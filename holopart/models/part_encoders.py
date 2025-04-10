import torch
import torch.nn as nn
from torch_cluster import fps
import numpy as np
from diffusers.models.normalization import LayerNorm
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .embeddings import FrequencyPositionalEmbedding
from .transformers.triposg_transformer import DiTBlock
from .autoencoders.autoencoder_kl_triposg import TripoSGEncoder
from ..schedulers.scheduling_rectified_flow import RectifiedFlowScheduler, compute_density_for_timestep_sampling
from ..utils.typing import *

class CrossAttentionEncoder(nn.Module):
    def __init__(
        self,
        q_in_channels: int = 3,
        kv_in_channels: int = 3,
        dim: int = 512,
        num_attention_heads: int = 8,
        num_layers: int = 8,
    ):
        super().__init__()

        self.q_proj_in = nn.Linear(q_in_channels, dim, bias=True)
        self.kv_proj_in = nn.Linear(kv_in_channels, dim, bias=True)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=False,
                    use_cross_attention=True,
                    cross_attention_dim=dim,
                    cross_attention_norm_type="layer_norm",
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",
                    norm_eps=1e-5,
                    qk_norm=False,
                    qkv_bias=False,
                )  # cross attention
            ]
            + [
                DiTBlock(
                    dim=dim,
                    num_attention_heads=num_attention_heads,
                    use_self_attention=True,
                    self_attention_norm_type="fp32_layer_norm",
                    use_cross_attention=False,
                    use_cross_attention_2=False,
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",
                    norm_eps=1e-5,
                    qk_norm=False,
                    qkv_bias=False,
                )
                for _ in range(num_layers)  # self attention
            ]
        )

        self.norm_out = LayerNorm(dim)

    def forward(self, sample_1: torch.Tensor, sample_2: torch.Tensor):
        hidden_states = self.q_proj_in(sample_1)
        encoder_hidden_states = self.kv_proj_in(sample_2)

        for layer, block in enumerate(self.blocks):
            if layer == 0:
                hidden_states = block(
                    hidden_states, encoder_hidden_states=encoder_hidden_states
                )
            else:
                hidden_states = block(hidden_states)

        hidden_states = self.norm_out(hidden_states)

        return hidden_states


class PartEncoder(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        position_channels: int = 3,
        part_feature_channels: int = 3,
        whole_feature_channels: int = 4,
        dim: int = 512,
        num_attention_heads: int = 8,
        num_layers: int = 8,
        num_tokens: int = 512,
        embedding_type: str = "frequency",
        embed_frequency: int = 8,
        embed_include_pi: bool = False,
        part_local: bool = False,
        init_weights: Optional[str] = None,
        noise_level: Optional[Tuple[float, float]] = None,
    ):

        super().__init__()

        if embedding_type == "frequency":
            self.embedder = FrequencyPositionalEmbedding(
                num_freqs=embed_frequency,
                logspace=True,
                input_dim=position_channels,
                include_pi=embed_include_pi,
            )
        else:
            raise NotImplementedError(
                f"Embedding type {embedding_type} is not supported."
            )
        
        if part_local:
            self.encoder_local = TripoSGEncoder(
                in_channels=part_feature_channels + self.embedder.out_dim,
                dim=dim,
                num_attention_heads=num_attention_heads,
                num_layers=num_layers
            )

        self.encoder_context = CrossAttentionEncoder(
            q_in_channels=part_feature_channels + self.embedder.out_dim,
            kv_in_channels=whole_feature_channels + self.embedder.out_dim,
            dim=dim,
            num_attention_heads=num_attention_heads,
            num_layers=num_layers
        )

        if noise_level is not None:
            self.noise_scheduler = RectifiedFlowScheduler()

        if init_weights is not None:
            from safetensors import safe_open
            state_dict = {}
            with safe_open(init_weights, framework="pt", device=0) as f:
                for k in f.keys():
                    state_dict[k] = f.get_tensor(k)
 
            whole_state_dict = {}
            for k, v in state_dict.items():
                if "encoder." in k:
                    k = k.replace("encoder.", "")
                    if "proj_in" in k:

                        # q_proj_in
                        if self.encoder_context.q_proj_in.weight.shape[-1] == v.shape[-1] + 1:
                            q_v = torch.cat([v, torch.ones_like(v[:, :1])], dim=-1)
                        else:
                            q_v = v
                        whole_state_dict[k.replace("proj_in", "q_proj_in")] = q_v

                        # kv_proj_in
                        if self.encoder_context.kv_proj_in.weight.shape[-1] == v.shape[-1] + 1:
                            kv_v = torch.cat([v, torch.ones_like(v[:, :1])], dim=-1)
                        else:
                            kv_v = v
                        whole_state_dict[k.replace("proj_in", "kv_proj_in")] = kv_v
                    else:
                        whole_state_dict[k] = v
            self.encoder_context.load_state_dict(whole_state_dict)

            if part_local:
                local_state_dict = {}
                for k, v in state_dict.items():
                    if "encoder." in k:
                        local_state_dict[k.replace("encoder.", "")] = v
                self.encoder_local.load_state_dict(local_state_dict)

    def _sample_features(
        self, x: torch.Tensor, num_tokens: int = 2048, seed: Optional[int] = None
    ):
        """
        Sample points from features of the input point cloud.

        Args:
            x (torch.Tensor): The input point cloud. shape: (B, N, C)
            num_tokens (int, optional): The number of points to sample. Defaults to 2048.
            seed (Optional[int], optional): The random seed. Defaults to None.
        """
        rng = np.random.default_rng(seed)
        indices = rng.choice(
            x.shape[1], num_tokens * 4, replace=num_tokens * 4 > x.shape[1]
        )
        selected_points = x[:, indices]

        batch_size, num_points, num_channels = selected_points.shape
        flattened_points = selected_points.view(batch_size * num_points, num_channels)
        batch_indices = (
            torch.arange(batch_size).to(x.device).repeat_interleave(num_points)
        )

        # fps sampling
        sampling_ratio = 1.0 / 4
        sampled_indices = fps(
            flattened_points[:, :3],
            batch_indices,
            ratio=sampling_ratio,
            random_start=self.training,
        )
        sampled_points = flattened_points[sampled_indices].view(
            batch_size, -1, num_channels
        )

        return sampled_points

    def add_noise(self, x, timesteps):
        noise = torch.randn_like(x)
        noisy_x = self.noise_scheduler.scale_noise(x, noise, timesteps)
        return noisy_x

    def forward(self, part: torch.Tensor, whole: torch.Tensor, part_local: Optional[torch.Tensor] = None, noise_level: float = 0.0, seed: Optional[int] = None):
        batch_size = part.shape[0]

        position_channels = self.config.position_channels
        sampled_part = self._sample_features(part, self.config.num_tokens, seed)
        part_sampled_positions, part_sampled_features = (
            sampled_part[..., :position_channels],
            sampled_part[..., position_channels:],
        )
        part_q = torch.cat([self.embedder(part_sampled_positions), part_sampled_features], dim=-1)

        # whole kv
        whole_positions, whole_features = whole[..., :position_channels], whole[..., position_channels:]
        whole_kv = torch.cat([self.embedder(whole_positions), whole_features], dim=-1)
        whole_cond = self.encoder_context(part_q, whole_kv)

        if self.config.part_local:
            assert part_local is not None
            sampled_part_local = self._sample_features(part_local, self.config.num_tokens, seed)
            part_local_sampled_positions, part_local_sampled_features = (
                sampled_part_local[..., :position_channels],
                sampled_part_local[..., position_channels:],
            )
            part_local_q = torch.cat([self.embedder(part_local_sampled_positions), part_local_sampled_features], dim=-1)
            part_local_positions, part_local_features = part_local[..., :position_channels], part_local[..., position_channels:]
            part_local_kv = torch.cat([self.embedder(part_local_positions), part_local_features], dim=-1)
            part_cond = self.encoder_local(part_local_q, part_local_kv)
        else:
            part_cond = None

        if (self.training and self.config.noise_level is not None) or (noise_level > 0.0):
            if noise_level > 0.0:
                sigmas = torch.ones(size=(batch_size, ), device=part.device) * noise_level
            else:
                sigmas = torch.rand(size=(batch_size, ), device=part.device) * (self.config.noise_level[1] - self.config.noise_level[0]) + self.config.noise_level[0]
            timesteps = self.noise_scheduler._sigma_to_t(sigmas)
            whole_cond = self.add_noise(whole_cond, timesteps)
            if part_cond is not None:
                part_cond = self.add_noise(part_cond, timesteps)

            # print("sigmas = ", sigmas)

        return whole_cond, part_cond