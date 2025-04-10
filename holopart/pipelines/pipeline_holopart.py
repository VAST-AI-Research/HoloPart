import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler  # not sure
from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
import trimesh
from transformers import CLIPModel

from ..models.autoencoders import TripoSGVAEModel
from ..models.transformers import TripoSGDiTModel
from ..models.part_encoders import PartEncoder
from .pipeline_holopart_output import HoloPartPipelineOutput
from .pipeline_utils import TransformerDiffusionMixin

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class HoloPartPipeline(DiffusionPipeline, TransformerDiffusionMixin):
    """
    Pipeline for part generation using HoloPart.
    """

    def __init__(
        self,
        vae: TripoSGVAEModel,
        transformer: TripoSGDiTModel,
        scheduler: HoloPartPipelineOutput,
        part_encoder: PartEncoder,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            part_encoder=part_encoder,
        )

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def interrupt(self):
        return self._interrupt

    def prepare_latents(
        self,
        batch_size,
        num_tokens,
        num_channels_latents,
        dtype,
        device,
        generator,
        latents: Optional[torch.Tensor] = None,
    ):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (batch_size, num_tokens, num_channels_latents)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)

        return latents
    
    def encode_context_local(self, part_surface, whole_surface, part_local_surface, noise_strength, device):
        dtype = next(self.part_encoder.parameters()).dtype

        part_surface = part_surface.to(device=device, dtype=dtype)
        whole_surface = whole_surface.to(device=device, dtype=dtype)
        if part_local_surface is not None:
            part_local_surface = part_local_surface.to(device=device, dtype=dtype)
        
        whole_embeds, part_embeds = self.part_encoder(part=part_surface, whole=whole_surface, part_local=part_local_surface, noise_level=noise_strength)
        uncond_whole_embeds = torch.zeros_like(whole_embeds) if whole_embeds is not None else None
        uncond_part_embeds = torch.zeros_like(part_embeds) if part_embeds is not None else None

        return part_embeds, whole_embeds, uncond_part_embeds, uncond_whole_embeds

    @torch.no_grad()
    def __call__(
        self,
        part_surface,
        whole_surface,
        num_inference_steps: int = 50,
        num_tokens: int = 2048,
        timesteps: List[int] = None,
        guidance_scale: float = 3.5,
        part_local_surface = None,
        image: Optional[PipelineImageInput] = None,
        cond_noise_strength: float = 0.0,
        num_images_per_prompt: int = 1,
        sampled_points: Optional[torch.Tensor] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        output_type: Optional[str] = "geometry",
        bounds: Union[Tuple[float], List[float], float] = (-1.05, -1.05, -1.05, 1.05, 1.05, 1.05),
        dense_octree_depth: float = 8,
        hierarchical_octree_depth: float = 10,
        final_octree_depth: float = 9,
        post_smooth: bool = True,
        return_dict: bool = True,
    ):
        # 1. Check inputs. Raise error if not correct

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if isinstance(part_surface, list):
            batch_size = len(part_surface)
            assert len(part_surface) == len(whole_surface)
        elif isinstance(part_surface, torch.Tensor):
            if part_surface.ndim == 2:
                part_surface = part_surface.unsqueeze(0)
                whole_surface = whole_surface.unsqueeze(0)
                if part_local_surface is not None:
                    part_local_surface = part_local_surface.unsqueeze(0)
                    assert part_local_surface.shape[0] == part_surface.shape[0]
            batch_size = part_surface.shape[0]
            assert part_surface.shape[0] == whole_surface.shape[0]

        if image is not None:
            if isinstance(image, torch.Tensor):
                if image.ndim == 3:
                    image = image.unsqueeze(0)
                batch_size = image.shape[0]

        device = self._execution_device

        # 3. Encode condition
        part_embeds, whole_embeds, negative_part_embeds, negative_whole_embeds = self.encode_context_local(part_surface, whole_surface, part_local_surface, cond_noise_strength, device)

        if image is not None:
            image_embeds_clip, negative_image_embeds_clip = self.encode_image_clip(image, device)
        else:
            image_embeds_clip, negative_image_embeds_clip = None, None

        if self.do_classifier_free_guidance:
            part_embeds = torch.cat([negative_part_embeds, part_embeds], dim=0) if part_embeds is not None else None
            whole_embeds = torch.cat([negative_whole_embeds, whole_embeds], dim=0) if whole_embeds is not None else None
            image_embeds_clip = torch.cat([negative_image_embeds_clip, image_embeds_clip], dim=0) if image_embeds_clip is not None else None
    
        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps
        )
        num_warmup_steps = max(
            len(timesteps) - num_inference_steps * self.scheduler.order, 0
        )
        self._num_timesteps = len(timesteps)

        # 5. Prepare latent variables
        # num_tokens = self.transformer.config.width
        num_channels_latents = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_tokens,
            num_channels_latents,
            whole_embeds.dtype,
            device,
            generator,
            latents,
        )

        additional_cross_attention_kwargs = {}
        image_weight_clip = 1.0
        if image_embeds_clip is not None:
            additional_cross_attention_kwargs.update(
                {
                    "clip": {
                        "weight": image_weight_clip,
                        "encoder_hidden_states": image_embeds_clip
                    }
                }
            )

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2)
                    if self.do_classifier_free_guidance
                    else latents
                )
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latent_model_input.shape[0])

                noise_pred = self.transformer(
                    latent_model_input,
                    timestep,
                    encoder_hidden_states=part_embeds,
                    encoder_hidden_states_2=whole_embeds,
                    attention_kwargs=attention_kwargs,
                    additional_cross_attention_kwargs=additional_cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_image = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (
                        noise_pred_image - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(
                    noise_pred, t, latents, return_dict=False
                )[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                    (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()

        meshes = None
        if output_type == "latent":
            output = latents
        elif output_type == "geometry":
            if sampled_points is None:
                raise ValueError(
                    "sampled_points must be provided when output_type is not 'latent'"
                )

            output = self.vae.decode(latents, sampled_points=sampled_points).sample
        elif output_type == "mesh":
            from holopart.inference_utils import hierarchical_extract_geometry
            
            geometric_func = lambda x: self.vae.decode(latents, sampled_points=x).sample
            output = hierarchical_extract_geometry(
                geometric_func,
                device,
                bounds=bounds,
                dense_octree_depth=dense_octree_depth,
                hierarchical_octree_depth=hierarchical_octree_depth,
                final_octree_depth=final_octree_depth,
                post_smooth=post_smooth
            )
            meshes = [trimesh.Trimesh(mesh_v_f[0].astype(np.float32), mesh_v_f[1]) for mesh_v_f in output]
        else:
            raise ValueError(
                f"output_type of {output_type} is not supported"
            )

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (output, meshes)

        return HoloPartPipelineOutput(samples=output, meshes=meshes)
