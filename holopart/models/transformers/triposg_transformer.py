# Copyright (c) 2025 VAST-AI-Research and contributors

# This code is based on Tencent HunyuanDiT (https://huggingface.co/Tencent-Hunyuan/HunyuanDiT),
# which is licensed under the Tencent Hunyuan Community License Agreement.
# Portions of this code are copied or adapted from HunyuanDiT.
# See the original license below:

# ---- Start of Tencent Hunyuan Community License Agreement ----

# TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT
# Tencent Hunyuan DiT Release Date: 14 May 2024
# THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION AND IS EXPRESSLY LIMITED TO THE TERRITORY, AS DEFINED BELOW.
# By clicking to agree or by using, reproducing, modifying, distributing, performing or displaying any portion or element of the Tencent Hunyuan Works, including via any Hosted Service, You will be deemed to have recognized and accepted the content of this Agreement, which is effective immediately.
# 1.	DEFINITIONS.
# a.	“Acceptable Use Policy” shall mean the policy made available by Tencent as set forth in the Exhibit A.
# b.	“Agreement” shall mean the terms and conditions for use, reproduction, distribution, modification, performance and displaying of Tencent Hunyuan Works or any portion or element thereof set forth herein.
# c.	“Documentation” shall mean the specifications, manuals and documentation for Tencent Hunyuan made publicly available by Tencent.
# d.	“Hosted Service” shall mean a hosted service offered via an application programming interface (API), web access, or any other electronic or remote means.
# e.	“Licensee,” “You” or “Your” shall mean a natural person or legal entity exercising the rights granted by this Agreement and/or using the Tencent Hunyuan Works for any purpose and in any field of use.
# f.	“Materials” shall mean, collectively, Tencent’s proprietary Tencent Hunyuan and Documentation (and any portion thereof) as made available by Tencent under this Agreement.
# g.	“Model Derivatives” shall mean all: (i) modifications to Tencent Hunyuan or any Model Derivative of Tencent Hunyuan; (ii) works based on Tencent Hunyuan or any Model Derivative of Tencent Hunyuan; or (iii) any other machine learning model which is created by transfer of patterns of the weights, parameters, operations, or Output of Tencent Hunyuan or any Model Derivative of Tencent Hunyuan, to that model in order to cause that model to perform similarly to Tencent Hunyuan or a Model Derivative of Tencent Hunyuan, including distillation methods, methods that use intermediate data representations, or methods based on the generation of synthetic data Outputs by Tencent Hunyuan or a Model Derivative of Tencent Hunyuan for training that model. For clarity, Outputs by themselves are not deemed Model Derivatives.
# h.	“Output” shall mean the information and/or content output of Tencent Hunyuan or a Model Derivative that results from operating or otherwise using Tencent Hunyuan or a Model Derivative, including via a Hosted Service.
# i.	“Tencent,” “We” or “Us” shall mean THL A29 Limited.
# j.	“Tencent Hunyuan” shall mean the large language models, text/image/video/audio/3D generation models, and multimodal large language models and their software and algorithms, including trained model weights, parameters (including optimizer states), machine-learning model code, inference-enabling code, training-enabling code, fine-tuning enabling code and other elements of the foregoing made publicly available by Us, including, without limitation to, Tencent Hunyuan DiT released at https://huggingface.co/Tencent-Hunyuan/HunyuanDiT.
# k.	“Tencent Hunyuan Works” shall mean: (i) the Materials; (ii) Model Derivatives; and (iii) all derivative works thereof.
# l.	“Territory” shall mean the worldwide territory, excluding the territory of the European Union.
# m.	“Third Party” or “Third Parties” shall mean individuals or legal entities that are not under common control with Us or You.
# n.	“including” shall mean including but not limited to.
# 2.	GRANT OF RIGHTS.
# We grant You, for the Territory only, a non-exclusive, non-transferable and royalty-free limited license under Tencent’s intellectual property or other rights owned by Us embodied in or utilized by the Materials to use, reproduce, distribute, create derivative works of (including Model Derivatives), and make modifications to the Materials, only in accordance with the terms of this Agreement and the Acceptable Use Policy, and You must not violate (or encourage or permit anyone else to violate) any term of this Agreement or the Acceptable Use Policy.
# 3.	DISTRIBUTION.
# You may, subject to Your compliance with this Agreement, distribute or make available to Third Parties the Tencent Hunyuan Works, exclusively in the Territory, provided that You meet all of the following conditions:
# a.	You must provide all such Third Party recipients of the Tencent Hunyuan Works or products or services using them a copy of this Agreement;
# b.	You must cause any modified files to carry prominent notices stating that You changed the files;
# c.	You are encouraged to: (i) publish at least one technology introduction blogpost or one public statement expressing Your experience of using the Tencent Hunyuan Works; and (ii) mark the products or services developed by using the Tencent Hunyuan Works to indicate that the product/service is “Powered by Tencent Hunyuan”; and
# d.	All distributions to Third Parties (other than through a Hosted Service) must be accompanied by a “Notice” text file that contains the following notice: “Tencent Hunyuan is licensed under the Tencent Hunyuan Community License Agreement, Copyright © 2024 Tencent. All Rights Reserved. The trademark rights of “Tencent Hunyuan” are owned by Tencent or its affiliate.”
# You may add Your own copyright statement to Your modifications and, except as set forth in this Section and in Section 5, may provide additional or different license terms and conditions for use, reproduction, or distribution of Your modifications, or for any such Model Derivatives as a whole, provided Your use, reproduction, modification, distribution, performance and display of the work otherwise complies with the terms and conditions of this Agreement (including as regards the Territory). If You receive Tencent Hunyuan Works from a Licensee as part of an integrated end user product, then this Section 3 of this Agreement will not apply to You.
# 4.	ADDITIONAL COMMERCIAL TERMS.
# If, on the Tencent Hunyuan version release date, the monthly active users of all products or services made available by or for Licensee is greater than 100 million monthly active users in the preceding calendar month, You must request a license from Tencent, which Tencent may grant to You in its sole discretion, and You are not authorized to exercise any of the rights under this Agreement unless or until Tencent otherwise expressly grants You such rights.
# 5.	RULES OF USE.
# a.	Your use of the Tencent Hunyuan Works must comply with applicable laws and regulations (including trade compliance laws and regulations) and adhere to the Acceptable Use Policy for the Tencent Hunyuan Works, which is hereby incorporated by reference into this Agreement. You must include the use restrictions referenced in these Sections 5(a) and 5(b) as an enforceable provision in any agreement (e.g., license agreement, terms of use, etc.) governing the use and/or distribution of Tencent Hunyuan Works and You must provide notice to subsequent users to whom You distribute that Tencent Hunyuan Works are subject to the use restrictions in these Sections 5(a) and 5(b).
# b.	You must not use the Tencent Hunyuan Works or any Output or results of the Tencent Hunyuan Works to improve any other large language model (other than Tencent Hunyuan or Model Derivatives thereof).
# c.	You must not use, reproduce, modify, distribute, or display the Tencent Hunyuan Works, Output or results of the Tencent Hunyuan Works outside the Territory. Any such use outside the Territory is unlicensed and unauthorized under this Agreement.
# 6.	INTELLECTUAL PROPERTY.
# a.	Subject to Tencent’s ownership of Tencent Hunyuan Works made by or for Tencent and intellectual property rights therein, conditioned upon Your compliance with the terms and conditions of this Agreement, as between You and Tencent, You will be the owner of any derivative works and modifications of the Materials and any Model Derivatives that are made by or for You.
# b.	No trademark licenses are granted under this Agreement, and in connection with the Tencent Hunyuan Works, Licensee may not use any name or mark owned by or associated with Tencent or any of its affiliates, except as required for reasonable and customary use in describing and distributing the Tencent Hunyuan Works. Tencent hereby grants You a license to use “Tencent Hunyuan” (the “Mark”) in the Territory solely as required to comply with the provisions of Section 3(c), provided that You comply with any applicable laws related to trademark protection. All goodwill arising out of Your use of the Mark will inure to the benefit of Tencent.
# c.	If You commence a lawsuit or other proceedings (including a cross-claim or counterclaim in a lawsuit) against Us or any person or entity alleging that the Materials or any Output, or any portion of any of the foregoing, infringe any intellectual property or other right owned or licensable by You, then all licenses granted to You under this Agreement shall terminate as of the date such lawsuit or other proceeding is filed. You will defend, indemnify and hold harmless Us from and against any claim by any Third Party arising out of or related to Your or the Third Party’s use or distribution of the Tencent Hunyuan Works.
# d.	Tencent claims no rights in Outputs You generate. You and Your users are solely responsible for Outputs and their subsequent uses.
# 7.	DISCLAIMERS OF WARRANTY AND LIMITATIONS OF LIABILITY.
# a.	We are not obligated to support, update, provide training for, or develop any further version of the Tencent Hunyuan Works or to grant any license thereto.
# b.	UNLESS AND ONLY TO THE EXTENT REQUIRED BY APPLICABLE LAW, THE TENCENT HUNYUAN WORKS AND ANY OUTPUT AND RESULTS THEREFROM ARE PROVIDED “AS IS” WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES OF ANY KIND INCLUDING ANY WARRANTIES OF TITLE, MERCHANTABILITY, NONINFRINGEMENT, COURSE OF DEALING, USAGE OF TRADE, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING, REPRODUCING, MODIFYING, PERFORMING, DISPLAYING OR DISTRIBUTING ANY OF THE TENCENT HUNYUAN WORKS OR OUTPUTS AND ASSUME ANY AND ALL RISKS ASSOCIATED WITH YOUR OR A THIRD PARTY’S USE OR DISTRIBUTION OF ANY OF THE TENCENT HUNYUAN WORKS OR OUTPUTS AND YOUR EXERCISE OF RIGHTS AND PERMISSIONS UNDER THIS AGREEMENT.
# c.	TO THE FULLEST EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT SHALL TENCENT OR ITS AFFILIATES BE LIABLE UNDER ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, TORT, NEGLIGENCE, PRODUCTS LIABILITY, OR OTHERWISE, FOR ANY DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL, EXEMPLARY, CONSEQUENTIAL OR PUNITIVE DAMAGES, OR LOST PROFITS OF ANY KIND ARISING FROM THIS AGREEMENT OR RELATED TO ANY OF THE TENCENT HUNYUAN WORKS OR OUTPUTS, EVEN IF TENCENT OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF ANY OF THE FOREGOING.
# 8.	SURVIVAL AND TERMINATION.
# a.	The term of this Agreement shall commence upon Your acceptance of this Agreement or access to the Materials and will continue in full force and effect until terminated in accordance with the terms and conditions herein.
# b.	We may terminate this Agreement if You breach any of the terms or conditions of this Agreement. Upon termination of this Agreement, You must promptly delete and cease use of the Tencent Hunyuan Works. Sections 6(a), 6(c), 7 and 9 shall survive the termination of this Agreement.
# 9.	GOVERNING LAW AND JURISDICTION.
# a.	This Agreement and any dispute arising out of or relating to it will be governed by the laws of the Hong Kong Special Administrative Region of the People’s Republic of China, without regard to conflict of law principles, and the UN Convention on Contracts for the International Sale of Goods does not apply to this Agreement.
# b.	Exclusive jurisdiction and venue for any dispute arising out of or relating to this Agreement will be a court of competent jurisdiction in the Hong Kong Special Administrative Region of the People’s Republic of China, and Tencent and Licensee consent to the exclusive jurisdiction of such court with respect to any such dispute.
#
# EXHIBIT A
# ACCEPTABLE USE POLICY

# Tencent reserves the right to update this Acceptable Use Policy from time to time.
# Last modified: [insert date]

# Tencent endeavors to promote safe and fair use of its tools and features, including Tencent Hunyuan. You agree not to use Tencent Hunyuan or Model Derivatives:
# 1.	Outside the Territory;
# 2.	In any way that violates any applicable national, federal, state, local, international or any other law or regulation;
# 3.	To harm Yourself or others;
# 4.	To repurpose or distribute output from Tencent Hunyuan or any Model Derivatives to harm Yourself or others;
# 5.	To override or circumvent the safety guardrails and safeguards We have put in place;
# 6.	For the purpose of exploiting, harming or attempting to exploit or harm minors in any way;
# 7.	To generate or disseminate verifiably false information and/or content with the purpose of harming others or influencing elections;
# 8.	To generate or facilitate false online engagement, including fake reviews and other means of fake online engagement;
# 9.	To intentionally defame, disparage or otherwise harass others;
# 10.	To generate and/or disseminate malware (including ransomware) or any other content to be used for the purpose of harming electronic systems;
# 11.	To generate or disseminate personal identifiable information with the purpose of harming others;
# 12.	To generate or disseminate information (including images, code, posts, articles), and place the information in any public context (including –through the use of bot generated tweets), without expressly and conspicuously identifying that the information and/or content is machine generated;
# 13.	To impersonate another individual without consent, authorization, or legal right;
# 14.	To make high-stakes automated decisions in domains that affect an individual’s safety, rights or wellbeing (e.g., law enforcement, migration, medicine/health, management of critical infrastructure, safety components of products, essential services, credit, employment, housing, education, social scoring, or insurance);
# 15.	In a manner that violates or disrespects the social ethics and moral standards of other countries or regions;
# 16.	To perform, facilitate, threaten, incite, plan, promote or encourage violent extremism or terrorism;
# 17.	For any use intended to discriminate against or harm individuals or groups based on protected characteristics or categories, online or offline social behavior or known or predicted personal or personality characteristics;
# 18.	To intentionally exploit any of the vulnerabilities of a specific group of persons based on their age, social, physical or mental characteristics, in order to materially distort the behavior of a person pertaining to that group in a manner that causes or is likely to cause that person or another person physical or psychological harm;
# 19.	For military purposes;
# 20.	To engage in the unauthorized or unlicensed practice of any profession including, but not limited to, financial, legal, medical/health, or other professional practices.

# ---- End of Tencent Hunyuan Community License Agreement ----

# Please note that the use of this code is subject to the terms and conditions
# of the Tencent Hunyuan Community License Agreement, including the Acceptable Use Policy.

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.models.attention import FeedForward
from diffusers.models.attention_processor import Attention, AttentionProcessor
from diffusers.models.embeddings import (
    GaussianFourierProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import (
    AdaLayerNormContinuous,
    FP32LayerNorm,
    LayerNorm,
)
from diffusers.utils import (
    USE_PEFT_BACKEND,
    is_torch_version,
    logging,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import maybe_allow_in_graph
from torch import nn

from ..attention_processor import FusedTripoSGAttnProcessor2_0, TripoSGAttnProcessor2_0
from .modeling_outputs import Transformer1DModelOutput
from ...utils.typing import *

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class DiTBlock(nn.Module):
    r"""
    Transformer block used in Hunyuan-DiT model (https://github.com/Tencent/HunyuanDiT). Allow skip connection and
    QKNorm

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of headsto use for multi-head attention.
        cross_attention_dim (`int`,*optional*):
            The size of the encoder_hidden_states vector for cross attention.
        dropout(`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        activation_fn (`str`,*optional*, defaults to `"geglu"`):
            Activation function to be used in feed-forward. .
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, *optional*, defaults to 1e-6):
            A small constant added to the denominator in normalization layers to prevent division by zero.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*):
            The size of the hidden layer in the feed-forward block. Defaults to `None`.
        ff_bias (`bool`, *optional*, defaults to `True`):
            Whether to use bias in the feed-forward block.
        skip (`bool`, *optional*, defaults to `False`):
            Whether to use skip connection. Defaults to `False` for down-blocks and mid-blocks.
        qk_norm (`bool`, *optional*, defaults to `True`):
            Whether to use normalization in QK calculation. Defaults to `True`.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        use_self_attention: bool = True,
        use_cross_attention: bool = False,
        self_attention_norm_type: Optional[str] = None,  # ada layer norm
        cross_attention_dim: Optional[int] = None,
        cross_attention_norm_type: Optional[str] = "fp32_layer_norm",
        use_cross_attention_2: bool = False,
        cross_attention_2_dim: Optional[int] = None,
        cross_attention_2_norm_type: Optional[str] = None,
        dropout=0.0,
        activation_fn: str = "gelu",
        norm_type: str = "fp32_layer_norm",  # TODO
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        ff_inner_dim: Optional[int] = None,  # int(dim * 4) if None
        ff_bias: bool = True,
        skip: bool = False,
        skip_concat_front: bool = False,  # [x, skip] or [skip, x]
        skip_norm_last: bool = False,  # this is an error
        qk_norm: bool = True,
        qkv_bias: bool = True,
    ):
        super().__init__()

        self.use_self_attention = use_self_attention
        self.use_cross_attention = use_cross_attention
        self.use_cross_attention_2 = use_cross_attention_2
        self.skip_concat_front = skip_concat_front
        self.skip_norm_last = skip_norm_last
        self.dim = dim
        self.num_attention_heads = num_attention_heads
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.qk_norm = qk_norm
        self.qkv_bias = qkv_bias
        self.gradient_checkpointing = False
        # Define 3 blocks. Each block has its own normalization layer.
        # NOTE: when new version comes, check norm2 and norm 3
        # 1. Self-Attn
        if use_self_attention:
            if (
                self_attention_norm_type == "fp32_layer_norm"
                or self_attention_norm_type is None
            ):
                self.norm1 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)
            else:
                raise NotImplementedError

            self.attn1 = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=dim // num_attention_heads,
                heads=num_attention_heads,
                qk_norm="rms_norm" if qk_norm else None,
                eps=1e-6,
                bias=qkv_bias,
                processor=TripoSGAttnProcessor2_0(),
            )

        # 2. Cross-Attn
        if use_cross_attention:
            assert cross_attention_dim is not None

            self.norm2 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                dim_head=dim // num_attention_heads,
                heads=num_attention_heads,
                qk_norm="rms_norm" if qk_norm else None,
                cross_attention_norm=cross_attention_norm_type,
                eps=1e-6,
                bias=qkv_bias,
                processor=TripoSGAttnProcessor2_0(),
            )

        # 2'. Second Cross-Attn
        if use_cross_attention_2:
            assert cross_attention_2_dim is not None

            self.norm2_2 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2_2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_2_dim,
                dim_head=dim // num_attention_heads,
                heads=num_attention_heads,
                qk_norm="rms_norm" if qk_norm else None,
                cross_attention_norm=cross_attention_2_norm_type,
                eps=1e-6,
                bias=qkv_bias,
                processor=TripoSGAttnProcessor2_0(),
            )

        self.additional_cross_attention = nn.ModuleDict()
        self.additional_norm = nn.ModuleDict()

        # 3. Feed-forward
        self.norm3 = FP32LayerNorm(dim, norm_eps, norm_elementwise_affine)

        self.ff = FeedForward(
            dim,
            dropout=dropout,  ### 0.0
            activation_fn=activation_fn,  ### approx GeLU
            final_dropout=final_dropout,  ### 0.0
            inner_dim=ff_inner_dim,  ### int(dim * mlp_ratio)
            bias=ff_bias,
        )

        # 4. Skip Connection
        if skip:
            self.skip_norm = FP32LayerNorm(dim, norm_eps, elementwise_affine=True)
            self.skip_linear = nn.Linear(2 * dim, dim)
        else:
            self.skip_linear = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_topk(self, topk):
        self.flash_processor.topk = topk

    def set_flash_processor(self, flash_processor):
        self.flash_processor = flash_processor
        self.attn2.processor = self.flash_processor

    def update_cross_attention(self, name, cross_attention_dim, cross_attention_norm_type = None):
        assert cross_attention_dim is not None
        norm = FP32LayerNorm(self.dim, self.norm_eps, self.norm_elementwise_affine)
        attn = Attention(
                query_dim=self.dim,
                cross_attention_dim=cross_attention_dim,
                dim_head=self.dim // self.num_attention_heads,
                heads=self.num_attention_heads,
                qk_norm="rms_norm" if self.qk_norm else None,
                cross_attention_norm=cross_attention_norm_type,
                eps=1e-6,
                bias=self.qkv_bias,
                processor=TripoSGAttnProcessor2_0(),
            )
        
        if name == "attn2":
            self.norm2 = norm
            self.attn2 = attn
        elif name == "attn2_2":
            self.norm2_2 = norm
            self.attn2_2 = attn
        else:
            raise NotImplementedError

    def add_additional_cross_attention(self, name, cross_attention_dim, cross_attention_norm_type=None):
        
        assert cross_attention_dim is not None
        norm = FP32LayerNorm(self.dim, self.norm_eps, self.norm_elementwise_affine)

        attn = Attention(
            query_dim=self.dim,
            cross_attention_dim=cross_attention_dim,
            dim_head=self.dim // self.num_attention_heads,
            heads=self.num_attention_heads,
            qk_norm="rms_norm" if self.qk_norm else None,
            cross_attention_norm=cross_attention_norm_type,
            eps=1e-6,
            bias=self.qkv_bias,
            processor=TripoSGAttnProcessor2_0(),
        )

        self.additional_cross_attention[name] = attn
        self.additional_norm[name] = norm

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def _set_gradient_checkpointing(self, module, value: bool = False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    def _forward_cross_attention(self, 
        hidden_states, 
        cross_attentions: List[Dict[str, Any]], 
        image_rotary_emb: Optional[torch.Tensor] = None, 
        attention_kwargs: Optional[Dict[str, Any]] = None, 
    ):
        attention_kwargs = attention_kwargs or {}

        for cross_attention in cross_attentions:
            attn_hidden_states = cross_attention["weight"] * cross_attention["module"](
                cross_attention["norm"](hidden_states), 
                cross_attention["encoder_hidden_states"], 
                image_rotary_emb=image_rotary_emb, 
                **attention_kwargs
            )
            hidden_states = hidden_states + attn_hidden_states

        return hidden_states
    
    def _build_cross_attenion_kwargs(self, name, encoder_hidden_states, weight=1.0):
        if name == "attn2":
            kwargs = {
                "module": self.attn2,
                "weight": weight,
                "norm": self.norm2,
                
            }
        elif name == "attn2_2":
            kwargs = {
                "module": self.attn2_2,
                "weight": weight,
                "norm": self.norm2_2,
            }
        else:
            kwargs = {
                "module": self.additional_cross_attention[name],
                "weight": weight,
                "norm": self.additional_norm[name],
            }

        kwargs["encoder_hidden_states"] = encoder_hidden_states
        kwargs["name"] = name
        return kwargs

    def _forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        skip: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        additional_cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        # Prepare attention kwargs
        attention_kwargs = attention_kwargs or {}

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Long Skip Connection
        if self.skip_linear is not None:
            cat = torch.cat(
                (
                    [skip, hidden_states]
                    if self.skip_concat_front
                    else [hidden_states, skip]
                ),
                dim=-1,
            )
            if self.skip_norm_last:
                # don't do this
                hidden_states = self.skip_linear(cat)
                hidden_states = self.skip_norm(hidden_states)
            else:
                cat = self.skip_norm(cat)
                hidden_states = self.skip_linear(cat)

        # 1. Self-Attention
        if self.use_self_attention:
            norm_hidden_states = self.norm1(hidden_states)
            attn_output = self.attn1(
                norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **attention_kwargs,
            )
            hidden_states = hidden_states + attn_output

        # 2. Cross-Attention
        hidden_states = self._forward_cross_attention(
            hidden_states,
            (
                [
                    self._build_cross_attenion_kwargs("attn2", encoder_hidden_states)
                ] if self.use_cross_attention else []
            ) + (
                [
                    self._build_cross_attenion_kwargs("attn2_2", encoder_hidden_states_2)  
                ] if self.use_cross_attention_2 else []
            ),
            image_rotary_emb=image_rotary_emb,
            attention_kwargs=attention_kwargs,
        )

        if additional_cross_attention_kwargs is not None:
            hidden_states = self._forward_cross_attention(
                hidden_states,
                [
                    self._build_cross_attenion_kwargs(name, value["encoder_hidden_states"], value["weight"])
                    for name, value in additional_cross_attention_kwargs.items()
                ],
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            )
            
        # FFN Layer ### TODO: switch norm2 and norm3 in the state dict
        mlp_inputs = self.norm3(hidden_states)
        hidden_states = hidden_states + self.ff(mlp_inputs)

        return hidden_states
    
    def forward(self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        skip: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        additional_cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if self.gradient_checkpointing:
            return torch.utils.checkpoint.checkpoint(
                self._forward,
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_2,
                temb,
                image_rotary_emb,
                skip,
                attention_kwargs,
                additional_cross_attention_kwargs,
                use_reentrant=False
            )
        else:
            return self._forward(
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_2,
                temb,
                image_rotary_emb,
                skip,
                attention_kwargs,
                additional_cross_attention_kwargs,
            )

class TripoSGDiTModel(ModelMixin, ConfigMixin, PeftAdapterMixin):
    """
    TripoSGDiT: Diffusion model with a Transformer backbone.

    Inherit ModelMixin and ConfigMixin to be compatible with the sampler StableDiffusionPipeline of diffusers.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88):
            The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        patch_size (`int`, *optional*):
            The size of the patch to use for the input.
        activation_fn (`str`, *optional*, defaults to `"geglu"`):
            Activation function to use in feed-forward.
        sample_size (`int`, *optional*):
            The width of the latent images. This is fixed during training since it is used to learn a number of
            position embeddings.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        cross_attention_dim (`int`, *optional*):
            The number of dimension in the clip text embedding.
        hidden_size (`int`, *optional*):
            The size of hidden layer in the conditioning embedding layers.
        num_layers (`int`, *optional*, defaults to 1):
            The number of layers of Transformer blocks to use.
        mlp_ratio (`float`, *optional*, defaults to 4.0):
            The ratio of the hidden layer size to the input size.
        learn_sigma (`bool`, *optional*, defaults to `True`):
             Whether to predict variance.
        cross_attention_dim_t5 (`int`, *optional*):
            The number dimensions in t5 text embedding.
        pooled_projection_dim (`int`, *optional*):
            The size of the pooled projection.
        text_len (`int`, *optional*):
            The length of the clip text embedding.
        text_len_t5 (`int`, *optional*):
            The length of the T5 text embedding.
        use_style_cond_and_image_meta_size (`bool`,  *optional*):
            Whether or not to use style condition and image meta size. True for version <=1.1, False for version >= 1.2
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        width: int = 2048,
        in_channels: int = 64,
        num_layers: int = 21,
        cross_attention_dim: int = 512,
        cross_attention_2_dim: int = 512,
        use_cross_attention: bool = False,
        use_cross_attention_2: bool = True
    ):
        super().__init__()
        self.out_channels = in_channels
        self.num_heads = num_attention_heads
        self.inner_dim = width
        self.mlp_ratio = 4.0

        time_embed_dim, timestep_input_dim = self._set_time_proj(
            "positional",
            inner_dim=self.inner_dim,
            flip_sin_to_cos=False,
            freq_shift=0,
            time_embedding_dim=None,
        )
        self.time_proj = TimestepEmbedding(
            timestep_input_dim, time_embed_dim, act_fn="gelu", out_dim=self.inner_dim
        )
        self.proj_in = nn.Linear(self.config.in_channels, self.inner_dim, bias=True)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    use_self_attention=True,
                    use_cross_attention=use_cross_attention,
                    self_attention_norm_type="fp32_layer_norm",
                    cross_attention_dim=self.config.cross_attention_dim,
                    cross_attention_norm_type=None,
                    use_cross_attention_2=use_cross_attention_2,
                    cross_attention_2_dim=self.config.cross_attention_2_dim,
                    cross_attention_2_norm_type=None,
                    activation_fn="gelu",
                    norm_type="fp32_layer_norm",  # TODO
                    norm_eps=1e-5,
                    ff_inner_dim=int(self.inner_dim * self.mlp_ratio),
                    skip=layer > num_layers // 2,
                    skip_concat_front=True,
                    skip_norm_last=True,  # this is an error
                    qk_norm=True,  # See http://arxiv.org/abs/2302.05442 for details.
                    qkv_bias=False,
                )
                for layer in range(num_layers)
            ]
        )

        self.norm_out = LayerNorm(self.inner_dim)
        self.proj_out = nn.Linear(self.inner_dim, self.out_channels, bias=True)

    def add_additional_cross_attention(self, name, cross_attention_dim, cross_attention_norm_type=None):
        for block in self.blocks:
            block.add_additional_cross_attention(name, cross_attention_dim, cross_attention_norm_type)

    def update_cross_attention(self, name, cross_attention_dim, cross_attention_norm_type = None):
        for block in self.blocks:
            block.update_cross_attention(name, cross_attention_dim, cross_attention_norm_type)

    def set_use_cross_attention(self, use_cross_attention, use_cross_attention_2):
        for block in self.blocks:
            block.use_cross_attention = use_cross_attention
            block.use_cross_attention_2 = use_cross_attention_2

    def _set_time_proj(
        self,
        time_embedding_type: str,
        inner_dim: int,
        flip_sin_to_cos: bool,
        freq_shift: float,
        time_embedding_dim: int,
    ) -> Tuple[int, int]:
        if time_embedding_type == "fourier":
            time_embed_dim = time_embedding_dim or inner_dim * 2
            if time_embed_dim % 2 != 0:
                raise ValueError(
                    f"`time_embed_dim` should be divisible by 2, but is {time_embed_dim}."
                )
            self.time_embed = GaussianFourierProjection(
                time_embed_dim // 2,
                set_W_to_weight=False,
                log=False,
                flip_sin_to_cos=flip_sin_to_cos,
            )
            timestep_input_dim = time_embed_dim
        elif time_embedding_type == "positional":
            time_embed_dim = time_embedding_dim or inner_dim * 4

            self.time_embed = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
            timestep_input_dim = inner_dim
        else:
            raise ValueError(
                f"{time_embedding_type} does not exist. Please make sure to use one of `fourier` or `positional`."
            )

        return time_embed_dim, timestep_input_dim

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.fuse_qkv_projections with FusedAttnProcessor2_0->FusedTripoSGAttnProcessor2_0
    def fuse_qkv_projections(self):
        """
        Enables fused QKV projections. For self-attention modules, all projection matrices (i.e., query, key, value)
        are fused. For cross-attention modules, key and value projection matrices are fused.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>
        """
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError(
                    "`fuse_qkv_projections()` is not supported for models having added KV projections."
                )

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedTripoSGAttnProcessor2_0())

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.unfuse_qkv_projections
    def unfuse_qkv_projections(self):
        """Disables the fused QKV projection if enabled.

        <Tip warning={true}>

        This API is 🧪 experimental.

        </Tip>

        """
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(
            name: str,
            module: torch.nn.Module,
            processors: Dict[str, AttentionProcessor],
        ):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(
        self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]
    ):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def set_default_attn_processor(self):
        """
        Disables custom attention processors and sets the default attention implementation.
        """
        self.set_attn_processor(TripoSGAttnProcessor2_0())

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        timestep: Union[int, float, torch.LongTensor],
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_hidden_states_2: Optional[torch.Tensor] = None,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        additional_cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        """
        The [`HunyuanDiT2DModel`] forward method.

        Args:
        hidden_states (`torch.Tensor` of shape `(batch size, dim, height, width)`):
            The input tensor.
        timestep ( `torch.LongTensor`, *optional*):
            Used to indicate denoising step.
        encoder_hidden_states ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer.
        encoder_hidden_states_2 ( `torch.Tensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
            Conditional embeddings for cross attention layer.
        return_dict: bool
            Whether to return a dictionary.
        """

        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if (
                attention_kwargs is not None
                and attention_kwargs.get("scale", None) is not None
            ):
                logger.warning(
                    "Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective."
                )

        _, N, _ = hidden_states.shape

        temb = self.time_embed(timestep).to(hidden_states.dtype)
        temb = self.time_proj(temb)
        temb = temb.unsqueeze(dim=1)  # unsqueeze to concat with hidden_states

        hidden_states = self.proj_in(hidden_states)

        # N + 1 token
        hidden_states = torch.cat([temb, hidden_states], dim=1)

        skips = []
        for layer, block in enumerate(self.blocks):
            if layer <= self.config.num_layers // 2:
                hidden_states = block(
                    hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_2=encoder_hidden_states_2,
                    attention_kwargs=attention_kwargs,
                    additional_cross_attention_kwargs=additional_cross_attention_kwargs,
                )  # (N, L, D)
            else:
                skip = skips.pop()
                hidden_states = block(
                    hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_2=encoder_hidden_states_2,
                    skip=skip,
                    attention_kwargs=attention_kwargs,
                    additional_cross_attention_kwargs=additional_cross_attention_kwargs,
                )  # (N, L, D)

            if layer < self.config.num_layers // 2:
                skips.append(hidden_states)

        # final layer
        hidden_states = self.norm_out(hidden_states)
        hidden_states = hidden_states[:, -N:]
        hidden_states = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (hidden_states,)

        return Transformer1DModelOutput(sample=hidden_states)
    
    def _set_gradient_checkpointing(self, module, value: bool = False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.enable_forward_chunking
    def enable_forward_chunking(
        self, chunk_size: Optional[int] = None, dim: int = 0
    ) -> None:
        """
        Sets the attention processor to use [feed forward
        chunking](https://huggingface.co/blog/reformer#2-chunked-feed-forward-layers).

        Parameters:
            chunk_size (`int`, *optional*):
                The chunk size of the feed-forward layers. If not specified, will run feed-forward layer individually
                over each tensor of dim=`dim`.
            dim (`int`, *optional*, defaults to `0`):
                The dimension over which the feed-forward computation should be chunked. Choose between dim=0 (batch)
                or dim=1 (sequence length).
        """
        if dim not in [0, 1]:
            raise ValueError(f"Make sure to set `dim` to either 0 or 1, not {dim}")

        # By default chunk size is 1
        chunk_size = chunk_size or 1

        def fn_recursive_feed_forward(
            module: torch.nn.Module, chunk_size: int, dim: int
        ):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, chunk_size, dim)

    # Copied from diffusers.models.unets.unet_3d_condition.UNet3DConditionModel.disable_forward_chunking
    def disable_forward_chunking(self):
        def fn_recursive_feed_forward(
            module: torch.nn.Module, chunk_size: int, dim: int
        ):
            if hasattr(module, "set_chunk_feed_forward"):
                module.set_chunk_feed_forward(chunk_size=chunk_size, dim=dim)

            for child in module.children():
                fn_recursive_feed_forward(child, chunk_size, dim)

        for module in self.children():
            fn_recursive_feed_forward(module, None, 0)