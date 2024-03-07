# Adapted from https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py

from dataclasses import dataclass
from typing import Callable, Optional 
from einops import rearrange, repeat

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.attention import FeedForward, AdaLayerNorm
from diffusers.models.attention_processor import Attention, AttnProcessor

from .utils import create_1d_absolute_sin_cos_embedding


@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        temp_attn: bool = False,
        ref_frame_type_2: Optional[str] = None,
        cross_frame_type: str = None,
        lan_cross_attn: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                    temp_attn=temp_attn,
                    ref_frame_type_2=ref_frame_type_2,
                    cross_frame_type=cross_frame_type,
                    lan_cross_attn=lan_cross_attn,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True):
        # Input
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
        temp_attn: bool = True,
        ref_frame_type_2: Optional[str] = None,
        cross_frame_type: str = None,
        lan_cross_attn: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None
        self.temp_attn = temp_attn

        # Spatial Attention
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        
        # Cross-Attn
        if lan_cross_attn:
            self.lan_cross_attn = Attention(
                query_dim=cross_attention_dim,
                cross_attention_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
            self.lan_norm = nn.LayerNorm(cross_attention_dim)
            nn.init.zeros_(self.lan_cross_attn.to_out[0].weight.data)
        else:
            self.lan_cross_attn = None
            self.lan_norm = None

        if cross_attention_dim is not None:
            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        # Cross-Frame-Attention
        if cross_frame_type is not None:
            self.cross_frame_attn = CrossFrameAttention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
                cross_frame_type=cross_frame_type,
            )
            nn.init.zeros_(self.cross_frame_attn.to_out[0].weight.data)
            self.norm_cross_frame = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.cross_frame_attn = None

        # Temp-Attn
        if temp_attn:
            self.attn_temp = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
            )
            nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
            self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.attn_temp = None
            self.norm_temp = None
        
    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None):
        # Spatial-Attention
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        if self.only_cross_attention:
            hidden_states = (
                self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask) + hidden_states
            )
        else:
            hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask) + hidden_states

        if self.lan_cross_attn is not None:
            # lan cross attention
            norm_encoder_hidden_states = (
                self.lan_norm(encoder_hidden_states)
            )
            encoder_hidden_states = (
                self.lan_cross_attn(
                    norm_encoder_hidden_states, encoder_hidden_states=hidden_states, attention_mask=attention_mask
                )
                + encoder_hidden_states
            )

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                )
                + hidden_states
            )

        if self.attn_temp:
            # Temporal-Attention
            d = hidden_states.shape[1]
            hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
            norm_hidden_states = (
                self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
            )
            if video_length > 1:
                self.attn_temp.set_use_memory_efficient_attention_xformers(True)
            else:
                self.attn_temp.set_use_memory_efficient_attention_xformers(False)
            hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
            hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        if self.cross_frame_attn is not None:
            norm_hidden_states = (
                self.norm_cross_frame(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_cross_frame(hidden_states)
            )
            hidden_states = (
                self.cross_frame_attn(
                    norm_hidden_states, attention_mask=attention_mask, video_length=video_length,
                )
                + hidden_states
            )
        
        return hidden_states


class SpatialTemporalAttention(Attention):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        processor: Optional["AttnProcessor"] = None,
        ref_frame_type_2: Optional[str] = None,
    ):
        super().__init__(query_dim, cross_attention_dim, heads, dim_head, dropout, bias, upcast_attention, upcast_softmax, cross_attention_norm, cross_attention_norm_num_groups, added_kv_proj_dim, norm_num_groups, out_bias, scale_qk, only_cross_attention, processor)

        self.ref_frame_type_2 = ref_frame_type_2
        self.spatial_norm = None

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        query = self.head_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        self_frame_index = torch.arange(video_length)
        former_frame_index = self_frame_index - 1
        former_frame_index[0] = 0

        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)

        if self.ref_frame_type_2 == None:
            key = torch.cat([key[:, former_frame_index]], dim=2)
            value = torch.cat([value[:, former_frame_index]], dim=2)
        elif self.ref_frame_type_2 == 'first':
            key = torch.cat([key[:, former_frame_index], key[:, [0] * video_length]], dim=2)
            value = torch.cat([value[:, former_frame_index], value[:, [0] * video_length]], dim=2)
        elif self.ref_frame_type_2 == 'random':
            random_ref_frame_offset = torch.randint(low=1, high=max(2, video_length - 1), size=(video_length,))
            random_ref_frame_index = (self_frame_index + random_ref_frame_offset) % video_length
            key = torch.cat([key[:, former_frame_index], key[:, random_ref_frame_index]], dim=2)
            value = torch.cat([value[:, former_frame_index], value[:, random_ref_frame_index]], dim=2)
        elif self.ref_frame_type_2 == 'self':
            key = torch.cat([key[:, former_frame_index], key[:, self_frame_index]], dim=2)
            value = torch.cat([value[:, former_frame_index], value[:, self_frame_index]], dim=2)
        
        key = rearrange(key, "b f d c -> (b f) d c")
        value = rearrange(value, "b f d c -> (b f) d c")

        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        attention_probs = self.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
    
class CrossFrameAttention(Attention):
    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias=False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        processor: Optional["AttnProcessor"] = None,
        cross_frame_type: Optional[str] = None,
    ):
        super().__init__(query_dim, cross_attention_dim, heads, dim_head, dropout, bias, upcast_attention, upcast_softmax, cross_attention_norm, cross_attention_norm_num_groups, added_kv_proj_dim, norm_num_groups, out_bias, scale_qk, only_cross_attention, processor)

        self.cross_frame_type = cross_frame_type
        self.spatial_norm = None

    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        query = self.head_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        self_frame_index = torch.arange(video_length)
        former_frame_index = self_frame_index - 1
        former_frame_index[0] = 0

        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)

        if self.cross_frame_type == 'previous':
            key = torch.cat([key[:, former_frame_index]], dim=2)
            value = torch.cat([value[:, former_frame_index]], dim=2)
        elif self.cross_frame_type == 'first':
            key = torch.cat([key[:, [0] * video_length]], dim=2)
            value = torch.cat([value[:, [0] * video_length]], dim=2)
        
        key = rearrange(key, "b f d c -> (b f) d c")
        value = rearrange(value, "b f d c -> (b f) d c")

        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)

        attention_probs = self.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = self.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states
