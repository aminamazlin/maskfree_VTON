from typing import Callable, List, Optional, Tuple, Union
import math
import torch
import torch.nn.functional as F
from torch import nn

from diffusers.models.attention_processor import Attention
from diffusers.utils import deprecate


class CustomAttnProcessor:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, store_attention: bool = False):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "CustomAttnProcessor requires PyTorch 2.0+. Please upgrade PyTorch to 2.0 or newer."
            )

        # Whether to store attention probabilities for visualization
        self.store_attn_map: bool = bool(store_attention)
        # Will be populated by caller (e.g., BasicTransformerBlock) before each forward
        self.timestep: Optional[int] = None

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        timestep: Optional[torch.LongTensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # Handle deprecated scale kwarg if passed through cross_attention_kwargs
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = (
                "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise "
                "an error in the future. `scale` should be passed to the pipeline via `cross_attention_kwargs`."
            )
            deprecate("scale", "1.0.0", deprecation_message)

        # Optionally accept timestep and store as int for later use by hooks
        if timestep is not None:
            try:
                self.timestep = int(torch.as_tensor(timestep).detach().view(-1)[0].item())
            except Exception:
                # Fallback to None if conversion fails
                self.timestep = None

        # Optionally accept explicit spatial size hints (used by viz)
        try:
            self.height = int(height) if height is not None else None
            self.width = int(width) if width is not None else None
        except Exception:
            self.height, self.width = None, None

        residual = hidden_states

        # Spatial norm (for 4D inputs)
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            # Persist spatial resolution for later visualization
            self.last_height = int(height)
            self.last_width = int(width)
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            # Not a spatial tensor; reset spatial hints
            self.last_height = None
            self.last_width = None

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # if attention_mask is not None:
        #     attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        #     # scaled_dot_product_attention expects attention_mask shape to be
        #     # (batch, heads, source_length, target_length)
        #     attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # Group Norm
        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        # Determine if self-attention or cross-attention
        is_cross_attention = encoder_hidden_states is not None
        self.last_is_cross = bool(is_cross_attention)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape for multi-head attention
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Optional normalization
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Track query/key lengths for reference
        self.last_q_len = int(query.shape[-2])
        self.last_k_len = int(key.shape[-2])

        # Only for self-attention (do NOT store maps here; user wants cross-attn only)
        if not is_cross_attention and batch_size == 2:
            # Split into person (batch 0) and object (batch 1)
            query_person = query[0:1]  # [1, heads, seq_len, head_dim]
            query_object = query[1:2]

            key_person = key[0:1]
            key_object = key[1:2]

            value_person = value[0:1]
            value_object = value[1:2]

            # Person's context: sees both person and object
            context_key_person = torch.cat([key_person, key_object], dim=2)  # [1, heads, 2*seq_len, head_dim]
            context_value_person = torch.cat([value_person, value_object], dim=2)

            # Object's context: sees only itself
            context_key_object = key_object  # [1, heads, seq_len, head_dim]
            context_value_object = value_object

            # Compute attention for person
            hidden_states_person = F.scaled_dot_product_attention(
                query_person,
                context_key_person,
                context_value_person,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )

            # Keep only own query length
            hidden_states_person = hidden_states_person[:, :, : query_person.shape[-2], :]

            # Compute attention for object
            hidden_states_object = F.scaled_dot_product_attention(
                query_object,
                context_key_object,
                context_value_object,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
            )

            # Combine back
            hidden_states = torch.cat([hidden_states_person, hidden_states_object], dim=0)
        else:
            # Cross-attention or regular self-attention
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                attention_mask = attention_mask.view(batch_size, attn.heads, 1, attention_mask.shape[-1])

            # Compute attention scores manually to optionally store them
            attn_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask

            attn_probs = torch.softmax(attn_scores, dim=-1)

            # Store attention weights only for cross-attention
            if self.store_attn_map and is_cross_attention:
                self.attn_map = attn_probs.detach().to("cpu")

            # Multiply by value to get output
            hidden_states = torch.matmul(attn_probs, value)

        # Reshape output
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        # Residual connection
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


