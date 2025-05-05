import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class ScaledDotProductAttention(nn.Module):
    """
    Implements Scaled Dot-Product Attention with optional residual attention
    (RealFormer) and optional learnable scale for locality self-attention (LSA).

    Args:
        model_dim (int): Dimension of the model embeddings.
        num_heads (int): Number of attention heads.
        attention_dropout (float, optional): Dropout for the attention weights. Default is 0.0.
        residual_attention (bool, optional): If True, adds previous layer's attention
            logits to the current ones for RealFormer-like behavior. Default is False.
        locality_self_attention (bool, optional): If True, allows learnable scale
            for LSA. Default is False.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        attention_dropout: float = 0.0,
        residual_attention: bool = False,
        locality_self_attention: bool = False,
    ):
        super().__init__()
        # Head dimension (d_k) is model_dim // num_heads
        head_dim = model_dim // num_heads

        # Learnable scale parameter for LSA
        # If `locality_self_attention` is True, this parameter is trainable
        # Otherwise it's a constant
        self.scaling_factor = nn.Parameter(
            torch.tensor(head_dim**-0.5), requires_grad=locality_self_attention
        )

        self.dropout = nn.Dropout(attention_dropout)
        self.residual_attention = residual_attention

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        values: Tensor,
        previous_scores: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ):
        """
        Forward pass for scaled dot-product attention.

        Shapes:
            queries: [batch_size, num_heads, query_len, d_k]
            keys:    [batch_size, num_heads, d_k, seq_len]
            values:  [batch_size, num_heads, seq_len, d_v]
            previous_scores (optional): [batch_size, num_heads, query_len, seq_len]
            key_padding_mask (optional): [batch_size, seq_len]
            attention_mask (optional): [query_len, seq_len] or broadcastable

        Returns:
            - output: [batch_size, num_heads, query_len, d_v]
            - attn_weights: [batch_size, num_heads, query_len, seq_len]
            - attn_scores (only if residual_attention=True): raw attention logits
        """
        # Compute raw attention logits: (q @ k) scaled by sqrt(d_k)
        attn_scores = torch.matmul(queries, keys) * self.scaling_factor

        # Optional RealFormer-like residual attention (add logits from previous layer)
        if previous_scores is not None:
            attn_scores = attn_scores + previous_scores

        # Optional attention mask
        if attention_mask is not None:
            # If mask is boolean, fill with -inf where True
            if attention_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attention_mask, float("-inf"))
            else:
                attn_scores += attention_mask

        # Key padding mask
        if key_padding_mask is not None:
            # Expand mask to match attention score dimensions
            # shape: [batch_size, 1, 1, seq_len]
            expanded_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores.masked_fill_(expanded_mask, float("-inf"))

        # Softmax over last dimension (seq_len)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute final values: (Attn Weights) x (V)
        output = torch.matmul(attn_weights, values)

        if self.residual_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class MultiheadAttention(nn.Module):
    """
    Multi-Head Attention module that projects inputs into multiple heads, applies
    scaled dot-product attention, and projects outputs back to model_dim.

    Args:
        model_dim (int): Dimensionality of input embeddings.
        num_heads (int): Number of attention heads.
        head_dim_q (int, optional): Dim of each head for Q/K. Defaults to model_dim//num_heads.
        head_dim_v (int, optional): Dim of each head for V. Defaults to model_dim//num_heads.
        residual_attention (bool, optional): Enable RealFormer-like residual attention. Default is False.
        attention_dropout (float, optional): Dropout rate for attention weights. Default is 0.0.
        projection_dropout (float, optional): Dropout rate after attention projection. Default is 0.0.
        qkv_bias (bool, optional): If True, includes bias term in Q, K, V linear layers. Default is True.
        locality_self_attention (bool, optional): If True, uses a learnable scale factor in attention. Default is False.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        head_dim_q: Optional[int] = None,
        head_dim_v: Optional[int] = None,
        residual_attention: bool = False,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        qkv_bias: bool = True,
        locality_self_attention: bool = False,
    ):
        super().__init__()

        # Default to splitting model_dim evenly among heads if not specified
        head_dim_q = head_dim_q or (model_dim // num_heads)
        head_dim_v = head_dim_v or (model_dim // num_heads)

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim_q = head_dim_q
        self.head_dim_v = head_dim_v
        self.residual_attention = residual_attention

        # Linear transformations for Q, K, V
        self.query_proj = nn.Linear(model_dim, num_heads * head_dim_q, bias=qkv_bias)
        self.key_proj = nn.Linear(model_dim, num_heads * head_dim_q, bias=qkv_bias)
        self.value_proj = nn.Linear(model_dim, num_heads * head_dim_v, bias=qkv_bias)

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            residual_attention=residual_attention,
            locality_self_attention=locality_self_attention,
        )

        # Final linear projection back to model_dim
        self.output_projection = nn.Sequential(
            nn.Linear(num_heads * head_dim_v, model_dim), nn.Dropout(projection_dropout)
        )

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        previous_attention: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ):
        """
        Forward pass of multi-head attention.

        Shapes:
            query:  [batch_size, query_len, model_dim]
            key:    [batch_size, seq_len, model_dim]  (defaults to query if None)
            value:  [batch_size, seq_len, model_dim]  (defaults to query if None)
            previous_attention (optional): [batch_size, num_heads, query_len, seq_len] for residual attention
            key_padding_mask (optional): [batch_size, seq_len]
            attention_mask (optional): broadcastable to [query_len, seq_len] or [batch_size, query_len, seq_len]

        Returns:
            - output: [batch_size, query_len, model_dim]
            - attn_weights: [batch_size, num_heads, query_len, seq_len]
            - attn_scores (only if residual_attention=True): raw attention logits
        """
        batch_size = query.size(0)

        # Default to self-attention
        if key is None:
            key = query
        if value is None:
            value = query

        # Project into Q, K, V for all heads
        q_proj = self.query_proj(query)
        k_proj = self.key_proj(key)
        v_proj = self.value_proj(value)

        # Reshap
        # For keys, we transpose after splitting
        q_proj = q_proj.view(batch_size, -1, self.num_heads, self.head_dim_q).transpose(
            1, 2
        )
        k_proj = k_proj.view(batch_size, -1, self.num_heads, self.head_dim_q).permute(
            0, 2, 3, 1
        )
        v_proj = v_proj.view(batch_size, -1, self.num_heads, self.head_dim_v).transpose(
            1, 2
        )

        # Compute scaled dot-product attention
        if self.residual_attention:
            output, weights, scores = self.attention(
                queries=q_proj,
                keys=k_proj,
                values=v_proj,
                previous_scores=previous_attention,
                key_padding_mask=key_padding_mask,
                attention_mask=attention_mask,
            )
        else:
            output, weights = self.attention(
                queries=q_proj,
                keys=k_proj,
                values=v_proj,
                previous_scores=None,
                key_padding_mask=key_padding_mask,
                attention_mask=attention_mask,
            )
            scores = None

        # Reshape
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_dim_v)
        )

        # Project output back to model_dim
        output = self.output_projection(output)

        if self.residual_attention:
            return output, weights, scores
        else:
            return output, weights
