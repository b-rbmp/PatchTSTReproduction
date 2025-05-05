import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple
from src.models.patchTST.attention.multihead import MultiheadAttention
from src.models.utils.tensor_ops import Transpose

class TSTEncoderLayer(nn.Module):
    """
    A single layer of the Time Series Transformer (TST) Encoder.

    This layer consists of:
    1. Multi-Head Self-Attention mechanism.
    2. Add & Norm (Residual connection followed by normalization).
    3. Position-wise Feed-Forward Network.
    4. Add & Norm (Residual connection followed by normalization).

    Supports Batch Normalization or Layer Normalization, pre/post normalization,
    and residual attention connections.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int = 256,
        use_batch_norm: bool = True,
        attention_dropout: float = 0.0,
        dropout_rate: float = 0.0,
        bias: bool = True,
        activation: str = "gelu",
        residual_attention: bool = False,
        pre_normalization: bool = False,
        store_attention: bool = False,
    ):
        """
        Initializes the TSTEncoderLayer.

        Args:
            model_dim: Dimension of the model.
            num_heads: Number of attention heads. Must divide model_dim.
            feedforward_dim: Hidden dimension of the FFN.
            use_batch_norm: If True, use BatchNorm; otherwise, use LayerNorm.
            attention_dropout: Dropout probability for attention weights.
            dropout_rate: Dropout probability for outputs of attention and FFN sublayers.
            bias: Whether linear layers have bias terms.
            activation: Activation function for FFN ('gelu' or 'relu').
            residual_attention: If True, attention scores are passed residually.
            pre_normalization: If True, normalization precedes sublayer execution.
            store_attention: If True, saves attention weights in `self.last_attention`.
        """
        super().__init__()

        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})."
            )

        # Calculate dimensions for key, query, value per head
        head_dim = model_dim // num_heads

        # Store flags
        self.pre_normalization = pre_normalization
        self.store_attention = store_attention
        self.residual_attention = residual_attention

        # 1. Multi-Head Self-Attention
        self.attention_block = MultiheadAttention(
            model_dim=model_dim,
            num_heads=num_heads,
            head_dim_q=head_dim,
            head_dim_v=head_dim,
            attention_dropout=attention_dropout,
            projection_dropout=dropout_rate,
            residual_attention=residual_attention,
        )

        # 2. Add & Norm (Attention)
        self.attn_dropout = nn.Dropout(dropout_rate)
        if use_batch_norm:
            # BatchNorm requires shape [batch, channels, length]
            # Transpose: [b, l, d] -> [b, d, l] -> BatchNorm -> [b, d, l] -> Transpose: [b, l, d]
            self.attn_normalization = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(model_dim), Transpose(1, 2)
            )
        else:
            # LayerNorm operates on the last dimension (features)
            self.attn_normalization = nn.LayerNorm(model_dim)

        # 3. Position-wise Feed-Forward Network (FFN)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feedforward_dim, bias=bias),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feedforward_dim, model_dim, bias=bias),
        )

        # 4. Add & Norm (FFN)
        self.ffn_dropout = nn.Dropout(dropout_rate)
        if use_batch_norm:
            # Same BatchNorm structure as after attention
            self.ffn_normalization = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(model_dim), Transpose(1, 2)
            )
        else:
            self.ffn_normalization = nn.LayerNorm(model_dim)

        # Attribute to store attention weights if needed
        self.last_attention = None

    def forward(
        self, inputs: Tensor, previous_attention: Optional[Tensor] = None
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass through a single encoder layer.

        Args:
            inputs (Tensor): Input tensor of shape [batch_size, sequence_length, model_dim].
            previous_attention (Optional[Tensor]): Attention scores from the previous layer, used only if `residual_attention` is True.

        Returns:
            Tuple[Tensor, Optional[Tensor]]:
                - Output tensor of shape [batch_size, sequence_length, model_dim].
                - Updated attention scores (Tensor) if `residual_attention` is True, else None.
        """
        residual = inputs # Store input for residual connection

        # 1. Multi-Head Self-Attention
        # Pre-Normalization (apply norm before the sublayer)
        if self.pre_normalization:
            attn_input = self.attn_normalization(inputs)
        else:
            attn_input = inputs

        # Apply attention mechanism
        if self.residual_attention:
            # MultiheadAttention returns (output, weights, scores) if residual_attention=True
            attended, attn_weights, scores = self.attention_block(
                query=attn_input, key=attn_input, value=attn_input, previous_attention=previous_attention
            )
        else:
            # MultiheadAttention returns (output, weights) otherwise
            attended, attn_weights = self.attention_block(
                query=attn_input, key=attn_input, value=attn_input
            )
            scores = None # No residual scores to pass on

        # Store attention weights if requested
        if self.store_attention:
            self.last_attention = attn_weights

        # 2. Add & Norm (Attention)
        # Apply dropout to the attention output before adding residual
        attended = self.attn_dropout(attended)
        # Add residual connection
        attn_output = residual + attended
        # Post-Normalization (apply norm after the sublayer and residual)
        if not self.pre_normalization:
            attn_output = self.attn_normalization(attn_output)

        # Store output for the next residual connection
        residual = attn_output

        # 3. Position-wise Feed-Forward
        # Pre-Normalization
        if self.pre_normalization:
            ffn_input = self.ffn_normalization(attn_output)
        else:
            ffn_input = attn_output

        # Apply feed-forward network
        ffn_out = self.feed_forward(ffn_input)

        # 4. Add & Norm (FFN)
        # Apply dropout to the FFN output
        ffn_out = self.ffn_dropout(ffn_out)
        # Add residual connection
        ffn_output = residual + ffn_out
        # Post-Normalization
        if not self.pre_normalization:
            ffn_output = self.ffn_normalization(ffn_output)

        # Return the final output of the layer and attention scores (if used)
        return ffn_output, scores

class TSTEncoder(nn.Module):
    """
    A stack of multiple TSTEncoderLayer modules.
    Processes an input sequence through all layers sequentially.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        feedforward_dim: int = 256,
        use_batch_norm: bool = True,
        attention_dropout: float = 0.0,
        dropout_rate: float = 0.0,
        activation: str = "gelu",
        residual_attention: bool = False,
        pre_normalization: bool = False,
        store_attention: bool = False,
        num_layers: int = 1,
    ):
        """
        Initializes the TSTEncoder.

        Args:
            model_dim, num_heads, ..., store_attention: Parameters for each TSTEncoderLayer.
            num_layers (int): The number of identical encoder layers in the stack.
        """
        super().__init__()

        self.residual_attention = residual_attention # Store flag

        # Create a ModuleList to hold the stack of encoder layers
        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    feedforward_dim=feedforward_dim,
                    use_batch_norm=use_batch_norm,
                    attention_dropout=attention_dropout,
                    dropout_rate=dropout_rate,
                    activation=activation,
                    residual_attention=residual_attention,
                    pre_normalization=pre_normalization,
                    store_attention=store_attention,
                )
                for _ in range(num_layers) # Create num_layers identical layers
            ]
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass through the entire stack of encoder layers.

        Args:
            inputs (Tensor): Input tensor of shape [batch_size, sequence_length, model_dim].

        Returns:
            Tensor: Output tensor of shape [batch_size, sequence_length, model_dim].
        """
        output = inputs
        attention_scores = None # Initialize attention scores for residual connection

        # Sequentially apply each layer
        for layer in self.layers:
            # Pass the output of the previous layer as input to the current layer
            # Also pass attention scores if residual_attention is enabled
            output, attention_scores = layer(output, previous_attention=attention_scores)
            # If residual_attention is False, attention_scores will remain None

        # Return the final output after passing through all layers
        return output