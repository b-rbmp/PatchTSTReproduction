import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch import Tensor

from src.models.patchTST.encoders.transformer import TSTEncoder
from src.models.patchTST.positional.encoding import generate_positional_encoding
from src.models.patchTST.heads.prediction import LinearPredictionHead
from src.models.patchTST.heads.regression import LinearRegressionHead
from src.models.patchTST.heads.classification import LinearClassificationHead
from src.models.patchTST.heads.pretrain import LinearPretrainHead

class PatchTSTEncoder(nn.Module):
    """
    Encoder for the unsupervised/general PatchTST model.

    Processes time series data by:
    1. Projecting patches into an embedding space (channel-dependent or shared).
    2. Adding positional encoding to the patch sequence.
    3. Passing the sequence through a stack of Transformer encoder layers (TSTEncoder).

    Input Shape: [batch_size, num_patches, num_channels, patch_length]
    Output Shape: [batch_size, num_channels, model_dim, num_patches]
    """

    def __init__(
        self,
        input_channels: int,
        num_patches: int,
        patch_length: int,
        num_layers: int = 3,
        model_dim: int = 128,
        num_heads: int = 16,
        feedforward_dim: int = 256,
        use_batch_norm: bool = True,
        attention_dropout: float = 0.0,
        activation: str = "gelu",
        store_attention: bool = False,
        residual_attention: bool = True,
        pre_normalization: bool = False,
        dropout_rate: float = 0.0,
        positional_encoding_type: str = "zeros",
        learn_positional_encoding: bool = True,
        shared_projection: bool = True,
    ):
        """
        Initializes the PatchTSTEncoder.

        Args:
            input_channels: Number of variables in the time series.
            num_patches: Number of patches created from the input sequence.
            patch_length: Length of each patch.
            num_layers: Number of transformer encoder layers.
            model_dim: The embedding dimension.
            num_heads: Number of attention heads.
            feedforward_dim: Hidden dimension in the FFN of transformer layers.
            use_batch_norm: Use BatchNorm (True) or LayerNorm (False).
            attention_dropout: Dropout for attention weights.
            dropout_rate: General dropout rate (after pos encoding).
            activation: Activation function ('gelu', 'relu').
            store_attention: Whether to store attention weights.
            residual_attention: Use residual attention connections.
            pre_normalization: Apply normalization before sublayers.
            positional_encoding_type: Type of positional encoding (e.g., 'zeros', 'sincos').
            learn_positional_encoding: If True, positional encodings are learnable parameters.
            shared_projection: If True, use one Linear layer to project patches for all channels.
                               If False, use a separate Linear layer per channel.
        """
        super().__init__()

        # Store dimensions and configurations
        self.num_channels = input_channels
        self.num_patches = num_patches
        self.patch_length = patch_length
        self.model_dim = model_dim
        self.shared_projection = shared_projection

        # 1. Patch Projection
        # Create the linear layer(s) to project patches to model_dim
        if not self.shared_projection:
            # Create a list of projection layers, one for each input channel
            self.patch_projection = nn.ModuleList([
                nn.Linear(self.patch_length, self.model_dim)
                for _ in range(self.num_channels)
            ])
        else:
            # Create a single projection layer shared across all channels
            self.patch_projection = nn.Linear(self.patch_length, self.model_dim)

        # 2. Positional Encoding
        # Generate positional encodings for the sequence of patches (length = num_patches)
        self.positional_encoding = generate_positional_encoding(
            encoding_type=positional_encoding_type,
            learnable=learn_positional_encoding,
            sequence_length=self.num_patches,
            model_dim=self.model_dim,
        )

        # 3. Dropout
        # Applied after adding positional encoding
        self.dropout = nn.Dropout(dropout_rate)

        # 4. Transformer Encoder Stack
        # The core sequence processing module
        self.encoder = TSTEncoder(
            model_dim=self.model_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            use_batch_norm=use_batch_norm,
            attention_dropout=attention_dropout,
            dropout_rate=dropout_rate,
            activation=activation,
            residual_attention=residual_attention,
            num_layers=num_layers,
            pre_normalization=pre_normalization,
            store_attention=store_attention,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass through the PatchTSTEncoder.

        Args:
            inputs (Tensor): Input tensor with shape

        Returns:
            Tensor: Encoded output tensor with shape
        """
        batch_size, num_patches, num_channels, patch_length = inputs.shape

        # 1. Patch Projection
        # Project patches to model_dim
        if not self.shared_projection:
            # Apply each channel's specific projection layer
            projected_channels = []
            for i in range(num_channels):
                # Input to linear layer
                channel_patches = inputs[:, :, i, :]
                # Output
                embedded = self.patch_projection[i](channel_patches)
                projected_channels.append(embedded)
            # Stack along the channel dimension
            x = torch.stack(projected_channels, dim=2)
        else:
            # Apply the shared projection layer across all channels simultaneously
            x = self.patch_projection(inputs)

        # Reshape for channel-independent processing by transformer:
        x = x.transpose(1, 2)

        # 2. Prepare for Transformer
        # Flatten batch and channel dimensions to treat each channel's patch sequence independently
        x_reshaped = x.reshape(batch_size * num_channels, num_patches, self.model_dim)

        # 3. Add Positional Encoding
        x_reshaped = x_reshaped + self.positional_encoding

        # 4. Apply Dropout
        x_reshaped = self.dropout(x_reshaped)

        # 5. Pass through Transformer Encoder
        encoded = self.encoder(x_reshaped)

        # 6. Reshape back
        encoded = encoded.reshape(batch_size, num_channels, num_patches, self.model_dim)
        # Permute to final output shape for heads:
        encoded = encoded.permute(0, 1, 3, 2)

        return encoded


class PatchTST(nn.Module):
    """
    Unsupervised/General PatchTST Model.

    Combines the PatchTSTEncoder with various task-specific heads (pretraining,
    prediction, regression, classification). 

    Input Shape (to forward method): [batch_size, num_patches, input_channels, patch_length]

    Output Shape (depends on head_type):
        - Pretraining: [batch_size, num_patches, input_channels, patch_length] (reconstructed patches)
        - Prediction:  [batch_size x target_dim x input_channels] (forecast)
        - Regression:  [batch_size x target_dim] (regression target)
        - Classification: [batch_size x target_dim] (class logits/probabilities)
    """

    def __init__(
        self,
        input_channels: int,
        num_patches: int,
        patch_length: int,
        target_dim: int,
        num_layers: int = 3,
        model_dim: int = 128,
        num_heads: int = 8,
        feedforward_dim: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        head_type: str = "prediction",
        y_range: Optional[Tuple[float, float]] = None,
        **encoder_kwargs,
    ):
        """
        Initializes the main PatchTST model.

        Args:
            input_channels: Number of input variables.
            num_patches: Number of patches.
            patch_length: Length of each patch.
            target_dim: Target output dimension (depends on task).
            num_layers: Number of layers in the encoder transformer.
            model_dim: Embedding dimension.
            num_heads: Number of attention heads in the encoder.
            feedforward_dim: Hidden dimension in the encoder's FFN.
            dropout: General dropout rate.
            activation: Activation function for the encoder.
            head_type: The task type, determines the head used.
            y_range: Optional output range for the regression head.
        """
        super().__init__()

        # Validation
        valid_head_types = {"pretrain", "prediction", "regression", "classification"}
        if head_type not in valid_head_types:
            raise ValueError(
                f"Invalid head_type '{head_type}'. Must be one of {valid_head_types}."
            )

        # Store config
        self.head_type = head_type
        self.num_vars = input_channels
        self.target_dim = target_dim

        # 1. Encoder Backbone
        # Initializes the patch processing and transformer encoding part
        self.encoder = PatchTSTEncoder(
            input_channels=input_channels,
            num_patches=num_patches,
            patch_length=patch_length,
            num_layers=num_layers,
            model_dim=model_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout_rate=dropout,
            activation=activation,
        )

        # 2. Task-Specific Head
        # Select and initialize the appropriate head based on head_type
        head_dropout = dropout

        if head_type == "pretrain":
            # Head for reconstructing masked patches
            self.head = LinearPretrainHead(
                model_dim=model_dim,
                patch_length=patch_length,
                dropout_rate=head_dropout # Pass dropout to head
            )
        elif head_type == "prediction":
            # Head for time series forecasting
            self.head = LinearPredictionHead(
                num_vars=self.num_vars,
                model_dim=model_dim,
                num_patches=num_patches,
                forecast_len=target_dim,
                dropout_rate=head_dropout,
                individual=True,
            )
        elif head_type == "regression":
            # Head for regression tasks (predicting continuous values)
            self.head = LinearRegressionHead(
                num_vars=self.num_vars,
                model_dim=model_dim,
                output_dim=target_dim,
                dropout_rate=head_dropout,
                y_range=y_range,
            )
        elif head_type == "classification":
            # Head for classification tasks (predicting discrete classes)
            self.head = LinearClassificationHead(
                num_vars=self.num_vars,
                model_dim=model_dim,
                num_classes=target_dim,
                dropout_rate=head_dropout,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the PatchTST model.

        Args:
            x (torch.Tensor): Input tensor of shape

        Returns:
            torch.Tensor: Output tensor whose shape depends on the head_type.
        """
        # 1. Encode the input patches
        encoded_features = self.encoder(x)

        # 2. Pass encoded features through the task-specific head
        output = self.head(encoded_features)

        return output

