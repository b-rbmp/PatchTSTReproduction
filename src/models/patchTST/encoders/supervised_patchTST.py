__all__ = ["PatchTST"]

from typing import Optional
from torch import nn, Tensor
import torch

from src.training.supervised.config import TrainingConfig
from src.models.patchTST.positional.encoding import generate_positional_encoding
from src.models.patchTST.heads.flatten import FlattenHead
from src.models.patchTST.encoders.transformer import TSTEncoder
from src.models.patchTST.revin.revin import RevIN

class MovingAvgBlock(nn.Module):
    """
    Moving average block to smooth time series and highlight trends.
    Applies a 1D average pooling operation.
    """
    def __init__(self, kernel_size: int, stride: int):
        """
        Initializes the moving average layer.

        Args:
            kernel_size (int): The size of the moving average window.
            stride (int): The stride of the moving average window.
        """
        super(MovingAvgBlock, self).__init__()
        self.kernel_size = kernel_size
        # 1D Average Pooling layer
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies moving average to the input tensor.

        Args:
            x (Tensor): Input tensor of shape [batch_size, sequence_length, num_features].

        Returns:
            Tensor: Smoothed tensor of the same shape as input (adjusted for pooling).
        """
        # Pad the time series on both ends to maintain sequence length after averaging
        # Calculate padding size
        padding = (self.kernel_size - 1) // 2
        # Pad the front by repeating the first element
        front = x[:, 0:1, :].repeat(1, padding, 1)
        # Pad the end by repeating the last element
        end = x[:, -1:, :].repeat(1, padding, 1)
        # Concatenate padding and original tensor
        x = torch.cat([front, x, end], dim=1)
        # Apply average pooling (needs shape [batch, features, seq_len])
        x = self.avg(x.permute(0, 2, 1))
        # Permute back to original shape [batch, seq_len, features]
        x = x.permute(0, 2, 1)
        return x


class SeriesDecompositionBlock(nn.Module):
    """
    Series decomposition block that separates a time series into trend and residual components.
    Uses a moving average to determine the trend.
    """
    def __init__(self, kernel_size: int):
        """
        Initializes the decomposition block.

        Args:
            kernel_size (int): The kernel size for the moving average used to find the trend.
        """
        super(SeriesDecompositionBlock, self).__init__()
        # Moving average module to calculate the trend
        self.moving_avg = MovingAvgBlock(kernel_size, stride=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        Decomposes the input time series.

        Args:
            x (Tensor): Input tensor of shape [batch_size, sequence_length, num_features].

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:
                - res (Tensor): The residual component (original series - trend).
                - moving_mean (Tensor): The trend component.
        """
        # Calculate the trend component using moving average
        moving_mean = self.moving_avg(x)
        # Calculate the residual component (seasonality/noise)
        res = x - moving_mean
        return res, moving_mean

class TSTiEncoder(nn.Module):
    """
    Channel-independent Time Series Transformer (TST) encoder backbone.
    Applies patching, embedding, positional encoding, and transformer blocks.
    'i' denotes channel-independence, meaning each channel is processed separately by the main transformer.
    """
    def __init__(
        self,
        c_in: int,
        patch_num: int, 
        patch_len: int,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        store_attn: bool = False,
        res_attention: bool = True,
        pre_norm: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        only_patching: bool = False,
    ):
        super().__init__()

        # Store configuration
        self.patch_num = patch_num
        self.patch_len = patch_len
        self.only_patching = only_patching

        # Input Encoding
        # Linear layer to project patches to the embedding dimension (d_model)
        self.W_P = nn.Linear(patch_len, d_model)

        # Determine the sequence length for positional encoding
        # If only_patching, sequence is nvars * patch_num concatenated
        # Otherwise, sequence is just patch_num (processed per variable)
        if only_patching:
            q_len = c_in * patch_num
        else:
            q_len = patch_num

        # Positional Encoding
        self.W_pos = generate_positional_encoding(
            encoding_type=pe,
            sequence_length=q_len,
            model_dim=d_model,
            learnable=learn_pe,
        )

        # Residual Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder
        # The core transformer block stack
        self.encoder = TSTEncoder(
            model_dim=d_model,
            num_heads=n_heads,
            feedforward_dim=d_ff,
            use_batch_norm=(norm == "BatchNorm"),
            attention_dropout=attn_dropout,
            dropout_rate=dropout, # Pass the general dropout rate
            activation=act,
            residual_attention=res_attention,
            num_layers=n_layers,
            pre_normalization=pre_norm,
            store_attention=store_attn,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the TSTiEncoder.

        Args:
            x (Tensor): Input tensor after patching
        Returns:
            Tensor: Output tensor
        """
        bs, nvars, patch_len, patch_num = x.shape

        # 1. Patch Projection
        x = x.permute(0, 1, 3, 2)
        x = self.W_P(x)

        # 2. Flatten and Prepare for Transformer
        # Depending on 'only_patching', reshape differently.
        if not self.only_patching: # Standard channel-independent processing
            # Reshape to treat each channel's patch sequence independently:
            x = x.reshape(bs * nvars, patch_num, x.shape[-1])
            # Select positional encoding for 'patch_num' length: [patch_num, d_model]
            pe_slice = self.W_pos[:patch_num, :]
        else: # Special mode: Concatenate patches across channels
            # Reshape to have one long sequence per batch item:
            x = x.reshape(bs, patch_num * nvars, x.shape[-1])
            # Select positional encoding for 'patch_num * nvars' length: [patch_num * nvars, d_model]
            pe_slice = self.W_pos[:(patch_num * nvars), :]

        # 3. Add Positional Encoding
        # Unsqueeze pe_slice to add batch dim for broadcasting: [1, seq_len, d_model]
        x = x + pe_slice.unsqueeze(0)
        # Apply dropout: [batch_dim, seq_len, d_model] (batch_dim is bs*nvars or bs)
        x = self.dropout(x)

        # 4. Transformer Encoder Pass
        # Input/Output shape: [batch_dim, seq_len, d_model]
        x = self.encoder(x)

        # 5. Un-flatten (Reverse step 2)
        if not self.only_patching:
            # Reshape back to 
            x = x.reshape(bs, nvars, patch_num, x.shape[-1])
        else:
            # Reshape back to 
            x = x.reshape(bs, patch_num, nvars, x.shape[-1])
            # Permute to match the standard output format:
            x = x.permute(0, 2, 1, 3)

        return x


class PatchTST_backbone(nn.Module):
    """
    The main PatchTST model backbone, combining RevIN, patching, the TSTiEncoder, and a head.
    """
    def __init__(
        self,
        c_in: int,
        context_window: int,
        target_window: int,
        patch_len: int,
        stride: int,
        padding_patch: Optional[str] = None,
        n_layers: int = 3,
        d_model: int = 128,
        n_heads: int = 16,
        d_ff: int = 256,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        act: str = "gelu",
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        fc_dropout: float = 0.0,
        head_dropout: float = 0.0,
        pretrain_head: bool = False,
        head_type: str = "flatten",
        individual: bool = False,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        only_patching: bool = False,
    ):
        super().__init__()

        # Store configuration
        self.n_vars = c_in
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.revin = revin
        self.head_type = head_type
        self.individual = individual
        self.pretrain_head = pretrain_head

        # 1. Reversible Instance Normalization (RevIN)
        if self.revin:
            # Normalizes each time series instance (across time) independently
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # 2. Patching Calculation
        # Calculate the number of patches
        patch_num = int((context_window - patch_len) / stride + 1)
        # Optional padding at the end to ensure all data is used
        if padding_patch == "end":
            # Pad the sequence length dimension (dim=-1)
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) # Pads last dim on the right
            patch_num += 1 # Account for the extra patch from padding
        self.patch_num = patch_num

        # 3. Backbone (TSTiEncoder)
        # The core transformer encoder that processes patches
        self.backbone = TSTiEncoder(
            c_in=c_in, patch_num=self.patch_num, patch_len=patch_len,
            n_layers=n_layers, d_model=d_model,
            n_heads=n_heads, d_ff=d_ff, norm=norm,
            attn_dropout=attn_dropout, dropout=dropout, act=act,
            res_attention=res_attention, pre_norm=pre_norm,
            store_attn=store_attn, pe=pe, learn_pe=learn_pe,
            only_patching=only_patching
        )

        # 4. Head
        # Calculates the flattened feature dimension after the backbone
        self.head_nf = d_model * self.patch_num # Features per variable
        # Select and initialize the appropriate head
        if self.pretrain_head:
            # Head for pretraining tasks (e.g., masked patch reconstruction)
            self.head = nn.Sequential(nn.Dropout(fc_dropout), nn.Conv1d(self.head_nf, c_in, 1))
        elif head_type == "flatten":
            # Standard forecasting head: flattens patch embeddings and projects to target window
            self.head = FlattenHead(
                individual=self.individual, # Use separate or shared projection layer
                n_vars=self.n_vars,
                nf=self.head_nf, # Input features to the head per variable
                target_window=target_window,
                head_dropout=head_dropout,
            )

    def forward(self, z: Tensor) -> Tensor:
        """
        Forward pass through the PatchTST backbone.

        Args:
            z (Tensor): Input tensor, shape [batch_size, input_length, n_vars].
                        Note: Assumes input is (batch, seq_len, vars).

        Returns:
            Tensor: Output tensor (forecast), shape based on RevIN settings,
                    matching the original implementation's final output shape.
                    If RevIN: [batch_size, n_vars, target_window]
                    If not RevIN: [batch_size, n_vars, target_window]
        """
        # Permute to backbone expected:
        z = z.permute(0, 2, 1)

        # 1. Apply RevIN normalization (if enabled)
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)

        # 2. Apply Patching
        # Optional padding at the end
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)
        # Unfold the sequence dimension (-1) into patches
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # Permute for TSTiEncoder
        z = z.permute(0, 1, 3, 2)

        # 3. Pass through TSTiEncoder Backbone
        z = self.backbone(z)

        # 4. Pass through Head
        z = self.head(z)

        # 5. Apply RevIN denormalization (if enabled)
        if self.revin:
            # Original sequence: permute -> denorm -> permute
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "denorm")
            z = z.permute(0, 2, 1)

        return z


class PatchTST(nn.Module):
    """
    Top-level PatchTST model class.
    Integrates the backbone and handles optional time series decomposition.
    """

    def __init__(
        self,
        configs: TrainingConfig,
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        act: str = "gelu",
        res_attention: bool = True,
        pre_norm: bool = False,
        store_attn: bool = False,
        pe: str = "zeros",
        learn_pe: bool = True,
        pretrain_head: bool = False,
        head_type: str = "flatten",
    ):
        """
        Initializes the PatchTST model.

        Args:
            configs (TrainingConfig): Configuration object with model hyperparameters.
            max_seq_len, d_k, d_v, norm, ..., verbose: Optional overrides for specific
                                                      backbone/encoder parameters.
        """
        super().__init__()

        # Data shape
        c_in = configs.encoder_input_size
        context_window = configs.input_length
        target_window = configs.prediction_length

        # Patching
        patch_len = configs.patch_length
        stride = configs.stride
        padding_patch = configs.patch_padding

        # Core Model Dimensions
        n_layers = configs.num_encoder_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_fcn

        # Regularization
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        # Head configuration
        individual = configs.individual_head

        # RevIN configuration
        revin = configs.revin
        affine = configs.revin_affine
        subtract_last = configs.subtract_last

        # Decomposition configuration
        self.decomposition = configs.decomposition
        self.kernel_size = configs.kernel_size

        # Check if time series decomposition is enabled
        if self.decomposition:
            # 1. Decomposition Module
            self.decomp_module = SeriesDecompositionBlock(self.kernel_size)

            # 2. Create two separate PatchTST backbones: one for residual, one for trend
            # Both backbones share the same configuration.
            common_backbone_args = dict(
                c_in=c_in, context_window=context_window, target_window=target_window,
                patch_len=patch_len, stride=stride, padding_patch=padding_patch,
                n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                dropout=dropout, act=act, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
                pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                revin=revin, affine=affine, subtract_last=subtract_last,
                only_patching=configs.only_patching
            )
            self.model_res = PatchTST_backbone(**common_backbone_args)
            self.model_trend = PatchTST_backbone(**common_backbone_args)

        else:
            # No decomposition: Create a single PatchTST backbone
            self.model = PatchTST_backbone(
                c_in=c_in, context_window=context_window, target_window=target_window,
                patch_len=patch_len, stride=stride, padding_patch=padding_patch,
                n_layers=n_layers, d_model=d_model, n_heads=n_heads,
                d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                dropout=dropout, act=act, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout,
                pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                revin=revin, affine=affine, subtract_last=subtract_last,
                only_patching=configs.only_patching
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the complete PatchTST model.

        Args:
            x (Tensor): Input tensor of shape [Batch, Input_length, Channels].

        Returns:
            Tensor: Forecast tensor of shape [Batch, Output_length, Channels].
        """
        if self.decomposition:
            # 1. Decompose input
            res_init, trend_init = self.decomp_module(x)

            # 2. Process residual and trend through backbones
            res_out = self.model_res(res_init)
            trend_out = self.model_trend(trend_init)

            # 3. Combine outputs
            combined_out = res_out + trend_out

            # 4. Permute to final desired shap
            final_output = combined_out.permute(0, 2, 1)

        else:
            # No decomposition
            # 1. Process directly through the single backbone
            backbone_output = self.model(x)

            # 2. Permute to final desired shape
            final_output = backbone_output.permute(0, 2, 1)


        return final_output