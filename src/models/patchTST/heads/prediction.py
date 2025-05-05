import torch
import torch.nn as nn
from torch import Tensor


class LinearPredictionHead(nn.Module):
    """
    Applies a linear projection for time-series forecasting (prediction).
    Can handle two modes:
      1) Individual: Each variable has its own linear layer.
      2) Shared: All variables share the same linear layer.

    Args:
        individual (bool): If True, use a separate linear layer per variable.
        num_vars (int): Number of variables (channels).
        model_dim (int): Dimensionality of the model embeddings.
        num_patches (int): Number of patches in the sequence dimension.
        forecast_len (int): Length of the forecast horizon.
        dropout_rate (float, optional): Dropout probability before the linear layer.
    """

    def __init__(
        self,
        individual: bool,
        num_vars: int,
        model_dim: int,
        num_patches: int,
        forecast_len: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.individual = individual
        self.num_vars = num_vars
        head_dim = model_dim * num_patches

        if self.individual:
            # One linear head per variable
            self.flatteners = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.linears = nn.ModuleList()
            for _ in range(self.num_vars):
                self.flatteners.append(
                    nn.Flatten(start_dim=-2)
                )  # flattens [model_dim, num_patches]
                self.dropouts.append(nn.Dropout(dropout_rate))
                self.linears.append(nn.Linear(head_dim, forecast_len))
        else:
            # Single linear head for all variables
            self.flatten = nn.Flatten(start_dim=-2)  # flattens [model_dim, num_patches]
            self.dropout = nn.Dropout(dropout_rate)
            self.linear = nn.Linear(head_dim, forecast_len)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for forecasting:
        Expects x of shape [batch_size, num_vars, model_dim, num_patches].

        Returns:
            Tensor of shape [batch_size, forecast_len, num_vars].
        """
        if self.individual:
            # Handle each variable independently
            outputs = []
            for var_idx in range(self.num_vars):
                z = self.flatteners[var_idx](
                    x[:, var_idx, :, :]
                )
                z = self.linears[var_idx](z)
                z = self.dropouts[var_idx](z)
                outputs.append(z)  # Collect forecast per variable
            # Stack along dim=1
            out = torch.stack(outputs, dim=1)
        else:
            # Shared linear projection
            out = self.flatten(
                x
            ) 
            out = self.dropout(out)
            out = self.linear(out)
            # Expand to [batch_size, 1, forecast_len] if needed
            out = out.unsqueeze(1) if self.num_vars == 1 else out

        # Return shape: [batch_size, forecast_len, num_vars]
        return out.transpose(2, 1)
