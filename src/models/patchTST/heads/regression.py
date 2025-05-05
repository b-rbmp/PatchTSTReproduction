import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple


class LinearRegressionHead(nn.Module):
    """
    Applies a linear projection for regression tasks.
    Optionally rescales outputs to a specified range using a SigmoidRange function.

    Args:
        num_vars (int): Number of input variables (channels).
        model_dim (int): Dimensionality of the model embeddings.
        output_dim (int): Size of the final output (e.g., 1 for univariate regression).
        dropout_rate (float): Dropout probability before the linear layer.
        y_range (Optional[Tuple[float, float]], optional): Range for the regression outputs.
            If provided, outputs are passed through SigmoidRange to map them to [low, high].
    """

    def __init__(
        self,
        num_vars: int,
        model_dim: int,
        output_dim: int,
        dropout_rate: float,
        y_range: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        self.y_range = y_range
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(num_vars * model_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for regression:
        Expects x of shape [batch_size, num_vars, model_dim, num_patches].
        Uses only the last patch: x[:, :, :, -1].

        Returns:
            Tensor of shape [batch_size, output_dim].
        """
        # Select the last patch
        x = x[:, :, :, -1]
        x = self.flatten(x)
        x = self.dropout(x)
        output = self.linear(x)

        # Optional range scaling
        if self.y_range is not None:
            output = SigmoidRange(*self.y_range)(output)
        return output


class SigmoidRange(nn.Module):
    """
    Scales the output of a sigmoid activation to a specified range [low, high].

    Args:
        low (float): The lower bound of the range.
        high (float): The upper bound of the range.

    Examples:
        >>> layer = SigmoidRange(low=0, high=10)
        >>> x = torch.tensor([-1.0, 0.0, 1.0])
        >>> y = layer(x)
        >>> print(y)  # Output: Tensor values between 0 and 10.
    """

    def __init__(self, low: float, high: float):
        super().__init__()
        if low >= high:
            raise ValueError(f"'low' ({low}) must be less than 'high' ({high}).")
        self.low = low
        self.high = high

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: scales the sigmoid output to the range [low, high].

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor scaled to the range [low, high].
        """
        sigmoid_output = torch.sigmoid(x)  # Outputs in [0, 1]
        return sigmoid_output * (self.high - self.low) + self.low
