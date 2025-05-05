import torch
import torch.nn as nn
from typing import Optional, Tuple


class RevIN(nn.Module):
    """
    Reverse Instance Normalization (RevIN). Supports two modes of shifting:
      1) Subtracting per-batch mean.
      2) Subtracting the last time-step value (if subtract_last=True).

    Args:
        num_features (int): Number of features or channels.
        eps (float): A small constant added for numerical stability.
        affine (bool): If True, RevIN has learnable affine parameters.
        subtract_last (bool): If True, subtract the last time-step value
                              instead of the mean during normalization.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        # Register buffers to store statistics on the correct device
        self.register_buffer("mean", None)
        self.register_buffer("stdev", None)
        self.register_buffer("last", None)

        if self.affine:
            # Initialize learnable affine parameters
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Forward pass of RevIN.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Time, Channels].
            mode (str): 'norm' to apply normalization or 'denorm' to revert it.

        Returns:
            torch.Tensor: Tensor after normalization or denormalization.
        """
        if mode == "norm":
            self.calculate_stats(x)
            x = self.normalize(x)
        elif mode == "denorm":
            x = self.denormalize(x)
        else:
            raise NotImplementedError(
                f"Unsupported mode: {mode}. Use 'norm' or 'denorm'."
            )
        return x


    def calculate_stats(self, x: torch.Tensor) -> None:
        """
        Calculates and stores mean/stdev or the last time-step value for normalization.

        Args:
            x (torch.Tensor): Input tensor of shape [Batch, Time, Channels].
        """
        # All dimensions except for batch and the last dimension (features).
        dims_to_reduce = tuple(range(1, x.ndim - 1))

        # Subtract either mean or last time-step:
        if self.subtract_last:
            # Store the last time-step's values: shape [Batch, 1, Channels]
            last_val = x[:, -1, :].unsqueeze(1)
            self.last = last_val.detach()
            # Mean not used in 'subtract_last' mode.
            self.mean = None
        else:
            # Store the mean over time (and possibly other dims except features)
            mean_val = torch.mean(x, dim=dims_to_reduce, keepdim=True)
            self.mean = mean_val.detach()
            # Last not used in 'mean' mode.
            self.last = None

        # Standard deviation calculation is the same for both modes.
        var_val = torch.var(x, dim=dims_to_reduce, keepdim=True, unbiased=False)
        stdev_val = torch.sqrt(var_val + self.eps)
        self.stdev = stdev_val.detach()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the input by subtracting the stored mean or last step,
        then dividing by stdev, and optionally applying an affine transform.

        Args:
            x (torch.Tensor): Input tensor to normalize.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean

        x = x / self.stdev

        if self.affine:
            x = x * self.affine_weight + self.affine_bias

        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reverts the normalization process. If affine is True, remove affine params first,
        then multiply by stdev and finally add the stored mean or last step.

        Args:
            x (torch.Tensor): Input tensor to denormalize.

        Returns:
            torch.Tensor: Denormalized tensor.
        """
        if self.affine:
            x = x - self.affine_bias
            # Add a small constant to handle near-zero values in weight:
            x = x / (self.affine_weight + self.eps * self.eps)

        x = x * self.stdev

        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean

        return x
