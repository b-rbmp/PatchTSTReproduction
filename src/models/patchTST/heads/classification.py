import torch.nn as nn
from torch import Tensor


class LinearClassificationHead(nn.Module):
    """
    Applies a linear projection for classification tasks.

    Args:
        num_vars (int): Number of input variables.
        model_dim (int): Dimensionality of model embeddings.
        num_classes (int): Number of classes for classification.
        dropout_rate (float): Dropout probability before the linear layer.
    """

    def __init__(
        self, num_vars: int, model_dim: int, num_classes: int, dropout_rate: float
    ):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(num_vars * model_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for classification:
        Expects x of shape [batch_size, num_vars, model_dim, num_patches].
        Uses only the last patch: x[:, :, :, -1].

        Returns:
            Tensor of shape [batch_size, num_classes].
        """
        # Select the last patch
        x = x[:, :, :, -1]
        x = self.flatten(x)
        x = self.dropout(x)
        output = self.linear(x)
        return output
