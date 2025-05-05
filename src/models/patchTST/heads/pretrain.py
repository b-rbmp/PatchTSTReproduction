import torch
import torch.nn as nn
from torch import Tensor


class LinearPretrainHead(nn.Module):
    """
    Applies a linear projection for pretraining tasks.
    Takes the encoded features and reconstructs the original patch values.

    Args:
        model_dim (int): Dimensionality of the model embeddings.
        patch_length (int): Length of each patch in the input data.
        dropout_rate (float): Dropout probability before the linear layer.
    """

    def __init__(self, model_dim: int, patch_length: int, dropout_rate: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(model_dim, patch_length)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for pretraining reconstruction:
        Expects x of shape [batch_size, num_vars, model_dim, num_patches].

        Returns:
            Tensor of shape [batch_size, num_patches, num_vars, patch_length],
            which is the reconstructed patch values.
        """
        # Move num_patches to the second-to-last dimension
        x = x.transpose(2, 3)
        x = self.dropout(x)
        x = self.linear(x)
        # Permute to [batch_size, num_patches, num_vars, patch_length]
        return x.permute(0, 2, 1, 3)
