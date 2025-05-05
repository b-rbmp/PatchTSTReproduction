import math
import torch
import torch.nn as nn
from typing import Optional


def sine_cosine_pos_encoding(
    sequence_length: int, model_dim: int, normalize: bool = True
) -> torch.Tensor:
    """
    Generates a sine-cosine positional encoding.

    Args:
        sequence_length (int): Length of the sequence to encode.
        model_dim (int): Dimensionality of the model (embedding size).
        normalize (bool, optional): If True, subtract mean and scale by std*10. Defaults to True.

    Returns:
        torch.Tensor: Shape [sequence_length, model_dim] with positional encodings.
    """
    # Initialize a buffer for positional embeddings
    encoding = torch.zeros(sequence_length, model_dim)

    # Positions: [0, 1, 2, ... sequence_length - 1]
    positions = torch.arange(0, sequence_length, dtype=torch.float).unsqueeze(1)

    # Compute exponential terms for scaling sine and cosine waves
    div_term = torch.exp(
        torch.arange(0, model_dim, 2, dtype=torch.float)
        * -(math.log(10000.0) / model_dim)
    )

    # Even indices → sine, odd indices → cosine
    encoding[:, 0::2] = torch.sin(positions * div_term)
    encoding[:, 1::2] = torch.cos(positions * div_term)

    # normalization
    if normalize:
        mean_val = encoding.mean()
        std_val = encoding.std()
        encoding = (encoding - mean_val) / (std_val * 10)

    return encoding


def generate_positional_encoding(
    encoding_type: Optional[str], learnable: bool, sequence_length: int, model_dim: int
) -> nn.Parameter:
    """
    Creates a parameter holding a positional encoding, based on the specified type.

    Args:
        encoding_type (str or None): Name of the encoding type (e.g. 'zeros', 'normal', 'sincos').
            If None, creates a randomly initialized tensor and sets learnable=False by default.
        learnable (bool): Whether the returned encoding is trainable.
        sequence_length (int): Size of the sequence axis.
        model_dim (int): Dimensionality of the model (embedding size).

    Returns:
        nn.Parameter: Parameter containing the positional encoding
            of shape [sequence_length, model_dim] or [sequence_length, 1],
            depending on the type.
    """
    if encoding_type is None:
        # Random initialization (uniform) for entire [sequence_length, model_dim]
        encoding_data = torch.empty(sequence_length, model_dim)
        nn.init.uniform_(encoding_data, -0.02, 0.02)
        # Typically set learnable = False if encoding_type is None
        learnable = False

    elif encoding_type.lower() in ["zero", "zeros"]:
        if encoding_type.lower() == "zero":
            encoding_data = torch.empty(sequence_length, 1)
        else:
            encoding_data = torch.empty(sequence_length, model_dim)
        nn.init.uniform_(encoding_data, -0.02, 0.02)

    elif encoding_type.lower() in ["normal", "gauss"]:
        encoding_data = torch.zeros(sequence_length, 1)
        nn.init.normal_(encoding_data, mean=0.0, std=0.1)

    elif encoding_type.lower() == "uniform":
        encoding_data = torch.zeros(sequence_length, 1)
        nn.init.uniform_(encoding_data, a=0.0, b=0.1)

    elif encoding_type.lower() == "sincos":
        # Use the sine-cosine encoding
        encoding_data = sine_cosine_pos_encoding(
            sequence_length, model_dim, normalize=True
        )

    else:
        raise ValueError(
            f"Invalid encoding_type '{encoding_type}'. "
            "Valid options: None, 'zero', 'zeros', 'normal'/'gauss', 'uniform', 'sincos'."
        )

    return nn.Parameter(encoding_data, requires_grad=learnable)
