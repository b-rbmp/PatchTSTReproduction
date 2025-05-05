import torch.nn as nn
from torch import Tensor


class Transpose(nn.Module):
    """
    Transposes a tensor along given dimensions, optionally making it contiguous.

    Args:
        dims (Tuple[int, int]): Dimensions to transpose.
        make_contiguous (bool, optional): If True, calls contiguous() on the result. Defaults to False.

    Examples:
        >>> layer = Transpose(1, 2, make_contiguous=True)
        >>> x = torch.randn(2, 3, 4)
        >>> y = layer(x)
        >>> y.shape
        torch.Size([2, 4, 3])
        >>> y.is_contiguous()
        True
    """

    def __init__(self, dim1: int, dim2: int, make_contiguous: bool = False):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2
        self.make_contiguous = make_contiguous

    def forward(self, x: Tensor) -> Tensor:
        transposed = x.transpose(self.dim1, self.dim2)
        if self.make_contiguous:
            transposed = transposed.contiguous()
        return transposed
