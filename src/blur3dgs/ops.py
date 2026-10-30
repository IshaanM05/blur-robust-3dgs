"""Shared differentiable ops. Blur is a true convolution with an odd-sized kernel centered at K//2."""
import torch
import torch.nn.functional as F


def apply_kernel(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Convolve img (C,H,W) or (B,C,H,W) with one KxK kernel (reflect padding, 'same' size)."""
    squeeze = img.dim() == 3
    x = img[None] if squeeze else img
    c, k = x.shape[1], kernel.shape[-1]
    assert kernel.shape[-2] == k and k % 2 == 1, "kernel must be square with odd size"
    p = k // 2
    x = F.pad(x, (p, p, p, p), mode="reflect")
    w = kernel.flip(-1).flip(-2)[None, None].expand(c, 1, k, k)  # conv2d is correlation -> flip
    y = F.conv2d(x, w.contiguous(), groups=c)
    return y[0] if squeeze else y
