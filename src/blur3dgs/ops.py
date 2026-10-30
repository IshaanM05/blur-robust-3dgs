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


def _gauss_window(size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return g[:, None] @ g[None]


def ssim_loss(a: torch.Tensor, b: torch.Tensor, size: int = 11) -> torch.Tensor:
    """1 - SSIM for (C,H,W) images in [0,1] (gaussian window, valid padding)."""
    c = a.shape[0]
    w = _gauss_window(size, 1.5, a.device)[None, None].expand(c, 1, size, size).contiguous()
    f = lambda x: F.conv2d(x[None], w, groups=c)[0]
    mu_a, mu_b = f(a), f(b)
    va, vb, cov = f(a * a) - mu_a**2, f(b * b) - mu_b**2, f(a * b) - mu_a * mu_b
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))
    return 1 - s.mean()
