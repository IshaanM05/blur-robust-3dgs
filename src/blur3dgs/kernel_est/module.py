"""Per-view learnable blur kernels: free-form KxK (softmax) and parametric polyline trajectory."""
import math

import torch
import torch.nn as nn


def _delta_logits(k: int, sharp: float = 6.0) -> torch.Tensor:
    """Logits whose softmax is close to a delta at the center, so training starts from 'no blur'."""
    z = torch.zeros(k, k)
    z[k // 2, k // 2] = sharp
    return z


class FreeFormKernels(nn.Module):
    """One KxK kernel per view; softmax enforces non-negativity and sum-to-one."""

    def __init__(self, n_views: int, ksize: int = 25, init_sharp: float = 6.0, init_sigma: float | None = None):
        super().__init__()
        self.ksize = ksize
        if init_sigma is not None:  # start from a mild Gaussian blur: unsaturated softmax -> useful gradients
            r = torch.arange(ksize, dtype=torch.float32) - ksize // 2
            z = -(r[:, None] ** 2 + r[None] ** 2) / (2 * init_sigma**2)
        else:
            z = _delta_logits(ksize, init_sharp)
        self.logits = nn.Parameter(z.expand(n_views, ksize, ksize).clone())

    def forward(self, idx: int) -> torch.Tensor:
        return torch.softmax(self.logits[idx].flatten(), 0).view(self.ksize, self.ksize)


class PolylineKernels(nn.Module):
    """Kernel = trajectory of P control points (px offsets from center) rasterized by bilinear splatting.

    Sampling T points along the polyline and splatting them bilinearly keeps the kernel differentiable
    w.r.t. the control points. Sum-to-one holds by construction.
    """

    def __init__(self, n_views: int, ksize: int = 25, n_ctrl: int = 6, n_samples: int = 64, init_len: float = 1.0):
        super().__init__()
        self.ksize, self.n_samples = ksize, n_samples
        t = torch.linspace(-0.5, 0.5, n_ctrl)
        base = torch.stack([t * init_len, torch.zeros_like(t)], 1)
        self.ctrl = nn.Parameter(base.expand(n_views, n_ctrl, 2).clone())

    def forward(self, idx: int) -> torch.Tensor:
        c = self.ctrl[idx]
        c = c - c.mean(0, keepdim=True)  # centroid at center removes the translation ambiguity
        seg = torch.linspace(0, len(c) - 1, self.n_samples, device=c.device)
        i0 = seg.floor().long().clamp(max=len(c) - 2)
        f = (seg - i0)[:, None]
        pts = c[i0] * (1 - f) + c[i0 + 1] * f  # (T,2) as (x,y)
        k, half = self.ksize, self.ksize // 2
        pos = (pts + half).clamp(0, k - 1.001)
        x0, y0 = pos[:, 0].floor().long(), pos[:, 1].floor().long()
        fx, fy = pos[:, 0] - x0, pos[:, 1] - y0
        kern = torch.zeros(k * k, device=c.device, dtype=c.dtype)
        for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
            kern = kern.index_add(0, (y0 + dy) * k + (x0 + dx), w)
        return (kern / kern.sum()).view(k, k)


class FixedKernels(nn.Module):
    """Known (oracle) per-view kernels, not trained. Gives the non-blind upper bound for the joint method."""

    def __init__(self, kernels):
        super().__init__()
        self.register_buffer("k", torch.as_tensor(kernels, dtype=torch.float32))

    def forward(self, idx: int) -> torch.Tensor:
        return self.k[idx]


class GaussianKernels(nn.Module):
    """Anisotropic Gaussian per view (log-sigma x/y + angle): a 3-parameter prior for defocus / Gaussian-like blur."""

    def __init__(self, n_views: int, ksize: int = 25, init_sigma: float = 1.5):
        super().__init__()
        self.ksize = ksize
        self.p = nn.Parameter(torch.tensor([math.log(init_sigma), math.log(init_sigma), 0.0]).repeat(n_views, 1))
        r = torch.arange(ksize, dtype=torch.float32) - ksize // 2
        yy, xx = torch.meshgrid(r, r, indexing="ij")
        self.register_buffer("xx", xx)
        self.register_buffer("yy", yy)

    def forward(self, idx: int) -> torch.Tensor:
        lsx, lsy, th = self.p[idx]
        c, s = torch.cos(th), torch.sin(th)
        u = c * self.xx + s * self.yy
        v = -s * self.xx + c * self.yy
        k = torch.exp(-0.5 * ((u / lsx.exp()) ** 2 + (v / lsy.exp()) ** 2))
        return k / k.sum()
