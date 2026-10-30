"""Differentiable 2D Gaussian rasterizer (from scratch, dense, for small images).

Each Gaussian i has mean mu_i (pixels, x,y), log-scales s_i, rotation theta_i, color c_i, opacity logit o_i and
a depth z_i. Covariance is Sigma = R diag(s^2) R^T, so the footprint is
    G_i(p) = exp(-0.5 (p-mu)^T Sigma^-1 (p-mu)).
Two compositing modes:
  'sum'   : I(p) = sum_i c_i G_i(p)                         (order-free, as in GaussianImage)
  'alpha' : I(p) = sum_i c_i a_i prod_{j<i}(1-a_j), a_i = sigmoid(o_i) G_i(p), sorted by z (front to back)
Memory is O(H*W*N), fine for ~128x128 and N up to a few thousand.
"""
import torch
import torch.nn as nn


class Gaussians2D(nn.Module):
    def __init__(self, n: int, h: int, w: int, mode: str = "sum", init_rgb: torch.Tensor | None = None,
                 seed: int = 0):
        super().__init__()
        assert mode in ("sum", "alpha")
        self.h, self.w, self.mode = h, w, mode
        g = torch.Generator().manual_seed(seed)
        side = int(n**0.5) + 1
        ys, xs = torch.meshgrid(torch.linspace(0, h - 1, side), torch.linspace(0, w - 1, side), indexing="ij")
        grid = torch.stack([xs.flatten(), ys.flatten()], 1)[:n]
        grid = grid + 0.3 * (torch.rand(grid.shape, generator=g) - 0.5) * (w / side)
        self.mu = nn.Parameter(grid)
        self.log_s = nn.Parameter(torch.full((n, 2), float(torch.log(torch.tensor(1.2 * w / side)))))
        self.theta = nn.Parameter(torch.zeros(n))
        rgb = torch.rand(n, 3, generator=g) * 0.1 if init_rgb is None else init_rgb.clone()
        self.rgb = nn.Parameter(rgb / (1.0 if mode == "alpha" else max(1.0, n / (side * side) * 1.0)))
        self.opacity = nn.Parameter(torch.full((n,), 2.0))
        self.depth = nn.Parameter(torch.rand(n, generator=g))
        ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32),
                                indexing="ij")
        self.register_buffer("pix", torch.stack([xs.flatten(), ys.flatten()], 1))

    def footprints(self) -> torch.Tensor:
        """(HW, N) Gaussian responses G_i(p)."""
        d = self.pix[:, None, :] - self.mu[None]  # (HW,N,2)
        c, s = torch.cos(self.theta), torch.sin(self.theta)
        u = c * d[..., 0] + s * d[..., 1]
        v = -s * d[..., 0] + c * d[..., 1]
        sx, sy = self.log_s[:, 0].exp(), self.log_s[:, 1].exp()
        return torch.exp(-0.5 * ((u / sx) ** 2 + (v / sy) ** 2))

    def forward(self) -> torch.Tensor:
        """Returns the rendered image (3,H,W)."""
        g = self.footprints()
        if self.mode == "sum":
            img = g @ self.rgb
        else:
            order = torch.argsort(self.depth)
            a = (torch.sigmoid(self.opacity)[order][None] * g[:, order]).clamp(max=0.999)
            trans = torch.cumprod(torch.cat([torch.ones_like(a[:, :1]), 1 - a[:, :-1]], 1), 1)
            img = (a * trans) @ self.rgb[order]
        return img.t().reshape(3, self.h, self.w)
