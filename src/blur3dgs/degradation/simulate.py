"""Apply per-view blur to clean multi-view images and keep the kernels as ground truth."""
import numpy as np
import torch

from ..ops import apply_kernel
from .kernels import make_kernel


def blur_views(images: np.ndarray, kind: str, severity: str, seed: int = 0, ksize: int = 25,
               noise_sigma: float = 0.0, per_view: bool = True):
    """images: (V,H,W,3) float in [0,1]. Returns (blurred (V,H,W,3), kernels (V,K,K)).

    per_view=False uses one shared kernel for all views (needed for the shared-vs-per-view ablation).
    Motion kernels differ per view (different random trajectories); defocus/gaussian are deterministic per
    severity, so per-view variation for them can be added by jittering the severity (not done here).
    """
    rng = np.random.default_rng(seed)
    v = len(images)
    kernels = np.stack([make_kernel(kind, severity, ksize, rng) for _ in range(v)]) if per_view else \
        np.repeat(make_kernel(kind, severity, ksize, rng)[None], v, 0)
    x = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    out = torch.stack([apply_kernel(x[i], torch.from_numpy(kernels[i])) for i in range(v)])
    out = out.permute(0, 2, 3, 1).numpy()
    if noise_sigma > 0:
        out = out + rng.normal(0, noise_sigma, out.shape).astype(np.float32)
    return np.clip(out, 0, 1).astype(np.float32), kernels
