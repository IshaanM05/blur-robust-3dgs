"""Kernel regularizers. Each takes a (K,K) kernel and returns a scalar; combine via `kernel_reg`."""

import torch


def centroid_loss(k: torch.Tensor) -> torch.Tensor:
    """Penalize center-of-mass offset from the center pixel (removes the shift ambiguity between image and kernel)."""
    n = k.shape[0]
    c = torch.arange(n, device=k.device, dtype=k.dtype) - n // 2
    cy = (k.sum(1) * c).sum()
    cx = (k.sum(0) * c).sum()
    return cx**2 + cy**2


def sparsity_loss(k: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    """Sum of k^p (p<1 promotes sparse kernels; for a sum-to-one kernel plain L1 is constant)."""
    return (k + 1e-8).pow(p).sum()


def tv_loss(k: torch.Tensor) -> torch.Tensor:
    return (k[1:] - k[:-1]).abs().sum() + (k[:, 1:] - k[:, :-1]).abs().sum()


def kernel_reg(k: torch.Tensor, w_centroid: float = 0.0, w_sparse: float = 0.0, w_tv: float = 0.0) -> torch.Tensor:
    out = k.new_zeros(())
    if w_centroid:
        out = out + w_centroid * centroid_loss(k)
    if w_sparse:
        out = out + w_sparse * sparsity_loss(k)
    if w_tv:
        out = out + w_tv * tv_loss(k)
    return out
