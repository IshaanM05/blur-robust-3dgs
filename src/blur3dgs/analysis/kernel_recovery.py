"""Kernel-recovery error against ground-truth kernels."""

import numpy as np


def kernel_l2(est: np.ndarray, gt: np.ndarray) -> float:
    return float(np.sqrt(((est - gt) ** 2).sum()))


def kernel_psnr(est: np.ndarray, gt: np.ndarray) -> float:
    """PSNR with both kernels rescaled so their max equals 1."""
    e, g = est / est.max(), gt / gt.max()
    mse = ((e - g) ** 2).mean()
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def best_shift_l2(est: np.ndarray, gt: np.ndarray, max_shift: int = 3) -> float:
    """Min L2 over integer shifts: a kernel is only identifiable up to translation, so score it fairly."""
    best = np.inf
    for dy in range(-max_shift, max_shift + 1):
        for dx in range(-max_shift, max_shift + 1):
            best = min(best, kernel_l2(np.roll(est, (dy, dx), (0, 1)), gt))
    return float(best)
