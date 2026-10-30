"""Richardson-Lucy deconvolution with a given (oracle or guessed) kernel."""
import numpy as np
from scipy.signal import fftconvolve

from ..degradation.kernels import gaussian_kernel


def richardson_lucy(img: np.ndarray, kernel: np.ndarray, iters: int = 30, eps: float = 1e-6) -> np.ndarray:
    """img: (H,W,3) float in [0,1]. Multiplicative update x <- x * (K^T * (y / (K * x)))."""
    k = kernel.shape[0]
    pad = k
    y = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect").astype(np.float64) + eps
    kern = kernel.astype(np.float64)[..., None]
    kern_t = kern[::-1, ::-1]
    x = y.copy()
    for _ in range(iters):
        est = fftconvolve(x, kern, mode="same", axes=(0, 1)) + eps
        x = x * fftconvolve(y / est, kern_t, mode="same", axes=(0, 1))
    return np.clip(x[pad:-pad, pad:-pad] - eps, 0, 1).astype(np.float32)


def guessed_kernel(size: int = 25, sigma: float = 2.0) -> np.ndarray:
    """Kernel used by the 'blind' baselines: an isotropic Gaussian guess, since the true kernel is unknown."""
    return gaussian_kernel(size, sigma)
