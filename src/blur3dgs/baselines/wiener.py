"""Wiener deconvolution with a given (oracle or guessed) kernel."""
import numpy as np


def psf2otf(kernel: np.ndarray, shape: tuple) -> np.ndarray:
    """Zero-pad the kernel to `shape` and circularly shift its center to (0,0), then FFT."""
    out = np.zeros(shape, np.float64)
    k = kernel.shape[0]
    out[:k, :k] = kernel
    out = np.roll(out, (-(k // 2), -(k // 2)), (0, 1))
    return np.fft.fft2(out)


def wiener_deconv(img: np.ndarray, kernel: np.ndarray, nsr: float = 1e-2) -> np.ndarray:
    """img: (H,W,3) float in [0,1]. nsr: noise-to-signal power ratio (regularizer)."""
    k = kernel.shape[0]
    pad = k
    p = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="reflect").astype(np.float64)
    h = psf2otf(kernel, p.shape[:2])
    filt = np.conj(h) / (np.abs(h) ** 2 + nsr)
    out = np.stack([np.real(np.fft.ifft2(np.fft.fft2(p[..., c]) * filt)) for c in range(p.shape[2])], -1)
    return np.clip(out[pad:-pad, pad:-pad], 0, 1).astype(np.float32)
