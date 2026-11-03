"""Radially averaged power spectrum: shows which spatial frequencies each method recovers."""

import numpy as np


def radial_power_spectrum(img: np.ndarray, n_bins: int = 64):
    """img (H,W,3) in [0,1]. Returns (freqs in cycles/pixel, mean power per radial bin), on log10 scale."""
    g = img.mean(-1)
    g = (g - g.mean()) * np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1]))  # window to reduce leakage
    p = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = p.shape
    yy, xx = np.mgrid[:h, :w]
    r = np.sqrt(((yy - h // 2) / h) ** 2 + ((xx - w // 2) / w) ** 2)
    bins = np.linspace(0, 0.5, n_bins + 1)
    idx = np.digitize(r.ravel(), bins) - 1
    ok = (idx >= 0) & (idx < n_bins)
    sums = np.bincount(idx[ok], p.ravel()[ok], n_bins)
    cnt = np.maximum(np.bincount(idx[ok], minlength=n_bins), 1)
    return 0.5 * (bins[:-1] + bins[1:]), np.log10(sums / cnt + 1e-12)
