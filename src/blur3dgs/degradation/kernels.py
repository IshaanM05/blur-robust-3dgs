"""Blur kernel generators: random-walk motion, disk defocus, Gaussian. Seeded, 3 severities.

All kernels are KxK (K odd), non-negative, sum to 1, centroid at the center pixel.
Severity values are in pixels at the `images_4` working resolution.
"""

import numpy as np

SEVERITIES = ("low", "medium", "high")
MOTION_LEN = {"low": 5.0, "medium": 11.0, "high": 19.0}
DEFOCUS_RADIUS = {"low": 1.5, "medium": 3.0, "high": 5.0}
GAUSS_SIGMA = {"low": 1.0, "medium": 2.0, "high": 3.0}
KINDS = ("motion", "defocus", "gaussian")


def _normalize(k: np.ndarray) -> np.ndarray:
    k = np.clip(k, 0, None)
    return (k / k.sum()).astype(np.float32)


def _splat(points: np.ndarray, size: int) -> np.ndarray:
    """Bilinearly splat unit masses at float (x, y) points (relative to center) into a size x size grid."""
    k = np.zeros((size, size), np.float64)
    c = size // 2
    for x, y in points:
        x, y = np.clip(x + c, 0, size - 1.001), np.clip(y + c, 0, size - 1.001)
        x0, y0 = int(x), int(y)
        fx, fy = x - x0, y - y0
        k[y0, x0] += (1 - fx) * (1 - fy)
        k[y0, x0 + 1] += fx * (1 - fy)
        k[y0 + 1, x0] += (1 - fx) * fy
        k[y0 + 1, x0 + 1] += fx * fy
    return k


def motion_trajectory(length: float, n_points: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth random-walk path of total arc length `length` (pixels), centered at its centroid. (n_points, 2)."""
    ang = rng.uniform(0, 2 * np.pi) + np.cumsum(rng.normal(0, 0.25, n_points))
    step = length / (n_points - 1)
    steps = step * np.stack([np.cos(ang), np.sin(ang)], 1)
    pts = np.concatenate([np.zeros((1, 2)), np.cumsum(steps[:-1], 0)])
    return pts - pts.mean(0)


def motion_kernel(size: int, length: float, rng: np.random.Generator) -> np.ndarray:
    pts = motion_trajectory(length, max(8, int(length * 8)), rng)
    return _normalize(_splat(pts, size))


def defocus_kernel(size: int, radius: float) -> np.ndarray:
    c = size // 2
    yy, xx = np.mgrid[:size, :size] - c
    r = np.sqrt(xx**2 + yy**2)
    return _normalize(np.clip(radius + 0.5 - r, 0, 1))  # anti-aliased disk edge


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    c = size // 2
    yy, xx = np.mgrid[:size, :size] - c
    return _normalize(np.exp(-(xx**2 + yy**2) / (2 * sigma**2)))


def make_kernel(kind: str, severity: str, size: int = 25, rng: np.random.Generator | None = None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    if kind == "motion":
        return motion_kernel(size, MOTION_LEN[severity], rng)
    if kind == "defocus":
        return defocus_kernel(size, DEFOCUS_RADIUS[severity])
    if kind == "gaussian":
        return gaussian_kernel(size, GAUSS_SIGMA[severity])
    raise ValueError(f"unknown kernel kind {kind!r}; expected one of {KINDS}")
