"""PSNR, SSIM, LPIPS on (H,W,3) float images in [0,1]."""
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_lpips_model = None


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(peak_signal_noise_ratio(a, b, data_range=1.0))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    return float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))


def lpips_dist(a: np.ndarray, b: np.ndarray, device: str = "cuda") -> float:
    global _lpips_model
    import lpips
    import torch
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None].float().to(device) * 2 - 1
    with torch.no_grad():
        return float(_lpips_model(t(a), t(b)))


def all_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    return {"psnr": psnr(pred, gt), "ssim": ssim(pred, gt), "lpips": lpips_dist(pred, gt)}
