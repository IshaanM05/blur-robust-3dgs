"""Fit an image with 2D Gaussians; optionally model blur with a known or learnable kernel (2D blind deconvolution).

Model: target ~= kernel * render(G). With a known kernel this is non-blind deconvolution through the Gaussian
representation; with `learn_kernel=True` the kernel is a free-form softmax kernel optimized jointly.
"""

import numpy as np
import torch

from ..kernel_est.module import FreeFormKernels
from ..kernel_est.regularizers import kernel_reg
from ..ops import apply_kernel
from .rasterizer import Gaussians2D


def fit_image(
    target: np.ndarray,
    n: int = 1024,
    iters: int = 1500,
    mode: str = "sum",
    kernel: np.ndarray | None = None,
    learn_kernel: bool = False,
    ksize: int = 15,
    reg: dict | None = None,
    lr: float = 5e-2,
    lr_kernel: float = 5e-2,
    seed: int = 0,
    device: str = "cuda",
    log_every: int = 0,
):
    """target: (H,W,3) float in [0,1]. Returns dict(render (H,W,3), kernel (K,K) or None, history, model)."""
    h, w = target.shape[:2]
    tgt = torch.from_numpy(target).permute(2, 0, 1).float().to(device)
    model = Gaussians2D(n, h, w, mode=mode, seed=seed).to(device)
    params = [{"params": [model.mu, model.log_s, model.theta, model.rgb, model.opacity, model.depth], "lr": lr}]
    est = None
    fixed = torch.from_numpy(kernel).float().to(device) if kernel is not None else None
    if learn_kernel:
        est = FreeFormKernels(1, ksize).to(device)
        params.append({"params": est.parameters(), "lr": lr_kernel})
    opt = torch.optim.Adam(params)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters, eta_min=lr * 0.02)
    hist = []
    for it in range(iters):
        opt.zero_grad(set_to_none=True)
        out = model()
        k = est(0) if est is not None else fixed
        pred = apply_kernel(out, k) if k is not None else out
        loss = (pred - tgt).abs().mean() + ((pred - tgt) ** 2).mean()
        if est is not None and reg:
            loss = loss + kernel_reg(k, **reg)
        loss.backward()
        opt.step()
        sched.step()
        hist.append(float(loss))
        if log_every and it % log_every == 0:
            print(f"it {it:5d} loss {hist[-1]:.5f}")
    with torch.no_grad():
        sharp = model().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        k_np = est(0) if est is not None else fixed
        k_np = k_np.cpu().numpy() if k_np is not None else None
    return {"render": sharp, "kernel": k_np, "history": hist, "model": model}
