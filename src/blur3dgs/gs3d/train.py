"""3DGS training loop on gsplat with an optional per-view blur-kernel forward model.

Training minimizes  L(kernel_i * render(G, pose_i), blurry_i)  over Gaussians G and per-view kernels.
At test time the kernel is NOT applied: the sharp render is compared to the sharp held-out ground truth.
gsplat provides only the rasterizer and the densification strategy; everything else is ours.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy
from scipy.spatial import cKDTree

from ..analysis.kernel_recovery import best_shift_l2
from ..analysis.metrics import all_metrics
from ..kernel_est.module import FixedKernels, FreeFormKernels, GaussianKernels, PolylineKernels
from ..kernel_est.regularizers import kernel_reg
from ..ops import apply_kernel, ssim_loss
from .dataset import Scene

SH0 = 0.28209479177387814


def _match_size(k: np.ndarray, n: int) -> np.ndarray:
    """Center-crop or zero-pad a square kernel to n x n so kernels of different ksize can be compared (renormalized)."""
    m = k.shape[0]
    if m > n:
        a = (m - n) // 2
        k = k[a:a + n, a:a + n]
    elif m < n:
        a = (n - m) // 2
        k = np.pad(k, ((a, n - m - a), (a, n - m - a)))
    return k / max(k.sum(), 1e-12)


def init_params(scene: Scene, sh_degree: int, device: str):
    pts = torch.from_numpy(scene.points)
    d = cKDTree(scene.points).query(scene.points, k=4)[0][:, 1:].mean(1)
    scales = torch.log(torch.from_numpy(d).float().clamp(min=1e-6))[:, None].repeat(1, 3)
    quats = F.normalize(torch.randn(len(pts), 4), dim=-1)
    rgb = torch.from_numpy(scene.colors).float() / 255
    sh = torch.zeros(len(pts), (sh_degree + 1) ** 2, 3)
    sh[:, 0] = (rgb - 0.5) / SH0
    p = torch.nn.ParameterDict({
        "means": pts, "scales": scales, "quats": quats,
        "opacities": torch.logit(torch.full((len(pts),), 0.1)),
        "sh0": sh[:, :1], "shN": sh[:, 1:]}).to(device)
    return {k: torch.nn.Parameter(v) for k, v in p.items()}


def render(params, viewmat, K, hw, sh_degree, device):
    colors = torch.cat([params["sh0"], params["shN"]], 1)
    out, alpha, info = rasterization(
        params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
        torch.sigmoid(params["opacities"]), colors, viewmat[None], K[None], hw[1], hw[0],
        sh_degree=sh_degree, packed=False)
    return out[0].permute(2, 0, 1).clamp(0, 1), info  # (3,H,W)


def train(scene: Scene, train_images: np.ndarray, out_dir: str, iters: int = 10000, kernel: str = "none",
          ksize: int = 25, kernel_start: int = 500, lr_kernel: float = 1e-2, reg: dict | None = None,
          shared_kernel: bool = False, kernel_init_sigma: float = 1.0, poly_init_len: float = 1.0, n_ctrl: int = 6, gt_kernels: np.ndarray | None = None, sh_degree: int = 3, seed: int = 0,
          device: str = "cuda", eval_every: int = 0):
    """train_images: (n_train,H,W,3) uint8 images the model is fit to (blurry, deblurred, or sharp).
    kernel: 'none' (vanilla) | 'free' | 'polyline'. Returns metrics dict (sharp held-out views)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hw = scene.hw
    params = init_params(scene, sh_degree, device)
    lrs = {"means": 1.6e-4 * scene.scene_scale, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
           "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    opts = {k: torch.optim.Adam([{"params": params[k], "lr": lrs[k]}], eps=1e-15) for k in params}
    strat = DefaultStrategy(refine_stop_iter=int(iters * 0.5), reset_every=max(iters // 10, 300), verbose=False)
    strat.check_sanity(params, opts)
    state = strat.initialize_state(scene_scale=scene.scene_scale)
    sched = torch.optim.lr_scheduler.ExponentialLR(opts["means"], gamma=0.01 ** (1.0 / iters))

    n_tr = len(scene.train_idx)
    kmod, kopt = None, None
    if kernel != "none":
        n_k = 1 if shared_kernel else n_tr
        if kernel == "oracle":
            kmod = FixedKernels(gt_kernels).to(device)
        elif kernel == "free":
            kmod = FreeFormKernels(n_k, ksize, init_sigma=kernel_init_sigma).to(device)
        elif kernel == "gauss":
            kmod = GaussianKernels(n_k, ksize).to(device)
        else:
            kmod = PolylineKernels(n_k, ksize, n_ctrl=n_ctrl, init_len=poly_init_len).to(device)
        if kernel != "oracle":
            kopt = torch.optim.Adam(kmod.parameters(), lr=lr_kernel)

    vm = torch.from_numpy(scene.viewmats).to(device)
    Ks = torch.from_numpy(scene.Ks).to(device)
    t0, hist = time.time(), []
    for step in range(iters):
        i = int(np.random.randint(n_tr))
        vi = scene.train_idx[i]
        gt = torch.from_numpy(train_images[i]).to(device).permute(2, 0, 1).float() / 255
        deg = min(step // 1000, sh_degree)
        pred, info = render(params, vm[vi], Ks[vi], hw, deg, device)
        k = None
        if kmod is not None and step >= kernel_start:
            k = kmod(0 if shared_kernel else i)
            pred = apply_kernel(pred, k)
        loss = 0.8 * (pred - gt).abs().mean() + 0.2 * ssim_loss(pred, gt)
        if k is not None and reg:
            loss = loss + kernel_reg(k, **reg)
        strat.step_pre_backward(params, opts, state, step, info)
        loss.backward()
        for o in opts.values():
            o.step(); o.zero_grad(set_to_none=True)
        if kopt is not None:
            kopt.step(); kopt.zero_grad(set_to_none=True)
        strat.step_post_backward(params, opts, state, step, info, packed=False)
        sched.step()
        hist.append(float(loss))
        if step % 500 == 0:
            print(f"step {step:6d} loss {np.mean(hist[-100:]):.4f} n={len(params['means'])} t={time.time()-t0:.0f}s", flush=True)

    metrics = evaluate(params, scene, vm, Ks, sh_degree, device, out)
    metrics.update(train_seconds=time.time() - t0, n_gaussians=int(len(params["means"])), iters=iters, kernel=kernel)
    if kmod is not None:
        with torch.no_grad():
            ks = torch.stack([kmod(0 if shared_kernel else i) for i in range(n_tr)]).cpu().numpy()
        np.save(out / "kernels.npy", ks)
        if gt_kernels is not None:
            metrics["kernel_best_shift_l2"] = float(np.mean([best_shift_l2(_match_size(ks[j if not shared_kernel else 0], gt_kernels[j].shape[0]), gt_kernels[j])
                                                             for j in range(n_tr)]))
    json.dump(metrics, open(out / "metrics.json", "w"), indent=1)
    return metrics


@torch.no_grad()
def evaluate(params, scene, vm, Ks, sh_degree, device, out: Path, save_n: int = 4):
    rows = []
    for j, vi in enumerate(scene.test_idx):
        pred, _ = render(params, vm[vi], Ks[vi], scene.hw, sh_degree, device)
        p = pred.permute(1, 2, 0).cpu().numpy()
        g = scene.images[vi].astype(np.float32) / 255
        rows.append(all_metrics(p, g))
        if j < save_n:
            import imageio.v2 as imageio
            imageio.imwrite(out / f"test_{j}.png", (p * 255).astype(np.uint8))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
