import numpy as np
import pytest
import torch
from scipy.signal import fftconvolve

from blur3dgs.analysis.kernel_recovery import best_shift_l2
from blur3dgs.analysis.metrics import psnr
from blur3dgs.baselines.richardson_lucy import richardson_lucy
from blur3dgs.baselines.wiener import wiener_deconv
from blur3dgs.degradation.kernels import KINDS, SEVERITIES, make_kernel
from blur3dgs.degradation.simulate import blur_views
from blur3dgs.gaussian2d.rasterizer import Gaussians2D
from blur3dgs.kernel_est.module import FreeFormKernels, PolylineKernels
from blur3dgs.kernel_est.regularizers import centroid_loss
from blur3dgs.ops import apply_kernel


def make_test_image(h=64, w=64, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:h, :w]
    base = 0.5 + 0.4 * np.sin(xx / 5.0) * np.cos(yy / 7.0)
    img = np.stack([base, base[::-1], base.T[:h, :w]], -1) + 0.05 * rng.standard_normal((h, w, 3))
    img[20:40, 20:40] = 1.0  # hard edges
    return np.clip(img, 0, 1).astype(np.float32)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("sev", SEVERITIES)
def test_kernels_valid(kind, sev):
    k = make_kernel(kind, sev, 25, np.random.default_rng(1))
    assert k.shape == (25, 25) and k.min() >= 0 and abs(k.sum() - 1) < 1e-5
    c = np.arange(25) - 12
    assert abs((k.sum(1) * c).sum()) < 0.75 and abs((k.sum(0) * c).sum()) < 0.75  # centroid near center


def test_apply_kernel_matches_scipy():
    img = make_test_image()
    k = make_kernel("motion", "medium", 25, np.random.default_rng(2))
    ours = apply_kernel(torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(k)).permute(1, 2, 0).numpy()
    padded = np.pad(img, ((12, 12), (12, 12), (0, 0)), mode="reflect")
    ref = np.stack([fftconvolve(padded[..., c], k, mode="same") for c in range(3)], -1)[12:-12, 12:-12]
    assert np.abs(ours - ref).max() < 1e-4


def test_deconv_beats_blurry_with_oracle_kernel():
    img = make_test_image()
    k = make_kernel("gaussian", "medium", 25)
    blurred, _ = blur_views(img[None], "gaussian", "medium", seed=0)
    blurred = blurred[0]
    assert psnr(wiener_deconv(blurred, k, 1e-3), img) > psnr(blurred, img)
    assert psnr(richardson_lucy(blurred, k, 30), img) > psnr(blurred, img)


def test_blur_views_per_view_and_shared():
    imgs = np.stack([make_test_image(seed=s) for s in range(3)])
    _, k_pv = blur_views(imgs, "motion", "medium", seed=0, per_view=True)
    _, k_sh = blur_views(imgs, "motion", "medium", seed=0, per_view=False)
    assert not np.allclose(k_pv[0], k_pv[1]) and np.allclose(k_sh[0], k_sh[1])


@pytest.mark.parametrize("cls", [FreeFormKernels, PolylineKernels])
def test_kernel_modules_normalized_and_differentiable(cls):
    m = cls(2, 25)
    k = m(1)
    assert k.shape == (25, 25) and abs(float(k.sum()) - 1) < 1e-5 and float(k.min()) >= 0
    (k * torch.randn(25, 25)).sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.parameters())


def test_centroid_loss_zero_for_centered_delta():
    k = torch.zeros(9, 9)
    k[4, 4] = 1
    assert float(centroid_loss(k)) == 0.0
    k2 = torch.zeros(9, 9)
    k2[4, 6] = 1
    assert float(centroid_loss(k2)) == pytest.approx(4.0)


@pytest.mark.parametrize("mode", ["sum", "alpha"])
def test_rasterizer_shape_and_grad(mode):
    g = Gaussians2D(64, 32, 32, mode=mode)
    out = g()
    assert out.shape == (3, 32, 32)
    out.mean().backward()
    assert g.mu.grad is not None and torch.isfinite(g.mu.grad).all()


def test_best_shift_l2_ignores_translation():
    k = make_kernel("motion", "medium", 25, np.random.default_rng(3))
    assert best_shift_l2(np.roll(k, (1, 2), (0, 1)), k) < 1e-6


def test_gaussian_kernels_normalized_differentiable():
    from blur3dgs.kernel_est.module import GaussianKernels

    m = GaussianKernels(2, 25)
    k = m(1)
    assert k.shape == (25, 25) and abs(float(k.sum()) - 1) < 1e-5 and float(k.min()) >= 0
    (k * torch.randn(25, 25)).sum().backward()
    assert m.p.grad is not None and m.p.grad.abs().sum() > 0


def test_match_size_roundtrip():
    from blur3dgs.gs3d.train import _match_size

    k = make_kernel("gaussian", "medium", 25)
    assert np.allclose(_match_size(_match_size(k, 35), 25), k, atol=1e-6)
    assert _match_size(k, 15).shape == (15, 15)
