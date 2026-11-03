"""Build all report/slide figures into docs/figures/ from whatever results exist (missing runs are skipped).
Usage: python scripts/make_figures.py
"""
import json
from pathlib import Path

import imageio.v2 as iio
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

from blur3dgs.analysis.spectrum import radial_power_spectrum
from blur3dgs.gs3d.dataset import load_scene

R, OUT = Path("results"), Path("docs/figures")
OUT.mkdir(parents=True, exist_ok=True)
LABEL = {"vanilla": "vanilla 3DGS", "wiener_oracle": "Wiener (true k)", "wiener_guess": "Wiener (guessed k)",
         "rl_guess": "RL (guessed k)", "ours_polyline": "ours: polyline", "ours_gauss": "ours: gaussian",
         "ours_oracle_kernel": "ours: oracle k", "ours_free": "ours: free-form"}
COLOR = {"vanilla": "#888888", "wiener_oracle": "#b58900", "wiener_guess": "#cb8f4a", "rl_guess": "#d8b080",
         "ours_polyline": "#1f77b4", "ours_gauss": "#17becf", "ours_oracle_kernel": "#2ca02c", "ours_free": "#9467bd"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def metric(scene, setting, method, key="psnr"):
    f = R / scene / setting / method / "metrics.json"
    return json.load(open(f))[key] if f.exists() else None


def qualitative(scene, setting="motion_medium", view=1, methods=("vanilla", "wiener_guess", "ours_polyline", "ours_oracle_kernel")):
    sc = load_scene(f"data/mipnerf360/{scene}")
    gt = sc.images[sc.test_idx[view]]
    items = [("ground truth (sharp)", gt)]
    for m in methods:
        f = R / scene / setting / m / f"test_{view}.png"
        if f.exists():
            im = iio.imread(f)
            p = peak_signal_noise_ratio(gt, im, data_range=255)
            items.append((f"{LABEL[m]}\n{p:.1f} dB (this view)", im))
    if len(items) < 3:
        return
    h, w = gt.shape[:2]
    y0, x0 = int(h * 0.35), int(w * 0.35)
    fig, ax = plt.subplots(2, len(items), figsize=(3.6 * len(items), 6.2))
    for j, (t, im) in enumerate(items):
        ax[0, j].imshow(im); ax[0, j].set_title(t, fontsize=9)
        ax[1, j].imshow(im[y0:y0 + h // 4, x0:x0 + w // 4])
        for a in ax[:, j]: a.axis("off")
    fig.suptitle(f"{scene}, {setting}: held-out test view (bottom row: zoom)")
    plt.tight_layout(); plt.savefig(OUT / f"qualitative_{scene}_{setting}.png", dpi=90); plt.close()


def severity(scene="bonsai", kind="motion", methods=("vanilla", "wiener_guess", "wiener_oracle", "ours_polyline", "ours_oracle_kernel")):
    sev = ["low", "medium", "high"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m in methods:
        ys = [metric(scene, f"{kind}_{s}", m) for s in sev]
        xs = [i for i, y in enumerate(ys) if y is not None]
        if len(xs) >= 2:
            ax.plot(xs, [ys[i] for i in xs], "o-", color=COLOR[m], label=LABEL[m], lw=2)
    ax.set_xticks(range(3)); ax.set_xticklabels(sev); ax.set_xlabel("blur severity"); ax.set_ylabel("PSNR on sharp test views (dB)")
    ax.set_title(f"{scene}, {kind} blur"); ax.legend(frameon=False, fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / f"severity_{scene}_{kind}.png", dpi=110); plt.close()


def blur_types(scene="bonsai", methods=("vanilla", "wiener_guess", "ours_polyline", "ours_gauss", "ours_oracle_kernel")):
    kinds = ["motion", "defocus", "gaussian"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    wdt = 0.8 / len(methods)
    for i, m in enumerate(methods):
        ys = [metric(scene, f"{k}_medium", m) for k in kinds]
        ax.bar(np.arange(3) + i * wdt, [y or 0 for y in ys], wdt, color=COLOR[m], label=LABEL[m])
    ax.set_xticks(np.arange(3) + 0.4 - wdt / 2); ax.set_xticklabels(kinds); ax.set_ylabel("PSNR (dB)")
    lo = min([metric(scene, f"{k}_medium", "vanilla") or 99 for k in kinds]) - 2
    ax.set_ylim(max(lo, 15), None); ax.set_title(f"{scene}, medium severity, by blur type"); ax.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(OUT / f"blur_types_{scene}.png", dpi=110); plt.close()


def spectra(scene="bonsai", setting="motion_medium", view=1, methods=("vanilla", "wiener_guess", "ours_polyline", "ours_oracle_kernel")):
    sc = load_scene(f"data/mipnerf360/{scene}")
    gt = sc.images[sc.test_idx[view]].astype(np.float32) / 255
    f, p = radial_power_spectrum(gt)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot(f, p, "k", lw=2.5, label="ground truth")
    for m in methods:
        fn = R / scene / setting / m / f"test_{view}.png"
        if fn.exists():
            f2, p2 = radial_power_spectrum(iio.imread(fn).astype(np.float32) / 255)
            ax.plot(f2, p2, color=COLOR[m], label=LABEL[m], lw=1.8)
    ax.set_xlabel("spatial frequency (cycles/pixel)"); ax.set_ylabel("log10 power"); ax.set_title(f"{scene} {setting}: radial power spectrum")
    ax.legend(frameon=False, fontsize=8); plt.tight_layout(); plt.savefig(OUT / f"spectrum_{scene}_{setting}.png", dpi=110); plt.close()


def kernels(scene="bonsai", settings=("motion_low", "motion_medium", "motion_high"), method="ours_polyline", views=(3, 90, 200)):
    rows = [s for s in settings if (R / scene / s / method / "kernels.npy").exists()]
    if not rows:
        return
    fig, ax = plt.subplots(2 * len(rows), len(views), figsize=(2.3 * len(views), 2.3 * 2 * len(rows)))
    ax = np.atleast_2d(ax).reshape(2 * len(rows), len(views))
    for r, s in enumerate(rows):
        g = np.load(f"data/blurred/{scene}/{s}/kernels.npy"); k = np.load(R / scene / s / method / "kernels.npy")
        for j, v in enumerate(views):
            ax[2 * r, j].imshow(g[v], cmap="gray"); ax[2 * r + 1, j].imshow(k[v], cmap="gray")
            ax[2 * r, j].set_title(f"{s}, view {v}\ntrue kernel", fontsize=8); ax[2 * r + 1, j].set_title(f"learned ({LABEL[method]})", fontsize=8)
            ax[2 * r, j].axis("off"); ax[2 * r + 1, j].axis("off")
    plt.tight_layout(); plt.savefig(OUT / f"kernels_{scene}_{method}.png", dpi=90); plt.close()


def ablations(scene="bonsai", setting="motion_medium"):
    names = [("vanilla", "vanilla"), ("ours_polyline", "ours (default)"), ("abl_ctrl3", "3 ctrl pts"), ("abl_ctrl12", "12 ctrl pts"),
             ("abl_k15", "kernel 15"), ("abl_k35", "kernel 35"), ("abl_shared", "shared kernel"), ("ours_free", "free-form kernel")]
    rows = [(l, metric(scene, setting, m)) for m, l in names if metric(scene, setting, m) is not None]
    views = [(f"{n} views", metric(scene, setting, f"abl_views{n}"), metric(scene, setting, f"vanilla_views{n}")) for n in (100, 50, 25)]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    if rows:
        ax[0].barh([r[0] for r in rows][::-1], [r[1] for r in rows][::-1], color="#1f77b4")
        ax[0].set_xlim(min(r[1] for r in rows) - 1, None); ax[0].set_xlabel("PSNR (dB)"); ax[0].set_title("kernel ablations (bonsai, motion medium)")
    v = [x for x in views if x[1] is not None]
    if v:
        full_o, full_v = metric(scene, setting, "ours_polyline"), metric(scene, setting, "vanilla")
        xs = ["255 views"] + [x[0] for x in v][::-1]
        ax[1].plot(xs, [full_o] + [x[1] for x in v][::-1], "o-", color=COLOR["ours_polyline"], label="ours: polyline", lw=2)
        vv = [full_v] + [x[2] for x in v][::-1]
        if all(y is not None for y in vv): ax[1].plot(xs, vv, "o-", color=COLOR["vanilla"], label="vanilla", lw=2)
        ax[1].set_ylabel("PSNR (dB)"); ax[1].set_title("number of training views"); ax[1].legend(frameon=False)
    plt.tight_layout(); plt.savefig(OUT / "ablations_bonsai.png", dpi=110); plt.close()


def seeds(scene="bonsai", setting="motion_medium", methods=("vanilla", "wiener_guess", "ours_polyline")):
    print("seed variability (training seeds; same blurred data):")
    for m in methods:
        v = [metric(scene, s, m) for s in (setting, setting + "_seed1", setting + "_seed2")]
        v = [x for x in v if x is not None]
        if len(v) > 1:
            print(f"  {m:15s} n={len(v)} PSNR mean {np.mean(v):.2f} std {np.std(v, ddof=1):.2f}  {['%.2f' % x for x in v]}")


if __name__ == "__main__":
    for scene in ("bonsai", "counter", "garden"):
        qualitative(scene)
    severity(); blur_types(); spectra(); kernels(); ablations(); seeds()
    for k in ("defocus", "gaussian"):
        qualitative("bonsai", f"{k}_medium", methods=("vanilla", "wiener_guess", "ours_polyline", "ours_gauss", "ours_oracle_kernel"))
    print("figures:", *sorted(p.name for p in OUT.glob("*.png")), sep="\n  ")
