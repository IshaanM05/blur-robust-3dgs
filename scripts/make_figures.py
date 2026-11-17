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
LABEL = {
    "vanilla": "vanilla 3DGS",
    "wiener_oracle": "Wiener (true k)",
    "wiener_guess": "Wiener (guess sigma 2)",
    "rl_guess": "RL (guessed k)",
    "ours_polyline": "ours: polyline",
    "ours_gauss": "ours: gaussian",
    "ours_oracle_kernel": "ours: oracle k",
    "ours_free": "ours: free-form",
}
COLOR = {
    "vanilla": "#888888",
    "wiener_oracle": "#b58900",
    "wiener_guess": "#cb8f4a",
    "rl_guess": "#d8b080",
    "ours_polyline": "#1f77b4",
    "ours_gauss": "#17becf",
    "ours_oracle_kernel": "#2ca02c",
    "ours_free": "#9467bd",
}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def metric(scene, setting, method, key="psnr"):
    f = R / scene / setting / method / "metrics.json"
    return json.load(open(f))[key] if f.exists() else None


def qualitative(
    scene, setting="motion_medium", view=1, methods=("vanilla", "wiener_guess", "ours_polyline", "ours_oracle_kernel")
):
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
        ax[0, j].imshow(im)
        ax[0, j].set_title(t, fontsize=9)
        ax[1, j].imshow(im[y0 : y0 + h // 4, x0 : x0 + w // 4])
        for a in ax[:, j]:
            a.axis("off")
    fig.suptitle(f"{scene}, {setting}: held-out test view (bottom row: zoom)")
    plt.tight_layout()
    plt.savefig(OUT / f"qualitative_{scene}_{setting}.png", dpi=90)
    plt.close()


def severity(
    scene="bonsai",
    kind="motion",
    methods=("vanilla", "wiener_guess", "wiener_oracle", "ours_polyline", "ours_oracle_kernel"),
):
    sev = ["low", "medium", "high"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for m in methods:
        ys = [metric(scene, f"{kind}_{s}", m) for s in sev]
        xs = [i for i, y in enumerate(ys) if y is not None]
        if len(xs) >= 2:
            ax.plot(xs, [ys[i] for i in xs], "o-", color=COLOR[m], label=LABEL[m], lw=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels(sev)
    ax.set_xlabel("blur severity")
    ax.set_ylabel("PSNR on sharp test views (dB)")
    ax.set_title(f"{scene}, {kind} blur")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / f"severity_{scene}_{kind}.png", dpi=110)
    plt.close()


def blur_types(
    scene="bonsai", methods=("vanilla", "wiener_guess", "ours_polyline", "ours_gauss", "ours_oracle_kernel")
):
    kinds = ["motion", "defocus", "gaussian"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    wdt = 0.8 / len(methods)
    for i, m in enumerate(methods):
        ys = [metric(scene, f"{k}_medium", m) for k in kinds]
        ax.bar(np.arange(3) + i * wdt, [y or 0 for y in ys], wdt, color=COLOR[m], label=LABEL[m])
    ax.set_xticks(np.arange(3) + 0.4 - wdt / 2)
    ax.set_xticklabels(kinds)
    ax.set_ylabel("PSNR (dB)")
    lo = min([metric(scene, f"{k}_medium", "vanilla") or 99 for k in kinds]) - 2
    ax.set_ylim(max(lo, 15), None)
    ax.set_title(f"{scene}, medium severity, by blur type")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT / f"blur_types_{scene}.png", dpi=110)
    plt.close()


def spectra(
    scene="bonsai",
    setting="motion_medium",
    view=1,
    methods=("vanilla", "wiener_guess", "ours_polyline", "ours_oracle_kernel"),
):
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
    ax.set_xlabel("spatial frequency (cycles/pixel)")
    ax.set_ylabel("log10 power")
    ax.set_title(f"{scene} {setting}: radial power spectrum")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / f"spectrum_{scene}_{setting}.png", dpi=110)
    plt.close()


def kernels(
    scene="bonsai", settings=("motion_low", "motion_medium", "motion_high"), method="ours_polyline", views=(3, 90, 200)
):
    rows = [s for s in settings if (R / scene / s / method / "kernels.npy").exists()]
    if not rows:
        return
    fig, ax = plt.subplots(2 * len(rows), len(views), figsize=(2.3 * len(views), 2.3 * 2 * len(rows)))
    ax = np.atleast_2d(ax).reshape(2 * len(rows), len(views))
    for r, s in enumerate(rows):
        g = np.load(f"data/blurred/{scene}/{s}/kernels.npy")
        k = np.load(R / scene / s / method / "kernels.npy")
        for j, v in enumerate(views):
            ax[2 * r, j].imshow(g[v], cmap="gray")
            ax[2 * r + 1, j].imshow(k[v], cmap="gray")
            ax[2 * r, j].set_title(f"{s}, view {v}\ntrue kernel", fontsize=8)
            ax[2 * r + 1, j].set_title(f"learned ({LABEL[method]})", fontsize=8)
            ax[2 * r, j].axis("off")
            ax[2 * r + 1, j].axis("off")
    plt.tight_layout()
    plt.savefig(OUT / f"kernels_{scene}_{method}.png", dpi=90)
    plt.close()


def ablations(scene="bonsai", setting="motion_medium"):
    names = [
        ("vanilla", "vanilla"),
        ("ours_polyline", "ours (default)"),
        ("abl_ctrl3", "3 ctrl pts"),
        ("abl_ctrl12", "12 ctrl pts"),
        ("abl_k15", "kernel 15"),
        ("abl_k35", "kernel 35"),
        ("abl_shared", "shared kernel"),
        ("ours_free", "free-form kernel"),
    ]
    rows = [(l, metric(scene, setting, m)) for m, l in names if metric(scene, setting, m) is not None]
    views = [
        (f"{n} views", metric(scene, setting, f"abl_views{n}"), metric(scene, setting, f"vanilla_views{n}"))
        for n in (100, 50, 25)
    ]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
    if rows:
        ax[0].barh([r[0] for r in rows][::-1], [r[1] for r in rows][::-1], color="#1f77b4")
        ax[0].set_xlim(min(r[1] for r in rows) - 1, None)
        ax[0].set_xlabel("PSNR (dB)")
        ax[0].set_title("kernel ablations (bonsai, motion medium)")
    v = [x for x in views if x[1] is not None]
    if v:
        full_o, full_v = metric(scene, setting, "ours_polyline"), metric(scene, setting, "vanilla")
        xs = ["255 views"] + [x[0] for x in v][::-1]
        ax[1].plot(
            xs, [full_o] + [x[1] for x in v][::-1], "o-", color=COLOR["ours_polyline"], label="ours: polyline", lw=2
        )
        vv = [full_v] + [x[2] for x in v][::-1]
        if all(y is not None for y in vv):
            ax[1].plot(xs, vv, "o-", color=COLOR["vanilla"], label="vanilla", lw=2)
        ax[1].set_ylabel("PSNR (dB)")
        ax[1].set_title("number of training views")
        ax[1].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT / "ablations_bonsai.png", dpi=110)
    plt.close()


def seeds(scene="bonsai", setting="motion_medium", methods=("vanilla", "wiener_guess", "ours_polyline")):
    print("seed variability (training seeds; same blurred data):")
    for m in methods:
        v = [metric(scene, s, m) for s in (setting, setting + "_seed1", setting + "_seed2")]
        v = [x for x in v if x is not None]
        if len(v) > 1:
            print(f"  {m:15s} n={len(v)} mean {np.mean(v):.2f} std {np.std(v, ddof=1):.2f}  {np.round(v, 2)}")


def guess_sensitivity(scene="bonsai"):
    """Wiener with a guessed Gaussian kernel (sigma 1, 2, 4) vs our blind methods. The true Gaussian-blur sigma is 2."""
    sets = [("gaussian_medium", "Gaussian blur"), ("defocus_medium", "defocus")]

    def best_ours(setting):
        vals = [metric(scene, setting, m) for m in ("ours_polyline", "ours_gauss")]
        return max(v for v in vals if v is not None)

    bars = [
        ("vanilla 3DGS", lambda st: metric(scene, st, "vanilla"), COLOR["vanilla"]),
        ("Wiener, guess sigma 1", lambda st: metric(scene, st, "wiener_guess_s1"), "#e3c9a6"),
        ("Wiener, guess sigma 2", lambda st: metric(scene, st, "wiener_guess"), COLOR["wiener_guess"]),
        ("Wiener, guess sigma 4", lambda st: metric(scene, st, "wiener_guess_s4"), "#8c5a2b"),
        ("ours (blind)", best_ours, COLOR["ours_polyline"]),
    ]
    if any(f(st) is None for _, f, _ in bars for st, _ in sets):
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    width = 0.8 / len(bars)
    for i, (label, f, color) in enumerate(bars):
        ax.bar(np.arange(len(sets)) + i * width, [f(st) for st, _ in sets], width, color=color, label=label)
    ax.set_xticks(np.arange(len(sets)) + 0.4 - width / 2)
    ax.set_xticklabels([name for _, name in sets])
    ax.set_ylim(21, 29.5)
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"{scene}: classical deconvolution depends on guessing the kernel")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT / f"guess_sensitivity_{scene}.png", dpi=110)
    plt.close()


def kernel_recovery_vs_severity(scenes=("bonsai", "counter")):
    """Kernel error of the polyline estimate against severity, next to what a delta kernel (no blur estimate) scores."""
    from blur3dgs.analysis.kernel_recovery import best_shift_l2

    sev = ["low", "medium", "high"]
    delta = np.zeros((25, 25))
    delta[12, 12] = 1
    fig, ax = plt.subplots(1, len(scenes), figsize=(5.2 * len(scenes), 4), sharey=True)
    ax = np.atleast_1d(ax)
    drawn = False
    for a, scene in zip(ax, scenes, strict=False):
        learned = [metric(scene, f"motion_{s}", "ours_polyline", "kernel_best_shift_l2") for s in sev]
        if any(v is None for v in learned):
            continue
        drawn = True
        no_est = [
            np.mean([best_shift_l2(delta, k) for k in np.load(f"data/blurred/{scene}/motion_{s}/kernels.npy")])
            for s in sev
        ]
        a.plot(sev, no_est, "o--", color=COLOR["vanilla"], lw=2, label="delta kernel (no blur estimate)")
        a.plot(sev, learned, "o-", color=COLOR["ours_polyline"], lw=2, label="ours: polyline")
        a.set_title(f"{scene}, motion blur")
        a.set_xlabel("blur severity")
    if not drawn:
        plt.close()
        return
    ax[0].set_ylabel("kernel error (L2, best shift)")
    ax[0].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / "kernel_recovery_vs_severity.png", dpi=110)
    plt.close()


def blur_realisations(scene="bonsai", methods=("vanilla", "wiener_guess", "ours_polyline")):
    """Mean and std of PSNR over the original blur and the extra random blur realisations (_b1, _b2)."""
    settings = ["motion_medium", "motion_medium_b1", "motion_medium_b2"]
    stats = {}
    for m in methods:
        v = [metric(scene, s, m) for s in settings]
        v = [x for x in v if x is not None]
        if len(v) > 1:
            stats[m] = (np.mean(v), np.std(v, ddof=1), v)
    if not stats:
        return
    print("random blur realisations (different kernels and blurred photos):")
    for m, (mu, sd, v) in stats.items():
        print(f"  {m:14s} n={len(v)} mean {mu:.2f} std {sd:.2f}  {np.round(v, 2)}")
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for i, (m, (mu, sd, _)) in enumerate(stats.items()):
        ax.bar(i, mu, yerr=sd, color=COLOR[m], capsize=5)
    ax.set_xticks(range(len(stats)))
    ax.set_xticklabels([LABEL[m] for m in stats], rotation=15, fontsize=8)
    ax.set_ylim(min(s[0] for s in stats.values()) - 2, None)
    ax.set_ylabel("PSNR (dB), mean +- std over blurs")
    ax.set_title(f"{scene}, medium motion blur")
    plt.tight_layout()
    plt.savefig(OUT / f"blur_realisations_{scene}.png", dpi=110)
    plt.close()


def noise_sweep(scene="bonsai", methods=("vanilla", "wiener_guess", "ours_gauss", "ours_oracle_kernel")):
    """PSNR on Gaussian blur as noise is added to the blurry photographs."""
    levels = [
        ("gaussian_medium", 0.0),
        ("gaussian_medium_n0.02", 0.02),
        ("gaussian_medium_n0.05", 0.05),
        ("gaussian_medium_n0.1", 0.1),
    ]
    fig, ax = plt.subplots(figsize=(6, 4))
    drawn = False
    for m in methods:
        pts = [(sig, metric(scene, st, m)) for st, sig in levels]
        pts = [(x, y) for x, y in pts if y is not None]
        if len(pts) >= 3:
            drawn = True
            ax.plot(*zip(*pts, strict=False), "o-", color=COLOR[m], lw=2, label=LABEL[m])
    if not drawn:
        plt.close()
        return
    ax.set_xlabel("noise standard deviation added to the blurry photographs")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title(f"{scene}, Gaussian blur, medium")
    ax.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / f"noise_sweep_{scene}.png", dpi=110)
    plt.close()


def longer_training(scene="bonsai", methods=("vanilla", "wiener_guess", "ours_polyline")):
    """10k against 30k iterations (printed; the conclusion is whether the ordering and the gaps change)."""
    rows = [(m, metric(scene, "motion_medium", m), metric(scene, "motion_medium_30k", m)) for m in methods]
    rows = [r for r in rows if r[1] is not None and r[2] is not None]
    if rows:
        print("10k vs 30k iterations (bonsai, medium motion blur, PSNR):")
        for m, a, b in rows:
            print(f"  {m:14s} {a:.2f} -> {b:.2f}")


def real_scenes(root="data/deblurnerf"):
    """Held-out views of the real blurry scenes: blurry photo, vanilla 3DGS and ours (no sharp ground truth)."""
    names = sorted(
        p.name[: -len("_vanilla")]
        for p in Path("results/real_scenes").glob("*_vanilla/metrics.json".replace("/metrics.json", ""))
    )
    for scene in names:
        if not (R / "real_scenes" / f"{scene}_ours" / "metrics.json").exists():
            continue
        sc = load_scene(f"{root}/{scene}")
        views = [
            j for j in range(len(sc.test_idx)) if (R / "real_scenes" / f"{scene}_ours" / f"test_{j}.png").exists()
        ][:2]
        fig, ax = plt.subplots(2 * len(views), 3, figsize=(13, 4.4 * len(views)))
        ax = ax.reshape(2 * len(views), 3)
        for r, j in enumerate(views):
            ims = [sc.images[sc.test_idx[j]]] + [
                iio.imread(R / "real_scenes" / f"{scene}_{m}" / f"test_{j}.png") for m in ("vanilla", "ours")
            ]
            h, w = ims[0].shape[:2]
            y0, x0 = int(h * 0.3), int(w * 0.3)
            for c, (title, im) in enumerate(
                zip(["blurry photo (held out)", "vanilla 3DGS", "ours (polyline kernel)"], ims, strict=False)
            ):
                ax[2 * r, c].imshow(im)
                ax[2 * r, c].set_title(title if r == 0 else "")
                ax[2 * r + 1, c].imshow(im[y0 : y0 + h // 3, x0 : x0 + w // 3])
                ax[2 * r, c].axis("off")
                ax[2 * r + 1, c].axis("off")
        fig.suptitle(f"{scene}: real camera-motion blur (no sharp ground truth); rows 2 and 4 are zooms")
        plt.tight_layout()
        plt.savefig(OUT / f"real_{scene}.png", dpi=80)
        plt.close()


if __name__ == "__main__":
    for scene in ("bonsai", "counter", "garden"):
        qualitative(scene)
    severity()
    severity(scene="counter")
    guess_sensitivity()
    guess_sensitivity(scene="counter")
    kernel_recovery_vs_severity()
    blur_realisations()
    noise_sweep()
    longer_training()
    real_scenes()
    blur_types()
    spectra()
    kernels()
    ablations()
    seeds()
    for k in ("defocus", "gaussian"):
        qualitative(
            "bonsai",
            f"{k}_medium",
            methods=("vanilla", "wiener_guess", "ours_polyline", "ours_gauss", "ours_oracle_kernel"),
        )
    print("figures:", *sorted(p.name for p in OUT.glob("*.png")), sep="\n  ")
