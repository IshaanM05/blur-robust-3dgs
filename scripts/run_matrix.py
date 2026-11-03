"""Resumable experiment matrix. Skips any run whose metrics.json already exists.
Usage: python scripts/run_matrix.py --combos bonsai:motion:medium counter:motion:medium --methods vanilla ...
Each combo is scene:kind:severity. Results: results/<scene>/<kind>_<severity>/<method>/metrics.json (+ <method>.log).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

METHODS = {  # name -> (deblur prep (method, kernel) or None, train args)
    "vanilla": (None, ["--input", "blurred"]),
    "wiener_oracle": (("wiener", "oracle"), ["--input", "deblur_wiener_oracle"]),
    "wiener_guess": (("wiener", "guess"), ["--input", "deblur_wiener_guess"]),
    "rl_oracle": (("rl", "oracle"), ["--input", "deblur_rl_oracle"]),
    "rl_guess": (("rl", "guess"), ["--input", "deblur_rl_guess"]),
    "ours_polyline": (
        None,
        [
            "--input",
            "blurred",
            "--kernel",
            "polyline",
            "--poly_init_len",
            "6",
            "--kernel_start",
            "0",
            "--lr_kernel",
            "0.05",
        ],
    ),
    "ours_oracle_kernel": (None, ["--input", "blurred", "--kernel", "oracle", "--kernel_start", "0"]),
    # parametric anisotropic-Gaussian kernel (for defocus / gaussian blur)
    "ours_gauss": (None, ["--input", "blurred", "--kernel", "gauss", "--kernel_start", "0", "--lr_kernel", "0.02"]),
    # ablations of ours_polyline (run on one scene)
    "abl_ctrl3": (
        None,
        [
            "--input",
            "blurred",
            "--kernel",
            "polyline",
            "--poly_init_len",
            "6",
            "--kernel_start",
            "0",
            "--lr_kernel",
            "0.05",
            "--n_ctrl",
            "3",
        ],
    ),
    "abl_ctrl12": (
        None,
        [
            "--input",
            "blurred",
            "--kernel",
            "polyline",
            "--poly_init_len",
            "6",
            "--kernel_start",
            "0",
            "--lr_kernel",
            "0.05",
            "--n_ctrl",
            "12",
        ],
    ),
    "abl_k15": (
        None,
        [
            "--input",
            "blurred",
            "--kernel",
            "polyline",
            "--poly_init_len",
            "6",
            "--kernel_start",
            "0",
            "--lr_kernel",
            "0.05",
            "--ksize",
            "15",
        ],
    ),
    "abl_k35": (
        None,
        [
            "--input",
            "blurred",
            "--kernel",
            "polyline",
            "--poly_init_len",
            "6",
            "--kernel_start",
            "0",
            "--lr_kernel",
            "0.05",
            "--ksize",
            "35",
        ],
    ),
    "abl_shared": (
        None,
        [
            "--input",
            "blurred",
            "--kernel",
            "polyline",
            "--poly_init_len",
            "6",
            "--kernel_start",
            "0",
            "--lr_kernel",
            "0.05",
            "--shared_kernel",
        ],
    ),
    "abl_views100": (
        None,
        [
            "--input",
            "blurred",
            "--kernel",
            "polyline",
            "--poly_init_len",
            "6",
            "--kernel_start",
            "0",
            "--lr_kernel",
            "0.05",
            "--train_views",
            "100",
        ],
    ),
    "abl_views50": (
        None,
        [
            "--input",
            "blurred",
            "--kernel",
            "polyline",
            "--poly_init_len",
            "6",
            "--kernel_start",
            "0",
            "--lr_kernel",
            "0.05",
            "--train_views",
            "50",
        ],
    ),
    "abl_views25": (
        None,
        [
            "--input",
            "blurred",
            "--kernel",
            "polyline",
            "--poly_init_len",
            "6",
            "--kernel_start",
            "0",
            "--lr_kernel",
            "0.05",
            "--train_views",
            "25",
        ],
    ),
    # vanilla with fewer views, to see how the gap changes
    "vanilla_views50": (None, ["--input", "blurred", "--train_views", "50"]),
    "vanilla_views25": (None, ["--input", "blurred", "--train_views", "25"]),
}
DEFAULT = ["vanilla", "wiener_oracle", "wiener_guess", "rl_guess", "ours_polyline", "ours_oracle_kernel"]

p = argparse.ArgumentParser()
p.add_argument("--combos", nargs="+", required=True)
p.add_argument("--methods", nargs="+", default=DEFAULT)
p.add_argument("--iters", type=int, default=10000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--suffix", default="", help="appended to the results folder name (e.g. _seed1)")
a = p.parse_args()
py = sys.executable


def run(cmd, log=None):
    t = time.time()
    with open(log, "w") if log else open("/dev/null", "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        print("FAILED:", " ".join(cmd), "(see", log, ")", flush=True)
    return r.returncode, time.time() - t


for combo in a.combos:
    scene, kind, sev = combo.split(":")
    tag = f"{kind}_{sev}"
    cache = Path("data/blurred") / scene / tag
    if not (cache / "blurred.npy").exists():
        run([py, "scripts/prepare_blur.py", "--scene", scene, "--kind", kind, "--severity", sev, "--seed", str(a.seed)])
    out_root = Path("results") / scene / (tag + a.suffix)
    out_root.mkdir(parents=True, exist_ok=True)
    for m in a.methods:
        out = out_root / m
        if (out / "metrics.json").exists():
            print("skip (done):", scene, tag, m, flush=True)
            continue
        prep, targs = METHODS[m]
        if prep and not (cache / f"deblur_{prep[0]}_{prep[1]}.npy").exists():
            run(
                [
                    py,
                    "scripts/prepare_deblur.py",
                    "--scene",
                    scene,
                    "--tag",
                    tag,
                    "--method",
                    prep[0],
                    "--kernel",
                    prep[1],
                ]
            )
        cmd = [
            py,
            "scripts/train.py",
            "--scene",
            scene,
            "--tag",
            tag,
            "--iters",
            str(a.iters),
            "--seed",
            str(a.seed),
            "--out",
            str(out),
        ] + targs
        rc, dt = run(cmd, str(out_root / f"{m}.log"))
        print(f"done {scene} {tag} {m} rc={rc} {dt:.0f}s", flush=True)
print("ALL DONE", flush=True)
