"""Resumable experiment matrix. Skips any run whose metrics.json already exists.

Usage: python scripts/run_matrix.py --combos bonsai:motion:medium counter:motion:medium --methods vanilla ...
A combo is scene:kind:severity, optionally followed by :noise (e.g. bonsai:gaussian:medium:0.02).
Results: results/<scene>/<kind>_<severity>/<method>/metrics.json, with the training log next to it as <method>.log.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

BLURRED = ["--input", "blurred"]
POLYLINE = BLURRED + ["--kernel", "polyline", "--poly_init_len", "6", "--kernel_start", "0", "--lr_kernel", "0.05"]
FREE_FORM = BLURRED + ["--kernel", "free", "--lr_kernel", "0.1"]
GAUSS = BLURRED + ["--kernel", "gauss", "--kernel_start", "0", "--lr_kernel", "0.02"]


def deblur(method, kernel, guess_sigma=None):
    """Pre-deblur baseline as (prepare_deblur step, train.py args). guess_sigma=None keeps the default guess (2.0)."""
    name = f"{method}_{kernel}" + (f"_s{guess_sigma:g}" if guess_sigma else "")
    return (method, kernel, guess_sigma), ["--input", f"deblur_{name}"]


# method name -> (pre-deblur step or None, train.py arguments)
METHODS = {
    "vanilla": (None, BLURRED),
    "wiener_oracle": deblur("wiener", "oracle"),
    "wiener_guess": deblur("wiener", "guess"),
    "wiener_guess_s1": deblur("wiener", "guess", 1.0),
    "wiener_guess_s4": deblur("wiener", "guess", 4.0),
    "rl_oracle": deblur("rl", "oracle"),
    "rl_guess": deblur("rl", "guess"),
    "ours_polyline": (None, POLYLINE),
    "ours_gauss": (None, GAUSS),
    "ours_oracle_kernel": (None, BLURRED + ["--kernel", "oracle", "--kernel_start", "0"]),
    # ablations of the polyline method
    "abl_ctrl3": (None, POLYLINE + ["--n_ctrl", "3"]),
    "abl_ctrl12": (None, POLYLINE + ["--n_ctrl", "12"]),
    "abl_k15": (None, POLYLINE + ["--ksize", "15"]),
    "abl_k35": (None, POLYLINE + ["--ksize", "35"]),
    "abl_shared": (None, POLYLINE + ["--shared_kernel"]),
    "abl_views100": (None, POLYLINE + ["--train_views", "100"]),
    "abl_views50": (None, POLYLINE + ["--train_views", "50"]),
    "abl_views25": (None, POLYLINE + ["--train_views", "25"]),
    "vanilla_views50": (None, BLURRED + ["--train_views", "50"]),
    "vanilla_views25": (None, BLURRED + ["--train_views", "25"]),
    # free-form kernel with regularizers: does a better prior alone rescue it?
    "abl_free_sparse": (None, FREE_FORM + ["--w_sparse", "0.01"]),
    "abl_free_tv": (None, FREE_FORM + ["--w_tv", "0.05"]),
    "abl_free_all": (None, FREE_FORM + ["--w_centroid", "1.0", "--w_sparse", "0.01", "--w_tv", "0.05"]),
}
DEFAULT = ["vanilla", "wiener_guess", "ours_polyline", "ours_oracle_kernel"]


def run(cmd, log=None):
    start = time.time()
    with open(log or "/dev/null", "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        print("FAILED:", " ".join(cmd), "(see", log, ")", flush=True)
    return rc, time.time() - start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--combos", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", default=DEFAULT)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--suffix", default="", help="appended to the results folder name (e.g. _seed1)")
    a = p.parse_args()
    py = sys.executable

    for combo in a.combos:
        scene, kind, sev, *noise = combo.split(":")
        tag = f"{kind}_{sev}" + (f"_n{noise[0]}" if noise else "")
        cache = Path("data/blurred") / scene / tag
        if not (cache / "blurred.npy").exists():
            cmd = [py, "scripts/prepare_blur.py", "--scene", scene, "--kind", kind, "--severity", sev]
            run(cmd + ["--seed", str(a.seed)] + (["--noise", noise[0]] if noise else []))
        out_root = Path("results") / scene / (tag + a.suffix)
        out_root.mkdir(parents=True, exist_ok=True)

        for m in a.methods:
            out = out_root / m
            if (out / "metrics.json").exists():
                print("skip (done):", scene, tag, m, flush=True)
                continue
            prep, train_args = METHODS[m]
            if prep:
                method, kernel, sigma = prep
                suffix = f"_s{sigma:g}" if sigma else ""
                if not (cache / f"deblur_{method}_{kernel}{suffix}.npy").exists():
                    cmd = [py, "scripts/prepare_deblur.py", "--scene", scene, "--tag", tag]
                    run(
                        cmd
                        + ["--method", method, "--kernel", kernel]
                        + (["--guess_sigma", str(sigma)] if sigma else [])
                    )
            cmd = [py, "scripts/train.py", "--scene", scene, "--tag", tag, "--iters", str(a.iters)]
            rc, seconds = run(cmd + ["--seed", str(a.seed), "--out", str(out)] + train_args, str(out_root / f"{m}.log"))
            print(f"done {scene} {tag} {m} rc={rc} {seconds:.0f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
