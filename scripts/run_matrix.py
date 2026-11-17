"""Resumable experiment matrix. Skips any run whose metrics.json already exists.

Usage: python scripts/run_matrix.py --combos bonsai:motion:medium counter:motion:medium --methods vanilla ...
A combo is scene:kind:severity, optionally followed by :noise (e.g. bonsai:gaussian:medium:0.02).
Results: results/<scene>/<kind>_<severity>/<method>/metrics.json, with the training log next to it as <method>.log.
Methods are defined in configs/methods.yaml (pass --config to use another file).
--blur-seed N uses another random blur realisation (results go to <kind>_<severity>_bN).
--dry-run only prints the commands.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

CONFIG = Path(__file__).resolve().parent.parent / "configs" / "methods.yaml"
DRY_RUN = False
# while this file exists no new run is started (scripts/pause.sh creates it); runs already started finish normally
PAUSE_FILE = Path(os.environ.get("BLUR_PAUSE_FILE", "results/PAUSE"))


def load_methods(path=CONFIG):
    """Read the method config. Returns ({name: (pre-deblur step or None, train.py args)}, default method list).

    A pre-deblur step is (method, kernel, guess_sigma); guess_sigma=None keeps the default guess (2.0).
    """
    cfg = yaml.safe_load(Path(path).read_text())
    presets = cfg.get("presets", {})
    methods = {}
    for name, spec in cfg["methods"].items():
        args = list(presets[spec["preset"]]) if "preset" in spec else []
        args += spec.get("args", [])
        prep = None
        if "deblur" in spec:
            d = spec["deblur"]
            sigma = d.get("guess_sigma")
            prep = (d["method"], d["kernel"], float(sigma) if sigma is not None else None)
            args = ["--input", f"deblur_{d['method']}_{d['kernel']}" + (f"_s{sigma:g}" if sigma else "")] + args
        methods[name] = (prep, [str(x) for x in args])
    unknown = [m for m in cfg.get("default", []) if m not in methods]
    if unknown:
        raise ValueError(f"default lists unknown methods: {unknown}")
    return methods, cfg.get("default", [])


def run(cmd, log=None):
    if DRY_RUN:
        print("dry run:", " ".join(cmd), flush=True)
        return 0, 0.0
    start = time.time()
    with open(log or "/dev/null", "w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        print("FAILED:", " ".join(cmd), "(see", log, ")", flush=True)
    return rc, time.time() - start


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(CONFIG), help="YAML file with the method definitions")
    p.add_argument("--combos", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", help="method names from the config (default: its `default` list)")
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0, help="training seed")
    p.add_argument("--blur-seed", type=int, default=0, help="seed of the blur realisation (0 is the original blur)")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    p.add_argument("--suffix", default="", help="appended to the results folder name (e.g. _seed1)")
    a = p.parse_args()
    global DRY_RUN
    DRY_RUN = a.dry_run
    py = sys.executable
    methods, default = load_methods(a.config)
    a.methods = a.methods or default
    unknown = [m for m in a.methods if m not in methods]
    if unknown:
        p.error(f"unknown methods {unknown}; defined in {a.config}: {', '.join(methods)}")

    for combo in a.combos:
        if PAUSE_FILE.exists():
            print(f"paused: not starting {combo}", flush=True)
            return
        scene, kind, sev, *noise = combo.split(":")
        tag = f"{kind}_{sev}" + (f"_n{noise[0]}" if noise else "") + (f"_b{a.blur_seed}" if a.blur_seed else "")
        cache = Path("data/blurred") / scene / tag
        if not (cache / "blurred.npy").exists():
            cmd = [py, "scripts/prepare_blur.py", "--scene", scene, "--kind", kind, "--severity", sev]
            run(cmd + ["--seed", str(a.blur_seed)] + (["--noise", noise[0]] if noise else []))
        out_root = Path("results") / scene / (tag + a.suffix)
        out_root.mkdir(parents=True, exist_ok=True)

        for m in a.methods:
            if PAUSE_FILE.exists():
                print(f"paused: not starting {scene} {tag} {m}", flush=True)
                return
            out = out_root / m
            if (out / "metrics.json").exists():
                print("skip (done):", scene, tag, m, flush=True)
                continue
            prep, train_args = methods[m]
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
