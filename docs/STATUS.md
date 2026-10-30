# Status and how to resume (paused 2026-10-31, ~04:00 IST)

Deadline: **Nov 6**. Plan: [plan.md](plan.md). Experiment results: [log.md](log.md). Figures: [figures/](figures/).

## Where we are
Phase 1 (2D + simulator + baselines + kernel estimators) and the 3D pipeline are **done, tested and working**.
First real result (bonsai, motion/medium, 10k iters, single seed):

| Method | PSNR | SSIM | LPIPS |
|---|---|---|---|
| vanilla 3DGS on blurry | 26.42 | 0.773 | 0.345 |
| Wiener pre-deblur, true kernel | 30.99 | 0.938 | 0.075 |
| ours, oracle kernels (upper bound) | 30.40 | 0.920 | 0.093 |
| ours, free-form kernels (all 3 settings tried) | 26.1-26.5 | ~0.77 | ~0.345 |
| **ours, polyline kernels, blind** | **29.93** | **0.910** | **0.111** |

Findings: forward model is correct; free-form per-view kernels collapse to a delta (scene absorbs the blur); the parametric polyline
kernel recovers the true kernels (L2 0.15 vs 0.99 for a delta) and lands within ~0.5 dB of the oracle. **Checkpoint decision: stay on the
proper plan** (the joint method works with the right kernel prior).

Caveats: one scene, one seed, one severity. Wiener with the TRUE kernel (31.0) still beats blind polyline (29.9); the fair blind baselines
(guessed kernel) are not trained yet.

## Settings that worked (use these as defaults)
`--kernel polyline --poly_init_len 6 --kernel_start 0 --lr_kernel 0.05 --iters 10000` (no regularizer). One 10k run takes ~4-7 min on the RTX 4080 laptop.

## Not done yet (resume here, in this order)
1. **Blind pre-deblur baselines on bonsai** (prepared, not trained): `deblur_wiener_guess.npy`, `deblur_rl_oracle.npy` exist in
   `data/blurred/bonsai/motion_medium/`; `deblur_rl_guess.npy` still needs `python scripts/prepare_deblur.py --scene bonsai --tag motion_medium --method rl --kernel guess`.
   Then train each with `scripts/train.py ... --input deblur_<name>` (`scripts/run_baselines_bonsai.sh` does it; interrupted, nothing finished).
2. **Main table:** repeat for `counter` (all severities) and `garden` (medium only); severities low/medium/high; motion + defocus + gaussian kernels.
   Needs `scripts/prepare_blur.py` per scene/kind/severity, then the methods: vanilla, wiener/RL oracle+guess, ours polyline, ours oracle, clean reference.
3. **Ablations (one scene):** polyline control points (n_ctrl, default 6), kernel size (ksize 25), regularizers (centroid/sparsity/TV; free-form fixes), shared vs
   per-view kernel (`--shared_kernel`), number of training views (`--max_views`).
4. **Analysis plots:** power spectra (`analysis/spectrum.py`), kernel-recovery vs severity (`analysis/kernel_recovery.py`), per-run tables.
5. **Optional / stretch:** Deblur-NeRF real scenes (qualitative), compare with official Deblurring-3DGS code, few-view ablation.
6. **Slides + README polish + rehearsal** (Nov 5), submit Nov 6. Add teammate name/roll number to README; teammate invite to @purvanyaTyagi was sent (check it was accepted).

## Environment (all local)
- Repo: `/home/ishaan/Desktop/blur-robust-3dgs` (private: https://github.com/IshaanM05/blur-robust-3dgs).
- `source .venv/bin/activate` (venv uses system site-packages; `src/` is on the path via a `.pth` file, not an editable install).
  If the venv is lost: `python3 -m venv .venv --system-site-packages && . .venv/bin/activate && pip install -r requirements.txt pycolmap` then add `<repo>/src` to a `.pth` file in site-packages.
- Data: `data/mipnerf360/{bonsai,counter,garden}` (extracted from `data/360_v2.zip`, 12.5 GB; the zip can be deleted to free space, all three scenes are already extracted).
  Blurred caches in `data/blurred/<scene>/<kind>_<severity>/` (git-ignored). Results in `results/` (git-ignored; numbers copied into `log.md`).
- First gsplat run compiles CUDA (~95 s, cached after). LPIPS AlexNet weights cached in `~/.cache/torch`.
- Entry points: `scripts/prepare_blur.py`, `prepare_deblur.py`, `train.py`, `exp_2d_blind.py`; batch scripts `scripts/run_*.sh` (they `nohup` in the background; wait for `DONE*` marker files).
- Tests: `python -m pytest -q tests` (18 pass).

## Gotchas learned
- Do not use `pgrep -f`/`pkill -f` with a pattern that appears in your own command line (matches itself).
- Free-form softmax kernels with near-delta init have vanishing gradients; Gaussian init + higher LR still collapse, so use the polyline kernel.
