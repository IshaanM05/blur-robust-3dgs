# Status and how to resume (paused 2026-10-31, ~16:55 IST; second pause)

Deadline: **Nov 6**. Plan: [plan.md](plan.md). Experiment results: [log.md](log.md). Figures: [figures/](figures/).

## Where we are
Pipeline, baselines and the proposed method are done and working. Full tables are auto-generated in [results.md](results.md)
(`python scripts/collect.py`). All numbers: held-out sharp views, 10k iterations, single seed, motion blur.

| Method | bonsai medium (PSNR) | counter medium (PSNR) | knows true kernel? |
|---|---|---|---|
| Wiener pre-deblur, true kernel (reference) | 30.99 | 28.19 | yes |
| ours, oracle kernels (upper bound) | 30.40 | 27.96 | yes |
| **ours, polyline kernels (blind)** | **29.93** | **27.72** | no |
| RL pre-deblur, guessed kernel | 26.84 | 25.56 | no |
| Wiener pre-deblur, guessed kernel | 26.82 | 25.46 | no |
| vanilla 3DGS on blurry input | 26.42 | 25.40 | no |
| ours, free-form kernels | 26.1-26.5 (collapse to delta) | not run | no |

Findings so far: forward model is correct (oracle kernels); free-form per-view kernels collapse to a delta; the polyline kernel prior recovers
kernels (L2 0.15 vs 0.99 for a delta) and beats the best blind classical baseline by ~3.1 dB (bonsai) and ~2.2 dB (counter), within 0.5 dB of the
oracle. Same ordering on both scenes. **Checkpoint decision: proper plan stays.**

**garden / motion_medium (partial, 2 of 6 done):** vanilla 23.18 dB, Wiener oracle 25.76 dB (~3M Gaussians, ~13 min per run).
Remaining there: `wiener_guess` (done, not yet in table: collect), `rl_guess`, `ours_polyline`, `ours_oracle_kernel`.

## Settings that worked (defaults in `scripts/run_matrix.py`)
`--kernel polyline --poly_init_len 6 --kernel_start 0 --lr_kernel 0.05 --iters 10000`. Runs take ~4 min (counter), ~6 min (bonsai), ~13-16 min (garden).

## RESUME HERE: one command
The matrix runner is resumable (skips anything with a `metrics.json`; a half-finished run restarts from scratch):
```bash
cd /home/ishaan/Desktop/blur-robust-3dgs && . .venv/bin/activate
nohup python scripts/run_matrix.py --combos garden:motion:medium bonsai:motion:low bonsai:motion:high > results/matrix_main2.log 2>&1 &
```
That finishes garden (3 runs left, ~45 min) then bonsai low/high severity (12 runs, ~70 min). Check progress with `cat results/matrix_main2.log`.
Do NOT start it twice. Stop it with `kill <pid of python scripts/run_matrix.py>` and also kill its child `scripts/train.py` or `scripts/prepare_deblur.py`.
Then `python scripts/collect.py` to refresh [results.md](results.md).

## After that (remaining work, in order)
1. **Other blur types:** `bonsai:defocus:medium`, `bonsai:gaussian:medium`. Expect polyline to struggle (a curve cannot represent a disk or blob);
   if so add an anisotropic-Gaussian parametric kernel as a third parameterization and report both. High value for the "when does it work" story.
2. **Ablations (bonsai medium motion):** polyline control points (`PolylineKernels(n_ctrl=...)`, add a CLI flag), kernel size (`--ksize`),
   regularizers (`--w_centroid/--w_sparse/--w_tv` for free-form), shared vs per-view (`--shared_kernel`), number of views (`--max_views`, e.g. 100/50/25).
   Add them as named methods in `scripts/run_matrix.py` (METHODS dict) so they are resumable too.
3. **Seeds:** rerun the 6 main methods on bonsai medium with `--seed 1 2` and `--suffix _seed1` for error bars (cheap, ~35 min per seed).
4. **Analysis plots:** radial power spectra (`analysis/spectrum.py`) of renders vs GT, kernel-recovery vs severity, qualitative render panels (see `docs/figures/`).
5. **Optional:** Deblur-NeRF real scenes (qualitative), official Deblurring-3DGS comparison.
6. **Slides + README polish + rehearsal** (Nov 5), submit Nov 6. Add teammate name/roll number to README (invite to @purvanyaTyagi was sent; confirm accepted).

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
