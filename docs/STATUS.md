# Status and how to resume

Last updated: 2026-11-03 night. **Deadline: Nov 6.** Plan: [plan.md](plan.md). Full tables: [results.md](results.md). Figures: [figures/](figures/). Talk outline: [slides_outline.md](slides_outline.md).

## Where we are
Pipeline, baselines and the proposed method are done. Experiments are running unattended (see below).
Motion blur results (PSNR, held-out sharp views, 10k iterations, one seed):

| Setting | vanilla | best blind classical | ours (polyline, blind) | ours (oracle kernels) |
|---|---|---|---|---|
| bonsai, low | 28.49 | 25.20 | 30.49 | 30.76 |
| bonsai, medium | 26.42 | 26.84 | 29.93 | 30.40 |
| bonsai, high | 25.04 | 25.31 | 29.50 | 30.03 |
| counter, medium | 25.40 | 25.56 | 27.72 | 27.96 |
| garden, medium | 23.18 | 22.86 | 25.17 | 25.27 |

Findings: the forward model is correct (oracle kernels); free-form per-view kernels collapse to a delta; the polyline kernel prior recovers
the kernels (L2 ~0.15 vs 0.99 for a delta) and stays within ~0.1-0.5 dB of the oracle. The gain over vanilla grows with severity
(+2.0 dB low, +3.5 medium, +4.5 high). Defocus blur (a disk) is weaker for the polyline prior (27.54 vs vanilla 26.70, oracle 28.62);
the anisotropic-Gaussian kernel (`ours_gauss`) is queued to address it.

## What is running
```bash
scripts/run_all.sh        # main matrix, severity sweep, blur types, ablations, seeds   (~midnight)
scripts/run_extras.sh     # waits for run_all.sh; regularizer ablation, counter severities, extra baselines (~3 AM)
scripts/sync_results.sh   # every 45 min: regenerate docs/results.md and push if changed
```
All are resumable (finished runs have a `metrics.json` and are skipped). If the machine restarts, re-run the three commands
(`nohup scripts/run_all.sh > results/run_all.log 2>&1 &`, same for the others). Check progress with `tail results/run_all.log`.

## Still to do
1. When queues finish: `python scripts/collect.py && python scripts/make_figures.py`, review every figure and table.
2. Write the analysis (why free-form collapses; ablations; seed variation; defocus vs motion; limits) into the README results section.
3. Build the slide deck from [slides_outline.md](slides_outline.md) with the final numbers.
4. Optional, needs download approval: real Deblur-NeRF scenes (qualitative; data is LLFF format, may need a pose converter),
   comparison with the official Deblurring-3DGS code (custom CUDA build, risky).
5. Add teammate name/roll number to README; final push; package code + results + slides for Moodle (Nov 5-6).

## Gotchas
- `pkill -f` / `pgrep -f` with a pattern that appears in your own command line matches itself; use `pgrep -x <name>` or kill by PID.
- Do not edit a bash script while it is running.
- Free-form softmax kernels collapse to a delta in every setting tried; use the polyline (motion) or gaussian (defocus/blur) prior.

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
