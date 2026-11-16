# Status and how to resume

Last updated: 2026-11-05. **Deadline: Nov 6.** All experiments are finished (85 runs, no failures). Full tables: [results.md](results.md).
Figures: [figures/](figures/). Talk outline: [slides_outline.md](slides_outline.md). Draft deck: `slides.pptx` (rebuild with `scripts/build_slides.py`).

## Done
- Pipeline, baselines, three kernel estimators, main table (3 scenes), severity sweeps (bonsai, counter), blur types, ablations, free-form regularizers,
  3 training seeds, wrong-guess Wiener and noisy-input checks. Findings are written up in the README.
- Headline: on motion blur the polyline prior is within 0.1 to 0.5 dB of true-kernel results and 2 to 3.5 dB above blind baselines; on defocus and
  Gaussian blur our gain is about 1 dB and a correctly guessed Wiener is stronger, but a wrong guess drops it 2.8 to 4.3 dB below vanilla.

Nov 16: added SSIM/LPIPS tables to the README, a draft paper-style report ([report.md](report.md)), CI, `pip install -e .`
support, a Makefile and `configs/methods.yaml`. The up-to-date to-do list is [NEXT_STEPS.md](NEXT_STEPS.md).

## Left to do (not code)
1. Review the deck (`docs/slides.pptx`): slide copy and speaker notes are drafted; adjust wording to your own voice and rehearse the 10-minute talk.
2. Add the teammate's name and roll number to the README.
3. Package for Moodle: code (this repo), results (`docs/results.md`, `docs/figures/`), presentation (`docs/slides.pptx`).
4. Optional, needs download approval: real Deblur-NeRF scenes (qualitative only) and the official Deblurring-3DGS comparison.

## Rebuilding everything
```bash
python scripts/collect.py && python scripts/make_figures.py && python scripts/build_slides.py
```
Re-running an experiment script skips finished runs: `scripts/run_all.sh`, `scripts/run_extras.sh`, `scripts/run_extras2.sh`.

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
