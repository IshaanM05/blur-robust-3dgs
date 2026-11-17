# Next steps

State of the project: all planned experiments are finished (85 runs), the README, results tables, figures and a draft deck are
written. This file lists what is still open, in priority order, with effort estimates and the exact commands.
Read [README.md](../README.md) and [results.md](results.md) first, and [STATUS.md](STATUS.md) for the environment notes.

Ground rules
- Do not overclaim. This is a re-implementation plus analysis of an existing idea (Deblurring-3DGS, BAD-Gaussians, Deblur-NeRF), on
  synthetic blur. Keep that wording.
- Every number in the README, figures and slides must come from `results/` through `scripts/collect.py`. Rebuild with
  `python scripts/collect.py && python scripts/make_figures.py && python scripts/build_slides.py`.
- Raw outputs (`results/`, `data/`) are git-ignored. Commit code, `docs/results.md`, figures and the deck only.
- Runs are resumable; finished runs (those with a `metrics.json`) are skipped.

Heads-up for a fresh clone
- The git history was rewritten once (on Nov 4, to remove commit trailers). If you cloned before that, delete your copy and clone again;
  `git pull` on the old clone will not work cleanly.
- `data/` and `results/` are not in the repo. You must download Mip-NeRF 360 yourself (about 12.5 GB, see [data/README.md](../data/README.md)); the final
  numbers are in `docs/results.md`, but new experiments need the data and a GPU with about 12 GB of memory.
- The first training run compiles gsplat's CUDA extension, which takes about 2 minutes.
- The deck was only checked in LibreOffice, which substitutes the fonts (Cambria and Calibri). Open `docs/slides.pptx` in real PowerPoint or Google
  Slides and check every slide for text overflow before presenting.

Update Nov 16: §3.2 and §5 are done and §3.4 is drafted (see each item). Everything else needs the data, a GPU, the raw
`results/` folder or a person. Note that `results/` exists only on the machine that ran the experiments; `docs/results.md` is the
only copy of the numbers in the repo.

## 1. Before submission (must do, no code)
1. (Done) Team names and roll numbers are in the README and report.
2. Read the five core papers listed in the README, then review `docs/slides.pptx`. The slide copy and speaker notes are drafts;
   put them in your own words. Rehearse the 10-minute talk out loud at least twice.
3. Make sure the professor can see the work: the repo is private, so either make it public or add them, then submit the code, results
   and slides on Moodle.

## 2. Finish the planned experiments (GPU only, no downloads, about 3 hours total)
Setup: Python venv from `requirements.txt`, `export PYTHONPATH=$PWD/src`, Mip-NeRF 360 scenes extracted per
[data/README.md](../data/README.md) (bonsai, counter, garden, `images_4`). A 12 GB GPU is enough.

1. **Clean-input reference** (training on the sharp photos, the true ceiling; planned but never run at full length). About 25 minutes.
   ```bash
   for s in bonsai counter garden; do
     python scripts/train.py --scene $s --input clean --iters 10000 --out results/$s/clean/sharp_input
   done
   ```
   Then add a "sharp input" row to the README table and slide 6 (`scripts/build_slides.py`, the `methods` list).
2. **Defocus and Gaussian blur on counter and garden**, to check that the mixed result on bonsai is not a one-scene effect. About 1.5 hours.
   ```bash
   python scripts/run_matrix.py --combos counter:defocus:medium counter:gaussian:medium \
       --methods vanilla wiener_guess wiener_guess_s1 wiener_guess_s4 ours_polyline ours_gauss ours_oracle_kernel
   ```
3. **Fill the remaining table gaps**: `rl_guess` on counter low and high (`--combos counter:motion:low counter:motion:high --methods rl_guess`).
4. **Longer training** (30k iterations) on bonsai medium for vanilla, Wiener guess and ours, to check the conclusions are not an
   artefact of stopping at 10k (`--iters 30000 --suffix _30k`).
5. **Several random blurs, not just several training seeds.** The seed study varies only the optimiser seed; the blurred photos are
   the same. Make `prepare_blur.py --seed N` produce a separate cache tag (for example `motion_medium_b1`), then run the three main
   methods on two or three blur realisations and report mean and standard deviation.
6. **Higher noise.** Noise at sigma 0.02 did not hurt Wiener. Try 0.05 and 0.1 on Gaussian blur
   (`--combos bonsai:gaussian:medium:0.05`; Wiener may start to lose to ours here, which is worth knowing either way).

## 3. Analysis and presentation (no GPU)
1. A figure of kernel-recovery error against severity (the `kernel_l2` column in `results/all_results.csv`; add a function to `scripts/make_figures.py`).
   Needs the raw `results/` folder (not in the repo).
2. ~~SSIM and LPIPS tables in the README next to PSNR.~~ Done (Nov 16): both README tables now have an SSIM / LPIPS companion.
3. A qualitative figure per blur type with the learned kernels next to the true ones. Needs `results/` (learned kernels) or new runs.
4. A short report (4 to 6 pages, PDF) in paper style. **Drafted** (Nov 16) as [report.md](report.md); every number is from `results.md`.
   Still to do: read it and put it in your own words, add the roll number for the second author, then convert to PDF
   (for example `pandoc docs/report.md -o report.pdf --pdf-engine=xelatex`, or paste into Overleaf; the figures are linked, not embedded).
5. A short demo video: a fly-through of held-out views, vanilla versus ours versus sharp, side by side. Needs data, a GPU and trained scenes.

## 4. Next level (research depth)
Pick one or two; each is a real extension rather than more of the same.
1. **Real blurry scenes (largest credibility gain).** Deblur-NeRF provides real camera-motion-blur captures
   (https://drive.google.com/drive/folders/1_TkpcJnw504ZOWmgVTD7vWqPdzbk9Wx_, start with `real_camera_motion_blur/blurball`, which the
   authors use for their quick demo). The data is in LLFF format (`poses_bounds.npy`); `gs3d/dataset.py` reads COLMAP `sparse/0`, so
   either check whether the folders include `sparse/0` (undistorted pinhole cameras are required) or write an LLFF loader. There is no sharp
   ground truth, so the result is qualitative: vanilla versus ours renders and learned kernels. Download size is unknown; check before fetching.
2. **Spatially varying blur.** Real blur changes across the image. Replace the single kernel per photo by a small grid of kernels (for example
   3 x 3) blended across the image, and test on blur simulated with a varying kernel field. This addresses the main limitation in the README.
3. **Choosing the prior automatically.** The polyline prior suits motion blur and the Gaussian prior suits defocus. Fit both per scene and pick by
   training loss, or predict a mixture, so the user does not have to know the blur type.
4. **Depth-dependent defocus.** Make the Gaussian kernel width depend on the rendered depth, which matches how defocus works physically.
5. **Compare with the official Deblurring-3DGS code** (`benhenryL/Deblurring-3D-Gaussian-Splatting`; check that it is the official repository for the
   paper). It needs its own custom CUDA build and is the riskiest item; only attempt it with a spare day.
6. **Few views.** The gain shrinks with fewer photos (255: +3.5 dB, 50: +2.4, 25: +1.3). A depth or smoothness prior on the kernels could help the sparse regime.

## 5. Engineering polish (done Nov 16, check on first push)
- ~~GitHub Actions workflow that runs `ruff check` and `pytest` on every push.~~ `.github/workflows/ci.yml` (CPU torch). Written on a
  machine without ruff or torch, so neither was run locally: watch the first CI run and fix anything it reports.
- ~~Make `pip install -e .` work in a clean environment.~~ Dependencies are now in `pyproject.toml` (`pip install -e ".[dev]"`); the CI
  test job installs this way, so it is the check.
- ~~A `Makefile` target for each of `run_all`, `figures`, `slides`, `test`.~~ `make help` lists them. `tables`, `figures` and `slides`
  refuse to run when `results/` is empty, so a fresh clone cannot overwrite `docs/results.md`.
- ~~Move the experiment configs from `scripts/run_matrix.py` to a YAML file.~~ `configs/methods.yaml`; verified to produce exactly the
  same 23 methods, arguments and default list as the old hard-coded dict. Adds `pyyaml` as a dependency.
