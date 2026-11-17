# Blur-Robust 3D Gaussian Splatting via Joint Per-View Blur-Kernel Estimation

CS 663 (Digital Image Processing) course project, IIT Bombay.

3D Gaussian Splatting (3DGS) rebuilds a scene from many posed photos, but it assumes the photos are sharp. When they
are blurred by camera motion or defocus, vanilla 3DGS bakes the blur into the scene. We treat blur as an image
restoration problem: each photo is the sharp rendering convolved with an unknown per-photo kernel,

    y_i = k_i * render(G, pose_i) + noise

and we optimise the Gaussians `G` and the kernels `k_i` together. The kernel is dropped at test time, so the scene is
rendered sharp. The question we study is **when joint estimation works, and why**.

## Team
- Ishaan Mondal (24B2118)
- [@purvanyaTyagi](https://github.com/purvanyaTyagi)

## Main result

Motion blur on three Mip-NeRF 360 scenes, PSNR (dB) on held-out sharp views, 10k iterations. "Blind" methods do not
know the true kernel; the last two rows use it and are upper-bound references.

| Method | bonsai | counter | garden |
|---|---|---|---|
| Vanilla 3DGS on blurry photos | 26.42 | 25.40 | 23.18 |
| Wiener deblur with a guessed kernel, then 3DGS | 26.82 | 25.46 | 22.86 |
| RL deblur with a guessed kernel, then 3DGS | 26.84 | 25.56 | 23.09 |
| **Ours, polyline kernel (blind)** | **29.93** | **27.72** | **25.17** |
| Ours with the true kernels | 30.40 | 27.96 | 25.27 |
| Wiener deblur with the true kernel, then 3DGS | 30.99 | 28.19 | 25.76 |

The same runs in SSIM (higher is better) / LPIPS (lower is better):

| Method | bonsai | counter | garden |
|---|---|---|---|
| Vanilla 3DGS on blurry photos | 0.773 / 0.345 | 0.745 / 0.365 | 0.528 / 0.563 |
| Wiener deblur with a guessed kernel, then 3DGS | 0.822 / 0.222 | 0.772 / 0.257 | 0.601 / 0.440 |
| RL deblur with a guessed kernel, then 3DGS | 0.821 / 0.230 | 0.776 / 0.262 | 0.608 / 0.458 |
| **Ours, polyline kernel (blind)** | **0.910 / 0.111** | **0.859 / 0.148** | **0.712 / 0.283** |
| Ours with the true kernels | 0.920 / 0.093 | 0.870 / 0.128 | 0.731 / 0.250 |
| Wiener deblur with the true kernel, then 3DGS | 0.938 / 0.075 | 0.894 / 0.112 | 0.814 / 0.148 |

- The polyline prior stays within 0.1 to 0.5 dB of the true-kernel result and 2 to 3.5 dB above every blind baseline.
- The gain over vanilla grows with blur severity on bonsai: +2.0 dB (low), +3.5 dB (medium), +4.5 dB (high).
- Over three training seeds on bonsai: ours 30.02 +- 0.08, vanilla 26.29 +- 0.18, guessed-kernel Wiener 26.76 +- 0.06.
- A guessed kernel can be worse than doing nothing: on garden, guessed-kernel Wiener (22.86) is below vanilla (23.18).

Full tables for every run are in [docs/results.md](docs/results.md); figures are in [docs/figures](docs/figures). A
paper-style write-up (draft) is in [docs/report.md](docs/report.md).

### Why the kernel parametrisation matters

A free-form per-photo kernel (a K x K softmax) is the obvious choice and it fails: the kernel collapses to a delta and
the scene absorbs the blur, which is the trivial solution of joint blind deconvolution (Levin et al., 2009). This
holds across learning rates, initialisations and regularisers (sparsity, TV, both): 25.9 to 26.5 dB, the same as vanilla.
A structured prior fixes it. The polyline kernel is an exposure path of 6 control points splatted into a kernel.

Ablations on bonsai (medium motion blur):

| Variant | PSNR |
|---|---|
| Default (6 control points, 25 px kernel) | 29.93 |
| 3 / 12 control points | 29.77 / 30.11 |
| Kernel size 15 / 35 | 29.95 / 30.08 |
| One kernel shared by all photos | 26.72 |
| Free-form kernel + sparsity / TV / both | 26.49 / 25.94 / 26.46 |
| 100 / 50 / 25 training views (vanilla at 50 / 25) | 29.52 / 27.46 / 23.64 (25.09 / 22.32) |

The method is insensitive to the polyline size, a per-photo kernel is essential, and the gain shrinks as photos get
scarce, because the kernels are only identifiable when enough views must agree on one scene.

### Defocus and Gaussian blur: a more mixed result

Same scene (bonsai, medium severity). The Wiener baseline guesses an isotropic Gaussian kernel. The true kernel of the
Gaussian-blur set is exactly a Gaussian with sigma 2, so a guess of sigma 2 is correct there, and it is a close match to
the defocus disk. We therefore also ran deliberately wrong guesses (sigma 1 and 4).

| Method | Gaussian blur | Defocus |
|---|---|---|
| Vanilla 3DGS | 26.24 | 26.70 |
| Wiener, guess sigma 1 | 26.51 | 26.45 |
| Wiener, guess sigma 2 (matches the truth) | 28.25 | 28.01 |
| Wiener, guess sigma 4 | 23.46 | 22.43 |
| Ours, polyline kernel (blind) | 27.29 | 27.54 |
| Ours, Gaussian kernel (blind) | 27.21 | 27.25 |
| Ours with the true kernels | 27.12 | 28.62 |

SSIM / LPIPS for the same runs:

| Method | Gaussian blur | Defocus |
|---|---|---|
| Vanilla 3DGS | 0.759 / 0.386 | 0.780 / 0.334 |
| Wiener, guess sigma 1 | 0.782 / 0.344 | 0.775 / 0.256 |
| Wiener, guess sigma 2 (matches the truth) | 0.855 / 0.242 | 0.852 / 0.232 |
| Wiener, guess sigma 4 | 0.676 / 0.404 | 0.650 / 0.413 |
| Ours, polyline kernel (blind) | 0.804 / 0.259 | 0.813 / 0.232 |
| Ours, Gaussian kernel (blind) | 0.815 / 0.213 | 0.806 / 0.194 |
| Ours with the true kernels | 0.820 / 0.192 | 0.872 / 0.148 |

- Our blind methods are 0.5 to 1 dB above vanilla on these blurs, a smaller gain than on motion blur, and they do not beat
  Wiener when Wiener is given the right kernel.
- Classical deconvolution is fragile: a guess of sigma 4 lands 2.8 dB (Gaussian) and 4.3 dB (defocus) below vanilla.
  Ours needs no guess, so it is the safer choice when the kernel is unknown.
- The perceptual metric tells a slightly different story: the blind Gaussian kernel has a lower (better) LPIPS than the
  correctly guessed Wiener (0.213 against 0.242 on Gaussian blur, 0.194 against 0.232 on defocus), although Wiener wins on
  PSNR and SSIM.
- The Gaussian kernel recovers the kernel better than the polyline (error 0.03 to 0.09 against 0.14 to 0.16) but does not
  score higher, so kernel accuracy is not what limits PSNR on these blurs.
- Even with the true kernel, the in-loop method (27.12) trails the Wiener pre-deblur (28.25) on Gaussian blur. A smooth
  blur removes high frequencies, and an explicit inverse filter can amplify them on this noise-free synthetic data.
- Adding sensor-like noise (sigma 0.02) did not change this on Gaussian blur: Wiener 28.05, ours 27.36, vanilla 26.33.

### Frequency view

The radial power spectrum of a rendered test view ([docs/figures/spectrum_bonsai_motion_medium.png](docs/figures/spectrum_bonsai_motion_medium.png))
shows vanilla 3DGS losing the high frequencies that blur removes, while our estimate recovers most of them and tracks
the true-kernel result.

## What is implemented from scratch
| Component | Location |
|---|---|
| 2D Gaussian rasterizer and fitter, 2D blind-deconvolution study | `src/blur3dgs/gaussian2d` |
| Blur simulator (motion, defocus, Gaussian; per-photo kernels; three severities) | `src/blur3dgs/degradation` |
| Wiener and Richardson-Lucy baselines (true or guessed kernel) | `src/blur3dgs/baselines` |
| Per-photo kernel estimators (free-form, polyline, Gaussian) and regularisers | `src/blur3dgs/kernel_est` |
| 3DGS training loop with the blur forward model (gsplat provides only the rasterizer and densification) | `src/blur3dgs/gs3d` |
| Metrics, power spectrum, kernel-recovery error | `src/blur3dgs/analysis` |

## Reproducing
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"      # or: pip install -r requirements.txt && export PYTHONPATH=$PWD/src
python -m pytest -q tests     # or: make test
```
`make help` lists the common tasks. The methods that `scripts/run_matrix.py` can run (baselines, our variants,
ablations) are defined in [configs/methods.yaml](configs/methods.yaml).
Download Mip-NeRF 360 (see [data/README.md](data/README.md)), then:
```bash
scripts/run_all.sh        # main table, severity sweep, blur types, ablations, seeds (several hours on one GPU)
scripts/run_extras.sh     # secondary experiments
scripts/run_extras2.sh    # robustness checks on defocus and Gaussian blur
python scripts/collect.py && python scripts/make_figures.py && python scripts/build_slides.py
```
Every run is resumable: finished runs have a `metrics.json` and are skipped. A single run, for example:
```bash
python scripts/prepare_blur.py --scene bonsai --kind motion --severity medium
python scripts/train.py --scene bonsai --tag motion_medium --input blurred --kernel polyline \
    --poly_init_len 6 --kernel_start 0 --lr_kernel 0.05 --out results/example
```
Experiments ran on one RTX 4080 laptop GPU (12 GB); a bonsai run takes 4 to 7 minutes, a garden run about 10 to 15.

## Layout
```
configs/       experiment method definitions (methods.yaml)
data/          datasets and blurred caches (git-ignored)
docs/          results tables, figures, experiment log, talk outline, draft slides
results/       run outputs (git-ignored)
scripts/       experiment runners, data preparation, figure and table generation
src/blur3dgs/  library code
tests/         unit tests
```

Open work and ideas for extending the project: [docs/NEXT_STEPS.md](docs/NEXT_STEPS.md).

## Limitations
- The blur is synthetic and global per photo. Real blur varies across the image and with depth.
- Each blur type needs a prior that can represent it. The polyline prior suits motion blur; on defocus and Gaussian
  blur our gain is 0.5 to 1 dB and a correctly guessed classical deconvolution is stronger.
- Three scenes (the severity sweep covers two), three training seeds on one scene, no real blurry captures.

## References
1. Kerbl et al., 3D Gaussian Splatting for Real-Time Radiance Field Rendering: https://arxiv.org/abs/2308.04079
2. Ma et al., Deblur-NeRF: https://arxiv.org/abs/2111.14292
3. Lee et al., Deblurring 3D Gaussian Splatting: https://arxiv.org/abs/2401.00834
4. Zhao et al., BAD-Gaussians: https://arxiv.org/abs/2403.11831
5. Yu et al., Mip-Splatting: https://arxiv.org/abs/2311.16493
6. Zhang et al., GaussianImage: https://arxiv.org/abs/2403.08551
7. Ye et al., gsplat: https://arxiv.org/abs/2409.06765
8. Levin et al., Understanding and Evaluating Blind Deconvolution Algorithms, CVPR 2009
9. Wiener deconvolution: https://en.wikipedia.org/wiki/Wiener_deconvolution; Richardson-Lucy: https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution

## Datasets
- Mip-NeRF 360: https://jonbarron.info/mipnerf360/ (synthetic blur applied to get ground truth)
- Deblur-NeRF blurry scenes: https://drive.google.com/drive/folders/1_TkpcJnw504ZOWmgVTD7vWqPdzbk9Wx_ (not used in the reported results)
