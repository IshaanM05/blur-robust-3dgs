# Blur-Robust 3D Gaussian Splatting via Joint Per-View Blur-Kernel Estimation

CS 663 (Digital Image Processing) course project, IIT Bombay.

Each blurry input view is modeled as the sharp rendering convolved with an unknown per-view
blur kernel, `y_i = k_i * render(G, pose_i) + noise`. The Gaussians `G` and the kernels `k_i`
are optimized jointly. The question we study is **when joint estimation works, and why**.

## Team
- Ishaan Mondal (24B2118)
- [@purvanyaTyagi](https://github.com/purvanyaTyagi) (name and roll number to be added)

## Plan
| Component | Location | From scratch? |
|---|---|---|
| 2D Gaussian fitter + 2D blind-deconvolution study | `src/blur3dgs/gaussian2d` | yes |
| Blur simulator (motion, defocus, Gaussian; per-view; 3 severities) | `src/blur3dgs/degradation` | yes |
| Wiener / Richardson-Lucy baselines (oracle and guessed kernel) | `src/blur3dgs/baselines` | yes |
| Per-view kernel estimator (free-form, parametric, regularizers) | `src/blur3dgs/kernel_est` | yes |
| 3D training loop on gsplat (rasterization only from gsplat) | `src/blur3dgs/gs3d` | partly |
| Kernel-recovery error, power-spectrum analysis, metrics | `src/blur3dgs/analysis` | yes |

Methods compared: vanilla 3DGS on blurry input, pre-deblur (oracle kernel) then 3DGS,
pre-deblur (blind) then 3DGS, ours (free-form kernel), ours (parametric kernel), clean reference.
Metrics: PSNR, SSIM, LPIPS on sharp held-out views.

See [docs/STATUS.md](docs/STATUS.md) for current status and how to resume, [docs/plan.md](docs/plan.md) for scope and schedule, and [docs/log.md](docs/log.md) for the experiment log.

## Layout
```
configs/       experiment configs (one YAML per run)
data/          datasets (git-ignored; see data/README.md)
docs/          plan, experiment log, notes
experiments/   per-experiment notes and result tables
results/       run outputs (git-ignored except .gitkeep)
scripts/       entry points (download, simulate, train, evaluate)
src/blur3dgs/  library code
tests/         unit tests
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## References
1. Kerbl et al., 3D Gaussian Splatting for Real-Time Radiance Field Rendering: https://arxiv.org/abs/2308.04079
2. Ma et al., Deblur-NeRF: https://arxiv.org/abs/2111.14292
3. Lee et al., Deblurring 3D Gaussian Splatting: https://arxiv.org/abs/2401.00834
4. BAD-Gaussians: https://arxiv.org/abs/2403.11831
5. Mip-Splatting: https://arxiv.org/abs/2311.16493
6. GaussianImage: https://arxiv.org/abs/2403.08551
7. gsplat: https://arxiv.org/abs/2409.06765
8. Wiener deconvolution: https://en.wikipedia.org/wiki/Wiener_deconvolution ; Richardson-Lucy: https://en.wikipedia.org/wiki/Richardson%E2%80%93Lucy_deconvolution

## Datasets
- Mip-NeRF 360: https://jonbarron.info/mipnerf360/ (synthetic blur applied for ground truth)
- Deblur-NeRF real/synthetic blurry scenes: https://drive.google.com/drive/folders/1_TkpcJnw504ZOWmgVTD7vWqPdzbk9Wx_
