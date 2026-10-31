# Experiment log

One entry per experiment: date, config, what changed, result, takeaway.

## Template
- **Date:**
- **Config:** `configs/...`
- **Hypothesis:**
- **Result:**
- **Takeaway:**

## 2026-10-31: 2D blind deconvolution toy (astronaut 128x128, motion/medium, N=1024 Gaussians, 1500 iters)
- **Config:** `scripts/exp_2d_blind.py --kind motion --severity medium`
- **Result (PSNR / SSIM vs sharp):** blurry 21.2 / 0.742; Wiener oracle k 30.8 / 0.935; RL oracle k 29.7 / 0.955;
  Wiener guessed gaussian k 22.2 / 0.782; RL guessed k 20.5 / 0.757; Gaussians + known k 27.5 / 0.925;
  Gaussians + learned k (blind) 21.4 / 0.754; blind kernel best-shift L2 0.94.
- **Takeaway:** with one image, blind joint estimation collapses to the trivial solution (sharp ~ blurry, kernel ~ spread),
  no better than the input. Wrong (guessed) kernels make classical deconvolution no better than the blurry input. This is the
  identifiability gap that multi-view 3D consistency should close: the Gaussians must explain many views with different kernels.
- **Caveat:** single seed, single image; Gaussians + known k is below Wiener oracle at this capacity (N=1024, 128px).

## 2026-10-31: bonsai, motion/medium, 10k iters, 3DGS on gsplat (single seed)
- **Configs:** `scripts/run_bonsai_first.sh`, `run_kernel_lr_sweep.sh`, `run_diagnosis.sh`, `run_baselines_bonsai.sh`
- **Result (held-out sharp views):**

| Method | PSNR | SSIM | LPIPS | kernel L2 (best shift) |
|---|---|---|---|---|
| vanilla 3DGS on blurry input | 26.42 | 0.773 | 0.345 | - |
| Wiener pre-deblur (oracle k) + 3DGS | 30.99 | 0.938 | 0.075 | - |
| ours: oracle kernels (non-blind upper bound) | 30.40 | 0.920 | 0.093 | 0 |
| ours: free-form, delta init | 26.51 | 0.775 | 0.343 | 0.92 (collapsed to delta) |
| ours: free-form, lr 0.1 / 0.3 | 26.40 / 26.33 | 0.771 / 0.769 | 0.343 / 0.346 | 0.91 / 0.94 |
| ours: free-form, gaussian init sigma 2.5, from step 0 | 26.14 | 0.767 | 0.345 | 0.91 |
| **ours: polyline (parametric), blind** | **29.93** | **0.910** | **0.111** | **0.15** |

- **Takeaway:** the forward model is correct (oracle kernels 30.4 dB). Free-form per-view kernels collapse to a delta, so the scene
  absorbs the blur and nothing is gained over vanilla. The parametric polyline kernel recovers kernels (L2 0.15 vs 0.99 for a delta)
  and gets within 0.5 dB of the oracle-kernel upper bound, +3.5 dB over vanilla, without knowing the kernels.
- **Caveats:** one scene, one seed, one severity. Wiener with the *true* kernel (31.0) is still above our blind polyline (29.9); the
  fair blind baselines (guessed kernel) are still running. Free-form tuning was limited (3 settings).

## 2026-10-31 (evening): matrix runs via `scripts/run_matrix.py` (see docs/results.md for all tables)
- **bonsai/motion_medium blind baselines:** Wiener guess 26.82, RL guess 26.84 (vs ours polyline 29.93, vanilla 26.42): ours is +3.1 dB over the best blind classical baseline.
- **counter/motion_medium:** Wiener oracle 28.19, ours oracle 27.96, **ours polyline 27.72**, RL guess 25.56, Wiener guess 25.46, vanilla 25.40 (kernel L2 0.145). Same ordering as bonsai.
- **garden/motion_medium (partial):** vanilla 23.18, Wiener oracle 25.76; remaining runs pending.
- **Caveat:** single seed. Garden runs grow to ~3M Gaussians and take 13-16 min each.
