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
