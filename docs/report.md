# When Does Joint Blur-Kernel Estimation Make 3D Gaussian Splatting Robust to Blur?

*Fundamentals of Digital Image Processing (CS 663) course project, IIT Bombay. Ishaan Mondal (24B2118), Purvanya Tyagi (24B0385).*

Every number below comes from [results.md](results.md), which `scripts/collect.py` builds from the run outputs. Except Section 5.5, all
results are on synthetic blur applied to real photographs, at 10,000 training iterations, and are single runs unless a standard deviation
is given.

## Abstract

3D Gaussian Splatting (3DGS) reconstructs a scene from posed photographs but assumes they are sharp; with blurred input the
blur is baked into the reconstruction. We re-implement and analyse an existing idea, also used by Deblur-NeRF, Deblurring 3DGS and
BAD-Gaussians: model each photograph as the sharp rendering convolved with an unknown per-photograph kernel, and optimise the Gaussians
and the kernels jointly. Our question is when this joint estimation works and why. On three Mip-NeRF 360 scenes with synthetic motion blur, a
low-dimensional *polyline* kernel recovers the true kernels and comes within 0.1 to 0.5 dB PSNR of training with the true
kernels, 2 to 3.5 dB above every blind baseline. A free-form kernel, by contrast, collapses to a delta in every setting we tried,
which is the trivial solution of blind deconvolution. On defocus and Gaussian blur the gain is 0.5 to 1 dB, and classical Wiener
deconvolution with a correctly guessed kernel is stronger in PSNR, although a wrong guess puts it 2.8 to 4.3 dB *below* doing nothing.

## 1. Introduction

Handheld capture, low light and moving platforms produce photographs blurred by camera motion or by defocus. Novel-view synthesis
methods such as 3DGS [1] fit a scene so that its renderings match the photographs, so a blurred input yields a blurred scene.

From the image-restoration point of view this is a deconvolution problem. Each photograph $y_i$ is modelled as

$$y_i = k_i * r_i(G) + n_i,$$

where $r_i(G)$ is the rendering of the Gaussian scene $G$ from camera pose $i$, $k_i$ is a blur kernel and $n_i$ noise. If the kernels were
known, deblurring each photograph first (Wiener, Richardson-Lucy) or putting the known kernel inside the training loss would both
work. They are not known, and single-image *blind* deconvolution is ill-posed: the pair (sharp image = blurry image, kernel = delta)
explains the observation perfectly [8].

Multi-view reconstruction changes this. One scene $G$ has to explain every photograph, each blurred differently. A kernel that
leaves the blur in one photograph conflicts with the sharper evidence the other photographs provide about the same surface. Deblur-NeRF
[2], Deblurring 3DGS [3] and BAD-Gaussians [4] exploit this with different blur models. This project re-implements the simplest version of
the idea, one learnable kernel per photograph, and studies it as a controlled experiment:

- Which kernel parametrisations make joint estimation identifiable, and which collapse?
- How does the gain depend on blur type, blur severity and the number of photographs?
- How does it compare with classical deconvolution, both when the kernel is guessed and when it is known?

We do not claim a new method. The contribution is an implementation written largely from scratch and an analysis on synthetic blur
with known ground truth.

## 2. Related work

**3DGS** [1] represents a scene as anisotropic 3D Gaussians that are projected and alpha-composited; it is fit by gradient descent
on a photometric loss, with densification and pruning. We use gsplat [7] for the rasteriser and densification only.

**Deblurring radiance fields.** Deblur-NeRF [2] models blur with a learned, spatially varying sparse kernel inside a NeRF.
Deblurring 3DGS [3] instead predicts per-Gaussian covariance changes with a small network. BAD-Gaussians [4] models motion blur
physically by interpolating camera poses over the exposure time and averaging the renderings. Our per-photograph kernel is the
simplest of these blur models: it cannot vary across the image or with depth, but it makes the identifiability question easy to study.

**Blind deconvolution.** Levin et al. [8] show that maximising the joint posterior of image and kernel favours the delta kernel
and the blurry image, and that estimating the kernel first, with a strong prior, avoids this. Our free-form result (Section 5.3) is
the same failure in a 3D setting, and the polyline kernel is the strong prior.

## 3. Method

### 3.1 Forward model and training

We optimise the Gaussian parameters (means, scales, rotations, opacities, spherical-harmonic colours) and one kernel per training
photograph. At each step a training photograph $i$ is sampled, the scene is rendered from pose $i$, the rendering is convolved with $k_i$
(true 2D convolution, reflect padding), and the loss

$$\mathcal{L} = 0.8\,\lVert k_i * r_i(G) - y_i \rVert_1 + 0.2\,(1 - \mathrm{SSIM}(k_i * r_i(G), y_i))$$

is back-propagated to both $G$ and $k_i$. At test time the kernel is dropped and the scene is rendered sharp. Training uses
10,000 iterations, Adam for both parts, and gsplat's default densification strategy (stopped at half the iterations).

### 3.2 Kernel parametrisations

All kernels are 25 x 25 pixels, non-negative and sum to one.

- **Free-form:** a softmax over 625 logits per photograph, initialised near a delta (or a small Gaussian).
- **Polyline (motion prior):** the exposure path of the camera as $P = 6$ control points in pixel offsets. 64 points are sampled
  along the path and splatted bilinearly into the kernel, which keeps it differentiable with respect to the control points. The centroid
  of the control points is fixed at the kernel centre, removing the translation ambiguity between the kernel and the scene. The path starts
  as a short horizontal segment. This is 12 parameters per photograph instead of 625.
- **Gaussian (defocus prior):** an anisotropic Gaussian with two log standard deviations and an angle, 3 parameters per photograph.
- **Oracle:** the true kernels, fixed. This is the non-blind upper bound for the joint method.

### 3.3 Baselines

- **Vanilla 3DGS** trained directly on the blurred photographs.
- **Pre-deblur + 3DGS:** each photograph is deblurred with Wiener deconvolution ($\hat X = \bar H Y / (|H|^2 + \mathrm{NSR})$, NSR = 0.01 for the guessed kernel and 0.001 for the true kernel)
  or Richardson-Lucy (30 iterations), then vanilla 3DGS is trained on the result. The *blind* variant uses a guessed isotropic Gaussian
  kernel (sigma 2 pixels unless stated); the *oracle* variant uses the true kernel and is a reference, not a fair competitor.

## 4. Experimental setup

**Data.** Three Mip-NeRF 360 scenes [10] (bonsai, counter, garden) at quarter resolution (`images_4`), with the provided poses. Every
eighth photograph is held out; the held-out photographs are kept *sharp* and are used only for evaluation. On bonsai this leaves 255
training photographs.

**Synthetic blur.** Only the training photographs are blurred, with a different kernel per photograph and three severities:

| Blur | low | medium | high |
|---|---|---|---|
| Motion: smooth random-walk path, arc length (px) | 5 | 11 | 19 |
| Defocus: anti-aliased disk, radius (px) | 1.5 | 3 | 5 |
| Gaussian: sigma (px) | 1 | 2 | 3 |

Motion kernels differ between photographs; defocus and Gaussian kernels are the same for every photograph at a given severity. A
noisy variant adds Gaussian noise of standard deviation 0.02.

**Metrics.** PSNR, SSIM and LPIPS (AlexNet) on the sharp held-out views. Kernel recovery is the L2 distance between the learned and the
true kernel, minimised over integer shifts of up to 3 pixels (a kernel is only identifiable up to a translation). For reference, a
delta kernel scores about 0.99 against the medium motion kernels.

**Implementation.** The blur simulator, Wiener and Richardson-Lucy baselines, kernel modules, training loop, metrics and a 2D
Gaussian-image fitter are our own code; gsplat provides the 3D rasteriser and densification. 20 unit tests cover the core
operations. Runs took 4 to 7 minutes on bonsai and 10 to 15 minutes on garden on one 12 GB laptop GPU; 85 runs in total.

## 5. Results

### 5.1 A 2D warm-up: single-image blind deconvolution collapses

Before the 3D experiments we fit 1,024 2D Gaussians to a single 128 x 128 image blurred with a medium motion kernel, jointly with a free-form
kernel (`scripts/exp_2d_blind.py`). The blind fit reaches 21.4 dB against 21.2 dB for the blurry input, and the learned kernel is a
delta (figure: [exp_2d_blind.png](figures/exp_2d_blind.png)). With the true kernel the same fitter reaches 27.5 dB. A Wiener deconvolution
with a guessed Gaussian kernel gives 22.2 dB, barely above the input. With one image, nothing prevents the trivial solution.

### 5.2 Main result: motion blur

**Table 1.** Medium motion blur, PSNR (dB) on held-out sharp views. The last two rows know the true kernels.

| Method | bonsai | counter | garden |
|---|---|---|---|
| Vanilla 3DGS | 26.42 | 25.40 | 23.18 |
| Wiener, guessed kernel, then 3DGS | 26.82 | 25.46 | 22.86 |
| Richardson-Lucy, guessed kernel, then 3DGS | 26.84 | 25.56 | 23.09 |
| **Ours, polyline kernel (blind)** | **29.93** | **27.72** | **25.17** |
| Ours with the true kernels | 30.40 | 27.96 | 25.27 |
| Wiener, true kernel, then 3DGS | 30.99 | 28.19 | 25.76 |

**Table 2.** The same runs, SSIM / LPIPS.

| Method | bonsai | counter | garden |
|---|---|---|---|
| Vanilla 3DGS | 0.773 / 0.345 | 0.745 / 0.365 | 0.528 / 0.563 |
| Wiener, guessed kernel | 0.822 / 0.222 | 0.772 / 0.257 | 0.601 / 0.440 |
| Richardson-Lucy, guessed kernel | 0.821 / 0.230 | 0.776 / 0.262 | 0.608 / 0.458 |
| **Ours, polyline kernel (blind)** | **0.910 / 0.111** | **0.859 / 0.148** | **0.712 / 0.283** |
| Ours with the true kernels | 0.920 / 0.093 | 0.870 / 0.128 | 0.731 / 0.250 |
| Wiener, true kernel | 0.938 / 0.075 | 0.894 / 0.112 | 0.814 / 0.148 |

On all three scenes the polyline kernel is 2.0 to 3.5 dB above vanilla and above both blind classical baselines, and 0.1 to 0.5 dB
below the same method with the true kernels; SSIM and LPIPS give the same ordering. The guessed-kernel baselines gain at most 0.4 dB and
on garden fall below vanilla, because an isotropic Gaussian is a poor model of a motion path. The learned kernels match the shape and
orientation of the true paths (figure: [kernels_bonsai_ours_polyline.png](figures/kernels_bonsai_ours_polyline.png)), with a recovery
error of 0.13 to 0.15 against about 0.99 for a delta. Over three training seeds on bonsai the polyline method scores 30.02 +- 0.08 dB, vanilla
26.29 +- 0.18 and guessed-kernel Wiener 26.76 +- 0.06, so the gaps are far larger than the seed noise. Visually, ours restores the knit texture
and the edges that vanilla 3DGS blurs (figure: [qualitative_bonsai_motion_medium.png](figures/qualitative_bonsai_motion_medium.png)).

Wiener deconvolution with the *true* kernel is the strongest entry. It uses information a blind method does not have, so we report it
as a reference rather than a competitor. It is also above our own true-kernel run, which suggests that the in-loop formulation loses
some detail that an explicit inverse filter recovers on this noise-free data (see Section 5.4).

**Severity.** The gain over vanilla grows with blur strength: +2.0, +3.5 and +4.5 dB on bonsai at low, medium and high severity, and
+1.0, +2.3 and +3.2 dB on counter (figures: [severity_bonsai_motion.png](figures/severity_bonsai_motion.png),
[severity_counter_motion.png](figures/severity_counter_motion.png)). At low severity on bonsai the guessed-kernel Wiener baseline (25.20) is
3.3 dB *below* vanilla (28.49): a plausible reason is that deconvolving with a kernel larger than the true one does more harm than the blur (we did not test this).

**Robustness checks.** (i) *Blur realisation:* two further random draws of the blur on bonsai give 29.98 and 29.86 dB for the polyline method
(29.93 for the first draw), against 26.28 for vanilla in both, so the result does not depend on one draw. (ii) *Longer training:* at 30,000
iterations on bonsai the polyline method reaches 30.92 dB (SSIM 0.922, LPIPS 0.085), vanilla 26.49 and guessed-kernel Wiener 26.41. Vanilla
gains only 0.07 dB from three times the training while ours gains 1.0 dB, so the gap widens rather than closing. (iii) *Richardson-Lucy with a
guessed kernel* on counter scores 24.21 and 24.48 dB at low and high severity, against 27.08 and 24.21 for vanilla: at low severity the wrong
guess costs 2.9 dB, and at high severity it gains only 0.3 dB, while ours scores 28.12 and 27.41. (iv) *Sharp-input ceiling:* 3DGS trained on the
unblurred photographs scores 30.88, 28.45 and 26.25 dB on bonsai, counter and garden; our blind method closes most of the gap on bonsai
(29.93) and counter (27.72).

**Frequency view.** The radial power spectrum of a rendered test view shows vanilla 3DGS losing the high frequencies removed by the blur,
while our estimate recovers most of them and tracks the true-kernel run (figure: [spectrum_bonsai_motion_medium.png](figures/spectrum_bonsai_motion_medium.png)).

### 5.3 Why the parametrisation matters

**Table 3.** Ablations on bonsai, medium motion blur (PSNR; kernel recovery error in brackets where relevant).

| Variant | PSNR |
|---|---|
| Default polyline (6 control points, 25 px kernel) | 29.93 (0.15) |
| 3 / 12 control points | 29.77 / 30.11 |
| Kernel size 15 / 35 px | 29.95 / 30.08 |
| One polyline kernel shared by all photographs | 26.72 (0.21) |
| Free-form kernel, delta initialisation | 26.51 (0.92) |
| Free-form, learning rate 0.1 / 0.3 | 26.40 / 26.33 |
| Free-form, Gaussian initialisation (sigma 2.5) | 26.14 (0.91) |
| Free-form + sparsity / TV / sparsity + TV + centroid | 26.49 / 25.94 / 26.46 |
| Vanilla 3DGS | 26.42 |

Every free-form variant lands within 0.5 dB of vanilla with a recovery error of 0.6 to 0.94: the kernel collapses towards a delta and the
scene absorbs the blur. This held across learning rates, initialisations and the standard kernel regularisers. The polyline kernel, with 12
parameters that can only describe a connected path, avoids the collapse, and the result is insensitive to the number of control points and the
kernel size. Sharing one kernel across all photographs fails (26.72), as expected when the true kernels differ per photograph: per-photograph
kernels are necessary, not merely convenient.

Our explanation follows Levin et al. [8]. With a free-form kernel the optimiser can lower the loss almost as much by making the scene blurry
as by making the kernel blurry, and the first is easier to reach from a delta initialisation, because the scene has far more parameters and
receives gradient from every photograph. A path-shaped prior removes most of the kernels that are not physically plausible, so the scene can
only explain the data well by becoming sharp.

**Number of photographs.** With 100, 50 and 25 training photographs the polyline method scores 29.52, 27.46 and 23.64 dB, against
vanilla's 25.09 at 50 and 22.32 at 25. The gain shrinks from +3.5 dB (255 photographs) to +2.4 (50) and +1.3 (25). This is the
multi-view argument from Section 1 in reverse: with fewer photographs there are fewer independent views of each surface to pin the kernels down.

### 5.4 Defocus and Gaussian blur: a mixed result

**Table 4.** bonsai, medium severity. The guessed Wiener kernel is an isotropic Gaussian; at sigma 2 it equals the true Gaussian-blur kernel
and is close to the defocus disk, so we also ran deliberately wrong guesses.

| Method | Gaussian blur PSNR | Defocus PSNR | Gaussian blur SSIM / LPIPS | Defocus SSIM / LPIPS |
|---|---|---|---|---|
| Vanilla 3DGS | 26.24 | 26.70 | 0.759 / 0.386 | 0.780 / 0.334 |
| Wiener, guess sigma 1 | 26.51 | 26.45 | 0.782 / 0.344 | 0.775 / 0.256 |
| Wiener, guess sigma 2 (correct) | 28.25 | 28.01 | 0.855 / 0.242 | 0.852 / 0.232 |
| Wiener, guess sigma 4 | 23.46 | 22.43 | 0.676 / 0.404 | 0.650 / 0.413 |
| Ours, polyline kernel | 27.29 | 27.54 | 0.804 / 0.259 | 0.813 / 0.232 |
| Ours, Gaussian kernel | 27.21 | 27.25 | 0.815 / 0.213 | 0.806 / 0.194 |
| Ours with the true kernels | 27.12 | 28.62 | 0.820 / 0.192 | 0.872 / 0.148 |

Here the picture is less favourable. Both blind variants are 0.5 to 1.1 dB above vanilla, and a correctly guessed Wiener deconvolution is 0.5 to
1.0 dB above them in PSNR and SSIM. The perceptual metric disagrees: the Gaussian kernel has a better LPIPS than the correct Wiener guess on both
blurs (0.213 against 0.242, and 0.194 against 0.232). Classical deconvolution is also fragile: a guess of sigma 4 lands 2.8 dB (Gaussian blur)
and 4.3 dB (defocus) below vanilla, while our methods need no guess (figure: [guess_sensitivity_bonsai.png](figures/guess_sensitivity_bonsai.png)).

Two further observations:

- **Kernel accuracy is not the bottleneck here.** The Gaussian kernel recovers the true kernel better than the polyline (error 0.03 to 0.09
  against 0.14 to 0.16) but does not score higher in PSNR. On Gaussian blur, even the true kernels in the loop (27.12) trail the correct Wiener
  pre-deblur (28.25).
- **Noise matters once it is large.** Table 5 sweeps the noise level on Gaussian blur. At standard deviation 0.02 the ordering is unchanged. At
  0.05 Wiener falls to the level of ours, and at 0.1 it is 1.8 dB below vanilla, while ours stays level with vanilla. The in-loop method with the
  true kernels degrades faster than with an estimated Gaussian, which suggests it fits the noise.

**Table 5.** bonsai, Gaussian blur, medium severity, PSNR by added noise standard deviation.

| Noise | Vanilla | Wiener (correct guess) | Ours, Gaussian kernel | Ours, true kernels |
|---|---|---|---|---|
| 0 | 26.24 | 28.25 | 27.21 | 27.12 |
| 0.02 | 26.33 | 28.05 | 27.36 | 26.76 |
| 0.05 | 26.36 | 26.59 | 26.62 | 24.95 |
| 0.1 | 25.67 | 23.84 | 25.67 | 22.93 |

A second scene gives the same ordering (`counter`, Gaussian blur, medium): vanilla 25.32, ours with the Gaussian kernel 26.17, ours with the
polyline 26.04, true kernels 26.56, correct Wiener 26.68, and Wiener with a wrong guess (sigma 4) 22.59.

Our interpretation, which we have not tested directly, is that a smooth blur removes high frequencies almost entirely. The in-loop method only
receives gradient through the blurred rendering, so it is never pushed to restore content that the blur suppressed. An explicit inverse filter
amplifies those frequencies directly, which works on noise-free synthetic data. A motion path, by contrast, preserves more high-frequency
content along the direction perpendicular to the motion, which may be why the in-loop method does better on motion blur.

### 5.5 Real blurry photographs (qualitative)

Deblur-NeRF [2] provides real camera-motion-blur captures. We downloaded the nine scenes that Google Drive let us fetch (`blurball`,
`blurbasket`, `blurbuick`, `blurcoffee`, `blurdecoration`, `blurgirl`, `blurheron`, `blurparterre`, `blurpuppet`) and trained vanilla 3DGS and the polyline method on them
with identical settings (7,000 iterations, the photographs as given, no synthetic blur added). The photographs have no sharp ground truth, so
PSNR against the held-out photographs is meaningless as a sharpness measure and we judge the renderings by eye (figure:
[real_summary.png](figures/real_summary.png); per-scene figures `real_<scene>.png`).

Vanilla 3DGS renders are clearly blurrier than the held-out photographs, because it averages the blur of its training photographs. The polyline
method recovers fine detail (the leaf edges and weave of the ball in `blurball`, the basket wires in `blurbasket`) at a sharpness comparable to the
photographs, and on `blurbuick`, where the held-out photograph is itself soft, sharper than it. It also introduces artefacts: bright, high-contrast edges
such as the chrome headlight strip in `blurbuick` show speckled over-sharpening. The results are uneven. `blurcoffee` and `blurdecoration` show dark or streaked
patches, and `blurgirl` (six photographs and a moving subject) fails for both methods, with extra speckle from ours. The method works best on static, well covered
scenes (`ball`, `basket`, `buick`, `heron`, `parterre`). This is qualitative evidence from nine scenes with one global kernel per
photograph, not a measured improvement.

## 6. Limitations

- **Synthetic, global blur.** Every kernel is uniform over the photograph. Real camera shake produces spatially varying blur, and defocus
  depends on depth. The evaluation on real blurry captures (Section 5.5) is qualitative and covers nine scenes and is uneven.
- **The prior must match the blur.** The polyline suits motion and the Gaussian suits defocus; the user must know which to use. On defocus and
  Gaussian blur our gain is 0.5 to 1 dB and a correctly guessed classical deconvolution is stronger in PSNR.
- **Limited breadth.** Three scenes for the main table, two for the severity sweep, one for the blur-type, ablation and seed studies. Defocus
  and Gaussian kernels are identical across photographs. The longer-training check (30,000 iterations) covers one scene; all other runs stop at
  10,000 iterations.
- **No comparison with the original implementations** of Deblur-NeRF, Deblurring 3DGS or BAD-Gaussians.

## 7. Conclusion

Joint estimation of per-photograph blur kernels with 3DGS works when the kernel model is constrained enough to rule out the trivial
explanation of the data. On synthetic motion blur a 12-parameter polyline kernel recovers the true kernels and nearly matches training with them,
2 to 3.5 dB above blind baselines, while an unconstrained kernel collapses to a delta whatever the tuning. The multi-view constraint is what
makes the kernels identifiable: the gain disappears with one image (2D study), shrinks with fewer photographs and needs a separate kernel per
photograph. On smooth blurs the method is safer than classical deconvolution, because it needs no guess, but not better than a correct guess.
Natural next steps are real blurry captures, spatially varying kernels and choosing the kernel prior automatically.

## References

1. B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. SIGGRAPH 2023. arXiv:2308.04079.
2. L. Ma et al. Deblur-NeRF: Neural Radiance Fields from Blurry Images. CVPR 2022. arXiv:2111.14292.
3. B. Lee et al. Deblurring 3D Gaussian Splatting. ECCV 2024. arXiv:2401.00834.
4. L. Zhao, P. Wang, P. Liu. BAD-Gaussians: Bundle Adjusted Deblur Gaussian Splatting. ECCV 2024. arXiv:2403.11831.
5. Z. Yu et al. Mip-Splatting: Alias-free 3D Gaussian Splatting. CVPR 2024. arXiv:2311.16493.
6. X. Zhang et al. GaussianImage: 1000 FPS Image Representation and Compression by 2D Gaussian Splatting. ECCV 2024. arXiv:2403.08551.
7. V. Ye et al. gsplat: An Open-Source Library for Gaussian Splatting. arXiv:2409.06765.
8. A. Levin, Y. Weiss, F. Durand, W. T. Freeman. Understanding and Evaluating Blind Deconvolution Algorithms. CVPR 2009.
9. N. Wiener. Extrapolation, Interpolation, and Smoothing of Stationary Time Series. MIT Press, 1949. W. H. Richardson, Bayesian-based iterative method of image restoration, JOSA 1972; L. B. Lucy, An iterative technique for the rectification of observed distributions, AJ 1974.
10. J. T. Barron et al. Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields. CVPR 2022. arXiv:2111.12077.
