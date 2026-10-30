# Project plan

## Scope
- Scenes: Mip-NeRF 360 `bonsai`, `counter` (full), `garden` (medium severity only), `images_4` resolution.
- Blur: motion, defocus, Gaussian kernels; different kernel per view; 3 severities; fixed seeds.
- Methods: vanilla 3DGS; pre-deblur oracle + 3DGS; pre-deblur blind + 3DGS; ours free-form; ours parametric; clean reference.
- Metrics: PSNR, SSIM, LPIPS on sharp held-out views; kernel-recovery error; radial power spectra.
- Ablations (one scene): kernel size, parameterization, each regularizer, per-view vs shared kernel, number of views.
- Bonus: Deblur-NeRF real scenes (qualitative). Stretch: official Deblurring-3DGS comparison.
- Compute: single RTX 4080 12 GB, local. ~35-40 runs of ~10 min (estimate, verify with one timed run).

## Milestones
| Dates | Goal |
|---|---|
| Oct 26-27 | Data download, vanilla gsplat on bonsai, time one run |
| Oct 27-29 | 2D Gaussian fitter + 2D blind-deconvolution study, simulator, Wiener/RL baselines |
| Oct 29-31 | Kernel module in the 3D loop. Checkpoint Oct 31: joint beats vanilla on one scene |
| Nov 1-3 | Main table, ablations, spectra and kernel-recovery plots |
| Nov 4 | Garden, real scenes, qualitative figures |
| Nov 5 | Slides, README, rehearsal |
| Nov 6 | Buffer and submission |

## Fallback
If joint estimation does not beat vanilla at the checkpoint, the project becomes an analysis of when
each approach works and fails (vanilla, pre-deblur, regularized kernel), with the full ablation set.
