# Talk outline (10 minutes, ~10 slides). Figures are in `docs/figures/`; tables in `docs/results.md`.

Map to the professor's 6 expectations: (1) papers, (2) math, (3) from-scratch implementation, (4) real datasets, (5) empirical analysis + ablations, (6) explanation of results.

| # | Slide | Content | Figure | ~min |
|---|---|---|---|---|
| 1 | Title + motivation | Blur from motion/defocus ruins 3D reconstruction (night driving, handheld). 3DGS assumes sharp inputs. | qualitative_bonsai (vanilla vs ours) | 0.5 |
| 2 | Background (papers read) | 3DGS (Kerbl), Deblur-NeRF, Deblurring-3DGS, BAD-Gaussians, Levin blind deconvolution, Mip-Splatting | none | 1 |
| 3 | Image formation model (DIP) | `y_i = k_i * render(G, pose_i) + n`; Wiener `H*/(|H|^2+NSR)`; Richardson-Lucy update; why single-image blind deconv is ill-posed | diagram | 1.5 |
| 4 | What we built from scratch | 2D Gaussian rasterizer (EWA-style covariance, alpha compositing), blur simulator, Wiener/RL, per-view kernel estimators, analysis tools. gsplat only for 3D rasterization. | pipeline diagram | 1 |
| 5 | 2D warm-up | Single-image blind deconv collapses to the trivial solution (sharp = blurry, kernel = delta) | exp_2d_blind.png | 1 |
| 6 | The 3D method | Joint optimisation of Gaussians + per-view kernels; test time: no kernel. Free-form kernel collapses; polyline prior (trajectory splatted into a kernel, centroid fixed) works. | kernels_bonsai_ours_polyline.png | 1.5 |
| 7 | Main results | 3 scenes, vs vanilla / Wiener / RL (guessed kernel) / oracle upper bounds | results table + qualitative_* | 1.5 |
| 8 | Analysis: severity, blur type, spectra | PSNR vs severity; polyline vs gaussian kernel across blur types; frequency recovery | severity_*, blur_types_*, spectrum_* | 1 |
| 9 | Ablations | control points, kernel size, shared vs per-view, number of views; seed std | ablations_bonsai.png | 1 |
| 10 | Takeaways + limits | Structured kernel prior makes joint estimation identifiable; limits: synthetic blur, per-view kernel is an approximation (blur really varies spatially/depth), single-seed scenes, free-form needs better priors. Future work. | none | 0.5 |

## Key numbers to quote (update from docs/results.md)
- bonsai motion medium: vanilla 26.42, blind classical (guessed k) 26.8, **ours polyline 29.93**, oracle-kernel 30.40, Wiener true-kernel 30.99.
- counter: 25.40 / 25.5 / **27.72** / 27.96 / 28.19.

## Likely questions from the professor (prepare answers)
- Why does the free-form kernel collapse and the polyline not? (Levin et al.: joint MAP has a trivial delta solution; polyline = low-dimensional, physically valid prior.)
- Is the blur realistic? (Synthetic, per-view global kernel; real Deblur-NeRF data is future/qualitative.)
- What is from scratch vs library? (see slide 4.)
- Why Wiener/RL with a guessed kernel? (the fair blind baseline; with the true kernel it is an oracle reference.)
- Wiener-oracle beats ours; why do you still win? (Oracle uses information we do not have; fair comparison is the blind row.)
