"""2D study: blur a sample image, then compare (a) blurry, (b) Wiener/RL with oracle kernel,
(c) Wiener/RL with a guessed Gaussian kernel, (d) Gaussians + known kernel, (e) Gaussians + learned kernel (blind).
Usage: python scripts/exp_2d_blind.py [--kind motion] [--severity medium] [--n 1024] [--iters 1500]
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib
import numpy as np
from skimage import data, transform

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from blur3dgs.analysis.kernel_recovery import best_shift_l2
from blur3dgs.analysis.metrics import psnr, ssim
from blur3dgs.baselines.richardson_lucy import guessed_kernel, richardson_lucy
from blur3dgs.baselines.wiener import wiener_deconv
from blur3dgs.degradation.simulate import blur_views
from blur3dgs.gaussian2d.fit import fit_image

p = argparse.ArgumentParser()
p.add_argument("--kind", default="motion")
p.add_argument("--severity", default="medium")
p.add_argument("--n", type=int, default=1024)
p.add_argument("--iters", type=int, default=1500)
p.add_argument("--size", type=int, default=128)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

img = transform.resize(data.astronaut(), (a.size, a.size), anti_aliasing=True).astype(np.float32)
blurred, ks = blur_views(img[None], a.kind, a.severity, seed=a.seed)
blurred, k_gt = blurred[0], ks[0]
rows = {
    "blurry input": blurred,
    "Wiener (oracle k)": wiener_deconv(blurred, k_gt, 1e-3),
    "RL (oracle k)": richardson_lucy(blurred, k_gt, 30),
    "Wiener (guessed gauss k)": wiener_deconv(blurred, guessed_kernel(25, 2.0), 1e-2),
    "RL (guessed gauss k)": richardson_lucy(blurred, guessed_kernel(25, 2.0), 30),
}
t = time.time()
known = fit_image(blurred, a.n, a.iters, kernel=k_gt, seed=a.seed)
rows["Gaussians + known k"] = known["render"]
blind = fit_image(
    blurred, a.n, a.iters, learn_kernel=True, ksize=25, reg={"w_centroid": 1.0, "w_sparse": 1e-3}, seed=a.seed
)
rows["Gaussians + learned k (blind)"] = blind["render"]
print(f"2D fits took {time.time() - t:.0f}s")
res = {k: {"psnr": psnr(v, img), "ssim": ssim(v, img)} for k, v in rows.items()}
res["kernel_best_shift_l2 (blind)"] = best_shift_l2(blind["kernel"], k_gt)
out = Path("results/exp_2d") / f"{a.kind}_{a.severity}"
out.mkdir(parents=True, exist_ok=True)
json.dump(res, open(out / "metrics.json", "w"), indent=1)
for k, v in res.items():
    print(f"{k:34s}", {m: round(x, 3) for m, x in v.items()} if isinstance(v, dict) else round(v, 4))


fig, ax = plt.subplots(2, 5, figsize=(17, 7))
items = [("ground truth", img)] + list(rows.items())
for i, (name, im) in enumerate(items[:7]):
    ax.flat[i].imshow(im)
    ax.flat[i].set_title(name, fontsize=9)
    ax.flat[i].axis("off")
ax.flat[7].imshow(k_gt, cmap="gray")
ax.flat[7].set_title("GT kernel")
ax.flat[7].axis("off")
ax.flat[8].imshow(blind["kernel"], cmap="gray")
ax.flat[8].set_title("learned kernel")
ax.flat[8].axis("off")
ax.flat[9].plot(blind["history"])
ax.flat[9].set_title("blind loss")
ax.flat[9].set_yscale("log")
plt.tight_layout()
plt.savefig(out / "figure.png", dpi=110)
