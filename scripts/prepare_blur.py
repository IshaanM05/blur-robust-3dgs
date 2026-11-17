"""Create blurred training views (+ ground-truth kernels) for one scene and cache them.
Usage: python scripts/prepare_blur.py --scene bonsai --kind motion --severity medium [--shared]
Output: data/blurred/<scene>/<kind>_<severity>[_shared][_n<noise>][_b<seed>]/{blurred.npy (uint8), kernels.npy}
A non-zero --seed is a different random blur realisation and gets its own folder. An existing cache is never
overwritten unless --overwrite is given.
"""

import argparse
from pathlib import Path

import numpy as np

from blur3dgs.degradation.simulate import blur_views
from blur3dgs.gs3d.dataset import load_scene

p = argparse.ArgumentParser()
p.add_argument("--root", default="data/mipnerf360")
p.add_argument("--scene", required=True)
p.add_argument("--kind", default="motion")
p.add_argument("--severity", default="medium")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--shared", action="store_true")
p.add_argument("--noise", type=float, default=0.0)
p.add_argument("--max_views", type=int, default=None)
p.add_argument("--overwrite", action="store_true")
a = p.parse_args()
tag = f"{a.kind}_{a.severity}" + ("_shared" if a.shared else "") + (f"_n{a.noise}" if a.noise else "")
tag += f"_b{a.seed}" if a.seed else ""
out = Path("data/blurred") / a.scene / tag
if (out / "blurred.npy").exists() and not a.overwrite:
    raise SystemExit(f"{out} already exists; not overwriting (use --overwrite to replace it)")
sc = load_scene(Path(a.root) / a.scene, max_views=a.max_views)
clean = sc.images[sc.train_idx].astype(np.float32) / 255
blurred, kernels = blur_views(clean, a.kind, a.severity, a.seed, noise_sigma=a.noise, per_view=not a.shared)
out.mkdir(parents=True, exist_ok=True)
np.save(out / "blurred.npy", (blurred * 255 + 0.5).astype(np.uint8))
np.save(out / "kernels.npy", kernels)
print("saved", out, blurred.shape)
