"""Pre-deblur cached blurry views (Wiener or RL), with oracle or guessed kernels -> input to vanilla 3DGS.
Usage: python scripts/prepare_deblur.py --scene bonsai --tag motion_medium --method wiener --kernel oracle|guess
"""
import argparse
from pathlib import Path

import numpy as np

from blur3dgs.baselines.richardson_lucy import guessed_kernel, richardson_lucy
from blur3dgs.baselines.wiener import wiener_deconv

p = argparse.ArgumentParser()
p.add_argument("--scene", required=True); p.add_argument("--tag", required=True)
p.add_argument("--method", choices=["wiener", "rl"], default="wiener")
p.add_argument("--kernel", choices=["oracle", "guess"], default="oracle")
p.add_argument("--guess_sigma", type=float, default=2.0); p.add_argument("--nsr", type=float, default=1e-3)
a = p.parse_args()
d = Path("data/blurred") / a.scene / a.tag
blurred, kernels = np.load(d / "blurred.npy"), np.load(d / "kernels.npy")
outs = []
for i, b in enumerate(blurred):
    k = kernels[i] if a.kernel == "oracle" else guessed_kernel(kernels.shape[-1], a.guess_sigma)
    x = b.astype(np.float32) / 255
    y = wiener_deconv(x, k, a.nsr if a.kernel == "oracle" else 1e-2) if a.method == "wiener" else richardson_lucy(x, k, 30)
    outs.append((y * 255 + 0.5).astype(np.uint8))
name = f"{a.method}_{a.kernel}"
np.save(d / f"deblur_{name}.npy", np.stack(outs))
print("saved", d / f"deblur_{name}.npy")
