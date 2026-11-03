"""Train one 3DGS run. Examples:
  clean reference : python scripts/train.py --scene bonsai --input clean --out results/bonsai/clean
  vanilla (blurry): python scripts/train.py --scene bonsai --tag motion_medium --input blurred --out results/bonsai/motion_medium/vanilla
  pre-deblur      : ... --input deblur_wiener_oracle ...
  ours            : ... --input blurred --kernel free|polyline ...
"""
import argparse
from pathlib import Path

import numpy as np

from blur3dgs.gs3d.dataset import load_scene
from blur3dgs.gs3d.train import train

p = argparse.ArgumentParser()
p.add_argument("--root", default="data/mipnerf360"); p.add_argument("--scene", required=True)
p.add_argument("--tag", default=None, help="blurred-cache tag, e.g. motion_medium")
p.add_argument("--input", default="blurred", help="clean | blurred | deblur_<method>_<kernel>")
p.add_argument("--kernel", default="none", choices=["none", "free", "polyline", "gauss", "oracle"])
p.add_argument("--ksize", type=int, default=25); p.add_argument("--iters", type=int, default=10000)
p.add_argument("--kernel_start", type=int, default=500); p.add_argument("--lr_kernel", type=float, default=1e-2)
p.add_argument("--w_centroid", type=float, default=0.0); p.add_argument("--w_sparse", type=float, default=0.0)
p.add_argument("--w_tv", type=float, default=0.0); p.add_argument("--shared_kernel", action="store_true")
p.add_argument("--kernel_init_sigma", type=float, default=1.0)
p.add_argument("--poly_init_len", type=float, default=1.0)
p.add_argument("--n_ctrl", type=int, default=6)
p.add_argument("--train_views", type=int, default=None, help="subsample TRAIN views only (test set unchanged)")
p.add_argument("--max_views", type=int, default=None); p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", required=True)
a = p.parse_args()

sc = load_scene(Path(a.root) / a.scene, max_views=a.max_views)
gt_k = None
if a.input == "clean":
    imgs = sc.images[sc.train_idx]
else:
    d = Path("data/blurred") / a.scene / a.tag
    imgs = np.load(d / ("blurred.npy" if a.input == "blurred" else f"{a.input}.npy"))
    gt_k = np.load(d / "kernels.npy")
if a.train_views and a.train_views < len(sc.train_idx):
    sel = np.linspace(0, len(sc.train_idx) - 1, a.train_views).round().astype(int)
    sc.train_idx = sc.train_idx[sel]; imgs = imgs[sel]
    gt_k = gt_k[sel] if gt_k is not None else None
reg = {k[2:]: v for k, v in (("w_centroid", a.w_centroid), ("w_sparse", a.w_sparse), ("w_tv", a.w_tv)) if v}
reg = {f"w_{k}": v for k, v in reg.items()}
m = train(sc, imgs, a.out, iters=a.iters, kernel=a.kernel, ksize=a.ksize, kernel_start=a.kernel_start,
          lr_kernel=a.lr_kernel, reg=reg or None, shared_kernel=a.shared_kernel, kernel_init_sigma=a.kernel_init_sigma, poly_init_len=a.poly_init_len, n_ctrl=a.n_ctrl, gt_kernels=gt_k, seed=a.seed)
print(m)
