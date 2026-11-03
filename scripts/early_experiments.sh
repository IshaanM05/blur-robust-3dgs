#!/bin/bash
# Free-form kernel variants tried before switching to the polyline prior (see docs/log.md).
# All on bonsai / motion_medium; compare against the polyline run from run_all.sh.
cd "$(dirname "$0")/.."
. .venv/bin/activate

B=results/bonsai/motion_medium
COMMON="--scene bonsai --tag motion_medium --iters 10000 --input blurred --kernel free --w_centroid 1.0"

python scripts/train.py $COMMON --lr_kernel 0.1 --out $B/ours_free_lr0.1
python scripts/train.py $COMMON --lr_kernel 0.3 --out $B/ours_free_lr0.3
python scripts/train.py $COMMON --lr_kernel 0.1 --kernel_init_sigma 2.5 --kernel_start 0 --out $B/ours_free_s2.5_k0
