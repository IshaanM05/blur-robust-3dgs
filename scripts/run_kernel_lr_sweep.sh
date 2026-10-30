#!/bin/bash
# Kernel learning-rate sweep on bonsai motion/medium (free-form, gaussian init sigma=1.0)
cd "$(dirname "$0")/.." && . .venv/bin/activate
B=results/bonsai/motion_medium
for LR in 0.1 0.3; do
  python scripts/train.py --scene bonsai --tag motion_medium --iters 10000 --input blurred --kernel free \
    --lr_kernel $LR --w_centroid 1.0 --out $B/ours_free_lr$LR > $B/ours_free_lr$LR.log 2>&1
done
echo done > $B/DONE_SWEEP
