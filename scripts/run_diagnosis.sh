#!/bin/bash
# Diagnosis on bonsai motion/medium: oracle kernels (non-blind upper bound), free-form from realistic blur at step 0, polyline kernels.
cd "$(dirname "$0")/.." && . .venv/bin/activate
B=results/bonsai/motion_medium; C="--scene bonsai --tag motion_medium --iters 10000 --input blurred"
python scripts/train.py $C --kernel oracle --kernel_start 0 --out $B/ours_oracle_kernel > $B/ours_oracle_kernel.log 2>&1
python scripts/train.py $C --kernel free --kernel_init_sigma 2.5 --kernel_start 0 --lr_kernel 0.1 --w_centroid 1.0 --out $B/ours_free_s2.5_k0 > $B/ours_free_s2.5_k0.log 2>&1
python scripts/train.py $C --kernel polyline --poly_init_len 6 --kernel_start 0 --lr_kernel 0.05 --out $B/ours_polyline > $B/ours_polyline.log 2>&1
echo done > $B/DONE_DIAG
