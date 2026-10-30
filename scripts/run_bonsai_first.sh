#!/bin/bash
# First real comparison: bonsai, motion/medium, 10k iters. Runs sequentially, logs to results/bonsai/motion_medium/*.log
cd "$(dirname "$0")/.." && . .venv/bin/activate
B=results/bonsai/motion_medium; mkdir -p $B
C="--scene bonsai --tag motion_medium --iters 10000"
python scripts/train.py $C --input blurred --out $B/vanilla > $B/vanilla.log 2>&1
python scripts/train.py $C --input deblur_wiener_oracle --out $B/wiener_oracle > $B/wiener_oracle.log 2>&1
python scripts/train.py $C --input blurred --kernel free --w_centroid 1.0 --out $B/ours_free > $B/ours_free.log 2>&1
echo done > $B/DONE
