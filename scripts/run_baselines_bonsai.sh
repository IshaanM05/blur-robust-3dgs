#!/bin/bash
# Remaining pre-deblur baselines on bonsai motion/medium (blind = guessed gaussian kernel; oracle = true kernel).
cd "$(dirname "$0")/.." && . .venv/bin/activate
B=results/bonsai/motion_medium; C="--scene bonsai --tag motion_medium --iters 10000"
for N in wiener_guess rl_oracle rl_guess; do
  python scripts/train.py $C --input deblur_$N --out $B/$N > $B/$N.log 2>&1
done
echo done > $B/DONE_BASE
