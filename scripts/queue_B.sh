#!/bin/bash
# Queue B (runs after queue A finishes): gaussian kernel on other blur types, ablations, extra seeds. Resumable.
cd "$(dirname "$0")/.." && . .venv/bin/activate
until [ -f results/DONE_QUEUE_A ]; do sleep 30; done
python scripts/run_matrix.py --combos bonsai:defocus:medium bonsai:gaussian:medium --methods ours_gauss
python scripts/run_matrix.py --combos bonsai:motion:medium --methods abl_ctrl3 abl_ctrl12 abl_k15 abl_k35 abl_shared abl_views100 abl_views50 abl_views25 vanilla_views50 vanilla_views25
for S in 1 2; do
  python scripts/run_matrix.py --combos bonsai:motion:medium --methods vanilla wiener_guess ours_polyline --seed $S --suffix _seed$S
done
echo done > results/DONE_QUEUE_B
