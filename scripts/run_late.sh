#!/bin/bash
# Runs that did not fit the first schedule: longer training on bonsai and RL on counter motion blur.
cd "$(dirname "$0")/.." && source .venv/bin/activate
python scripts/run_matrix.py --combos bonsai:motion:medium --iters 30000 --suffix _30k --methods vanilla wiener_guess ours_polyline
python scripts/run_matrix.py --combos counter:motion:low counter:motion:high --methods rl_guess
touch results/DONE_LATE
