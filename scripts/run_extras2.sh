#!/bin/bash
# Robustness checks on the Gaussian and defocus blur sets. Runs after run_extras.sh (waits for it). Safe to re-run.
cd "$(dirname "$0")/.."
. .venv/bin/activate
while pgrep -x run_extras.sh > /dev/null; do sleep 30; done

# Wiener with a deliberately wrong kernel guess (sigma 1 and 4; the default guess is sigma 2)
python scripts/run_matrix.py --combos bonsai:gaussian:medium bonsai:defocus:medium \
    --methods wiener_guess_s1 wiener_guess_s4

# the same comparison with sensor-like noise added to the blurry inputs (sigma 0.02 on a 0-1 scale)
python scripts/run_matrix.py --combos bonsai:gaussian:medium:0.02 \
    --methods vanilla wiener_guess ours_gauss ours_oracle_kernel
