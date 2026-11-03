#!/bin/bash
# Secondary experiments: run after run_all.sh (waits for it to finish). Safe to re-run; finished runs are skipped.
cd "$(dirname "$0")/.."
. .venv/bin/activate
while pgrep -x run_all.sh > /dev/null; do sleep 30; done

# does a stronger regularizer rescue the free-form kernel?
python scripts/run_matrix.py --combos bonsai:motion:medium --methods abl_free_sparse abl_free_tv abl_free_all

# severity sweep on a second scene
python scripts/run_matrix.py --combos counter:motion:low counter:motion:high \
    --methods vanilla wiener_oracle wiener_guess ours_polyline ours_oracle_kernel

# reference baselines for the bonsai sweeps and the other blur types
python scripts/run_matrix.py --combos bonsai:motion:low bonsai:motion:high bonsai:defocus:medium bonsai:gaussian:medium \
    --methods wiener_oracle rl_guess

# RL baseline on garden
python scripts/run_matrix.py --combos garden:motion:medium --methods rl_guess
