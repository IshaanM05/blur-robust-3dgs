#!/bin/bash
# Last round of experiments, in two groups that run side by side on one GPU. Resumable: finished runs are skipped.
#   A: sharp-input reference, defocus and Gaussian blur on counter
#   B: other random blurs, more noise, longer training, last table gap
cd "$(dirname "$0")/.."
. .venv/bin/activate

group_a() {
    python scripts/run_matrix.py --combos bonsai:motion:medium counter:motion:medium garden:motion:medium \
        --methods sharp_input
    python scripts/run_matrix.py --combos counter:defocus:medium counter:gaussian:medium \
        --methods vanilla wiener_guess wiener_guess_s4 ours_polyline ours_gauss ours_oracle_kernel
}

group_b() {
    for blur_seed in 1 2; do
        python scripts/run_matrix.py --combos bonsai:motion:medium --blur-seed $blur_seed \
            --methods vanilla wiener_guess ours_polyline
    done
    python scripts/run_matrix.py --combos bonsai:gaussian:medium:0.05 bonsai:gaussian:medium:0.1 \
        --methods vanilla wiener_guess ours_gauss ours_oracle_kernel
    python scripts/run_matrix.py --combos bonsai:motion:medium --iters 30000 --suffix _30k \
        --methods vanilla wiener_guess ours_polyline
    python scripts/run_matrix.py --combos counter:motion:low counter:motion:high --methods rl_guess
}

group_a > results/final_a.log 2>&1 &
group_b > results/final_b.log 2>&1 &
wait
echo done > results/DONE_FINAL
