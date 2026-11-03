#!/bin/bash
# Runs every experiment in the report. Safe to re-run: finished runs (metrics.json present) are skipped.
# Results: results/<scene>/<kind>_<severity>/<method>/, tables via scripts/collect.py, figures via scripts/make_figures.py.
set -e
cd "$(dirname "$0")/.."
. .venv/bin/activate

CORE="vanilla wiener_guess ours_polyline ours_oracle_kernel"

# main table: motion blur, medium severity, three scenes
python scripts/run_matrix.py --combos bonsai:motion:medium counter:motion:medium \
    --methods vanilla wiener_oracle wiener_guess rl_guess ours_polyline ours_oracle_kernel
# garden is large (~3M Gaussians, ~15 min per run), so the RL baseline is skipped there
python scripts/run_matrix.py --combos garden:motion:medium \
    --methods vanilla wiener_oracle wiener_guess ours_polyline ours_oracle_kernel

# severity sweep and other blur types (bonsai)
python scripts/run_matrix.py --combos bonsai:motion:low bonsai:motion:high bonsai:defocus:medium bonsai:gaussian:medium \
    --methods $CORE
python scripts/run_matrix.py --combos bonsai:defocus:medium bonsai:gaussian:medium --methods ours_gauss

# ablations on bonsai / motion / medium
python scripts/run_matrix.py --combos bonsai:motion:medium \
    --methods abl_ctrl3 abl_ctrl12 abl_k15 abl_k35 abl_shared abl_views100 abl_views50 abl_views25 \
              vanilla_views50 vanilla_views25

# training-seed variation (same blurred data)
for seed in 1 2; do
    python scripts/run_matrix.py --combos bonsai:motion:medium --methods vanilla wiener_guess ours_polyline \
        --seed $seed --suffix _seed$seed
done
