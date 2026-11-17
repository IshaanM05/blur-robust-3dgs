#!/bin/bash
# Runs every experiment in the report. Safe to re-run: finished runs (metrics.json present) are skipped.
# Results: results/<scene>/<kind>_<severity>/<method>/; tables via scripts/collect.py, figures via scripts/make_figures.py.
# Needs the Mip-NeRF 360 scenes (see data/README.md) and a GPU. The whole set takes about a day on one laptop GPU.
set -e
cd "$(dirname "$0")/.."

CORE="vanilla wiener_guess ours_polyline ours_oracle_kernel"
run() { python scripts/run_matrix.py "$@"; }

# main table: motion blur, medium severity, three scenes (garden is large, about 15 min per run)
run --combos bonsai:motion:medium counter:motion:medium garden:motion:medium \
    --methods vanilla wiener_oracle wiener_guess rl_guess ours_polyline ours_oracle_kernel sharp_input

# severity sweep on two scenes
run --combos bonsai:motion:low bonsai:motion:high counter:motion:low counter:motion:high \
    --methods vanilla wiener_oracle wiener_guess rl_guess ours_polyline ours_oracle_kernel

# defocus and Gaussian blur, with deliberately wrong Wiener guesses (the default guess is sigma 2)
run --combos bonsai:defocus:medium bonsai:gaussian:medium counter:defocus:medium counter:gaussian:medium \
    --methods vanilla wiener_oracle wiener_guess wiener_guess_s1 wiener_guess_s4 rl_guess \
              ours_polyline ours_gauss ours_oracle_kernel

# noise added to the blurry inputs (standard deviation on a 0-1 scale)
for noise in 0.02 0.05 0.1; do
    run --combos bonsai:gaussian:medium:$noise --methods vanilla wiener_guess ours_gauss ours_oracle_kernel
done

# ablations on bonsai, motion blur, medium severity
run --combos bonsai:motion:medium \
    --methods abl_ctrl3 abl_ctrl12 abl_k15 abl_k35 abl_shared abl_views100 abl_views50 abl_views25 \
              vanilla_views50 vanilla_views25 abl_free_sparse abl_free_tv abl_free_all

# free-form kernel variants (initialisation and learning rate), to check that the collapse is not a tuning issue
B=results/bonsai/motion_medium
COMMON="--scene bonsai --tag motion_medium --iters 10000 --input blurred --kernel free --w_centroid 1.0"
[ -e $B/ours_free_lr0.1/metrics.json ] || python scripts/train.py $COMMON --lr_kernel 0.1 --out $B/ours_free_lr0.1
[ -e $B/ours_free_lr0.3/metrics.json ] || python scripts/train.py $COMMON --lr_kernel 0.3 --out $B/ours_free_lr0.3
[ -e $B/ours_free_s2.5_k0/metrics.json ] || \
    python scripts/train.py $COMMON --lr_kernel 0.1 --kernel_init_sigma 2.5 --kernel_start 0 --out $B/ours_free_s2.5_k0

# training seeds (same blurred data) and blur realisations (different random blur)
for seed in 1 2; do
    run --combos bonsai:motion:medium --methods vanilla wiener_guess ours_polyline --seed $seed --suffix _seed$seed
    run --combos bonsai:motion:medium --methods vanilla wiener_guess ours_polyline --blur-seed $seed
done

# longer training
run --combos bonsai:motion:medium --iters 30000 --suffix _30k --methods vanilla wiener_guess ours_polyline
