#!/bin/bash
# Train vanilla 3DGS and our polyline-kernel method on every real Deblur-NeRF scene that has been downloaded.
# The photographs are genuinely blurry and there is no sharp ground truth, so the output is qualitative: compare the
# held-out renders in results/real_scenes/<scene>_{vanilla,ours}/test_*.png. Resumable; a scene that does not load
# (for example because its download is unfinished) is skipped. To stop after the current run: touch results/STOP_REAL
cd "$(dirname "$0")/.."
. .venv/bin/activate
mkdir -p results/real_scenes
iters=${ITERS:-7000}  # these scenes have only 25 to 45 photographs

for dir in data/deblurnerf/*/; do
    scene=$(basename "$dir")
    python -c "import sys; from blur3dgs.gs3d.dataset import load_scene; load_scene(sys.argv[1])" "$dir" 2> /dev/null \
        || { echo "skip $scene (not fully downloaded)"; continue; }
    for method in vanilla ours; do
        out=results/real_scenes/${scene}_$method
        [ -e results/STOP_REAL ] && { echo "stopped before $scene $method"; exit 0; }
        [ -e "$out/metrics.json" ] && { echo "skip (done) $scene $method"; continue; }
        args="--root data/deblurnerf --scene $scene --input clean --iters $iters --out $out"
        [ "$method" = ours ] && args="$args --kernel polyline --poly_init_len 6 --kernel_start 0 --lr_kernel 0.05"
        python scripts/train.py $args > "$out.log" 2>&1
        echo "done $scene $method rc=$?"
    done
done
echo "ALL DONE"
