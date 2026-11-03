#!/bin/bash
# Queue A (Nov 3): finish garden, severity sweep, other blur types. Resumable (run_matrix skips finished runs).
cd "$(dirname "$0")/.." && . .venv/bin/activate
CORE="vanilla wiener_guess ours_polyline ours_oracle_kernel"
python scripts/run_matrix.py --combos garden:motion:medium --methods ours_polyline ours_oracle_kernel
python scripts/run_matrix.py --combos bonsai:motion:low bonsai:motion:high --methods $CORE
python scripts/run_matrix.py --combos bonsai:defocus:medium bonsai:gaussian:medium --methods $CORE
echo done > results/DONE_QUEUE_A
