#!/bin/bash
# Restart the experiment queue after scripts/pause.sh (or after a crash or power cut). Finished runs are kept and skipped.
cd "$(dirname "$0")/.."
. .venv/bin/activate

if pgrep -x run_final.sh > /dev/null; then
    echo "the queue is already running"
    exit 1
fi
python scripts/check_state.py || { echo "fix the problems above before restarting"; exit 1; }

rm -f results/PAUSE results/DONE_FINAL
nohup scripts/run_final.sh > results/run_final.log 2>&1 &
echo "queue restarted; follow it with: tail -f results/final_a.log results/final_b.log"
