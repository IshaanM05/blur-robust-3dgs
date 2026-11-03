#!/bin/bash
# While experiments run, regenerate docs/results.md every 45 minutes and push it if it changed.
cd "$(dirname "$0")/.."
. .venv/bin/activate

sync_results() {
    python scripts/collect.py > /dev/null
    git add docs/results.md
    if ! git diff --cached --quiet; then
        git commit -q -m "Update results table ($(ls results/*/*/*/metrics.json | wc -l) runs)"
        git push -q origin main || true
    fi
}

while pgrep -x run_all.sh > /dev/null || pgrep -x run_extras.sh > /dev/null; do
    sleep 2700
    sync_results
done
sync_results
