#!/bin/bash
# Keep trying to download the remaining Deblur-NeRF real scenes (Google Drive rate limits them), and train on each new
# scene as it arrives. Waits 8 minutes between attempts; stops after 8 attempts or once every scene is on disk.
cd "$(dirname "$0")/.."
. .venv/bin/activate
for attempt in 1 2 3 4 5 6 7 8; do
    python scripts/download_deblurnerf.py && { scripts/run_real.sh; exit 0; }
    scripts/run_real.sh
    sleep 480
done
