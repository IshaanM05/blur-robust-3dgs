#!/bin/bash
# Pause the experiment queue safely: start no new runs, wait for the runs in progress to finish, then say when it is
# safe to shut the machine down. Resume afterwards with scripts/resume.sh.
cd "$(dirname "$0")/.."

touch results/PAUSE
echo "pause requested: no new runs will start"

busy() { pgrep -f "scripts/(train|prepare_blur|prepare_deblur|run_matrix)\.py" > /dev/null; }

quiet_checks=0
while [ "$quiet_checks" -lt 2 ]; do
    if busy; then
        quiet_checks=0
        echo "$(date +%H:%M:%S) waiting for $(pgrep -fc 'scripts/(train|prepare_blur|prepare_deblur|run_matrix)\.py') running process(es)"
    else
        quiet_checks=$((quiet_checks + 1))
    fi
    sleep 10
done
echo "$(date +%H:%M:%S) everything has finished. SAFE TO SHUT DOWN. Restart later with scripts/resume.sh"
