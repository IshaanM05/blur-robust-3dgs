"""Check that the cached blurred photographs and the finished-run files can be read.

A power cut can leave a half-written file. Run before restarting; exits with status 1 if anything is damaged.
"""

import json
import sys
from pathlib import Path

import numpy as np

damaged = []
caches = sorted(Path("data/blurred").glob("*/*/*.npy"))
for path in caches:
    try:
        np.load(path, mmap_mode="r")  # raises if the file is shorter than its header says
    except Exception as err:
        damaged.append((path, err))
results = sorted(Path("results").glob("*/*/*/metrics.json"))
for path in results:
    try:
        json.load(open(path))
    except Exception as err:
        damaged.append((path, err))

print(f"checked {len(caches)} cached arrays and {len(results)} finished runs")
for path, err in damaged:
    print(f"DAMAGED: {path} ({err})")
sys.exit(1 if damaged else 0)
