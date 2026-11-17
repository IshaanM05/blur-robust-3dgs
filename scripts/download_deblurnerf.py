"""Download the real camera-motion-blur scenes of the Deblur-NeRF dataset, keeping only what this project uses.

Per scene: images_4/ (quarter resolution), sparse/ (COLMAP reconstruction), poses_bounds.npy and the holdout file.
The full-resolution images, pretrained weights and Blender files are skipped. Scenes already on disk are skipped.

Usage: python scripts/download_deblurnerf.py --list
       python scripts/download_deblurnerf.py [--scenes blurball blurbuick ...] [--out data/deblurnerf]
Dataset: https://drive.google.com/drive/folders/1_TkpcJnw504ZOWmgVTD7vWqPdzbk9Wx_ (folder real_camera_motion_blur)
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import gdown
from gdown.exceptions import FileURLRetrievalError

REAL_CAMERA_MOTION_BLUR = "https://drive.google.com/drive/folders/1X-NfxsZXWH5c4vjaKVjlFnEQU-l54ag_"
WANTED_DIRS = ("images_4", "sparse")
WANTED_FILES = ("poses_bounds.npy", "hold=7")


def wanted(rel_parts):
    """rel_parts: path below the scene folder, e.g. ('images_4', '000.png')."""
    return rel_parts[0] in WANTED_DIRS or (len(rel_parts) == 1 and rel_parts[0] in WANTED_FILES)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="*", help="scene folder names (default: all)")
    p.add_argument("--out", default="data/deblurnerf")
    p.add_argument("--list", action="store_true", help="only list what would be downloaded")
    a = p.parse_args()

    entries = gdown.download_folder(url=REAL_CAMERA_MOTION_BLUR, skip_download=True, quiet=True, remaining_ok=True)
    todo = {}
    for e in entries:
        parts = Path(e.path).parts
        if len(parts) < 2 or any(x in ("", ".", "..") for x in parts) or Path(e.path).is_absolute():
            continue  # never write outside the output folder
        scene, rel = parts[0], parts[1:]
        if (not a.scenes or scene in a.scenes) and wanted(rel):
            todo.setdefault(scene, []).append((e.id, rel))

    print(f"{len(todo)} scenes:", ", ".join(f"{s} ({len(f)} files)" for s, f in sorted(todo.items())))
    if a.list:
        print("file types:", dict(Counter(Path(r[-1]).suffix or r[-1] for f in todo.values() for _, r in f)))
        return

    failures = []
    for scene, files in sorted(todo.items()):
        root = Path(a.out) / scene
        if (root / "sparse" / "0" / "cameras.bin").exists():
            print(f"{scene}: already on disk, skipped")
            continue
        for file_id, rel in files:
            dest = root.joinpath(*rel)
            if dest.exists() and dest.stat().st_size > 0:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                if gdown.download(id=file_id, output=str(dest), quiet=True) is None:
                    failures.append(f"{scene}/{'/'.join(rel)}")
            except FileURLRetrievalError:
                # Drive limits downloads of popular public files; retrying at once only prolongs the block
                print(f"{scene}: Google Drive is rate limiting downloads. Stopping; run again later.", flush=True)
                sys.exit(2)
        print(f"{scene}: done ({len(files)} files)", flush=True)
    if failures:
        print(f"{len(failures)} files failed (Drive may be rate limiting); run again to fetch the rest:", failures[:5])
        sys.exit(1)


if __name__ == "__main__":
    main()
