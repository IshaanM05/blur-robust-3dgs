"""Mip-NeRF 360 loader (COLMAP poses, `images_{factor}` folder, pinhole cameras).

Standard split: every `test_every`-th image (sorted by name) is a test view.
"""

from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


@dataclass
class Scene:
    names: list
    images: np.ndarray  # (V,H,W,3) uint8, sharp ground truth
    viewmats: np.ndarray  # (V,4,4) world->camera
    Ks: np.ndarray  # (V,3,3) at the images' resolution
    points: np.ndarray  # (P,3)
    colors: np.ndarray  # (P,3) uint8
    train_idx: np.ndarray
    test_idx: np.ndarray
    scene_scale: float

    @property
    def hw(self):
        return self.images.shape[1:3]


def _pose(img):
    cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
    m = np.eye(4)
    m[:3] = cfw.matrix()
    return m


def _find_image(img_dir: Path, name: str) -> Path:
    """The file for a COLMAP image name. Some datasets list `.jpg` in COLMAP but ship `.png` in the resized folders."""
    path = img_dir / name
    if path.exists():
        return path
    matches = sorted(img_dir.glob(Path(name).stem + ".*"))
    if not matches:
        raise FileNotFoundError(f"no image for {name} in {img_dir}")
    return matches[0]


def load_scene(root: str | Path, factor: int = 4, test_every: int = 8, max_views: int | None = None) -> Scene:
    import pycolmap

    root = Path(root)
    rec = pycolmap.Reconstruction(str(root / "sparse" / "0"))
    img_dir = root / f"images_{factor}"
    items = sorted(rec.images.values(), key=lambda im: im.name)
    if max_views:
        items = items[:: max(1, len(items) // max_views)][:max_views]
    names, images, viewmats, Ks = [], [], [], []
    for im in items:
        cam = rec.cameras[im.camera_id]
        model = str(cam.model).split(".")[-1]
        if model not in ("PINHOLE", "SIMPLE_PINHOLE"):
            raise ValueError(f"camera model {model} not supported (need undistorted pinhole images)")
        p = np.asarray(cam.params, dtype=np.float64)
        fx, fy, cx, cy = (p[0], p[0], p[1], p[2]) if model == "SIMPLE_PINHOLE" else (p[0], p[1], p[2], p[3])
        pic = imageio.imread(_find_image(img_dir, im.name))[..., :3]
        s = pic.shape[1] / cam.width
        names.append(im.name)
        images.append(pic)
        viewmats.append(_pose(im))
        Ks.append([[fx * s, 0, cx * s], [0, fy * s, cy * s], [0, 0, 1]])
    images = np.stack(images)
    viewmats = np.stack(viewmats).astype(np.float32)
    pts = np.array([p.xyz for p in rec.points3D.values()], dtype=np.float32)
    cols = np.array([p.color for p in rec.points3D.values()], dtype=np.uint8)
    idx = np.arange(len(names))
    test = idx[idx % test_every == 0]
    train = np.setdiff1d(idx, test)
    centers = np.linalg.inv(viewmats)[:, :3, 3]
    scale = float(np.linalg.norm(centers - centers.mean(0), axis=1).max() * 1.1)
    return Scene(names, images, viewmats, np.array(Ks, np.float32), pts, cols, train, test, scale)
