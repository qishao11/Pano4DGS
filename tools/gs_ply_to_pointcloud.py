"""Export a 3DGS ply as a plain colored point cloud (gaussian centers + SH0 color).

Example:
    python tools/gs_ply_to_pointcloud.py --ply outputs/replica/room0/3dgs_final.ply \
        --out /tmp/room0_points.ply --min-opacity 0.1
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement

logger = logging.getLogger(__name__)

SH_C0 = 0.28209479177387814


def export(ply_path: Path, out_path: Path, min_opacity: float, max_scale: float) -> None:
    ply = PlyData.read(str(ply_path))["vertex"]
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], axis=-1).astype(np.float32)
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(ply["opacity"], dtype=np.float32)))
    scale = np.exp(
        np.stack([ply[f"scale_{i}"] for i in range(3)], axis=-1).astype(np.float32)
    )
    sh = np.stack([ply[f"f_dc_{i}"] for i in range(3)], axis=-1).astype(np.float32)
    rgb = np.clip(sh * SH_C0 + 0.5, 0.0, 1.0)

    keep = opacity >= min_opacity
    if max_scale > 0:
        keep &= scale.max(axis=-1) <= max_scale
    logger.info("%s: keep %d / %d gaussians", ply_path.parent.name, keep.sum(), len(xyz))

    verts = np.empty(
        int(keep.sum()),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
               ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    verts["x"], verts["y"], verts["z"] = xyz[keep].T
    rgb8 = (rgb[keep] * 255).astype(np.uint8)
    verts["red"], verts["green"], verts["blue"] = rgb8.T

    out_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(verts, "vertex")], text=False).write(str(out_path))
    logger.info("wrote %s", out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-opacity", type=float, default=0.1)
    parser.add_argument(
        "--max-scale", type=float, default=0.0,
        help="drop gaussians whose largest axis exceeds this (meters); 0 disables",
    )
    args = parser.parse_args()
    export(Path(args.ply), Path(args.out), args.min_opacity, args.max_scale)


if __name__ == "__main__":
    main()
