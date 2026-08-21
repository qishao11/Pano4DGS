"""Convert a 3DGS ply into the antimatter15 `.splat` format for web/desktop viewers.

Layout is 32 bytes per gaussian: position f32[3], scale f32[3], color u8[4] (rgba),
rotation u8[4] (wxyz quaternion mapped to 0..255). Gaussians are written in
descending importance so viewers that truncate keep the significant ones.

Example:
    python tools/gs_ply_to_splat.py --ply outputs/replica/room0/3dgs_final.ply \
        --out /tmp/room0.splat
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from plyfile import PlyData

logger = logging.getLogger(__name__)

SH_C0 = 0.28209479177387814


def convert(ply_path: Path, out_path: Path, min_opacity: float, max_scale: float) -> None:
    ply = PlyData.read(str(ply_path))["vertex"]
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], axis=-1).astype(np.float32)
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(ply["opacity"], dtype=np.float32)))
    scale = np.exp(
        np.stack([ply[f"scale_{i}"] for i in range(3)], axis=-1).astype(np.float32)
    )
    rot = np.stack([ply[f"rot_{i}"] for i in range(4)], axis=-1).astype(np.float32)
    rot = rot / (np.linalg.norm(rot, axis=-1, keepdims=True) + 1e-9)
    sh = np.stack([ply[f"f_dc_{i}"] for i in range(3)], axis=-1).astype(np.float32)
    rgb = np.clip(sh * SH_C0 + 0.5, 0.0, 1.0)

    keep = opacity >= min_opacity
    if max_scale > 0:
        keep &= scale.max(axis=-1) <= max_scale
    xyz, opacity, scale, rot, rgb = (
        arr[keep] for arr in (xyz, opacity, scale, rot, rgb)
    )

    importance = opacity * scale.prod(axis=-1)
    order = np.argsort(-importance)
    xyz, opacity, scale, rot, rgb = (
        arr[order] for arr in (xyz, opacity, scale, rot, rgb)
    )

    n = len(xyz)
    buf = np.zeros((n, 32), dtype=np.uint8)
    buf[:, 0:12] = xyz.astype("<f4").view(np.uint8).reshape(n, 12)
    buf[:, 12:24] = scale.astype("<f4").view(np.uint8).reshape(n, 12)
    buf[:, 24:27] = np.clip(rgb * 255, 0, 255).astype(np.uint8)
    buf[:, 27] = np.clip(opacity * 255, 0, 255).astype(np.uint8)
    buf[:, 28:32] = np.clip(rot * 128 + 128, 0, 255).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(buf.tobytes())
    logger.info("%s: wrote %d gaussians -> %s", ply_path.parent.name, n, out_path)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-opacity", type=float, default=0.0)
    parser.add_argument(
        "--max-scale", type=float, default=0.0,
        help="drop gaussians whose largest axis exceeds this (meters); 0 disables",
    )
    args = parser.parse_args()
    convert(Path(args.ply), Path(args.out), args.min_opacity, args.max_scale)


if __name__ == "__main__":
    main()
