"""Is the bank centroid biased toward the camera, and by how much?

Sections 3.40/3.48 attribute the residual placement error to a structural bias:
each view seeds only the surface facing it, so a bank is a shell patch rather
than a full object, and its centroid sits toward the camera by roughly the
object's radius. That has been an explanation, never a measurement. Everything
below is the measurement.

Two properties separate this from ordinary noise:

  direction   the residual should point from the object toward the camera. A
              random error has cosine 0 on average; a camera-facing bias has
              cosine near 1 and, crucially, stays near 1 as the camera moves.
  magnitude   it should be on the order of the object's radius, not of the
              reconstruction's noise floor.

The similarity fit is the reason this is worth measuring rather than assuming.
Umeyama absorbs any *constant* offset into its translation, so a bias fixed in
world space would leave no residual at all. What survives the fit is the part
that rotates with the camera -- which is exactly the part a camera-facing bias
contributes, and exactly the part no global alignment can remove.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint_replay import (  # noqa: E402
    apply_similarity,
    bank_statistics,
    ground_truth_centre,
    load_gaussians,
    load_ground_truth,
    observed_times,
    umeyama,
)


def load_camera_centres(path):
    """``{tstamp: centre}`` from a TUM-style trajectory file.

    The saved rows are built from a world-to-camera matrix, so the camera centre
    is ``-R^T t`` rather than the stored translation. Both readings are returned
    so the caller can check which one places the cameras inside the room instead
    of trusting a convention that has bitten this project before (section 3.8).
    """
    rows = np.loadtxt(path)
    if rows.ndim == 1:
        rows = rows[None]
    as_stored, inverted = {}, {}
    for row in rows:
        stamp, translation, quat = float(row[0]), row[1:4], row[4:8]
        x, y, z, w = quat
        rotation = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        as_stored[stamp] = torch.tensor(translation, dtype=torch.float32)
        inverted[stamp] = torch.tensor(-rotation.T @ translation,
                                       dtype=torch.float32)
    return as_stored, inverted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gt", required=True)
    parser.add_argument("--traj", default=None,
                        help="the run's traj_mcgs.txt; defaults to the "
                             "checkpoint's own directory")
    parser.add_argument("--max-extent", type=float, default=None)
    args = parser.parse_args()

    gaussians, _ = load_gaussians(args.checkpoint)
    traj = args.traj or str(Path(args.checkpoint).parent / "traj_mcgs.txt")
    gt_entries = load_ground_truth(args.gt)
    times = observed_times(gaussians)

    if args.max_extent is None:
        static = gaussians.get_xyz[~gaussians.dynamic_mask]
        scene = float((static.quantile(0.98, dim=0)
                       - static.quantile(0.02, dim=0)).max())
        args.max_extent = 2.0 * scene

    centroids, extents, clean = {}, {}, []
    for time in times:
        _, centroid, extent = bank_statistics(gaussians, time)
        centroids[time], extents[time] = centroid, extent
        if extent <= args.max_extent:
            clean.append(time)
    if len(clean) < 3:
        raise SystemExit("need three uncontaminated banks to fit the alignment")

    transform = umeyama(
        torch.stack([centroids[t] for t in clean]),
        torch.stack([ground_truth_centre(gt_entries, t) for t in clean]))
    scale = transform[0]

    stored, inverted = load_camera_centres(traj)
    # pick the convention that puts the cameras nearer the object, i.e. inside
    # the room rather than mirrored through the origin
    def mean_distance(centres):
        distances = []
        for time in clean:
            centre = centres.get(time)
            if centre is None:
                continue
            distances.append(float(torch.linalg.vector_norm(
                apply_similarity(transform, centre[None])[0]
                - ground_truth_centre(gt_entries, time))))
        return float(np.mean(distances)) if distances else float("inf")

    label, centres = min((("-R^T t", inverted), ("stored t", stored)),
                         key=lambda pair: mean_distance(pair[1]))
    print(f"checkpoint   : {args.checkpoint}")
    print(f"alignment    : scale {scale:.3f} on {len(clean)} clean banks")
    print(f"camera centre: using {label} "
          f"(mean camera-object distance {mean_distance(centres):.3f})\n")

    object_size = min(extents[t] for t in clean) * scale
    print(f"   object size in GT units: {object_size:.3f}\n")
    print(f"   {'time':>6}{'|residual|':>12}{'cos(to camera)':>16}"
          f"{'cam distance':>14}")

    residuals, cosines, ratios = [], [], []
    for time in clean:
        centre = centres.get(time)
        if centre is None:
            continue
        placed = apply_similarity(transform, centroids[time][None])[0]
        truth = ground_truth_centre(gt_entries, time)
        residual = placed - truth
        camera = apply_similarity(transform, centre[None])[0]
        to_camera = camera - truth
        distance = float(torch.linalg.vector_norm(to_camera))
        norm = float(torch.linalg.vector_norm(residual))
        cosine = float(torch.dot(residual, to_camera)
                       / (norm * distance + 1e-12))
        residuals.append(norm)
        cosines.append(cosine)
        ratios.append(norm / max(object_size, 1e-9))
        print(f"   {time:>6.1f}{norm:>12.3f}{cosine:>16.3f}{distance:>14.3f}")

    mean_cos = float(np.mean(cosines))
    print(f"\n   mean |residual| {np.mean(residuals):.3f} "
          f"({np.mean(ratios):.2f} object sizes)")
    print(f"   mean cos(residual, toward camera) {mean_cos:+.3f} "
          f"over {len(cosines)} banks, sd {np.std(cosines):.3f}")
    print(f"   banks pointing at the camera: "
          f"{sum(c > 0 for c in cosines)}/{len(cosines)}")
    if mean_cos > 0.5:
        print("\n   -> camera-facing bias confirmed: correcting it needs an "
              "offset along the view ray, which a global alignment cannot do")
    elif mean_cos < -0.5:
        print("\n   -> residual points *away* from the camera, which the "
              "surface-patch story does not predict")
    else:
        print("\n   -> no consistent camera-facing direction; the residual is "
              "not explained by the surface-patch bias")


if __name__ == "__main__":
    main()
