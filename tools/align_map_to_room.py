"""Align a Gaussian map to the synthetic room, without touching the object.

Placement accuracy has been measured against a similarity fitted on the dynamic
object's own bank centroids -- the very quantity under test. Section 3.51 shows
what that costs: the self-fit lands on scale 2.258 where the scene's is ~4.6, and
that one factor is the whole "the object only travels 0.55x as far as it should"
observation. An alignment estimated from anything other than the object removes
the circularity.

The static room is the obvious anchor and it is known exactly: an axis-aligned
box of half-extents (HX, HY, HZ) centred on the origin, so every static Gaussian
should sit on one of six known planes. That makes this a point-to-plane fit with
no correspondence search and no reference point cloud to compare against.

The initialisation is nearly free. The map's frame is the first camera's, and the
generator's first camera sits at (0, 0, -1.5) with yaw and pitch both
``sin(0) = 0`` -- exactly the identity rotation. So only the scale is really
unknown, and even that starts from a robust ratio of median wall distances.

Rotation is refined too rather than assumed: the first pose being identity is a
property of the *generator*, while the map's frame comes out of tracking, and the
whole point of this tool is to stop assuming things about frames.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from checkpoint_replay import load_gaussians  # noqa: E402

HX, HY, HZ = 4.0, 2.2, 4.0  # room half-extents, make_synthetic_erp_room.py


def room_planes(hx=HX, hy=HY, hz=HZ):
    """The six walls as ``(unit normal, offset)`` with ``n . p = offset``."""
    return [
        (np.array([1.0, 0.0, 0.0]), hx), (np.array([-1.0, 0.0, 0.0]), hx),
        (np.array([0.0, 1.0, 0.0]), hy), (np.array([0.0, -1.0, 0.0]), hy),
        (np.array([0.0, 0.0, 1.0]), hz), (np.array([0.0, 0.0, -1.0]), hz),
    ]


def nearest_plane_residual(points, planes):
    """Signed distance to the nearest wall, and which wall that was."""
    distances = np.stack([points @ n - d for n, d in planes], axis=1)
    index = np.argmin(np.abs(distances), axis=1)
    return distances[np.arange(points.shape[0]), index], index


def apply(points, scale, rotation, translation):
    return scale * (points @ rotation.T) + translation


def fit(points, scale, rotation, translation, planes, iterations=30,
        trim=0.2):
    """Refine sim(3) so the points land on the walls.

    Trimmed: a reconstruction has floaters and the room has an object in it, and
    a least-squares fit with no rejection would let those set the scale. Section
    3.29's outlier banks are the standing reminder of what that does.
    """
    def cost(s, r, t):
        residual, _ = nearest_plane_residual(apply(points, s, r, t), planes)
        kept = np.abs(residual) <= np.quantile(np.abs(residual), 1.0 - trim)
        return float(np.sqrt((residual[kept] ** 2).mean()))

    best = cost(scale, rotation, translation)
    for _ in range(iterations):
        placed = apply(points, scale, rotation, translation)
        residual, index = nearest_plane_residual(placed, planes)
        keep = np.abs(residual) <= np.quantile(np.abs(residual), 1.0 - trim)
        source = points[keep]
        normals = np.stack([planes[i][0] for i in index[keep]])
        offsets = np.array([planes[i][1] for i in index[keep]])

        # Gauss-Newton on (log scale, rotation vector, translation): the
        # measurement is scalar per point, so this stays a 7-column normal
        # equation however many Gaussians are involved.
        placed_keep = apply(source, scale, rotation, translation)
        rotated = scale * (source @ rotation.T)
        jacobian = np.concatenate([
            (normals * rotated).sum(axis=1, keepdims=True),          # d/dlog s
            # d(n . R p)/dw for R <- (I + [w]x) R is w . (Rp x n); getting this
            # cross product backwards spins the fit into a 96-degree "solution"
            # that fits nothing, which is exactly what it did first time round
            np.cross(rotated, normals),                              # d/drotvec
            normals,                                                 # d/dt
        ], axis=1)
        error = (normals * placed_keep).sum(axis=1) - offsets
        step, *_ = np.linalg.lstsq(jacobian, -error, rcond=None)

        # accept only what actually helps: the correspondences are recomputed
        # every iteration, so a step can improve its own linearisation while
        # making the real objective worse
        damping = 1.0
        for _ in range(8):
            trial_scale = scale * float(np.exp(damping * step[0]))
            trial_rotation = rodrigues(damping * step[1:4]) @ rotation
            trial_translation = translation + damping * step[4:7]
            trial = cost(trial_scale, trial_rotation, trial_translation)
            if trial < best:
                scale, rotation, translation, best = (
                    trial_scale, trial_rotation, trial_translation, trial)
                break
            damping *= 0.5
        else:
            break  # no step of any length helps; the fit has converged
    return scale, rotation, translation


def rodrigues(vector):
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    cross = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    return (np.eye(3) + np.sin(angle) * cross
            + (1 - np.cos(angle)) * (cross @ cross))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trim", type=float, default=0.2)
    parser.add_argument("--save", default=None,
                        help="write the fitted transform as an .npz")
    args = parser.parse_args()

    gaussians, _ = load_gaussians(args.checkpoint)
    static = gaussians.get_xyz[~gaussians.dynamic_mask].detach().cpu().numpy()
    # floaters would drag the initial scale; the fit trims, the init cannot
    keep = np.abs(static - np.median(static, axis=0)).max(axis=1)
    static = static[keep <= np.quantile(keep, 0.98)]

    planes = room_planes()
    # initial scale: the map's own half-extent against the room's, on the axis
    # the walls are best observed on
    extent = np.quantile(static, 0.98, axis=0) - np.quantile(static, 0.02, axis=0)
    scale = float(np.mean([2 * HX / extent[0], 2 * HY / extent[1],
                           2 * HZ / extent[2]]))
    rotation = np.eye(3)
    translation = np.array([0.0, 0.0, -1.5])  # generator's first camera

    before, _ = nearest_plane_residual(
        apply(static, scale, rotation, translation), planes)
    scale, rotation, translation = fit(
        static, scale, rotation, translation, planes, trim=args.trim)
    after, _ = nearest_plane_residual(
        apply(static, scale, rotation, translation), planes)

    angle = float(np.degrees(np.arccos(
        np.clip((np.trace(rotation) - 1) / 2, -1, 1))))
    print(f"checkpoint : {args.checkpoint}")
    print(f"initial    : scale {scale:.3f} "
          f"(per-axis {[round(float(h / e), 3) for h, e in zip((2*HX, 2*HY, 2*HZ), extent)]})")
    print(f"fitted     : scale {scale:.3f}, rotation {angle:.2f} deg, "
          f"translation {[round(float(v), 3) for v in translation]}")
    print(f"wall residual (m): median {np.median(np.abs(before)):.3f} -> "
          f"{np.median(np.abs(after)):.3f}, "
          f"90th pct {np.quantile(np.abs(before), 0.9):.3f} -> "
          f"{np.quantile(np.abs(after), 0.9):.3f}")
    print(f"room is {2*HX} x {2*HY} x {2*HZ}, so a median residual of "
          f"{np.median(np.abs(after)):.3f} is "
          f"{100 * np.median(np.abs(after)) / (2 * HY):.1f}% of its shortest side")

    if args.save:
        np.savez(args.save, scale=scale, rotation=rotation,
                 translation=translation)
        print(f"saved transform -> {args.save}")


if __name__ == "__main__":
    main()
