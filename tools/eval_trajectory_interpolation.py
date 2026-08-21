"""Does the object-level SE(3) trajectory actually interpolate to unobserved times?

This is the one capability the current per-timestamp oracle bank lacks: it stores
an independent point cloud per observed timestamp, so at any time it was not
shown, the dynamic object simply is not rendered. The replacement representation
(object-local coordinates + an SE(3) trajectory) is supposed to predict those
times instead.

The test fits a trajectory to a *subset* of the ground-truth sphere positions and
measures error on the held-out ones, alongside what the lookup table can do at
those same times (nothing -- reported as a miss).

Honest caveat: the synthetic motions are piecewise linear, so linear translation
interpolation is expected to be near-exact away from direction changes. This
measures whether the plumbing works and quantifies the capability gap, not how
well the representation handles hard motion. `bounce_x` reverses mid-sequence and
is the only case here that stresses the interpolant at all.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))

from gaussian.deform.object_trajectory import ObjectSE3Trajectory  # noqa: E402


def load_ground_truth(path):
    entries = json.load(open(path))
    times = [float(e["frame"]) for e in entries]
    centers = [[float(v) for v in e["center_xyz"]] for e in entries]
    return times, centers


def split_observed(times, centers, stride):
    """Observe every `stride`-th frame; hold out the rest."""
    observed, held_out = [], []
    for index, (t, c) in enumerate(zip(times, centers)):
        (observed if index % stride == 0 else held_out).append((t, c))
    # keep the last frame observed so held-out times stay inside the fitted span
    # (outside it every representation just clamps, which tests nothing)
    if held_out and held_out[-1][0] > observed[-1][0]:
        observed.append(held_out.pop())
    return observed, held_out


def fit_trajectory(observed):
    times = [t for t, _ in observed]
    centers = [c for _, c in observed]
    return ObjectSE3Trajectory(times, centers)


def evaluate(trajectory, held_out):
    errors = []
    for t, target in held_out:
        predicted, _ = trajectory.evaluate(t)
        errors.append(torch.linalg.vector_norm(
            predicted - torch.tensor(target, dtype=predicted.dtype)).item())
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", nargs="+", required=True,
                        help="ground-truth sphere trajectory JSON file(s)")
    parser.add_argument("--stride", type=int, default=4,
                        help="observe every Nth frame, hold out the rest")
    args = parser.parse_args()

    print(f"observing every {args.stride}th frame, holding out the rest\n")
    header = (f"{'sequence':<34}{'obs':>5}{'held':>6}"
              f"{'SE3 mean':>11}{'SE3 max':>10}{'lookup':>10}")
    print(header)
    print("-" * len(header))

    spans = []
    for path in args.gt:
        times, centers = load_ground_truth(path)
        observed, held_out = split_observed(times, centers, args.stride)
        if not held_out:
            print(f"{Path(path).stem:<34} (no held-out frames at this stride)")
            continue

        errors = evaluate(fit_trajectory(observed), held_out)
        spans.append(max(max(c) for c in centers) - min(min(c) for c in centers))
        name = Path(path).stem.replace("_sphere_gt_trajectory", "")
        # the per-timestamp bank has no entry at a held-out time, so the object
        # is absent from the render there -- no prediction at all
        print(f"{name:<34}{len(observed):>5}{len(held_out):>6}"
              f"{sum(errors) / len(errors):>11.4f}{max(errors):>10.4f}"
              f"{'absent':>10}")

    if spans:
        print(f"\n(scene extent for scale: sphere travels within a room of order "
              f"{max(spans):.1f} units; 'lookup' = what the per-timestamp oracle "
              f"bank renders at a held-out time)")


if __name__ == "__main__":
    main()
