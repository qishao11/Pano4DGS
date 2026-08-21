"""Write the ground-truth camera trajectory of a synthetic room sequence.

The poses were never missing. ``make_synthetic_erp_room.camera_pose`` defines
them analytically -- a forward dolly with a small sinusoidal weave -- but they
lived inside the render loop and were discarded once the frames were written, so
every synthetic sequence shipped with a sphere trajectory and no camera
trajectory. On that basis the project recorded that ATE was unmeasurable on
synthetic data (panoramic_4dgs_status.md 3.51 corrects this).

Writing them out buys two things:

  ATE          evo can now compare traj_mcgs.txt against a real reference, so
               "how good is the panoramic SLAM" stops being an empty column.
  alignment    placement error was being measured against a similarity fitted on
               the *object's own* centroids, which is circular and, worse, picked
               a scale roughly half the scene's (section 3.51). Fitting on camera
               poses instead is independent of the thing being measured.

Output is TUM format -- ``timestamp tx ty tz qx qy qz qw``, camera-to-world --
matching what the pipeline writes to traj_mcgs.txt so the two can be compared
directly.
"""

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_synthetic_erp_room import camera_pose  # noqa: E402


def quaternion_from_matrix(rotation):
    """``(qx, qy, qz, qw)`` from a rotation matrix, via the largest-trace branch.

    The branch matters: the naive w-first formula divides by a quantity that
    approaches zero for 180-degree rotations. These trajectories never rotate
    that far, but a helper that quietly breaks outside its tested range is how a
    diagnostic ends up lying later.
    """
    m, trace = rotation, float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        return np.array([(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s,
                         (m[1, 0] - m[0, 1]) / s, 0.25 * s])
    diagonal = np.diag(m)
    axis = int(np.argmax(diagonal))
    if axis == 0:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        return np.array([0.25 * s, (m[0, 1] + m[1, 0]) / s,
                         (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s])
    if axis == 1:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        return np.array([(m[0, 1] + m[1, 0]) / s, 0.25 * s,
                         (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s])
    s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
    return np.array([(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s,
                     0.25 * s, (m[1, 0] - m[0, 1]) / s])


def frame_count(sequence):
    """How many frames a generated sequence holds, from its own files."""
    frames = [f for f in os.listdir(sequence) if re.fullmatch(r"\d+\.png", f)]
    if not frames:
        raise SystemExit(f"no numbered .png frames in {sequence}")
    return max(int(Path(f).stem) for f in frames) + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence", required=True,
                        help="a generated ERP frame directory, e.g. "
                             "data/synth_erp_room_dynamic_fast_hires")
    parser.add_argument("--out", default=None,
                        help="defaults to <sequence>_gt_poses.txt")
    parser.add_argument("--nframes", type=int, default=None,
                        help="defaults to the sequence's own frame count")
    parser.add_argument("--timescale", type=float, default=1.0,
                        help="same --timescale the run used, so the stamps line "
                             "up with traj_mcgs.txt")
    args = parser.parse_args()

    count = args.nframes or frame_count(args.sequence)
    out = args.out or (args.sequence.rstrip("/") + "_gt_poses.txt")

    rows = []
    for t in range(count):
        position, rotation = camera_pose(t)
        qx, qy, qz, qw = quaternion_from_matrix(rotation)
        rows.append([t / args.timescale, *position, qx, qy, qz, qw])

    np.savetxt(out, np.asarray(rows), fmt="%.9f")
    span = np.linalg.norm(np.diff(np.asarray(rows)[:, 1:4], axis=0), axis=1).sum()
    print(f"wrote {count} camera-to-world poses -> {out}")
    print(f"   path length {span:.3f}, "
          f"start {rows[0][1:4]}, end {rows[-1][1:4]}")


if __name__ == "__main__":
    main()
