"""Does object_se3 placement hold up on a real trained map, not just unit tests?

``eval_trajectory_interpolation.py`` fitted a trajectory to *ground-truth* sphere
centres and measured how well it interpolates.  This tool asks the two questions
that one could not:

  1. does the trajectory built the way the pipeline actually builds it -- from
     the centroids of the dynamic Gaussians a real 30000-step run produced --
     still place the object correctly, or does the surface-facing bias in those
     centroids wreck it?
  2. at an observed time, does carrying the rows reproduce the per-timestamp
     bank, so no previously recorded time-slice result moves?

It reads a finished checkpoint (``4dgs_final.pt``), so it needs no training and
no GPU time.  Checkpoints written in ``oracle_time_slice`` mode work fine: the
rows, their object IDs and their source times are identical -- only what happens
to them at render time differs, which is exactly what is being compared here.

Rendered images are not produced; the comparison is over Gaussian positions and
visibility, which is where the capability gap lives.

On comparing against ground truth: the reconstruction lives in its own world
frame (first camera pose = origin, and its own scale), while the synthetic
ground truth lives in the generator's room frame.  On the hires sphere sequence
the two differ by roughly 13x in scale, so raw coordinate differences are
meaningless.  Positions are therefore compared through a least-squares
similarity fit (Umeyama) between the estimated knots and the ground-truth
centres, and it is the *residual after* that fit -- plus the recovered scale --
that says whether the trajectory tracks the object.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))

from gaussian.deform.object_render import (  # noqa: E402
    nearest_observed_time,
    object_se3_overrides,
)
from gaussian.deform.object_trajectory import ObjectTrajectoryTable  # noqa: E402
from gaussian.deform.oracle_motion_gate import time_slice_opacities  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checkpoint_replay import (  # noqa: E402
    apply_similarity,
    ground_truth_centre,
    load_gaussians,
    load_ground_truth,
    umeyama,
)


def dynamic_rows_visible(gaussians, opacities):
    """Dynamic Gaussians the renderer would actually see."""
    dynamic = gaussians.dynamic_score.reshape(-1) > 0.5
    return int(torch.logical_and(dynamic, opacities.reshape(-1) > 0).sum().item())


def dynamic_centroid(xyz, gaussians, opacities):
    dynamic = gaussians.dynamic_score.reshape(-1) > 0.5
    shown = torch.logical_and(dynamic, opacities.reshape(-1) > 0)
    if not bool(shown.any()):
        return None
    return xyz[shown].mean(dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="a 4dgs_final.pt produced by any dynamic mode")
    parser.add_argument("--gt", default=None,
                        help="optional sphere trajectory JSON, to score the "
                             "placement against the true object position")
    parser.add_argument("--tolerance", type=float, default=1e-4,
                        help="source-time match tolerance, as in the config")
    parser.add_argument("--max-extent", type=float, default=1.0,
                        help="a bank wider than this is treated as outlier-"
                             "contaminated and left out of the ground-truth fit")
    args = parser.parse_args()

    gaussians, saved_mode = load_gaussians(args.checkpoint)
    trajectories = ObjectTrajectoryTable()
    observed = trajectories.observe_centroids(
        gaussians.get_xyz, gaussians.dynamic_object_id,
        gaussians.dynamic_source_time)
    if not observed:
        raise SystemExit(
            f"{args.checkpoint} contains no dynamic Gaussians to place")

    rows = gaussians.get_xyz.shape[0]
    dynamic_rows = int((gaussians.dynamic_score > 0.5).sum().item())
    print(f"checkpoint      : {args.checkpoint}")
    print(f"saved mode      : {saved_mode}")
    print(f"Gaussians       : {rows} ({dynamic_rows} dynamic, "
          f"{100.0 * dynamic_rows / max(1, rows):.2f}%)")
    print(f"objects         : {trajectories.object_ids}")
    print(f"observed times  : {len(observed)} in "
          f"[{observed[0]:g}, {observed[-1]:g}]\n")

    gt_entries = load_ground_truth(args.gt) if args.gt else None

    print("0. what each observed time contributes")
    print("   (extent = widest axis span of that bank; a clean sphere bank "
          "should be about one\n    sphere across. median is reported next to "
          "the mean because the mean is what the\n    trajectory is currently "
          "built from, and outliers move it.)")
    print(f"   {'time':>6}{'rows':>7}{'mean':>26}{'median':>26}"
          f"{'extent':>9}{'p10-p90':>9}")
    object_ids = gaussians.dynamic_object_id.reshape(-1)
    source_times = gaussians.dynamic_source_time.reshape(-1)
    knots, clean_times = {}, []
    for time in observed:
        mask = torch.logical_and(object_ids >= 0, source_times == time)
        rows_here = gaussians.get_xyz[mask]
        centroid = rows_here.mean(dim=0)
        median = rows_here.median(dim=0).values
        extent = (rows_here.max(dim=0).values
                  - rows_here.min(dim=0).values).max().item()
        robust_extent = (
            torch.quantile(rows_here, 0.9, dim=0)
            - torch.quantile(rows_here, 0.1, dim=0)).max().item()
        knots[time] = centroid
        clean = extent <= args.max_extent
        if clean:
            clean_times.append(time)
        line = (f"   {time:>6g}{rows_here.shape[0]:>7}"
                f"   {centroid[0]:>7.2f}{centroid[1]:>8.2f}{centroid[2]:>8.2f}"
                f"   {median[0]:>7.2f}{median[1]:>8.2f}{median[2]:>8.2f}"
                f"{extent:>9.2f}{robust_extent:>9.2f}"
                f"{'' if clean else '   contaminated'}")
        print(line)
    print(f"   {len(clean_times)}/{len(observed)} banks are within "
          f"--max-extent {args.max_extent:g}\n")

    print("1. observed times must reproduce the per-timestamp bank")
    worst_shift = 0.0
    worst_time = None
    for time in observed:
        overrides = object_se3_overrides(
            gaussians, trajectories, observed, time, tolerance=args.tolerance)
        shift = (overrides["means3D_override"]
                 - gaussians.get_xyz).abs().max().item()
        bank = time_slice_opacities(
            gaussians.get_opacity, gaussians.dynamic_score,
            gaussians.dynamic_source_time, time, tolerance=args.tolerance)
        if not torch.equal(bank, overrides["opacities_override"]):
            raise SystemExit(f"visibility differs from the bank at t={time:g}")
        if shift > worst_shift:
            worst_shift, worst_time = shift, time
    print(f"   visibility identical at all {len(observed)} observed times")
    where = "" if worst_time is None else f" (at t={worst_time:g})"
    print(f"   largest position shift: {worst_shift:.3e} world units{where}"
          f" -- float error in T(t).T(t)^-1, not motion\n")

    transform = None
    if gt_entries:
        print("2. do the estimated knots track the real object?")
        if len(clean_times) < 3:
            print("   too few uncontaminated banks to fit a similarity "
                  "transform; skipping\n")
        else:
            source = torch.stack([knots[t] for t in clean_times])
            target = torch.stack(
                [ground_truth_centre(gt_entries, t) for t in clean_times])
            transform = umeyama(source, target)
            residuals = torch.linalg.vector_norm(
                apply_similarity(transform, source) - target, dim=1)
            span = float(torch.linalg.vector_norm(
                target.max(dim=0).values - target.min(dim=0).values))
            print(f"   fitted on the {len(clean_times)} uncontaminated banks: "
                  f"scale {transform[0]:.3f} "
                  f"(the reconstruction is ~{transform[0]:.0f}x smaller than "
                  f"the generator's room)")
            print(f"   residual after alignment: mean "
                  f"{residuals.mean().item():.3f}, max "
                  f"{residuals.max().item():.3f} GT units, against a "
                  f"{span:.2f}-unit object path")
            print("   (a residual far below the path length means the knots "
                  "follow the object; the\n    similarity fit only removes the "
                  "frame difference, it cannot manufacture motion)\n")

    print("3. midpoints between observed times: what each path renders")
    header = (f"   {'time':>9}{'bank shows':>12}{'object_se3 shows':>18}"
              f"{'moved by':>10}{'knots l/ref/r':>16}")
    if transform is not None:
        header += f"{'GT error':>10}"
    print(header)
    errors, clean_errors = [], []
    for left, right in zip(observed, observed[1:]):
        midpoint = 0.5 * (left + right)
        overrides = object_se3_overrides(
            gaussians, trajectories, observed, midpoint,
            tolerance=args.tolerance)
        bank = time_slice_opacities(
            gaussians.get_opacity, gaussians.dynamic_score,
            gaussians.dynamic_source_time, midpoint, tolerance=args.tolerance)
        bank_visible = dynamic_rows_visible(gaussians, bank)
        se3_visible = dynamic_rows_visible(
            gaussians, overrides["opacities_override"])
        centre = dynamic_centroid(
            overrides["means3D_override"], gaussians,
            overrides["opacities_override"])
        reference_centre = dynamic_centroid(
            gaussians.get_xyz, gaussians, overrides["opacities_override"])
        moved = torch.linalg.vector_norm(centre - reference_centre).item()
        # the pose comes from interpolating the two knots bracketing this time,
        # so a bad knot on either side ruins it -- being carried *from* a clean
        # bank is not enough
        reference = nearest_observed_time(observed, midpoint)
        knots_clean = left in clean_times and right in clean_times
        label = f"{left:g}/{reference:g}/{right:g}" + ("" if knots_clean else "*")
        line = (f"   {midpoint:>9.3f}{bank_visible:>12}{se3_visible:>18}"
                f"{moved:>10.3f}{label:>16}")
        if transform is not None:
            error = torch.linalg.vector_norm(
                apply_similarity(transform, centre[None])[0]
                - ground_truth_centre(gt_entries, midpoint)).item()
            errors.append(error)
            if knots_clean:
                clean_errors.append(error)
            line += f"{error:>10.3f}"
        print(line)

    if errors:
        print("\n   * = a bracketing knot is contaminated, so that row inherits "
              "its bad centroid;\n     the error there is a bank-quality "
              "problem, not an interpolation problem.")
        if clean_errors:
            print(f"   GT error over the {len(clean_errors)} rows bracketed by "
                  f"clean knots: mean "
                  f"{sum(clean_errors) / len(clean_errors):.3f}"
                  f", max {max(clean_errors):.3f} GT units")
        print(f"   GT error over all {len(errors)} rows: "
              f"mean {sum(errors) / len(errors):.3f}, max {max(errors):.3f}")


if __name__ == "__main__":
    main()
