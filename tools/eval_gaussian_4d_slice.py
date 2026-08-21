"""Does slicing native 4D Gaussians fix what the earlier representations could not?

Replays a finished checkpoint (no training, no GPU) and compares what is visible
at times between observations under four settings:

  bank      the per-timestamp oracle: a dynamic row exists at exactly one time
            (panoramic_4dgs_status.md 3.19). Between observations: nothing.
  v=0       every dynamic row gets a temporal radius but no drift. The object
            appears between observations -- but as the union of the neighbouring
            banks sitting where they were, i.e. ghosts rather than one object.
  v=est     drift estimated offline from where the neighbouring bank centroids
            are. This is the SE(3) route's information in a different shape, and
            it inherits the same problem: 6 of 12 bank centroids are wrecked by
            outliers (section 3.29).
  v=gt      drift taken from the ground-truth object motion, mapped into the
            reconstruction frame. Not a method -- an upper bound that separates
            "can the representation do this" from "can we estimate the velocity".
  v=icp     drift from registering adjacent banks' point clouds instead of
            differencing their centroids. Geometry only -- never passes through
            the photometric loss, so the failure mode of section 3.35 cannot
            apply. This is the one that should beat v=est.
  v=trained the velocity the checkpoint actually learned, shown only when the
            run was trained in gaussian_4d mode. This is the one that has to beat
            v=0 for stage 1 to have worked; v=gt bounds how much is available.

The headline number is `spread`: a robust (p10-p90) axis span of the rows that
are actually visible. One object should span about one object; ghosts span the
distance the object travelled between the two observations. A centroid cannot
tell those apart -- two symmetric ghosts have a centroid in exactly the right
place -- which is why spread is reported alongside the GT error.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaussian.deform.gaussian_4d import slice_at_time
from gaussian.deform.motion_estimation import (
    estimate_bank_velocities,
    velocities_to_rows,
)  # noqa: E402
from gaussian.deform.oracle_motion_gate import time_slice_opacities  # noqa: E402
from checkpoint_replay import (  # noqa: E402
    apply_similarity,
    bank_statistics,
    ground_truth_centre,
    load_gaussians,
    load_ground_truth,
    observed_times,
    umeyama,
)


# below this an opacity contributes nothing a viewer could see; also the cutoff
# that keeps --stride's suppressed banks from being counted
VISIBLE_OPACITY = 1e-6


def rows_of_bank(gaussians, time):
    return torch.logical_and(
        gaussians.dynamic_mask,
        gaussians.dynamic_source_time.reshape(-1) == time)


def estimated_velocity(gaussians, times, centroids):
    """Per-row drift from where the neighbouring bank centroids sit.

    Central difference where both neighbours exist. Stand-in for the parameter
    the real system should train; it inherits whatever is wrong with the bank
    centroids, which is the point of comparing it against v=gt.
    """
    velocity = torch.zeros_like(gaussians.get_xyz)
    for index, time in enumerate(times):
        slopes = [
            (centroids[times[other]] - centroids[time]) / (times[other] - time)
            for other in (index - 1, index + 1)
            if 0 <= other < len(times)
        ]
        if slopes:
            velocity[rows_of_bank(gaussians, time)] = torch.stack(slopes).mean(0)
    return velocity


def ground_truth_velocity(gaussians, times, gt_entries, transform):
    """True object velocity expressed in the reconstruction's frame.

    ``apply_similarity`` maps reconstruction -> ground truth as
    ``y = s*R*x + t``, so a direction maps back as ``R^T v / s``.
    """
    scale, rotation, _ = transform
    velocity = torch.zeros_like(gaussians.get_xyz)
    step = 0.5
    for time in times:
        forward = ground_truth_centre(gt_entries, time + step)
        backward = ground_truth_centre(gt_entries, time - step)
        v_gt = (forward - backward) / (2.0 * step)
        velocity[rows_of_bank(gaussians, time)] = (rotation.T @ v_gt) / scale
    return velocity


def visible_spread(xyz, weights, dynamic, threshold=1e-3):
    """Robust axis span and presence-weighted centre of the visible dynamic rows.

    p10-p90 rather than max-min: 6 of 12 banks carry outlier Gaussians reaching
    100 world units (section 3.29), and max-min would report those instead of
    the object this measurement is about.
    """
    shown = torch.logical_and(dynamic, weights.reshape(-1) > threshold)
    if int(shown.sum()) < 2:
        return None, None, int(shown.sum())
    rows = xyz[shown]
    spread = (torch.quantile(rows, 0.9, dim=0)
              - torch.quantile(rows, 0.1, dim=0)).max().item()
    w = weights.reshape(-1)[shown][:, None]
    return spread, (rows * w).sum(dim=0) / w.sum(), int(shown.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--gt", default=None)
    parser.add_argument("--time-scale", type=float, default=None,
                        help="temporal radius for every dynamic row; default is "
                             "half the median gap between observed times")
    parser.add_argument("--stride", type=int, default=1,
                        help="keep every Nth observed time, treat the rest as "
                             "unobserved. The sphere moves ~0.02 between "
                             "consecutive frames while being 0.23 across, so at "
                             "stride 1 neighbouring banks overlap and the "
                             "velocity term has nothing to do; widening the gap "
                             "is what makes it testable")
    parser.add_argument("--max-extent", type=float, default=None,
                        help="banks wider than this are outlier-contaminated "
                             "(section 3.29) and excluded from the GT fit. "
                             "Default adapts to the reconstruction's own scale: "
                             "an absolute threshold silently misfires when the "
                             "scale changes -- switching from monocular to "
                             "ground-truth depth grows it 3.35x, which alone "
                             "dropped usable rows from 12 to 2 (section 3.40)")
    args = parser.parse_args()

    gaussians, saved_mode = load_gaussians(args.checkpoint)
    if args.max_extent is None:
        # 2x the static map's own 2-98% extent: reproduces the historical 1.0 on
        # monocular-depth runs (extent 0.530) and scales with the reconstruction
        static = gaussians.get_xyz[~gaussians.dynamic_mask]
        scene = float((static.quantile(0.98, dim=0)
                       - static.quantile(0.02, dim=0)).max())
        args.max_extent = 2.0 * scene
    times = observed_times(gaussians)
    if args.stride > 1:
        kept = times[::args.stride]
        if kept[-1] != times[-1]:
            kept.append(times[-1])
        # rows of dropped banks must not render, or they appear as extra ghosts
        # that no representation asked for
        keep = torch.zeros_like(gaussians.dynamic_score.reshape(-1),
                                dtype=torch.bool)
        for time in kept:
            keep = torch.logical_or(keep, rows_of_bank(gaussians, time))
        gaussians._opacity = gaussians._opacity.clone()
        gaussians._opacity[torch.logical_and(gaussians.dynamic_mask, ~keep)] = -30.0
        times = kept
    if len(times) < 2:
        raise SystemExit("need at least two observed times")

    gaps = [b - a for a, b in zip(times, times[1:])]
    median_gap = float(sorted(gaps)[len(gaps) // 2])
    time_scale_value = args.time_scale if args.time_scale is not None \
        else 0.5 * median_gap

    centroids, clean = {}, []
    for time in times:
        _, centroid, extent = bank_statistics(gaussians, time)
        centroids[time] = centroid
        if extent <= args.max_extent:
            clean.append(time)

    dynamic = gaussians.dynamic_mask
    rows = gaussians.get_xyz.shape[0]
    time_center = gaussians.dynamic_source_time.reshape(-1)
    time_scale = torch.full((rows,), float(time_scale_value))

    print(f"checkpoint     : {args.checkpoint}")
    print(f"saved mode     : {saved_mode}")
    print(f"Gaussians      : {rows} ({int(dynamic.sum())} dynamic)")
    print(f"observed times : {len(times)} in [{times[0]:g}, {times[-1]:g}], "
          f"{len(clean)} uncontaminated"
          + (f" (stride {args.stride})" if args.stride > 1 else ""))
    print(f"time scale     : {time_scale_value:g} "
          f"(median gap between observations is {median_gap:g})")

    transform = None
    gt_entries = load_ground_truth(args.gt) if args.gt else None
    if gt_entries and len(clean) >= 3:
        transform = umeyama(
            torch.stack([centroids[t] for t in clean]),
            torch.stack([ground_truth_centre(gt_entries, t) for t in clean]))
        print(f"GT alignment   : scale {transform[0]:.3f} on {len(clean)} "
              f"clean banks")

    object_size = min(
        (bank_statistics(gaussians, t)[2] for t in clean), default=float("nan"))
    print(f"object size    : {object_size:.3f} world units "
          f"(narrowest clean bank)\n")

    settings = [("v=0", torch.zeros((rows, 3)))]
    settings.append(("v=est", estimated_velocity(gaussians, times, centroids)))
    settings.append(("v=icp", velocities_to_rows(
        estimate_bank_velocities(
            gaussians.get_xyz, gaussians.dynamic_object_id,
            gaussians.dynamic_source_time, times),
        gaussians.dynamic_object_id, gaussians.dynamic_source_time,
        gaussians.get_xyz)))
    if transform is not None:
        settings.append(
            ("v=gt", ground_truth_velocity(gaussians, times, gt_entries,
                                           transform)))
    # The trained parameters, if this checkpoint has any. Its own time_scale is
    # used too -- comparing a trained velocity under a different radius than it
    # was trained with would not measure what the run learned.
    trained_time_scale = None
    if gaussians.has_trained_4d and float(gaussians.get_velocity.abs().max()) > 0:
        settings.append(("v=trained", gaussians.get_velocity))
        trained_time_scale = gaussians.get_time_scale.reshape(-1)

    header = f"   {'time':>7}{'bank':>6}{'shown':>7}"
    for name, _ in settings:
        header += f"{name + ' spread':>14}"
        if transform is not None:
            header += f"{name + ' err':>12}"
    print(header)

    summary = {name: {"spread": [], "err": []} for name, _ in settings}
    travelled_all = []
    for left, right in zip(times, times[1:]):
        midpoint = 0.5 * (left + right)
        both_clean = left in clean and right in clean
        bank = time_slice_opacities(
            gaussians.get_opacity, gaussians.dynamic_score,
            gaussians.dynamic_source_time, midpoint)
        # not `> 0`: --stride drops banks by pushing opacity to sigmoid(-30),
        # which is tiny but strictly positive, and would be counted as visible
        bank_rows = int(torch.logical_and(
            dynamic, bank.reshape(-1) > VISIBLE_OPACITY).sum().item())

        line = f"   {midpoint:>7.2f}{bank_rows:>6}"
        shown_reported = False
        for name, velocity in settings:
            row_time_scale = (trained_time_scale
                              if name == "v=trained" and trained_time_scale is not None
                              else time_scale)
            moved, _, weights = slice_at_time(
                gaussians.get_xyz, gaussians.get_opacity, time_center,
                row_time_scale, velocity, midpoint, gaussians.dynamic_score)
            spread, centre, shown = visible_spread(moved, weights, dynamic)
            if not shown_reported:
                line += f"{shown:>7}"
                shown_reported = True
            line += "           n/a" if spread is None else f"{spread:>14.3f}"
            if spread is not None and both_clean:
                summary[name]["spread"].append(spread)
            if transform is None:
                continue
            if centre is None:
                line += "         n/a"
                continue
            error = torch.linalg.vector_norm(
                apply_similarity(transform, centre[None])[0]
                - ground_truth_centre(gt_entries, midpoint)).item()
            line += f"{error:>12.3f}"
            if both_clean:
                summary[name]["err"].append(error)
        if both_clean:
            travelled_all.append(torch.linalg.vector_norm(
                centroids[right] - centroids[left]).item())
        print(line + ("" if both_clean else "  *"))

    print("\n   * = at least one bracketing bank is outlier-contaminated "
          "(section 3.29); those rows say\n     more about bank quality than "
          "about the representation. Summary below uses clean rows only.")
    if travelled_all:
        print(f"\n   clean rows: {len(travelled_all)}, object size "
              f"{object_size:.3f}, object travelled "
              f"{sum(travelled_all) / len(travelled_all):.3f} between "
              f"observations on average")
        for name, _ in settings:
            spreads = summary[name]["spread"]
            errs = summary[name]["err"]
            if not spreads:
                continue
            text = f"   {name:<6} spread {sum(spreads) / len(spreads):>8.3f}"
            if errs:
                text += f"   GT err {sum(errs) / len(errs):>8.3f}"
            print(text)


if __name__ == "__main__":
    main()
