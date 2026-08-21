"""Estimate object motion from geometry instead of learning it photometrically.

Why geometry. Eight attempts across three parameterizations and two training
designs (panoramic_4dgs_status.md sections 3.5-3.9, 3.32-3.35) all failed the
same way, and section 3.35 pinned the mechanism: when every observed time owns
its own Gaussians and the loss is photometric at that time only, a row carried
from a neighbouring time is either redundant (and gets pushed away, learning a
*reversed* velocity) or absent (and the hole is cheaper to fill some other way).
The optimizer is never *required* to place it correctly.

Meanwhile the best-scoring method in that whole comparison, v=est at 0.734
against v=0's 0.799 and every trained variant's 0.842+, was not learned at all --
it differenced consecutive bank centroids. This module improves on that estimate
while keeping it out of the photometric loop entirely.

A centroid reduces a bank of thousands of points to three numbers, and carries a
bias: each view seeds only the surface facing it, so the centroid sits toward the
camera and that offset moves as the camera does. Registering the two point clouds
uses the whole geometry instead.

Only translation is estimated. For the synthetic sphere rotation is not
observable at all, and velocity is the only quantity the 4D slice consumes --
fitting a rotation would let it absorb translation error into a meaningless
rotation.
"""

import torch


def translation_icp(source, target, iterations=20, trim=0.2, initial=None):
    """Translation-only ICP: the shift carrying ``source`` onto ``target``.

    ``trim`` discards that fraction of the worst-matching correspondences each
    iteration.  The two banks see different surface patches of the same object --
    the camera moved between them -- so a portion of each cloud has no true
    counterpart, and a plain mean over all correspondences would be dragged by
    exactly those points.

    ``initial`` seeds the search, normally the centroid difference: ICP only
    finds the nearest local optimum, and for a smooth object like a sphere the
    basin is not wide.
    """
    if source.numel() == 0 or target.numel() == 0:
        return torch.zeros(3, dtype=torch.float32)
    shift = (torch.zeros(3, dtype=source.dtype, device=source.device)
             if initial is None else initial.clone().to(source))
    keep = max(1, int(round(source.shape[0] * (1.0 - trim))))
    for _ in range(iterations):
        distances = torch.cdist(source + shift, target)
        closest, index = distances.min(dim=1)
        # trimmed: keep the best-matching correspondences only
        order = torch.argsort(closest)[:keep]
        delta = (target[index[order]] - (source[order] + shift)).mean(dim=0)
        shift = shift + delta
        if float(delta.norm()) < 1e-6:
            break
    return shift


def bank_rows(xyz, object_ids, source_times, object_id, time):
    mask = torch.logical_and(object_ids.reshape(-1) == float(object_id),
                             source_times.reshape(-1) == float(time))
    return xyz[mask], mask


def estimate_bank_velocities(xyz, object_ids, source_times, times,
                             max_points=2000, refine=True, **icp_kwargs):
    """Velocity per (object, observed time), from registering adjacent banks.

    Returns ``{(object_id, time): velocity}``.  Each bank is matched against both
    neighbours where they exist and the two estimates averaged -- a central
    difference, which is what the centroid-based estimator does too, so the
    comparison isolates *how* the shift is measured rather than which times it
    is measured between.

    ``refine`` chooses what measures the shift:

      True   trimmed translation-only ICP over the two point clouds, seeded from
             their centroid difference. Introduced in section 3.36 because bank
             centroids were being wrecked by outlier Gaussians (6 of 12 banks,
             section 3.29) under monocular depth, where it scored decisively
             better -- direction cosine 0.797 against 0.244.
      False  the centroid difference alone, i.e. the ICP's own seed without the
             refinement. Section 3.47: under ground-truth depth every bank is
             clean (13/13), the contamination ICP was introduced to work around
             is gone, and the plain centroid now scores *better* in all six runs
             measured -- 0.884 against 0.906 on four faces, 0.962 against 0.975
             on six. It is also far cheaper: no cdist, no iteration.

    Neither is right unconditionally, which is why this is a parameter and not a
    replacement: the choice tracks whether the banks are clean, and that tracks
    the depth source.
    """
    object_ids = object_ids.reshape(-1)
    source_times = source_times.reshape(-1)
    estimates = {}
    for object_id in sorted({int(v) for v in object_ids.tolist() if v >= 0}):
        clouds, centroids = {}, {}
        for time in times:
            rows, _ = bank_rows(xyz, object_ids, source_times, object_id, time)
            # the centroid is taken over the whole bank; only the ICP needs a
            # subsample, and seeding it from a subsampled centroid is what it has
            # always done, so that path is left bit-for-bit unchanged
            centroids[time] = rows.mean(dim=0) if rows.shape[0] else None
            if rows.shape[0] > max_points:
                # cdist is O(n*m), so subsample -- but deterministically. A
                # randperm here made the production velocity estimate differ
                # between runs on the very same checkpoint, which silently put
                # noise into every number measured from it. Striding keeps the
                # points spatially spread, since storage order follows seeding
                # order rather than position.
                stride = rows.shape[0] // max_points
                rows = rows[::stride][:max_points]
            clouds[time] = rows
        for index, time in enumerate(times):
            source = clouds[time]
            if source.shape[0] == 0:
                continue
            velocities = []
            for other in (index - 1, index + 1):
                if not 0 <= other < len(times):
                    continue
                target_time = times[other]
                target = clouds[target_time]
                if target.shape[0] == 0:
                    continue
                span = target_time - time
                if not refine:
                    velocities.append(
                        (centroids[target_time] - centroids[time]) / span)
                    continue
                initial = target.mean(dim=0) - source.mean(dim=0)
                shift = translation_icp(source, target, initial=initial,
                                        **icp_kwargs)
                velocities.append(shift / span)
            if velocities:
                estimates[(object_id, time)] = torch.stack(velocities).mean(dim=0)
    return estimates


def velocities_to_rows(estimates, object_ids, source_times, like):
    """Broadcast per-bank velocities onto the per-Gaussian layout."""
    velocity = torch.zeros_like(like)
    object_ids = object_ids.reshape(-1)
    source_times = source_times.reshape(-1)
    for (object_id, time), value in estimates.items():
        mask = torch.logical_and(object_ids == float(object_id),
                                 source_times == float(time))
        velocity[mask] = value.to(velocity)
    return velocity
