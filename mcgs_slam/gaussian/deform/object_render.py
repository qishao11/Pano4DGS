"""Place object-owned Gaussians at an arbitrary physical time for rendering.

This is the render-side half of the object-level SE(3) representation
(``panoramic_4dgs_status.md`` sections 3.27/3.28).  The oracle time-slice mode it
replaces keeps one independent bank of dynamic Gaussians per observed timestamp
and shows only the bank whose timestamp is being rendered: exact where it was
observed, and undefined everywhere else -- a lookup table, not a 4D
representation.

Here the same rows are instead *carried* by their object's trajectory:

  1. pick the observed time whose bank is used as the object's canonical copy
     (``nearest_observed_time`` -- the nearest one, so an observed time picks
     itself),
  2. show only that bank (every other dynamic row keeps opacity 0, exactly as
     before, so the object never appears in N overlapping copies),
  3. move it from that time to the requested time with the object's own relative
     motion ``T(target) . T(reference)^-1``.

Step 3 is the identity when reference == target, so at observed times this
reproduces the time-slice bank up to float error, while unobserved times -- where
the bank renders nothing at all -- now render the object at an interpolated pose.
That capability gap, not a PSNR delta, is what this path exists to demonstrate.
"""

import torch

from gaussian.deform.oracle_motion_gate import time_slice_opacities


def nearest_observed_time(observed_times, target_time):
    """Observed time whose dynamic bank should stand in for ``target_time``.

    Returns ``None`` when no dynamic rows have been registered yet (a static
    scene, or before the first dynamic keyframe), which callers treat as "render
    with no overrides at all".
    """
    if observed_times is None or len(observed_times) == 0:
        return None
    target = float(target_time)
    return min((float(t) for t in observed_times), key=lambda t: abs(t - target))


def object_se3_overrides(gaussians, trajectories, observed_times, target_time,
                         tolerance=1e-4):
    """``render()`` override kwargs placing the dynamic rows at ``target_time``.

    Returns an empty dict when there is nothing dynamic to place, so the caller
    can splat it into ``render(**overrides)`` unconditionally.
    """
    reference = nearest_observed_time(observed_times, target_time)
    if reference is None or not trajectories.object_ids:
        return {}

    opacities = time_slice_opacities(
        gaussians.get_opacity,
        gaussians.dynamic_score,
        gaussians.dynamic_source_time,
        reference,
        tolerance=tolerance,
    )

    xyz = gaussians.get_xyz
    object_ids = gaussians.dynamic_object_id.reshape(-1).to(device=xyz.device)
    source_times = gaussians.dynamic_source_time.reshape(-1).to(
        device=xyz.device, dtype=xyz.dtype)
    # Only the reference bank is visible, so only it is worth moving. Masking
    # the rest to -1 (static) here -- with the *same* predicate that gated their
    # opacity above -- keeps "which rows are shown" and "which rows are moved"
    # from ever disagreeing, and lets transform_to_time skip its device->host
    # scans via `groups`.
    from_reference = torch.isclose(
        source_times,
        torch.as_tensor(reference, device=xyz.device, dtype=xyz.dtype),
        atol=float(tolerance),
        rtol=0.0,
    )
    active_ids = torch.where(
        from_reference, object_ids, torch.full_like(object_ids, -1.0))
    groups = [
        (object_id, float(reference))
        for object_id in trajectories.object_ids
    ]

    moved_xyz, moved_rotations = trajectories.transform_to_time(
        xyz,
        active_ids,
        source_times,
        float(target_time),
        rotations=gaussians.get_rotation,
        groups=groups,
    )
    return {
        "means3D_override": moved_xyz,
        "rotations_override": moved_rotations,
        "opacities_override": opacities,
    }


def observed_times_from_trajectories(trajectories):
    """Rebuild the observed-time list from a restored trajectory table.

    Checkpoints store the trajectories but not this list; every knot was created
    by a dynamic keyframe, so the knots are the same set of times.  Kept out of
    the render path -- reading knot times syncs to host.
    """
    times = set()
    for object_id in trajectories.object_ids:
        trajectory = trajectories.trajectories[str(object_id)]
        times.update(float(t) for t in trajectory.knot_times.reshape(-1).tolist())
    return sorted(times)
