"""Native 4D Gaussians: conditioning a spacetime primitive on a query time.

Three dynamic representations came before this one (see
``panoramic_4dgs_status.md``):

  - a canonical map plus a global ``(xyz,t) -> delta`` MLP (§3.19: disproven --
    it learned to repaint canonical Gaussians rather than move them),
  - the oracle per-timestamp bank (§3.19/3.20: works, but is a lookup table --
    a dynamic row is visible at exactly one timestamp and nowhere else),
  - object-level SE(3) trajectories (§3.27-3.29: interpolates, but needs an
    object segmentation, a centroid per bank, and it turned out 6 of 12 banks
    have centroids wrecked by outlier Gaussians).

This module takes the fourth route, the one the 4D-Gaussian literature converged
on (4D-Rotor GS; "native 4D primitives", arXiv:2412.20720): keep no canonical
space and no deformation field, and instead give each Gaussian an extent in time
as well as space.  Rendering a moment means *slicing* that 4D primitive at
``t``.

For a Gaussian with spacetime mean ``(mu_x, t_c)`` and covariance blocks
``Sigma_xx, Sigma_xt, Sigma_tt``, conditioning on time ``t`` gives

    weight(t) = exp(-(t - t_c)^2 / (2 * Sigma_tt))          -- how present it is
    mu_x(t)   = mu_x + Sigma_xt / Sigma_tt * (t - t_c)      -- where it is

so the conditional mean moves *linearly* in time and the primitive fades out
away from its own moment.  Rather than storing a full 4D covariance, this module
uses the two equivalent quantities directly: ``time_scale`` (= sqrt(Sigma_tt),
the primitive's temporal radius) and ``velocity`` (= Sigma_xt / Sigma_tt, its
drift per unit time).  They are exactly the numbers the slice needs, they are
individually interpretable, and each stays a plain per-Gaussian parameter.

Why this is the natural successor to the per-timestamp bank: with
``time_scale -> 0`` the weight becomes a delta and this reduces to the bank
exactly.  Widening it is what buys continuity, and unlike the bank the
in-between times are defined; unlike the object-SE(3) route it needs no object
IDs, no centroids and no trajectory table, so the bank-contamination problem in
§3.29 cannot reach it.

It also restores a training signal the SE(3) route did not have.  There, a row
was only ever rendered at its own observed time, where the transform is the
identity, so the trajectory knots received exactly zero photometric gradient.
Here a row with a finite temporal radius contributes to *neighbouring* times as
well, so rendering an observed frame produces gradients for the velocity and
time scale of the rows around it.

Static rows (motion score 0) are untouched: they are present at all times.
"""

import torch


def temporal_weights(time_center, time_scale, target_time, motion_score,
                     per_observation=False):
    """How present each Gaussian is at ``target_time``.

    Static rows -- ``motion_score <= 0.5`` -- get weight 1 at every time.
    ``motion_score`` is deliberately required rather than defaulting to "all
    dynamic": with the project's convention that static rows carry
    ``source_time = -1``, omitting it silently fades the entire static map out
    at any time far from -1, which renders as a black frame rather than as an
    error.

    ``time_scale`` is the primitive's temporal radius (sqrt of the temporal
    variance) and must be non-negative.  Exactly 0 is allowed and means "this
    moment only" -- the delta the per-timestamp bank implements.  Negative
    values are rejected instead of being clamped: clamping would leave the row
    permanently stuck as a delta with zero gradient, unable to recover, which is
    the worst kind of silent failure.  A trainable time scale must therefore be
    parameterized to stay non-negative (softplus/exp), not stored raw.

    ``per_observation`` fixes a missing normalizer.  A bank is one *observation*
    of the object -- one instant's worth of seeded surface -- and how many
    Gaussians it happens to hold is an artefact of seeding and pruning, not a
    statement about the object.  Weighting each row independently therefore lets
    a bank's influence scale with its row count: banks in a finished map range
    from 211 rows to 2241, a factor of ten, so the same kernel value of 0.135
    means something ten times larger for one bank than another.  Dividing each
    row by its own bank's mass and renormalizing across banks makes the temporal
    conditional weight observations rather than primitives, which is what the
    slice is supposed to mean.  Total dynamic mass is then the same at every
    query time, and each bank's share depends only on its distance in time.
    """
    time_center = time_center.reshape(-1)
    time_scale = time_scale.reshape(-1).to(dtype=time_center.dtype)
    if time_scale.shape != time_center.shape:
        raise ValueError("time_scale and time_center must have equal rows")
    if bool((time_scale < 0).any()):
        raise ValueError(
            "time_scale must be non-negative; parameterize it with softplus or "
            "exp rather than letting the optimizer push it through zero")

    target = torch.as_tensor(
        target_time, device=time_center.device, dtype=time_center.dtype)
    # a zero radius would divide by zero, so evaluate the delta case separately
    positive = time_scale > 0
    safe_scale = torch.where(positive, time_scale, torch.ones_like(time_scale))
    offset = (target - time_center) / safe_scale
    weights = torch.where(
        positive,
        torch.exp(-0.5 * offset * offset),
        (time_center == target).to(time_center.dtype))

    dynamic = motion_score.reshape(-1).to(device=weights.device) > 0.5
    if per_observation and bool(dynamic.any()):
        weights = _normalize_per_observation(weights, time_center, dynamic)
    return torch.where(dynamic, weights, torch.ones_like(weights))


def _normalize_per_observation(weights, time_center, dynamic):
    """Rescale dynamic weights so a bank's influence is independent of its size.

    Each bank keeps a single share of the frame, ``w_b / sum_b' w_b'``, and that
    share is spread over however many rows the bank has.  The mean bank size
    puts the result back on the same scale as the unnormalized weights, so
    opacities stay in their usual range instead of collapsing as banks grow.
    """
    groups = time_center[dynamic]
    unique_groups, inverse = torch.unique(groups, return_inverse=True)
    counts = torch.zeros(unique_groups.shape[0], device=weights.device,
                         dtype=weights.dtype)
    counts = counts.index_add(
        0, inverse, torch.ones_like(inverse, dtype=weights.dtype))
    # one kernel value per bank: every row of a bank shares a time centre, and
    # they share a time scale too unless it has been trained per row, in which
    # case the bank's mean is the honest summary
    totals = torch.zeros_like(counts).index_add(0, inverse, weights[dynamic])
    per_bank = totals / counts.clamp_min(1.0)
    share = per_bank / per_bank.sum().clamp_min(torch.finfo(weights.dtype).tiny)
    scaled = share * counts.mean() / counts.clamp_min(1.0)

    normalized = weights.clone()
    normalized[dynamic] = scaled[inverse]
    return normalized


def pool_velocity(velocity, time_center, motion_score):
    """Share one velocity across each bank, instead of one per Gaussian.

    Motivation (section 3.34): every failure so far has the same shape -- the
    optimizer finds a cheaper explanation than the object's real motion. A
    per-Gaussian velocity offers ~33000 degrees of freedom to a motion that
    physically has 3 per instant, and that surplus *is* the cheat: individual
    Gaussians drift wherever the residual wants them. Averaging within a bank
    removes the surplus without changing the storage layout or any propagation
    point, and it is differentiable, so the gradient reaching the shared value
    is the sum over the bank -- noise cancels, signal accumulates.

    Static rows keep their own (unused, identically zero) velocity.
    """
    dynamic = motion_score.reshape(-1).to(device=velocity.device) > 0.5
    if not bool(dynamic.any()):
        return velocity
    groups = time_center.reshape(-1)[dynamic]
    unique_groups, inverse = torch.unique(groups, return_inverse=True)
    totals = torch.zeros(
        (unique_groups.shape[0], velocity.shape[1]),
        device=velocity.device, dtype=velocity.dtype)
    totals = totals.index_add(0, inverse, velocity[dynamic])
    counts = torch.zeros(
        unique_groups.shape[0], device=velocity.device, dtype=velocity.dtype)
    counts = counts.index_add(
        0, inverse, torch.ones_like(inverse, dtype=velocity.dtype))
    means = totals / counts[:, None].clamp_min(1.0)
    pooled = velocity.clone()
    pooled[dynamic] = means[inverse]
    return pooled


def widen_for_interpolation(time_scale, widened):
    """Temporal radius to use when rendering a time nobody observed.

    The radius has two jobs and section 3.49/3.49.1 showed they pull opposite
    ways.  During training only observed times are rendered, and there a radius
    wide enough to reach the neighbouring bank makes the frame's own rows a
    *redundant* explanation of it: the photometric loss dims them (median opacity
    fell 0.722 -> 0.137 -> 0.063 across consecutive banks at 0.5, against a flat
    1.000 at 0.25) until pruning finishes them off, leaving a bank too faint to
    draw the object.  Between observations the opposite is needed -- a midpoint
    is half a step from both banks and gets nothing unless the radius reaches it.

    One number cannot do both, so interpolation gets its own.  Training and
    observed-time rendering keep the narrow radius that stops the redundancy;
    only queries at unobserved times widen, and by then the banks are intact and
    fully opaque, so there is real mass to interpolate between.

    ``widened`` of ``None`` keeps the stored radius, which is the historical
    behaviour and what every observed-time render uses.
    """
    if widened is None:
        return time_scale
    widened = float(widened)
    if widened <= 0:
        raise ValueError("interpolation time scale must be positive")
    return torch.full_like(time_scale.reshape(-1), widened)


def slice_at_time(xyz, opacities, time_center, time_scale, velocity,
                  target_time, motion_score, exclude_own_bank=False,
                  per_observation=False):
    """Condition the 4D primitives on ``target_time``.

    Returns ``(moved_xyz, faded_opacities, weights)``.  The positional drift is
    applied to dynamic rows only -- a static row has no meaningful time centre to
    measure ``t - t_c`` from.  See ``temporal_weights`` for why ``motion_score``
    is required.

    ``per_observation`` makes the temporal conditional weight banks rather than
    individual Gaussians; see ``temporal_weights``.

    ``exclude_own_bank`` hides the dynamic rows whose own time *is* the rendered
    time, and exists to fix a specific training failure (section 3.32): with
    them visible, they carry weight 1 and zero elapsed time, so they alone
    explain the frame and every row arriving from a neighbouring time is
    redundant.  A redundant degree of freedom gets spent fitting residual noise,
    which is what made the learned velocity land at cosine 0.05 against the true
    motion -- right magnitude, random direction.  Hiding them leaves the object
    renderable only by carrying a neighbouring bank into place, so the
    photometric loss can only be reduced by getting the velocity right.

    Train with it on, evaluate with it off: at evaluation the map should use
    everything it has.
    """
    rows = xyz.shape[0]
    for name, tensor in (("opacities", opacities), ("time_center", time_center),
                         ("time_scale", time_scale), ("velocity", velocity),
                         ("motion_score", motion_score)):
        if tensor.shape[0] != rows:
            raise ValueError(
                f"{name} has {tensor.shape[0]} rows for {rows} Gaussians")

    weights = temporal_weights(time_center, time_scale, target_time,
                               motion_score, per_observation=per_observation)

    target = torch.as_tensor(target_time, device=xyz.device, dtype=xyz.dtype)
    elapsed = target - time_center.reshape(-1).to(dtype=xyz.dtype)
    dynamic = motion_score.reshape(-1).to(device=xyz.device) > 0.5
    if exclude_own_bank:
        # tolerance-free: source times are written from the same float that is
        # rendered back, so equality is exact for a row's own bank
        own = torch.logical_and(dynamic, elapsed == 0)
        weights = torch.where(own, torch.zeros_like(weights), weights)
    elapsed = torch.where(dynamic, elapsed, torch.zeros_like(elapsed))
    moved_xyz = xyz + velocity * elapsed[:, None]

    return moved_xyz, opacities * weights[:, None], weights


def visible_rows(weights, threshold=1e-3):
    """Rows a renderer would actually contribute, for diagnostics."""
    return int((weights.reshape(-1) > threshold).sum().item())
