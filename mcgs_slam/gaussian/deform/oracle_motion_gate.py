"""Oracle-only motion gating helpers for synthetic dynamic diagnostics.

This module is deliberately not a production motion segmenter.  It lets the
synthetic moving-sphere experiment answer a narrower question: can the current
canonical deformation field help when the dynamic/static assignment is known?
"""

import numpy as np
import torch
import torch.nn.functional as F


def color_motion_scores(colors, palette, threshold):
    """Return Nx1 hard motion scores for normalized colors.

    ``colors`` are expected in the same channel order as the configured palette.
    Palette values may be normalized to [0, 1] or supplied as [0, 255].
    """
    colors = np.asarray(colors, dtype=np.float32)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError("colors must have shape (N, 3)")
    if palette is None or len(palette) == 0:
        return np.zeros((len(colors), 1), dtype=np.float32)

    palette = np.asarray(palette, dtype=np.float32)
    if palette.ndim != 2 or palette.shape[1] != 3:
        raise ValueError("palette must have shape (K, 3)")
    if palette.size and float(palette.max()) > 1.0:
        palette = palette / 255.0

    min_distance = np.linalg.norm(
        colors[:, None, :] - palette[None, :, :], axis=-1).min(axis=1)
    return (min_distance <= float(threshold)).astype(np.float32)[:, None]


def apply_motion_gate(
    dxyz, drot_quat, dscale_log, motion_score, translation_only=False
):
    """Gate deformation and optionally restrict the oracle object to translation.

    ``translation_only`` is appropriate for the synthetic rigid sphere and blocks
    the scale-inflation shortcut exposed by ROI-weighted training.  It is not a
    claim that real dynamic scenes are translation-only.
    """
    gate = motion_score.to(device=dxyz.device, dtype=dxyz.dtype).reshape(-1, 1)
    if gate.shape[0] != dxyz.shape[0]:
        raise ValueError(
            f"motion gate has {gate.shape[0]} rows for {dxyz.shape[0]} Gaussians")

    gated_dxyz = dxyz * gate
    gated_dscale = dscale_log * gate
    identity = torch.zeros_like(drot_quat)
    identity[:, 0] = 1.0
    if translation_only:
        return gated_dxyz, identity, torch.zeros_like(dscale_log)
    gated_drot = F.normalize(identity + gate * (drot_quat - identity), dim=-1)
    return gated_dxyz, gated_drot, gated_dscale


def time_slice_opacities(
    opacities, motion_score, source_time, target_time, tolerance=1e-4
):
    """Hide dynamic Gaussians that do not belong to ``target_time``.

    Static Gaussians remain visible at every time.  This is an oracle diagnostic
    for testing explicit dynamic ownership; it is not a temporal interpolation
    model and therefore should only be evaluated at observed physical times.
    """
    gate = motion_score.to(device=opacities.device).reshape(-1) > 0.5
    source = source_time.to(
        device=opacities.device, dtype=opacities.dtype).reshape(-1)
    if gate.shape[0] != opacities.shape[0]:
        raise ValueError(
            f"motion gate has {gate.shape[0]} rows for {opacities.shape[0]} opacities")
    if source.shape[0] != opacities.shape[0]:
        raise ValueError(
            f"source time has {source.shape[0]} rows for {opacities.shape[0]} opacities")

    target = torch.as_tensor(
        target_time, device=opacities.device, dtype=opacities.dtype)
    active_dynamic = torch.isclose(
        source, target, atol=float(tolerance), rtol=0.0)
    active = torch.logical_or(~gate, active_dynamic)
    return opacities * active.to(opacities.dtype).reshape(-1, 1)


def oracle_color_mask(image, palette, threshold):
    """Return an HxW torch mask for a normalized CHW image.

    This is only for the synthetic diagnostic.  It deliberately uses the same
    known palette as Gaussian seeding, but remains independent of the rendered
    prediction so a wrong prediction cannot change its own supervision mask.
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("image must have shape (3, H, W)")
    if palette is None or len(palette) == 0:
        return torch.zeros(image.shape[1:], dtype=torch.bool, device=image.device)

    palette_tensor = torch.as_tensor(palette, dtype=image.dtype, device=image.device)
    if palette_tensor.ndim != 2 or palette_tensor.shape[1] != 3:
        raise ValueError("palette must have shape (K, 3)")
    if palette_tensor.numel() and palette_tensor.max() > 1.0:
        palette_tensor = palette_tensor / 255.0

    pixels = image.permute(1, 2, 0).unsqueeze(-2)
    distance = torch.linalg.vector_norm(
        pixels - palette_tensor.reshape(1, 1, -1, 3), dim=-1)
    return distance.min(dim=-1).values <= float(threshold)


def oracle_roi_l1(prediction, target, palette, threshold):
    """Mean RGB L1 inside the oracle dynamic ROI, or differentiable zero if absent."""
    mask = oracle_color_mask(target, palette, threshold)
    if not mask.any():
        return prediction.sum() * 0.0
    error = torch.abs(prediction - target).permute(1, 2, 0)
    return error[mask].mean()


def zero_masked_rows_(gradient, motion_score):
    """Zero optimizer gradients for oracle-dynamic tensor rows in place."""
    if gradient is None:
        return
    mask = motion_score.to(device=gradient.device).reshape(-1) > 0.5
    if mask.shape[0] != gradient.shape[0]:
        raise ValueError(
            f"motion mask has {mask.shape[0]} rows for gradient with {gradient.shape[0]} rows")
    gradient[mask] = 0


def backward_with_auxiliary_params(base_loss, auxiliary_loss, auxiliary_params):
    """Backpropagate ``base_loss`` normally and ``auxiliary_loss`` only to selected params.

    The oracle ROI loss must train the deformation field without giving the
    canonical Gaussian map a shortcut to repaint or inflate the target region.
    """
    params = tuple(param for param in auxiliary_params if param.requires_grad)
    if not params:
        base_loss.backward()
        return

    base_loss.backward(retain_graph=True)
    auxiliary_grads = torch.autograd.grad(
        auxiliary_loss, params, allow_unused=True)
    with torch.no_grad():
        for param, gradient in zip(params, auxiliary_grads):
            if gradient is None:
                continue
            if param.grad is None:
                param.grad = gradient.detach()
            else:
                param.grad.add_(gradient.detach())
