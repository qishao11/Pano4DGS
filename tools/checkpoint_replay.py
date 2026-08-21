"""Shared pieces for replaying a finished checkpoint offline.

The dynamic-representation tools all want the same three things: the Gaussian
tensors out of a ``4dgs_final.pt`` without dragging in CUDA-only extensions, the
ground-truth object trajectory, and a way to compare the two despite living in
different world frames.
"""

import json

import torch


class CheckpointGaussians:
    """The fields the render override paths read, served from a checkpoint.

    Loading the real GaussianModel would pull in CUDA-only extensions
    (simple_knn) and an optimizer none of these tools need.
    """

    def __init__(self, tensors):
        self._xyz = tensors["_xyz"]
        self._rotation = tensors["_rotation"]
        self._opacity = tensors["_opacity"]
        self.dynamic_score = tensors["dynamic_score"]
        self.dynamic_source_time = tensors["dynamic_source_time"]
        self.dynamic_object_id = tensors["dynamic_object_id"]
        # Present only once stage 1 landed; older checkpoints have neither.
        self._velocity = tensors.get("_velocity")
        self._time_scale_raw = tensors.get("_time_scale_raw")

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_opacity(self):
        # the checkpoint stores the pre-activation value
        return torch.sigmoid(self._opacity)

    @property
    def has_trained_4d(self):
        """Whether this checkpoint carries trained 4D parameters at all."""
        return self._velocity is not None and self._time_scale_raw is not None

    @property
    def get_velocity(self):
        return self._velocity

    @property
    def get_time_scale(self):
        return torch.nn.functional.softplus(self._time_scale_raw)

    @property
    def dynamic_mask(self):
        return self.dynamic_score.reshape(-1) > 0.5


def load_gaussians(path):
    """Return ``(CheckpointGaussians, saved_dynamic_mode)``."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint["gaussians"]
    if "dynamic_object_id" not in state["tensors"]:
        raise SystemExit(
            f"{path} predates per-Gaussian object IDs (section 3.28); "
            "it cannot be replayed through the dynamic paths")
    return CheckpointGaussians(state["tensors"]), checkpoint.get("dynamic_mode")


def observed_times(gaussians):
    """Distinct physical times that seeded dynamic Gaussians, sorted."""
    source = gaussians.dynamic_source_time.reshape(-1)
    return sorted(
        float(t) for t in torch.unique(source[gaussians.dynamic_mask]).tolist())


def load_ground_truth(path):
    """``{frame: [x, y, z]}`` from a synthetic sphere trajectory JSON."""
    return {
        float(entry["frame"]): [float(v) for v in entry["center_xyz"]]
        for entry in json.load(open(path))
    }


def ground_truth_centre(gt_entries, time):
    """Linearly interpolate the ground-truth object centre at ``time``."""
    times = sorted(gt_entries)
    if time <= times[0]:
        return torch.tensor(gt_entries[times[0]])
    if time >= times[-1]:
        return torch.tensor(gt_entries[times[-1]])
    for left, right in zip(times, times[1:]):
        if left <= time <= right:
            alpha = (time - left) / (right - left)
            return torch.lerp(torch.tensor(gt_entries[left]),
                              torch.tensor(gt_entries[right]), alpha)
    raise AssertionError("unreachable")


def umeyama(source, target):
    """Least-squares similarity transform (scale, R, t) taking source to target.

    Needed because the reconstruction has its own world frame and scale (first
    camera pose = origin); on the hires sphere sequence it is ~10x smaller than
    the generator's room, so raw coordinate differences mean nothing.
    """
    count = source.shape[0]
    mu_source, mu_target = source.mean(dim=0), target.mean(dim=0)
    centred_source = source - mu_source
    centred_target = target - mu_target
    covariance = centred_target.T @ centred_source / count
    u, singular_values, vt = torch.linalg.svd(covariance)
    correction = torch.eye(3, dtype=source.dtype)
    if torch.det(u) * torch.det(vt) < 0:
        correction[2, 2] = -1.0
    rotation = u @ correction @ vt
    variance = (centred_source ** 2).sum() / count
    scale = float((singular_values * correction.diagonal()).sum() / variance)
    translation = mu_target - scale * (rotation @ mu_source)
    return scale, rotation, translation


def apply_similarity(transform, points):
    scale, rotation, translation = transform
    return scale * (points @ rotation.T) + translation


def bank_statistics(gaussians, time):
    """Rows, centroid and widest axis span of the bank seeded at ``time``."""
    mask = torch.logical_and(
        gaussians.dynamic_mask,
        gaussians.dynamic_source_time.reshape(-1) == time)
    rows = gaussians.get_xyz[mask]
    extent = (rows.max(dim=0).values - rows.min(dim=0).values).max().item()
    return rows.shape[0], rows.mean(dim=0), extent
