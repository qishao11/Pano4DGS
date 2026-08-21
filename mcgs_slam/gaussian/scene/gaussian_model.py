#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

import numpy as np
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from torch import nn

from gaussian.utils.general_utils import (
    build_rotation,
    build_scaling_rotation,
    helper,
    inverse_sigmoid,
    strip_symmetric,
)
from gaussian.utils.graphics_utils import BasicPointCloud, getWorld2View2
from gaussian.utils.sh_utils import RGB2SH
from gaussian.deform.oracle_motion_gate import color_motion_scores, time_slice_opacities


class GaussianModel:
    def __init__(self, sh_degree: int, config=None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree

        self._xyz = torch.empty(0, device="cuda")
        self._features_dc = torch.empty(0, device="cuda")
        self._features_rest = torch.empty(0, device="cuda")
        self._scaling = torch.empty(0, device="cuda")
        self._rotation = torch.empty(0, device="cuda")
        self._opacity = torch.empty(0, device="cuda")
        self.max_radii2D = torch.empty(0, device="cuda")
        self.xyz_gradient_accum = torch.empty(0, device="cuda")

        self.unique_kfIDs = torch.empty(0).int()
        self.n_obs = torch.empty(0).int()
        self.dynamic_score = torch.empty((0, 1), device="cuda")
        self.dynamic_source_time = torch.empty((0, 1), device="cuda")
        # -1 = static / unassigned. Dynamic rows carry the ID of the rigid
        # object they belong to, so an object-level SE(3) trajectory can move
        # them to times they were never observed at.
        self.dynamic_object_id = torch.empty((0, 1), device="cuda")
        # Native 4D Gaussians (gaussian/deform/gaussian_4d.py): each row gets an
        # extent in time and a drift, so rendering a moment is a slice of a 4D
        # primitive rather than a lookup into a per-timestamp bank.
        # `_time_scale_raw` is pre-softplus: the radius must stay positive, and
        # a raw value the optimizer can push through zero would leave the row
        # permanently stuck as a delta with no gradient (section 3.30 review).
        self._velocity = nn.Parameter(torch.empty((0, 3), device="cuda"))
        self._time_scale_raw = nn.Parameter(torch.empty((0, 1), device="cuda"))

        self.optimizer = None

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = self.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.config = config
        self.ply_input = None

        deform_cfg = (config or {}).get("Training", {}).get("deform_cfg", {})
        self.oracle_dynamic_colors = deform_cfg.get("oracle_dynamic_colors")
        self.oracle_color_threshold = deform_cfg.get("oracle_color_threshold", 0.02)
        self.oracle_dynamic_gate = bool(self.oracle_dynamic_colors)
        self.oracle_translation_only = bool(
            deform_cfg.get("oracle_translation_only", False))
        self.oracle_freeze_dynamic_scaling = bool(
            deform_cfg.get("oracle_freeze_dynamic_scaling", False))
        self.oracle_time_sliced_dynamic = bool(
            deform_cfg.get("oracle_time_sliced_dynamic", False))
        self.oracle_time_tolerance = float(
            deform_cfg.get("oracle_time_tolerance", 1e-4))
        self.oracle_dynamic_downsample = deform_cfg.get(
            "oracle_dynamic_downsample")
        # Seed dynamic pixels from the depth prior rather than BA's disparity,
        # whose multi-view constraint is invalid for a moving object. See
        # _depth_for_dynamic_pixels and section 3.53.
        self.dynamic_depth_from_prior = bool(
            deform_cfg.get("dynamic_depth_from_prior", False))
        # Hold dynamic rows at their seeded position through refinement, where a
        # single supervising view cannot constrain depth along its own ray. See
        # gs_backend._freeze_dynamic_positions and section 3.54.
        self.freeze_dynamic_positions = bool(
            deform_cfg.get("freeze_dynamic_positions", False))
        # Minimum initial size of a seeded Gaussian, in projected pixels.
        # 0 disables it, which is the historical behaviour. See section 3.46.
        self.min_projected_pixels = float(
            (config or {}).get("Dataset", {}).get("min_projected_pixels", 0.0))
        # Temporal radius a freshly seeded dynamic row starts with. Half the
        # median gap between observed times, so neighbouring banks just overlap.
        self.time_scale_init = float(deform_cfg.get("time_scale_init", 0.5))
        if self.time_scale_init <= 0:
            raise ValueError("time_scale_init must be positive")
        if (
            self.oracle_dynamic_downsample is not None
            and float(self.oracle_dynamic_downsample) < 1.0
        ):
            raise ValueError("oracle_dynamic_downsample must be at least 1")

        self.isotropic = False

    def build_covariance_from_scaling_rotation(
        self, scaling, scaling_modifier, rotation
    ):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_velocity(self):
        return self._velocity

    @property
    def get_time_scale(self):
        """Temporal radius, kept positive by construction (see ``_time_scale_raw``)."""
        return torch.nn.functional.softplus(self._time_scale_raw)

    @staticmethod
    def inverse_softplus(value):
        """Raw value whose softplus is ``value``; for initializing time scales."""
        tensor = torch.as_tensor(value, dtype=torch.float32)
        return torch.log(torch.expm1(tensor))

    def get_opacity_at_time(self, target_time):
        """Return an optional oracle visibility override for a physical time."""
        if not self.oracle_time_sliced_dynamic:
            return None
        return time_slice_opacities(
            self.get_opacity,
            self.dynamic_score,
            self.dynamic_source_time,
            target_time,
            tolerance=self.oracle_time_tolerance,
        )

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def checkpoint_state(self):
        """Capture all Gaussian and temporal tensors needed for exact replay."""
        tensor_names = (
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_scaling",
            "_rotation",
            "_opacity",
            "max_radii2D",
            "xyz_gradient_accum",
            "denom",
            "unique_kfIDs",
            "n_obs",
            "dynamic_score",
            "dynamic_source_time",
            "dynamic_object_id",
            "_velocity",
            "_time_scale_raw",
        )
        return {
            "active_sh_degree": int(self.active_sh_degree),
            "max_sh_degree": int(self.max_sh_degree),
            "spatial_lr_scale": float(getattr(self, "spatial_lr_scale", 1.0)),
            "tensors": {
                name: getattr(self, name).detach().cpu().clone()
                for name in tensor_names
            },
        }

    def restore_checkpoint_state(self, state, training_args=None, device="cuda"):
        """Restore a :meth:`checkpoint_state` payload and optionally resume training."""
        tensors = state.get("tensors", {})
        required = (
            "_xyz",
            "_features_dc",
            "_features_rest",
            "_scaling",
            "_rotation",
            "_opacity",
            "unique_kfIDs",
            "n_obs",
            "dynamic_score",
            "dynamic_source_time",
        )
        missing = [name for name in required if name not in tensors]
        if missing:
            raise ValueError(f"Gaussian checkpoint is missing tensors: {missing}")

        point_count = int(tensors["_xyz"].shape[0])
        mismatched = {
            name: int(tensors[name].shape[0])
            for name in required
            if int(tensors[name].shape[0]) != point_count
        }
        # Optional (older payloads predate it) but must still line up when present,
        # otherwise a silently short object-ID column would mis-assign every row.
        if "dynamic_object_id" in tensors:
            object_id_rows = int(tensors["dynamic_object_id"].shape[0])
            if object_id_rows != point_count:
                mismatched["dynamic_object_id"] = object_id_rows
        for name in ("_velocity", "_time_scale_raw"):
            if name in tensors and int(tensors[name].shape[0]) != point_count:
                mismatched[name] = int(tensors[name].shape[0])
        if mismatched:
            raise ValueError(
                f"Gaussian checkpoint point-count mismatch: expected {point_count}, "
                f"got {mismatched}")

        parameter_names = (
            "_xyz", "_features_dc", "_features_rest", "_scaling",
            "_rotation", "_opacity",
        )
        for name in parameter_names:
            value = tensors[name].to(device=device).contiguous()
            setattr(self, name, nn.Parameter(value.requires_grad_(True)))

        self.unique_kfIDs = tensors["unique_kfIDs"].cpu().int().clone()
        self.n_obs = tensors["n_obs"].cpu().int().clone()
        self.dynamic_score = tensors["dynamic_score"].to(device=device).clone()
        self.dynamic_source_time = tensors["dynamic_source_time"].to(
            device=device).clone()
        # Older checkpoints predate object IDs; default them to static so the
        # payload stays loadable instead of failing the required-tensor check.
        if "dynamic_object_id" in tensors:
            self.dynamic_object_id = tensors["dynamic_object_id"].to(
                device=device).clone()
        else:
            # Derive from dynamic_score exactly as create_pcd_from_image does.
            # Defaulting every row to -1 would label the restored map's dynamic
            # Gaussians as static, so a resumed run would behave differently
            # from an identical run that never checkpointed -- silently, since
            # the tensor is present and correctly shaped.
            self.dynamic_object_id = torch.where(
                self.dynamic_score > 0.5,
                torch.zeros_like(self.dynamic_score),
                torch.full_like(self.dynamic_score, -1.0))
        # Checkpoints predating the 4D parameters restore to "no motion, default
        # radius", which is the same state a fresh seed starts from -- not a
        # silent behaviour change, because a run that never had them behaved
        # exactly this way.
        if "_velocity" in tensors:
            velocity = tensors["_velocity"].to(device=device).clone()
        else:
            velocity = torch.zeros(
                (point_count, 3), device=device, dtype=self._xyz.dtype)
        if "_time_scale_raw" in tensors:
            time_scale_raw = tensors["_time_scale_raw"].to(device=device).clone()
        else:
            time_scale_raw = torch.full(
                (point_count, 1), float(self.inverse_softplus(self.time_scale_init)),
                device=device, dtype=self._xyz.dtype)
        # Parameters, and assigned before training_setup below builds the
        # optimizer over them -- otherwise the optimizer would hold the old
        # empty tensors (the detached-parameter failure from section 3.27).
        self._velocity = nn.Parameter(velocity.requires_grad_(True))
        self._time_scale_raw = nn.Parameter(time_scale_raw.requires_grad_(True))
        self.active_sh_degree = int(state.get("active_sh_degree", 0))
        self.max_sh_degree = int(state.get("max_sh_degree", self.max_sh_degree))
        self.spatial_lr_scale = float(state.get("spatial_lr_scale", 1.0))

        if training_args is not None:
            self.training_setup(training_args)
        else:
            self.optimizer = None

        optional_defaults = {
            "max_radii2D": (point_count,),
            "xyz_gradient_accum": (point_count, 1),
            "denom": (point_count, 1),
        }
        for name, shape in optional_defaults.items():
            value = tensors.get(name)
            if value is None or int(value.shape[0]) != point_count:
                value = torch.zeros(shape, dtype=self._xyz.dtype)
            setattr(self, name, value.to(device=device).clone())

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_pcd_from_image(self, cam_info, init=False, scale=2.0, depthmap=None,
                              prior_depthmap=None):
        cam = cam_info
        rgb_raw = (cam.original_image * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
        depthmap = self._depth_for_dynamic_pixels(rgb_raw, depthmap, prior_depthmap)
        rgb = o3d.geometry.Image(rgb_raw.astype(np.uint8))
        depth = o3d.geometry.Image(depthmap.astype(np.float32))

        return self.create_pcd_from_image_and_depth(cam, rgb, depth, init)

    def _depth_for_dynamic_pixels(self, rgb_raw, depthmap, prior_depthmap):
        """Seed the moving object from the depth prior instead of BA's disparity.

        A dynamic pixel's multi-view constraint is invalid -- the object moved
        between the frames being matched -- yet bundle adjustment updates its
        disparity from that constraint anyway, overwriting the ground-truth
        depth. Section 3.52 measured the result: 86.5% of dynamic Gaussians land
        inside the object's silhouette and only 21.7% at the right distance
        along the ray, so the renders look right while the geometry is a
        diameter off. Back-projecting the same pixels with the prior instead puts
        100.0% of them on the sphere, at every keyframe (section 3.53).

        Only dynamic pixels are touched. Static geometry is exactly what BA is
        good at, and section 3.52 measured it accurate to 0.02 m.

        The prior is metric while the map is not (the reconstruction runs ~4.7x
        smaller), so the two are reconciled per frame by the ratio the *static*
        pixels agree on -- the places where BA and the prior should say the same
        thing. Hardcoding a factor would silently break the moment the scale
        convention changed.
        """
        if prior_depthmap is None or not self.oracle_dynamic_colors:
            return depthmap
        if not bool(getattr(self, "dynamic_depth_from_prior", False)):
            return depthmap

        scores = color_motion_scores(
            rgb_raw.reshape(-1, 3).astype(np.float32) / 255.0,
            self.oracle_dynamic_colors,
            self.oracle_color_threshold,
        ).reshape(rgb_raw.shape[:2])
        dynamic = scores > 0.5
        usable = np.logical_and(prior_depthmap > 0, depthmap > 0)
        calibration = np.logical_and(usable, ~dynamic)
        if not dynamic.any() or calibration.sum() < 100:
            return depthmap

        ratio = float(np.median(depthmap[calibration] / prior_depthmap[calibration]))
        if not np.isfinite(ratio) or ratio <= 0:
            return depthmap

        adjusted = depthmap.copy()
        swap = np.logical_and(dynamic, usable)
        adjusted[swap] = prior_depthmap[swap] * ratio
        return adjusted

    def create_pcd_from_image_and_depth(self, cam, rgb, depth, init=False):
        if init:
            downsample_factor = self.config["Dataset"]["pcd_downsample_init"]
        else:
            downsample_factor = self.config["Dataset"]["pcd_downsample"]
        point_size = self.config["Dataset"]["point_size"]
        if "adaptive_pointsize" in self.config["Dataset"]:
            if self.config["Dataset"]["adaptive_pointsize"]:
                point_size = min(0.05, point_size * np.median(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1.0,
            depth_trunc=100.0,
            convert_rgb_to_intensity=False,
        )

        W2C = getWorld2View2(cam.R, cam.T).cpu().numpy()
        pcd_tmp = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd,
            o3d.camera.PinholeCameraIntrinsic(
                cam.image_width,
                cam.image_height,
                cam.fx,
                cam.fy,
                cam.cx,
                cam.cy,
            ),
            extrinsic=W2C,
            project_valid_depth_only=True,
        )
        # A per-time dynamic bank cannot borrow density from the same object at
        # other timestamps. Uniform 1/64 sampling left only tens of points when
        # the synthetic sphere was small, so preserve a denser oracle-dynamic
        # subset while retaining the original static-map sampling budget.
        if self.oracle_dynamic_downsample is not None:
            full_rgb = np.asarray(pcd_tmp.colors)
            full_dynamic_score = color_motion_scores(
                full_rgb,
                self.oracle_dynamic_colors,
                self.oracle_color_threshold,
            ).reshape(-1)
            dynamic_indices = np.flatnonzero(full_dynamic_score > 0.5).tolist()
            if dynamic_indices:
                static_pcd = pcd_tmp.select_by_index(
                    dynamic_indices, invert=True).random_down_sample(
                        1.0 / downsample_factor)
                dynamic_pcd = pcd_tmp.select_by_index(
                    dynamic_indices).random_down_sample(
                        1.0 / float(self.oracle_dynamic_downsample))
                pcd_tmp = static_pcd + dynamic_pcd
            else:
                pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        else:
            pcd_tmp = pcd_tmp.random_down_sample(1.0 / downsample_factor)
        new_xyz = np.asarray(pcd_tmp.points)
        new_rgb = np.asarray(pcd_tmp.colors)
        new_dynamic_score = color_motion_scores(
            new_rgb, self.oracle_dynamic_colors, self.oracle_color_threshold)

        pcd = BasicPointCloud(
            points=new_xyz, colors=new_rgb, normals=np.zeros((new_xyz.shape[0], 3))
        )
        self.ply_input = pcd

        fused_point_cloud = torch.from_numpy(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.from_numpy(np.asarray(pcd.colors)).float().cuda())
        features = (
            torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2))
            .float()
            .cuda()
        )
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = (
            torch.clamp_min(
                distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()),
                0.0000001,
            )
            * point_size
        )
        scales = torch.log(torch.sqrt(dist2))[..., None]
        if self.min_projected_pixels > 0:
            # Floor the initial scale so every seeded Gaussian projects to at
            # least this many pixels. Section 3.46: seeded Gaussians start
            # sub-pixel (0.3-0.4 px here). The horizontal cubemap faces grow ~6x
            # during training and recover, but the pole faces grow only 1.6x,
            # 42-57% of their Gaussians never train at all, and they render pure
            # black. A Gaussian too small to cover a pixel contributes almost
            # nothing to the photometric loss and so receives almost no gradient
            # to grow with; this breaks that loop at initialisation instead.
            camera_centre = np.linalg.inv(W2C)[:3, 3]
            distance = np.linalg.norm(
                np.asarray(pcd.points) - camera_centre[None], axis=1)
            floor = (self.min_projected_pixels
                     * np.clip(distance, 1e-6, None) / float(cam.fx))
            floor = torch.from_numpy(floor).float().cuda()[..., None]
            scales = torch.maximum(scales, torch.log(floor))
        if not self.isotropic:
            scales = scales.repeat(1, 3)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        dynamic_score = torch.from_numpy(new_dynamic_score).float().cuda()
        physical_tstamp = getattr(
            cam, "physical_tstamp",
            cam.tstamp - 500 * getattr(cam, "cam_idx", 0))
        if torch.is_tensor(physical_tstamp):
            physical_tstamp = physical_tstamp.item()
        dynamic_source_time = torch.full_like(dynamic_score, -1.0)
        dynamic_source_time[dynamic_score > 0.5] = float(physical_tstamp)
        # The oracle motion gate only answers "is this row dynamic", not which
        # object it belongs to, and the synthetic scenes contain exactly one
        # moving body -- so every dynamic row is assigned object 0. Real
        # multi-object data will need per-instance IDs from a segmenter here.
        dynamic_object_id = torch.full_like(dynamic_score, -1.0)
        dynamic_object_id[dynamic_score > 0.5] = 0.0
        # A fresh row has no observed motion yet: zero drift, default radius.
        velocity = torch.zeros(
            (fused_point_cloud.shape[0], 3), device=fused_point_cloud.device,
            dtype=fused_point_cloud.dtype)
        time_scale_raw = torch.full_like(
            dynamic_score, float(self.inverse_softplus(self.time_scale_init)))
        return (
            fused_point_cloud, features, scales, rots, opacities,
            dynamic_score, dynamic_source_time, dynamic_object_id,
            velocity, time_scale_raw,
        )

    def init_lr(self, spatial_lr_scale):
        self.spatial_lr_scale = spatial_lr_scale

    def extend_from_pcd(
        self, fused_point_cloud, features, scales, rots, opacities,
        dynamic_score, dynamic_source_time, dynamic_object_id, kf_id,
        velocity=None, time_scale_raw=None,
    ):
        new_xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        new_features_dc = nn.Parameter(
            features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True)
        )
        new_scaling = nn.Parameter(scales.requires_grad_(True))
        new_rotation = nn.Parameter(rots.requires_grad_(True))
        new_opacity = nn.Parameter(opacities.requires_grad_(True))

        new_unique_kfIDs = torch.ones((new_xyz.shape[0])).int() * kf_id
        new_n_obs = torch.zeros((new_xyz.shape[0])).int()
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_unique_kfIDs,
            new_n_obs=new_n_obs,
            new_dynamic_score=dynamic_score,
            new_dynamic_source_time=dynamic_source_time,
            new_dynamic_object_id=dynamic_object_id,
            new_velocity=velocity,
            new_time_scale_raw=time_scale_raw,
        )

    def extend_from_pcd_seq(
        self, cam_info, kf_id=-1, init=False, scale=2.0, depthmap=None,
        prior_depthmap=None
    ):
        (
            fused_point_cloud, features, scales, rots, opacities,
            dynamic_score, dynamic_source_time, dynamic_object_id,
            velocity, time_scale_raw,
        ) = (
            self.create_pcd_from_image(cam_info, init, scale=scale, depthmap=depthmap,
                                       prior_depthmap=prior_depthmap)
        )
        self.extend_from_pcd(
            fused_point_cloud, features, scales, rots, opacities,
            dynamic_score, dynamic_source_time, dynamic_object_id, kf_id,
            velocity=velocity, time_scale_raw=time_scale_raw,
        )

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {
                "params": [self._features_dc],
                "lr": training_args.feature_lr,
                "name": "f_dc",
            },
            {
                "params": [self._features_rest],
                "lr": training_args.feature_lr / 20.0,
                "name": "f_rest",
            },
            {
                "params": [self._opacity],
                "lr": training_args.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr * self.spatial_lr_scale,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
            # 4D parameters. Same magnitude as position: velocity is world units
            # per unit time, and one time unit is one frame here. No decay
            # schedule -- one scheduled quantity (xyz) is enough to reason about.
            {
                "params": [self._velocity],
                "lr": getattr(training_args, "velocity_lr", 0.001),
                "name": "velocity",
            },
            {
                "params": [self._time_scale_raw],
                "lr": getattr(training_args, "time_scale_lr", 0.001),
                "name": "time_scale",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.lr_init = training_args.position_lr_init * self.spatial_lr_scale
        self.lr_final = training_args.position_lr_final * self.spatial_lr_scale
        self.max_steps = training_args.position_lr_max_steps

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = helper(
                    iteration,
                    lr_init=self.lr_init,
                    lr_final=self.lr_final,
                    max_steps=self.max_steps+1000,
                )

                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = (
            self._features_dc.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        f_rest = (
            self._features_rest.detach()
            .transpose(1, 2)
            .flatten(start_dim=1)
            .contiguous()
            .cpu()
            .numpy()
        )
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.01)
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def reset_opacity_nonvisible(
        self, visibility_filters
    ):  ##Reset opacity for only non-visible gaussians
        opacities_new = inverse_sigmoid(torch.ones_like(self.get_opacity) * 0.4)

        for filter in visibility_filters:
            opacities_new[filter] = self.get_opacity[filter]
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._velocity = optimizable_tensors["velocity"]
        self._time_scale_raw = optimizable_tensors["time_scale"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.unique_kfIDs = self.unique_kfIDs[valid_points_mask.cpu()]
        self.n_obs = self.n_obs[valid_points_mask.cpu()]
        self.dynamic_score = self.dynamic_score[valid_points_mask]
        self.dynamic_source_time = self.dynamic_source_time[valid_points_mask]
        self.dynamic_object_id = self.dynamic_object_id[valid_points_mask]
        self._velocity = optimizable_tensors["velocity"]
        self._time_scale_raw = optimizable_tensors["time_scale"]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_features_dc,
        new_features_rest,
        new_opacities,
        new_scaling,
        new_rotation,
        new_kf_ids=None,
        new_n_obs=None,
        new_dynamic_score=None,
        new_dynamic_source_time=None,
        new_dynamic_object_id=None,
        new_velocity=None,
        new_time_scale_raw=None,
    ):
        if new_velocity is None:
            new_velocity = torch.zeros(
                (new_xyz.shape[0], 3), device=new_xyz.device, dtype=new_xyz.dtype)
        if new_time_scale_raw is None:
            new_time_scale_raw = torch.full(
                (new_xyz.shape[0], 1),
                float(self.inverse_softplus(self.time_scale_init)),
                device=new_xyz.device, dtype=new_xyz.dtype)
        d = {
            "xyz": new_xyz,
            "f_dc": new_features_dc,
            "f_rest": new_features_rest,
            "opacity": new_opacities,
            "scaling": new_scaling,
            "rotation": new_rotation,
            "velocity": new_velocity,
            "time_scale": new_time_scale_raw,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._velocity = optimizable_tensors["velocity"]
        self._time_scale_raw = optimizable_tensors["time_scale"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        if new_kf_ids is not None:
            self.unique_kfIDs = torch.cat((self.unique_kfIDs, new_kf_ids)).int()
        if new_n_obs is not None:
            self.n_obs = torch.cat((self.n_obs, new_n_obs)).int()
        if new_dynamic_score is None:
            new_dynamic_score = torch.zeros(
                (new_xyz.shape[0], 1), device=new_xyz.device, dtype=new_xyz.dtype)
        self.dynamic_score = torch.cat((self.dynamic_score, new_dynamic_score), dim=0)
        if new_dynamic_source_time is None:
            new_dynamic_source_time = torch.full(
                (new_xyz.shape[0], 1), -1.0,
                device=new_xyz.device, dtype=new_xyz.dtype)
        self.dynamic_source_time = torch.cat(
            (self.dynamic_source_time, new_dynamic_source_time), dim=0)
        if new_dynamic_object_id is None:
            new_dynamic_object_id = torch.full(
                (new_xyz.shape[0], 1), -1.0,
                device=new_xyz.device, dtype=new_xyz.dtype)
        self.dynamic_object_id = torch.cat(
            (self.dynamic_object_id, new_dynamic_object_id), dim=0)


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[: grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            > self.percent_dense * scene_extent,
        )

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[
            selected_pts_mask
        ].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N)
        )
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()].repeat(N)
        new_n_obs = self.n_obs[selected_pts_mask.cpu()].repeat(N)
        new_dynamic_score = self.dynamic_score[selected_pts_mask].repeat(N, 1)
        new_dynamic_source_time = self.dynamic_source_time[
            selected_pts_mask].repeat(N, 1)
        new_dynamic_object_id = self.dynamic_object_id[
            selected_pts_mask].repeat(N, 1)
        # a split child belongs to the same object, so it moves the same way
        new_velocity = self._velocity[selected_pts_mask].repeat(N, 1)
        new_time_scale_raw = self._time_scale_raw[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
            new_dynamic_score=new_dynamic_score,
            new_dynamic_source_time=new_dynamic_source_time,
            new_dynamic_object_id=new_dynamic_object_id,
            new_velocity=new_velocity,
            new_time_scale_raw=new_time_scale_raw,
        )

        prune_filter = torch.cat(
            (
                selected_pts_mask,
                torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool),
            )
        )

        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(
            torch.norm(grads, dim=-1) >= grad_threshold, True, False
        )
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values
            <= self.percent_dense * scene_extent,
        )

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_kf_id = self.unique_kfIDs[selected_pts_mask.cpu()]
        new_n_obs = self.n_obs[selected_pts_mask.cpu()]
        new_dynamic_score = self.dynamic_score[selected_pts_mask]
        new_dynamic_source_time = self.dynamic_source_time[selected_pts_mask]
        new_dynamic_object_id = self.dynamic_object_id[selected_pts_mask]
        new_velocity = self._velocity[selected_pts_mask]
        new_time_scale_raw = self._time_scale_raw[selected_pts_mask]
        self.densification_postfix(
            new_xyz,
            new_features_dc,
            new_features_rest,
            new_opacities,
            new_scaling,
            new_rotation,
            new_kf_ids=new_kf_id,
            new_n_obs=new_n_obs,
            new_dynamic_score=new_dynamic_score,
            new_dynamic_source_time=new_dynamic_source_time,
            new_dynamic_object_id=new_dynamic_object_id,
        )

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = torch.logical_and((self.get_opacity < min_opacity).squeeze(), (self.unique_kfIDs != self.unique_kfIDs.max()).cuda())
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent

            prune_mask = torch.logical_or(
                torch.logical_or(prune_mask, big_points_vs), big_points_ws
            )
        self.prune_points(prune_mask)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True
        )
        self.denom[update_filter] += 1
