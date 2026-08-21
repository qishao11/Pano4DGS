import copy
import os
import random
import time
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import trange
from munch import munchify
from lietorch import SE3, SO3

from utils.utils import Log, clone_obj
from gaussian.renderer import render
from gaussian.utils.loss_utils import l1_loss, ssim
from gaussian.scene.gaussian_model import GaussianModel
from gaussian.utils.graphics_utils import getProjectionMatrix2
from gaussian.utils.slam_utils import update_pose, to_se3_vec, get_loss_normal, get_loss_mapping_rgbd
from gaussian.utils.camera_utils import Camera
from gaussian.utils.eval_utils import eval_rendering, eval_rendering_kf
from gaussian.gui import gui_utils, slam_gui
from gaussian.deform.deform_net import DeformNet, apply_deform
from gaussian.deform.dynamic_modes import (
    DYNAMIC_MODE_DEFORM,
    DYNAMIC_MODE_GAUSSIAN_4D,
    DYNAMIC_MODE_OBJECT_SE3,
    resolve_dynamic_mode,
)
from gaussian.deform.gaussian_4d import (
    pool_velocity,
    slice_at_time,
    widen_for_interpolation,
)
from gaussian.deform.motion_estimation import (
    estimate_bank_velocities,
    velocities_to_rows,
)
from gaussian.deform.object_trajectory import ObjectTrajectoryTable
from gaussian.deform.object_render import (
    object_se3_overrides,
    observed_times_from_trajectories,
)
from gaussian.deform.oracle_motion_gate import (
    apply_motion_gate,
    backward_with_auxiliary_params,
    oracle_roi_l1,
    zero_masked_rows_,
)
from physical_view_window import PhysicalViewWindow
import cubemap

class GSBackEnd(mp.Process):
    CHECKPOINT_FORMAT = "mcgs_slam_temporal_gaussians"
    CHECKPOINT_VERSION = 1

    def __init__(self, config, save_dir, use_gui=False):
        super().__init__()
        self.config = config
        self.dynamic_mode = resolve_dynamic_mode(self.config["Training"])
        self.config["Training"]["dynamic_mode"] = self.dynamic_mode
        
        self.iteration_count = 0
        self.viewpoints = {}
        self.current_window = []
        self.initialized = False
        self.save_dir = save_dir
        self.use_gui = use_gui

        self.opt_params = munchify(config["opt_params"])
        self.config["Training"]["monocular"] = False

        self.gaussians = GaussianModel(sh_degree=0, config=self.config)
        self.gaussians.init_lr(6.0)
        self.gaussians.training_setup(self.opt_params)
        self.background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        self.time_metadata = {
            "origin": None,
            "scale": 1.0,
            "unit": "legacy_frame_index",
            "value_semantics": "legacy frame index",
        }

        # 4DGS: canonical Gaussians (above) + optional time-conditioned deformation field.
        # See panoramic_support_feasibility.md section 4 / mcgs_slam/gaussian/deform/deform_net.py.
        self.deform_net = None
        self.deform_optimizer = None
        self.object_trajectories = ObjectTrajectoryTable().cuda()
        # Physical times that seeded dynamic Gaussians, i.e. the times whose
        # dynamic bank exists. object_se3 rendering picks the nearest one as the
        # canonical copy to carry; kept python-side so the render path never
        # syncs the device to read it back.
        self.dynamic_observed_times = []
        # Leave-one-out training for gaussian_4d. Default OFF: it was tried in
        # section 3.33 and made things worse -- hiding the frame's own bank
        # creates a hole where the object should be, and the cheapest way to
        # fill that hole is not learning the velocity. Kept because the negative
        # result is worth being able to reproduce.
        self.gaussian_4d_leave_one_out = bool(
            self.config["Training"].get("deform_cfg", {}).get(
                "leave_one_out", False))
        # Share one velocity per bank rather than one per Gaussian (3.34).
        self.gaussian_4d_pool_velocity = bool(
            self.config["Training"].get("deform_cfg", {}).get(
                "pool_velocity", True))
        # Weight the temporal conditional per observation rather than per
        # Gaussian, so a bank's influence does not scale with how many rows
        # seeding and pruning happened to leave it (3.49). Disproven in 3.49.1;
        # kept switchable so the negative result stays reproducible.
        self.gaussian_4d_per_observation = bool(
            self.config["Training"].get("deform_cfg", {}).get(
                "per_observation_weights", False))
        # Temporal radius used only at times nobody observed. None keeps the
        # stored one. See gaussian_4d.widen_for_interpolation for why the two
        # cannot be the same number (3.50).
        interp_scale = self.config["Training"].get("deform_cfg", {}).get(
            "interp_time_scale")
        self.gaussian_4d_interp_time_scale = (
            None if interp_scale is None else float(interp_scale))
        # How the geometric velocity measures the shift between adjacent banks:
        # 'icp' (3.36, needed while bank centroids were outlier-contaminated) or
        # 'centroid' (3.47, better once ground-truth depth made every bank clean).
        # Default stays 'icp' so existing monocular-depth configs are unaffected.
        self.gaussian_4d_velocity_estimator = str(
            self.config["Training"].get("deform_cfg", {}).get(
                "velocity_estimator", "icp")).lower()
        if self.gaussian_4d_velocity_estimator not in ("icp", "centroid"):
            raise ValueError(
                "velocity_estimator must be 'icp' or 'centroid', got "
                f"{self.gaussian_4d_velocity_estimator!r}")
        if self.dynamic_mode == DYNAMIC_MODE_DEFORM:
            dcfg = self.config["Training"].get("deform_cfg", {})
            self.deform_net = DeformNet(
                xyz_freqs=dcfg.get("xyz_freqs", 6),
                t_freqs=dcfg.get("t_freqs", 4),
                hidden_dim=dcfg.get("hidden_dim", 128),
                n_layers=dcfg.get("n_layers", 3),
                t_scale=dcfg.get("t_scale", 1.0),
                max_dxyz=dcfg.get("max_dxyz", 1.0),
                max_daxis_angle=dcfg.get("max_daxis_angle", 1.0),
                max_dscale=dcfg.get("max_dscale", 2.0),
            ).cuda()
            self.deform_optimizer = torch.optim.Adam(
                self.deform_net.parameters(), lr=dcfg.get("lr", 1e-3)
            )
            self.deform_reg_weight = dcfg.get("reg_weight", 0.1)
            # Curriculum (panoramic_4dgs_status.md section 3.15): ramp deform_net's pre-tanh
            # output linearly from 0 to full strength over the first `deform_ramp_iters`
            # training iterations (see DeformNet.forward()'s `ramp` docstring for why this
            # replaced an earlier on/off warmup -- that only delayed saturation by the
            # warmup length, since tanh could still saturate within ~100 iterations of being
            # unfrozen; ramping the *pre-tanh* value keeps tanh's argument -- and so its
            # gradient -- from vanishing early, giving regularization a real chance to
            # compete instead of losing before it can act).
            self.deform_ramp_iters = dcfg.get("ramp_iters", dcfg.get("warmup_iters", 1000))
            # L2 penalty on deform_net's pre-tanh raw output (section 3.15) -- see
            # DeformNet.regularization_loss()'s docstring for why this is needed in addition
            # to (not instead of) the existing L1 penalty on tanh's output.
            self.deform_raw_reg_weight = dcfg.get("raw_reg_weight", 0.01)
            # Oracle-only diagnostic: counter the tiny synthetic sphere ROI being
            # diluted by a full-frame photometric mean.  Production configs omit
            # this key, so real-data behavior is unchanged.
            self.oracle_roi_weight = dcfg.get("oracle_roi_weight", 0.0)

        self.cameras_extent = 6.0
        self.set_hyperparams()
        self.physical_view_window = PhysicalViewWindow(self.window_size)
        # Compatibility alias for GUI/debug call sites. Entries are physical
        # timestamps now, not synthetic per-camera viewpoint keys.
        self.current_window = self.physical_view_window.current_times

        if self.use_gui:
            self.q_main2vis = mp.Queue()
            self.q_vis2main = mp.Queue()
            self.params_gui = gui_utils.ParamsGUI(
                background=self.background,
                gaussians=self.gaussians,
                q_main2vis=self.q_main2vis,
                q_vis2main=self.q_vis2main,
            )
            gui_process = mp.Process(target=slam_gui.run, args=(self.params_gui,))
            gui_process.start()
            time.sleep(3)

    def render_at(self, viewpoint, time_override=None):
        """render() wrapper that queries the deformation field (if enabled) at this
        viewpoint's timestamp first. Returns (render_pkg, deform_reg_loss) -- deform_reg_loss
        is a zero tensor when deform_net is disabled, so callers can unconditionally add it.

        ``time_override`` renders this viewpoint's *pose* at a different *time*.
        Training never passes it; it exists so render_erp_panoramas can stitch a
        full 360 image at an instant that was never observed, reusing the nearest
        observed instant's face poses. Leave-one-out is forced off in that case:
        it is a training device that hides a row at its own timestamp, and at an
        unobserved time there is no own timestamp to hide.

        NOTE: viewpoint.tstamp remains the synthetic unique view key used by the legacy
        Gaussian ownership path.  The explicit viewpoint.physical_tstamp added by the
        physical-time window is the deformation time shared by every cubemap/rig view of the
        same instant.  Legacy viewpoints without that field retain the old offset fallback.

        REVERTED ARCHITECTURE EXPERIMENT (panoramic_4dgs_status.md section 3.7): a
        `.detach()` was tried here on deform_net's xyz input, to sever dxyz's gradient from
        flowing back into the canonical Gaussians' own _xyz (the coupling section 4.1
        originally wanted, so deformation could help refine canonical geometry). The
        hypothesis was that this coupling's Fourier-encoded local Jacobian was amplifying
        into _xyz's gradient and driving dxyz's tanh output into saturation. Tested and
        disproven: dxyz saturated on virtually the same schedule with the coupling severed,
        and PSNR was worse (11.76 vs 14.06 without detaching) -- so the coupling was never
        the cause of the saturation, and detaching only cost the "deformation helps fix
        canonical geometry" capability for no benefit. Reverted; kept for the record since
        the negative result is informative (rules out the coupling-feedback hypothesis)."""
        real_t = getattr(
            viewpoint, 'physical_tstamp',
            viewpoint.tstamp - 500 * getattr(viewpoint, 'cam_idx', 0))
        if time_override is not None:
            # Render this pose at a time it never observed. Only the *time* is
            # borrowed; the pose stays the viewpoint's own, which is what makes a
            # panorama at an unobserved instant possible at all (no face poses
            # exist between observations).
            real_t = float(time_override)
        if self.dynamic_mode == DYNAMIC_MODE_OBJECT_SE3:
            # No deformation regularizer here: the motion lives in the object's
            # own SE(3) knots, not in a field that needs to be kept small.
            return render(
                viewpoint, self.gaussians, self.background,
                **object_se3_overrides(
                    self.gaussians, self.object_trajectories,
                    self.dynamic_observed_times, real_t,
                    tolerance=self.gaussians.oracle_time_tolerance)), 0.0
        if self.dynamic_mode == DYNAMIC_MODE_GAUSSIAN_4D:
            # Slice the 4D primitives at this instant. No regularizer: the motion
            # lives in each Gaussian's own extent in time, not in a field that
            # has to be kept small.
            # render_at is only ever called from the three training loops, so
            # leave-one-out belongs here; eval_utils renders with the full map.
            velocity = self.gaussians.get_velocity
            if self.gaussian_4d_pool_velocity:
                velocity = pool_velocity(
                    velocity, self.gaussians.dynamic_source_time,
                    self.gaussians.dynamic_score)
            # only a time nobody observed widens; training and every observed
            # frame keep the stored radius (3.50)
            time_scale = self.gaussians.get_time_scale
            if time_override is not None:
                time_scale = widen_for_interpolation(
                    time_scale, self.gaussian_4d_interp_time_scale)
            moved_xyz, faded_opacity, _ = slice_at_time(
                self.gaussians.get_xyz, self.gaussians.get_opacity,
                self.gaussians.dynamic_source_time, time_scale,
                velocity, real_t, self.gaussians.dynamic_score,
                exclude_own_bank=(self.gaussian_4d_leave_one_out
                                  and time_override is None),
                per_observation=self.gaussian_4d_per_observation)
            return render(
                viewpoint, self.gaussians, self.background,
                means3D_override=moved_xyz,
                opacities_override=faded_opacity), 0.0
        opacity_at_time = self.gaussians.get_opacity_at_time(real_t)
        if self.deform_net is None:
            return render(
                viewpoint, self.gaussians, self.background,
                opacities_override=opacity_at_time), 0.0

        ramp = min(1.0, self.iteration_count / max(1, self.deform_ramp_iters))
        dxyz, drot, dscale, raw = self.deform_net(self.gaussians.get_xyz, real_t, ramp=ramp)
        motion_gate = None
        raw_for_reg = raw
        if self.gaussians.oracle_dynamic_gate:
            motion_gate = self.gaussians.dynamic_score
            dxyz, drot, dscale = apply_motion_gate(
                dxyz, drot, dscale, motion_gate,
                translation_only=self.gaussians.oracle_translation_only)
            raw_for_reg = raw * motion_gate
        self._log_deform_health(
            dxyz, drot, dscale, viewpoint, real_t, raw=raw_for_reg,
            motion_gate=motion_gate)
        new_xyz, new_scaling_log, new_rotation = apply_deform(
            self.gaussians.get_xyz, self.gaussians._scaling, self.gaussians.get_rotation,
            dxyz, drot, dscale,
        )
        render_pkg = render(
            viewpoint, self.gaussians, self.background,
            means3D_override=new_xyz,
            scales_override=self.gaussians.scaling_activation(new_scaling_log),
            rotations_override=torch.nn.functional.normalize(new_rotation),
            opacities_override=opacity_at_time,
        )
        reg_loss = self.deform_reg_weight * self.deform_net.regularization_loss(
            dxyz, drot, dscale, raw=raw_for_reg, raw_weight=self.deform_raw_reg_weight)
        return render_pkg, reg_loss

    def _oracle_roi_loss(self, image, viewpoint):
        if self.deform_net is None or self.oracle_roi_weight <= 0:
            return image.sum() * 0.0
        return self.oracle_roi_weight * oracle_roi_l1(
            image,
            viewpoint.original_image.cuda(),
            self.gaussians.oracle_dynamic_colors,
            self.gaussians.oracle_color_threshold,
        )

    def _mask_oracle_shape_grads(self):
        """Block the ROI loss's canonical-scale inflation shortcut."""
        if not self.gaussians.oracle_freeze_dynamic_scaling:
            return
        zero_masked_rows_(
            self.gaussians._scaling.grad,
            self.gaussians.dynamic_score,
        )

    def _freeze_dynamic_positions(self):
        """Hold the moving object's Gaussians where seeding put them.

        Section 3.53 left the object half-fixed and said where the other half
        went. Seeding is now exact -- back-projecting the colour-gated pixels
        with the depth prior lands 100.0% of them on the sphere -- yet the
        finished map has only 44.8% there, while the fraction still inside the
        object's silhouette barely moved (86.3% -> 85.0%). The rows stay on the
        view ray and slide along it, so the drift happens after seeding, during
        refinement.

        It happens because a dynamic row is supervised by one frame. With
        time_scale 0.25 a bank contributes essentially nothing to its
        neighbours, and a single view cannot constrain depth along its own ray:
        every position on that ray reprojects to the same pixel, so the
        photometric loss is indifferent to where on it the Gaussian sits, and
        indifferent gradients are noise. Freezing the position spends that
        indifference on keeping the correct seed instead.

        Only the dynamic rows freeze, and only their position. Colour, opacity
        and shape still train, and the static map -- which does have multi-view
        support, and which section 3.52 measured accurate to 0.02 m -- is
        untouched.
        """
        if not self.gaussians.freeze_dynamic_positions:
            return
        zero_masked_rows_(
            self.gaussians._xyz.grad,
            self.gaussians.dynamic_score,
        )

    def _backward_with_oracle_roi(self, base_loss, roi_loss):
        if self.deform_net is None or self.oracle_roi_weight <= 0:
            base_loss.backward()
            return
        backward_with_auxiliary_params(
            base_loss,
            roi_loss,
            self.deform_net.parameters(),
        )

    def _log_deform_health(
            self, dxyz, drot, dscale, viewpoint, real_t, warn_thresh=5.0,
            raw=None, motion_gate=None):
        """Diagnostic added while tracking down the panoramic (cubemap) + 4DGS NaN
        instability (see panoramic_4dgs_status.md section 3). The existing NaN check in
        color_refinement() only fires once `loss` is already NaN, by which point xyz/dxyz
        themselves are typically already NaN too (max() over a NaN tensor is NaN, so the
        existing log line prints unhelpful "nan"s). This runs on every render_at() call
        instead, before anything has gone fully non-finite, so the growth trend leading up
        to the collapse -- and which cam_idx/cubemap face it starts on -- is visible in the
        log rather than only the moment of collapse. `raw` (added section 3.15) is
        deform_net's pre-tanh, pre-ramp value -- tracking it separately from dxyz_max exposes
        cases where tanh's output looks "stuck" at its bound while the network's true
        internal magnitude keeps growing far beyond it.

        dxyz_mean/dxyz_frac_sat (added section 3.19, panoramic_4dgs_status.md): dxyz_max alone
        is a worst-case statistic -- it only ever tells us about the single most-saturated
        Gaussian, and section 3.18 found a case where dxyz_max was pinned at the tanh cap for
        an entire 30000-step run while PSNR still improved substantially, which dxyz_max alone
        can't explain (did most Gaussians get better while a few stayed saturated, or did
        nothing really change?). dxyz_mean is the per-component mean magnitude across ALL
        Gaussians; dxyz_frac_sat is the fraction of (Gaussian, xyz-component) entries within 5%
        of the tanh cap (self.deform_net.max_dxyz) -- together they distinguish "almost
        everything is saturated" from "a small tail is saturated but the bulk isn't"."""
        with torch.no_grad():
            dxyz_finite, drot_finite, dscale_finite = torch.isfinite(dxyz), torch.isfinite(drot), torch.isfinite(dscale)
            nonfinite = not (dxyz_finite.all() and drot_finite.all() and dscale_finite.all())
            dxyz_abs_finite = dxyz[dxyz_finite].abs()
            dxyz_max = dxyz_abs_finite.max().item() if dxyz_finite.any() else float('nan')
            dxyz_mean = dxyz_abs_finite.mean().item() if dxyz_finite.any() else float('nan')
            cap = getattr(self.deform_net, 'max_dxyz', None)
            dxyz_frac_sat = (dxyz_abs_finite > 0.95 * cap).float().mean().item() if (dxyz_finite.any() and cap) else float('nan')
            drot_max = drot[drot_finite].abs().max().item() if drot_finite.any() else float('nan')
            dscale_max = dscale[dscale_finite].abs().max().item() if dscale_finite.any() else float('nan')
            raw_max = raw.abs().max().item() if raw is not None and torch.isfinite(raw).any() else float('nan')
            gate_fraction = motion_gate.float().mean().item() if motion_gate is not None else 1.0
            should_warn = nonfinite or dxyz_max > warn_thresh or drot_max > warn_thresh or dscale_max > warn_thresh
            periodic = (self.iteration_count % 200 == 0)
            if should_warn or periodic:
                Log(f"iter={self.iteration_count} cam_idx={getattr(viewpoint, 'cam_idx', None)} "
                    f"real_t={real_t} dxyz_max={dxyz_max:.3e} dxyz_mean={dxyz_mean:.3e} "
                    f"dxyz_frac_sat={dxyz_frac_sat:.3f} drot_max={drot_max:.3e} "
                    f"dscale_max={dscale_max:.3e} raw_max={raw_max:.3e} "
                    f"motion_gate_frac={gate_fraction:.3f} nonfinite={nonfinite} warn={should_warn}",
                    tag="DeformHealth")

    def set_hyperparams(self):
        self.init_itr_num = self.config["Training"]["init_itr_num"]
        self.init_gaussian_update = self.config["Training"]["init_gaussian_update"]
        self.init_gaussian_reset = self.config["Training"]["init_gaussian_reset"]
        self.init_gaussian_th = self.config["Training"]["init_gaussian_th"]
        self.init_gaussian_extent = self.cameras_extent * self.config["Training"]["init_gaussian_extent"]
        self.gaussian_update_every = self.config["Training"]["gaussian_update_every"]
        self.gaussian_update_offset = self.config["Training"]["gaussian_update_offset"]
        self.gaussian_th = self.config["Training"]["gaussian_th"]
        self.gaussian_extent = self.cameras_extent * self.config["Training"]["gaussian_extent"]
        self.gaussian_reset = self.config["Training"]["gaussian_reset"]
        self.size_threshold = self.config["Training"]["size_threshold"]
        self.window_size = self.config["Training"]["window_size"]
        self.lambda_dnormal = self.config["Training"]["lambda_dnormal"]


    def _projection_for_cam(self, cam_idx, packet):
        """Returns (K, projection_matrix) for this cam_idx, computed from this camera's own
        intrinsics and cached per cam_idx. Previously this was computed once (from whichever
        camera's packet happened to arrive first, always cam0 in practice) and reused
        unconditionally for every other camera's Camera objects -- silently baking cam0's
        fx/fy/cx/cy into every other camera's rendering/FoV forever, regardless of that
        camera's real intrinsics (see panoramic_4dgs_status.md section 3.8). For cubemap/
        equirect rigs all faces share identical intrinsics by construction (cubemap.py tiles
        the same fx/fy/cx/cy across all slots) so this had no effect there, but for a real
        multi-camera pinhole rig with genuinely different per-camera intrinsics it does."""
        if not hasattr(self, "_projection_by_cam"):
            self._projection_by_cam = {}
        if cam_idx not in self._projection_by_cam:
            H, W = packet["images"].shape[-2:]
            K = list(packet["intrinsics"][0]) + [W, H]
            projection_matrix = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=K[0], fy=K[1], cx=K[2], cy=K[3], W=W, H=H).transpose(0, 1).cuda()
            self._projection_by_cam[cam_idx] = (K, projection_matrix)
        return self._projection_by_cam[cam_idx]

    @staticmethod
    def _physical_tstamp(packet, packet_idx, view_key, cam_idx):
        """Read the explicit physical time, with legacy packet compatibility."""
        physical_tstamps = packet.get("physical_tstamp")
        if physical_tstamps is not None:
            return physical_tstamps[packet_idx].item()
        return view_key - 500 * cam_idx

    def _record_time_metadata(self, packet):
        metadata = packet.get("time_metadata")
        if metadata is not None:
            self.time_metadata = copy.deepcopy(metadata)

    def _store_viewpoint(self, viewpoint, view_key, physical_tstamp, cam_idx, depth_map,
                         prior_depth_map=None):
        """Store one virtual view while updating a physical-time window."""
        is_new_view = view_key not in self.viewpoints
        if not self.initialized:
            self.reset()

        viewpoint.physical_tstamp = physical_tstamp
        self.viewpoints[view_key] = viewpoint
        self.physical_view_window.register(physical_tstamp, cam_idx, view_key)

        if not self.initialized:
            self.add_next_kf(0, viewpoint, depth_map=depth_map,
                             prior_depth_map=prior_depth_map, init=True)
            self.initialize_map(0, viewpoint)
            self.initialized = True
        elif is_new_view:
            self.add_next_kf(view_key, viewpoint, depth_map=depth_map,
                             prior_depth_map=prior_depth_map)

    def _publish_mapping_gui(self, current_viewpoint):
        if not self.use_gui or current_viewpoint is None:
            return
        window_keys = self.physical_view_window.window_view_keys(self.iteration_count)
        if not window_keys:
            return
        keyframes = [self.viewpoints[view_key] for view_key in window_keys]
        current_window_dict = {window_keys[0]: window_keys[1:]}
        self.q_main2vis.put(
            gui_utils.GaussianPacket(
                gaussians=clone_obj(self.gaussians),
                current_frame=current_viewpoint,
                keyframes=keyframes,
                kf_window=current_window_dict,
                gtcolor=current_viewpoint.original_image,
                gtdepth=current_viewpoint.depth.numpy()))

    def process_track_data(self, packet):
        self._record_time_metadata(packet)
        cam_idx = packet.get('cam_idx', 0)
        K, projection_matrix = self._projection_for_cam(cam_idx, packet)
        if not hasattr(self, "projection_matrix"):
            # kept for eval_rendering()/finalize()'s legacy single-camera use (cam0's own)
            self.K, self.projection_matrix = K, projection_matrix

        w2c = SE3(packet["poses"]).matrix().cuda()
        viewpoint = None
        for i, _ in enumerate(packet['viz_idx']):
            view_key = packet['tstamp'][i].item()
            physical_tstamp = self._physical_tstamp(packet, i, view_key, cam_idx)
            viewpoint = Camera.init_from_tracking(
                packet["images"][i] / 255.0, packet["depths"][i], packet["normals"][i],
                w2c[i], view_key, projection_matrix, K, view_key, cam_idx=cam_idx)
            prior = packet.get("prior_depths")
            self._store_viewpoint(
                viewpoint, view_key, physical_tstamp, cam_idx,
                depth_map=packet["depths"][i].numpy(),
                prior_depth_map=None if prior is None else prior[i].numpy())

        # call_gs() sends camera blocks sequentially. Delay mapping until the
        # last block, then preserve the old total optimizer/render budget.
        if not packet.get("finalize_batch", True):
            return
        map_iterations = int(packet.get("map_iterations", 10))
        active_views = sum(
            len(self.physical_view_window.groups[physical_tstamp])
            for physical_tstamp in self.current_window)
        Log(
            f"physical_times={len(self.current_window)} active_views={active_views} "
            f"views_per_iteration={len(self.current_window)} map_iterations={map_iterations}",
            tag="PhysicalWindow")
        self.map(self.current_window, iters=map_iterations)
        self._publish_mapping_gui(viewpoint)

    def process_global_track_data(self, packet, cam_num):
        self._record_time_metadata(packet)
        if not hasattr(self, "projection_matrix"):
            H, W = packet["images"].shape[-2:]
            self.K = K = list(packet["intrinsics"][0]) + [W, H]
            self.projection_matrix = getProjectionMatrix2(znear=0.01, zfar=100.0, fx=K[0], fy=K[1], cx=K[2], cy=K[3], W=W, H=H).transpose(0, 1).cuda()

        # packet stacks all cameras in order [cam0 frames, cam1 frames, ...], so the camera
        # index of frame i is i // n_per_cam, and each camera's own intrinsics can be read
        # from the first row of its block -- see the matching bugfix/comment in
        # _projection_for_cam() (panoramic_4dgs_status.md section 3.8).
        n_per_cam = max(1, len(packet['viz_idx']) // cam_num)
        for _cam_idx in range(cam_num):
            self._projection_for_cam(_cam_idx, {
                "images": packet["images"],
                "intrinsics": packet["intrinsics"][_cam_idx * n_per_cam:],
            })

        if packet['pose_updates'] is not None:
            with torch.no_grad():
                tstamps = packet['tstamp']
                indices = (tstamps.unsqueeze(1) == self.gaussians.unique_kfIDs.unsqueeze(0)).nonzero()[:, 0] % int(tstamps.shape[0] / cam_num)
                updates = packet['pose_updates'].cuda()[indices]
                updates_scale = packet['scale_updates'].cuda()[indices]
                
                xyz = self.gaussians.get_xyz
                xyz = (updates * xyz) / updates_scale
                self.gaussians._xyz[:] = xyz

                scale = self.gaussians.get_scaling
                scale = scale / updates_scale
                self.gaussians._scaling[:] = self.gaussians.scaling_inverse_activation(scale)
 
                rot = SO3(self.gaussians.get_rotation)
                rot = SO3(updates.data[:,3:]) * rot
                self.gaussians._rotation[:] = rot.data

        w2c = SE3(packet["poses"]).matrix().cuda()
        viewpoint = None
        for i, _ in enumerate(packet['viz_idx']):
            view_key = packet['tstamp'][i].item()
            cam_idx = i // n_per_cam
            physical_tstamp = self._physical_tstamp(packet, i, view_key, cam_idx)
            K, projection_matrix = self._projection_by_cam[cam_idx]
            viewpoint = Camera.init_from_tracking(
                packet["images"][i] / 255.0, packet["depths"][i], packet["normals"][i],
                w2c[i], view_key, projection_matrix, K, view_key, cam_idx=cam_idx)
            prior = packet.get("prior_depths")
            self._store_viewpoint(
                viewpoint, view_key, physical_tstamp, cam_idx,
                depth_map=packet["depths"][i].numpy(),
                prior_depth_map=None if prior is None else prior[i].numpy())

        self.map(self.current_window, iters=10)
        # self.map(self.current_window, iters=1, prune=True)
        self._publish_mapping_gui(viewpoint)

    def estimate_object_motion(self):
        """Fill each dynamic row's velocity by registering adjacent banks.

        Geometry, not learning. Sections 3.32-3.35 showed the photometric loss
        cannot learn object motion here -- it always finds a cheaper explanation,
        and the learned velocity scored worse than no velocity at all. Section
        3.36 measured the alternative: registration gets direction cosine 0.797
        against learning's 0.244, and is the best-scoring option overall.

        Whether the shift is measured by ICP or by the plain centroid difference
        is ``deform_cfg.velocity_estimator``; see estimate_bank_velocities for
        which one wins when.

        Runs once at the end, when the banks are final. Returns how many rows
        were assigned.
        """
        if self.dynamic_mode != DYNAMIC_MODE_GAUSSIAN_4D:
            return 0
        if not self.dynamic_observed_times:
            # gaussian_4d does not register trajectories, so recover the observed
            # times from the rows themselves
            source = self.gaussians.dynamic_source_time.reshape(-1)
            dynamic = self.gaussians.dynamic_score.reshape(-1) > 0.5
            if not bool(dynamic.any()):
                return 0
            self.dynamic_observed_times = sorted(
                float(v) for v in torch.unique(source[dynamic]).tolist())
        estimates = estimate_bank_velocities(
            self.gaussians.get_xyz.detach(),
            self.gaussians.dynamic_object_id,
            self.gaussians.dynamic_source_time,
            self.dynamic_observed_times,
            refine=(self.gaussian_4d_velocity_estimator == "icp"))
        if not estimates:
            return 0
        velocity = velocities_to_rows(
            estimates, self.gaussians.dynamic_object_id,
            self.gaussians.dynamic_source_time, self.gaussians.get_xyz)
        with torch.no_grad():
            self.gaussians._velocity.copy_(velocity)
            # Drop any optimizer momentum carried by the old values. Harmless
            # today because finalize() does not train afterwards, but a stale
            # exp_avg would quietly drag the estimate away the moment anything
            # steps this parameter again.
            if self.gaussians.optimizer is not None:
                state = self.gaussians.optimizer.state.get(
                    self.gaussians._velocity)
                if state:
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in state:
                            state[key].zero_()
        assigned = int((velocity.norm(dim=1) > 0).sum().item())
        Log(f"object motion from bank registration: {len(estimates)} banks, "
            f"{assigned} rows assigned", tag="4DGS")
        return assigned

    def render_time_sweep(self, steps_per_gap=4, max_frames=200):
        """Render a fixed viewpoint at a continuous sweep of times.

        This is what "4D" buys over the per-timestamp bank: the camera is held
        still while time advances in fractional steps, so most of the frames are
        at times that were never observed. The bank renders nothing dynamic at
        those; here the object should move smoothly through them.
        """
        if self.dynamic_mode != DYNAMIC_MODE_GAUSSIAN_4D:
            return 0
        times = sorted(self.dynamic_observed_times)
        if len(times) < 2:
            return 0
        # one fixed camera: cam0 at the middle observed time, so the object
        # passes across the view rather than starting off-screen
        middle = times[len(times) // 2]
        # nearest cam0 view rather than an exact time match: an exact match can
        # be missing (that instant's cam0 pruned, or the dynamic times not being
        # a subset of the viewpoint times), and silently rendering nothing is a
        # worse outcome than sweeping from a neighbouring pose
        anchor, best = None, None
        for viewpoint in self.viewpoints.values():
            if getattr(viewpoint, "cam_idx", 0) != 0:
                continue
            distance = abs(getattr(viewpoint, "physical_tstamp", 1e9) - middle)
            if best is None or distance < best:
                anchor, best = viewpoint, distance
        if anchor is None:
            Log("time sweep skipped: no cam0 viewpoint available", tag="4DGS")
            return 0

        out_dir = os.path.join(self.save_dir, "renders", "time_sweep")
        os.makedirs(out_dir, exist_ok=True)
        sweep = []
        for left, right in zip(times, times[1:]):
            for step in range(steps_per_gap):
                sweep.append(left + (right - left) * step / steps_per_gap)
        sweep.append(times[-1])
        sweep = sweep[:max_frames]

        observed = {round(float(t), 6) for t in times}
        for index, t in enumerate(sweep):
            with torch.no_grad():
                velocity = self.gaussians.get_velocity
                if self.gaussian_4d_pool_velocity:
                    velocity = pool_velocity(
                        velocity, self.gaussians.dynamic_source_time,
                        self.gaussians.dynamic_score)
                time_scale = self.gaussians.get_time_scale
                if round(float(t), 6) not in observed:
                    time_scale = widen_for_interpolation(
                        time_scale, self.gaussian_4d_interp_time_scale)
                moved, faded, _ = slice_at_time(
                    self.gaussians.get_xyz, self.gaussians.get_opacity,
                    self.gaussians.dynamic_source_time,
                    time_scale, velocity, t,
                    self.gaussians.dynamic_score,
                    per_observation=self.gaussian_4d_per_observation)
                pkg = render(anchor, self.gaussians, self.background,
                             means3D_override=moved, opacities_override=faded)
            image = torch.clamp(pkg["render"], 0.0, 1.0)
            frame = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255)
            frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)
            tag = "obs" if round(float(t), 6) in observed else "interp"
            cv2.imwrite(os.path.join(out_dir, f"{index:04d}_t{t:07.3f}_{tag}.jpg"), frame)
        Log(f"time sweep: {len(sweep)} frames "
            f"({sum(1 for t in sweep if round(float(t), 6) not in observed)} at "
            f"never-observed times) -> {out_dir}", tag="4DGS")
        return len(sweep)

    def render_erp_panoramas(self, max_instants=8, target_times=None,
                             subdir="erp_panorama"):
        """Render full equirectangular panoramas -- the actual deliverable.

        The pipeline reconstructs from cubemap faces and, until now, only ever
        rendered faces back. But the output of a *panoramic* reconstruction
        should be a 360 image, so this renders every face of an instant from its
        own pose and stitches them with cubemap.cubemap_to_erp.

        With the four-face calib only the horizontal band exists (the poles were
        deliberately skipped to avoid the ERP singularity, see
        panoramic_support_feasibility.md 2.2) and the top and bottom of every
        panorama stay black; calib/equirect_6face.yml adds up/down and closes
        that. Either way the covered fraction is reported rather than hidden.

        ``target_times`` renders at arbitrary instants instead of the observed
        ones. No face poses exist between observations, so each requested time
        borrows the *poses* of the nearest observed instant and supplies its own
        *time* to the 4D slice. That combination -- a full sphere at a moment
        that was never captured -- is the panoramic half of "panoramic 4DGS",
        which render_time_sweep only ever showed on a single cubemap face.
        """
        # cam_idx order after motion_filter drops slot 1 (the front duplicate):
        # front, then the extra faces from the calib file in their listed order.
        # mcgs.py mirrors these from args; the fallback only applies to configs
        # that predate that (and matches the default calib).
        training = self.config.get("Training", {})
        faces_by_cam = ['front'] + list(
            training.get("cubemap_faces") or ['right', 'back', 'left'])
        fov = float(training.get("fov", 90.0))
        groups = getattr(self.physical_view_window, "groups", {})
        if not groups:
            return 0
        out_dir = os.path.join(self.save_dir, "renders", subdir)
        os.makedirs(out_dir, exist_ok=True)

        observed = sorted(groups)
        if target_times is None:
            # each observed instant rendered at its own time
            schedule = [(t, t, None) for t in observed[:max_instants]]
        else:
            schedule = []
            for t in list(target_times)[:max_instants]:
                pose_time = min(observed, key=lambda o: abs(o - float(t)))
                schedule.append((float(t), pose_time, float(t)))

        written, coverages, interpolated = 0, [], 0
        for stamp, pose_time, time_override in schedule:
            entries = groups[pose_time]
            face_images = {}
            for cam_idx, view_key in (entries.items() if isinstance(entries, dict)
                                      else entries):
                viewpoint = self.viewpoints.get(view_key)
                if viewpoint is None or cam_idx >= len(faces_by_cam):
                    continue
                with torch.no_grad():
                    pkg, _ = self.render_at(viewpoint, time_override=time_override)
                # same exposure compensation eval_rendering_kf applies; without it
                # every face carries a systematic brightness offset
                image = (torch.exp(viewpoint.exposure_a) * pkg["render"]
                         + viewpoint.exposure_b)
                image = torch.clamp(image, 0.0, 1.0)
                frame = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255)
                # render() works in RGB; cv2 wants BGR, same as eval_rendering_kf
                face_images[faces_by_cam[cam_idx]] = cv2.cvtColor(
                    frame.astype(np.uint8), cv2.COLOR_RGB2BGR)
            if len(face_images) < 2:
                continue
            size = next(iter(face_images.values())).shape[0]
            # ERP height = face size: a 90-degree face spans 90 degrees of
            # latitude, which is half an ERP's height, so this keeps the face's
            # own sampling density without inventing resolution
            erp, covered = cubemap.cubemap_to_erp(
                face_images, size * 2, size * 4, fov_deg=fov)
            # tag the frames whose time no camera ever saw, so the deliverable is
            # self-describing rather than needing the launch command to interpret
            tag = "obs" if abs(stamp - pose_time) < 1e-6 else "interp"
            interpolated += tag == "interp"
            cv2.imwrite(
                os.path.join(out_dir, f"erp_{stamp:08.3f}_{tag}.jpg"), erp)
            coverages.append(float(covered.mean()))
            written += 1
        if not written:
            Log(f"ERP panoramas: nothing written -- {len(groups)} instants known, "
                f"but none had 2+ renderable faces (faces_by_cam={faces_by_cam})",
                tag="4DGS")
        if written:
            Log(f"ERP panoramas: {written} instants ({interpolated} at "
                f"never-observed times), "
                f"{100 * sum(coverages) / len(coverages):.1f}% of the sphere covered "
                f"by {len(faces_by_cam)} faces -> {out_dir}", tag="4DGS")
        return written

    def render_erp_panoramas_interpolated(self, max_instants=8):
        """Full-sphere panoramas at the midpoints between observed instants.

        This is the deliverable the project has been missing: render_time_sweep
        proved the 4D representation is defined between observations but showed
        it on one 384x384 cubemap face, while render_erp_panoramas produced real
        360 images but only at instants a camera actually captured. Neither one
        on its own demonstrates "panoramic 4DGS"; this is both at once.

        Midpoints specifically, because that is where the per-timestamp bank
        renders exactly nothing -- the widest gap between what the old
        representation could do and what this one can.

        Skipped for static maps, where every time renders the same image and an
        "interpolated" panorama would be a duplicate rather than evidence.
        """
        if self.dynamic_mode not in (DYNAMIC_MODE_GAUSSIAN_4D,
                                     DYNAMIC_MODE_OBJECT_SE3):
            return 0
        times = sorted(self.dynamic_observed_times)
        if len(times) < 2:
            return 0
        midpoints = [0.5 * (a + b) for a, b in zip(times, times[1:])]
        return self.render_erp_panoramas(
            max_instants=max_instants, target_times=midpoints,
            subdir="erp_panorama_interp")

    def log_viewpoint_poses(self):
        """Print each cubemap face's actual camera centre and optical axis.

        Diagnostic for section 3.46: the pole faces render pure black even though
        their Gaussians sit at correct world positions with adequate size, which
        points at the viewpoint pose rather than the map. All faces of one
        instant share an optical centre, and their optical axes should point
        along +z/+x/-z/-x/-y/+y for front/right/back/left/up/down.
        """
        faces = ['front'] + list(
            self.config.get("Training", {}).get("cubemap_faces")
            or ['right', 'back', 'left'])
        seen = {}
        views = {}
        for view_key, viewpoint in sorted(self.viewpoints.items()):
            cam_idx = int(getattr(viewpoint, "cam_idx", 0))
            if cam_idx in seen:
                continue
            R = viewpoint.R.detach().cpu().numpy()
            T = viewpoint.T.detach().cpu().numpy()
            centre = -R.T @ T
            axis = R.T @ np.array([0.0, 0.0, 1.0])   # camera +z in world
            seen[cam_idx] = (centre, axis, R, T)
            views[cam_idx] = viewpoint
        for cam_idx in sorted(seen):
            centre, axis, R, T = seen[cam_idx]
            name = faces[cam_idx] if cam_idx < len(faces) else str(cam_idx)
            # print, not Log: Log crops to a fixed width and silently ate the
            # second half of this line the first time round.
            print(f"[POSE] cam{cam_idx} {name} centre "
                  f"{centre[0]:+.3f} {centre[1]:+.3f} {centre[2]:+.3f}",
                  flush=True)
            print(f"[POSE] cam{cam_idx} {name} axis "
                  f"{axis[0]:+.3f} {axis[1]:+.3f} {axis[2]:+.3f}", flush=True)
            viewpoint = views[cam_idx]
            with torch.no_grad():
                pkg = render(viewpoint, self.gaussians, self.background)
                image, radii = pkg["render"], pkg["radii"]
                # Where do the map's Gaussians land in THIS camera's frame?
                xyz = self.gaussians.get_xyz
                Rt = torch.from_numpy(R).float().cuda()
                Tt = torch.from_numpy(T).float().cuda()
                cam_xyz = xyz @ Rt.T + Tt[None]
                z = cam_xyz[:, 2]
                infront = z > 0.01
                # 90 deg face: |x|,|y| < z inside the frustum
                inside = infront & (cam_xyz[:, 0].abs() < z) & (cam_xyz[:, 1].abs() < z)
            print(f"[POSE] cam{cam_idx} {name} fov "
                  f"{viewpoint.FoVx:.3f}/{viewpoint.FoVy:.3f} "
                  f"size {viewpoint.image_width}x{viewpoint.image_height}", flush=True)
            print(f"[POSE] cam{cam_idx} {name} gaussians total {xyz.shape[0]} "
                  f"infront {int(infront.sum())} infrustum {int(inside.sum())} "
                  f"rasterised {int((radii > 0).sum())}", flush=True)
            print(f"[POSE] cam{cam_idx} {name} image mean {float(image.mean()):.5f} "
                  f"max {float(image.max()):.5f}", flush=True)
            zin = z[inside]
            if zin.numel():
                q = torch.quantile(zin, torch.tensor([0.0, 0.5, 1.0], device=zin.device))
                print(f"[POSE] cam{cam_idx} {name} frustum-z min {float(q[0]):.5f} "
                      f"median {float(q[1]):.5f} max {float(q[2]):.5f} "
                      f"znear {float(getattr(viewpoint, 'znear', -1)):.5f} "
                      f"zfar {float(getattr(viewpoint, 'zfar', -1)):.5f}", flush=True)
                op = self.gaussians.get_opacity.reshape(-1)[inside]
                sc = self.gaussians.get_scaling[inside].mean(dim=1)
                print(f"[POSE] cam{cam_idx} {name} frustum-opacity median "
                      f"{float(op.median()):.5f} frac>0.05 "
                      f"{float((op > 0.05).float().mean()):.3f} scale median "
                      f"{float(sc.median()):.6f}", flush=True)

    def finalize(self):
        if os.environ.get("MCGS_POSE_DEBUG"):
            self.log_viewpoint_poses()
        self.color_refinement(iteration_total=self.gaussians.max_steps)
        # after refinement, so registration sees the final bank geometry
        self.estimate_object_motion()
        self.render_time_sweep()
        self.render_erp_panoramas()
        self.render_erp_panoramas_interpolated()
        self.gaussians.save_ply(f'{self.save_dir}/3dgs_final.ply')
        self.save_checkpoint(os.path.join(self.save_dir, "4dgs_final.pt"))

        # BUGFIX (panoramic_4dgs_status.md section 3.8): self.viewpoints holds every camera's
        # views keyed by tstamp+500*cam_idx (see the NOTE in render_at()), so iterating over
        # *all* of them here mixed multiple cameras' poses into one "trajectory" sorted by
        # those fake, offset keys -- garbage for any multi-camera/cubemap run (single-camera
        # runs happened to be unaffected since cam_idx is always 0 there, offset=0). The saved
        # trajectory is meant to be the single primary tracked camera's pose history (matches
        # how README's evo_ape usage compares it against one gt_poses.txt), so restrict to
        # cam_idx==0 views, whose tstamp already needs no offset correction.
        poses_cw = []
        for view in self.viewpoints.values():
            if getattr(view, 'cam_idx', 0) != 0:
                continue
            T_w2c = np.eye(4)
            T_w2c[0:3, 0:3] = view.R.cpu().numpy()
            T_w2c[0:3, 3] = view.T.cpu().numpy()
            poses_cw.append(np.hstack(([view.tstamp], to_se3_vec(T_w2c))))
        poses_cw.sort(key=lambda x: x[0])
        return np.stack(poses_cw)

    def checkpoint_payload(self):
        """Build a versioned replay/resume payload for the temporal Gaussian map."""
        deform_state = None
        if self.deform_net is not None:
            deform_state = {
                key: value.detach().cpu().clone()
                for key, value in self.deform_net.state_dict().items()
            }
        return {
            "format": self.CHECKPOINT_FORMAT,
            "version": self.CHECKPOINT_VERSION,
            "config": copy.deepcopy(self.config),
            "iteration_count": int(self.iteration_count),
            "time": copy.deepcopy(self.time_metadata),
            "dynamic_mode": self.dynamic_mode,
            "background": self.background.detach().cpu().clone(),
            "gaussians": self.gaussians.checkpoint_state(),
            "object_trajectories": self.object_trajectories.checkpoint_state(),
            "deform_state_dict": deform_state,
            "gaussian_optimizer_state": (
                self.gaussians.optimizer.state_dict()
                if self.gaussians.optimizer is not None else None),
            "deform_optimizer_state": (
                self.deform_optimizer.state_dict()
                if self.deform_optimizer is not None else None),
            "viewpoints_included": False,
        }

    def save_checkpoint(self, path):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(self.checkpoint_payload(), path)

    @classmethod
    def from_checkpoint(cls, path, save_dir=None, use_gui=False,
                        resume_optimizers=False):
        """Construct a backend from the exact config stored in a checkpoint."""
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint.get("format") != cls.CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported checkpoint format in {path}")
        if save_dir is None:
            save_dir = os.path.dirname(os.path.abspath(path))
        backend = cls(checkpoint["config"], save_dir, use_gui=use_gui)
        backend.load_checkpoint(path, resume_optimizers=resume_optimizers)
        return backend

    def load_checkpoint(self, path, resume_optimizers=False):
        """Restore temporal Gaussian state into a backend built from saved config."""
        checkpoint = torch.load(path, map_location="cpu")
        if checkpoint.get("format") != self.CHECKPOINT_FORMAT:
            raise ValueError(f"unsupported checkpoint format in {path}")
        if int(checkpoint.get("version", -1)) != self.CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported checkpoint version {checkpoint.get('version')}; "
                f"expected {self.CHECKPOINT_VERSION}")
        saved_mode = checkpoint.get("dynamic_mode", self.dynamic_mode)
        if saved_mode != self.dynamic_mode:
            raise ValueError(
                f"checkpoint dynamic_mode '{saved_mode}' does not match runtime "
                f"mode '{self.dynamic_mode}'")

        self.gaussians.restore_checkpoint_state(
            checkpoint["gaussians"],
            training_args=self.opt_params if resume_optimizers else None,
        )
        deform_state = checkpoint.get("deform_state_dict")
        if deform_state is not None:
            if self.deform_net is None:
                raise ValueError(
                    "checkpoint contains DeformNet weights; construct GSBackEnd with "
                    "the checkpoint's saved config before loading")
            self.deform_net.load_state_dict(deform_state)
        self.object_trajectories.restore_checkpoint_state(
            checkpoint.get("object_trajectories", {}), device="cuda")
        # Not stored separately: every knot came from a dynamic keyframe, so the
        # knot times are exactly the observed times.
        self.dynamic_observed_times = observed_times_from_trajectories(
            self.object_trajectories)
        self.iteration_count = int(checkpoint.get("iteration_count", 0))
        self.time_metadata = copy.deepcopy(checkpoint.get("time", self.time_metadata))
        background = checkpoint.get("background")
        if background is not None:
            self.background = background.to(device="cuda").clone()

        if resume_optimizers:
            gaussian_optimizer = checkpoint.get("gaussian_optimizer_state")
            if gaussian_optimizer is not None:
                self.gaussians.optimizer.load_state_dict(gaussian_optimizer)
            deform_optimizer = checkpoint.get("deform_optimizer_state")
            if deform_optimizer is not None and self.deform_optimizer is not None:
                self.deform_optimizer.load_state_dict(deform_optimizer)
        return checkpoint

    @torch.no_grad()
    def eval_rendering(self, gtimages, gtdepthdir, traj, kf_idx):
        eval_rendering(gtimages, gtdepthdir, traj, self.gaussians,self.save_dir, self.background,
            self.projection_matrix, self.K, kf_idx, iteration="after_opt")
    
    def eval_rendering_kf(self):
        object_se3 = self.dynamic_mode == DYNAMIC_MODE_OBJECT_SE3
        gaussian_4d = self.dynamic_mode == DYNAMIC_MODE_GAUSSIAN_4D
        eval_rendering_kf(self.viewpoints, self.gaussians, self.save_dir, self.background,
                           iteration="after_opt", deform_net=self.deform_net,
                           object_trajectories=(
                               self.object_trajectories if object_se3 else None),
                           observed_times=self.dynamic_observed_times,
                           gaussian_4d=gaussian_4d,
                           per_observation=self.gaussian_4d_per_observation)

    def add_next_kf(self, frame_idx, viewpoint, init=False, scale=2.0, depth_map=None,
                    prior_depth_map=None):
        rows_before = self.gaussians.get_xyz.shape[0]
        self.gaussians.extend_from_pcd_seq(
            viewpoint, kf_id=frame_idx, init=init, scale=scale, depthmap=depth_map,
            prior_depthmap=prior_depth_map
        )
        if self.dynamic_mode == DYNAMIC_MODE_OBJECT_SE3:
            self._observe_object_trajectories(rows_before)

    def _observe_object_trajectories(self, rows_before):
        """Register a trajectory knot per object seen in the rows just seeded.

        The knot is the centroid of the object's freshly seeded Gaussians (see
        ``ObjectTrajectoryTable.observe_centroids`` for what that does and does
        not measure).  Several rig/cubemap views share one physical time;
        ``observe`` averages repeated times into the same knot, so all of them
        contribute.

        ``optimizer=None`` is deliberate, not the silent-no-training hazard from
        section 3.28: training only ever renders *observed* times, and carrying a
        row from an observed time to itself is the identity regardless of the
        knots, so photometric loss produces exactly zero gradient for them.
        Attaching an optimizer here would step nothing. Supervising the knots
        needs a loss at times between observations, which this stage does not
        have.
        """
        registered = self.object_trajectories.observe_centroids(
            self.gaussians.get_xyz[rows_before:],
            self.gaussians.dynamic_object_id[rows_before:],
            self.gaussians.dynamic_source_time[rows_before:],
            optimizer=None,
        )
        for source_time in registered:
            if source_time not in self.dynamic_observed_times:
                self.dynamic_observed_times.append(source_time)
        self.dynamic_observed_times.sort()

    def reset(self):
        self.iteration_count = 0
        if hasattr(self, "physical_view_window"):
            self.physical_view_window.clear()
            self.current_window = self.physical_view_window.current_times
        else:
            self.current_window = []
        self.initialized = False
        # remove all gaussians
        self.gaussians.prune_points(self.gaussians.unique_kfIDs >= 0)
        # ...and the object state that describes them: keeping trajectories for
        # banks that no longer exist would leave observed times pointing at
        # nothing.  An empty payload rebuilds an empty table.
        self.object_trajectories.restore_checkpoint_state({}, device="cuda")
        self.dynamic_observed_times = []
        # Leave-one-out training for gaussian_4d. Default OFF: it was tried in
        # section 3.33 and made things worse -- hiding the frame's own bank
        # creates a hole where the object should be, and the cheapest way to
        # fill that hole is not learning the velocity. Kept because the negative
        # result is worth being able to reproduce.
        self.gaussian_4d_leave_one_out = bool(
            self.config["Training"].get("deform_cfg", {}).get(
                "leave_one_out", False))
        # Share one velocity per bank rather than one per Gaussian (3.34).
        self.gaussian_4d_pool_velocity = bool(
            self.config["Training"].get("deform_cfg", {}).get(
                "pool_velocity", True))
        # Weight the temporal conditional per observation rather than per
        # Gaussian, so a bank's influence does not scale with how many rows
        # seeding and pruning happened to leave it (3.49). Disproven in 3.49.1;
        # kept switchable so the negative result stays reproducible.
        self.gaussian_4d_per_observation = bool(
            self.config["Training"].get("deform_cfg", {}).get(
                "per_observation_weights", False))
        # Temporal radius used only at times nobody observed. None keeps the
        # stored one. See gaussian_4d.widen_for_interpolation for why the two
        # cannot be the same number (3.50).
        interp_scale = self.config["Training"].get("deform_cfg", {}).get(
            "interp_time_scale")
        self.gaussian_4d_interp_time_scale = (
            None if interp_scale is None else float(interp_scale))
        # How the geometric velocity measures the shift between adjacent banks:
        # 'icp' (3.36, needed while bank centroids were outlier-contaminated) or
        # 'centroid' (3.47, better once ground-truth depth made every bank clean).
        # Default stays 'icp' so existing monocular-depth configs are unaffected.
        self.gaussian_4d_velocity_estimator = str(
            self.config["Training"].get("deform_cfg", {}).get(
                "velocity_estimator", "icp")).lower()
        if self.gaussian_4d_velocity_estimator not in ("icp", "centroid"):
            raise ValueError(
                "velocity_estimator must be 'icp' or 'centroid', got "
                f"{self.gaussian_4d_velocity_estimator!r}")

    def initialize_map(self, cur_frame_idx, viewpoint):
        for mapping_iteration in range(self.init_itr_num):
            self.iteration_count += 1
            render_pkg, deform_reg = self.render_at(viewpoint)
            (image, viewspace_point_tensor, visibility_filter, radii, depth, n_touched) = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
                render_pkg["depth"],
                render_pkg["n_touched"]
            )
            loss_init = get_loss_mapping_rgbd(self.config, image, depth, viewpoint) + deform_reg
            roi_loss_init = self._oracle_roi_loss(image, viewpoint)
            self._backward_with_oracle_roi(loss_init, roi_loss_init)
            self._mask_oracle_shape_grads()
            self._freeze_dynamic_positions()

            with torch.no_grad():
                self.gaussians.max_radii2D[visibility_filter] = torch.max(
                    self.gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                self.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter
                )
                if mapping_iteration % self.init_gaussian_update == 0:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.init_gaussian_th,
                        self.init_gaussian_extent,
                        None,
                    )

                if self.iteration_count == self.init_gaussian_reset:
                    self.gaussians.reset_opacity()

                if self.deform_net is not None:
                    # BUGFIX (panoramic_4dgs_status.md section 3.6): dxyz's gradient flows
                    # straight through into the canonical Gaussians' own _xyz (by design --
                    # see deform_net.py docstring), but self.gaussians.optimizer.step() below
                    # applies that gradient BEFORE the clip_grad_norm_ two lines down even
                    # runs -- and that clip only ever touched self.deform_net.parameters(),
                    # a completely different tensor from self.gaussians._xyz. So _xyz's own
                    # gradient (the one actually carrying the amplified coupling term) was
                    # never clipped at all; the old comment here describing this clip as a
                    # guard against exactly this failure mode was incorrect about what the
                    # code did. Clipping _xyz's gradient directly, before it's applied, is
                    # the fix that actually protects the vulnerable tensor.
                    torch.nn.utils.clip_grad_norm_([self.gaussians._xyz], max_norm=1.0)
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                if self.deform_optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(self.deform_net.parameters(), max_norm=1.0)
                    self.deform_optimizer.step()
                    self.deform_optimizer.zero_grad(set_to_none=True)

        Log("Initialized map")
        return render_pkg

    def map(self, current_window, iters, prune=False):
        if len(current_window) == 0:
            return

        for _ in range(iters):
            self.iteration_count += 1

            loss_mapping = 0
            roi_loss_mapping = 0
            viewspace_point_tensor_acm = []
            visibility_filter_acm = []
            radii_acm = []

            window_keys = self.physical_view_window.window_view_keys(self.iteration_count)
            replay_keys = self.physical_view_window.replay_view_keys(
                limit=2, step=self.iteration_count)
            viewpoints = [self.viewpoints[view_key] for view_key in window_keys + replay_keys]
            if not viewpoints:
                continue
            for viewpoint in viewpoints:
                render_pkg, deform_reg = self.render_at(viewpoint)
                image, viewspace_point_tensor, visibility_filter, radii, depth, n_touched = (
                    render_pkg["render"],
                    render_pkg["viewspace_points"],
                    render_pkg["visibility_filter"],
                    render_pkg["radii"],
                    render_pkg["depth"],
                    render_pkg["n_touched"])

                loss_mapping += self.lambda_dnormal * get_loss_normal(depth, viewpoint) / 10.
                loss_mapping += get_loss_mapping_rgbd(self.config, image, depth, viewpoint)
                loss_mapping += deform_reg
                roi_loss_mapping += self._oracle_roi_loss(image, viewpoint)
                viewspace_point_tensor_acm.append(viewspace_point_tensor)
                visibility_filter_acm.append(visibility_filter)
                radii_acm.append(radii)

            scaling = self.gaussians.get_scaling
            isotropic_loss = torch.abs(scaling - scaling.mean(dim=1).view(-1, 1))
            loss_mapping += 10 * isotropic_loss.mean()
            self._backward_with_oracle_roi(loss_mapping, roi_loss_mapping)
            self._mask_oracle_shape_grads()
            self._freeze_dynamic_positions()
            ## Deinsifying / Pruning Gaussians
            with torch.no_grad():
                for idx in range(len(viewspace_point_tensor_acm)):
                    self.gaussians.max_radii2D[visibility_filter_acm[idx]] = torch.max(
                        self.gaussians.max_radii2D[visibility_filter_acm[idx]],
                        radii_acm[idx][visibility_filter_acm[idx]],
                    )
                    self.gaussians.add_densification_stats(
                        viewspace_point_tensor_acm[idx], visibility_filter_acm[idx]
                    )

                update_gaussian = self.iteration_count % self.gaussian_update_every == self.gaussian_update_offset
                if update_gaussian:
                    self.gaussians.densify_and_prune(
                        self.opt_params.densify_grad_threshold,
                        self.gaussian_th,
                        self.gaussian_extent,
                        self.size_threshold,
                    )

                ## Opacity reset
                if (self.iteration_count % self.gaussian_reset) == 0 and (not update_gaussian):
                    Log("Resetting the opacity of non-visible Gaussians")
                    self.gaussians.reset_opacity_nonvisible(visibility_filter_acm)

                if self.deform_net is not None:
                    # BUGFIX (panoramic_4dgs_status.md section 3.6) -- see the matching
                    # comment in initialize_map(): this clips _xyz's own gradient, which is
                    # what actually carries the amplified dxyz->xyz coupling term. The old
                    # clip below only ever touched deform_net's own parameters, a different
                    # tensor, so _xyz's gradient was never bounded at all.
                    torch.nn.utils.clip_grad_norm_([self.gaussians._xyz], max_norm=1.0)
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                if self.deform_optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(self.deform_net.parameters(), max_norm=1.0)
                    self.deform_optimizer.step()
                    self.deform_optimizer.zero_grad(set_to_none=True)
                # self.gaussians.update_learning_rate(self.iteration_count)

    def color_refinement(self, iteration_total):
        Log("Starting color refinement")

        opt_params = []
        for view in self.viewpoints.values():
            opt_params.append({
                    "params": [view.cam_rot_delta],
                    "lr": self.config["opt_params"]["pose_lr"],
                    "name": "rot_{}".format(view.uid)})
            opt_params.append({
                    "params": [view.cam_trans_delta],
                    "lr": self.config["opt_params"]["pose_lr"],
                    "name": "trans_{}".format(view.uid)})
            if self.config["Training"]["compensate_exposure"]:
                opt_params.append({
                        "params": [view.exposure_a],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_a_{}".format(view.uid)})
                opt_params.append({
                        "params": [view.exposure_b],
                        "lr": self.config["opt_params"]["exposure_lr"],
                        "name": "exposure_b_{}".format(view.uid)})
        self.keyframe_optimizers = torch.optim.Adam(opt_params)

        for iteration in (pbar := trange(1, iteration_total + 1)):
            # BUGFIX (panoramic_4dgs_status.md section 3.16): self.iteration_count (read by
            # render_at()'s `ramp` curriculum and by _log_deform_health()'s periodic-print
            # check) was never incremented in this loop -- it stayed frozen at whatever value
            # tracking (initialize_map()/map()) left it at, for all `iteration_total` (30000)
            # steps of this function. Since that means the ramp curriculum was already pinned
            # at 1.0 before color_refinement even starts (init_itr_num=1050 alone exceeds the
            # default ramp_iters=1000), the curriculum never actually acted during this phase
            # -- the one it was built to protect (the original NaN divergence in section 3.4
            # started ~step 360 of this exact loop). Incrementing it here restores both the
            # curriculum's intended effect during this phase and DeformHealth's visibility
            # into it (previously zero DeformHealth log lines ever came from this loop).
            self.iteration_count += 1
            viewpoint_idx_stack = list(self.viewpoints.keys())
            viewpoint_cam_idx = viewpoint_idx_stack.pop(random.randint(0, len(viewpoint_idx_stack) - 1))
            viewpoint_cam = self.viewpoints[viewpoint_cam_idx]
            render_pkg, deform_reg = self.render_at(viewpoint_cam)
            image, depth = render_pkg["render"], render_pkg["depth"]
            image = (torch.exp(viewpoint_cam.exposure_a)) * image + viewpoint_cam.exposure_b

            gt_image = viewpoint_cam.original_image.cuda()
            loss = (1.0 - self.opt_params.lambda_dssim) * l1_loss(image, gt_image) + self.opt_params.lambda_dssim * (1.0 - ssim(image, gt_image))
            loss += get_loss_mapping_rgbd(self.config, image, depth, viewpoint_cam)
            if iteration < 7000:
                loss += self.lambda_dnormal * get_loss_normal(depth, viewpoint_cam)
            else:
                loss += self.lambda_dnormal * get_loss_normal(depth, viewpoint_cam) / 2
            loss += deform_reg
            roi_loss = self._oracle_roi_loss(image, viewpoint_cam)
            total_loss = loss + roi_loss
            if torch.isnan(total_loss):
                with torch.no_grad():
                    xyz = self.gaussians.get_xyz
                    scl = self.gaussians.get_scaling
                    Log(f"NaN loss at iter {iteration}, cam_idx={getattr(viewpoint_cam, 'cam_idx', None)}, "
                        f"tstamp={viewpoint_cam.tstamp}, xyz absmax={xyz.abs().max().item():.3e}, "
                        f"scaling absmax={scl.abs().max().item():.3e}, "
                        f"image nan={torch.isnan(image).any().item()} inf={torch.isinf(image).any().item()}, "
                        f"depth nan={torch.isnan(depth).any().item()} inf={torch.isinf(depth).any().item()} "
                        f"depth absmax={depth[torch.isfinite(depth)].abs().max().item() if torch.isfinite(depth).any() else float('nan'):.3e}, "
                        f"gt_depth absmax={viewpoint_cam.depth.abs().max().item():.3e}", tag="DeformDebug")
                    if self.deform_net is not None:
                        real_t = getattr(
                            viewpoint_cam, 'physical_tstamp',
                            viewpoint_cam.tstamp - 500 * getattr(viewpoint_cam, 'cam_idx', 0))
                        dxyz, drot, dscale, raw = self.deform_net(xyz, real_t)
                        Log(f"dxyz absmax={dxyz.abs().max().item():.3e}, drot absmax={drot.abs().max().item():.3e}, "
                            f"dscale absmax={dscale.abs().max().item():.3e}, raw absmax={raw.abs().max().item():.3e}", tag="DeformDebug")
                break
            self._backward_with_oracle_roi(loss, roi_loss)
            self._mask_oracle_shape_grads()
            self._freeze_dynamic_positions()
            with torch.no_grad():
                if self.deform_net is not None:
                    # BUGFIX (panoramic_4dgs_status.md section 3.6) -- see the matching
                    # comment in initialize_map(): this clips _xyz's own gradient, which is
                    # what actually carries the amplified dxyz->xyz coupling term. The old
                    # clip below only ever touched deform_net's own parameters, a different
                    # tensor, so _xyz's gradient was never bounded at all.
                    torch.nn.utils.clip_grad_norm_([self.gaussians._xyz], max_norm=1.0)
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
                lr = self.gaussians.update_learning_rate(iteration)

                self.keyframe_optimizers.step()
                self.keyframe_optimizers.zero_grad(set_to_none=True)
                if self.deform_optimizer is not None:
                    torch.nn.utils.clip_grad_norm_(self.deform_net.parameters(), max_norm=1.0)
                    self.deform_optimizer.step()
                    self.deform_optimizer.zero_grad(set_to_none=True)
                update_pose(viewpoint_cam)

            if self.use_gui and iteration % 50 == 0:
                self.q_main2vis.put(gui_utils.GaussianPacket(gaussians=clone_obj(self.gaussians)))

            pbar.set_description(
                f"Global GS Refinement lr {lr:.3E} loss {total_loss.item():.3f}")

        Log("Map refinement done")
