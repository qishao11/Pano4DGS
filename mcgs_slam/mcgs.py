import os
import cv2
import torch
import lietorch
import numpy as np

from droid_net import DroidNet
from depth_video import DepthVideo
from motion_filter import MotionFilter
from droid_frontend import DroidFrontend
from droid_backend import DroidBackend

from collections import OrderedDict
from torch.multiprocessing import Process
from tqdm import trange

from gs_backend import GSBackEnd
from gaussian.deform.dynamic_modes import (
    DYNAMIC_MODE_DEFORM,
    resolve_dynamic_mode,
)
from temporal import SourceTimeNormalizer, source_timestamps_for_indices
from utils.utils import load_config


def _depth_from_disparity(disparity):
    """Depth from a stored disparity map, with 0 meaning "no measurement".

    ``disps_prior_up`` holds ``1/depth`` where a depth was supplied and exactly 0
    where none was, so a plain reciprocal would turn every unmeasured pixel into
    an infinity that then propagates silently. Zeros are kept as zeros; the
    seeding treats them as missing.
    """
    return torch.where(disparity > 0, 1.0 / disparity.clamp(min=1e-8),
                       torch.zeros_like(disparity))


class Mcgs:
    def __init__(self, args, video=None):
        super(Mcgs, self).__init__()
        self.load_weights(args.weights)
        self.args = args
        self.config = load_config(args.config)
        if getattr(args, 'deform', False):
            explicit_mode = self.config["Training"].get("dynamic_mode")
            if explicit_mode not in (None, DYNAMIC_MODE_DEFORM):
                raise ValueError(
                    f"--deform conflicts with configured dynamic_mode "
                    f"'{explicit_mode}'")
            self.config["Training"]["deform"] = True
            self.config["Training"]["dynamic_mode"] = DYNAMIC_MODE_DEFORM
        self.dynamic_mode = resolve_dynamic_mode(self.config["Training"])
        self.config["Training"]["dynamic_mode"] = self.dynamic_mode
        self.scale_factor = 0.2
        time_cfg = self.config["Training"].get("deform_cfg", {})
        self.source_time = SourceTimeNormalizer(
            scale=time_cfg.get("source_time_scale", 1.0))

        # store images, depth, poses, intrinsics (shared between processes)
        if video is None:
            self.video = DepthVideo(args, args.image_size, args.buffer, stereo=args.stereo)
        else:
            self.video = video

        # filter incoming frames so that there is enough motion
        self.filterx = MotionFilter(self.net, self.video, thresh=args.filter_thresh, args=args)

        # frontend process
        self.frontend = DroidFrontend(self.net, self.video, self.args)

        # backend process
        self.backend = DroidBackend(self.net, self.video, self.args)
        
        # 3dgs
        # Panoramic geometry lives on args (options.py::_load_equirect_configs)
        # but the backend only receives config, so mirror it across. Without this
        # the ERP panorama path silently falls back to a default face order and
        # a 90-degree fov, which stitches faces into the wrong directions for any
        # calib that differs -- and looks plausible rather than failing.
        if getattr(args, "camera", "pinhole") == "equirect":
            self.config["Training"]["cubemap_faces"] = list(args.cubemap_faces)
            self.config["Training"]["face_size"] = int(args.face_size)
            self.config["Training"]["fov"] = float(args.fov)
        self.gs = GSBackEnd(self.config, self.args.output, args.gsvis)

        # visualizer
        if args.vis:
            from visualization import droid_visualization
            self.visualizer = Process(target=droid_visualization, args=(self.video,))
            self.visualizer.start()

    def load_weights(self, weights):
        """ load trained model weights """

        self.net = DroidNet()
        state_dict = OrderedDict([
            (k.replace("module.", ""), v) for (k, v) in torch.load(weights).items()])

        state_dict["update.weight.2.weight"] = state_dict["update.weight.2.weight"][:2]
        state_dict["update.weight.2.bias"] = state_dict["update.weight.2.bias"][:2]
        state_dict["update.delta.2.weight"] = state_dict["update.delta.2.weight"][:2]
        state_dict["update.delta.2.bias"] = state_dict["update.delta.2.bias"][:2]

        self.net.load_state_dict(state_dict)
        self.net.to("cuda:0").eval()
    
    def _scale_poses(self, poses, scale_factor=0.2):
        scaled_poses = poses.clone()
        scaled_poses[:, :3] *= scale_factor
        return scaled_poses

    def _source_time_tensor(self, viz_idx):
        """Return normalized capture times for GS, separate from view IDs."""
        indices = viz_idx.detach().cpu().reshape(-1).tolist()
        timestamps = source_timestamps_for_indices(self.video.kf_stamps, indices)
        normalized = self.source_time.normalize_many(timestamps)
        # Keep packet precision until GaussianModel stores small relative values.
        return torch.tensor(normalized, dtype=torch.float64)

    def call_gs(self, viz_idx, dposes=None, dscale=None):
        num_gs_cameras = max(1, int(self.video.multi) - 1) if self.video.multi else 1
        view_ids = self.video.tstamp[viz_idx].to(device='cpu')
        physical_tstamp = self._source_time_tensor(viz_idx)
        data = {'viz_idx':  viz_idx.to(device='cpu'),
                'tstamp':   view_ids,
                'physical_tstamp': physical_tstamp,
                'time_metadata': self.source_time.state_dict(),
                'poses':    self._scale_poses(self.video.poses[viz_idx].to(device='cpu'), scale_factor = self.scale_factor),
                'images':   self.video.images_up[viz_idx.cpu()],
                'normals':  self.video.normals[viz_idx.cpu()],
                'depths':   self.scale_factor / self.video.disps_up[viz_idx.cpu()].to(device='cpu'),
                # The ground-truth/sensor depth, kept separate from BA's answer so
                # the seeding can prefer it for the moving object, whose multi-view
                # constraint BA has no right to trust (section 3.53).
                'prior_depths': _depth_from_disparity(
                    self.video.disps_prior_up[viz_idx.cpu()].to(device='cpu')),
                'intrinsics':   self.video.intrinsics[viz_idx].to(device='cpu')[:, 0, :4] * 8,
                'cam_idx':  0,
                'finalize_batch': num_gs_cameras == 1,
                'map_iterations': 10 * num_gs_cameras,
                'pose_updates':  dposes.to(device='cpu') if dposes is not None else None,
                'scale_updates': dscale.to(device='cpu') if dscale is not None else None}

        self.gs.process_track_data(data)
        
        if self.video.multi:
            for i in range(1, self.video.multi-1):
                
                T_ci_c0 = self.video.T_ci_c0[i]
                
                data = {'viz_idx':  viz_idx.to(device='cpu'),
                        'tstamp':   view_ids + 500 * i,
                        'physical_tstamp': physical_tstamp,
                        'time_metadata': self.source_time.state_dict(),
                        'poses':    self._scale_poses((T_ci_c0.cpu() * lietorch.SE3(self.video.poses[viz_idx].to(device='cpu')[None])).data[0],
                                                        scale_factor = self.scale_factor),
                        'images':   self.video.images_up_list[i][viz_idx.cpu()],
                        'normals':  self.video.normals_list[i][viz_idx.cpu()],
                        'depths':   self.scale_factor / self.video.disps_up_list[i][viz_idx.cpu()].to(device='cpu'),
                        'intrinsics':   self.video.intrinsics[viz_idx].to(device='cpu')[:, i + 1, :4] * 8,
                        'cam_idx':  i,
                        'finalize_batch': i == num_gs_cameras - 1,
                        'map_iterations': 10 * num_gs_cameras,
                        'pose_updates':  dposes.to(device='cpu') if dposes is not None else None,
                        'scale_updates': dscale.to(device='cpu') if dscale is not None else None}
            
                self.gs.process_track_data(data)
    
    def call_global_gs(self, viz_idx, dposes=None, dscale=None):
        
        multi_cam_data = {
            'viz_idx': [],
            'tstamp': [],
            'physical_tstamp': [],
            'poses': [],
            'images': [],
            'normals': [],
            'depths': [],
            'prior_depths': [],
            'intrinsics': []
        }

        multi_cam_data['viz_idx'].append(viz_idx.to(device='cpu'))
        view_ids = self.video.tstamp[viz_idx].to(device='cpu')
        physical_tstamp = self._source_time_tensor(viz_idx)
        multi_cam_data['tstamp'].append(view_ids)
        multi_cam_data['physical_tstamp'].append(physical_tstamp)
        multi_cam_data['poses'].append(
            self._scale_poses(self.video.poses[viz_idx].to(device='cpu'), scale_factor = self.scale_factor)
        )
        multi_cam_data['images'].append(self.video.images_up[viz_idx.cpu()])
        multi_cam_data['normals'].append(self.video.normals[viz_idx.cpu()])
        multi_cam_data['depths'].append(self.scale_factor / self.video.disps_up[viz_idx.cpu()].to(device='cpu'))
        multi_cam_data['prior_depths'].append(
            _depth_from_disparity(self.video.disps_prior_up[viz_idx.cpu()].to(device='cpu')))
        multi_cam_data['intrinsics'].append(self.video.intrinsics[viz_idx].to(device='cpu')[:, 0, :4] * 8)

        if self.video.multi:
            for i in range(1, self.video.multi - 1):
                T_ci_c0 = self.video.T_ci_c0[i]

                multi_cam_data['viz_idx'].append(viz_idx.to(device='cpu'))
                multi_cam_data['tstamp'].append(view_ids + 500 * i)
                multi_cam_data['physical_tstamp'].append(physical_tstamp)
                multi_cam_data['poses'].append(self._scale_poses((T_ci_c0.cpu() * lietorch.SE3(self.video.poses[viz_idx].to(device='cpu')[None])).data[0], 
                                                        scale_factor = self.scale_factor))
                multi_cam_data['images'].append(self.video.images_up_list[i][viz_idx.cpu()])
                multi_cam_data['normals'].append(self.video.normals_list[i][viz_idx.cpu()])
                multi_cam_data['depths'].append(self.scale_factor / self.video.disps_up_list[i][viz_idx.cpu()].to(device='cpu'))
                multi_cam_data['prior_depths'].append(_depth_from_disparity(
                    self.video.disps_prior_up_list[i][viz_idx.cpu()].to(device='cpu')))
                multi_cam_data['intrinsics'].append(self.video.intrinsics[viz_idx].to(device='cpu')[:, i + 1, :4] * 8)
                
        final_data = {
            k: torch.cat(v, dim=0) if isinstance(v[0], torch.Tensor) else v
            for k, v in multi_cam_data.items()
        }
        
        final_data['pose_updates'] = lietorch.SE3(self._scale_poses(dposes.to(device='cpu').data, self.scale_factor)) if dposes is not None else None
        final_data['scale_updates'] = dscale.to(device='cpu') if dscale is not None else None
        final_data['time_metadata'] = self.source_time.state_dict()
            
        self.gs.process_global_track_data(final_data, self.video.multi - 1)

    def track(self, t, tstamp, image, intrinsics, measurement_depth=None):
        """ main thread - update map """

        with torch.no_grad():
            # check there is enough motion
            self.filterx.track(t, tstamp, image, intrinsics, measurement_depth)

            # local bundle adjustment
            viz_idx = self.frontend()

            if self.video.counter.value >= (self.video.buffer - 15):
                window = 35
                self.frontend.release_buffer(window=window)
                self.video.release_buffer(window=window)
        
        if len(viz_idx):
            self.call_gs(viz_idx)

    def save_kf_poses(self, args, video, filename='traj_mcgs.txt'):
        N = video.total_counter
        kf_poses = lietorch.SE3(video.globuf.poses_all[:N][None]).inv().data.cpu().numpy()[0]    # poses_wc
        kf_stamps = sorted(video.globuf.kf_stamps_all.values())
        traj_file = os.path.join(args.output, filename)
        with open(traj_file, 'w') as f:
            for stamp, pose in zip(kf_stamps, kf_poses):
                pose = [stamp] + list(pose)
                pose = [str(i) for i in pose]
                pose = " ".join(pose)
                f.write(pose + "\n")
        print('saved pose file to', traj_file)

    def global_pose_ba(self):
        """ terminate the visualization process, return poses [t, q] """
        torch.cuda.empty_cache()

        for iter in trange(5, desc='Global BA outer loop'):
            print("#"*64, f" iter {iter} ", "#"*64)
            self.backend.pose_ba(iter, 6)
            torch.cuda.empty_cache()

        # self.video.globuf.poses_all = self.video.poses.cpu()
        # self.video.globuf.disps_all = self.video.disps.cpu()
        # self.video.globuf.images_all = self.video.images
        # if self.video.multi:
        #     self.video.globuf.images_all_list = self.video.images_list
        #     self.video.globuf.disps_all_list = [d.cpu() for d in self.video.disps_list]
    
    def terminate(self):
        
        del self.frontend

        # global bundle adjustment
        poses_pre = self.video.poses[:self.video.counter.value].clone()
        
        self.global_pose_ba()
        del self.backend
        
        poses_pos = self.video.poses[:self.video.counter.value].clone()
        
        dposes = lietorch.SE3(poses_pos).inv() * lietorch.SE3(poses_pre)
        dscale = torch.ones(self.video.counter.value, 1)
        torch.cuda.empty_cache()

        # final refinement
        self.call_global_gs(torch.arange(0, self.video.counter.value, device='cuda'), dposes, dscale)
        self.gs.finalize()
        
        self.gs.eval_rendering_kf()
