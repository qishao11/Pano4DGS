import json
import os

import cv2
import numpy as np
import torch
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from utils.utils import Log
from gaussian.renderer import render
from gaussian.utils.loss_utils import ssim, psnr
from gaussian.utils.camera_utils import Camera
from gaussian.deform.deform_net import apply_deform
from gaussian.deform.oracle_motion_gate import apply_motion_gate
from gaussian.deform.object_render import object_se3_overrides
from gaussian.deform.gaussian_4d import slice_at_time


def _render_maybe_deformed(frame, gaussians, background, deform_net,
                           object_trajectories=None, observed_times=None,
                           gaussian_4d=False, per_observation=False):
    """Render at the physical time shared by all views of a rig/cubemap instant.

    ``object_trajectories``/``observed_times`` select the object_se3 path, which
    must match what training rendered -- evaluating those Gaussians without their
    trajectory would leave every dynamic row at whichever time it was seeded.
    """
    real_t = getattr(
        frame, 'physical_tstamp',
        frame.tstamp - 500 * getattr(frame, 'cam_idx', 0))
    if gaussian_4d:
        moved_xyz, faded_opacity, _ = slice_at_time(
            gaussians.get_xyz, gaussians.get_opacity, gaussians.dynamic_source_time,
            gaussians.get_time_scale, gaussians.get_velocity, real_t,
            gaussians.dynamic_score, per_observation=per_observation)
        return render(frame, gaussians, background,
                      means3D_override=moved_xyz,
                      opacities_override=faded_opacity)
    if object_trajectories is not None:
        return render(
            frame, gaussians, background,
            **object_se3_overrides(
                gaussians, object_trajectories, observed_times, real_t,
                tolerance=gaussians.oracle_time_tolerance))
    opacity_at_time = gaussians.get_opacity_at_time(real_t)
    if deform_net is None:
        return render(
            frame, gaussians, background,
            opacities_override=opacity_at_time)
    dxyz, drot, dscale, _raw = deform_net(gaussians.get_xyz, real_t)
    if gaussians.oracle_dynamic_gate:
        dxyz, drot, dscale = apply_motion_gate(
            dxyz, drot, dscale, gaussians.dynamic_score,
            translation_only=gaussians.oracle_translation_only)
    new_xyz, new_scaling_log, new_rotation = apply_deform(
        gaussians.get_xyz, gaussians._scaling, gaussians.get_rotation, dxyz, drot, dscale,
    )
    return render(
        frame, gaussians, background,
        means3D_override=new_xyz,
        scales_override=gaussians.scaling_activation(new_scaling_log),
        rotations_override=torch.nn.functional.normalize(new_rotation),
        opacities_override=opacity_at_time,
    )


def eval_rendering(
    gtimages,
    gtdepthdir,
    traj,
    gaussians,
    save_dir,
    background,
    projection_matrix,
    K,
    kf_idx,
    iteration="final",
):
    gtdepths = sorted(os.listdir(gtdepthdir)) if gtdepthdir is not None else None
    psnr_array, ssim_array, lpips_array, l1_array = [], [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")
    
    image_save_dir = f'{save_dir}/renders/image_{iteration}'
    depth_save_dir = f'{save_dir}/renders/depth_{iteration}'
    # vis_save_dir = f'{save_dir}/renders/vis_{iteration}'  
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(depth_save_dir, exist_ok=True)
    # os.makedirs(vis_save_dir, exist_ok=True)
    
    for i, (idx, image) in enumerate(gtimages.items()):
        if idx % 5 != 0 and idx not in kf_idx and i != len(gtimages) - 1:
            continue
        frame = Camera.init_from_tracking(image.squeeze()/255.0, None, None, traj[idx], idx, projection_matrix, K)
        gtimage = frame.original_image.cuda()

        rendering = render(
            frame, gaussians, background,
            opacities_override=gaussians.get_opacity_at_time(idx))
        image = torch.clamp(rendering["render"], 0.0, 1.0)
        depth = rendering["depth"].detach().squeeze().cpu().numpy()

        if gtdepthdir is not None:
            gtdepth = cv2.imread(os.path.join(gtdepthdir, gtdepths[idx]), cv2.IMREAD_ANYDEPTH) / 6553.5 # 1000.
            gtdepth = cv2.resize(gtdepth, (depth.shape[-1], depth.shape[-2]), interpolation=cv2.INTER_NEAREST)
            invalid = gtdepth <= 0
            depth[invalid] = 0

        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f'{image_save_dir}/{idx:06d}.jpg', pred)
        cv2.imwrite(f'{depth_save_dir}/{idx:06d}.png', np.clip(depth*6553.5, 0, 65535).astype(np.uint16))
        # vis = np.concatenate((pred, cv2.imread(f'{save_dir}/renders/depth_{iteration}/{idx:06d}.png')), axis=0)
        # cv2.imwrite(f'{vis_save_dir}/{idx:06d}.jpg', vis)

        if gtdepthdir is not None and idx in kf_idx:
            l1_array.append(np.abs(gtdepth[depth > 0] - depth[depth > 0]).mean().item()) 

        # if idx in kf_idx:
        #     continue
        mask = gtimage > 0
        psnr_score = psnr((image[mask]).unsqueeze(0), (gtimage[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gtimage).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gtimage).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    output["mean_l1"] = float(np.mean(l1_array)) if l1_array else 0

    Log(f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}, depth l1: {output["mean_l1"]}', tag="Eval")

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    os.makedirs(psnr_save_dir, exist_ok=True)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output

def eval_rendering_kf(
    viewpoints,
    gaussians,
    save_dir,
    background,
    iteration="final",
    deform_net=None,
    object_trajectories=None,
    observed_times=None,
    gaussian_4d=False,
    per_observation=False,
):
    psnr_array, ssim_array, lpips_array = [], [], []
    per_cam_psnr, per_cam_ssim, per_cam_lpips = {}, {}, {}
    cal_lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to("cuda")
    
    image_save_dir = f'{save_dir}/renders/image_{iteration}'
    depth_save_dir = f'{save_dir}/renders/depth_{iteration}'

    # per-camera frame counter so each camera's renders are numbered independently
    cam_counters = {}
    for frame in viewpoints.values():
        gtimage = frame.original_image.cuda()

        rendering = _render_maybe_deformed(
            frame, gaussians, background, deform_net,
            object_trajectories=object_trajectories,
            observed_times=observed_times,
            gaussian_4d=gaussian_4d,
            per_observation=per_observation)
        image = (torch.exp(frame.exposure_a)) * rendering["render"] + frame.exposure_b
        image = torch.clamp(image, 0.0, 1.0)
        depth = rendering["depth"].detach().squeeze().cpu().numpy()

        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)

        cam_idx = getattr(frame, 'cam_idx', 0)
        cam_image_dir = f'{image_save_dir}/cam{cam_idx}'
        cam_depth_dir = f'{depth_save_dir}/cam{cam_idx}'
        os.makedirs(cam_image_dir, exist_ok=True)
        os.makedirs(cam_depth_dir, exist_ok=True)
        save_idx = cam_counters.get(cam_idx, 0)
        cam_counters[cam_idx] = save_idx + 1
        cv2.imwrite(f'{cam_image_dir}/{save_idx:06d}.jpg', pred)
        cv2.imwrite(f'{cam_depth_dir}/{save_idx:06d}.png', np.clip(depth*6553.5, 0, 65535).astype(np.uint16))

        mask = gtimage > 0
        psnr_score = psnr((image[mask]).unsqueeze(0), (gtimage[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gtimage).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gtimage).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())
        per_cam_psnr.setdefault(cam_idx, []).append(psnr_score.item())
        per_cam_ssim.setdefault(cam_idx, []).append(ssim_score.item())
        per_cam_lpips.setdefault(cam_idx, []).append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))
    # per-camera breakdown -- the overall mean above averages across every camera in a
    # multi-camera/cubemap rig, which can hide a large primary-vs-auxiliary-view quality gap
    # (see panoramic_4dgs_status.md section 3.11: the mean alone wasn't enough to tell whether
    # a quality gap came from the ERP->cubemap conversion itself or from auxiliary views just
    # being under-trained relative to the primary tracked camera).
    output["per_cam"] = {
        str(cam_idx): {
            "mean_psnr": float(np.mean(per_cam_psnr[cam_idx])),
            "mean_ssim": float(np.mean(per_cam_ssim[cam_idx])),
            "mean_lpips": float(np.mean(per_cam_lpips[cam_idx])),
            "n": len(per_cam_psnr[cam_idx]),
        }
        for cam_idx in sorted(per_cam_psnr)
    }

    Log(f'kf mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}', tag="Eval")

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    os.makedirs(psnr_save_dir, exist_ok=True)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result_kf.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output

def save_gaussians(gaussians, name, iteration, final=False):
    if name is None:
        return
    if final:
        point_cloud_path = os.path.join(name, "point_cloud/final")
    else:
        point_cloud_path = os.path.join(
            name, "point_cloud/iteration_{}".format(str(iteration))
        )
    gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
    print('saved to ', point_cloud_path)
