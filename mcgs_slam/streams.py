import os
import re
import numpy as np
import cv2
import torch

import cubemap

base = 8


def map_filename(x):
    return float(re.findall(r"[+]?(?:\d*\.\d+|\d+)", x)[-1])


def resize_depth(depth, h1, w1):
    """Nearest-neighbour, for the same reason erp_depth_to_cubemap uses it:
    interpolating across a depth edge invents a surface that is not there."""
    depth = cv2.resize(depth, (w1, h1), interpolation=cv2.INTER_NEAREST)
    return torch.as_tensor(depth[:h1 - h1 % base, :w1 - w1 % base])


def resize(image, h1, w1):
    image = cv2.resize(image, (w1, h1))
    image = image[:h1-h1 % base, :w1-w1 % base]
    image = torch.as_tensor(image).permute(2, 0, 1)
    return image


def image_stream(imagedirs, calib, args):
    """ image generator """
    if getattr(args, 'camera', 'pinhole') == 'equirect':
        yield from equirect_cubemap_stream(imagedirs, calib, args)
        return

    image_lists = []
    for imagedir in imagedirs:
        image_lists.append(sorted(os.listdir(imagedir), key=map_filename)[::args.stride])

    h1, w1 = args.ht, args.wd
    for t in range(len((image_lists[0]))):
        images = []
        for i, imagelist in enumerate(image_lists):
            timestamp = float(re.findall(r"[+]?(?:\d*\.\d+|\d+)", imagelist[t])[-1]) / args.timescale
            if i == 0:
                t0 = timestamp
            else:
                assert abs(timestamp - t0) < 2e-2, f"{timestamp} vs. {t0}"

            image = cv2.imread(os.path.join(imagedirs[i], imagelist[t]))
            
            # TODO
            if calib.shape[1] > 4:
                K = np.array([[calib[i][0], 0, calib[i][2]], [0, calib[i][1], calib[i][3]], [0, 0, 1]])
                image = cv2.undistort(image, K, calib[i][4:])
            
            h0, w0, _ = image.shape
            image = resize(image, h1, w1)
            images.append(image)

        images = torch.stack(images, dim=0)

        intrinsics = torch.zeros((len(imagedirs), 8))
        intrinsics[:, :4] = torch.tensor(calib[:, :4])

        intrinsics[:, [0, 2]] *= (w1 / w0)
        intrinsics[:, [1, 3]] *= (h1 / h0)

        if t == 0:
            print(
                f'Orig size: {h0}-{w0}; Down to size: {h1}-{w1}\nInput calib: {list(calib)}\nAdapt calib: {list(intrinsics.numpy())}')

        # pinhole path has no ground-truth depth source
        yield t, images, intrinsics, timestamp, None


def equirect_cubemap_stream(imagedirs, calib, args):
    """ Panoramic image generator: splits each equirectangular (ERP) frame into cubemap-face
    virtual pinhole images and yields them in the same [front, front-dup(stereo placeholder),
    extra faces...] slot layout that options.py::_load_equirect_configs() built `calib`/
    `args.T_cami_cam0` for, so the rest of the pipeline (motion_filter/droid_frontend/
    gs_backend) needs no changes. `imagedirs` is a single-element list: one directory of ERP
    frames. """
    erp_dir = imagedirs[0]
    frame_list = sorted(os.listdir(erp_dir), key=map_filename)[::args.stride]
    # Ground-truth depth, if the synthetic generator wrote it next to the frames
    # (tools/make_synthetic_erp_room.py --depth). Only used with --rgbd; it exists
    # to test whether monocular depth is the accuracy bottleneck (section 3.39).
    depth_dir = erp_dir.rstrip('/') + '_depth'
    has_depth = os.path.isdir(depth_dir)

    extra_faces = list(args.cubemap_faces)
    unique_faces = ['front'] + extra_faces
    slot_faces = ['front', 'front'] + extra_faces  # slot 1 = stereo placeholder (front dup)

    h1 = w1 = args.face_size

    for t, fname in enumerate(frame_list):
        timestamp = float(re.findall(r"[+]?(?:\d*\.\d+|\d+)", fname)[-1]) / args.timescale

        erp_img = cv2.imread(os.path.join(erp_dir, fname))
        face_imgs = cubemap.erp_to_cubemap(erp_img, args.face_size, faces=unique_faces, fov_deg=args.fov)

        images = torch.stack([resize(face_imgs[f], h1, w1) for f in slot_faces], dim=0)

        depths = None
        if has_depth:
            stem = os.path.splitext(fname)[0]
            depth_path = os.path.join(depth_dir, stem + '.npy')
            if os.path.exists(depth_path):
                erp_depth = np.load(depth_path)
                face_depths = cubemap.erp_depth_to_cubemap(
                    erp_depth, args.face_size, faces=unique_faces, fov_deg=args.fov)
                depths = torch.stack(
                    [resize_depth(face_depths[f], h1, w1) for f in slot_faces], dim=0)

        intrinsics = torch.zeros((len(slot_faces), 8))
        intrinsics[:, :4] = torch.tensor(calib[:, :4])

        if t == 0:
            erp_h, erp_w = erp_img.shape[:2]
            min_w = cubemap.min_erp_width_for_face(args.face_size, args.fov)
            if erp_w < min_w:
                print(f'WARNING: ERP source resolution ({erp_w}x{erp_h}) is below what '
                      f'face_size={args.face_size}/fov={args.fov} needs at the face center '
                      f'(>= {min_w}px wide) -- cv2.remap() will upsample-interpolate that '
                      f'region, costing real sharpness (see panoramic_4dgs_status.md section '
                      f'3.12, ~4.4dB PSNR lost at this exact mismatch). Raise the ERP capture/'
                      f'render resolution, or lower face_size/fov, to fix.')
            print(
                f'Equirect ERP -> cubemap faces {slot_faces}, face size {h1}x{w1}\nAdapt calib: {list(intrinsics.numpy())}')

        yield t, images, intrinsics, timestamp, depths
