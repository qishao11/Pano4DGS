import os
import cv2
import yaml
import argparse
import numpy as np
import torch

import cubemap


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPU', default=0, type=int)

    parser.add_argument("--imagedir", type=str, help="path to color image directory", nargs='+')
    parser.add_argument("--posefile", type=str, help="path to pose file")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")
    parser.add_argument("--stereo", default=True, type=bool, help="stereo")
    parser.add_argument("--output", default="output", type=str, help="output file")
    parser.add_argument("--early_stop", default=-1, type=int, help="stoping frame")

    parser.add_argument("--weights", default=os.path.dirname(__file__) + "/../pretrained_models/droid.pth")
    parser.add_argument("--config", default=os.path.dirname(__file__) + "/../config/config.yaml")
    parser.add_argument("--buffer", type=int, default=200)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--jdsa", action="store_true")
    parser.add_argument("--rgbd", action="store_true")
    parser.add_argument("--prgbd", action="store_true", help="pseudo rgbd")
    parser.add_argument("--depth_prior_mode", default="full",
                        choices=("full", "scale_only"),
                        help="what to keep from the monocular depth prior. "
                             "'scale_only' replaces it with its median, keeping "
                             "the scale anchor BA needs while discarding the "
                             "structure, which measured negatively correlated "
                             "with ground truth (section 3.43/3.45)")
    parser.add_argument("--gsvis", action="store_true")
    parser.add_argument("--deform", action="store_true", help="4DGS: enable time-conditioned Gaussian deformation field (see gaussian/deform/deform_net.py)")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=10, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=20.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)

    args = parser.parse_args()
    args.multi = len(args.imagedir) if len(args.imagedir) > 2 else False
    return args


def load_configs(args):
    params = yaml.load(open(args.calib, 'r'), Loader=yaml.FullLoader)
    args.camera = params['camera']
    print("Camera model:", args.camera)

    if args.camera == 'equirect':
        return _load_equirect_configs(args, params)

    args.calib = np.array(params['intrinsic'])
    args.base = np.array(params['baseline'])
    if args.multi:
        args.T_cami_cam0 = torch.as_tensor(params['T_cami_cam0'], dtype=torch.float, device="cuda")

    args.timescale = float(params['timescale'])

    if 'ht' in params and 'wd' in params:
        args.ht, args.wd = params['ht'], params['wd']
    else:
        limit = 384 * 512
        h0, w0 = cv2.imread(os.path.join(args.imagedir[0], os.listdir(args.imagedir[0])[0])).shape[:2]
        ht = int(h0 * np.sqrt((limit) / (h0 * w0)))
        wd = int(w0 * np.sqrt((limit) / (h0 * w0)))
        args.ht, args.wd = ht-ht % 8, wd-wd % 8
    args.image_size = [args.ht, args.wd]
    print(f'Input image resolution {args.ht} x {args.wd}')

    return args


def _load_equirect_configs(args, params):
    """Panoramic (equirectangular) camera: decompose each ERP frame into cubemap-face
    virtual pinhole cameras and feed them through the existing multi-camera rig pipeline.
    See panoramic_support_feasibility.md and mcgs_slam/cubemap.py for the design.

    calib slot layout matches the existing rig convention (mcgs.py::call_gs /
    depth_video.py): [front (cam0, drives tracking), front-duplicate (stereo-pair
    placeholder slot that motion_filter.py always strips), <extra faces...>].
    """
    faces = params.get('faces', ['front', 'right', 'back', 'left'])
    assert faces[0] == 'front', "first equirect face must be 'front' (drives DROID tracking)"
    extra_faces = faces[1:]
    face_size = int(params['face_size'])
    fov = float(params.get('fov', 90.0))

    args.cubemap_faces = extra_faces
    args.face_size = face_size
    args.fov = fov

    fx, fy, cx, cy = cubemap.face_intrinsics(face_size, fov)
    n_slots = 2 + len(extra_faces)  # front + stereo-placeholder(front dup) + extras
    args.calib = np.tile(np.array([fx, fy, cx, cy]), (n_slots, 1))
    args.multi = n_slots

    T_rows = [[0., 0., 0., 0., 0., 0., 1.]]  # stereo-placeholder slot (front dup, identity)
    T_rows += [cubemap.face_rotation_quat(f) for f in extra_faces]
    args.T_cami_cam0 = torch.as_tensor(T_rows, dtype=torch.float, device="cuda")

    args.base = np.array(params.get('baseline', [0., 0., 0., 0., 0., 0., 1.]))
    args.timescale = float(params['timescale'])
    args.ht = args.wd = face_size
    args.image_size = [args.ht, args.wd]
    print(f'Equirect cubemap: front + {extra_faces}, face_size={face_size}, fov={fov}')

    return args
