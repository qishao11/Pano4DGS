#!/usr/bin/env python3
"""Generate a synthetic ERP (equirectangular) video of a camera moving through a textured
box room, for end-to-end integration testing of the panoramic (cubemap) pipeline without
needing a real panoramic dataset.

This is a *plumbing* test, not a photometric-accuracy test: the room geometry gives real
depth/parallax (so DROID-SLAM tracking has something real to solve), and each wall has a
distinct, richly-textured pattern (so feature tracking has something to grab onto). It does
NOT validate reconstruction quality against ground truth beyond "did tracking/mapping run to
completion without diverging".

Camera convention matches mcgs_slam/cubemap.py: X-right, Y-down, Z-forward. World frame is
defined identically (no separate world/camera convention), camera orientation is a yaw-only
rotation about the world Y axis plus small pitch/roll jitter.

Optional `--dynamic_object`: adds a sphere that moves independently of the room's static
geometry/texture. `--dynamic_motion` selects a monotonic lateral sweep, a direction-reversing
bounce, or a lateral-plus-depth diagonal trajectory. This exists specifically to
validate DeformNet against *genuine* motion (panoramic_4dgs_status.md section 3.9's leading
open question -- P4a/2.2 never actually confirmed DeformNet helps on real dynamic content,
only that it doesn't hurt on an essentially-static clip). The sphere uses a plain, low-
frequency pattern (not the walls' deliberately adversarial checkerboard) so the test isolates
"can DeformNet track real motion" from "does DeformNet cope with adversarial high-frequency
texture" (section 3.5/3.7's finding) -- those are two different questions.

Usage:
    python tools/make_synthetic_erp_room.py --out data/synth_erp_room --nframes 20
    python tools/make_synthetic_erp_room.py --out data/synth_erp_room_dynamic --nframes 20 --dynamic_object
    python tools/make_synthetic_erp_room.py --out data/synth_erp_room_dynamic_bounce --nframes 20 --dynamic_object --dynamic_motion bounce_x
"""
import os
import argparse

import numpy as np
import cv2

HX, HY, HZ = 4.0, 2.2, 4.0  # room half-extents (X, Y, Z)
TEX_SIZE = 512


def make_natural_wall_texture(seed, base_hue):
    """Lower-frequency wall texture, for monocular-depth friendliness.

    The default make_wall_texture() is deliberately adversarial -- a 16x16
    checkerboard plus random blobs -- which suits feature tracking but is exactly
    the kind of image a monocular depth network has never seen. Measured on the
    default room, metric3d's depth correlates *negatively* with ground truth
    (-0.28 to -0.50): not a scale error, a complete OOD failure
    (panoramic_4dgs_status.md section 3.43).

    This variant keeps mid-frequency structure so tracking still has corners to
    latch onto, but drops the high-frequency checkerboard and adds a smooth
    vertical gradient, which is the sort of shading cue depth networks rely on.
    """
    rng = np.random.RandomState(seed)
    hsv = np.zeros((TEX_SIZE, TEX_SIZE, 3), dtype=np.uint8)
    hsv[..., 0] = base_hue
    hsv[..., 1] = 90
    # smooth vertical shading gradient
    hsv[..., 2] = np.linspace(120, 220, TEX_SIZE, dtype=np.uint8)[:, None]
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # a few large, soft patches -- mid-frequency, trackable, not adversarial
    for _ in range(6):
        c = tuple(int(v) for v in rng.randint(90, 210, size=3))
        r = rng.randint(60, 130)
        cx, cy = rng.randint(0, TEX_SIZE), rng.randint(0, TEX_SIZE)
        cv2.circle(img, (cx, cy), r, c, -1)
    img = cv2.GaussianBlur(img, (31, 31), 0)

    # sparse sharp marks so feature tracking still has corners
    for _ in range(12):
        c = tuple(int(v) for v in rng.randint(0, 255, size=3))
        cx, cy = rng.randint(20, TEX_SIZE - 20), rng.randint(20, TEX_SIZE - 20)
        cv2.rectangle(img, (cx - 9, cy - 9), (cx + 9, cy + 9), c, -1)
    return img


def make_wall_texture(seed, base_hue):
    rng = np.random.RandomState(seed)
    hsv = np.zeros((TEX_SIZE, TEX_SIZE, 3), dtype=np.uint8)
    hsv[..., 0] = base_hue
    hsv[..., 1] = 160
    hsv[..., 2] = 200
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # checkerboard for dense low-level texture / grid-line cues
    cell = TEX_SIZE // 16
    for i in range(16):
        for j in range(16):
            if (i + j) % 2 == 0:
                y0, y1 = i * cell, (i + 1) * cell
                x0, x1 = j * cell, (j + 1) * cell
                img[y0:y1, x0:x1] = (img[y0:y1, x0:x1].astype(np.int32) * 0.6).astype(np.uint8)

    # random colored blobs for distinctive, non-repeating trackable features
    for _ in range(25):
        c = tuple(int(v) for v in rng.randint(0, 255, size=3))
        r = rng.randint(8, 28)
        cx, cy = rng.randint(0, TEX_SIZE), rng.randint(0, TEX_SIZE)
        cv2.circle(img, (cx, cy), r, c, -1)

    # thin border so face seams are visually identifiable during debugging
    cv2.rectangle(img, (0, 0), (TEX_SIZE - 1, TEX_SIZE - 1), (255, 255, 255), 3)
    return img


SPHERE_RADIUS = 0.6
SPHERE_TEX_SIZE = 256
_SPHERE_TEX = None  # populated lazily


def make_sphere_texture():
    """Plain, low-frequency pattern (a few broad color bands), deliberately NOT the walls'
    high-frequency checkerboard -- see module docstring."""
    global _SPHERE_TEX
    if _SPHERE_TEX is None:
        img = np.full((SPHERE_TEX_SIZE, SPHERE_TEX_SIZE, 3), (60, 220, 240), dtype=np.uint8)  # bright yellow-ish (BGR)
        cv2.circle(img, (SPHERE_TEX_SIZE // 2, SPHERE_TEX_SIZE // 2), SPHERE_TEX_SIZE // 3, (40, 40, 220), -1)  # red disc
        cv2.circle(img, (SPHERE_TEX_SIZE // 2, SPHERE_TEX_SIZE // 2), SPHERE_TEX_SIZE // 6, (220, 220, 40), -1)  # cyan-ish core
        _SPHERE_TEX = img
    return _SPHERE_TEX


def sphere_center(t, nframes, motion="sweep_x", cycles=1.0):
    """Return one of the synthetic object's controlled motion profiles.

    ``sweep_x`` preserves the original constant-velocity trajectory. ``bounce_x``
    reverses direction halfway through the sequence, while ``diagonal_xz`` changes
    both lateral position and depth.  The latter two are dataset variants for
    checking that a dynamic representation did not overfit one monotonic path.

    ``cycles`` (``bounce_x`` only) is how many out-and-back traversals the sphere
    makes over the sequence, and exists to raise the *per-frame displacement*
    without touching anything else.  Why that knob and not a longer path: at the
    default the sphere moves 0.263 per frame against its own diameter of 1.2, so
    consecutive frames overlap by ~78% and every dynamic representation tested so
    far is indistinguishable from simply widening in time (panoramic_4dgs_status
    .md section 3.30).  Widening the sweep instead would push the sphere through
    the walls and force the camera path to change with it -- two variables at
    once.  Bouncing more often keeps the room, the frame count, the camera path
    and the swept volume identical, and moves only the quantity under test.

    ``cycles=1.0`` reproduces the original ``bounce_x`` value-for-value.
    """
    frac = t / max(1, nframes - 1)
    if motion != "bounce_x" and float(cycles) != 1.0:
        raise ValueError(
            f"cycles is only meaningful for bounce_x, not {motion!r}; a "
            "multi-cycle sweep/diagonal would teleport at each wrap")
    if motion == "sweep_x":
        x = -2.5 + 5.0 * frac
        z = 1.0
    elif motion == "bounce_x":
        # triangular wave, `cycles` full out-and-back traversals over the run
        u = (2.0 * float(cycles) * frac) % 2.0
        triangular_phase = u if u <= 1.0 else 2.0 - u
        x = -2.5 + 5.0 * triangular_phase
        z = 1.0
    elif motion == "diagonal_xz":
        x = -2.5 + 5.0 * frac
        z = 1.8 - 1.6 * frac
    else:
        raise ValueError(f"unknown sphere motion profile: {motion}")
    y = 0.8
    return np.array([x, y, z])


FACE_TEXTURES = None  # populated lazily: dict face_name -> HxWx3 uint8


NATURAL_TEXTURE = False  # set by --natural_texture


def get_textures():
    global FACE_TEXTURES
    if FACE_TEXTURES is None:
        maker = make_natural_wall_texture if NATURAL_TEXTURE else make_wall_texture
        FACE_TEXTURES = {
            'x+': maker(1, 0),    # red
            'x-': maker(2, 60),   # green
            'y+': maker(3, 100),  # cyan-ish (floor, Y-down positive)
            'y-': maker(4, 20),   # orange (ceiling)
            'z+': maker(5, 140),  # blue (far wall)
            'z-': maker(6, 170),  # magenta (start wall)
        }
    return FACE_TEXTURES


def erp_ray_grid(h, w):
    """Camera-frame unit ray direction for every ERP pixel, matching cubemap.py's
    theta/phi convention (X-right, Y-down, Z-forward)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    theta = (xx / w - 0.5) * 2 * np.pi
    phi = (0.5 - yy / h) * np.pi
    rays = np.stack([np.cos(phi) * np.sin(theta),
                      -np.sin(phi),
                      np.cos(phi) * np.cos(theta)], axis=-1)
    return rays


def pinhole_ray_grid(h, w, fx, fy, cx, cy):
    """Camera-frame unit ray direction for every pixel of a standard pinhole camera (same
    X-right, Y-down, Z-forward convention as erp_ray_grid). Used to render the identical
    room directly as a plain pinhole video -- see render_pinhole_vs_cubemap.py -- for an
    apples-to-apples comparison against the ERP->cubemap round trip with everything else
    (room geometry, texture, camera trajectory, resolution, intrinsics, training budget)
    held fixed, isolating whether the cubemap porting itself costs quality."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    x = (xx - cx) / fx
    y = (yy - cy) / fy
    z = np.ones_like(x)
    rays = np.stack([x, y, z], axis=-1)
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    return rays


def camera_pose(t):
    """Camera-to-world pose of frame ``t``: ``(position, rotation)``.

    Split out of ``main`` so that the ground-truth trajectory has exactly one
    definition. It used to live inline in the render loop, which meant the poses
    existed only while a dataset was being generated and were never written
    down -- the reason this project spent months believing its synthetic
    sequences had no pose ground truth and that ATE was unmeasurable on them
    (panoramic_4dgs_status.md 3.51). They were always analytic; nobody exported
    them. tools/export_synthetic_gt_poses.py now does, from this function.

    The dolly stays well inside +/-HZ and the weave exists to give the bundle
    adjustment parallax that a rig sharing one optical centre cannot supply on
    its own (section 3.45).
    """
    position = np.array([
        0.3 * np.sin(t * 0.35),    # small lateral weave for extra parallax
        0.1 * np.sin(t * 0.5),
        -1.5 + 0.15 * t,           # forward dolly
    ])
    rotation = yaw_pitch_roll_matrix(
        np.radians(6.0 * np.sin(t * 0.3)),
        np.radians(2.0 * np.sin(t * 0.4)),
        0.0,
    )
    return position, rotation


def yaw_pitch_roll_matrix(yaw, pitch, roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    cp, sp = np.cos(pitch), np.sin(pitch)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    return Ry @ Rx @ Rz


def render_room(cam_pos, R_wc, h, w, obj_center=None, rays_cam=None,
                return_depth=False):
    """Render one frame via ray intersection against the room walls, plus (if obj_center is
    given) a moving sphere -- see module docstring. rays_cam defaults to the ERP ray grid
    (360-degree equirectangular); pass pinhole_ray_grid(...)'s output instead to render the
    identical room as a plain pinhole camera (see --pinhole)."""
    if rays_cam is None:
        rays_cam = erp_ray_grid(h, w)                   # (h,w,3) in camera frame
    rays_world = rays_cam @ R_wc.T                       # rotate to world frame

    ox, oy, oz = cam_pos
    dx, dy, dz = rays_world[..., 0], rays_world[..., 1], rays_world[..., 2]

    eps = 1e-9
    cands = []  # (t, face_id)
    # face_id: 0=x+ 1=x- 2=y+ 3=y- 4=z+ 5=z- 6=dynamic sphere (if present)
    for axis, (o, d, half) in enumerate([(ox, dx, HX), (oy, dy, HY), (oz, dz, HZ)]):
        for sign, face_id in [(1, axis * 2), (-1, axis * 2 + 1)]:
            denom = np.where(np.abs(d) < eps, eps, d)
            t = (sign * half - o) / denom
            valid = (np.sign(d) == sign) & (t > 0)
            cands.append((np.where(valid, t, np.inf), face_id))

    if obj_center is not None:
        # ray-sphere intersection: |o + t*d - c|^2 = r^2, take the nearer positive root.
        ocx, ocy, ocz = ox - obj_center[0], oy - obj_center[1], oz - obj_center[2]
        a = dx * dx + dy * dy + dz * dz
        b = 2 * (dx * ocx + dy * ocy + dz * ocz)
        c = ocx * ocx + ocy * ocy + ocz * ocz - SPHERE_RADIUS ** 2
        disc = b * b - 4 * a * c
        valid_disc = disc >= 0
        sqrt_disc = np.sqrt(np.where(valid_disc, disc, 0.0))
        t_near = (-b - sqrt_disc) / (2 * a)
        t_far = (-b + sqrt_disc) / (2 * a)
        t_sphere = np.where(t_near > eps, t_near, t_far)
        valid_sphere = valid_disc & (t_sphere > eps)
        cands.append((np.where(valid_sphere, t_sphere, np.inf), 6))

    t_stack = np.stack([c[0] for c in cands], axis=0)  # (6 or 7,h,w)
    best = np.argmin(t_stack, axis=0)                  # (h,w) which face
    t_best = np.take_along_axis(t_stack, best[None], axis=0)[0]

    hit = np.stack([ox + dx * t_best, oy + dy * t_best, oz + dz * t_best], axis=-1)

    out = np.zeros((h, w, 3), dtype=np.uint8)
    tex = get_textures()
    face_names = ['x+', 'x-', 'y+', 'y-', 'z+', 'z-']
    for face_id, name in enumerate(face_names):
        mask = best == face_id
        if not mask.any():
            continue
        hx_, hy_, hz_ = hit[..., 0], hit[..., 1], hit[..., 2]
        if face_id in (0, 1):     # x+/x- : uv from (z,y)
            u = (hz_ / HZ * 0.5 + 0.5)
            v = (hy_ / HY * 0.5 + 0.5)
        elif face_id in (2, 3):   # y+/y- : uv from (x,z)
            u = (hx_ / HX * 0.5 + 0.5)
            v = (hz_ / HZ * 0.5 + 0.5)
        else:                     # z+/z- : uv from (x,y)
            u = (hx_ / HX * 0.5 + 0.5)
            v = (hy_ / HY * 0.5 + 0.5)
        ui = np.clip((u[mask] * (TEX_SIZE - 1)).astype(np.int32), 0, TEX_SIZE - 1)
        vi = np.clip((v[mask] * (TEX_SIZE - 1)).astype(np.int32), 0, TEX_SIZE - 1)
        out[mask] = tex[name][vi, ui]

    if obj_center is not None:
        mask = best == 6
        if mask.any():
            nx = (hit[..., 0][mask] - obj_center[0]) / SPHERE_RADIUS
            ny = (hit[..., 1][mask] - obj_center[1]) / SPHERE_RADIUS
            nz = (hit[..., 2][mask] - obj_center[2]) / SPHERE_RADIUS
            u = 0.5 + np.arctan2(nx, nz) / (2 * np.pi)
            v = 0.5 - np.arcsin(np.clip(ny, -1.0, 1.0)) / np.pi
            sphere_tex = make_sphere_texture()
            ui = np.clip((u * (SPHERE_TEX_SIZE - 1)).astype(np.int32), 0, SPHERE_TEX_SIZE - 1)
            vi = np.clip((v * (SPHERE_TEX_SIZE - 1)).astype(np.int32), 0, SPHERE_TEX_SIZE - 1)
            out[mask] = sphere_tex[vi, ui]

    if return_depth:
        # t_best is distance along a unit ray, i.e. *radial* depth from the camera
        # centre -- not the along-axis z a pinhole camera reports. Converting it is
        # the consumer's job (cubemap.erp_depth_to_cubemap), because the conversion
        # depends on which face a pixel lands in.
        return out, t_best.astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/synth_erp_room')
    ap.add_argument('--nframes', type=int, default=20)
    ap.add_argument('--erp_h', type=int, default=640,
                     help='default raised from 320 (panoramic_4dgs_status.md section 3.12): 320x640 is '
                          'below what the default calib/equirect_test.yml face_size=384/fov=90 needs at '
                          'the cubemap face center, costing ~4.4dB PSNR to avoidable resampling blur -- '
                          'see cubemap.py::min_erp_width_for_face()')
    ap.add_argument('--erp_w', type=int, default=1280)
    ap.add_argument('--dynamic_object', action='store_true',
                     help='add a moving sphere with a known ground-truth trajectory -- see module docstring')
    ap.add_argument(
        '--dynamic_motion',
        choices=('sweep_x', 'bounce_x', 'diagonal_xz'),
        default='sweep_x',
        help='moving-sphere trajectory; default sweep_x preserves the original dataset',
    )
    ap.add_argument(
        '--natural_texture', action='store_true',
        help='use lower-frequency, depth-network-friendly wall textures instead '
             'of the default adversarial checkerboard (see section 3.43: '
             'metric3d correlates negatively with ground truth on the default)')
    ap.add_argument(
        '--depth', action='store_true',
        help='also write ground-truth radial depth per frame as .npy into '
             '<out>_depth/. Exists to answer whether monocular depth is the '
             'accuracy bottleneck (panoramic_4dgs_status.md section 3.39) by '
             'rerunning the identical sequence with exact depth instead.')
    ap.add_argument(
        '--motion_cycles', type=float, default=1.0,
        help='bounce_x only: out-and-back traversals over the sequence. Raises '
             'per-frame displacement while keeping the room, frame count, camera '
             'path and swept volume fixed. Default 1.0 reproduces the original '
             'bounce_x exactly; 3.5 gives ~1.26 sphere diameters per frame',
    )
    ap.add_argument('--pinhole', action='store_true',
                     help='render the identical room/texture/camera trajectory directly as a plain '
                          'pinhole video instead of ERP -- for an apples-to-apples cubemap-vs-pinhole '
                          'quality comparison (panoramic_4dgs_status.md section 3.11). Writes a matching '
                          'calib/<out_basename>_pinhole.yml using the exact same fx/fy/cx/cy formula as '
                          'cubemap.py::face_intrinsics(), so the only difference from a cubemap face is '
                          'whether the frame went through the ERP encode + cubemap decode round trip.')
    ap.add_argument('--face_size', type=int, default=384, help='pinhole mode: square output resolution')
    ap.add_argument('--fov', type=float, default=90.0, help='pinhole mode: field of view in degrees')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.pinhole:
        f = args.face_size / (2.0 * np.tan(np.radians(args.fov) / 2.0))
        c = args.face_size / 2.0
        rays_cam = pinhole_ray_grid(args.face_size, args.face_size, f, f, c, c)
        out_h = out_w = args.face_size
        calib_path = os.path.join('calib', os.path.basename(args.out.rstrip('/')) + '.yml')
        with open(calib_path, 'w') as cf:
            cf.write(f"intrinsic:  [[{f}, {f}, {c}, {c}, 0, 0, 0, 0]]\n\ncamera:     'pinhole'\n\n"
                      f"baseline:   [0, 0, 0, 0, 0, 0, 1]\n\ntimescale:  1\n")
    else:
        rays_cam = None  # render_room defaults to the ERP grid
        out_h, out_w = args.erp_h, args.erp_w

    global NATURAL_TEXTURE
    NATURAL_TEXTURE = args.natural_texture

    depth_dir = args.out.rstrip('/') + '_depth'
    if args.depth:
        os.makedirs(depth_dir, exist_ok=True)

    gt_trajectory = []
    for t in range(args.nframes):
        cam_pos, R_wc = camera_pose(t)

        obj_center = (
            sphere_center(t, args.nframes, motion=args.dynamic_motion,
                          cycles=args.motion_cycles)
            if args.dynamic_object else None
        )
        if args.depth:
            frame, depth = render_room(cam_pos, R_wc, out_h, out_w,
                                       obj_center=obj_center, rays_cam=rays_cam,
                                       return_depth=True)
            np.save(os.path.join(depth_dir, f'{t}.npy'), depth)
        else:
            frame = render_room(cam_pos, R_wc, out_h, out_w, obj_center=obj_center, rays_cam=rays_cam)
        cv2.imwrite(os.path.join(args.out, f'{t}.png'), frame)
        if obj_center is not None:
            gt_trajectory.append({'frame': t, 'center_xyz': obj_center.tolist(), 'radius': SPHERE_RADIUS})

    if args.dynamic_object:
        import json
        # NOTE: written *outside* args.out, not inside it -- mcgs_slam/streams.py's
        # equirect_cubemap_stream() scans every file in the frame directory and sorts by a
        # trailing number extracted from the filename (map_filename()), so a non-numeric
        # filename dropped in the same directory as the frames crashes that sort.
        gt_path = args.out.rstrip('/') + '_sphere_gt_trajectory.json'
        with open(gt_path, 'w') as f:
            json.dump(gt_trajectory, f, indent=2)
        kind = 'pinhole' if args.pinhole else 'ERP'
        print(f'Wrote {args.nframes} synthetic {kind} frames (with moving sphere) to {args.out}, ground truth trajectory at {gt_path}')
    else:
        kind = 'pinhole' if args.pinhole else 'ERP'
        print(f'Wrote {args.nframes} synthetic {kind} frames to {args.out}')
    if args.pinhole:
        print(f'Wrote matching calib to {calib_path}')


if __name__ == '__main__':
    main()
