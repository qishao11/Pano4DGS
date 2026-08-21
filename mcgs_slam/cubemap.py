"""Equirectangular (ERP) <-> cubemap-face conversion for panoramic camera support.

Camera coordinate convention (matches the rest of mcgs_slam / DROID-SLAM):
X-right, Y-down, Z-forward. The 'front' face is the reference frame (== cam0):
looking down +Z. 'right'/'back'/'left' are the remaining horizontal faces,
each a fixed 90-degree yaw around the shared optical center (zero baseline).
Only the four horizontal faces are implemented -- P1 deliberately skips
'up'/'down' to avoid the equirectangular pole-singularity risk documented in
panoramic_support_feasibility.md (section 2.2).
"""
import numpy as np
import cv2

_HORIZONTAL_FACES = ('front', 'right', 'back', 'left')
_ALL_FACES = _HORIZONTAL_FACES + ('up', 'down')

_grid_cache = {}


def _face_axes(face):
    """(right_axis, down_axis, forward_axis) of `face`'s camera frame, expressed
    in the front/cam0 camera frame."""
    if face == 'front':
        return np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])
    if face == 'right':
        return np.array([0., 0., -1.]), np.array([0., 1., 0.]), np.array([1., 0., 0.])
    if face == 'back':
        return np.array([-1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., -1.])
    if face == 'left':
        return np.array([0., 0., 1.]), np.array([0., 1., 0.]), np.array([-1., 0., 0.])
    # Poles. Derived so that right x down == forward (the same right-handed
    # convention the four horizontal faces satisfy), and verified by the
    # round-trip in erp_selftest: with all six faces the ERP is fully covered.
    # They were left unimplemented in P1 to avoid the ERP pole singularity; the
    # singularity is a *sampling* issue (ERP packs enormous pixel area into the
    # poles), not a projection error, so the faces themselves are well defined.
    if face == 'up':        # looks along -Y
        return np.array([1., 0., 0.]), np.array([0., 0., 1.]), np.array([0., -1., 0.])
    if face == 'down':      # looks along +Y
        return np.array([1., 0., 0.]), np.array([0., 0., -1.]), np.array([0., 1., 0.])
    raise NotImplementedError(f"cubemap face '{face}' is not implemented")


def face_rotation_matrix(face):
    """R such that X_face = R @ X_front, for a point expressed in the front/cam0 frame."""
    right, down, forward = _face_axes(face)
    return np.stack([right, down, forward], axis=0)


def rotmat_to_quat(R):
    """3x3 rotation matrix -> (qx, qy, qz, qw), via Shepperd's method."""
    m = R
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m[2, 1] - m[1, 2]) / S
        qy = (m[0, 2] - m[2, 0]) / S
        qz = (m[1, 0] - m[0, 1]) / S
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / S
        qx = 0.25 * S
        qy = (m[0, 1] + m[1, 0]) / S
        qz = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / S
        qx = (m[0, 1] + m[1, 0]) / S
        qy = 0.25 * S
        qz = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / S
        qx = (m[0, 2] + m[2, 0]) / S
        qy = (m[1, 2] + m[2, 1]) / S
        qz = 0.25 * S
    return np.array([qx, qy, qz, qw])


def face_rotation_quat(face):
    """T_cami_cam0-style row [tx, ty, tz, qx, qy, qz, qw] for `face` relative to 'front'.
    Translation is always zero: cubemap faces share one optical center."""
    qx, qy, qz, qw = rotmat_to_quat(face_rotation_matrix(face))
    return [0., 0., 0., float(qx), float(qy), float(qz), float(qw)]


def face_intrinsics(face_size, fov_deg=90.0):
    """(fx, fy, cx, cy) for a square cubemap face of `face_size` pixels covering `fov_deg`."""
    f = face_size / (2.0 * np.tan(np.radians(fov_deg) / 2.0))
    c = face_size / 2.0
    return f, f, c, c


def min_erp_width_for_face(face_size, fov_deg=90.0):
    """Minimum ERP source image width (for the standard 2:1 ERP aspect ratio) below which
    erp_to_cubemap()'s cv2.remap() has to *upsample*-interpolate the cubemap face's center
    region -- introducing blur there, not because the projection math is wrong, but because
    the face simply asks for more angular resolution than the ERP source has to give.

    For a rectilinear (pinhole) cubemap face, angular pixel density is highest at the face
    center (d(px)/d(angle) = fx there) and falls off toward the edges (down to roughly half
    that at a 90-degree-FOV face's edge) -- see panoramic_4dgs_status.md section 3.12, which
    found this exact mismatch cost ~4.4dB PSNR on cam0/front alone (the best-covered view)
    versus rendering the identical scene directly as pinhole, no ERP round trip involved.
    Below this width the face center is being asked to show detail the ERP frame never had
    in the first place; caller should either raise the ERP capture/render resolution or
    accept the blur (e.g. lower face_size/fov, or downstream quality requirements that don't
    need face-center sharpness)."""
    fx, _, _, _ = face_intrinsics(face_size, fov_deg)
    return int(np.ceil(fx * 2 * np.pi))


def _face_sampling_grid(face, face_size, erp_h, erp_w, fov_deg=90.0):
    """cv2.remap-compatible (map_x, map_y) turning a face pixel into its ERP source pixel."""
    key = (face, face_size, erp_h, erp_w, fov_deg)
    if key in _grid_cache:
        return _grid_cache[key]

    fx, fy, cx, cy = face_intrinsics(face_size, fov_deg)
    u, v = np.meshgrid(np.arange(face_size, dtype=np.float64),
                        np.arange(face_size, dtype=np.float64))
    rays_face = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)
    rays_face /= np.linalg.norm(rays_face, axis=-1, keepdims=True)

    R_face_front = face_rotation_matrix(face)
    rays_front = rays_face @ R_face_front  # row-vector form of R_face_front.T @ ray

    xf, yf, zf = rays_front[..., 0], rays_front[..., 1], rays_front[..., 2]
    theta = np.arctan2(xf, zf)
    phi = np.arcsin(np.clip(-yf, -1.0, 1.0))

    map_x = (((theta / (2 * np.pi)) + 0.5) * erp_w).astype(np.float32)
    map_y = ((0.5 - phi / np.pi) * erp_h).astype(np.float32)

    _grid_cache[key] = (map_x, map_y)
    return map_x, map_y


def erp_to_cubemap(erp_img, face_size, faces=_HORIZONTAL_FACES, fov_deg=90.0):
    """Split an equirectangular image into {face_name: HxWx3 image}."""
    h, w = erp_img.shape[:2]
    out = {}
    for face in faces:
        map_x, map_y = _face_sampling_grid(face, face_size, h, w, fov_deg)
        out[face] = cv2.remap(erp_img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_WRAP)
    return out


def _radial_to_z_factor(face_size, fov_deg=90.0):
    """Per-pixel factor converting radial depth to a pinhole face's along-axis z.

    An ERP frame's depth is the distance from the camera centre along a unit ray.
    A pinhole camera -- which is what each cubemap face pretends to be, and what
    every downstream consumer assumes -- reports the z component instead. For a
    face pixel whose unnormalized ray is ((u-cx)/fx, (v-cy)/fy, 1), the unit ray's
    z is 1/||.||, so z = radial / ||.||. The factor is 1 at the face centre and
    falls to about 0.58 at a 90-degree face's corner, so skipping it would put the
    face's edges systematically too far away.
    """
    key = ("radial_to_z", face_size, fov_deg)
    if key in _grid_cache:
        return _grid_cache[key]
    fx, fy, cx, cy = face_intrinsics(face_size, fov_deg)
    u, v = np.meshgrid(np.arange(face_size, dtype=np.float64),
                        np.arange(face_size, dtype=np.float64))
    norm = np.sqrt(((u - cx) / fx) ** 2 + ((v - cy) / fy) ** 2 + 1.0)
    factor = (1.0 / norm).astype(np.float32)
    _grid_cache[key] = factor
    return factor


def erp_depth_to_cubemap(erp_depth, face_size, faces=_HORIZONTAL_FACES, fov_deg=90.0):
    """Split an equirectangular *radial* depth map into per-face pinhole z depth.

    Nearest-neighbour sampling on purpose: linear interpolation across a depth
    discontinuity invents a surface halfway between foreground and background,
    which is exactly the kind of point that shows up later as an outlier
    Gaussian.
    """
    h, w = erp_depth.shape[:2]
    factor = _radial_to_z_factor(face_size, fov_deg)
    out = {}
    for face in faces:
        map_x, map_y = _face_sampling_grid(face, face_size, h, w, fov_deg)
        radial = cv2.remap(erp_depth.astype(np.float32), map_x, map_y,
                            interpolation=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_WRAP)
        out[face] = radial * factor
    return out


def cubemap_to_erp(face_images, erp_h, erp_w, fov_deg=90.0):
    """Stitch rendered cubemap faces back into one equirectangular image.

    The inverse of erp_to_cubemap, and what makes a panoramic *output* possible:
    the pipeline renders pinhole faces, but the deliverable of a panoramic
    reconstruction is a full 360 image. Overlapping faces are averaged.

    Returns ``(erp_image_uint8, covered_mask)``. Anything the given faces do not
    span -- the poles, when only the four horizontal faces are used -- stays
    black and is reported as uncovered rather than silently filled.
    """
    faces = list(face_images)
    face_size = next(iter(face_images.values())).shape[0]
    fx, fy, cx, cy = face_intrinsics(face_size, fov_deg)

    recon = np.zeros((erp_h, erp_w, 3), dtype=np.float64)
    coverage = np.zeros((erp_h, erp_w), dtype=np.float64)

    yy, xx = np.mgrid[0:erp_h, 0:erp_w].astype(np.float64)
    theta = (xx / erp_w - 0.5) * 2 * np.pi
    phi = (0.5 - yy / erp_h) * np.pi
    rays_front = np.stack([np.cos(phi) * np.sin(theta),
                            -np.sin(phi),
                            np.cos(phi) * np.cos(theta)], axis=-1)

    for face in faces:
        face_img = face_images[face]
        rays_face = rays_front @ face_rotation_matrix(face).T
        valid = rays_face[..., 2] > 1e-6
        safe_z = np.where(valid, rays_face[..., 2], 1.0)
        u = fx * rays_face[..., 0] / safe_z + cx
        v = fy * rays_face[..., 1] / safe_z + cy
        valid &= (u >= 0) & (u < face_size) & (v >= 0) & (v < face_size)
        sampled = cv2.remap(face_img, u.astype(np.float32), v.astype(np.float32),
                             interpolation=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
        mask = valid.astype(np.float64)
        recon += sampled.astype(np.float64) * mask[..., None]
        coverage += mask

    covered = coverage > 0
    recon[covered] /= coverage[covered, None]
    return recon.astype(np.uint8), covered


def erp_selftest(erp_img, face_size=384, faces=_HORIZONTAL_FACES, fov_deg=90.0):
    """Round-trip check: split `erp_img` into cubemap faces, reproject each face back onto
    the ERP grid, and return (reconstruction, abs-diff-map, covered-mask). Validates the
    projection math with any synthetic test pattern -- no real panoramic data required."""
    h, w = erp_img.shape[:2]
    faces_imgs = erp_to_cubemap(erp_img, face_size, faces, fov_deg)
    recon, covered = cubemap_to_erp(faces_imgs, h, w, fov_deg)

    diff = np.zeros((h, w), dtype=np.float64)
    diff[covered] = np.abs(
        recon[covered].astype(np.float64) - erp_img[covered].astype(np.float64)).mean(axis=-1)
    return recon, diff, covered
