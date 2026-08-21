"""Offline Gaussian-ellipsoid ("高斯球") renderer.

Reuses the MCGS-SLAM OpenGL splat shader (gl_render, render_mod=-4) to render a
saved 3DGS ply headlessly through EGL, so single-agent MAGS-SLAM Replica runs can
be inspected without a display.

Example:
    python tools/render_gaussian_balls.py \
        --ply outputs/replica/room0/3dgs_final.ply \
        --traj outputs/replica/room0/traj_kf.txt \
        --out /tmp/room0
"""

import argparse
import ctypes
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from plyfile import PlyData

logger = logging.getLogger(__name__)

_MCGS_ROOT = Path(__file__).resolve().parents[1]
if str(_MCGS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MCGS_ROOT))


def create_egl_context(width: int, height: int) -> None:
    """Create a headless OpenGL context backed by an EGL pbuffer.

    The EGL imports stay local: setting PYOPENGL_PLATFORM at module import time would
    break callers that already own a GLX context (e.g. tools/view_gaussian_balls.py).
    """
    os.environ["PYOPENGL_PLATFORM"] = "egl"

    from OpenGL import EGL
    from OpenGL.EGL import (
        EGL_BLUE_SIZE, EGL_DEFAULT_DISPLAY, EGL_DEPTH_SIZE, EGL_GREEN_SIZE, EGL_HEIGHT,
        EGL_NO_CONTEXT, EGL_NONE, EGL_OPENGL_API, EGL_OPENGL_BIT, EGL_PBUFFER_BIT,
        EGL_RED_SIZE, EGL_RENDERABLE_TYPE, EGL_SURFACE_TYPE, EGL_WIDTH, EGLConfig,
        eglBindAPI, eglChooseConfig, eglCreateContext, eglCreatePbufferSurface,
        eglGetDisplay, eglInitialize, eglMakeCurrent,
    )

    display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
    major, minor = ctypes.c_long(), ctypes.c_long()
    if not eglInitialize(display, major, minor):
        raise RuntimeError("eglInitialize failed")

    cfg_attribs = [
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_NONE,
    ]
    configs = (EGLConfig * 1)()
    num_cfg = ctypes.c_long()
    attribs = (ctypes.c_int * len(cfg_attribs))(*cfg_attribs)
    if not eglChooseConfig(display, attribs, configs, 1, num_cfg) or num_cfg.value == 0:
        raise RuntimeError("eglChooseConfig found no OpenGL-capable config")

    surf_attribs = (ctypes.c_int * 5)(EGL_WIDTH, width, EGL_HEIGHT, height, EGL_NONE)
    surface = eglCreatePbufferSurface(display, configs[0], surf_attribs)
    eglBindAPI(EGL_OPENGL_API)
    context = eglCreateContext(display, configs[0], EGL_NO_CONTEXT, None)
    if context == EGL.EGL_NO_CONTEXT:
        raise RuntimeError("eglCreateContext failed")
    eglMakeCurrent(display, surface, surface, context)


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Right-handed OpenGL view matrix (numpy replacement for glm.lookAt)."""
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-9)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-9)
    u = np.cross(s, f)
    view = np.eye(4, dtype=np.float32)
    view[0, :3], view[1, :3], view[2, :3] = s, u, -f
    view[:3, 3] = [-np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye)]
    return view


def _perspective(fovy: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    """Right-handed OpenGL projection matrix (numpy replacement for glm.perspective)."""
    t = 1.0 / np.tan(fovy / 2.0)
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = t / aspect
    proj[1, 1] = t
    proj[2, 2] = -(zfar + znear) / (zfar - znear)
    proj[2, 3] = -2.0 * zfar * znear / (zfar - znear)
    proj[3, 2] = -1.0
    return proj


def patch_camera_math() -> None:
    """The installed `glm` package is not PyGLM, so supply numpy matrix math."""
    from mcgs_slam.gaussian.gui.gl_render import util

    util.Camera.get_view_matrix = lambda self: _look_at(
        np.asarray(self.position, dtype=np.float32),
        np.asarray(self.target, dtype=np.float32),
        np.asarray(self.up, dtype=np.float32),
    )
    util.Camera.get_project_matrix = lambda self: _perspective(
        self.fovy, self.w / self.h, self.znear, self.zfar
    )

    from OpenGL import GL as gl

    def set_uniform_mat4(shader, content, name):
        gl.glUseProgram(shader)
        gl.glUniformMatrix4fv(
            gl.glGetUniformLocation(shader, name), 1, gl.GL_FALSE,
            np.ascontiguousarray(content.T, dtype=np.float32),
        )

    util.set_uniform_mat4 = set_uniform_mat4
    from mcgs_slam.gaussian.gui.gl_render import render_ogl

    render_ogl.util.set_uniform_mat4 = set_uniform_mat4


def load_gaussians(ply_path: Path):
    """Load a 3DGS ply into the gl_render GaussianData layout (activated params)."""
    from mcgs_slam.gaussian.gui.gl_render import util_gau

    ply = PlyData.read(str(ply_path))["vertex"]
    xyz = np.stack([ply["x"], ply["y"], ply["z"]], axis=-1).astype(np.float32)
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(ply["opacity"], dtype=np.float32)))
    scale = np.exp(
        np.stack([ply[f"scale_{i}"] for i in range(3)], axis=-1).astype(np.float32)
    )
    rot = np.stack([ply[f"rot_{i}"] for i in range(4)], axis=-1).astype(np.float32)
    rot = rot / (np.linalg.norm(rot, axis=-1, keepdims=True) + 1e-9)
    sh = np.stack([ply[f"f_dc_{i}"] for i in range(3)], axis=-1).astype(np.float32)
    return util_gau.GaussianData(
        xyz=xyz,
        rot=rot,
        scale=scale,
        opacity=opacity.reshape(-1, 1),
        sh=sh,
    )


def load_kf_poses(traj_path: Path) -> Optional[np.ndarray]:
    """Read TUM-style keyframe poses (t tx ty tz qx qy qz qw) as 4x4 cam-to-world."""
    if not traj_path.exists():
        return None
    data = np.loadtxt(str(traj_path))
    if data.ndim == 1:
        data = data[None]
    trans = data[:, 1:4]
    quat = data[:, 4:8]  # xyzw
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    rot = np.stack(
        [
            1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2),
        ],
        axis=-1,
    ).reshape(-1, 3, 3)
    poses = np.tile(np.eye(4), (len(data), 1, 1))
    poses[:, :3, :3] = rot
    poses[:, :3, 3] = trans
    return poses.astype(np.float32)


def keyframe_views(poses: np.ndarray, fracs: Tuple[float, ...]) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """Eye/target/up triplets taken from keyframe camera poses (OpenCV convention)."""
    views = []
    for frac in fracs:
        idx = int(np.clip(round(frac * (len(poses) - 1)), 0, len(poses) - 1))
        c2w = poses[idx]
        eye = c2w[:3, 3]
        forward = c2w[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
        up = c2w[:3, :3] @ np.array([0.0, -1.0, 0.0], dtype=np.float32)
        views.append((f"kf{idx:04d}", eye, eye + forward * 3.0, up))
    return views


def orbit_views(xyz: np.ndarray, n_views: int) -> List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    """External orbit around the robust scene centroid (top-down-ish elevation)."""
    lo, hi = np.percentile(xyz, 2, axis=0), np.percentile(xyz, 98, axis=0)
    center = (lo + hi) / 2.0
    radius = float(np.linalg.norm(hi - lo)) * 0.75
    views = []
    for i in range(n_views):
        theta = 2 * np.pi * i / n_views
        eye = center + np.array(
            [radius * np.cos(theta), -radius * 0.55, radius * np.sin(theta)],
            dtype=np.float32,
        )
        views.append(
            (f"orbit{i:04d}", eye.astype(np.float32), center.astype(np.float32),
             np.array([0.0, -1.0, 0.0], dtype=np.float32))
        )
    return views


def render_scene(args: argparse.Namespace) -> None:
    from OpenGL import GL as gl

    from mcgs_slam.gaussian.gui.gl_render import render_ogl, util

    patch_camera_math()
    ply_path = Path(args.ply)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gaussians = load_gaussians(ply_path)
    logger.info("loaded %d gaussians from %s", len(gaussians), ply_path)

    renderer = render_ogl.OpenGLRenderer(args.width, args.height)
    renderer.update_gaussian_data(gaussians)
    renderer.set_render_mod(args.render_mod)
    renderer.set_scale_modifier(args.scale_modifier)

    camera = util.Camera(args.height, args.width)
    camera.fovy = np.deg2rad(args.fovy)

    views: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    poses = load_kf_poses(Path(args.traj)) if args.traj else None
    if poses is not None:
        views += keyframe_views(poses, tuple(args.kf_fracs))
    if args.orbit > 0:
        views += orbit_views(gaussians.xyz, args.orbit)

    from PIL import Image

    for name, eye, target, up in views:
        camera.position = np.asarray(eye, dtype=np.float32)
        camera.target = np.asarray(target, dtype=np.float32)
        camera.up = np.asarray(up, dtype=np.float32)

        renderer.sort_and_update(camera)
        renderer.update_camera_pose(camera)
        renderer.update_camera_intrin(camera)
        renderer.set_render_reso(args.width, args.height)

        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        renderer.draw()
        gl.glFinish()

        buf = gl.glReadPixels(0, 0, args.width, args.height, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        img = np.frombuffer(buf, dtype=np.uint8).reshape(args.height, args.width, 3)[::-1]
        out_path = out_dir / f"{ply_path.parent.name}_{name}.png"
        Image.fromarray(img).save(out_path)
        logger.info("wrote %s", out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--traj", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fovy", type=float, default=60.0)
    parser.add_argument("--scale-modifier", type=float, default=1.0)
    parser.add_argument(
        "--render-mod", type=int, default=-4,
        help="-4 gaussian ball, -3 flat ball, -2 billboard, -1 depth, >=0 SH color",
    )
    parser.add_argument("--kf-fracs", type=float, nargs="*", default=[0.25, 0.6])
    parser.add_argument("--orbit", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    create_egl_context(args.width, args.height)
    render_scene(args)


if __name__ == "__main__":
    main()
