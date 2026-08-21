"""Interactive Gaussian-ellipsoid viewer (needs a real display, i.e. run it from the desktop).

Same shader as tools/render_gaussian_balls.py, but in a GLFW window with mouse orbit,
so the "gaussian ball" mode can be inspected from any viewpoint.

Controls:
    left drag   orbit          scroll      zoom
    right drag  pan            B           cycle render mode (ball/flat/billboard/splat)
    [ / ]       scale modifier R           reset view            ESC  quit

Example:
    python tools/view_gaussian_balls.py --ply outputs/replica/room0/3dgs_final.ply
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Must be pinned before PyOpenGL is imported: the window is GLX, so GL entry points
# have to be resolved through GLX too (mixing in EGL yields a SIGBUS).
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import glfw
import numpy as np

_MCGS_ROOT = Path(__file__).resolve().parents[1]
if str(_MCGS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MCGS_ROOT))

from tools.render_gaussian_balls import load_gaussians, patch_camera_math  # noqa: E402

logger = logging.getLogger(__name__)

RENDER_MODES = [(-4, "gaussian ball"), (-3, "flat ball"), (-2, "billboard"), (0, "splat")]


class OrbitState:
    """Spherical camera state driven by the mouse."""

    def __init__(self, center: np.ndarray, radius: float):
        self.center = center.astype(np.float32)
        self.radius = float(radius)
        self.azimuth = 0.0
        self.elevation = 0.5
        self.mode_idx = 0
        self.scale_modifier = 1.0
        self.dirty = True
        self._last = None

    def eye(self) -> np.ndarray:
        ce = np.cos(self.elevation)
        return self.center + self.radius * np.array(
            [ce * np.cos(self.azimuth), -np.sin(self.elevation), ce * np.sin(self.azimuth)],
            dtype=np.float32,
        )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fovy", type=float, default=60.0)
    args = parser.parse_args()

    if not glfw.init():
        raise RuntimeError("glfw init failed (no display?) - use render_gaussian_balls.py instead")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(args.width, args.height, "MCGS gaussian balls", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("glfw window creation failed")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    from OpenGL import GL as gl

    from mcgs_slam.gaussian.gui.gl_render import render_ogl, util

    patch_camera_math()
    gaussians = load_gaussians(Path(args.ply))
    logger.info("loaded %d gaussians", len(gaussians))

    lo, hi = np.percentile(gaussians.xyz, 2, axis=0), np.percentile(gaussians.xyz, 98, axis=0)
    state = OrbitState((lo + hi) / 2.0, float(np.linalg.norm(hi - lo)) * 0.75)
    reset = (state.azimuth, state.elevation, state.radius, state.center.copy())

    renderer = render_ogl.OpenGLRenderer(args.width, args.height)
    renderer.update_gaussian_data(gaussians)
    camera = util.Camera(args.height, args.width)
    camera.fovy = np.deg2rad(args.fovy)
    camera.up = np.array([0.0, -1.0, 0.0], dtype=np.float32)

    def on_mouse_move(_win, x, y):
        if state._last is None:
            state._last = (x, y)
            return
        dx, dy = x - state._last[0], y - state._last[1]
        state._last = (x, y)
        if glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS:
            state.azimuth += dx * 0.005
            state.elevation = float(np.clip(state.elevation + dy * 0.005, -1.5, 1.5))
            state.dirty = True
        elif glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS:
            right = np.cross(state.eye() - state.center, camera.up)
            right /= np.linalg.norm(right) + 1e-9
            state.center = state.center - right * dx * state.radius * 0.001
            state.center = state.center + camera.up * dy * state.radius * 0.001
            state.dirty = True

    def on_scroll(_win, _dx, dy):
        state.radius = float(np.clip(state.radius * (0.9 ** dy), 0.05, 1e4))
        state.dirty = True

    def on_key(_win, key, _sc, action, _mods):
        if action != glfw.PRESS:
            return
        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_B:
            state.mode_idx = (state.mode_idx + 1) % len(RENDER_MODES)
            logger.info("render mode: %s", RENDER_MODES[state.mode_idx][1])
            state.dirty = True
        elif key in (glfw.KEY_LEFT_BRACKET, glfw.KEY_RIGHT_BRACKET):
            factor = 0.8 if key == glfw.KEY_LEFT_BRACKET else 1.25
            state.scale_modifier = float(np.clip(state.scale_modifier * factor, 0.02, 3.0))
            logger.info("scale modifier: %.3f", state.scale_modifier)
            state.dirty = True
        elif key == glfw.KEY_R:
            state.azimuth, state.elevation, state.radius, state.center = (
                reset[0], reset[1], reset[2], reset[3].copy()
            )
            state.dirty = True

    glfw.set_cursor_pos_callback(window, on_mouse_move)
    glfw.set_scroll_callback(window, on_scroll)
    glfw.set_key_callback(window, on_key)

    while not glfw.window_should_close(window):
        glfw.poll_events()
        w, h = glfw.get_framebuffer_size(window)
        if (w, h) != (camera.w, camera.h):
            camera.update_resolution(h, w)
            renderer.set_render_reso(w, h)
            state.dirty = True

        if state.dirty:
            camera.position = state.eye()
            camera.target = state.center
            renderer.sort_and_update(camera)
            renderer.update_camera_pose(camera)
            renderer.update_camera_intrin(camera)
            renderer.set_render_mod(RENDER_MODES[state.mode_idx][0])
            renderer.set_scale_modifier(state.scale_modifier)
            state.dirty = False

        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        renderer.draw()
        glfw.swap_buffers(window)

    glfw.terminate()


if __name__ == "__main__":
    main()
