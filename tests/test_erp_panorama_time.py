"""Rendering a full panorama at an instant no camera observed.

render_time_sweep already renders unobserved times, but on a single cubemap
face; render_erp_panoramas already renders full spheres, but only at observed
instants. The gap between them is the entire "panoramic 4DGS" claim, and it is
closed by letting a panorama borrow the nearest observed instant's face poses
while supplying its own time to the 4D slice.

These tests drive the scheduling half of that on a stub backend -- which poses
get reused, which time each face is asked for, how frames are tagged -- because
the rendering half needs CUDA and a trained map.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))

from gs_backend import GSBackEnd  # noqa: E402
from gaussian.deform.dynamic_modes import (  # noqa: E402
    DYNAMIC_MODE_GAUSSIAN_4D,
    DYNAMIC_MODE_NONE,
)


class _Viewpoint:
    """Just enough of a Camera for the panorama path."""

    def __init__(self, cam_idx, physical_tstamp):
        self.cam_idx = cam_idx
        self.physical_tstamp = physical_tstamp
        self.exposure_a = torch.zeros(())
        self.exposure_b = torch.zeros(())


class _Window:
    def __init__(self, groups):
        self.groups = groups


class _StubBackend:
    """A GSBackEnd stand-in carrying only what the panorama path touches."""

    render_erp_panoramas = GSBackEnd.render_erp_panoramas
    render_erp_panoramas_interpolated = GSBackEnd.render_erp_panoramas_interpolated

    def __init__(self, save_dir, times=(0.0, 1.0, 2.0), faces=('right', 'back', 'left')):
        self.save_dir = save_dir
        self.config = {"Training": {"cubemap_faces": list(faces), "fov": 90.0}}
        self.dynamic_mode = DYNAMIC_MODE_GAUSSIAN_4D
        self.dynamic_observed_times = list(times)
        n_faces = 1 + len(faces)
        self.viewpoints = {}
        groups = {}
        for t in times:
            entries = {}
            for cam_idx in range(n_faces):
                key = (t, cam_idx)
                self.viewpoints[key] = _Viewpoint(cam_idx, t)
                entries[cam_idx] = key
            groups[t] = entries
        self.physical_view_window = _Window(groups)
        self.asked = []          # (pose_time, cam_idx, time_override)

    def render_at(self, viewpoint, time_override=None):
        self.asked.append((viewpoint.physical_tstamp, viewpoint.cam_idx,
                           time_override))
        # a constant grey face: cubemap_to_erp only needs a real image here
        return {"render": torch.full((3, 8, 8), 0.5)}, 0.0


def _stub(save_dir, **kwargs):
    backend = _StubBackend(save_dir, **kwargs)
    return backend


class TestObservedPanoramas(unittest.TestCase):
    def test_observed_times_ask_for_no_override(self):
        """The pre-existing path must keep asking each face for its own time."""
        with tempfile.TemporaryDirectory() as tmp:
            backend = _stub(tmp)
            written = backend.render_erp_panoramas()
            self.assertEqual(written, 3)
            self.assertTrue(all(override is None for _, _, override in backend.asked))
            names = sorted(p.name for p in
                           (Path(tmp) / "renders" / "erp_panorama").iterdir())
            self.assertTrue(all("_obs" in n for n in names), names)


class TestInterpolatedPanoramas(unittest.TestCase):
    def test_midpoints_reuse_nearest_poses_but_their_own_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _stub(tmp)
            written = backend.render_erp_panoramas_interpolated()
            self.assertEqual(written, 2)  # midpoints of 3 observed times

            overrides = sorted({o for _, _, o in backend.asked})
            self.assertEqual(overrides, [0.5, 1.5])
            # every face of a midpoint borrows an *observed* pose
            for pose_time, _, override in backend.asked:
                self.assertIn(pose_time, backend.dynamic_observed_times)
                self.assertAlmostEqual(abs(pose_time - override), 0.5)

    def test_interpolated_frames_are_tagged_and_kept_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _stub(tmp)
            backend.render_erp_panoramas()
            backend.render_erp_panoramas_interpolated()
            renders = Path(tmp) / "renders"
            observed = sorted(p.name for p in (renders / "erp_panorama").iterdir())
            interp = sorted(p.name for p in
                            (renders / "erp_panorama_interp").iterdir())
            self.assertEqual(len(observed), 3)
            self.assertEqual(len(interp), 2)
            # the two sets must not collide, or the deliverable overwrites itself
            self.assertFalse(set(observed) & set(interp))
            self.assertTrue(all("_interp" in n for n in interp), interp)

    def test_every_face_is_rendered_for_a_midpoint(self):
        """A panorama missing a face is a panorama with a hole in it."""
        with tempfile.TemporaryDirectory() as tmp:
            backend = _stub(tmp)
            backend.render_erp_panoramas_interpolated()
            for midpoint in (0.5, 1.5):
                faces = {cam for _, cam, o in backend.asked if o == midpoint}
                self.assertEqual(faces, {0, 1, 2, 3})

    def test_static_maps_produce_nothing(self):
        """Interpolating a static map would duplicate frames, not prove anything."""
        with tempfile.TemporaryDirectory() as tmp:
            backend = _stub(tmp)
            backend.dynamic_mode = DYNAMIC_MODE_NONE
            self.assertEqual(backend.render_erp_panoramas_interpolated(), 0)
            self.assertEqual(backend.asked, [])

    def test_single_observation_produces_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = _stub(tmp, times=(0.0,))
            self.assertEqual(backend.render_erp_panoramas_interpolated(), 0)


if __name__ == "__main__":
    unittest.main()
