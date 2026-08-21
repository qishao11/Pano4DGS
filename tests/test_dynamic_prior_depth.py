"""Seeding the moving object from the depth prior instead of BA's disparity.

A dynamic pixel's multi-view constraint is invalid -- the object moved between
the frames being matched -- but bundle adjustment updates its disparity from
that constraint anyway. Section 3.52 measured the damage: 86.5% of dynamic
Gaussians land inside the object's silhouette and only 21.7% at the right
distance, so renders look right while the geometry is a diameter out.
Back-projecting the same pixels with the prior puts 100.0% of them on the
sphere (section 3.53).

The two properties worth pinning are that only dynamic pixels move, and that the
metric prior is reconciled with a non-metric map by the ratio the *static*
pixels agree on rather than by a constant somebody typed in.
"""

import sys
import types
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))

from gaussian.scene.gaussian_model import GaussianModel  # noqa: E402

YELLOW = [240, 220, 60]


class _Model:
    """Just the fields ``_depth_for_dynamic_pixels`` reads."""

    _depth_for_dynamic_pixels = GaussianModel._depth_for_dynamic_pixels

    def __init__(self, enabled=True, colors=(YELLOW,), threshold=0.1):
        self.oracle_dynamic_colors = [list(c) for c in colors]
        self.oracle_color_threshold = threshold
        self.dynamic_depth_from_prior = enabled


def scene(dynamic_columns=slice(0, 8), map_scale=0.25, height=16, width=32):
    """A grey wall with a yellow patch, plus a metric prior for both."""
    rgb = np.full((height, width, 3), 90, dtype=np.uint8)
    rgb[:, dynamic_columns] = YELLOW

    prior = np.full((height, width), 4.0, dtype=np.float32)   # metric depth
    prior[:, dynamic_columns] = 2.5                           # object is nearer

    # what BA produced: the static wall at the map's scale, the object wrong
    depth = prior * map_scale
    depth[:, dynamic_columns] = 9.9
    return rgb, depth, prior


class SwapTest(unittest.TestCase):
    def test_dynamic_pixels_take_the_prior_at_the_map_s_scale(self):
        rgb, depth, prior = scene()
        out = _Model()._depth_for_dynamic_pixels(rgb, depth, prior)
        np.testing.assert_allclose(out[:, 0:8], 2.5 * 0.25, rtol=1e-5)

    def test_static_pixels_are_untouched(self):
        rgb, depth, prior = scene()
        out = _Model()._depth_for_dynamic_pixels(rgb, depth, prior)
        np.testing.assert_array_equal(out[:, 8:], depth[:, 8:])

    def test_the_scale_is_calibrated_not_assumed(self):
        """A different map scale must come out right without touching the code."""
        for map_scale in (1.0, 0.25, 4.0):
            rgb, depth, prior = scene(map_scale=map_scale)
            out = _Model()._depth_for_dynamic_pixels(rgb, depth, prior)
            np.testing.assert_allclose(out[:, 0:8], 2.5 * map_scale, rtol=1e-5)

    def test_the_input_is_not_mutated(self):
        rgb, depth, prior = scene()
        before = depth.copy()
        _Model()._depth_for_dynamic_pixels(rgb, depth, prior)
        np.testing.assert_array_equal(depth, before)


class DisabledTest(unittest.TestCase):
    def test_off_by_default_leaves_depth_alone(self):
        rgb, depth, prior = scene()
        out = _Model(enabled=False)._depth_for_dynamic_pixels(rgb, depth, prior)
        np.testing.assert_array_equal(out, depth)

    def test_no_prior_leaves_depth_alone(self):
        rgb, depth, _ = scene()
        out = _Model()._depth_for_dynamic_pixels(rgb, depth, None)
        np.testing.assert_array_equal(out, depth)

    def test_no_oracle_palette_leaves_depth_alone(self):
        rgb, depth, prior = scene()
        out = _Model(colors=())._depth_for_dynamic_pixels(rgb, depth, prior)
        np.testing.assert_array_equal(out, depth)


class DegenerateTest(unittest.TestCase):
    def test_a_frame_with_no_dynamic_pixels_is_unchanged(self):
        rgb, depth, prior = scene(dynamic_columns=slice(0, 0))
        out = _Model()._depth_for_dynamic_pixels(rgb, depth, prior)
        np.testing.assert_array_equal(out, depth)

    def test_too_few_static_pixels_to_calibrate_is_refused(self):
        """Better the old depth than a scale fitted to a handful of pixels."""
        rgb, depth, prior = scene(dynamic_columns=slice(0, 31))
        out = _Model()._depth_for_dynamic_pixels(rgb, depth, prior)
        np.testing.assert_array_equal(out, depth)

    def test_pixels_without_a_prior_keep_their_ba_depth(self):
        """0 in the prior means "never measured", not "at zero distance"."""
        rgb, depth, prior = scene()
        prior[0, 0:8] = 0.0
        out = _Model()._depth_for_dynamic_pixels(rgb, depth, prior)
        np.testing.assert_array_equal(out[0, 0:8], depth[0, 0:8])
        np.testing.assert_allclose(out[1, 0:8], 2.5 * 0.25, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
