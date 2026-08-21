import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tools.make_synthetic_erp_room import SPHERE_RADIUS, sphere_center


class SyntheticMotionProfilesTest(unittest.TestCase):
    def test_default_sweep_preserves_original_endpoints(self):
        np.testing.assert_allclose(sphere_center(0, 20), [-2.5, 0.8, 1.0])
        np.testing.assert_allclose(sphere_center(19, 20), [2.5, 0.8, 1.0])

    def test_bounce_reverses_lateral_direction(self):
        start = sphere_center(0, 21, motion="bounce_x")
        midpoint = sphere_center(10, 21, motion="bounce_x")
        end = sphere_center(20, 21, motion="bounce_x")

        np.testing.assert_allclose(start, [-2.5, 0.8, 1.0])
        np.testing.assert_allclose(midpoint, [2.5, 0.8, 1.0])
        np.testing.assert_allclose(end, start)

    def test_diagonal_changes_depth_and_lateral_position(self):
        start = sphere_center(0, 20, motion="diagonal_xz")
        end = sphere_center(19, 20, motion="diagonal_xz")

        self.assertLess(start[0], end[0])
        self.assertGreater(start[2], end[2])

    def test_unknown_motion_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown sphere motion"):
            sphere_center(0, 20, motion="invalid")

    def test_one_cycle_reproduces_the_original_bounce(self):
        """Existing bounce datasets must stay reproducible value-for-value."""
        for t in range(21):
            np.testing.assert_allclose(
                sphere_center(t, 21, motion="bounce_x", cycles=1.0),
                sphere_center(t, 21, motion="bounce_x"))

    def test_more_cycles_raise_per_frame_displacement(self):
        """The whole point: bigger steps, same swept volume (section 3.30)."""
        def steps(cycles):
            centres = np.array([
                sphere_center(t, 20, motion="bounce_x", cycles=cycles)
                for t in range(20)])
            return np.linalg.norm(np.diff(centres, axis=0), axis=1), centres

        slow_steps, _ = steps(1.0)
        fast_steps, fast_centres = steps(3.5)

        # displacement per frame goes up by about the cycle ratio
        self.assertGreater(fast_steps.mean(), 2.0 * slow_steps.mean())
        # ...while the sphere stays inside the same swept volume. Compared against
        # the profile's own bounds, not against what cycles=1 happened to sample:
        # at 20 frames that one never lands on its turning point (max x 2.24).
        self.assertLessEqual(fast_centres[:, 0].max(), 2.5 + 1e-9)
        self.assertGreaterEqual(fast_centres[:, 0].min(), -2.5 - 1e-9)

    def test_per_frame_step_clears_one_sphere_diameter(self):
        """The measurement threshold stage 0 exists to cross.

        At the default the sphere moves 0.263 per frame against a diameter of
        1.2, so consecutive frames overlap ~78% and no dynamic representation is
        distinguishable from widening in time.
        """
        centres = np.array([
            sphere_center(t, 20, motion="bounce_x", cycles=3.5)
            for t in range(20)])
        step = np.linalg.norm(np.diff(centres, axis=0), axis=1).mean()
        # 1.51 vs a 1.2 diameter; turning frames pull the mean below the 1.84
        # median, which is why 2.5 cycles (0.96 diameters) was not enough
        self.assertGreater(step, 2.0 * SPHERE_RADIUS)

    def test_cycles_on_a_non_bounce_profile_is_rejected(self):
        """Silently ignoring it would make a dataset that is not what was asked."""
        with self.assertRaisesRegex(ValueError, "only meaningful for bounce_x"):
            sphere_center(0, 20, motion="sweep_x", cycles=2.5)


if __name__ == "__main__":
    unittest.main()
