"""Fitting a reconstruction onto the synthetic room's six known walls.

This exists to break a circularity: placement accuracy used to be measured
against a similarity fitted on the dynamic object's own centroids, which landed
on roughly half the scene's true scale (panoramic_4dgs_status.md 3.51). Aligning
to the static room instead is independent of the object being measured.

The tests below run on synthetic wall points rather than a checkpoint, so they
pin the estimator's behaviour without needing a trained map.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from align_map_to_room import (  # noqa: E402
    HX, HY, HZ,
    apply,
    fit,
    nearest_plane_residual,
    rodrigues,
    room_planes,
)


def wall_points(count=4000, seed=0):
    """Points scattered over the six walls of the true room."""
    rng = np.random.default_rng(seed)
    half = np.array([HX, HY, HZ])
    points = rng.uniform(-half, half, size=(count, 3))
    axis = rng.integers(0, 3, size=count)
    side = rng.choice([-1.0, 1.0], size=count)
    points[np.arange(count), axis] = side * half[axis]
    return points


class PlaneGeometryTest(unittest.TestCase):
    def test_wall_points_sit_on_the_walls(self):
        residual, _ = nearest_plane_residual(wall_points(), room_planes())
        self.assertLess(float(np.abs(residual).max()), 1e-9)

    def test_rodrigues_is_a_rotation(self):
        for vector in ([0.0, 0.0, 0.0], [0.1, -0.2, 0.05], [1.0, 1.0, 1.0]):
            rotation = rodrigues(np.array(vector))
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3),
                                       atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)


class FitTest(unittest.TestCase):
    def _recover(self, true_scale, true_rotvec, true_translation, seed=0,
                 noise=0.0, **kwargs):
        """Distort a known room, then check the fit undoes it."""
        truth = wall_points(seed=seed)
        rotation = rodrigues(np.array(true_rotvec))
        # points as they would appear in the map's own frame
        source = (truth - true_translation) @ rotation / true_scale
        if noise:
            source = source + np.random.default_rng(1).normal(
                0.0, noise, size=source.shape)
        return fit(source, true_scale, np.eye(3), np.array(true_translation),
                   room_planes(), **kwargs), source

    def test_recovers_a_pure_scale(self):
        (scale, rotation, translation), source = self._recover(
            4.75, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(scale, 4.75, places=3)
        residual, _ = nearest_plane_residual(
            apply(source, scale, rotation, translation), room_planes())
        self.assertLess(float(np.median(np.abs(residual))), 1e-3)

    def test_recovers_a_small_rotation(self):
        """The sign of the rotation Jacobian: getting it backwards once sent the
        fit to a 96-degree 'solution' that fit nothing."""
        (scale, rotation, translation), source = self._recover(
            4.0, [0.03, -0.02, 0.01], [0.1, 0.0, -1.5])
        residual, _ = nearest_plane_residual(
            apply(source, scale, rotation, translation), room_planes())
        self.assertLess(float(np.median(np.abs(residual))), 5e-2)

    def test_never_returns_a_worse_fit_than_it_started_with(self):
        """Correspondences are recomputed each step, so a step can improve its
        own linearisation and the real objective at the same time get worse."""
        planes = room_planes()
        source = wall_points(seed=3) / 4.0
        def rms(s, r, t):
            residual, _ = nearest_plane_residual(apply(source, s, r, t), planes)
            return float(np.sqrt((residual ** 2).mean()))
        start = (2.0, np.eye(3), np.zeros(3))   # deliberately bad scale
        before = rms(*start)
        after = rms(*fit(source, *start, planes))
        self.assertLessEqual(after, before)

    def test_survives_outliers(self):
        """Floaters in the room's interior must not set the scale (3.29)."""
        truth = wall_points(seed=4)
        rng = np.random.default_rng(5)
        floaters = rng.uniform([-HX, -HY, -HZ], [HX, HY, HZ], size=(1200, 3))
        source = np.concatenate([truth, floaters]) / 4.75
        scale, rotation, translation = fit(
            source, 4.75, np.eye(3), np.zeros(3), room_planes(), trim=0.3)
        self.assertAlmostEqual(scale, 4.75, delta=0.15)

    def test_noise_does_not_bias_the_scale(self):
        (scale, _, _), _ = self._recover(
            4.75, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], noise=0.01)
        self.assertAlmostEqual(scale, 4.75, delta=0.1)


if __name__ == "__main__":
    unittest.main()
