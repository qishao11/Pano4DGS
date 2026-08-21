"""The synthetic sequences' camera trajectory, which was always analytic.

``camera_pose`` used to be inline in the generator's render loop, so the poses
were computed, used, and thrown away. On that basis the project recorded for
months that ATE could not be measured on synthetic data. These tests pin the
formulas down now that a second caller depends on them.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from make_synthetic_erp_room import camera_pose  # noqa: E402
from export_synthetic_gt_poses import quaternion_from_matrix  # noqa: E402


class CameraPoseTest(unittest.TestCase):
    def test_matches_the_formulas_the_render_loop_used(self):
        """Extracting the function must not have moved a single camera."""
        for t in range(20):
            position, _ = camera_pose(t)
            expected = np.array([0.3 * np.sin(t * 0.35),
                                 0.1 * np.sin(t * 0.5),
                                 -1.5 + 0.15 * t])
            np.testing.assert_allclose(position, expected, atol=1e-12)

    def test_rotation_is_orthonormal_and_right_handed(self):
        for t in (0, 7, 19):
            _, rotation = camera_pose(t)
            np.testing.assert_allclose(rotation @ rotation.T, np.eye(3),
                                       atol=1e-12)
            self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)

    def test_the_dolly_stays_inside_the_room(self):
        """HZ is 4.0; a camera outside the room would render nonsense."""
        for t in range(20):
            position, _ = camera_pose(t)
            self.assertLess(abs(position[2]), 4.0)
            self.assertLess(abs(position[0]), 4.0)
            self.assertLess(abs(position[1]), 2.2)

    def test_the_weave_actually_supplies_lateral_parallax(self):
        """A pure dolly gives a rig sharing one optical centre nothing (3.45)."""
        lateral = [camera_pose(t)[0][0] for t in range(20)]
        self.assertGreater(max(lateral) - min(lateral), 0.4)


class QuaternionTest(unittest.TestCase):
    def _rotation(self, axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        cross = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
        return (np.eye(3) + np.sin(angle) * cross
                + (1 - np.cos(angle)) * (cross @ cross))

    def _back(self, quat):
        x, y, z, w = quat
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    def test_round_trips_the_trajectory_rotations(self):
        for t in range(20):
            _, rotation = camera_pose(t)
            np.testing.assert_allclose(
                self._back(quaternion_from_matrix(rotation)), rotation, atol=1e-9)

    def test_round_trips_large_rotations_too(self):
        """The trace branch exists precisely for the cases the dolly never hits."""
        for angle in (np.pi / 2, np.pi - 1e-4, 2.0, -2.5):
            for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, -2, 0.5]):
                rotation = self._rotation(axis, angle)
                np.testing.assert_allclose(
                    self._back(quaternion_from_matrix(rotation)), rotation,
                    atol=1e-7)

    def test_quaternions_are_unit_norm(self):
        for t in range(20):
            _, rotation = camera_pose(t)
            self.assertAlmostEqual(
                float(np.linalg.norm(quaternion_from_matrix(rotation))), 1.0,
                places=9)


if __name__ == "__main__":
    unittest.main()
