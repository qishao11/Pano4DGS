import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))

from gaussian.deform.object_trajectory import (
    ObjectSE3Trajectory,
    ObjectTrajectoryTable,
    quaternion_slerp,
)


class ObjectTrajectoryTest(unittest.TestCase):
    def test_linear_translation_interpolation(self):
        trajectory = ObjectSE3Trajectory(
            [0.0, 2.0],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        )

        translation, rotation = trajectory.evaluate(0.5)

        torch.testing.assert_close(translation, torch.tensor([1.0, 0.0, 0.0]))
        torch.testing.assert_close(rotation, torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def test_slerp_rotates_halfway_about_z(self):
        trajectory = ObjectSE3Trajectory(
            [0.0, 2.0],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
        )

        transformed = trajectory.transform_points(
            torch.tensor([[1.0, 0.0, 0.0]]), 1.0)

        torch.testing.assert_close(
            transformed,
            torch.tensor([[0.0, 1.0, 0.0]]),
            atol=1e-6,
            rtol=0.0,
        )

    def test_slerp_uses_quaternion_shortest_path(self):
        q = torch.tensor([0.9238795, 0.0, 0.3826834, 0.0])
        interpolated = quaternion_slerp(q, -q, 0.5)
        torch.testing.assert_close(interpolated, q, atol=1e-6, rtol=0.0)

    def test_repeated_time_averages_multiview_observations(self):
        trajectory = ObjectSE3Trajectory([1.0], [[0.0, 0.0, 0.0]])

        index, inserted = trajectory.add_observation(
            1.0, [2.0, 0.0, 0.0])

        self.assertEqual(index, 0)
        self.assertFalse(inserted)
        torch.testing.assert_close(
            trajectory.translations[0], torch.tensor([1.0, 0.0, 0.0]))
        self.assertEqual(trajectory.observation_counts.tolist(), [2])

    def test_new_observation_is_inserted_in_time_order(self):
        trajectory = ObjectSE3Trajectory([2.0], [[2.0, 0.0, 0.0]])
        trajectory.add_observation(0.0, [0.0, 0.0, 0.0])
        trajectory.add_observation(1.0, [1.0, 0.0, 0.0])

        self.assertEqual(trajectory.knot_times.tolist(), [0.0, 1.0, 2.0])
        translation, _ = trajectory.evaluate(0.5)
        torch.testing.assert_close(
            translation, torch.tensor([0.5, 0.0, 0.0]))

    def test_table_transforms_dynamic_and_preserves_static_rows(self):
        table = ObjectTrajectoryTable()
        table.observe(3, 0.0, [0.0, 0.0, 0.0])
        table.observe(3, 1.0, [2.0, 0.0, 0.0])
        xyz = torch.tensor([[5.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        object_ids = torch.tensor([-1, 3])

        transformed = table.transform_gaussians(xyz, object_ids, 0.5)

        torch.testing.assert_close(
            transformed,
            torch.tensor([[5.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        )

    def test_lifecycle_hides_objects_outside_observed_times(self):
        table = ObjectTrajectoryTable()
        table.observe(0, 1.0, [0.0, 0.0, 0.0])
        table.observe(0, 2.0, [1.0, 0.0, 0.0])
        ids = torch.tensor([-1, 0, 0])

        torch.testing.assert_close(
            table.visibility_mask(ids, 0.5),
            torch.tensor([True, False, False]),
        )
        torch.testing.assert_close(
            table.visibility_mask(ids, 1.5),
            torch.tensor([True, True, True]),
        )

    def test_checkpoint_round_trip_preserves_multiple_objects(self):
        table = ObjectTrajectoryTable()
        table.observe(0, 0.0, [0.0, 0.0, 0.0])
        table.observe(0, 1.0, [1.0, 0.0, 0.0])
        table.observe(4, 2.0, [0.0, 3.0, 0.0])

        restored = ObjectTrajectoryTable()
        restored.restore_checkpoint_state(table.checkpoint_state())

        self.assertEqual(restored.object_ids, [0, 4])
        expected, _ = table.evaluate(0, 0.25)
        actual, _ = restored.evaluate(0, 0.25)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
