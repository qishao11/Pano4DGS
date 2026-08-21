"""Carrying dynamic Gaussians from their observed time to a requested time.

The per-timestamp oracle bank multiplies a dynamic Gaussian's opacity by zero
unless its source time matches the render time (oracle_motion_gate.
time_slice_opacities), so at an unobserved time the object vanishes entirely.
transform_to_time replaces that switch with the object's own relative motion,
and must reduce to the identity at observed times so it does not perturb any
already-recorded result.
"""

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))

from gaussian.deform.object_trajectory import ObjectTrajectoryTable  # noqa: E402


def _linear_table():
    """Object 0 moves from the origin to (2,0,0) between t=0 and t=2."""
    table = ObjectTrajectoryTable()
    table.observe(0, 0.0, [0.0, 0.0, 0.0])
    table.observe(0, 2.0, [2.0, 0.0, 0.0])
    return table


class TransformToTimeTest(unittest.TestCase):
    def test_identity_at_the_observed_time(self):
        """The property that keeps this backward compatible with the bank."""
        table = _linear_table()
        xyz = torch.tensor([[0.3, 0.4, 0.5], [1.0, -1.0, 2.0]])
        object_ids = torch.tensor([0.0, 0.0])
        source_times = torch.tensor([1.0, 1.0])

        moved = table.transform_to_time(xyz, object_ids, source_times, 1.0)

        torch.testing.assert_close(moved, xyz, atol=1e-6, rtol=0.0)

    def test_static_rows_are_untouched(self):
        table = _linear_table()
        xyz = torch.tensor([[5.0, 5.0, 5.0], [0.0, 0.0, 0.0]])
        object_ids = torch.tensor([-1.0, 0.0])
        source_times = torch.tensor([-1.0, 0.0])

        moved = table.transform_to_time(xyz, object_ids, source_times, 2.0)

        torch.testing.assert_close(moved[0], xyz[0], atol=1e-6, rtol=0.0)
        # the dynamic row did move
        self.assertGreater((moved[1] - xyz[1]).abs().sum().item(), 1.0)

    def test_translation_is_carried_by_the_object_motion(self):
        table = _linear_table()
        xyz = torch.tensor([[0.0, 0.0, 0.0]])
        object_ids = torch.tensor([0.0])
        source_times = torch.tensor([0.0])

        moved = table.transform_to_time(xyz, object_ids, source_times, 1.0)

        # halfway along a 2-unit sweep
        torch.testing.assert_close(
            moved, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-5, rtol=0.0)

    def test_rows_with_different_source_times_are_grouped_correctly(self):
        """Rows of one object can come from different observed times; each must
        be carried by its own relative transform, not a shared one."""
        table = _linear_table()
        xyz = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        object_ids = torch.tensor([0.0, 0.0])
        source_times = torch.tensor([0.0, 2.0])

        moved = table.transform_to_time(xyz, object_ids, source_times, 2.0)

        # row 0 observed at t=0 is carried the full +2; row 1 already at t=2
        torch.testing.assert_close(
            moved[0], torch.tensor([2.0, 0.0, 0.0]), atol=1e-5, rtol=0.0)
        torch.testing.assert_close(
            moved[1], torch.tensor([0.0, 0.0, 0.0]), atol=1e-6, rtol=0.0)

    def test_rotation_is_applied_about_the_object_frame(self):
        table = ObjectTrajectoryTable()
        table.observe(0, 0.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        # 180 degrees about z
        table.observe(0, 1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])

        xyz = torch.tensor([[1.0, 0.0, 0.0]])
        moved = table.transform_to_time(
            xyz, torch.tensor([0.0]), torch.tensor([0.0]), 1.0)

        torch.testing.assert_close(
            moved, torch.tensor([[-1.0, 0.0, 0.0]]), atol=1e-5, rtol=0.0)

    def test_defined_at_unobserved_times_where_the_bank_is_not(self):
        """The capability the per-timestamp bank lacks entirely."""
        table = _linear_table()
        xyz = torch.tensor([[0.0, 0.0, 0.0]])
        object_ids = torch.tensor([0.0])
        source_times = torch.tensor([0.0])

        for unobserved in (0.25, 0.5, 1.3, 1.75):
            moved = table.transform_to_time(
                xyz, object_ids, source_times, unobserved)
            self.assertTrue(torch.isfinite(moved).all())
            torch.testing.assert_close(
                moved, torch.tensor([[unobserved, 0.0, 0.0]]),
                atol=1e-5, rtol=0.0)

    def test_rotating_object_also_carries_gaussian_orientations(self):
        """Carrying only centres leaves an anisotropic Gaussian in the right
        place with a stale covariance. The per-timestamp bank got this right for
        free by storing a separate row per time."""
        table = ObjectTrajectoryTable()
        table.observe(0, 0.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        table.observe(0, 1.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])  # 180deg z

        xyz = torch.tensor([[1.0, 0.0, 0.0]])
        gaussian_rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

        moved, moved_rotations = table.transform_to_time(
            xyz, torch.tensor([0.0]), torch.tensor([0.0]), 1.0,
            rotations=gaussian_rotations)

        torch.testing.assert_close(
            moved, torch.tensor([[-1.0, 0.0, 0.0]]), atol=1e-5, rtol=0.0)
        # the Gaussian's own orientation must have picked up the same 180deg
        self.assertAlmostEqual(abs(moved_rotations[0, 3].item()), 1.0, places=4)
        self.assertAlmostEqual(moved_rotations[0, 0].item(), 0.0, places=4)

    def test_orientation_transport_is_identity_at_the_observed_time(self):
        table = _linear_table()
        gaussian_rotations = torch.tensor([[0.6, 0.8, 0.0, 0.0]])
        _, moved_rotations = table.transform_to_time(
            torch.zeros((1, 3)), torch.tensor([0.0]), torch.tensor([1.0]), 1.0,
            rotations=gaussian_rotations)
        torch.testing.assert_close(
            moved_rotations, gaussian_rotations, atol=1e-5, rtol=0.0)

    def test_missing_trajectory_raises_instead_of_silently_leaving_rows(self):
        """A row stamped with an object that has no trajectory cannot be placed;
        leaving it at a stale position would render wrongly with no diagnostic.
        Matches transform_gaussians/relative_transform."""
        table = ObjectTrajectoryTable()  # nothing registered
        with self.assertRaises(KeyError):
            table.transform_to_time(
                torch.zeros((1, 3)), torch.tensor([0.0]), torch.tensor([0.0]),
                0.5)

    def test_new_object_is_added_to_the_optimizer(self):
        """observe()'s create branch used to drop the optimizer, so a newly
        seen object's trajectory silently never trained."""
        table = ObjectTrajectoryTable()
        table.observe(0, 0.0, [0.0, 0.0, 0.0])
        optimizer = torch.optim.Adam(table.parameters(), lr=0.1)

        table.observe(7, 0.0, [1.0, 0.0, 0.0], optimizer=optimizer)

        tracked = {id(t) for group in optimizer.param_groups
                   for t in group["params"]}
        new_object = {id(p) for p in table.trajectories["7"].parameters()}
        self.assertTrue(new_object <= tracked,
                        "new object's knots were never given to the optimizer")

    def test_new_trajectory_lands_on_the_table_device(self):
        """nn.ModuleDict.__setitem__ does not inherit the parent's device, so a
        table already moved to a device would otherwise hold CPU submodules."""
        table = ObjectTrajectoryTable()
        table.observe(0, 0.0, [0.0, 0.0, 0.0])
        expected = next(table.parameters()).device

        table.observe(1, 0.0, [1.0, 0.0, 0.0])

        for parameter in table.trajectories["1"].parameters():
            self.assertEqual(parameter.device.type, expected.type)
        for buffer in table.trajectories["1"].buffers():
            self.assertEqual(buffer.device.type, expected.type)

    def test_is_differentiable_to_the_trajectory(self):
        table = _linear_table()
        xyz = torch.tensor([[0.5, 0.0, 0.0]], requires_grad=True)

        table.transform_to_time(
            xyz, torch.tensor([0.0]), torch.tensor([0.0]), 1.0).sum().backward()

        self.assertTrue(torch.isfinite(xyz.grad).all())
        touched = [p for p in table.parameters() if p.grad is not None]
        self.assertTrue(touched, "no trajectory parameter received a gradient")
        for parameter in touched:
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
