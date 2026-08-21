"""Training-path tests for the object-level SE(3) trajectory.

tests/test_object_trajectory.py covers forward correctness only -- none of its
cases touch backward/grad/optimizer, and the module is never exercised by a
training loop in the main code (gs_backend.py only instantiates the table).
These tests cover the gradient path instead, which is where the representation
has to work if it is ever going to replace the per-timestamp oracle bank.
"""

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))

from gaussian.deform.object_trajectory import (  # noqa: E402
    ObjectSE3Trajectory,
    ObjectTrajectoryTable,
    quaternion_slerp,
)


class SlerpGradientTest(unittest.TestCase):
    def test_gradients_finite_for_near_identical_knots(self):
        """Adjacent trajectory knots are almost always near-identical rotations.

        acos'(dot) = -1/sqrt(1-dot^2) blows up at dot = +-1. torch.where picks
        the safe forward value but still backprops through *both* branches, so
        inf * 0 = NaN reaches the inputs unless dot is clamped away from +-1
        before acos.
        """
        q0 = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
        q1 = torch.tensor([1.0, 1e-7, 0.0, 0.0], requires_grad=True)

        quaternion_slerp(q0, q1, 0.5).sum().backward()

        self.assertTrue(torch.isfinite(q0.grad).all(),
                        f"q0 grad not finite: {q0.grad}")
        self.assertTrue(torch.isfinite(q1.grad).all(),
                        f"q1 grad not finite: {q1.grad}")

    def test_gradients_finite_for_exactly_identical_knots(self):
        q0 = torch.tensor([0.0, 0.0, 0.0, 1.0], requires_grad=True)
        q1 = torch.tensor([0.0, 0.0, 0.0, 1.0], requires_grad=True)

        quaternion_slerp(q0, q1, 0.3).sum().backward()

        self.assertTrue(torch.isfinite(q0.grad).all())
        self.assertTrue(torch.isfinite(q1.grad).all())

    def test_gradients_finite_for_well_separated_knots(self):
        """The spherical branch must keep working -- guard against a fix that
        simply routes everything through the linear branch."""
        q0 = torch.tensor([1.0, 0.0, 0.0, 0.0], requires_grad=True)
        q1 = torch.tensor([0.0, 1.0, 0.0, 0.0], requires_grad=True)

        quaternion_slerp(q0, q1, 0.5).sum().backward()

        self.assertTrue(torch.isfinite(q0.grad).all())
        self.assertTrue(torch.isfinite(q1.grad).all())
        self.assertGreater(q0.grad.abs().sum().item(), 0.0)


class TrajectoryOptimizerTest(unittest.TestCase):
    def test_add_observation_keeps_optimizer_parameters_live(self):
        """Inserting a knot must not orphan the optimizer it is handed.

        A leaf Parameter cannot be resized in place (its AccumulateGrad node
        caches the row count), so a new Parameter is unavoidable. The contract
        is therefore: hand add_observation the optimizer and it swaps the
        tensor inside param_groups, so the optimizer keeps stepping the tensor
        that actually feeds evaluate().
        """
        trajectory = ObjectSE3Trajectory([0.0, 1.0], [[0.0, 0.0, 0.0],
                                                      [1.0, 0.0, 0.0]])
        optimizer = torch.optim.SGD(trajectory.parameters(), lr=0.1)

        trajectory.add_observation(0.5, [0.5, 1.0, 0.0], optimizer=optimizer)

        optimized = {id(t) for group in optimizer.param_groups
                     for t in group["params"]}
        live = {id(t) for t in trajectory.parameters()}
        self.assertTrue(
            live <= optimized,
            "trajectory parameters are not the ones the optimizer holds; "
            "add_observation orphaned the optimizer")

    def test_adam_keeps_stepping_after_a_knot_is_inserted_mid_training(self):
        """Online knot insertion is the actual use case: knots arrive while the
        map is being optimized. Adam keeps per-parameter, row-indexed moment
        buffers, so they must grow with the knots or the next step() either
        throws on shape or silently uses stale moments."""
        trajectory = ObjectSE3Trajectory([0.0, 1.0], [[0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0]])
        optimizer = torch.optim.Adam(trajectory.parameters(), lr=0.05)
        target = torch.tensor([1.0, 1.0, 1.0])

        for _ in range(20):  # build up Adam moment state first
            optimizer.zero_grad()
            translation, _ = trajectory.evaluate(1.0)
            torch.nn.functional.mse_loss(translation, target).backward()
            optimizer.step()

        trajectory.add_observation(0.5, [0.4, 0.4, 0.4], optimizer=optimizer)
        self.assertEqual(trajectory.translations.shape[0], 3)

        for _ in range(200):
            optimizer.zero_grad()
            translation, _ = trajectory.evaluate(1.0)
            loss = torch.nn.functional.mse_loss(translation, target)
            loss.backward()
            optimizer.step()

        self.assertTrue(torch.isfinite(trajectory.translations).all())
        self.assertLess(loss.item(), 1e-4,
                        f"training broke after knot insertion, loss={loss}")

    def test_trajectory_fits_a_translation_by_gradient_descent(self):
        """End-to-end: the representation must actually be trainable."""
        trajectory = ObjectSE3Trajectory([0.0, 1.0], [[0.0, 0.0, 0.0],
                                                      [0.0, 0.0, 0.0]])
        target = torch.tensor([2.0, -1.0, 0.5])
        optimizer = torch.optim.Adam(trajectory.parameters(), lr=0.1)

        for _ in range(300):
            optimizer.zero_grad()
            translation, _ = trajectory.evaluate(1.0)
            loss = torch.nn.functional.mse_loss(translation, target)
            loss.backward()
            optimizer.step()

        final, _ = trajectory.evaluate(1.0)
        self.assertLess(loss.item(), 1e-4, f"failed to converge, loss={loss}")
        torch.testing.assert_close(final, target, atol=1e-2, rtol=1e-2)

    def test_interpolated_pose_carries_gradient_to_both_knots(self):
        """The whole point of this representation is interpolation, so a query
        at an unobserved time must produce gradients on the surrounding knots."""
        trajectory = ObjectSE3Trajectory([0.0, 2.0], [[0.0, 0.0, 0.0],
                                                      [2.0, 0.0, 0.0]])

        translation, rotation = trajectory.evaluate(1.0)
        (translation.sum() + rotation.sum()).backward()

        self.assertIsNotNone(trajectory.translations.grad)
        self.assertTrue(torch.isfinite(trajectory.translations.grad).all())
        self.assertTrue(torch.isfinite(trajectory.rotations.grad).all())
        # both surrounding knots must receive translation gradient
        self.assertGreater(trajectory.translations.grad[0].abs().sum().item(), 0.0)
        self.assertGreater(trajectory.translations.grad[1].abs().sum().item(), 0.0)

    def test_transform_gaussians_is_differentiable_through_the_table(self):
        table = ObjectTrajectoryTable()
        table.observe(0, 0.0, [0.0, 0.0, 0.0])
        table.observe(0, 1.0, [1.0, 0.0, 0.0])

        canonical = torch.tensor([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]],
                                 requires_grad=True)
        object_ids = torch.tensor([0, -1])

        table.transform_gaussians(canonical, object_ids, 0.5).sum().backward()

        self.assertTrue(torch.isfinite(canonical.grad).all())
        params = [p for p in table.parameters() if p.grad is not None]
        self.assertTrue(params, "no trajectory parameter received a gradient")
        for p in params:
            self.assertTrue(torch.isfinite(p.grad).all())


class TableObserveTest(unittest.TestCase):
    def test_first_observation_accepts_tensor_translation(self):
        """observe() forwards [translation] into ObjectSE3Trajectory, whose
        torch.as_tensor() call raises on a list holding a multi-element tensor.
        Callers naturally hold tensors, and add_observation already accepts
        them, so the two paths must agree."""
        table = ObjectTrajectoryTable()
        table.observe(0, 0.0, torch.tensor([1.0, 2.0, 3.0]))
        translation, _ = table.evaluate(0, 0.0)
        torch.testing.assert_close(translation, torch.tensor([1.0, 2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
