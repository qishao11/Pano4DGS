"""Render-time placement of object-owned Gaussians (object_se3 mode).

The property under test is the capability gap against the per-timestamp oracle
bank: at an observed time both must render the same thing (so no recorded result
moves), while at an unobserved time the bank renders no dynamic Gaussians at all
and this path renders them at an interpolated pose.
"""

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))

from gaussian.deform.object_render import (  # noqa: E402
    nearest_observed_time,
    object_se3_overrides,
    observed_times_from_trajectories,
)
from gaussian.deform.object_trajectory import ObjectTrajectoryTable  # noqa: E402
from gaussian.deform.oracle_motion_gate import time_slice_opacities  # noqa: E402


class FakeGaussians:
    """The handful of GaussianModel fields the render override path reads.

    GaussianModel itself pulls in CUDA-only extensions (simple_knn), so the
    placement logic is exercised on CPU tensors here.
    """

    def __init__(self, xyz, object_ids, source_times, rotations=None,
                 opacities=None):
        self._xyz = torch.as_tensor(xyz, dtype=torch.float32)
        rows = self._xyz.shape[0]
        self.dynamic_object_id = torch.as_tensor(
            object_ids, dtype=torch.float32).reshape(rows, 1)
        self.dynamic_source_time = torch.as_tensor(
            source_times, dtype=torch.float32).reshape(rows, 1)
        self.dynamic_score = (self.dynamic_object_id >= 0).float()
        if rotations is None:
            rotations = torch.zeros((rows, 4))
            rotations[:, 0] = 1.0
        self._rotation = torch.as_tensor(rotations, dtype=torch.float32)
        if opacities is None:
            opacities = torch.full((rows, 1), 0.8)
        self._opacity = torch.as_tensor(opacities, dtype=torch.float32)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_rotation(self):
        return torch.nn.functional.normalize(self._rotation)

    @property
    def get_opacity(self):
        return self._opacity


def _sweep_table():
    """Object 0 sweeps from the origin to (2,0,0) between t=0 and t=2."""
    table = ObjectTrajectoryTable()
    table.observe(0, 0.0, [0.0, 0.0, 0.0])
    table.observe(0, 2.0, [2.0, 0.0, 0.0])
    return table


def _two_bank_scene():
    """One static row plus one dynamic row seeded at each of t=0 and t=2."""
    return FakeGaussians(
        xyz=[[5.0, 5.0, 5.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        object_ids=[-1.0, 0.0, 0.0],
        source_times=[-1.0, 0.0, 2.0],
    )


class NearestObservedTimeTest(unittest.TestCase):
    def test_no_observations_yields_none(self):
        self.assertIsNone(nearest_observed_time([], 1.0))
        self.assertIsNone(nearest_observed_time(None, 1.0))

    def test_an_observed_time_picks_itself(self):
        self.assertEqual(nearest_observed_time([0.0, 2.0, 4.0], 2.0), 2.0)

    def test_an_unobserved_time_picks_the_closest_bank(self):
        self.assertEqual(nearest_observed_time([0.0, 2.0, 4.0], 2.4), 2.0)
        self.assertEqual(nearest_observed_time([0.0, 2.0, 4.0], 3.6), 4.0)


class ObjectSE3OverridesTest(unittest.TestCase):
    def test_nothing_to_place_returns_no_overrides(self):
        """Callers splat the result into render(), so it must be empty, not None."""
        gaussians = FakeGaussians(
            xyz=[[1.0, 1.0, 1.0]], object_ids=[-1.0], source_times=[-1.0])
        self.assertEqual(
            object_se3_overrides(gaussians, ObjectTrajectoryTable(), [], 1.0), {})
        self.assertEqual(
            object_se3_overrides(gaussians, _sweep_table(), [], 1.0), {})

    def test_observed_time_reproduces_the_bank(self):
        """No recorded time-slice result may move: the transform is the identity."""
        gaussians = _two_bank_scene()
        overrides = object_se3_overrides(
            gaussians, _sweep_table(), [0.0, 2.0], 2.0)

        torch.testing.assert_close(
            overrides["means3D_override"], gaussians.get_xyz,
            atol=1e-5, rtol=0.0)
        torch.testing.assert_close(
            overrides["opacities_override"],
            time_slice_opacities(
                gaussians.get_opacity, gaussians.dynamic_score,
                gaussians.dynamic_source_time, 2.0),
            atol=0.0, rtol=0.0)

    def test_unobserved_time_places_the_object_where_the_bank_shows_nothing(self):
        gaussians = _two_bank_scene()
        overrides = object_se3_overrides(
            gaussians, _sweep_table(), [0.0, 2.0], 1.4)

        # nearest bank is t=2 (the row at (2,0,0)); the object is at x=1.4 then,
        # so that row moves back by 0.6
        moved = overrides["means3D_override"]
        torch.testing.assert_close(
            moved[2], torch.tensor([1.4, 0.0, 0.0]), atol=1e-5, rtol=0.0)
        self.assertGreater(overrides["opacities_override"][2].item(), 0.0)
        # what the per-timestamp bank alone would have done at this time
        bank = time_slice_opacities(
            gaussians.get_opacity, gaussians.dynamic_score,
            gaussians.dynamic_source_time, 1.4)
        self.assertEqual(bank[1].item(), 0.0)
        self.assertEqual(bank[2].item(), 0.0)

    def test_only_the_reference_bank_is_shown_and_moved(self):
        """Otherwise every bank of the same object would stack at one place."""
        gaussians = _two_bank_scene()
        overrides = object_se3_overrides(
            gaussians, _sweep_table(), [0.0, 2.0], 1.4)

        opacities = overrides["opacities_override"]
        self.assertEqual(opacities[1].item(), 0.0)      # t=0 bank stays hidden
        self.assertGreater(opacities[2].item(), 0.0)    # t=2 bank is the reference
        # the hidden bank is also left alone rather than moved
        torch.testing.assert_close(
            overrides["means3D_override"][1], gaussians.get_xyz[1],
            atol=0.0, rtol=0.0)

    def test_static_rows_are_neither_moved_nor_hidden(self):
        gaussians = _two_bank_scene()
        overrides = object_se3_overrides(
            gaussians, _sweep_table(), [0.0, 2.0], 1.4)

        torch.testing.assert_close(
            overrides["means3D_override"][0], gaussians.get_xyz[0],
            atol=0.0, rtol=0.0)
        self.assertEqual(
            overrides["opacities_override"][0].item(),
            gaussians.get_opacity[0].item())

    def test_gaussian_orientation_is_carried_for_a_rotating_object(self):
        """Position without orientation leaves a stale covariance (section 3.28)."""
        table = ObjectTrajectoryTable()
        table.observe(0, 0.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0])
        # 180 degrees about z
        table.observe(0, 2.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        gaussians = FakeGaussians(
            xyz=[[1.0, 0.0, 0.0]], object_ids=[0.0], source_times=[0.0])

        # t=0.8 is 40% of the way, i.e. 72 degrees of the 180
        overrides = object_se3_overrides(gaussians, table, [0.0, 2.0], 0.8)

        angle = torch.tensor(0.4 * torch.pi)
        torch.testing.assert_close(
            overrides["means3D_override"],
            torch.tensor([[torch.cos(angle), torch.sin(angle), 0.0]]),
            atol=1e-5, rtol=0.0)
        # the Gaussian's own orientation turns by the same 72 degrees
        rotated = overrides["rotations_override"][0]
        self.assertAlmostEqual(rotated[0].item(), torch.cos(angle / 2).item(),
                               places=5)
        self.assertAlmostEqual(rotated[3].item(), torch.sin(angle / 2).item(),
                               places=5)

    def test_group_fast_path_matches_the_generic_scan(self):
        """The render path skips transform_to_time's device->host unique scans."""
        gaussians = _two_bank_scene()
        table = _sweep_table()
        overrides = object_se3_overrides(gaussians, table, [0.0, 2.0], 1.4)

        # the same request expressed without `groups`: hide the non-reference
        # bank by marking it static, exactly as the override path does
        object_ids = torch.tensor([-1.0, -1.0, 0.0])
        generic = table.transform_to_time(
            gaussians.get_xyz, object_ids, gaussians.dynamic_source_time, 1.4)

        torch.testing.assert_close(
            overrides["means3D_override"], generic, atol=1e-6, rtol=0.0)

    def test_gradients_reach_the_canonical_positions(self):
        """Moved rows must stay differentiable w.r.t. the Gaussians' own xyz."""
        gaussians = _two_bank_scene()
        gaussians._xyz.requires_grad_(True)

        overrides = object_se3_overrides(
            gaussians, _sweep_table(), [0.0, 2.0], 1.4)
        overrides["means3D_override"].sum().backward()

        self.assertIsNotNone(gaussians._xyz.grad)
        self.assertTrue(torch.isfinite(gaussians._xyz.grad).all())


class RenderContractTest(unittest.TestCase):
    def test_override_keys_are_accepted_by_render(self):
        """The overrides are splatted into render(**overrides).

        Both call sites (gs_backend.render_at and eval_utils) pass this dict
        straight through, so a renamed or misspelled key is a TypeError that
        only surfaces on a real GPU run.
        """
        import inspect

        from gaussian.renderer import render

        accepted = set(inspect.signature(render).parameters)
        overrides = object_se3_overrides(
            _two_bank_scene(), _sweep_table(), [0.0, 2.0], 1.4)

        self.assertTrue(overrides)
        self.assertLessEqual(set(overrides), accepted)


class ObservedTimesFromTrajectoriesTest(unittest.TestCase):
    def test_knot_times_round_trip(self):
        self.assertEqual(
            observed_times_from_trajectories(_sweep_table()), [0.0, 2.0])

    def test_empty_table_has_no_observed_times(self):
        self.assertEqual(
            observed_times_from_trajectories(ObjectTrajectoryTable()), [])

    def test_survives_a_checkpoint_round_trip(self):
        """Observed times are not stored; a resumed run rebuilds them from knots.

        If that rebuild lost a time, the resumed map would pick a different
        reference bank than the run that saved it -- a silent divergence, which
        is exactly the class of checkpoint bug section 3.28 had to chase down.
        """
        table = ObjectTrajectoryTable()
        table.observe_centroids(
            torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [9.0, 9.0, 9.0]]),
            torch.tensor([0.0, 0.0, -1.0]),
            torch.tensor([0.0, 3.0, -1.0]))
        before = observed_times_from_trajectories(table)

        restored = ObjectTrajectoryTable()
        restored.restore_checkpoint_state(table.checkpoint_state())

        self.assertEqual(before, [0.0, 3.0])
        self.assertEqual(observed_times_from_trajectories(restored), before)

    def test_centroids_become_the_knots(self):
        """The knot is the centroid of that bank, and static rows are excluded."""
        table = ObjectTrajectoryTable()
        registered = table.observe_centroids(
            torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [9.0, 9.0, 9.0]]),
            torch.tensor([0.0, 0.0, -1.0]),
            torch.tensor([5.0, 5.0, -1.0]))

        self.assertEqual(registered, [5.0])
        translation, _ = table.evaluate(0, 5.0)
        torch.testing.assert_close(
            translation, torch.tensor([1.0, 0.0, 0.0]), atol=1e-6, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
