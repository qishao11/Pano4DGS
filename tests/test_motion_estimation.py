"""Translation-only registration of adjacent dynamic banks.

This is the geometry-side alternative to learning velocity photometrically,
after section 3.35 showed the learned route is structurally blocked.
"""

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))

from gaussian.deform.motion_estimation import (  # noqa: E402
    estimate_bank_velocities,
    translation_icp,
    velocities_to_rows,
)


def _sphere(centre, count=200, radius=0.6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    directions = torch.randn((count, 3), generator=generator)
    directions = directions / directions.norm(dim=1, keepdim=True)
    return directions * radius + torch.tensor(centre, dtype=torch.float32)


class TranslationICPTest(unittest.TestCase):
    def test_recovers_a_known_shift(self):
        source = _sphere([0.0, 0.0, 0.0])
        shift = torch.tensor([0.7, -0.3, 0.2])
        recovered = translation_icp(source, source + shift, initial=shift * 0)
        torch.testing.assert_close(recovered, shift, atol=2e-2, rtol=0.0)

    def test_beats_the_centroid_when_only_part_of_the_object_is_seen(self):
        """The case that motivates this: each view seeds the facing surface only.

        Two banks then contain different patches, so their centroids differ by
        more than the object actually moved -- the bias the centroid estimator
        cannot see and registration can.
        """
        shift = torch.tensor([1.0, 0.0, 0.0])
        full = _sphere([0.0, 0.0, 0.0], count=400)
        # bank A sees the -x facing half, bank B (camera moved) the +x half
        source = full[full[:, 0] < 0.0]
        target = (full + shift)[(full[:, 0] > 0.0)]

        centroid_estimate = target.mean(dim=0) - source.mean(dim=0)
        icp_estimate = translation_icp(source, target, initial=centroid_estimate)

        centroid_error = (centroid_estimate - shift).norm()
        icp_error = (icp_estimate - shift).norm()
        self.assertLess(float(icp_error), float(centroid_error))

    def test_empty_cloud_is_handled(self):
        self.assertEqual(
            translation_icp(torch.zeros((0, 3)), _sphere([0.0, 0.0, 0.0])).shape,
            (3,))


class EstimateBankVelocitiesTest(unittest.TestCase):
    def test_constant_motion_over_three_banks(self):
        times = [0.0, 1.0, 2.0]
        clouds = [_sphere([t, 0.0, 0.0], seed=1) for t in times]
        xyz = torch.cat(clouds)
        object_ids = torch.zeros((xyz.shape[0], 1))
        source_times = torch.cat([
            torch.full((c.shape[0], 1), t) for c, t in zip(clouds, times)])

        estimates = estimate_bank_velocities(xyz, object_ids, source_times, times)

        self.assertEqual(len(estimates), 3)
        for (_, _), value in estimates.items():
            torch.testing.assert_close(
                value, torch.tensor([1.0, 0.0, 0.0]), atol=5e-2, rtol=0.0)

    def test_static_rows_get_no_velocity(self):
        times = [0.0, 1.0]
        dynamic = torch.cat([_sphere([t, 0.0, 0.0], count=50) for t in times])
        static = torch.zeros((10, 3))
        xyz = torch.cat([dynamic, static])
        object_ids = torch.cat([torch.zeros((dynamic.shape[0], 1)),
                                torch.full((10, 1), -1.0)])
        source_times = torch.cat([
            torch.full((50, 1), 0.0), torch.full((50, 1), 1.0),
            torch.full((10, 1), -1.0)])

        estimates = estimate_bank_velocities(xyz, object_ids, source_times, times)
        rows = velocities_to_rows(estimates, object_ids, source_times, xyz)

        self.assertTrue(bool((rows[-10:] == 0).all()))
        self.assertGreater(float(rows[:50].norm(dim=1).mean()), 0.5)


class CentroidEstimatorTest(unittest.TestCase):
    """``refine=False``: the ICP's own seed, without the refinement.

    Section 3.47 made this the production choice under ground-truth depth, where
    every bank is clean and the contamination ICP was introduced to survive
    (3.29/3.36) no longer exists.
    """

    def _sequence(self, count=200):
        times = [0.0, 1.0, 2.0]
        clouds = [_sphere([t, 0.0, 0.0], count=count, seed=3) for t in times]
        xyz = torch.cat(clouds)
        object_ids = torch.zeros((xyz.shape[0], 1))
        source_times = torch.cat([
            torch.full((c.shape[0], 1), t) for c, t in zip(clouds, times)])
        return xyz, object_ids, source_times, times

    def test_recovers_constant_motion_from_clean_banks(self):
        xyz, object_ids, source_times, times = self._sequence()
        estimates = estimate_bank_velocities(
            xyz, object_ids, source_times, times, refine=False)
        self.assertEqual(len(estimates), 3)
        for value in estimates.values():
            torch.testing.assert_close(
                value, torch.tensor([1.0, 0.0, 0.0]), atol=5e-2, rtol=0.0)

    def test_is_the_plain_centroid_difference(self):
        """Not merely close to it -- exactly it, or the config name is a lie."""
        xyz, object_ids, source_times, times = self._sequence()
        estimates = estimate_bank_velocities(
            xyz, object_ids, source_times, times, refine=False)
        centroids = [xyz[source_times.reshape(-1) == t].mean(dim=0)
                     for t in times]
        # middle bank: central difference over both neighbours
        expected = 0.5 * ((centroids[0] - centroids[1]) / (0.0 - 1.0)
                          + (centroids[2] - centroids[1]) / (2.0 - 1.0))
        torch.testing.assert_close(estimates[(0, 1.0)], expected,
                                   atol=1e-6, rtol=0.0)

    def test_centroid_uses_every_row_not_the_icp_subsample(self):
        """Subsampling exists for cdist; the centroid has no reason to pay it."""
        xyz, object_ids, source_times, times = self._sequence(count=300)
        few = estimate_bank_velocities(
            xyz, object_ids, source_times, times, max_points=20, refine=False)
        many = estimate_bank_velocities(
            xyz, object_ids, source_times, times, max_points=10000, refine=False)
        for key in few:
            torch.testing.assert_close(few[key], many[key], atol=0.0, rtol=0.0)

    def test_matches_icp_when_the_banks_are_identical_copies(self):
        """Clean, complete banks: the refinement has nothing left to correct.

        This is the ground-truth-depth case in miniature, and why 3.47 could drop
        the ICP without losing anything -- when both banks hold the same surface,
        their centroid difference already *is* the shift.
        """
        xyz, object_ids, source_times, times = self._sequence()
        icp = estimate_bank_velocities(xyz, object_ids, source_times, times)
        centroid = estimate_bank_velocities(
            xyz, object_ids, source_times, times, refine=False)
        for key in icp:
            torch.testing.assert_close(icp[key], centroid[key],
                                       atol=1e-3, rtol=0.0)

    def test_differs_from_icp_when_banks_see_different_patches(self):
        """Partial banks: the two estimators must not silently be the same thing.

        Each view seeds only the surface facing it, so consecutive banks hold
        different patches and their centroids differ by more than the object
        moved. That bias is exactly what the ICP was introduced to remove, and
        what the centroid estimator accepts in exchange for being cheaper and --
        on clean banks -- more accurate.
        """
        times = [0.0, 1.0]
        full = [_sphere([t, 0.0, 0.0], count=400, seed=4) for t in times]
        # bank 0 keeps the -x facing half, bank 1 the +x half: the camera moved
        banks = [full[0][full[0][:, 0] < 0.0],
                 full[1][full[1][:, 0] > 1.0]]
        xyz = torch.cat(banks)
        object_ids = torch.zeros((xyz.shape[0], 1))
        source_times = torch.cat([
            torch.full((b.shape[0], 1), t) for b, t in zip(banks, times)])

        icp = estimate_bank_velocities(xyz, object_ids, source_times, times)
        centroid = estimate_bank_velocities(
            xyz, object_ids, source_times, times, refine=False)

        self.assertTrue(
            any(float((icp[k] - centroid[k]).norm()) > 1e-3 for k in icp),
            "refine=False produced the same numbers as ICP on partial banks")
        # and on partial banks the ICP is the more accurate of the two (3.36)
        truth = torch.tensor([1.0, 0.0, 0.0])
        key = (0, 0.0)
        self.assertLess(float((icp[key] - truth).norm()),
                        float((centroid[key] - truth).norm()))


class DeterminismTest(unittest.TestCase):
    def test_estimates_are_reproducible(self):
        """Same checkpoint must give the same velocity, every time.

        The subsampling used to be a randperm, so repeated estimation on one
        finished map produced different velocities -- noise that would show up in
        every downstream number without ever announcing itself.
        """
        times = [0.0, 1.0, 2.0]
        # more points than max_points, to force the subsampling path
        clouds = [_sphere([t, 0.0, 0.0], count=300, seed=2) for t in times]
        xyz = torch.cat(clouds)
        object_ids = torch.zeros((xyz.shape[0], 1))
        source_times = torch.cat([
            torch.full((c.shape[0], 1), t) for c, t in zip(clouds, times)])

        first = estimate_bank_velocities(
            xyz, object_ids, source_times, times, max_points=100)
        second = estimate_bank_velocities(
            xyz, object_ids, source_times, times, max_points=100)

        self.assertEqual(sorted(first), sorted(second))
        for key in first:
            torch.testing.assert_close(first[key], second[key],
                                       atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
