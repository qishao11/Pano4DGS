"""Slicing native 4D Gaussians at a query time.

The properties that matter are continuity with what came before and the two
things the earlier representations could not do: be defined between observations
(the per-timestamp bank is not) without needing an object segmentation (the
SE(3) route does).
"""

import sys
import unittest
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))

from gaussian.deform.gaussian_4d import (  # noqa: E402
    pool_velocity,
    slice_at_time,
    temporal_weights,
    visible_rows,
    widen_for_interpolation,
)
from gaussian.deform.oracle_motion_gate import time_slice_opacities  # noqa: E402


def _scene():
    """Two dynamic rows observed at t=0 and t=2, plus one static row."""
    return {
        "xyz": torch.tensor([[5.0, 5.0, 5.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        "opacities": torch.full((3, 1), 0.8),
        "time_center": torch.tensor([-1.0, 0.0, 2.0]),
        "motion_score": torch.tensor([0.0, 1.0, 1.0]),
        "velocity": torch.zeros((3, 3)),
    }


class TemporalWeightsTest(unittest.TestCase):
    def test_zero_radius_reproduces_the_per_timestamp_bank(self):
        """The bank is the degenerate case, so no recorded result is at risk."""
        scene = _scene()
        source_time = scene["time_center"].reshape(-1, 1)
        for target in (0.0, 2.0, 1.0):
            weights = temporal_weights(
                scene["time_center"], torch.zeros(3), target,
                scene["motion_score"])
            bank = time_slice_opacities(
                scene["opacities"], scene["motion_score"].reshape(-1, 1),
                source_time, target)
            torch.testing.assert_close(
                scene["opacities"] * weights[:, None], bank,
                atol=1e-6, rtol=0.0)

    def test_static_rows_are_present_at_every_time(self):
        scene = _scene()
        weights = temporal_weights(
            scene["time_center"], torch.ones(3), 100.0, scene["motion_score"])
        self.assertEqual(weights[0].item(), 1.0)
        self.assertLess(weights[1].item(), 1e-3)

    def test_weight_peaks_at_the_gaussian_own_moment(self):
        scene = _scene()
        at_centre = temporal_weights(
            scene["time_center"], torch.ones(3), 0.0, scene["motion_score"])
        self.assertAlmostEqual(at_centre[1].item(), 1.0, places=6)
        # one radius away is exp(-1/2)
        one_sigma = temporal_weights(
            scene["time_center"], torch.ones(3), 1.0, scene["motion_score"])
        self.assertAlmostEqual(one_sigma[1].item(), float(torch.exp(
            torch.tensor(-0.5))), places=6)

    def test_between_observations_something_is_visible(self):
        """The capability the bank lacks, without any object machinery."""
        scene = _scene()
        bank = time_slice_opacities(
            scene["opacities"], scene["motion_score"].reshape(-1, 1),
            scene["time_center"].reshape(-1, 1), 1.0)
        self.assertEqual(bank[1].item(), 0.0)
        self.assertEqual(bank[2].item(), 0.0)

        weights = temporal_weights(
            scene["time_center"], torch.ones(3), 1.0, scene["motion_score"])
        self.assertEqual(visible_rows(weights), 3)


class InterpolationRadiusTest(unittest.TestCase):
    """Training and interpolation need opposite temporal radii (section 3.50).

    Narrow keeps the neighbouring bank out of an observed frame, so the frame's
    own rows are not a redundant explanation the loss can dim away. Wide is what
    lets a midpoint draw on both banks at all. One number cannot do both, which
    is what sections 3.49/3.49.1 spent two failed fixes establishing.
    """

    def _banks(self, radius):
        time_center = torch.cat([torch.full((200,), t) for t in (0.0, 1.0, 2.0)])
        return (time_center,
                torch.full_like(time_center, radius),
                torch.ones_like(time_center))

    def test_none_keeps_the_stored_radius(self):
        _, scale, _ = self._banks(0.25)
        torch.testing.assert_close(widen_for_interpolation(scale, None), scale)

    def test_widening_lifts_a_midpoint_out_of_the_tail(self):
        tc, scale, ms = self._banks(0.25)
        narrow = temporal_weights(tc, scale, 0.5, ms)
        wide = temporal_weights(tc, widen_for_interpolation(scale, 0.5), 0.5, ms)
        # a midpoint is half a step from both banks
        self.assertAlmostEqual(float(narrow[tc == 0.0].max()), 0.1353, places=3)
        self.assertAlmostEqual(float(wide[tc == 0.0].max()), 0.6065, places=3)

    def test_widening_does_not_touch_an_observed_time(self):
        """The narrow radius is the whole point at observed times."""
        tc, scale, ms = self._banks(0.25)
        own_narrow = float(temporal_weights(tc, scale, 1.0, ms)[tc == 1.0].max())
        self.assertAlmostEqual(own_narrow, 1.0, places=5)

    def test_nonpositive_widening_is_rejected(self):
        _, scale, _ = self._banks(0.25)
        for bad in (0.0, -0.5):
            with self.assertRaises(ValueError):
                widen_for_interpolation(scale, bad)

    def test_shape_and_dtype_survive(self):
        _, scale, _ = self._banks(0.25)
        widened = widen_for_interpolation(scale, 0.5)
        self.assertEqual(widened.shape, scale.reshape(-1).shape)
        self.assertEqual(widened.dtype, scale.dtype)


class PerObservationWeightsTest(unittest.TestCase):
    """The temporal conditional should weight observations, not primitives.

    A bank is one instant's worth of seeded surface; its row count comes from
    seeding and pruning, not from the object. Section 3.49 measured banks from
    211 to 2241 rows in one finished map, so weighting rows independently let a
    thin bank be outvoted by its own neighbours' carry-over.
    """

    def _banks(self, sizes, times):
        """One bank per (size, time), all rows dynamic."""
        time_center = torch.cat([torch.full((n,), t)
                                 for n, t in zip(sizes, times)])
        return (time_center,
                torch.full_like(time_center, 0.5),
                torch.ones_like(time_center))

    def _bank_mass(self, weights, time_center, time):
        return float(weights[time_center == time].sum())

    def test_a_banks_share_does_not_depend_on_its_row_count(self):
        times = [0.0, 1.0, 2.0]
        shares = []
        for middle in (50, 500, 5000):
            tc, ts, ms = self._banks([1000, middle, 1000], times)
            w = temporal_weights(tc, ts, 1.0, ms, per_observation=True)
            total = float(w.sum())
            shares.append(self._bank_mass(w, tc, 1.0) / total)
        for share in shares[1:]:
            self.assertAlmostEqual(share, shares[0], places=5)

    def test_a_thin_bank_is_not_outvoted_by_its_neighbours(self):
        """The section 3.49 failure, reproduced and then removed."""
        times = [0.0, 1.0, 2.0]
        tc, ts, ms = self._banks([1050, 211, 1050], times)

        plain = temporal_weights(tc, ts, 1.0, ms)
        own = self._bank_mass(plain, tc, 1.0)
        ghost = float(plain.sum()) - own
        self.assertGreater(ghost / own, 1.0)      # neighbours win: the bug

        fixed = temporal_weights(tc, ts, 1.0, ms, per_observation=True)
        own = self._bank_mass(fixed, tc, 1.0)
        ghost = float(fixed.sum()) - own
        self.assertLess(ghost / own, 0.5)

    def test_total_dynamic_mass_is_the_same_at_every_query_time(self):
        """Otherwise the object's brightness would flicker with the clock."""
        tc, ts, ms = self._banks([1050, 211, 1050], [0.0, 1.0, 2.0])
        masses = [float(temporal_weights(tc, ts, t, ms,
                                         per_observation=True).sum())
                  for t in (0.0, 0.5, 1.0, 1.5, 2.0)]
        for mass in masses[1:]:
            self.assertAlmostEqual(mass, masses[0], places=3)

    def test_a_midpoint_splits_evenly_between_its_two_banks(self):
        tc, ts, ms = self._banks([1050, 211, 1050], [0.0, 1.0, 2.0])
        w = temporal_weights(tc, ts, 0.5, ms, per_observation=True)
        left = self._bank_mass(w, tc, 0.0)
        right = self._bank_mass(w, tc, 1.0)
        # relative, not absolute: these are sums over ~1000 float32 rows
        self.assertLess(abs(left - right) / max(left, right), 1e-5)

    def test_static_rows_are_untouched(self):
        time_center = torch.tensor([-1.0, 0.0, 0.0, 1.0])
        motion = torch.tensor([0.0, 1.0, 1.0, 1.0])
        w = temporal_weights(time_center, torch.full((4,), 0.5), 0.0, motion,
                             per_observation=True)
        self.assertEqual(float(w[0]), 1.0)

    def test_equal_sized_banks_keep_the_plain_ordering(self):
        """Normalizing must not reorder banks that were already comparable."""
        tc, ts, ms = self._banks([500, 500, 500], [0.0, 1.0, 2.0])
        plain = temporal_weights(tc, ts, 1.0, ms)
        fixed = temporal_weights(tc, ts, 1.0, ms, per_observation=True)
        for t in (0.0, 1.0, 2.0):
            order_plain = self._bank_mass(plain, tc, t)
            order_fixed = self._bank_mass(fixed, tc, t)
            self.assertEqual(order_plain > self._bank_mass(plain, tc, 0.0),
                             order_fixed > self._bank_mass(fixed, tc, 0.0))

    def test_slice_at_time_threads_the_option(self):
        tc, ts, ms = self._banks([1050, 211, 1050], [0.0, 1.0, 2.0])
        xyz = torch.zeros((tc.shape[0], 3))
        opacity = torch.full((tc.shape[0], 1), 0.8)
        velocity = torch.zeros((tc.shape[0], 3))
        _, _, plain = slice_at_time(xyz, opacity, tc, ts, velocity, 1.0, ms)
        _, _, fixed = slice_at_time(xyz, opacity, tc, ts, velocity, 1.0, ms,
                                    per_observation=True)
        self.assertFalse(torch.allclose(plain, fixed))


class SliceAtTimeTest(unittest.TestCase):
    def test_velocity_moves_the_row_off_its_own_moment(self):
        scene = _scene()
        velocity = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                                 [1.0, 0.0, 0.0]])
        moved, _, _ = slice_at_time(
            scene["xyz"], scene["opacities"], scene["time_center"],
            torch.ones(3), velocity, 1.0, scene["motion_score"])

        # the t=0 row has drifted one unit forward, the t=2 row one unit back
        torch.testing.assert_close(
            moved[1], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(
            moved[2], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6, rtol=0.0)

    def test_a_row_does_not_move_at_its_own_moment(self):
        scene = _scene()
        velocity = torch.full((3, 3), 3.0)
        moved, _, _ = slice_at_time(
            scene["xyz"], scene["opacities"], scene["time_center"],
            torch.ones(3), velocity, 0.0, scene["motion_score"])
        torch.testing.assert_close(
            moved[1], scene["xyz"][1], atol=1e-6, rtol=0.0)

    def test_static_rows_never_drift(self):
        scene = _scene()
        velocity = torch.full((3, 3), 5.0)
        moved, opacities, _ = slice_at_time(
            scene["xyz"], scene["opacities"], scene["time_center"],
            torch.ones(3), velocity, 7.0, scene["motion_score"])
        torch.testing.assert_close(
            moved[0], scene["xyz"][0], atol=0.0, rtol=0.0)
        self.assertEqual(opacities[0].item(), scene["opacities"][0].item())

    def test_gradients_reach_velocity_and_time_scale(self):
        """The signal the SE(3) route structurally could not produce."""
        scene = _scene()
        velocity = torch.zeros((3, 3), requires_grad=True)
        time_scale = torch.ones(3, requires_grad=True)

        moved, faded, _ = slice_at_time(
            scene["xyz"], scene["opacities"], scene["time_center"],
            time_scale, velocity, 1.0, scene["motion_score"])
        (moved.sum() + faded.sum()).backward()

        self.assertTrue(torch.isfinite(velocity.grad).all())
        self.assertTrue(torch.isfinite(time_scale.grad).all())
        # a row rendered away from its own moment must produce a real gradient
        self.assertGreater(velocity.grad[1].abs().sum().item(), 0.0)
        self.assertGreater(time_scale.grad[1].abs().item(), 0.0)

    def test_negative_time_scale_is_rejected_not_clamped(self):
        """Clamping would leave the row a delta with zero gradient, forever.

        A silently un-recoverable parameter is worse than a crash, so a trainable
        time scale has to be parameterized non-negative instead.
        """
        scene = _scene()
        with self.assertRaises(ValueError):
            temporal_weights(
                scene["time_center"], torch.full((3,), -0.5), 1.0,
                scene["motion_score"])

    def test_motion_score_is_required(self):
        """Omitting it used to fade the whole static map out -- a black frame."""
        scene = _scene()
        with self.assertRaises(TypeError):
            temporal_weights(scene["time_center"], torch.ones(3), 1.0)

    def test_excluding_own_bank_hides_exactly_that_bank(self):
        """Leave-one-out: the frame's own dynamic rows must go, others stay."""
        scene = _scene()
        _, _, weights = slice_at_time(
            scene["xyz"], scene["opacities"], scene["time_center"],
            torch.ones(3), scene["velocity"], 0.0, scene["motion_score"],
            exclude_own_bank=True)

        self.assertEqual(weights[1].item(), 0.0)      # the t=0 row: its own time
        self.assertGreater(weights[2].item(), 0.0)    # the t=2 row still carries
        self.assertEqual(weights[0].item(), 1.0)      # static rows unaffected

    def test_excluding_own_bank_makes_velocity_the_only_way_to_place_the_object(self):
        """The point of the mechanism: with the own bank hidden, the rendered
        position depends on velocity, so a photometric loss can push it."""
        scene = _scene()
        velocity = torch.zeros((3, 3), requires_grad=True)

        moved, faded, _ = slice_at_time(
            scene["xyz"], scene["opacities"], scene["time_center"],
            torch.ones(3), velocity, 0.0, scene["motion_score"],
            exclude_own_bank=True)
        # only rows that survived the mask can contribute to a loss
        (moved * faded).sum().backward()

        self.assertTrue(torch.isfinite(velocity.grad).all())
        # the carried row (t=2) gets gradient; the hidden own-bank row does not
        self.assertGreater(velocity.grad[2].abs().sum().item(), 0.0)
        self.assertEqual(velocity.grad[1].abs().sum().item(), 0.0)

    def test_pooling_shares_one_velocity_per_bank(self):
        """The whole point: 3 degrees of freedom per instant, not 3 per Gaussian."""
        velocity = torch.tensor([[9.0, 9.0, 9.0],    # static, must be untouched
                                 [1.0, 0.0, 0.0],    # t=0 bank
                                 [3.0, 0.0, 0.0]])   # t=2 bank
        time_center = torch.tensor([-1.0, 0.0, 2.0])
        motion = torch.tensor([0.0, 1.0, 1.0])

        pooled = pool_velocity(velocity, time_center, motion)

        torch.testing.assert_close(pooled[0], velocity[0])   # static untouched
        torch.testing.assert_close(pooled[1], velocity[1])   # alone in its bank
        torch.testing.assert_close(pooled[2], velocity[2])

        # two rows in one bank average to a single shared value
        velocity = torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        time_center = torch.tensor([0.0, 0.0])
        motion = torch.tensor([1.0, 1.0])
        pooled = pool_velocity(velocity, time_center, motion)
        expected = torch.tensor([2.0, 0.0, 0.0])
        torch.testing.assert_close(pooled[0], expected)
        torch.testing.assert_close(pooled[1], expected)

    def test_pooling_sums_the_gradient_over_the_bank(self):
        """Why pooling should also help statistically: noise cancels."""
        velocity = torch.zeros((3, 3), requires_grad=True)
        time_center = torch.tensor([0.0, 0.0, 0.0])
        motion = torch.ones(3)

        pooled = pool_velocity(velocity, time_center, motion)
        # a loss that pulls the shared value one way from all three rows
        (pooled[:, 0].sum()).backward()

        # each row receives the same share of the shared value's gradient
        self.assertTrue(torch.allclose(velocity.grad[:, 0],
                                       torch.full((3,), 1.0)))

    def test_row_count_mismatch_is_rejected(self):
        scene = _scene()
        with self.assertRaises(ValueError):
            slice_at_time(
                scene["xyz"], scene["opacities"], scene["time_center"],
                torch.ones(2), scene["velocity"], 1.0, scene["motion_score"])


if __name__ == "__main__":
    unittest.main()
