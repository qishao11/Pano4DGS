"""Holding the moving object where seeding put it.

Section 3.53 measured the remaining gap precisely: seeding lands 100.0% of the
object's Gaussians on the sphere, the finished map has 44.8%, and the fraction
still inside the object's silhouette barely moves (86.3% -> 85.0%). The rows stay
on the view ray and slide along it during refinement, because one view cannot
constrain depth along its own ray -- every position on it reprojects to the same
pixel.

What must hold: dynamic positions stop moving, and nothing else does.
"""

import sys
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))

from gaussian.deform.oracle_motion_gate import zero_masked_rows_  # noqa: E402


class _Gaussians:
    def __init__(self, motion, freeze=True, rows=6):
        self._xyz = torch.zeros((rows, 3), requires_grad=True)
        self._scaling = torch.zeros((rows, 3), requires_grad=True)
        self._features = torch.zeros((rows, 3), requires_grad=True)
        self.dynamic_score = torch.tensor(motion, dtype=torch.float32)
        self.freeze_dynamic_positions = freeze
        self.oracle_freeze_dynamic_scaling = False


class _Backend:
    """The two gradient hooks, lifted off GSBackEnd without its constructor."""

    def __init__(self, gaussians):
        self.gaussians = gaussians

    def _freeze_dynamic_positions(self):
        from gs_backend import GSBackEnd
        return GSBackEnd._freeze_dynamic_positions(self)

    def _mask_oracle_shape_grads(self):
        from gs_backend import GSBackEnd
        return GSBackEnd._mask_oracle_shape_grads(self)


def _with_gradients(motion, **kwargs):
    gaussians = _Gaussians(motion, **kwargs)
    for tensor in (gaussians._xyz, gaussians._scaling, gaussians._features):
        tensor.grad = torch.ones_like(tensor)
    return gaussians, _Backend(gaussians)


class FreezeTest(unittest.TestCase):
    MOTION = [0.0, 1.0, 0.0, 1.0, 1.0, 0.0]

    def test_dynamic_rows_stop_receiving_position_gradient(self):
        gaussians, backend = _with_gradients(self.MOTION)
        backend._freeze_dynamic_positions()
        dynamic = torch.tensor(self.MOTION) > 0.5
        self.assertTrue(bool((gaussians._xyz.grad[dynamic] == 0).all()))

    def test_static_rows_keep_theirs(self):
        """The static map has real multi-view support; freezing it would be a loss."""
        gaussians, backend = _with_gradients(self.MOTION)
        backend._freeze_dynamic_positions()
        static = torch.tensor(self.MOTION) <= 0.5
        self.assertTrue(bool((gaussians._xyz.grad[static] == 1).all()))

    def test_only_position_is_frozen(self):
        """Colour, opacity and shape still have to train on the object."""
        gaussians, backend = _with_gradients(self.MOTION)
        backend._freeze_dynamic_positions()
        self.assertTrue(bool((gaussians._scaling.grad == 1).all()))
        self.assertTrue(bool((gaussians._features.grad == 1).all()))

    def test_disabled_by_default_changes_nothing(self):
        gaussians, backend = _with_gradients(self.MOTION, freeze=False)
        backend._freeze_dynamic_positions()
        self.assertTrue(bool((gaussians._xyz.grad == 1).all()))

    def test_a_map_with_no_dynamic_rows_is_untouched(self):
        gaussians, backend = _with_gradients([0.0] * 6)
        backend._freeze_dynamic_positions()
        self.assertTrue(bool((gaussians._xyz.grad == 1).all()))

    def test_a_missing_gradient_is_not_an_error(self):
        """The first step of a loop can reach the hook before any backward pass."""
        gaussians, backend = _with_gradients(self.MOTION)
        gaussians._xyz.grad = None
        backend._freeze_dynamic_positions()  # must not raise

    def test_it_composes_with_the_scale_freeze(self):
        gaussians, backend = _with_gradients(self.MOTION)
        gaussians.oracle_freeze_dynamic_scaling = True
        backend._mask_oracle_shape_grads()
        backend._freeze_dynamic_positions()
        dynamic = torch.tensor(self.MOTION) > 0.5
        self.assertTrue(bool((gaussians._xyz.grad[dynamic] == 0).all()))
        self.assertTrue(bool((gaussians._scaling.grad[dynamic] == 0).all()))
        self.assertTrue(bool((gaussians._scaling.grad[~dynamic] == 1).all()))


class MaskShapeTest(unittest.TestCase):
    def test_a_mismatched_mask_is_rejected_not_broadcast(self):
        """Silently broadcasting would freeze the wrong rows."""
        gradient = torch.ones((6, 3))
        with self.assertRaises(ValueError):
            zero_masked_rows_(gradient, torch.ones(4))


if __name__ == "__main__":
    unittest.main()
