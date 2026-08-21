import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))

from gaussian.deform.oracle_motion_gate import (
    apply_motion_gate,
    backward_with_auxiliary_params,
    color_motion_scores,
    oracle_color_mask,
    oracle_roi_l1,
    time_slice_opacities,
    zero_masked_rows_,
)


class OracleMotionGateTest(unittest.TestCase):
    def test_color_scores_accept_255_palette(self):
        colors = np.array([
            [60, 220, 240],
            [0, 0, 0],
            [40, 40, 220],
        ], dtype=np.float32) / 255.0
        palette = [[60, 220, 240], [40, 40, 220]]

        scores = color_motion_scores(colors, palette, threshold=0.01)

        np.testing.assert_array_equal(scores[:, 0], [1.0, 0.0, 1.0])

    def test_motion_gate_forces_static_identity(self):
        dxyz = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        drot = torch.tensor([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]])
        dscale = torch.ones((2, 3))
        score = torch.tensor([[1.0], [0.0]])

        gated_xyz, gated_rot, gated_scale = apply_motion_gate(
            dxyz, drot, dscale, score)

        torch.testing.assert_close(gated_xyz[0], dxyz[0])
        torch.testing.assert_close(gated_xyz[1], torch.zeros(3))
        torch.testing.assert_close(gated_scale[1], torch.zeros(3))
        torch.testing.assert_close(gated_rot[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))

    def test_oracle_roi_loss_uses_only_palette_pixels(self):
        target = torch.zeros((3, 1, 2))
        target[:, 0, 0] = torch.tensor([240.0, 220.0, 60.0]) / 255.0
        prediction = target.clone()
        prediction[:, 0, 0] = 0.0
        prediction[:, 0, 1] = 1.0
        palette = [[240, 220, 60]]

        mask = oracle_color_mask(target, palette, threshold=0.01)
        loss = oracle_roi_l1(prediction, target, palette, threshold=0.01)

        torch.testing.assert_close(mask, torch.tensor([[True, False]]))
        torch.testing.assert_close(loss, target[:, 0, 0].mean())

    def test_translation_only_disables_rotation_and_scale(self):
        dxyz = torch.ones((1, 3))
        drot = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        dscale = torch.ones((1, 3))
        score = torch.ones((1, 1))

        gated_xyz, gated_rot, gated_scale = apply_motion_gate(
            dxyz, drot, dscale, score, translation_only=True)

        torch.testing.assert_close(gated_xyz, dxyz)
        torch.testing.assert_close(gated_rot, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
        torch.testing.assert_close(gated_scale, torch.zeros_like(dscale))

    def test_time_slice_keeps_static_and_only_matching_dynamic_rows(self):
        opacity = torch.tensor([[0.2], [0.4], [0.6], [0.8]])
        motion_score = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
        source_time = torch.tensor([[-1.0], [3.0], [4.0], [-1.0]])

        sliced = time_slice_opacities(
            opacity, motion_score, source_time, target_time=3.0)

        torch.testing.assert_close(
            sliced, torch.tensor([[0.2], [0.4], [0.0], [0.8]]))

    def test_time_slice_accepts_small_timestamp_roundoff(self):
        sliced = time_slice_opacities(
            torch.ones((1, 1)),
            torch.ones((1, 1)),
            torch.tensor([[2.00005]]),
            target_time=2.0,
            tolerance=1e-4,
        )

        torch.testing.assert_close(sliced, torch.ones((1, 1)))

    def test_zero_masked_rows_preserves_static_gradients(self):
        gradient = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        zero_masked_rows_(gradient, torch.tensor([[1.0], [0.0]]))
        torch.testing.assert_close(
            gradient,
            torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 5.0]]),
        )

    def test_auxiliary_gradient_isolated_to_selected_parameter(self):
        canonical = torch.tensor(2.0, requires_grad=True)
        deform = torch.tensor(3.0, requires_grad=True)
        image = canonical + deform
        base_loss = image.square()
        auxiliary_loss = 3.0 * image

        backward_with_auxiliary_params(base_loss, auxiliary_loss, [deform])

        torch.testing.assert_close(canonical.grad, torch.tensor(10.0))
        torch.testing.assert_close(deform.grad, torch.tensor(13.0))


if __name__ == "__main__":
    unittest.main()
