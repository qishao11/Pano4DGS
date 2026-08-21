import unittest

import numpy as np

from tools.eval_synthetic_dynamic_roi import RegionAccumulator, dynamic_color_mask


class SyntheticDynamicRoiTest(unittest.TestCase):
    def test_palette_mask_rejects_static_colors(self):
        image = np.asarray([[[60, 220, 240], [0, 0, 0]]], dtype=np.uint8)
        np.testing.assert_array_equal(
            dynamic_color_mask(image, threshold=0.02),
            np.asarray([[True, False]]),
        )

    def test_accumulator_uses_only_masked_pixels(self):
        target = np.zeros((1, 2, 3), dtype=np.uint8)
        prediction = target.copy()
        prediction[0, 0] = 10
        prediction[0, 1] = 200
        accumulator = RegionAccumulator()
        accumulator.update(prediction, target, np.asarray([[True, False]]))
        result = accumulator.result()
        self.assertAlmostEqual(result["mae"], 10.0)
        self.assertAlmostEqual(result["psnr"], 20.0 * np.log10(255.0 / 10.0))
        self.assertEqual(result["pixel_count"], 1)


if __name__ == "__main__":
    unittest.main()
