import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))

from temporal import SourceTimeNormalizer, source_timestamps_for_indices


class SourceTimeNormalizerTest(unittest.TestCase):
    def test_irregular_source_times_remain_irregular(self):
        normalizer = SourceTimeNormalizer()
        values = normalizer.normalize_many([1000.0, 1000.1, 1000.4])

        self.assertAlmostEqual(values[0], 0.0)
        self.assertAlmostEqual(values[1], 0.1)
        self.assertAlmostEqual(values[2], 0.4)

    def test_state_round_trip_preserves_origin_and_scale(self):
        normalizer = SourceTimeNormalizer(scale=0.5, unit="seconds")
        self.assertEqual(normalizer.normalize(10.0), 0.0)
        self.assertEqual(normalizer.normalize(10.5), 1.0)

        restored = SourceTimeNormalizer()
        restored.load_state_dict(normalizer.state_dict())
        self.assertEqual(restored.normalize(11.0), 2.0)
        self.assertEqual(restored.state_dict(), normalizer.state_dict())

    def test_select_source_timestamps_preserves_index_order(self):
        stamps = {0: 1000.0, 1: 1000.1, 2: 1000.4}
        self.assertEqual(
            source_timestamps_for_indices(stamps, [2, 0, 1]),
            [1000.4, 1000.0, 1000.1],
        )

    def test_missing_keyframe_timestamp_fails_loudly(self):
        with self.assertRaisesRegex(KeyError, "keyframe index 3"):
            source_timestamps_for_indices({0: 1.0}, [3])


if __name__ == "__main__":
    unittest.main()
