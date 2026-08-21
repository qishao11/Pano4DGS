import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))

from gaussian.deform.dynamic_modes import resolve_dynamic_mode


class DynamicModeTest(unittest.TestCase):
    def test_legacy_configs_are_inferred(self):
        self.assertEqual(resolve_dynamic_mode({"deform": False}), "none")
        self.assertEqual(resolve_dynamic_mode({"deform": True}), "deform")
        self.assertEqual(resolve_dynamic_mode({
            "deform": False,
            "deform_cfg": {"oracle_time_sliced_dynamic": True},
        }), "oracle_time_slice")

    def test_ambiguous_legacy_combination_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "both enabled"):
            resolve_dynamic_mode({
                "deform": True,
                "deform_cfg": {"oracle_time_sliced_dynamic": True},
            })

    def test_explicit_object_mode_is_accepted(self):
        self.assertEqual(resolve_dynamic_mode({
            "dynamic_mode": "object_se3",
            "deform": False,
        }), "object_se3")

    def test_explicit_mode_conflict_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            resolve_dynamic_mode({
                "dynamic_mode": "none",
                "deform": True,
            })


if __name__ == "__main__":
    unittest.main()
