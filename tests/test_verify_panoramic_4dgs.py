"""The acceptance check for "panoramic 4DGS", and that it can still fail.

A verifier nobody has seen reject anything is decoration. These tests drive its
two judgement calls -- sphere coverage and per-frame spread -- from both sides,
because both thresholds exist to catch failures this project actually shipped:
the four-face calib leaving 53% of every sphere black, and a mean of 25.9 dB
sitting on top of a frame that had collapsed to 16.3 (sections 3.47/3.49).
"""

import re
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools"))

from verify_panoramic_4dgs import (  # noqa: E402
    COVERAGE_LIMIT,
    SPREAD_LIMIT_DB,
    Report,
    stamps_of,
)


class ReportTest(unittest.TestCase):
    def test_exit_code_is_zero_only_when_nothing_failed(self):
        report = Report()
        report.add("a", True, "")
        report.add("b", None, "skipped")
        self.assertEqual(report.show(), 0)

    def test_a_single_failure_fails_the_run(self):
        report = Report()
        report.add("a", True, "")
        report.add("b", False, "")
        self.assertEqual(report.show(), 1)

    def test_skipped_claims_are_not_failures(self):
        report = Report()
        report.add("optional", None, "not requested")
        self.assertEqual(report.show(), 0)


class ThresholdTest(unittest.TestCase):
    def test_four_horizontal_faces_do_not_count_as_a_sphere(self):
        """46.5% is what the four-face calib covers; it must not pass."""
        self.assertLess(0.465, COVERAGE_LIMIT)

    def test_six_faces_do(self):
        self.assertGreaterEqual(0.999, COVERAGE_LIMIT)

    def test_the_spread_limit_rejects_the_failure_it_was_written_for(self):
        """Section 3.49: frames ran 16.33 to 27.85 while the mean read 25.9."""
        per_frame = [27.69, 16.33, 26.69, 27.68, 24.77, 27.85, 26.70, 26.97]
        self.assertGreater(max(per_frame) - min(per_frame), SPREAD_LIMIT_DB)

    def test_the_spread_limit_accepts_the_fixed_run(self):
        """Section 3.50, same sequence after splitting the temporal radius."""
        per_frame = [27.27, 26.88, 26.59, 26.81, 27.57, 26.92, 27.54, 27.54]
        self.assertLessEqual(max(per_frame) - min(per_frame), SPREAD_LIMIT_DB)


class StampParsingTest(unittest.TestCase):
    def _touch(self, directory, names):
        for name in names:
            (directory / name).write_bytes(b"")

    def test_reads_both_obs_and_interp_names(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._touch(directory, ["erp_0000.000_obs.jpg",
                                    "erp_0000.500_interp.jpg",
                                    "erp_0012.000_obs.jpg"])
            stamps = stamps_of(directory)
            self.assertEqual(sorted(stamps), [0.0, 0.5, 12.0])

    def test_ignores_unrelated_files(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self._touch(directory, ["erp_0000.000_obs.jpg", "notes.txt",
                                    "depth_0000.000.png"])
            self.assertEqual(list(stamps_of(directory)), [0.0])

    def test_survives_a_directory_with_nothing_in_it(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(stamps_of(Path(tmp)), {})


class PerFrameParsingTest(unittest.TestCase):
    """The regex that reads eval_erp_panorama's table.

    Kept honest here because a silently-empty parse would turn the spread check
    into an unconditional pass -- the exact failure mode the check exists to
    prevent.
    """

    PATTERN = r"^\s+\d+\.\d{3}\s+\d+\.\d\s+(\d+\.\d+)"

    def test_reads_the_covered_psnr_column(self):
        table = (
            "      stamp  covered%  cov PSNR  cov SSIM  full PSNR  full SSIM\n"
            "      0.000      99.9     27.69     0.918      27.49      0.918\n"
            "      1.000      99.9     16.33     0.855      16.32      0.854\n"
        )
        values = [float(m.group(1)) for m in re.finditer(self.PATTERN, table, re.M)]
        self.assertEqual(values, [27.69, 16.33])

    def test_does_not_match_the_summary_line(self):
        summary = "   mean over 8 frames: covered 25.59 dB / 0.907 SSIM\n"
        self.assertEqual(list(re.finditer(self.PATTERN, summary, re.M)), [])


if __name__ == "__main__":
    unittest.main()
