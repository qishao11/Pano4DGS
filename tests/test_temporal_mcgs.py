import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))

from mcgs import Mcgs
from temporal import SourceTimeNormalizer


class McgsSourceTimePacketTest(unittest.TestCase):
    def test_gs_time_uses_capture_timestamp_not_frame_index(self):
        mcgs = Mcgs.__new__(Mcgs)
        mcgs.video = SimpleNamespace(kf_stamps={0: 1000.0, 1: 1000.1, 2: 1000.4})
        mcgs.source_time = SourceTimeNormalizer()

        model_time = mcgs._source_time_tensor(torch.tensor([0, 1, 2]))

        torch.testing.assert_close(
            model_time,
            torch.tensor([0.0, 0.1, 0.4], dtype=torch.float64),
            rtol=0.0,
            atol=1e-12,
        )
        self.assertEqual(mcgs.source_time.state_dict()["origin"], 1000.0)


if __name__ == "__main__":
    unittest.main()
