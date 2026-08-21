import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mcgs_slam"))

from physical_view_window import PhysicalViewWindow


class PhysicalViewWindowTest(unittest.TestCase):
    def test_capacity_counts_physical_times_not_virtual_views(self):
        window = PhysicalViewWindow(capacity=3)
        for physical_time in range(5):
            for cam_idx in range(4):
                window.register(physical_time, cam_idx, physical_time + 500 * cam_idx)

        self.assertEqual(window.current_times, [4, 3, 2])
        self.assertEqual(len(window.groups[4]), 4)

    def test_window_selection_rotates_all_faces(self):
        window = PhysicalViewWindow(capacity=2)
        for cam_idx in range(4):
            window.register(7, cam_idx, 7 + 500 * cam_idx)

        selected = {window.window_view_keys(step)[0] for step in range(4)}
        self.assertEqual(selected, {7, 507, 1007, 1507})

    def test_late_auxiliary_view_does_not_reinsert_evicted_time(self):
        window = PhysicalViewWindow(capacity=2)
        window.register(0, 0, 0)
        window.register(1, 0, 1)
        window.register(2, 0, 2)
        window.register(0, 1, 500)

        self.assertEqual(window.current_times, [2, 1])
        self.assertEqual(window.groups[0][1], 500)

    def test_replay_excludes_active_times_and_balances_cameras(self):
        window = PhysicalViewWindow(capacity=1)
        for physical_time in range(3):
            for cam_idx in range(4):
                window.register(physical_time, cam_idx, physical_time + 500 * cam_idx)

        replay = window.replay_view_keys(limit=2, step=0)
        self.assertEqual(len(replay), 2)
        self.assertTrue(all(view_key % 500 != 2 for view_key in replay))
        self.assertNotEqual(replay[0] // 500, replay[1] // 500)

    def test_single_camera_matches_one_view_per_physical_time(self):
        window = PhysicalViewWindow(capacity=3)
        for physical_time in range(4):
            window.register(physical_time, 0, physical_time)

        self.assertEqual(window.window_view_keys(step=99), [3, 2, 1])


if __name__ == "__main__":
    unittest.main()
