"""Physical-time grouping and balanced view scheduling for multi-camera mapping.

The GS backend uses a unique numeric key for every virtual camera view.  For a
cubemap frame that means one physical instant produces four keys, so treating
those keys as independent entries in the mapping window shortens the temporal
span by four.  This module keeps the unique view keys for Gaussian ownership,
but schedules mapping in units of physical time.
"""

from collections import defaultdict


class PhysicalViewWindow:
    """Group virtual views by physical timestamp and schedule them fairly.

    ``current_times`` is newest-first and contains at most ``capacity`` physical
    instants.  ``groups`` retains all registered instants so older views remain
    available to the replay sampler after leaving the active window.
    """

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.current_times = []
        self.groups = {}

    def clear(self):
        self.current_times.clear()
        self.groups.clear()

    def register(self, physical_tstamp, cam_idx, view_key):
        """Register a view and return whether its physical instant is new.

        Views from an already-known instant never reinsert that instant into the
        active window.  This matters when camera blocks arrive sequentially: an
        auxiliary face belonging to an old/evicted instant must not displace a
        genuinely newer physical frame.
        """
        is_new_time = physical_tstamp not in self.groups
        if is_new_time:
            self.groups[physical_tstamp] = {}
            self.current_times.insert(0, physical_tstamp)
            del self.current_times[self.capacity:]
        self.groups[physical_tstamp][int(cam_idx)] = view_key
        return is_new_time

    def window_view_keys(self, step):
        """Select one view per active physical instant using face round-robin."""
        selected = []
        for time_offset, physical_tstamp in enumerate(self.current_times):
            group = self.groups.get(physical_tstamp, {})
            cam_indices = sorted(group)
            if not cam_indices:
                continue
            cam_idx = cam_indices[(int(step) + time_offset) % len(cam_indices)]
            selected.append(group[cam_idx])
        return selected

    def replay_view_keys(self, limit, step):
        """Select old views with round-robin camera balance.

        Camera buckets are traversed cyclically before taking a second item from
        the same camera.  Selection within each bucket is also rotated by
        ``step`` so repeated mapping iterations do not replay one fixed frame.
        """
        limit = max(0, int(limit))
        if limit == 0:
            return []

        active_times = set(self.current_times)
        by_camera = defaultdict(list)
        for physical_tstamp, group in self.groups.items():
            if physical_tstamp in active_times:
                continue
            for cam_idx, view_key in group.items():
                by_camera[cam_idx].append(view_key)

        cam_indices = sorted(by_camera)
        if not cam_indices:
            return []

        # Rotate both the camera order and each camera's temporal candidates.
        cam_start = int(step) % len(cam_indices)
        cam_indices = cam_indices[cam_start:] + cam_indices[:cam_start]
        buckets = {}
        for cam_idx in cam_indices:
            values = by_camera[cam_idx]
            value_start = int(step) % len(values)
            buckets[cam_idx] = values[value_start:] + values[:value_start]

        selected = []
        depth = 0
        while len(selected) < limit:
            added = False
            for cam_idx in cam_indices:
                values = buckets[cam_idx]
                if depth < len(values):
                    selected.append(values[depth])
                    added = True
                    if len(selected) == limit:
                        break
            if not added:
                break
            depth += 1
        return selected

