"""Temporal helpers shared by tracking packets and dynamic Gaussian models.

The tracking stack needs an integer sample index for array addressing while a
dynamic scene model needs the source capture time.  Keeping those values
separate avoids accidentally treating a frame index as physical time.
"""


class SourceTimeNormalizer:
    """Translate source timestamps to a small, stable relative time domain.

    ``scale`` declares how many source timestamp units make one model-time
    unit.  Streams in this repository already apply their calibration
    ``timescale`` before yielding timestamps, so the default scale is one.
    """

    def __init__(self, scale=1.0, unit="stream_timestamp"):
        scale = float(scale)
        if scale <= 0:
            raise ValueError("source time scale must be positive")
        self.scale = scale
        self.unit = str(unit)
        self.origin = None

    def normalize(self, timestamp):
        timestamp = float(timestamp)
        if self.origin is None:
            self.origin = timestamp
        return (timestamp - self.origin) / self.scale

    def normalize_many(self, timestamps):
        return [self.normalize(timestamp) for timestamp in timestamps]

    def state_dict(self):
        return {
            "origin": self.origin,
            "scale": self.scale,
            "unit": self.unit,
            "value_semantics": "(source_timestamp - origin) / scale",
        }

    def load_state_dict(self, state):
        scale = float(state["scale"])
        if scale <= 0:
            raise ValueError("source time scale must be positive")
        origin = state.get("origin")
        self.origin = None if origin is None else float(origin)
        self.scale = scale
        self.unit = str(state.get("unit", "stream_timestamp"))


def source_timestamps_for_indices(kf_stamps, indices):
    """Return capture timestamps for local keyframe indices in input order."""
    timestamps = []
    for index in indices:
        index = int(index)
        if index not in kf_stamps:
            raise KeyError(f"missing source timestamp for keyframe index {index}")
        timestamps.append(float(kf_stamps[index]))
    return timestamps
