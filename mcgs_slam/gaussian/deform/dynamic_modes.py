"""Resolve one explicit dynamic-scene mode from legacy configuration flags."""


DYNAMIC_MODE_NONE = "none"
DYNAMIC_MODE_DEFORM = "deform"
DYNAMIC_MODE_ORACLE_TIME_SLICE = "oracle_time_slice"
DYNAMIC_MODE_OBJECT_SE3 = "object_se3"
DYNAMIC_MODE_GAUSSIAN_4D = "gaussian_4d"

VALID_DYNAMIC_MODES = {
    DYNAMIC_MODE_NONE,
    DYNAMIC_MODE_DEFORM,
    DYNAMIC_MODE_ORACLE_TIME_SLICE,
    DYNAMIC_MODE_OBJECT_SE3,
    DYNAMIC_MODE_GAUSSIAN_4D,
}


def resolve_dynamic_mode(training_config):
    """Return a single mode and reject ambiguous mode combinations.

    Existing configs predate ``dynamic_mode``.  They remain compatible by
    inferring the mode from ``deform`` and ``oracle_time_sliced_dynamic``.
    Once a config declares ``dynamic_mode``, that declaration is authoritative
    and inconsistent legacy flags fail loudly.
    """
    deform_cfg = training_config.get("deform_cfg", {})
    deform_enabled = bool(training_config.get("deform", False))
    time_slice_enabled = bool(
        deform_cfg.get("oracle_time_sliced_dynamic", False))
    explicit = training_config.get("dynamic_mode")

    if explicit is None:
        if deform_enabled and time_slice_enabled:
            raise ValueError(
                "ambiguous dynamic configuration: DeformNet and oracle time-slice "
                "are both enabled")
        if time_slice_enabled:
            return DYNAMIC_MODE_ORACLE_TIME_SLICE
        if deform_enabled:
            return DYNAMIC_MODE_DEFORM
        return DYNAMIC_MODE_NONE

    mode = str(explicit).strip().lower()
    if mode not in VALID_DYNAMIC_MODES:
        raise ValueError(
            f"unknown dynamic_mode '{explicit}'; expected one of "
            f"{sorted(VALID_DYNAMIC_MODES)}")
    if deform_enabled != (mode == DYNAMIC_MODE_DEFORM):
        raise ValueError(
            f"dynamic_mode '{mode}' conflicts with Training.deform="
            f"{deform_enabled}")
    if time_slice_enabled != (mode == DYNAMIC_MODE_ORACLE_TIME_SLICE):
        raise ValueError(
            f"dynamic_mode '{mode}' conflicts with "
            f"oracle_time_sliced_dynamic={time_slice_enabled}")
    return mode
