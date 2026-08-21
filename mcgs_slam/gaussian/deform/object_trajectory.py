"""Object-centric continuous rigid motion for dynamic Gaussian maps.

Quaternions use the Gaussian convention ``(w, x, y, z)``.  Knot times are
buffers, while translation and rotation knots are parameters so a later
mapping stage can optimize them jointly with object-local Gaussians.
"""

import torch
from torch import nn


def normalize_quaternion(quaternion, eps=1e-8):
    return quaternion / torch.linalg.vector_norm(
        quaternion, dim=-1, keepdim=True).clamp_min(eps)


def quaternion_slerp(q0, q1, alpha):
    """Shortest-path spherical interpolation for ``(w,x,y,z)`` quaternions."""
    q0 = normalize_quaternion(q0)
    q1 = normalize_quaternion(q1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.sum(q0 * q1, dim=-1, keepdim=True).clamp(-1.0, 1.0)

    alpha = torch.as_tensor(alpha, dtype=q0.dtype, device=q0.device)
    while alpha.ndim < dot.ndim:
        alpha = alpha.unsqueeze(-1)

    linear = normalize_quaternion((1.0 - alpha) * q0 + alpha * q1)
    # acos'(dot) = -1/sqrt(1-dot^2) is infinite at dot = +-1. torch.where below
    # selects the linear branch there, but autograd still evaluates the gradient
    # of *both* branches, so an infinite d(theta)/d(dot) becomes inf * 0 = NaN on
    # q0/q1. Adjacent trajectory knots are almost always near-identical rotations
    # (dot ~ 1), so this fired constantly. Clamp strictly inside (-1, 1) to keep
    # the unselected branch's gradient finite.
    safe_dot = dot.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    theta = torch.acos(safe_dot)
    sin_theta = torch.sin(theta)
    safe_sin = sin_theta.clamp_min(1e-8)
    spherical = (
        torch.sin((1.0 - alpha) * theta) / safe_sin * q0
        + torch.sin(alpha * theta) / safe_sin * q1
    )
    return normalize_quaternion(
        torch.where(dot.abs() > 0.9995, linear, spherical))


def quaternion_multiply(a, b):
    """Hamilton product of ``(w,x,y,z)`` quaternions (applies ``b`` then ``a``)."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), dim=-1)


def quaternion_conjugate(quaternion):
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack((w, -x, -y, -z), dim=-1)


def _stack_rows(rows, width, dtype):
    """Build an (N, width) tensor from a sequence that may hold tensors.

    ``torch.as_tensor([t])`` raises when ``t`` is a multi-element tensor, which
    is exactly what callers holding tensors naturally pass (see
    ``ObjectTrajectoryTable.observe``). Stack instead when we are given a
    sequence of per-row tensors.
    """
    if isinstance(rows, torch.Tensor):
        return rows.to(dtype=dtype).reshape(-1, width)
    rows = list(rows)
    if rows and any(isinstance(row, torch.Tensor) for row in rows):
        return torch.stack([
            torch.as_tensor(row, dtype=dtype).reshape(width) for row in rows])
    return torch.as_tensor(rows, dtype=dtype)


def quaternion_to_matrix(quaternion):
    """Convert a normalized ``(w,x,y,z)`` quaternion to a 3x3 matrix."""
    w, x, y, z = normalize_quaternion(quaternion).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quaternion.shape[:-1] + (3, 3))


class ObjectSE3Trajectory(nn.Module):
    """Piecewise-continuous SE(3) trajectory with linear translation and SLERP."""

    def __init__(self, knot_times, translations, rotations=None,
                 observation_counts=None):
        super().__init__()
        times = torch.as_tensor(knot_times, dtype=torch.float64).reshape(-1)
        # Checked before the shape test below: with no knots `translations` is an
        # empty list whose tensor has shape (0,), not (0, 3), so the shape test
        # would fire first and report a misleading message.
        if times.numel() == 0:
            raise ValueError("an object trajectory needs at least one knot")
        translations = _stack_rows(translations, 3, torch.float32)
        if translations.shape != (times.numel(), 3):
            raise ValueError("translations must have shape (num_knots, 3)")
        if rotations is None:
            rotations = torch.zeros((times.numel(), 4), dtype=torch.float32)
            rotations[:, 0] = 1.0
        rotations = _stack_rows(rotations, 4, torch.float32)
        if rotations.shape != (times.numel(), 4):
            raise ValueError("rotations must have shape (num_knots, 4)")

        order = torch.argsort(times)
        times = times[order]
        if times.numel() > 1 and torch.any(times[1:] <= times[:-1]):
            raise ValueError("knot times must be unique")
        translations = translations[order]
        rotations = normalize_quaternion(rotations[order])
        if observation_counts is None:
            observation_counts = torch.ones(times.numel(), dtype=torch.long)
        counts = torch.as_tensor(observation_counts, dtype=torch.long).reshape(-1)[order]
        if counts.shape != times.shape or torch.any(counts <= 0):
            raise ValueError("observation_counts must be positive per-knot values")

        self.register_buffer("knot_times", times)
        self.register_buffer("observation_counts", counts)
        self.translations = nn.Parameter(translations)
        self.rotations = nn.Parameter(rotations)

    @property
    def first_time(self):
        return float(self.knot_times[0].item())

    @property
    def last_time(self):
        return float(self.knot_times[-1].item())

    def is_alive(self, time, tolerance=1e-6):
        time = float(time)
        return self.first_time - tolerance <= time <= self.last_time + tolerance

    @staticmethod
    def _grow_parameter(parameter, new_data, insert_at, optimizer):
        """Replace a knot Parameter with a longer one, keeping training intact.

        A leaf Parameter cannot simply be resized via ``.data``: its cached
        autograd metadata (the ``AccumulateGrad`` node) keeps the original row
        count, so the next backward fails with "returned an invalid gradient".
        A new Parameter is therefore required -- which is exactly why
        ``GaussianModel.cat_tensors_to_optimizer`` also builds one.

        The previous behaviour built that new Parameter but told nobody, so any
        optimizer created earlier over ``parameters()`` kept stepping the old,
        detached tensor that no longer feeds ``evaluate()``. Passing
        ``optimizer`` swaps the tensor inside its ``param_groups`` and grows its
        row-indexed per-parameter state (Adam moments, SGD momentum) at the same
        position, so training continues uninterrupted.
        """
        def _grow_rows(buffer):
            return torch.cat((buffer[:insert_at],
                              torch.zeros_like(buffer[:1]),
                              buffer[insert_at:]), dim=0)

        grown = nn.Parameter(new_data, requires_grad=parameter.requires_grad)
        if parameter.grad is not None:
            grown.grad = _grow_rows(parameter.grad)

        if optimizer is not None:
            for group in optimizer.param_groups:
                for position, existing in enumerate(group["params"]):
                    if existing is not parameter:
                        continue
                    state = optimizer.state.pop(parameter, None)
                    if state:
                        for key, buffer in list(state.items()):
                            if (isinstance(buffer, torch.Tensor)
                                    and buffer.shape[:1] == parameter.shape[:1]):
                                state[key] = _grow_rows(buffer)
                        optimizer.state[grown] = state
                    group["params"][position] = grown
        return grown

    def add_observation(self, time, translation, rotation=None, tolerance=1e-6,
                        optimizer=None):
        """Insert a knot or average a repeated-time multi-view observation.

        Pass ``optimizer`` when knots are inserted during training so its
        per-parameter state is grown alongside the knots.
        """
        time = float(time)
        translation = torch.as_tensor(
            translation, dtype=self.translations.dtype,
            device=self.translations.device).reshape(3)
        if rotation is None:
            rotation = torch.tensor(
                [1.0, 0.0, 0.0, 0.0], dtype=self.rotations.dtype,
                device=self.rotations.device)
        rotation = normalize_quaternion(torch.as_tensor(
            rotation, dtype=self.rotations.dtype,
            device=self.rotations.device).reshape(4))

        distances = torch.abs(self.knot_times - time)
        existing = torch.nonzero(distances <= tolerance).reshape(-1)
        if existing.numel():
            index = int(existing[0].item())
            count = int(self.observation_counts[index].item())
            alpha = 1.0 / (count + 1)
            with torch.no_grad():
                self.translations[index].lerp_(translation, alpha)
                self.rotations[index].copy_(quaternion_slerp(
                    self.rotations[index], rotation, alpha))
                self.observation_counts[index] += 1
            return index, False

        insert_at = int(torch.searchsorted(
            self.knot_times, torch.tensor(time, dtype=torch.float64,
                                          device=self.knot_times.device)).item())
        time_value = torch.tensor(
            [time], dtype=self.knot_times.dtype, device=self.knot_times.device)
        count_value = torch.ones(
            1, dtype=self.observation_counts.dtype,
            device=self.observation_counts.device)
        new_times = torch.cat(
            (self.knot_times[:insert_at], time_value,
             self.knot_times[insert_at:]))
        new_counts = torch.cat(
            (self.observation_counts[:insert_at], count_value,
             self.observation_counts[insert_at:]))
        new_translations = torch.cat(
            (self.translations.detach()[:insert_at], translation[None],
             self.translations.detach()[insert_at:]))
        new_rotations = torch.cat(
            (self.rotations.detach()[:insert_at], rotation[None],
             self.rotations.detach()[insert_at:]))
        self.knot_times = new_times
        self.observation_counts = new_counts
        self.translations = self._grow_parameter(
            self.translations, new_translations, insert_at, optimizer)
        self.rotations = self._grow_parameter(
            self.rotations, new_rotations, insert_at, optimizer)
        return insert_at, True

    def evaluate(self, time):
        """Evaluate a pose, clamping outside the observed lifecycle."""
        time = torch.as_tensor(
            time, dtype=self.knot_times.dtype, device=self.knot_times.device)
        if time.numel() != 1:
            raise ValueError("trajectory evaluation expects one time value")
        time = time.reshape(())
        if self.knot_times.numel() == 1 or time <= self.knot_times[0]:
            return self.translations[0], normalize_quaternion(self.rotations[0])
        if time >= self.knot_times[-1]:
            return self.translations[-1], normalize_quaternion(self.rotations[-1])

        right = int(torch.searchsorted(self.knot_times, time).item())
        left = right - 1
        span = self.knot_times[right] - self.knot_times[left]
        alpha = ((time - self.knot_times[left]) / span).to(
            dtype=self.translations.dtype)
        translation = torch.lerp(
            self.translations[left], self.translations[right], alpha)
        rotation = quaternion_slerp(
            self.rotations[left], self.rotations[right], alpha)
        return translation, rotation

    def transform_points(self, local_points, time):
        translation, rotation = self.evaluate(time)
        matrix = quaternion_to_matrix(rotation)
        return local_points @ matrix.transpose(-1, -2) + translation

    def checkpoint_state(self):
        return {
            "knot_times": self.knot_times.detach().cpu().clone(),
            "translations": self.translations.detach().cpu().clone(),
            "rotations": normalize_quaternion(
                self.rotations.detach()).cpu().clone(),
            "observation_counts": self.observation_counts.detach().cpu().clone(),
        }

    @classmethod
    def from_checkpoint_state(cls, state):
        return cls(
            state["knot_times"],
            state["translations"],
            state["rotations"],
            state.get("observation_counts"),
        )


class ObjectTrajectoryTable(nn.Module):
    """Collection of independently moving rigid-object trajectories."""

    def __init__(self):
        super().__init__()
        self.trajectories = nn.ModuleDict()

    @staticmethod
    def _key(object_id):
        object_id = int(object_id)
        if object_id < 0:
            raise ValueError("dynamic object IDs must be non-negative")
        return str(object_id)

    @property
    def object_ids(self):
        return sorted(int(key) for key in self.trajectories.keys())

    def _table_device(self, fallback=None):
        """Device the table's existing tensors live on.

        ``nn.ModuleDict.__setitem__`` does not inherit the parent module's
        device, so a submodule built inside an already-``.cuda()``-ed table
        would otherwise stay on the CPU until something crashes on mixed
        devices at render time.
        """
        for tensor in self.parameters():
            return tensor.device
        for tensor in self.buffers():
            return tensor.device
        if isinstance(fallback, torch.Tensor):
            return fallback.device
        return None

    def observe(self, object_id, time, translation, rotation=None,
                optimizer=None):
        key = self._key(object_id)
        if key not in self.trajectories:
            trajectory = ObjectSE3Trajectory(
                [time], [translation], None if rotation is None else [rotation])
            device = self._table_device(fallback=translation)
            if device is not None:
                trajectory = trajectory.to(device)
            self.trajectories[key] = trajectory
            if optimizer is not None:
                # A brand-new object's knots belong to no param group yet, so
                # without this its trajectory would silently never train while
                # its Gaussians are already being rendered through it.
                optimizer.add_param_group(
                    {"params": list(trajectory.parameters())})
            return 0, True
        return self.trajectories[key].add_observation(
            time, translation, rotation, optimizer=optimizer)

    def observe_centroids(self, xyz, object_ids, source_times, optimizer=None):
        """Register a knot per (object, source time) from that group's centroid.

        This is how trajectories are actually built: the oracle motion gate says
        which rows are dynamic and which object they belong to, and the centroid
        of those rows is the object's observed position at that time.

        It is not the object's true centre -- a view only seeds the surface
        facing it, so the centroid sits toward the camera. What the render path
        uses is the *relative* transform between two nearby times, where a bias
        that changes slowly with viewpoint largely cancels.

        Returns the sorted distinct source times registered, so callers can keep
        their own list of observed times without re-scanning the device tensors.
        """
        object_ids = object_ids.reshape(-1)
        source_times = source_times.reshape(-1)
        dynamic = object_ids >= 0
        if not bool(dynamic.any()):
            return []
        registered = set()
        for object_id in torch.unique(object_ids[dynamic]).tolist():
            object_mask = object_ids == object_id
            for source in torch.unique(source_times[object_mask]).tolist():
                mask = torch.logical_and(object_mask, source_times == source)
                self.observe(
                    int(object_id), float(source),
                    xyz[mask].mean(dim=0).detach(), optimizer=optimizer)
                registered.add(float(source))
        return sorted(registered)

    def evaluate(self, object_id, time):
        key = self._key(object_id)
        if key not in self.trajectories:
            raise KeyError(f"unknown dynamic object ID {object_id}")
        return self.trajectories[key].evaluate(time)

    def transform_gaussians(self, canonical_xyz, object_ids, time):
        """Transform object-local rows; rows with object ID -1 remain static."""
        object_ids = object_ids.reshape(-1).to(device=canonical_xyz.device)
        if object_ids.shape[0] != canonical_xyz.shape[0]:
            raise ValueError("object_ids and canonical_xyz must have equal rows")
        output = canonical_xyz.clone()
        for object_id in torch.unique(object_ids[object_ids >= 0]).tolist():
            key = self._key(object_id)
            if key not in self.trajectories:
                raise KeyError(f"unknown dynamic object ID {object_id}")
            mask = object_ids == int(object_id)
            output[mask] = self.trajectories[key].transform_points(
                canonical_xyz[mask], time)
        return output

    def relative_transform(self, object_id, from_time, to_time):
        """Rigid motion carrying the object's pose at ``from_time`` to ``to_time``.

        Returns ``(rotation_matrix, translation, rotation_quaternion)`` for
        ``T(to) @ inv(T(from))``. The quaternion is the same rotation, supplied
        so callers can carry Gaussian orientations without a matrix->quaternion
        conversion.
        """
        key = self._key(object_id)
        if key not in self.trajectories:
            raise KeyError(f"unknown dynamic object ID {object_id}")
        trajectory = self.trajectories[key]
        t_from, q_from = trajectory.evaluate(from_time)
        t_to, q_to = trajectory.evaluate(to_time)
        q_relative = normalize_quaternion(
            quaternion_multiply(q_to, quaternion_conjugate(q_from)))
        rotation = quaternion_to_matrix(q_relative)
        translation = t_to - rotation @ t_from
        return rotation, translation, q_relative

    def transform_to_time(self, xyz, object_ids, source_times, target_time,
                          rotations=None, groups=None):
        """Move each dynamic row from the time it was observed to ``target_time``.

        World-frame ``xyz`` is kept as-is (no object-local storage change): a row
        captured at ``source_times[i]`` is carried by the object's own relative
        motion, so the result is the identity when the two times match -- i.e.
        this reproduces the per-timestamp bank exactly at observed times while
        also being defined at times in between, which the bank is not.
        Rows with object ID < 0 are static and pass through untouched.

        ``rotations`` are the Gaussians' own ``(w,x,y,z)`` orientation
        quaternions. Pass them whenever the object can rotate: carrying only
        centres leaves an anisotropic Gaussian at the right place with a stale
        covariance, which the per-timestamp bank got right for free by storing a
        separate row per time. Returns ``moved_xyz`` alone when ``rotations`` is
        ``None``, otherwise ``(moved_xyz, moved_rotations)``.

        ``groups`` is an optional list of ``(object_id, source_time)`` pairs the
        caller already knows.  Deriving them here costs two
        ``torch.unique(...).tolist()`` scans, each a device->host sync, on every
        call -- acceptable offline but not on the render hot path.  Supplying a
        pair asserts that *every* row with that object ID came from that source
        time, so the per-row source check is skipped; the caller owns that
        guarantee (see ``object_render.object_se3_overrides``, which masks the
        rows itself with the same predicate that gates their opacity).

        Raises ``KeyError`` for an object ID with no registered trajectory,
        matching ``transform_gaussians``/``relative_transform``: a missing
        trajectory means the object cannot be placed, and silently leaving it at
        a stale position would be a wrong render with no diagnostic.
        """
        object_ids = object_ids.reshape(-1).to(device=xyz.device)
        source_times = source_times.reshape(-1).to(device=xyz.device)
        if object_ids.shape[0] != xyz.shape[0]:
            raise ValueError("object_ids and xyz must have equal rows")
        if source_times.shape[0] != xyz.shape[0]:
            raise ValueError("source_times and xyz must have equal rows")
        if rotations is not None and rotations.shape[0] != xyz.shape[0]:
            raise ValueError("rotations and xyz must have equal rows")

        derived = groups is None
        if derived:
            groups = []
            for object_id in torch.unique(object_ids[object_ids >= 0]).tolist():
                object_mask = object_ids == int(object_id)
                # rows of one object can carry different source times, and the
                # relative transform depends on that time, so group by it
                for source in torch.unique(source_times[object_mask]).tolist():
                    groups.append((int(object_id), float(source)))

        output = xyz.clone()
        output_rotations = None if rotations is None else rotations.clone()
        for object_id, source in groups:
            key = self._key(int(object_id))
            if key not in self.trajectories:
                raise KeyError(f"unknown dynamic object ID {int(object_id)}")
            mask = object_ids == int(object_id)
            if derived:
                mask = torch.logical_and(mask, source_times == source)
            rotation, translation, q_relative = self.relative_transform(
                int(object_id), source, target_time)
            output[mask] = (
                xyz[mask] @ rotation.transpose(-1, -2) + translation)
            if output_rotations is not None:
                output_rotations[mask] = normalize_quaternion(
                    quaternion_multiply(
                        q_relative.expand_as(rotations[mask]),
                        rotations[mask]))
        if output_rotations is None:
            return output
        return output, output_rotations

    def visibility_mask(self, object_ids, time, tolerance=1e-6):
        """Keep static rows and objects inside their observed lifecycle."""
        object_ids = object_ids.reshape(-1)
        visible = object_ids < 0
        for object_id in torch.unique(object_ids[object_ids >= 0]).tolist():
            key = self._key(object_id)
            alive = (
                key in self.trajectories
                and self.trajectories[key].is_alive(time, tolerance=tolerance)
            )
            visible = torch.logical_or(
                visible, torch.logical_and(
                    object_ids == int(object_id),
                    torch.tensor(alive, device=visible.device)))
        return visible

    def checkpoint_state(self):
        return {
            key: trajectory.checkpoint_state()
            for key, trajectory in self.trajectories.items()
        }

    def restore_checkpoint_state(self, state, device=None):
        self.trajectories = nn.ModuleDict({
            str(key): ObjectSE3Trajectory.from_checkpoint_state(value)
            for key, value in state.items()
        })
        if device is not None:
            self.to(device)
