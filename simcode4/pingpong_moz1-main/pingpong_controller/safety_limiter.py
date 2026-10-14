"""Publish-time safety limiter for the right-arm joint command.

Hardcoded joint position, velocity, and acceleration limits for the
right arm.  The limiter projects every command onto a discrete-time
viability interval: besides satisfying the limits on this tick, the
result can still brake to rest before either position boundary.  This
avoids a subtle failure of a final position clip, which can turn a
bounded velocity into an unbounded one-tick deceleration.

Joint order is fixed: RightArm-0, RightArm-1, ..., RightArm-6.
"""

from __future__ import annotations

import numpy as np


def stopping_velocity_limit(
    distance_rad: np.ndarray,
    acc_limit_rad_s2: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Return the largest outward velocity that can stop in ``distance``.

    The command integrator uses the semi-implicit discrete update
    ``q_next = q + v_next * dt``.  If ``h = a_max * dt`` and the chosen
    speed falls in ``[n*h, (n+1)*h)``, maximum braking needs the distance

    ``dt * ((n + 1) * speed - h * n * (n + 1) / 2)``.

    This function is the exact piecewise inverse of that distance.  It is
    vectorised per joint and intentionally contains no continuous-time
    approximation.
    """

    distance = np.maximum(np.asarray(distance_rad, dtype=np.float64), 0.0)
    acc_limit = np.asarray(acc_limit_rad_s2, dtype=np.float64)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive and finite, got {dt}")
    if np.any(~np.isfinite(acc_limit)) or np.any(acc_limit <= 0.0):
        raise ValueError("acceleration limits must be positive and finite")

    h = acc_limit * float(dt)
    scaled_distance = 8.0 * distance / (acc_limit * float(dt) ** 2)
    brake_steps = np.floor(
        0.5 * (np.sqrt(1.0 + scaled_distance) - 1.0)
    )
    return (
        distance / (float(dt) * (brake_steps + 1.0))
        + 0.5 * h * brake_steps
    )


def project_safe_command_step(
    target_rad: np.ndarray,
    current_cmd_rad: np.ndarray,
    current_vel_rad_s: np.ndarray,
    pos_low_rad: np.ndarray,
    pos_high_rad: np.ndarray,
    vel_limit_rad_s: np.ndarray,
    acc_limit_rad_s2: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project one command tick onto the position/velocity/acceleration set.

    Returns ``(q_next, v_next, interval_low, interval_high)``.  A valid
    initial state (position in range, zero velocity is sufficient) remains
    valid forever.  If an externally supplied state is outside the
    viability kernel, no command can satisfy all hard limits; in that case
    this function raises instead of silently breaking one of them.
    """

    target = np.asarray(target_rad, dtype=np.float64)
    current_q = np.asarray(current_cmd_rad, dtype=np.float64)
    current_v = np.asarray(current_vel_rad_s, dtype=np.float64)
    pos_low = np.asarray(pos_low_rad, dtype=np.float64)
    pos_high = np.asarray(pos_high_rad, dtype=np.float64)
    vel_limit = np.asarray(vel_limit_rad_s, dtype=np.float64)
    acc_limit = np.asarray(acc_limit_rad_s2, dtype=np.float64)

    arrays = (target, current_q, current_v, pos_low, pos_high, vel_limit, acc_limit)
    if any(array.shape != current_q.shape for array in arrays):
        raise ValueError("all limiter arrays must have the same shape")
    if any(np.any(~np.isfinite(array)) for array in arrays):
        raise ValueError("all limiter arrays must be finite")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"dt must be positive and finite, got {dt}")
    if np.any(pos_low >= pos_high):
        raise ValueError("each lower position limit must be below its upper limit")
    if np.any(vel_limit <= 0.0) or np.any(acc_limit <= 0.0):
        raise ValueError("velocity and acceleration limits must be positive")
    tolerance = 64.0 * np.finfo(np.float64).eps
    if np.any(current_q < pos_low - tolerance) or np.any(current_q > pos_high + tolerance):
        raise RuntimeError("limiter position state is outside its hard bounds")

    current_q = np.minimum(np.maximum(current_q, pos_low), pos_high)
    target = np.minimum(np.maximum(target, pos_low), pos_high)
    accel_step = acc_limit * float(dt)
    lower_stop = stopping_velocity_limit(current_q - pos_low, acc_limit, dt)
    upper_stop = stopping_velocity_limit(pos_high - current_q, acc_limit, dt)

    interval_low = np.maximum.reduce(
        (
            -vel_limit,
            current_v - accel_step,
            (pos_low - current_q) / float(dt),
            -lower_stop,
        )
    )
    interval_high = np.minimum.reduce(
        (
            vel_limit,
            current_v + accel_step,
            (pos_high - current_q) / float(dt),
            upper_stop,
        )
    )
    interval_tolerance = 256.0 * np.finfo(np.float64).eps
    if np.any(interval_low > interval_high + interval_tolerance):
        raise RuntimeError(
            "limiter state is outside the position/velocity/acceleration "
            "viability kernel"
        )
    interval_mid = 0.5 * (interval_low + interval_high)
    interval_low = np.minimum(interval_low, interval_mid)
    interval_high = np.maximum(interval_high, interval_mid)

    desired_vel = (target - current_q) / float(dt)
    next_vel = np.minimum(np.maximum(desired_vel, interval_low), interval_high)
    next_q = current_q + next_vel * float(dt)
    return next_q, next_vel, interval_low, interval_high


def project_target_tracking_command_step(
    target_rad: np.ndarray,
    current_cmd_rad: np.ndarray,
    current_vel_rad_s: np.ndarray,
    pos_low_rad: np.ndarray,
    pos_high_rad: np.ndarray,
    vel_limit_rad_s: np.ndarray,
    acc_limit_rad_s2: np.ndarray,
    dt: float,
    target_vel_rad_s: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Plan one bounded trapezoidal-profile step toward a position target.

    ``project_safe_command_step`` is a safety projection: it uses the fastest
    feasible velocity toward the requested position and only reserves braking
    distance for the hard joint boundaries.  That is appropriate as a final
    safety envelope, but a hardware motion planner must also brake before the
    *current target*.  Otherwise a rapidly changing inverse-MPC position input
    can make the planned command overshoot and chase the target indefinitely.

    This planner reuses the exact position-bound viability interval, then caps
    the desired target-directed velocity by the exact discrete stopping speed
    for the remaining target distance.  The returned q/dq therefore satisfy
    position, velocity and acceleration limits while following a conventional
    acceleration-limited trapezoidal position profile.
    """

    target = np.asarray(target_rad, dtype=np.float64)
    current_q = np.asarray(current_cmd_rad, dtype=np.float64)
    current_v = np.asarray(current_vel_rad_s, dtype=np.float64)
    acc_limit = np.asarray(acc_limit_rad_s2, dtype=np.float64)
    _, _, interval_low, interval_high = project_safe_command_step(
        target,
        current_q,
        current_v,
        pos_low_rad,
        pos_high_rad,
        vel_limit_rad_s,
        acc_limit,
        dt,
    )
    position_error = target - current_q
    if target_vel_rad_s is None:
        target_vel = np.zeros_like(position_error)
    else:
        target_vel = np.asarray(target_vel_rad_s, dtype=np.float64)
        if target_vel.shape != position_error.shape or np.any(~np.isfinite(target_vel)):
            raise ValueError("target_vel_rad_s must match the finite target shape")
        target_vel = np.minimum(
            np.maximum(target_vel, -np.asarray(vel_limit_rad_s, dtype=np.float64)),
            np.asarray(vel_limit_rad_s, dtype=np.float64),
        )
    stop_speed = stopping_velocity_limit(np.abs(position_error), acc_limit, dt)
    relative_vel = np.sign(position_error) * np.minimum(
        np.abs(position_error) / float(dt),
        stop_speed,
    )
    desired_vel = target_vel + relative_vel
    next_vel = np.minimum(np.maximum(desired_vel, interval_low), interval_high)
    next_q = current_q + next_vel * float(dt)
    return next_q, next_vel, interval_low, interval_high


class RightArmCommandSafetyLimiter:
    """Position-aware velocity/acceleration safety limiter.

    Usage
    -----
    >>> limiter = RightArmCommandSafetyLimiter(initial_cmd_rad, dt=0.005)
    >>> safe_cmd = limiter.filter(raw_cmd)   # call every control tick
    >>> counts = limiter.consume_clip_counts()  # periodic throttled log

    All internal math is float64 to match `mc_core_interface/MechUnitCmd`
    (`float64[] jnt_pos`).
    """

    N_JOINTS = 7

    # Position limits (rad) from models/moz1_pd.xml RightArm joint ranges.
    POS_LIMIT_LOW_RAD = np.array([
        -2.0944,    # RightArm-0
        -2.96706,   # RightArm-1
        -3.05433,   # RightArm-2
        -0.174533,  # RightArm-3
        -3.05433,   # RightArm-4
        -1.65806,   # RightArm-5
        -1.5708,    # RightArm-6
    ], dtype=np.float64)
    POS_LIMIT_HIGH_RAD = np.array([
        3.14159,    # RightArm-0
        0.15708,    # RightArm-1
        3.05433,    # RightArm-2
        2.25147,    # RightArm-3
        3.05433,    # RightArm-4
        1.65806,    # RightArm-5
        1.5708,     # RightArm-6
    ], dtype=np.float64)

    # Velocity / acceleration limits (deg/s, deg/s^2).
    VEL_LIMIT_DEG_S = np.array(
        [210.0, 210.0, 240.0, 240.0, 300.0, 300.0, 300.0], dtype=np.float64)
    ACC_LIMIT_DEG_S2 = np.array(
        [1300.0, 1300.0, 1800.0, 3000.0, 3000.0, 3000.0, 3000.0],
        dtype=np.float64)

    def __init__(self, initial_cmd_rad, *, dt: float = 0.005):
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be positive and finite, got {dt}")
        self.dt = float(dt)
        self.vel_limit_rad_s = np.deg2rad(self.VEL_LIMIT_DEG_S)
        self.acc_limit_rad_s2 = np.deg2rad(self.ACC_LIMIT_DEG_S2)

        init = np.asarray(initial_cmd_rad, dtype=np.float64).reshape(-1)
        if init.shape[0] != self.N_JOINTS or not np.all(np.isfinite(init)):
            raise ValueError(
                f"initial_cmd_rad must be a finite length-{self.N_JOINTS} "
                "array (rad)")
        init = np.clip(init, self.POS_LIMIT_LOW_RAD, self.POS_LIMIT_HIGH_RAD)
        self.last_cmd = init.copy()
        self.last_vel = np.zeros(self.N_JOINTS, dtype=np.float64)
        self.last_acc = np.zeros(self.N_JOINTS, dtype=np.float64)

        # Running clip counters; reset by consume_clip_counts().
        self._n_invalid = 0
        self._n_pos_clip = 0
        self._n_vel_clip = 0
        self._n_acc_clip = 0

    def reset(self, current_cmd_rad) -> None:
        """Re-anchor the limiter to a known joint state (rad).

        Use this at most once on startup, once the first valid right-arm
        `/joint_states` feedback arrives, so the next safety-limited
        command starts from the real position instead of the coarse
        initial pose. Calling this mid-run would let the integrator
        jump and break the velocity/acceleration limits, so don't.
        """
        current = np.asarray(current_cmd_rad, dtype=np.float64).reshape(-1)
        if (current.shape[0] != self.N_JOINTS
                or not np.all(np.isfinite(current))):
            raise ValueError(
                f"current_cmd_rad must be a finite length-{self.N_JOINTS} "
                "array (rad)")
        self.last_cmd = np.clip(
            current, self.POS_LIMIT_LOW_RAD, self.POS_LIMIT_HIGH_RAD)
        self.last_vel = np.zeros(self.N_JOINTS, dtype=np.float64)
        self.last_acc = np.zeros(self.N_JOINTS, dtype=np.float64)

    def filter(self, raw_cmd) -> np.ndarray:
        """Return a safety-bounded 7-DOF joint command (rad) and update state.

        A malformed input (wrong length, NaN, Inf) requests zero velocity
        and therefore performs an acceleration-bounded stop.  Freezing the
        position immediately while the previous command is moving would
        itself violate the acceleration limit.
        """
        raw = np.asarray(raw_cmd, dtype=np.float64).reshape(-1)
        invalid = raw.shape[0] != self.N_JOINTS or not np.all(np.isfinite(raw))
        if invalid:
            self._n_invalid += 1
            raw = self.last_cmd.copy()

        # (2) Clip raw command to position limits.
        raw_clipped = np.clip(
            raw, self.POS_LIMIT_LOW_RAD, self.POS_LIMIT_HIGH_RAD)
        if np.any(raw != raw_clipped):
            self._n_pos_clip += 1

        # Desired per-tick velocity to track the target.  The counters below
        # retain their historical meanings even though the final projection
        # is now one joint-wise feasible interval rather than sequential
        # clips followed by an unsafe final position clip.
        desired_vel = (raw_clipped - self.last_cmd) / self.dt
        vel_clamped = np.clip(
            desired_vel, -self.vel_limit_rad_s, self.vel_limit_rad_s)
        if np.any(vel_clamped != desired_vel):
            self._n_vel_clip += 1
        acc_lo = self.last_vel - self.acc_limit_rad_s2 * self.dt
        acc_hi = self.last_vel + self.acc_limit_rad_s2 * self.dt
        acc_clamped = np.clip(vel_clamped, acc_lo, acc_hi)
        if np.any(acc_clamped != vel_clamped):
            self._n_acc_clip += 1

        candidate, actual_vel, feasible_lo, feasible_hi = project_safe_command_step(
            raw_clipped,
            self.last_cmd,
            self.last_vel,
            self.POS_LIMIT_LOW_RAD,
            self.POS_LIMIT_HIGH_RAD,
            self.vel_limit_rad_s,
            self.acc_limit_rad_s2,
            self.dt,
        )
        if np.any(acc_clamped < feasible_lo) or np.any(acc_clamped > feasible_hi):
            self._n_pos_clip += 1
        prev_vel = self.last_vel.copy()
        actual_acc = (actual_vel - prev_vel) / self.dt
        self.last_cmd = candidate.copy()
        self.last_vel = actual_vel.copy()
        self.last_acc = actual_acc.copy()
        return candidate.copy()

    def consume_clip_counts(self) -> dict:
        """Return running clip counts and reset them to zero."""
        counts = {
            "invalid": int(self._n_invalid),
            "pos": int(self._n_pos_clip),
            "vel": int(self._n_vel_clip),
            "acc": int(self._n_acc_clip),
        }
        self._n_invalid = 0
        self._n_pos_clip = 0
        self._n_vel_clip = 0
        self._n_acc_clip = 0
        return counts
