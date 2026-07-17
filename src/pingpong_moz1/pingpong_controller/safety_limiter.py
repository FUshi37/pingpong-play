"""Publish-time safety limiter for the right-arm joint command.

Hardcoded joint position, velocity, and acceleration limits for the
right arm. Applies them in order so that the MechUnitCmd sent to the
robot never exceeds hardware-safe bounds, regardless of what the RL
policy emits. Limits come from moz1_pd.xml RightArm joint ranges and
the training-env velocity/acceleration configuration.

Joint order is fixed: RightArm-0, RightArm-1, ..., RightArm-6.
"""

from __future__ import annotations

import numpy as np


class RightArmCommandSafetyLimiter:
    """Three-layer (position / velocity / acceleration) safety limiter.

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

    def __init__(
            self, initial_cmd_rad, *, dt: float = 0.005,
            parking_velocity_scale: float = 0.08,
            parking_acceleration_scale: float = 0.06):
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"dt must be positive and finite, got {dt}")
        self.dt = float(dt)
        self.vel_limit_rad_s = np.deg2rad(self.VEL_LIMIT_DEG_S)
        self.acc_limit_rad_s2 = np.deg2rad(self.ACC_LIMIT_DEG_S2)
        self.parking_velocity_scale = self._finite_scale(
            parking_velocity_scale, default=0.08)
        self.parking_acceleration_scale = self._finite_scale(
            parking_acceleration_scale, default=0.06)

        init = np.asarray(initial_cmd_rad, dtype=np.float64).reshape(-1)
        if init.shape[0] != self.N_JOINTS or not np.all(np.isfinite(init)):
            raise ValueError(
                f"initial_cmd_rad must be a finite length-{self.N_JOINTS} "
                "array (rad)")
        init = np.clip(init, self.POS_LIMIT_LOW_RAD, self.POS_LIMIT_HIGH_RAD)
        self.last_cmd = init.copy()
        self.last_vel = np.zeros(self.N_JOINTS, dtype=np.float64)

        # Running clip counters; reset by consume_clip_counts().
        self._n_invalid = 0
        self._n_pos_clip = 0
        self._n_vel_clip = 0
        self._n_acc_clip = 0

    @staticmethod
    def _finite_scale(value: float, *, default: float) -> float:
        try:
            scale = float(value)
        except (TypeError, ValueError):
            scale = float(default)
        if not np.isfinite(scale):
            scale = float(default)
        return float(np.clip(scale, 1e-3, 1.0))

    def profile_limits(self, profile: str = "normal") -> tuple[np.ndarray, np.ndarray]:
        """Return per-joint velocity/acceleration limits for a command profile."""
        name = str(profile or "normal").strip().lower()
        if name in {"parking", "park", "safe", "reset", "failsafe"}:
            return (
                self.vel_limit_rad_s * self.parking_velocity_scale,
                self.acc_limit_rad_s2 * self.parking_acceleration_scale,
            )
        return self.vel_limit_rad_s, self.acc_limit_rad_s2

    def reset(self, current_cmd_rad) -> None:
        """Re-anchor the limiter to a known joint state (rad).

        Use this on startup or when entering a deliberate parking/reset path,
        after sampling the real current arm state. The next filtered command
        then starts from that state with zero commanded velocity instead of
        jumping from a stale internal target.
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

    def observe(self, current_cmd_rad) -> None:
        """Track an externally-published command without modifying it.

        Use this when the normal limiter is disabled. It keeps the internal
        position/velocity state warm so a later parking/failsafe profile starts
        from the latest command instead of an old limiter state.
        """
        current = np.asarray(current_cmd_rad, dtype=np.float64).reshape(-1)
        if (current.shape[0] != self.N_JOINTS
                or not np.all(np.isfinite(current))):
            self._n_invalid += 1
            return
        current = np.clip(
            current, self.POS_LIMIT_LOW_RAD, self.POS_LIMIT_HIGH_RAD)
        self.last_vel = (current - self.last_cmd) / self.dt
        self.last_cmd = current.copy()

    def filter(self, raw_cmd, *, profile: str = "normal") -> np.ndarray:
        """Return a safety-bounded 7-DOF joint command (rad) and update state.

        On any malformed input (wrong length, NaN, Inf) the previous safe
        command is returned unchanged so the robot sees a frozen target.
        """
        raw = np.asarray(raw_cmd, dtype=np.float64).reshape(-1)
        if raw.shape[0] != self.N_JOINTS or not np.all(np.isfinite(raw)):
            self._n_invalid += 1
            return self.last_cmd.copy()

        # (2) Clip raw command to position limits.
        raw_clipped = np.clip(
            raw, self.POS_LIMIT_LOW_RAD, self.POS_LIMIT_HIGH_RAD)
        if np.any(raw != raw_clipped):
            self._n_pos_clip += 1

        vel_limit_rad_s, acc_limit_rad_s2 = self.profile_limits(profile)

        # (3) Desired per-tick velocity to track the target.
        desired_vel = (raw_clipped - self.last_cmd) / self.dt

        # (4) Velocity limit.
        vel_clamped = np.clip(
            desired_vel, -vel_limit_rad_s, vel_limit_rad_s)
        if np.any(vel_clamped != desired_vel):
            self._n_vel_clip += 1
        desired_vel = vel_clamped

        # (5) Acceleration limit (relative to last commanded velocity).
        acc_lo = self.last_vel - acc_limit_rad_s2 * self.dt
        acc_hi = self.last_vel + acc_limit_rad_s2 * self.dt
        acc_clamped = np.clip(desired_vel, acc_lo, acc_hi)
        if np.any(acc_clamped != desired_vel):
            self._n_acc_clip += 1
        desired_vel = acc_clamped

        # (6) Integrate to candidate position.
        candidate = self.last_cmd + desired_vel * self.dt

        # (7) Final position-limit clip.
        candidate_clipped = np.clip(
            candidate, self.POS_LIMIT_LOW_RAD, self.POS_LIMIT_HIGH_RAD)
        if np.any(candidate != candidate_clipped):
            self._n_pos_clip += 1

        # (8) Recompute actual velocity from the clipped candidate.
        actual_vel = (candidate_clipped - self.last_cmd) / self.dt

        # (9) Update state.
        self.last_cmd = candidate_clipped.copy()
        self.last_vel = actual_vel.copy()

        # (10) Return the safe command.
        return candidate_clipped.copy()

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
