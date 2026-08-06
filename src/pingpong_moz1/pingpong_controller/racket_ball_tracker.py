"""Cartesian racket-centre tracking for the phase-1 hand-held-ball test.

The tracker is deliberately policy-independent.  It consumes a ball state in
the deployment control frame and the MuJoCo FK/Jacobian already owned by the
policy controller, then produces a rate-limited seven-joint position command.
Both solvers use the same target and safety limits:

``target = ball_position - [0, 0, height_offset]``

The non-MPC solver is a damped-least-squares Cartesian velocity servo.  The
MPC solver is a receding-horizon, locally linear controller that optimises one
constant joint velocity over a configurable horizon against the predicted
ball target trajectory.  Only the first control interval is applied before
the problem is rebuilt from fresh feedback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


KinematicsFn = Callable[
    [np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool],
]


@dataclass(frozen=True)
class RacketMpcTuning:
    """Runtime MPC tuning shared by tracking and ball-juggling control.

    ``velocity_weight`` penalises Cartesian target-velocity error, while
    ``effort_weight`` penalises joint velocity.  The latter distinction keeps
    the legacy ``RacketBallTrackerConfig.mpc_velocity_weight`` behaviour
    unambiguous when it is converted to this runtime representation.
    """

    horizon_steps: int = 12
    position_weight: float = 1.0
    velocity_weight: float = 1.0
    effort_weight: float = 1.0e-4
    smooth_weight: float = 2.0e-4
    posture_weight: float = 1.0e-4
    orientation_weight: float = 0.0
    orientation_gain_s: float = 0.0
    max_cartesian_speed_m_s: float = 0.20
    max_cartesian_acceleration_m_s2: float = 2.0
    joint_velocity_scale: float = 0.15
    joint_acceleration_scale: float = 0.10

    def __post_init__(self) -> None:
        if (isinstance(self.horizon_steps, (bool, np.bool_))
                or not isinstance(self.horizon_steps, (int, np.integer))
                or not 1 <= int(self.horizon_steps) <= 128):
            raise ValueError("horizon_steps must be an integer in [1, 128]")
        for name, value in {
            "position_weight": self.position_weight,
            "max_cartesian_speed_m_s": self.max_cartesian_speed_m_s,
            "max_cartesian_acceleration_m_s2": (
                self.max_cartesian_acceleration_m_s2),
        }.items():
            numeric = self._finite_number(name, value)
            if numeric <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        for name, value in {
            "velocity_weight": self.velocity_weight,
            "effort_weight": self.effort_weight,
            "smooth_weight": self.smooth_weight,
            "posture_weight": self.posture_weight,
            "orientation_weight": self.orientation_weight,
            "orientation_gain_s": self.orientation_gain_s,
        }.items():
            numeric = self._finite_number(name, value)
            if numeric < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        joint_velocity_scale = self._finite_number(
            "joint_velocity_scale", self.joint_velocity_scale)
        if not 0.0 <= joint_velocity_scale <= 1.0:
            raise ValueError("joint_velocity_scale must be in [0, 1]")
        joint_acceleration_scale = self._finite_number(
            "joint_acceleration_scale", self.joint_acceleration_scale)
        if not 0.0 < joint_acceleration_scale <= 1.0:
            raise ValueError("joint_acceleration_scale must be in (0, 1]")

    @staticmethod
    def _finite_number(name: str, value: object) -> float:
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(
                    value, (int, float, np.integer, np.floating))):
            raise ValueError(f"{name} must be a finite number")
        numeric = float(value)
        if not np.isfinite(numeric):
            raise ValueError(f"{name} must be a finite number")
        return numeric

    @classmethod
    def from_config(
        cls,
        config: "RacketBallTrackerConfig",
    ) -> "RacketMpcTuning":
        """Build the default per-tick tuning from a legacy tracker config."""

        if not isinstance(config, RacketBallTrackerConfig):
            raise TypeError("config must be a RacketBallTrackerConfig")
        return cls(
            horizon_steps=int(config.mpc_horizon_steps),
            position_weight=float(config.mpc_position_weight),
            velocity_weight=float(config.mpc_target_velocity_weight),
            # The old name meant a joint-velocity/effort regulariser.
            effort_weight=float(config.mpc_velocity_weight),
            smooth_weight=float(config.mpc_smooth_weight),
            posture_weight=float(config.mpc_posture_weight),
            orientation_weight=float(config.mpc_orientation_weight),
            orientation_gain_s=float(config.orientation_gain_s),
            max_cartesian_speed_m_s=float(
                config.max_cartesian_speed_m_s),
            max_cartesian_acceleration_m_s2=float(
                config.max_cartesian_acceleration_m_s2),
            joint_velocity_scale=float(config.joint_velocity_scale),
            joint_acceleration_scale=float(config.joint_acceleration_scale),
        )

    @classmethod
    def from_tracker_config(
        cls,
        config: "RacketBallTrackerConfig",
    ) -> "RacketMpcTuning":
        """Explicitly named alias useful at tracker/juggler integration sites."""

        return cls.from_config(config)

    def diagnostics(self) -> dict[str, float | int]:
        """Return a JSON-safe snapshot for control-loop diagnostics."""

        return {
            "horizonSteps": int(self.horizon_steps),
            "positionWeight": float(self.position_weight),
            "velocityWeight": float(self.velocity_weight),
            "effortWeight": float(self.effort_weight),
            "smoothWeight": float(self.smooth_weight),
            "postureWeight": float(self.posture_weight),
            "orientationWeight": float(self.orientation_weight),
            "orientationGainS": float(self.orientation_gain_s),
            "maxCartesianSpeedMps": float(self.max_cartesian_speed_m_s),
            "maxCartesianAccelerationMps2": float(
                self.max_cartesian_acceleration_m_s2),
            "jointVelocityScale": float(self.joint_velocity_scale),
            "jointAccelerationScale": float(self.joint_acceleration_scale),
        }


@dataclass(frozen=True)
class RacketBallTrackerConfig:
    """Static tuning shared by the DLS and linear MPC tracking solvers."""

    dt: float = 0.005
    position_gain_s: float = 4.0
    dls_damping: float = 0.04
    max_cartesian_speed_m_s: float = 0.20
    max_cartesian_acceleration_m_s2: float = 2.0
    joint_velocity_scale: float = 0.15
    joint_acceleration_scale: float = 0.10
    posture_gain_s: float = 0.12
    orientation_gain_s: float = 0.0
    mpc_horizon_steps: int = 12
    mpc_position_weight: float = 1.0
    mpc_target_velocity_weight: float = 1.0
    mpc_velocity_weight: float = 1.0e-4
    mpc_smooth_weight: float = 2.0e-4
    mpc_posture_weight: float = 1.0e-4
    mpc_orientation_weight: float = 0.0
    target_boundary_margin_m: float = 0.005
    max_command_feedback_error_rad: float = 0.20

    def __post_init__(self) -> None:
        positive = {
            "dt": self.dt,
            "position_gain_s": self.position_gain_s,
            "dls_damping": self.dls_damping,
            "max_cartesian_speed_m_s": self.max_cartesian_speed_m_s,
            "max_cartesian_acceleration_m_s2": (
                self.max_cartesian_acceleration_m_s2),
            "mpc_position_weight": self.mpc_position_weight,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        for name, value in {
            "posture_gain_s": self.posture_gain_s,
            "orientation_gain_s": self.orientation_gain_s,
            "mpc_target_velocity_weight": self.mpc_target_velocity_weight,
            "mpc_velocity_weight": self.mpc_velocity_weight,
            "mpc_smooth_weight": self.mpc_smooth_weight,
            "mpc_posture_weight": self.mpc_posture_weight,
            "mpc_orientation_weight": self.mpc_orientation_weight,
            "target_boundary_margin_m": self.target_boundary_margin_m,
            "max_command_feedback_error_rad": (
                self.max_command_feedback_error_rad),
        }.items():
            if not np.isfinite(value) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and >= 0")
        if not 1 <= int(self.mpc_horizon_steps) <= 128:
            raise ValueError("mpc_horizon_steps must be in [1, 128]")
        if (not np.isfinite(self.joint_velocity_scale)
                or not 0.0 <= float(self.joint_velocity_scale) <= 1.0):
            raise ValueError("joint_velocity_scale must be in [0, 1]")
        if (not np.isfinite(self.joint_acceleration_scale)
                or not 0.0 < float(self.joint_acceleration_scale) <= 1.0):
            raise ValueError("joint_acceleration_scale must be in (0, 1]")
        if float(self.max_command_feedback_error_rad) <= 0.0:
            raise ValueError("max_command_feedback_error_rad must be > 0")

    def mpc_tuning(self) -> RacketMpcTuning:
        """Return the immutable default runtime tuning for this config."""

        return RacketMpcTuning.from_config(self)


def racket_target_below_ball(
    ball_position_m: np.ndarray,
    height_offset_m: float,
) -> np.ndarray:
    """Return the racket-face centre target directly below the ball centre."""

    ball = np.asarray(ball_position_m, dtype=np.float64).reshape(3)
    offset = float(height_offset_m)
    if not np.all(np.isfinite(ball)):
        raise ValueError("ball_position_m must contain three finite values")
    if not np.isfinite(offset) or offset < 0.0:
        raise ValueError("height_offset_m must be finite and >= 0")
    target = ball.copy()
    target[2] -= offset
    return target


def _normalize_quaternion_xyzw(
    quaternion_xyzw: np.ndarray,
    name: str,
) -> np.ndarray:
    quat = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if not np.all(np.isfinite(quat)) or norm <= 1.0e-12:
        raise ValueError(f"{name} must be a finite non-zero quaternion")
    return quat / norm


def _rotation_matrix_from_quaternion_xyzw(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = _normalize_quaternion_xyzw(
        quaternion_xyzw, "quaternion_xyzw")
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def _rotation_vector_between_quaternions_xyzw(
    current_quaternion_xyzw: np.ndarray,
    reference_quaternion_xyzw: np.ndarray,
) -> np.ndarray:
    """Return the world-frame rotation vector from current to reference."""

    current_R = _rotation_matrix_from_quaternion_xyzw(
        current_quaternion_xyzw)
    reference_R = _rotation_matrix_from_quaternion_xyzw(
        reference_quaternion_xyzw)
    delta_R = reference_R @ current_R.T
    cos_angle = 0.5 * (float(np.trace(delta_R)) - 1.0)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    skew_vector = np.array([
        delta_R[2, 1] - delta_R[1, 2],
        delta_R[0, 2] - delta_R[2, 0],
        delta_R[1, 0] - delta_R[0, 1],
    ], dtype=np.float64)
    sin_angle = float(np.sin(angle))
    if angle < 1.0e-6:
        return 0.5 * skew_vector
    if abs(sin_angle) > 1.0e-6:
        return (angle / (2.0 * sin_angle)) * skew_vector

    # Near pi radians, extract the axis from the diagonal terms.  This branch
    # is rarely hit during juggling, but it keeps diagnostics finite if the arm
    # is started far from the nominal racket attitude.
    axis = np.empty(3, dtype=np.float64)
    axis[0] = np.sqrt(max(0.0, (delta_R[0, 0] + 1.0) * 0.5))
    axis[1] = np.sqrt(max(0.0, (delta_R[1, 1] + 1.0) * 0.5))
    axis[2] = np.sqrt(max(0.0, (delta_R[2, 2] + 1.0) * 0.5))
    if delta_R[2, 1] - delta_R[1, 2] < 0.0:
        axis[0] = -axis[0]
    if delta_R[0, 2] - delta_R[2, 0] < 0.0:
        axis[1] = -axis[1]
    if delta_R[1, 0] - delta_R[0, 1] < 0.0:
        axis[2] = -axis[2]
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1.0e-12:
        return np.zeros(3, dtype=np.float64)
    return angle * axis / axis_norm


class RacketBallTracker:
    """Rate-limited Cartesian tracker with DLS and linear-MPC strategies."""

    def __init__(
        self,
        *,
        kinematics: KinematicsFn,
        nominal_q_rad: np.ndarray,
        joint_lower_rad: np.ndarray,
        joint_upper_rad: np.ndarray,
        joint_velocity_limit_rad_s: np.ndarray,
        joint_acceleration_limit_rad_s2: np.ndarray,
        config: RacketBallTrackerConfig,
    ) -> None:
        self.kinematics = kinematics
        self.config = config
        self.default_mpc_tuning = config.mpc_tuning()
        self.nominal_q = self._vector(nominal_q_rad, "nominal_q_rad")
        self.joint_lower = self._vector(joint_lower_rad, "joint_lower_rad")
        self.joint_upper = self._vector(joint_upper_rad, "joint_upper_rad")
        self.joint_velocity_limit = self._vector(
            joint_velocity_limit_rad_s, "joint_velocity_limit_rad_s")
        self.joint_acceleration_limit = self._vector(
            joint_acceleration_limit_rad_s2,
            "joint_acceleration_limit_rad_s2",
        )
        self.n_joints = int(self.nominal_q.size)
        for name, value in (
            ("joint_lower_rad", self.joint_lower),
            ("joint_upper_rad", self.joint_upper),
            ("joint_velocity_limit_rad_s", self.joint_velocity_limit),
            ("joint_acceleration_limit_rad_s2", self.joint_acceleration_limit),
        ):
            if value.size != self.n_joints:
                raise ValueError(
                    f"{name} has {value.size} values; expected {self.n_joints}")
        if np.any(self.joint_lower >= self.joint_upper):
            raise ValueError("joint limits require lower < upper")
        if np.any(self.joint_velocity_limit <= 0.0):
            raise ValueError("joint velocity limits must be > 0")
        if np.any(self.joint_acceleration_limit <= 0.0):
            raise ValueError("joint acceleration limits must be > 0")
        self.command_q = np.clip(
            self.nominal_q, self.joint_lower, self.joint_upper)
        self.command_dq = np.zeros(self.n_joints, dtype=np.float64)
        self.reference_quat_xyzw = self._reference_quaternion_from_nominal()
        self.active = False
        self.last_diagnostics = self._idle_diagnostics("not_started")

    @staticmethod
    def _vector(value: np.ndarray, name: str) -> np.ndarray:
        out = np.asarray(value, dtype=np.float64).reshape(-1)
        if out.size == 0 or not np.all(np.isfinite(out)):
            raise ValueError(f"{name} must be a non-empty finite vector")
        return out.copy()

    def _reference_quaternion_from_nominal(self) -> np.ndarray | None:
        try:
            pose = self.kinematics(self.nominal_q)
        except Exception:
            return None
        if len(pose) != 5 or not bool(pose[4]):
            return None
        try:
            return _normalize_quaternion_xyzw(
                np.asarray(pose[1], dtype=np.float64),
                "nominal racket quaternion",
            )
        except ValueError:
            return None

    def reset(
        self,
        q_rad: np.ndarray,
        dq_rad_s: np.ndarray | None = None,
    ) -> None:
        """Seed the command trajectory from current feedback on mode entry."""

        q = self._vector(q_rad, "q_rad")
        if q.size != self.n_joints:
            raise ValueError(f"q_rad has {q.size} values; expected {self.n_joints}")
        if dq_rad_s is None:
            dq = np.zeros(self.n_joints, dtype=np.float64)
        else:
            dq = self._vector(dq_rad_s, "dq_rad_s")
            if dq.size != self.n_joints:
                raise ValueError(
                    f"dq_rad_s has {dq.size} values; expected {self.n_joints}")
        velocity_limit = (
            self.joint_velocity_limit * self.config.joint_velocity_scale)
        self.command_q = np.clip(q, self.joint_lower, self.joint_upper)
        self.command_dq = np.clip(dq, -velocity_limit, velocity_limit)
        self.active = True
        self.last_diagnostics = self._idle_diagnostics("reset")

    def deactivate(self, reason: str = "inactive") -> None:
        self.active = False
        self.command_dq = np.zeros(self.n_joints, dtype=np.float64)
        self.last_diagnostics = self._idle_diagnostics(reason)

    def observe_nominal_applied(
        self,
        q_rad: np.ndarray,
        dq_rad_s: np.ndarray | None = None,
    ) -> None:
        """Back-calculate a downstream nominal-layer safety correction.

        The caller must pass the corrected *nominal* command, not an actuator-
        compensated or transport-layer reference.  This keeps a workspace
        guard from rejecting an outward increment while the tracker silently
        integrates the same rejected motion.
        """

        q = self._vector(q_rad, "q_rad")
        if q.size != self.n_joints:
            raise ValueError(f"q_rad has {q.size} values; expected {self.n_joints}")
        previous_q = self.command_q.copy()
        self.command_q = np.clip(q, self.joint_lower, self.joint_upper)
        if dq_rad_s is None:
            dq = (self.command_q - previous_q) / self.config.dt
        else:
            dq = self._vector(dq_rad_s, "dq_rad_s")
            if dq.size != self.n_joints:
                raise ValueError(
                    f"dq_rad_s has {dq.size} values; expected {self.n_joints}")
        velocity_limit = (
            self.joint_velocity_limit * self.config.joint_velocity_scale)
        self.command_dq = np.clip(dq, -velocity_limit, velocity_limit)
        if isinstance(self.last_diagnostics, dict):
            self.last_diagnostics["antiWindupApplied"] = True
            self.last_diagnostics["commandPosition"] = self.command_q.tolist()
            self.last_diagnostics["commandVelocity"] = self.command_dq.tolist()

    def hold(
        self,
        q_rad: np.ndarray | None = None,
        reason: str = "hold",
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Stop trajectory integration while retaining a finite position hold."""

        if q_rad is not None:
            q = self._vector(q_rad, "q_rad")
            if q.size != self.n_joints:
                raise ValueError(
                    f"q_rad has {q.size} values; expected {self.n_joints}")
            self.command_q = np.clip(q, self.joint_lower, self.joint_upper)
        self.command_dq = np.zeros(self.n_joints, dtype=np.float64)
        self.active = False
        self.last_diagnostics = self._idle_diagnostics(reason)
        self.last_diagnostics["commandPosition"] = self.command_q.tolist()
        return self.command_q.astype(np.float32), dict(self.last_diagnostics)

    def move_to_joint_position(
        self,
        *,
        feedback_q_rad: np.ndarray,
        feedback_dq_rad_s: np.ndarray | None,
        target_q_rad: np.ndarray,
        mpc_tuning: RacketMpcTuning | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Rate-limit a joint-space move using the tracker's command state.

        This is intended for a safe pre-juggle positioning phase.  It shares
        the same joint velocity, acceleration and hard-position constraints as
        Cartesian tracking, so switching from this move into tracking does not
        introduce a second command trajectory or a velocity discontinuity.
        """

        tuning = self._resolve_mpc_tuning(mpc_tuning)
        feedback_q = self._vector(feedback_q_rad, "feedback_q_rad")
        target_q = self._vector(target_q_rad, "target_q_rad")
        if feedback_q.size != self.n_joints:
            raise ValueError(
                f"feedback_q_rad has {feedback_q.size} values; "
                f"expected {self.n_joints}")
        if target_q.size != self.n_joints:
            raise ValueError(
                f"target_q_rad has {target_q.size} values; "
                f"expected {self.n_joints}")
        if feedback_dq_rad_s is None:
            feedback_dq = np.zeros(self.n_joints, dtype=np.float64)
        else:
            feedback_dq = self._vector(
                feedback_dq_rad_s, "feedback_dq_rad_s")
            if feedback_dq.size != self.n_joints:
                raise ValueError(
                    f"feedback_dq_rad_s has {feedback_dq.size} values; "
                    f"expected {self.n_joints}")

        if not self.active:
            self.reset(feedback_q, feedback_dq)
        command_feedback_error_abs_max = float(np.max(np.abs(
            self.command_q - feedback_q)))
        command_reanchored = bool(
            command_feedback_error_abs_max
            > float(self.config.max_command_feedback_error_rad))
        velocity_limit = (
            self.joint_velocity_limit * tuning.joint_velocity_scale)
        if command_reanchored:
            self.command_q = np.clip(
                feedback_q, self.joint_lower, self.joint_upper)
            self.command_dq = np.clip(
                feedback_dq, -velocity_limit, velocity_limit)

        previous_dq = np.clip(
            self.command_dq, -velocity_limit, velocity_limit)
        previous_joint_velocity_reanchored = bool(np.any(
            np.abs(previous_dq - self.command_dq) > 1e-12))
        self.command_dq = previous_dq

        safe_target = np.clip(target_q, self.joint_lower, self.joint_upper)
        target_clipped = bool(np.any(np.abs(safe_target - target_q) > 1e-12))
        desired_dq = (safe_target - self.command_q) / self.config.dt
        limited_dq, velocity_limited, acceleration_limited = (
            self._limit_joint_velocity(desired_dq, tuning))
        requested_q = self.command_q + limited_dq * self.config.dt
        next_q = np.clip(requested_q, self.joint_lower, self.joint_upper)
        position_limited = bool(np.any(
            np.abs(next_q - requested_q) > 1e-12))
        limited_dq = (next_q - self.command_q) / self.config.dt

        acceleration_delta = (
            self.joint_acceleration_limit
            * tuning.joint_acceleration_scale * self.config.dt)
        acceleration_constraint_overridden = bool(np.any(
            np.abs(limited_dq - self.command_dq)
            > acceleration_delta + 1e-12))
        self.command_q = next_q
        self.command_dq = limited_dq
        self.active = True
        feedback_error = safe_target - feedback_q
        command_remaining_error = safe_target - self.command_q
        diagnostics: dict[str, Any] = {
            "active": True,
            "status": "moving_to_joint_position",
            "solver": "joint_position_ramp",
            "mpcRequested": False,
            "mpcActive": False,
            "mpcTuning": tuning.diagnostics(),
            "targetPosition": safe_target.tolist(),
            "targetClipped": target_clipped,
            "jointError": feedback_error.tolist(),
            "jointErrorAbsMaxRad": float(np.max(np.abs(feedback_error))),
            "commandRemainingError": command_remaining_error.tolist(),
            "commandRemainingErrorAbsMaxRad": float(np.max(
                np.abs(command_remaining_error))),
            "settled": bool(
                np.max(np.abs(feedback_error)) <= 0.02
                and np.max(np.abs(feedback_dq)) <= 0.02),
            "commandPosition": self.command_q.tolist(),
            "commandVelocity": self.command_dq.tolist(),
            "commandFeedbackErrorAbsMaxRad": (
                command_feedback_error_abs_max),
            "commandReanchored": command_reanchored,
            "previousJointVelocityReanchored": bool(
                previous_joint_velocity_reanchored),
            "velocityLimited": bool(velocity_limited),
            "accelerationLimited": bool(acceleration_limited),
            "accelerationConstraintOverridden": bool(
                acceleration_constraint_overridden),
            "positionLimited": bool(position_limited),
            "saturated": bool(
                target_clipped or previous_joint_velocity_reanchored
                or velocity_limited or acceleration_limited
                or acceleration_constraint_overridden or position_limited),
        }
        self.last_diagnostics = diagnostics
        return self.command_q.astype(np.float32), dict(diagnostics)

    def step(
        self,
        *,
        feedback_q_rad: np.ndarray,
        feedback_dq_rad_s: np.ndarray | None,
        ball_position_m: np.ndarray,
        ball_velocity_m_s: np.ndarray,
        height_offset_m: float,
        use_mpc: bool,
        mpc_tuning: RacketMpcTuning | None = None,
        position_limits_m: tuple[
            tuple[float, float] | None,
            tuple[float, float] | None,
            tuple[float, float] | None,
        ] = (None, None, None),
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Advance one control tick and return ``(joint_position, diagnostics)``."""

        tick_tuning = self._resolve_mpc_tuning(mpc_tuning)

        q_feedback = self._vector(feedback_q_rad, "feedback_q_rad")
        if q_feedback.size != self.n_joints:
            raise ValueError(
                f"feedback_q_rad has {q_feedback.size} values; "
                f"expected {self.n_joints}")
        if feedback_dq_rad_s is None:
            dq_feedback = np.zeros(self.n_joints, dtype=np.float64)
        else:
            dq_feedback = self._vector(
                feedback_dq_rad_s, "feedback_dq_rad_s")
            if dq_feedback.size != self.n_joints:
                raise ValueError(
                    f"feedback_dq_rad_s has {dq_feedback.size} values; "
                    f"expected {self.n_joints}")
        ball_position = np.asarray(
            ball_position_m, dtype=np.float64).reshape(3)
        ball_velocity = np.asarray(
            ball_velocity_m_s, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(ball_velocity)):
            raise ValueError("ball_velocity_m_s must contain three finite values")
        raw_target = racket_target_below_ball(
            ball_position, height_offset_m)
        target, target_clipped, effective_velocity = self._bounded_target(
            raw_target, ball_velocity, position_limits_m, tick_tuning)

        if not self.active:
            self.reset(q_feedback, dq_feedback)

        command_feedback_error = self.command_q - q_feedback
        command_feedback_error_abs_max = float(np.max(np.abs(
            command_feedback_error)))
        command_reanchored = bool(
            command_feedback_error_abs_max
            > float(self.config.max_command_feedback_error_rad))
        if command_reanchored:
            velocity_limit = (
                self.joint_velocity_limit
                * tick_tuning.joint_velocity_scale)
            self.command_q = np.clip(
                q_feedback, self.joint_lower, self.joint_upper)
            self.command_dq = np.clip(
                dq_feedback, -velocity_limit, velocity_limit)

        tick_velocity_limit = (
            self.joint_velocity_limit * tick_tuning.joint_velocity_scale)
        safe_previous_command_dq = np.clip(
            self.command_dq, -tick_velocity_limit, tick_velocity_limit)
        previous_joint_velocity_reanchored = bool(np.any(
            np.abs(safe_previous_command_dq - self.command_dq) > 1e-12))
        if previous_joint_velocity_reanchored:
            self.command_dq = safe_previous_command_dq

        pose = self.kinematics(q_feedback)
        if len(pose) != 5:
            raise ValueError("kinematics must return pose, quat, Jpos, Jrot, ok")
        racket_position, racket_quaternion, jacobian_position, jacobian_rotation, ok = pose
        racket_position = np.asarray(
            racket_position, dtype=np.float64).reshape(3)
        racket_quaternion = _normalize_quaternion_xyzw(
            racket_quaternion, "racket quaternion")
        jacobian = np.asarray(jacobian_position, dtype=np.float64)
        jacobian_rot = np.asarray(jacobian_rotation, dtype=np.float64)
        if (not bool(ok) or jacobian.shape != (3, self.n_joints)
                or jacobian_rot.shape != (3, self.n_joints)
                or not np.all(np.isfinite(racket_position))
                or not np.all(np.isfinite(jacobian))
                or not np.all(np.isfinite(jacobian_rot))):
            raise RuntimeError("racket FK/Jacobian is unavailable")

        orientation_error = np.zeros(3, dtype=np.float64)
        if self.reference_quat_xyzw is not None:
            orientation_error = _rotation_vector_between_quaternions_xyzw(
                racket_quaternion, self.reference_quat_xyzw)
        orientation_control_active = bool(
            self.reference_quat_xyzw is not None
            and float(tick_tuning.orientation_weight) > 0.0
            and float(tick_tuning.orientation_gain_s) > 0.0)

        # Keep the previous point of the rate-limited trajectory Cartesian-
        # feasible.  This is normally already true, but feedback re-anchoring
        # or downstream anti-windup may have replaced command_dq.
        safe_previous_dq, previous_cartesian_reanchored = (
            self._limit_cartesian_speed(
                self.command_dq, jacobian, tick_tuning))
        if previous_cartesian_reanchored:
            self.command_dq = safe_previous_dq

        position_error = target - racket_position
        solver = "dls"
        fallback_reason = None
        if bool(use_mpc):
            try:
                desired_dq = self._linear_mpc_velocity(
                    q_feedback,
                    racket_position,
                    jacobian,
                    jacobian_rot,
                    orientation_error,
                    target,
                    effective_velocity,
                    position_limits_m,
                    tick_tuning,
                )
                solver = "linear_mpc"
            except (np.linalg.LinAlgError, ValueError) as exc:
                desired_dq = self._dls_velocity(
                    q_feedback,
                    jacobian,
                    jacobian_rot,
                    position_error,
                    effective_velocity,
                    orientation_error,
                    tick_tuning,
                )
                solver = "dls_fallback"
                fallback_reason = str(exc)
        else:
            desired_dq = self._dls_velocity(
                q_feedback,
                jacobian,
                jacobian_rot,
                position_error,
                effective_velocity,
                orientation_error,
                tick_tuning,
            )

        desired_dq, cartesian_speed_limited = self._limit_cartesian_speed(
            desired_dq, jacobian, tick_tuning)
        limited_dq, velocity_limited, acceleration_limited = (
            self._limit_joint_velocity(desired_dq, tick_tuning))
        limited_dq, cartesian_acceleration_limited = (
            self._limit_cartesian_acceleration_from_previous(
                limited_dq,
                jacobian,
                self.command_dq,
                tick_tuning,
            ))
        requested_next_q = self.command_q + limited_dq * self.config.dt
        position_safe_next_q = np.clip(
            requested_next_q, self.joint_lower, self.joint_upper)
        position_limited = bool(np.any(
            np.abs(position_safe_next_q - requested_next_q) > 1e-12))
        limited_dq = (
            (position_safe_next_q - self.command_q) / self.config.dt)

        # Non-uniform per-joint clipping can destroy a null-space cancellation:
        # a dq with J@dq == 0 may acquire a large task velocity after one joint
        # saturates.  Project from the previous feasible command so the final
        # result retains the joint acceleration/velocity box constraints.
        limited_dq, final_cartesian_speed_limited = (
            self._limit_cartesian_speed_from_previous(
                limited_dq, jacobian, self.command_dq, tick_tuning))

        # The previous command can be incompatible with a newly reached hard
        # joint boundary.  Position safety takes precedence; a final radial
        # Cartesian projection cannot reintroduce a joint-limit crossing.
        projected_next_q = self.command_q + limited_dq * self.config.dt
        final_position_q = np.clip(
            projected_next_q, self.joint_lower, self.joint_upper)
        if np.any(np.abs(final_position_q - projected_next_q) > 1e-12):
            position_limited = True
            limited_dq = (
                (final_position_q - self.command_q) / self.config.dt)
        limited_dq, cartesian_safety_override = self._limit_cartesian_speed(
            limited_dq, jacobian, tick_tuning)
        final_cartesian_speed_limited = bool(
            final_cartesian_speed_limited or cartesian_safety_override)
        next_q = self.command_q + limited_dq * self.config.dt
        next_q = np.clip(next_q, self.joint_lower, self.joint_upper)

        acceleration_delta = (
            self.joint_acceleration_limit
            * tick_tuning.joint_acceleration_scale
            * self.config.dt)
        acceleration_constraint_overridden = bool(np.any(
            np.abs(limited_dq - self.command_dq)
            > acceleration_delta + 1e-12))
        cartesian_acceleration_m_s2 = float(np.linalg.norm(
            jacobian @ (limited_dq - self.command_dq)
            / self.config.dt))
        cartesian_acceleration_constraint_overridden = bool(
            cartesian_acceleration_m_s2
            > tick_tuning.max_cartesian_acceleration_m_s2 + 1e-10)
        self.command_q = next_q
        self.command_dq = limited_dq
        self.active = True
        vertical_gap = float(ball_position[2] - racket_position[2])
        diagnostics: dict[str, Any] = {
            "active": True,
            "status": "tracking",
            "solver": solver,
            "mpcRequested": bool(use_mpc),
            "mpcActive": solver == "linear_mpc",
            "mpcTuning": tick_tuning.diagnostics(),
            "fallbackReason": fallback_reason,
            "heightOffsetM": float(height_offset_m),
            "rawTarget": raw_target.tolist(),
            "target": target.tolist(),
            "targetClipped": bool(target_clipped),
            "racket": racket_position.tolist(),
            "racketQuaternion": racket_quaternion.tolist(),
            "referenceQuaternion": (
                None
                if self.reference_quat_xyzw is None
                else self.reference_quat_xyzw.tolist()),
            "orientationError": orientation_error.tolist(),
            "orientationErrorRad": float(np.linalg.norm(orientation_error)),
            "orientationControlActive": orientation_control_active,
            "error": position_error.tolist(),
            "errorNormM": float(np.linalg.norm(position_error)),
            "horizontalErrorM": float(np.linalg.norm(position_error[:2])),
            "verticalGapM": vertical_gap,
            "heightErrorM": vertical_gap - float(height_offset_m),
            "commandPosition": self.command_q.tolist(),
            "commandVelocity": self.command_dq.tolist(),
            "commandFeedbackErrorAbsMaxRad": (
                command_feedback_error_abs_max),
            "commandReanchored": bool(command_reanchored),
            "cartesianSpeedMps": float(np.linalg.norm(
                jacobian @ self.command_dq)),
            "cartesianAccelerationMps2": cartesian_acceleration_m_s2,
            "cartesianAccelerationLimited": bool(
                cartesian_acceleration_limited),
            "cartesianAccelerationConstraintOverridden": bool(
                cartesian_acceleration_constraint_overridden),
            "cartesianSpeedLimited": bool(
                cartesian_speed_limited
                or final_cartesian_speed_limited
                or previous_cartesian_reanchored),
            "finalCartesianSpeedLimited": bool(
                final_cartesian_speed_limited),
            "previousCartesianVelocityReanchored": bool(
                previous_cartesian_reanchored),
            "previousJointVelocityReanchored": bool(
                previous_joint_velocity_reanchored),
            "antiWindupApplied": False,
            "velocityLimited": bool(velocity_limited),
            "accelerationLimited": bool(acceleration_limited),
            "accelerationConstraintOverridden": bool(
                acceleration_constraint_overridden),
            "positionLimited": bool(position_limited),
            "saturated": bool(
                target_clipped or cartesian_speed_limited
                or final_cartesian_speed_limited
                or previous_cartesian_reanchored
                or previous_joint_velocity_reanchored or velocity_limited
                or acceleration_limited or cartesian_acceleration_limited
                or acceleration_constraint_overridden
                or cartesian_acceleration_constraint_overridden
                or position_limited),
        }
        self.last_diagnostics = diagnostics
        return self.command_q.astype(np.float32), dict(diagnostics)

    def _bounded_target(
        self,
        raw_target: np.ndarray,
        ball_velocity: np.ndarray,
        limits: tuple[
            tuple[float, float] | None,
            tuple[float, float] | None,
            tuple[float, float] | None,
        ],
        tuning: RacketMpcTuning,
    ) -> tuple[np.ndarray, bool, np.ndarray]:
        target = np.asarray(raw_target, dtype=np.float64).copy()
        velocity = np.asarray(ball_velocity, dtype=np.float64).copy()
        clipped = False
        effective_limits = self.inset_position_limits(limits)
        for axis, axis_limits in enumerate(effective_limits):
            if axis_limits is None:
                continue
            inner_lo, inner_hi = axis_limits
            original = float(target[axis])
            target[axis] = np.clip(original, inner_lo, inner_hi)
            if abs(float(target[axis]) - original) > 1e-12:
                clipped = True
            # Project feed-forward motion so even a target exactly on (or one
            # tick before) a boundary cannot ask either solver to move farther
            # outward.  Tangential and inward components remain untouched.
            old_velocity = float(velocity[axis])
            min_velocity = (
                (inner_lo - float(target[axis])) / self.config.dt)
            max_velocity = (
                (inner_hi - float(target[axis])) / self.config.dt)
            velocity[axis] = min(
                max(old_velocity, min(min_velocity, max_velocity)),
                max(min_velocity, max_velocity),
            )
            if abs(float(velocity[axis]) - old_velocity) > 1e-12:
                clipped = True
        speed = float(np.linalg.norm(velocity))
        max_speed = float(tuning.max_cartesian_speed_m_s)
        if speed > max_speed and speed > 1e-12:
            velocity *= max_speed / speed
        return target, clipped, velocity

    def inset_position_limits(
        self,
        limits: tuple[
            tuple[float, float] | None,
            tuple[float, float] | None,
            tuple[float, float] | None,
        ],
    ) -> tuple[
        tuple[float, float] | None,
        tuple[float, float] | None,
        tuple[float, float] | None,
    ]:
        """Return the exact workspace box used for low-level targets."""

        if len(limits) != 3:
            raise ValueError("position_limits_m must contain x/y/z limits")
        margin = float(self.config.target_boundary_margin_m)
        effective: list[tuple[float, float] | None] = []
        for axis_limits in limits:
            if axis_limits is None:
                effective.append(None)
                continue
            lower, upper = float(axis_limits[0]), float(axis_limits[1])
            if (not np.isfinite(lower) or not np.isfinite(upper)
                    or lower >= upper):
                raise ValueError(
                    "position limits require finite lower < upper")
            inner_lower = lower + margin
            inner_upper = upper - margin
            if inner_lower > inner_upper:
                inner_lower = inner_upper = 0.5 * (lower + upper)
            effective.append((inner_lower, inner_upper))
        return tuple(effective)

    def _resolve_mpc_tuning(
        self,
        tuning: RacketMpcTuning | None,
    ) -> RacketMpcTuning:
        if tuning is None:
            return self.default_mpc_tuning
        if not isinstance(tuning, RacketMpcTuning):
            raise TypeError("mpc_tuning must be a RacketMpcTuning or None")
        return tuning

    def _dls_velocity(
        self,
        q_feedback: np.ndarray,
        jacobian: np.ndarray,
        jacobian_rotation: np.ndarray,
        position_error: np.ndarray,
        target_velocity: np.ndarray,
        orientation_error: np.ndarray,
        tuning: RacketMpcTuning,
    ) -> np.ndarray:
        desired_cartesian_velocity = (
            target_velocity
            + float(self.config.position_gain_s) * position_error)
        speed = float(np.linalg.norm(desired_cartesian_velocity))
        max_speed = float(self.config.max_cartesian_speed_m_s)
        if speed > max_speed and speed > 1e-12:
            desired_cartesian_velocity *= max_speed / speed
        task_jacobian = jacobian
        task_velocity = desired_cartesian_velocity
        orientation_weight = float(tuning.orientation_weight)
        orientation_gain_s = float(tuning.orientation_gain_s)
        if orientation_weight > 0.0 and orientation_gain_s > 0.0:
            orientation_scale = float(np.sqrt(orientation_weight))
            task_jacobian = np.vstack((
                task_jacobian,
                orientation_scale * jacobian_rotation,
            ))
            task_velocity = np.concatenate((
                task_velocity,
                orientation_scale * orientation_gain_s * orientation_error,
            ))
        damping_sq = float(self.config.dls_damping) ** 2
        task_rows = int(task_jacobian.shape[0])
        task_inverse = task_jacobian.T @ np.linalg.solve(
            task_jacobian @ task_jacobian.T + damping_sq * np.eye(task_rows),
            np.eye(task_rows),
        )
        solved_velocity = task_inverse @ task_velocity
        nullspace = np.eye(self.n_joints) - task_inverse @ task_jacobian
        posture_velocity = (
            float(self.config.posture_gain_s)
            * (self.nominal_q - q_feedback))
        return solved_velocity + nullspace @ posture_velocity

    def _linear_mpc_velocity(
        self,
        q_feedback: np.ndarray,
        racket_position: np.ndarray,
        jacobian: np.ndarray,
        jacobian_rotation: np.ndarray,
        orientation_error: np.ndarray,
        target: np.ndarray,
        target_velocity: np.ndarray,
        position_limits: tuple[
            tuple[float, float] | None,
            tuple[float, float] | None,
            tuple[float, float] | None,
        ],
        tuning: RacketMpcTuning,
    ) -> np.ndarray:
        horizon = int(tuning.horizon_steps)
        dt = float(self.config.dt)
        rows = []
        rhs = []
        for step in range(1, horizon + 1):
            time_s = float(step) * dt
            predicted_target = target + target_velocity * time_s
            predicted_target, _clipped, _velocity = self._bounded_target(
                predicted_target, target_velocity, position_limits, tuning)
            # Later target errors carry slightly more weight while retaining
            # enough near-term authority for a smooth first receding step.
            weight = np.sqrt(
                float(tuning.position_weight)
                * float(step) / float(horizon))
            rows.append(weight * time_s * jacobian)
            rhs.append(weight * (predicted_target - racket_position))
        A = np.vstack(rows)
        b = np.concatenate(rhs)
        velocity_weight = float(tuning.velocity_weight)
        effort_weight = float(tuning.effort_weight)
        smooth_weight = float(tuning.smooth_weight)
        posture_weight = float(tuning.posture_weight)
        orientation_weight = float(tuning.orientation_weight)
        orientation_gain_s = float(tuning.orientation_gain_s)
        terminal_time = float(horizon) * dt
        hessian = A.T @ A
        gradient = A.T @ b
        if velocity_weight > 0.0:
            # The position rollout alone anticipates where a moving target
            # will be.  This separate term also asks the local task velocity
            # to match that target: J*dq ~= target_velocity.
            velocity_A = jacobian
            velocity_b = target_velocity
            hessian += velocity_weight * (velocity_A.T @ velocity_A)
            gradient += velocity_weight * (velocity_A.T @ velocity_b)
        if orientation_weight > 0.0 and orientation_gain_s > 0.0:
            desired_omega = orientation_gain_s * orientation_error
            hessian += orientation_weight * (
                jacobian_rotation.T @ jacobian_rotation)
            gradient += orientation_weight * (
                jacobian_rotation.T @ desired_omega)
        if effort_weight > 0.0:
            hessian += effort_weight * np.eye(self.n_joints)
        if smooth_weight > 0.0:
            hessian += smooth_weight * np.eye(self.n_joints)
            gradient += smooth_weight * self.command_dq
        if posture_weight > 0.0 and terminal_time > 0.0:
            posture_A = terminal_time * np.eye(self.n_joints)
            posture_b = self.nominal_q - q_feedback
            hessian += posture_weight * (posture_A.T @ posture_A)
            gradient += posture_weight * (posture_A.T @ posture_b)
        # A tiny diagonal is numerical regularisation, not a tuning weight.
        hessian += 1.0e-10 * np.eye(self.n_joints)
        solution = np.linalg.solve(hessian, gradient)
        if not np.all(np.isfinite(solution)):
            raise ValueError("linear MPC produced non-finite joint velocity")
        return solution

    def _limit_cartesian_speed(
        self,
        desired_dq: np.ndarray,
        jacobian: np.ndarray,
        tuning: RacketMpcTuning | None = None,
    ) -> tuple[np.ndarray, bool]:
        desired = np.asarray(desired_dq, dtype=np.float64).reshape(
            self.n_joints)
        task_velocity = np.asarray(
            jacobian, dtype=np.float64).reshape(3, self.n_joints) @ desired
        speed = float(np.linalg.norm(task_velocity))
        active_tuning = tuning or self.default_mpc_tuning
        limit = float(active_tuning.max_cartesian_speed_m_s)
        if speed <= limit or speed <= 1e-12:
            return desired, False
        return desired * (limit / speed), True

    def _limit_cartesian_speed_from_previous(
        self,
        candidate_dq: np.ndarray,
        jacobian: np.ndarray,
        previous_dq: np.ndarray,
        tuning: RacketMpcTuning | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Keep task speed feasible without breaking the joint rate box.

        Both endpoints are expected to obey the joint velocity/acceleration
        constraints.  Their line segment therefore obeys the same convex box.
        The Cartesian-speed ball is convex too, so bisection finds the farthest
        feasible point toward the new candidate.
        """

        candidate = np.asarray(candidate_dq, dtype=np.float64).reshape(
            self.n_joints)
        previous = np.asarray(previous_dq, dtype=np.float64).reshape(
            self.n_joints)
        J = np.asarray(jacobian, dtype=np.float64).reshape(
            3, self.n_joints)
        active_tuning = tuning or self.default_mpc_tuning
        limit = float(active_tuning.max_cartesian_speed_m_s)
        if float(np.linalg.norm(J @ candidate)) <= limit + 1e-12:
            return candidate, False

        previous_speed = float(np.linalg.norm(J @ previous))
        if previous_speed > limit + 1e-12:
            previous = previous * (limit / previous_speed)

        low = 0.0
        high = 1.0
        delta = candidate - previous
        for _ in range(48):
            mid = 0.5 * (low + high)
            trial = previous + mid * delta
            if float(np.linalg.norm(J @ trial)) <= limit:
                low = mid
            else:
                high = mid
        return previous + low * delta, True

    def _limit_cartesian_acceleration_from_previous(
        self,
        candidate_dq: np.ndarray,
        jacobian: np.ndarray,
        previous_dq: np.ndarray,
        tuning: RacketMpcTuning | None = None,
    ) -> tuple[np.ndarray, bool]:
        """Limit the local task-velocity change over one control interval.

        Scaling on the segment from the previous feasible joint velocity also
        preserves the per-joint velocity/acceleration boxes and the Cartesian
        speed ball.  A later hard joint-position clip can override this soft
        acceleration constraint; that conflict is exposed in diagnostics.
        """

        candidate = np.asarray(candidate_dq, dtype=np.float64).reshape(
            self.n_joints)
        previous = np.asarray(previous_dq, dtype=np.float64).reshape(
            self.n_joints)
        J = np.asarray(jacobian, dtype=np.float64).reshape(
            3, self.n_joints)
        active_tuning = tuning or self.default_mpc_tuning
        task_velocity_delta = J @ (candidate - previous)
        maximum_delta = (
            float(active_tuning.max_cartesian_acceleration_m_s2)
            * self.config.dt)
        delta_norm = float(np.linalg.norm(task_velocity_delta))
        if delta_norm <= maximum_delta + 1e-12 or delta_norm <= 1e-15:
            return candidate, False
        fraction = maximum_delta / delta_norm
        return previous + fraction * (candidate - previous), True

    def _limit_joint_velocity(
        self,
        desired_dq: np.ndarray,
        tuning: RacketMpcTuning | None = None,
    ) -> tuple[np.ndarray, bool, bool]:
        desired = np.asarray(desired_dq, dtype=np.float64).reshape(
            self.n_joints)
        if not np.all(np.isfinite(desired)):
            raise ValueError("solver produced non-finite joint velocity")
        active_tuning = tuning or self.default_mpc_tuning
        velocity_limit = (
            self.joint_velocity_limit * active_tuning.joint_velocity_scale)
        velocity_clipped = np.clip(desired, -velocity_limit, velocity_limit)
        velocity_limited = bool(np.any(np.abs(velocity_clipped - desired) > 1e-12))
        acceleration_delta = (
            self.joint_acceleration_limit
            * active_tuning.joint_acceleration_scale
            * self.config.dt)
        lower = self.command_dq - acceleration_delta
        upper = self.command_dq + acceleration_delta
        acceleration_clipped = np.clip(velocity_clipped, lower, upper)
        acceleration_limited = bool(np.any(
            np.abs(acceleration_clipped - velocity_clipped) > 1e-12))
        return acceleration_clipped, velocity_limited, acceleration_limited

    def _idle_diagnostics(self, reason: str) -> dict[str, Any]:
        return {
            "active": False,
            "status": str(reason),
            "solver": "none",
            "mpcRequested": False,
            "mpcActive": False,
            "mpcTuning": self.default_mpc_tuning.diagnostics(),
            "fallbackReason": None,
            "heightOffsetM": None,
            "rawTarget": None,
            "target": None,
            "targetClipped": False,
            "racket": None,
            "racketQuaternion": None,
            "referenceQuaternion": (
                None
                if self.reference_quat_xyzw is None
                else self.reference_quat_xyzw.tolist()),
            "orientationError": None,
            "orientationErrorRad": None,
            "orientationControlActive": False,
            "error": None,
            "errorNormM": None,
            "horizontalErrorM": None,
            "verticalGapM": None,
            "heightErrorM": None,
            "commandPosition": self.command_q.tolist(),
            "commandVelocity": self.command_dq.tolist(),
            "commandFeedbackErrorAbsMaxRad": None,
            "commandReanchored": False,
            "cartesianSpeedMps": 0.0,
            "cartesianSpeedLimited": False,
            "finalCartesianSpeedLimited": False,
            "previousCartesianVelocityReanchored": False,
            "previousJointVelocityReanchored": False,
            "antiWindupApplied": False,
            "velocityLimited": False,
            "accelerationLimited": False,
            "accelerationConstraintOverridden": False,
            "positionLimited": False,
            "saturated": False,
        }
