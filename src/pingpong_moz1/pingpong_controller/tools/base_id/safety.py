#!/usr/bin/env python3
"""Fail-closed safety gates, real-time geofence, staleness watchdog, and braking bursts.

Pure Python + NumPy (zero ROS runtime dependency) so every safety invariant can be
exhaustively verified by offline pytest unit tests before touching hardware.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from .waveforms import TierLimits
except ImportError:
    from waveforms import TierLimits  # type: ignore[no-redef]

LIVE_CONFIRM_ENV_VAR = "CONFIRM_MOZ1_BASE_ID_LIVE"
LIVE_CONFIRM_ENV_VALUE = "YES"


class LiveGateError(RuntimeError):
    """Raised when any mandatory live-execution gate fails."""


class SafetyTripError(RuntimeError):
    """Raised when a real-time runtime safety monitor trips."""

    def __init__(self, reason_code: str, detail: str) -> None:
        super().__init__(f"[{reason_code}] {detail}")
        self.reason_code = reason_code
        self.detail = detail


class OutputExistsError(FileExistsError):
    """Raised when an experiment or analysis output directory already exists."""


@dataclass
class FeedbackSnapshot:
    """Latest timestamped telemetry sample from one sensor/topic stream."""

    wall_time_s: float
    header_time_s: Optional[float]
    values: np.ndarray
    pose_xyyaw: Optional[Tuple[float, float, float]] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RobotStatusSnapshot:
    """Latest parsed ``/robot_states`` safety/motor/mode status."""

    wall_time_s: float
    safety_status: int
    motor_status: Tuple[int, ...]
    control_mode: Tuple[int, ...]
    task_mode: int
    motion_status: int


@dataclass
class SafetyTripRecord:
    """Audit log entry emitted whenever a safety monitor triggers."""

    wall_time_s: float
    trial_id: str
    reason_code: str
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "wall_time_s": self.wall_time_s,
            "trial_id": self.trial_id,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }


def enforce_live_gates(
    *,
    live: bool,
    env: Mapping[str, str],
    stage: str,
    prompt_fn: Callable[[str], str] = input,
) -> str:
    """Enforce the 3-factor live gate (``--live`` + env var + interactive stage code).

    Returns ``"dry_run"`` if ``live`` is False, or ``"live"`` only when all three gates
    pass. Raises ``LiveGateError`` on any missing or mismatched gate.
    """
    if not live:
        return "dry_run"

    env_val = env.get(LIVE_CONFIRM_ENV_VAR, "")
    if env_val != LIVE_CONFIRM_ENV_VALUE:
        raise LiveGateError(
            f"Refusing --live execution: environment variable "
            f"{LIVE_CONFIRM_ENV_VAR} must be exactly '{LIVE_CONFIRM_ENV_VALUE}' "
            f"(currently {env_val!r})."
        )

    prompt = (
        f"[LIVE SAFETY GATE] Confirm physical E-Stop is in hand and area (>=3m x 3m) is clear.\n"
        f"Type exact stage code '{stage}' to arm live base commands: "
    )
    try:
        typed = prompt_fn(prompt).strip()
    except (EOFError, KeyboardInterrupt) as exc:
        raise LiveGateError(
            f"Refusing --live execution: interactive confirmation aborted ({exc})."
        ) from exc

    if typed != stage:
        raise LiveGateError(
            f"Refusing --live execution: expected stage confirmation {stage!r}, got {typed!r}."
        )
    return "live"


def create_unique_run_dir(
    out_root: Path | str,
    stage: str,
    *,
    explicit_dir: Optional[Path | str] = None,
    timestamp_str: Optional[str] = None,
) -> Path:
    """Create a fresh, never-overwritten output directory for one experiment run."""
    if explicit_dir is not None:
        target = Path(explicit_dir)
    else:
        ts = timestamp_str or datetime.now().strftime("%Y%m%d_%H%M%S")
        target = Path(out_root) / f"base_id_{ts}_{stage}"
    if target.exists():
        raise OutputExistsError(
            f"Refusing to overwrite existing experiment directory: {target}"
        )
    target.mkdir(parents=True, exist_ok=False)
    return target


class DeadReckoningOdom:
    """SE(2) dead-reckoning integrator from body-frame ``(vx, vy, wz)`` telemetry.

    Essential on MOZ1 when ``frame_type == 0`` (``REF_COORD_UNIT_BASE``), where
    ``/mx_robot_state`` reports ``base.cartesian.pose == Identity`` and only
    ``base.cartesian.twist`` carries live base velocity.
    """

    def __init__(self, max_dt_s: float = 0.25) -> None:
        self._max_dt_s = float(max_dt_s)
        self.x_m: float = 0.0
        self.y_m: float = 0.0
        self.yaw_rad: float = 0.0
        self._last_time_s: Optional[float] = None
        self._last_twist: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def reset(self, x_m: float = 0.0, y_m: float = 0.0, yaw_rad: float = 0.0) -> None:
        self.x_m = float(x_m)
        self.y_m = float(y_m)
        self.yaw_rad = float(yaw_rad)
        self._last_time_s = None
        self._last_twist = (0.0, 0.0, 0.0)

    def update(self, wall_time_s: float, vx_m_s: float, vy_m_s: float, wz_rad_s: float) -> Tuple[float, float, float]:
        if not (
            math.isfinite(wall_time_s)
            and math.isfinite(vx_m_s)
            and math.isfinite(vy_m_s)
            and math.isfinite(wz_rad_s)
        ):
            return self.pose()
        if self._last_time_s is not None:
            dt = wall_time_s - self._last_time_s
            if 0.0 < dt <= self._max_dt_s:
                vx_mid = 0.5 * (self._last_twist[0] + vx_m_s)
                vy_mid = 0.5 * (self._last_twist[1] + vy_m_s)
                wz_mid = 0.5 * (self._last_twist[2] + wz_rad_s)
                yaw_mid = self.yaw_rad + 0.5 * wz_mid * dt
                c = math.cos(yaw_mid)
                s = math.sin(yaw_mid)
                self.x_m += (vx_mid * c - vy_mid * s) * dt
                self.y_m += (vx_mid * s + vy_mid * c) * dt
                self.yaw_rad += wz_mid * dt
        self._last_time_s = wall_time_s
        self._last_twist = (vx_m_s, vy_m_s, wz_rad_s)
        return self.pose()

    def pose(self) -> Tuple[float, float, float]:
        return (self.x_m, self.y_m, self.yaw_rad)


def _angle_diff_rad(a: float, b: float) -> float:
    return math.atan2(math.sin(a - b), math.cos(a - b))


class RealTimeSafetySupervisor:
    """Real-time safety fence, feedback health watchdog, and tier command limiter."""

    def __init__(
        self,
        *,
        geofence_radius_m: float,
        geofence_yaw_rad: float,
        feedback_stale_timeout_s: float,
        tier_limits: Mapping[str, TierLimits],
        require_robot_status_normal: bool = True,
        require_base_motor_power_on: bool = True,
        require_target_tracking_mode: bool = True,
        require_legwaist_impedance_mode: bool = False,
        apply_command_slew_clamp: bool = False,
        base_mu_idx: int = 0,
        legwaist_mu_idx: int = 1,
    ) -> None:
        self.geofence_radius_m = float(geofence_radius_m)
        self.geofence_yaw_rad = float(geofence_yaw_rad)
        self.feedback_stale_timeout_s = float(feedback_stale_timeout_s)
        self.tier_limits = dict(tier_limits)
        self.require_robot_status_normal = bool(require_robot_status_normal)
        self.require_base_motor_power_on = bool(require_base_motor_power_on)
        self.require_target_tracking_mode = bool(require_target_tracking_mode)
        self.require_legwaist_impedance_mode = bool(require_legwaist_impedance_mode)
        self.apply_command_slew_clamp = bool(apply_command_slew_clamp)
        self.base_mu_idx = int(base_mu_idx)
        self.legwaist_mu_idx = int(legwaist_mu_idx)

        self._odom_origin: Optional[Tuple[float, float, float]] = None
        self._mocap_origin: Optional[Tuple[float, float, float]] = None
        self._prev_base_wall_s: Optional[float] = None
        self._prev_base_header_s: Optional[float] = None
        self._prev_arm_wall_s: Optional[float] = None
        self._prev_arm_header_s: Optional[float] = None
        self.trip_records: List[SafetyTripRecord] = []

    def latch_trial_origin(
        self,
        *,
        odom_pose_xyyaw: Tuple[float, float, float],
        mocap_pose_xyyaw: Optional[Tuple[float, float, float]] = None,
    ) -> None:
        """Latch initial chassis pose ``(x0, y0, yaw0)`` at the start of a trial."""
        if not all(math.isfinite(v) for v in odom_pose_xyyaw):
            raise SafetyTripError(
                "ORIGIN_NON_FINITE", f"Cannot latch non-finite odom origin: {odom_pose_xyyaw}"
            )
        self._odom_origin = tuple(float(v) for v in odom_pose_xyyaw)  # type: ignore[assignment]
        if mocap_pose_xyyaw is not None:
            if not all(math.isfinite(v) for v in mocap_pose_xyyaw):
                raise SafetyTripError(
                    "MOCAP_ORIGIN_NON_FINITE",
                    f"Cannot latch non-finite mocap origin: {mocap_pose_xyyaw}",
                )
            self._mocap_origin = tuple(float(v) for v in mocap_pose_xyyaw)  # type: ignore[assignment]
        else:
            self._mocap_origin = None

    def _record_and_raise(self, wall_time_s: float, trial_id: str, code: str, detail: str) -> None:
        self.trip_records.append(SafetyTripRecord(wall_time_s, trial_id, code, detail))
        raise SafetyTripError(code, detail)

    def _validate_stream(
        self,
        stream_name: str,
        snap: Optional[FeedbackSnapshot],
        now_wall_s: float,
        trial_id: str,
        prev_wall_s: Optional[float],
        prev_header_s: Optional[float],
    ) -> Tuple[float, Optional[float]]:
        if snap is None:
            self._record_and_raise(
                now_wall_s, trial_id, f"{stream_name.upper()}_MISSING",
                f"Required feedback stream '{stream_name}' has no samples.",
            )
        assert snap is not None
        if not math.isfinite(snap.wall_time_s):
            self._record_and_raise(
                now_wall_s, trial_id, f"{stream_name.upper()}_NON_FINITE_TIME",
                f"Feedback '{stream_name}' wall_time_s is non-finite: {snap.wall_time_s}",
            )
        age_s = now_wall_s - snap.wall_time_s
        if age_s > self.feedback_stale_timeout_s:
            self._record_and_raise(
                now_wall_s, trial_id, f"{stream_name.upper()}_STALE",
                f"Feedback '{stream_name}' stale: age={age_s * 1000.0:.1f} ms > "
                f"{self.feedback_stale_timeout_s * 1000.0:.1f} ms.",
            )
        if prev_wall_s is not None and snap.wall_time_s < prev_wall_s - 1e-6:
            self._record_and_raise(
                now_wall_s, trial_id, f"{stream_name.upper()}_WALL_TIME_BACKWARD",
                f"Feedback '{stream_name}' wall_time went backward: "
                f"{snap.wall_time_s:.6f} < {prev_wall_s:.6f}.",
            )
        if snap.header_time_s is not None:
            if not math.isfinite(snap.header_time_s):
                self._record_and_raise(
                    now_wall_s, trial_id, f"{stream_name.upper()}_NON_FINITE_HEADER",
                    f"Feedback '{stream_name}' header_time_s is non-finite.",
                )
            if prev_header_s is not None and snap.header_time_s < prev_header_s - 1e-6:
                self._record_and_raise(
                    now_wall_s, trial_id, f"{stream_name.upper()}_HEADER_TIME_BACKWARD",
                    f"Feedback '{stream_name}' header_time_s went backward: "
                    f"{snap.header_time_s:.6f} < {prev_header_s:.6f}.",
                )
        arr = np.asarray(snap.values, dtype=np.float64)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            self._record_and_raise(
                now_wall_s, trial_id, f"{stream_name.upper()}_NON_FINITE_VALUE",
                f"Feedback '{stream_name}' contains NaN/Inf or empty values.",
            )
        if snap.pose_xyyaw is not None and not all(math.isfinite(v) for v in snap.pose_xyyaw):
            self._record_and_raise(
                now_wall_s, trial_id, f"{stream_name.upper()}_NON_FINITE_POSE",
                f"Feedback '{stream_name}' pose contains NaN/Inf: {snap.pose_xyyaw}",
            )
        return snap.wall_time_s, snap.header_time_s

    def preflight_check(
        self,
        *,
        now_wall_s: float,
        base_snap: Optional[FeedbackSnapshot],
        arm_snap: Optional[FeedbackSnapshot],
        robot_status: Optional[RobotStatusSnapshot] = None,
        mocap_snap: Optional[FeedbackSnapshot] = None,
        mocap_required: bool = False,
        existing_base_cmd_publishers: int = 0,
        allow_existing_publishers: bool = False,
        check_motion_ready: bool = True,
    ) -> None:
        """Fail-closed startup self-check before any command is published."""
        trial_id = "PREFLIGHT"
        self._prev_base_wall_s, self._prev_base_header_s = self._validate_stream(
            "base", base_snap, now_wall_s, trial_id, None, None
        )
        self._prev_arm_wall_s, self._prev_arm_header_s = self._validate_stream(
            "arm", arm_snap, now_wall_s, trial_id, None, None
        )
        if mocap_required:
            self._validate_stream("mocap", mocap_snap, now_wall_s, trial_id, None, None)

        if existing_base_cmd_publishers > 0 and not allow_existing_publishers:
            self._record_and_raise(
                now_wall_s,
                trial_id,
                "CONFLICTING_BASE_PUBLISHER",
                f"Detected {existing_base_cmd_publishers} existing publisher(s) on base command "
                "topic (e.g., MozTeleopRobotNode). Stop teleop or explicitly set "
                "base_command.allow_existing_publishers=true after confirming it is idle.",
            )

        if check_motion_ready and (
            self.require_robot_status_normal
            or self.require_base_motor_power_on
            or self.require_target_tracking_mode
        ):
            if robot_status is None:
                self._record_and_raise(
                    now_wall_s, trial_id, "ROBOT_STATUS_MISSING",
                    "/robot_states telemetry has not arrived during preflight.",
                )
            assert robot_status is not None
            if now_wall_s - robot_status.wall_time_s > max(1.0, self.feedback_stale_timeout_s):
                self._record_and_raise(
                    now_wall_s, trial_id, "ROBOT_STATUS_STALE",
                    "/robot_states telemetry is stale during preflight.",
                )
            if self.require_robot_status_normal and robot_status.safety_status != 0:
                self._record_and_raise(
                    now_wall_s, trial_id, "ROBOT_SAFETY_NOT_NORMAL",
                    f"/robot_states safety_status={robot_status.safety_status} (expected 0=NORMAL).",
                )
            if self.require_base_motor_power_on:
                if (
                    self.base_mu_idx >= len(robot_status.motor_status)
                    or robot_status.motor_status[self.base_mu_idx] != 1
                ):
                    self._record_and_raise(
                        now_wall_s, trial_id, "BASE_MOTOR_NOT_POWER_ON",
                        f"/robot_states motor_status[{self.base_mu_idx}]="
                        f"{robot_status.motor_status} (expected 1=POWER_ON).",
                    )
            if self.require_target_tracking_mode and robot_status.task_mode != 1:
                self._record_and_raise(
                    now_wall_s, trial_id, "ROBOT_NOT_IN_TRACKING_MODE",
                    f"/robot_states task_mode={robot_status.task_mode} "
                    "(expected 1=PLAN_MODE_TARGET_TRACKING).",
                )
            if self.require_legwaist_impedance_mode:
                if (
                    self.legwaist_mu_idx >= len(robot_status.control_mode)
                    or robot_status.control_mode[self.legwaist_mu_idx] not in (3, 4)
                ):
                    self._record_and_raise(
                        now_wall_s, trial_id, "LEGWAIST_NOT_IN_IMPEDANCE_MODE",
                        f"/robot_states control_mode[{self.legwaist_mu_idx}]="
                        f"{robot_status.control_mode} (expected 3=JOINT_IMPEDANCE or 4=CART_IMPEDANCE).",
                    )

        assert base_snap is not None
        odom_pose = base_snap.pose_xyyaw if base_snap.pose_xyyaw is not None else (0.0, 0.0, 0.0)
        mocap_pose = mocap_snap.pose_xyyaw if mocap_snap is not None else None
        self.latch_trial_origin(odom_pose_xyyaw=odom_pose, mocap_pose_xyyaw=mocap_pose)

    def check_step_safety(
        self,
        *,
        now_wall_s: float,
        trial_id: str,
        base_snap: Optional[FeedbackSnapshot],
        arm_snap: Optional[FeedbackSnapshot],
        odom_pose_xyyaw: Tuple[float, float, float],
        mocap_snap: Optional[FeedbackSnapshot] = None,
        mocap_required: bool = False,
        robot_status: Optional[RobotStatusSnapshot] = None,
    ) -> None:
        """Run every real-time check on each control tick; raises ``SafetyTripError`` on violation."""
        if self._odom_origin is None:
            self._record_and_raise(
                now_wall_s, trial_id, "ORIGIN_NOT_LATCHED",
                "Trial origin pose was not latched before control execution.",
            )

        self._prev_base_wall_s, self._prev_base_header_s = self._validate_stream(
            "base", base_snap, now_wall_s, trial_id,
            self._prev_base_wall_s, self._prev_base_header_s,
        )
        self._prev_arm_wall_s, self._prev_arm_header_s = self._validate_stream(
            "arm", arm_snap, now_wall_s, trial_id,
            self._prev_arm_wall_s, self._prev_arm_header_s,
        )

        if robot_status is not None and self.require_robot_status_normal:
            if robot_status.safety_status != 0:
                self._record_and_raise(
                    now_wall_s, trial_id, "ROBOT_SAFETY_TRIP",
                    f"/robot_states safety_status changed to {robot_status.safety_status}.",
                )

        # Software geofence on dead-reckoned / odom pose.
        if not all(math.isfinite(v) for v in odom_pose_xyyaw):
            self._record_and_raise(
                now_wall_s, trial_id, "ODOM_POSE_NON_FINITE",
                f"Odom pose is non-finite: {odom_pose_xyyaw}",
            )
        assert self._odom_origin is not None
        dx = odom_pose_xyyaw[0] - self._odom_origin[0]
        dy = odom_pose_xyyaw[1] - self._odom_origin[1]
        dyaw = abs(_angle_diff_rad(odom_pose_xyyaw[2], self._odom_origin[2]))
        radius_m = math.hypot(dx, dy)
        if radius_m > self.geofence_radius_m:
            self._record_and_raise(
                now_wall_s, trial_id, "GEOFENCE_RADIUS_ODOM",
                f"Odom displacement {radius_m:.3f} m exceeded geofence_radius_m="
                f"{self.geofence_radius_m:.3f} m.",
            )
        if dyaw > self.geofence_yaw_rad:
            self._record_and_raise(
                now_wall_s, trial_id, "GEOFENCE_YAW_ODOM",
                f"Odom yaw excursion {dyaw:.3f} rad exceeded geofence_yaw_rad="
                f"{self.geofence_yaw_rad:.3f} rad.",
            )

        # Software geofence on external mocap pose whenever available/required.
        if mocap_required:
            self._validate_stream("mocap", mocap_snap, now_wall_s, trial_id, None, None)
        if (
            mocap_snap is not None
            and mocap_snap.pose_xyyaw is not None
            and self._mocap_origin is not None
            and all(math.isfinite(v) for v in mocap_snap.pose_xyyaw)
        ):
            mdx = mocap_snap.pose_xyyaw[0] - self._mocap_origin[0]
            mdy = mocap_snap.pose_xyyaw[1] - self._mocap_origin[1]
            mrad = math.hypot(mdx, mdy)
            if mrad > self.geofence_radius_m:
                self._record_and_raise(
                    now_wall_s, trial_id, "GEOFENCE_RADIUS_MOCAP",
                    f"Mocap displacement {mrad:.3f} m exceeded geofence_radius_m="
                    f"{self.geofence_radius_m:.3f} m.",
                )
            if mocap_snap.extra.get("has_yaw", True):
                mdyaw = abs(_angle_diff_rad(mocap_snap.pose_xyyaw[2], self._mocap_origin[2]))
                if mdyaw > self.geofence_yaw_rad:
                    self._record_and_raise(
                        now_wall_s, trial_id, "GEOFENCE_YAW_MOCAP",
                        f"Mocap yaw excursion {mdyaw:.3f} rad exceeded geofence_yaw_rad="
                        f"{self.geofence_yaw_rad:.3f} rad.",
                    )

    def clamp_command(
        self,
        vx_m_s: float,
        vy_m_s: float,
        wz_rad_s: float,
        *,
        tier: str,
        dt_s: float,
        prev_cmd: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        now_wall_s: float = 0.0,
        trial_id: str = "",
    ) -> Tuple[float, float, float]:
        """Validate finiteness and clamp velocity (and optional accel ramp) to tier caps."""
        if not (math.isfinite(vx_m_s) and math.isfinite(vy_m_s) and math.isfinite(wz_rad_s)):
            self._record_and_raise(
                now_wall_s, trial_id, "CMD_NON_FINITE",
                f"Refusing non-finite command ({vx_m_s}, {vy_m_s}, {wz_rad_s}).",
            )
        tcfg = self.tier_limits[tier]
        vx = float(np.clip(vx_m_s, -tcfg.v_max_m_s, tcfg.v_max_m_s))
        vy = float(np.clip(vy_m_s, -tcfg.v_max_m_s, tcfg.v_max_m_s))
        planar = math.hypot(vx, vy)
        if planar > tcfg.v_max_m_s and planar > 1e-12:
            scale = tcfg.v_max_m_s / planar
            vx *= scale
            vy *= scale
        wz = float(np.clip(wz_rad_s, -tcfg.w_max_rad_s, tcfg.w_max_rad_s))

        # Optional soft slew clamp — never limits an immediate zero-brake command.
        is_zero_brake = (abs(vx) < 1e-12 and abs(vy) < 1e-12 and abs(wz) < 1e-12)
        if self.apply_command_slew_clamp and not is_zero_brake and dt_s > 0.0:
            dv_max = tcfg.a_max_m_s2 * dt_s
            dw_max = tcfg.alpha_max_rad_s2 * dt_s
            vx = float(np.clip(vx, prev_cmd[0] - dv_max, prev_cmd[0] + dv_max))
            vy = float(np.clip(vy, prev_cmd[1] - dv_max, prev_cmd[1] + dv_max))
            wz = float(np.clip(wz, prev_cmd[2] - dw_max, prev_cmd[2] + dw_max))
        return (vx, vy, wz)


def execute_zero_brake_burst(
    publish_zero_fn: Callable[[], None],
    get_measured_twist_fn: Optional[Callable[[], Optional[Tuple[float, float, float]]]] = None,
    *,
    rate_hz: float = 50.0,
    min_duration_s: float = 0.5,
    max_duration_s: float = 3.0,
    stop_vel_eps_m_s: float = 0.01,
    stop_yaw_rate_eps_rad_s: float = 0.02,
    clock_fn: Callable[[], float] = time.monotonic,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Dict[str, Any]:
    """Continuously publish zero-velocity packets at >=50 Hz for >=0.5 s until stopped."""
    eff_rate_hz = max(50.0, float(rate_hz))
    eff_min_s = max(0.5, float(min_duration_s))
    eff_max_s = max(eff_min_s, float(max_duration_s))
    period_s = 1.0 / eff_rate_hz

    t0 = clock_fn()
    sent_count = 0
    stopped_confirmed = False
    last_speed_m_s: Optional[float] = None
    last_abs_wz_rad_s: Optional[float] = None

    while True:
        now = clock_fn()
        publish_zero_fn()
        sent_count += 1
        elapsed = now - t0

        if get_measured_twist_fn is not None:
            meas = get_measured_twist_fn()
            if meas is not None and all(math.isfinite(v) for v in meas):
                last_speed_m_s = math.hypot(meas[0], meas[1])
                last_abs_wz_rad_s = abs(meas[2])
                if (
                    last_speed_m_s < stop_vel_eps_m_s
                    and last_abs_wz_rad_s < stop_yaw_rate_eps_rad_s
                ):
                    stopped_confirmed = True

        if elapsed >= eff_min_s and (stopped_confirmed or get_measured_twist_fn is None):
            break
        if elapsed >= eff_max_s:
            break

        next_tick = t0 + sent_count * period_s
        sleep_dt = next_tick - clock_fn()
        if sleep_dt > 0.0:
            sleep_fn(sleep_dt)

    return {
        "zero_packets_sent": sent_count,
        "duration_s": round(clock_fn() - t0, 4),
        "rate_hz_target": eff_rate_hz,
        "stopped_confirmed": stopped_confirmed,
        "final_speed_m_s": last_speed_m_s,
        "final_abs_wz_rad_s": last_abs_wz_rad_s,
    }
