"""Done condition monitor for real-time safety checking.

Replicates the 9 termination conditions from the training environment
(mjx_juggle_env.py) to prevent the RL policy from operating outside its
trained safe boundaries. When any condition fires, the controller stops RL
and transitions to a safe fallback posture.
"""
import math
from typing import Optional, Tuple
import numpy as np


def _normalize_quat_xyzw(quat) -> Optional[np.ndarray]:
    if quat is None:
        return None
    q = np.asarray(quat, dtype=np.float64).reshape(-1)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        return None
    norm = float(np.linalg.norm(q))
    if norm < 1e-9:
        return None
    return q / norm


def _quat_angle_error_deg(quat_xyzw, reference_xyzw) -> Optional[float]:
    q = _normalize_quat_xyzw(quat_xyzw)
    ref = _normalize_quat_xyzw(reference_xyzw)
    if q is None or ref is None:
        return None
    # q and -q represent the same orientation.
    dot = min(1.0, max(-1.0, abs(float(np.dot(q, ref)))))
    return float(math.degrees(2.0 * math.acos(dot)))


def _quat_to_euler_xyz_rad(quat_xyzw) -> Optional[np.ndarray]:
    """Return XYZ Euler angles [A, B, C] from [x, y, z, w] quaternion."""
    q = _normalize_quat_xyzw(quat_xyzw)
    if q is None:
        return None
    x, y, z, w = [float(v) for v in q]
    a = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y))
    sin_b = 2.0 * (w * y - z * x)
    b = math.asin(max(-1.0, min(1.0, sin_b)))
    c = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z))
    return np.asarray([a, b, c], dtype=np.float64)


def _wrap_angle_rad(angle: float) -> float:
    return float((float(angle) + math.pi) % (2.0 * math.pi) - math.pi)


def _angle_in_limits_rad(angle: float, limits: Tuple[float, float]) -> bool:
    a = _wrap_angle_rad(angle)
    lo = _wrap_angle_rad(float(limits[0]))
    hi = _wrap_angle_rad(float(limits[1]))
    if lo <= hi:
        return bool(lo <= a <= hi)
    # Wrapped interval, e.g. [170deg, -170deg].
    return bool(a >= lo or a <= hi)


def _angle_in_limits_rad_margin(
        angle: float, limits: Tuple[float, float], margin: float) -> bool:
    """Return True if angle is inside limits after shrinking by margin."""
    margin = max(0.0, float(margin))
    a = _wrap_angle_rad(angle)
    lo = _wrap_angle_rad(float(limits[0]))
    hi = _wrap_angle_rad(float(limits[1]))
    if lo <= hi:
        span = hi - lo
        if span <= 2.0 * margin:
            return False
        return bool((lo + margin) <= a <= (hi - margin))
    # Wrapped interval. Shrink each open edge; this is conservative near +/-pi.
    return bool(a >= _wrap_angle_rad(lo + margin)
                or a <= _wrap_angle_rad(hi - margin))


def _angle_above_min_rad(angle: float, minimum: float) -> bool:
    return bool(_wrap_angle_rad(angle) >= _wrap_angle_rad(minimum))


def _angle_below_max_rad(angle: float, maximum: float) -> bool:
    return bool(_wrap_angle_rad(angle) <= _wrap_angle_rad(maximum))


def _finite_vec_or_none(value, length: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (int(length),) or not np.all(np.isfinite(arr)):
        return None
    return arr


def _finite_nonnegative(value: float, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = float(default)
    if not math.isfinite(out) or out < 0.0:
        out = float(default)
    return max(0.0, out)


class DoneConditionMonitor:
    """Real-time monitor for simulation done conditions.

    Evaluates the same 9 termination conditions used in training to detect
    when the RL policy is operating outside its safe working envelope. All
    thresholds match the training environment (mjx_juggle_env.py:2778-2786).
    """

    def __init__(
        self,
        # Ball state limits (sim frame, meters)
        ball_z_floor: float = 0.8,
        ball_z_ceiling: float = 1.9,
        ball_x_range: float = 0.5,
        ball_y_range: float = 0.5,
        # Base position limits (sim frame, meters; stationary on real robot)
        base_x_limit: float = 2.6,
        base_y_limit: float = 2.6,
        # Racket position limits relative to anchor (meters)
        racket_anchor_distance_limit: float = 1.1,
        racket_z_high_offset: float = 0.24,
        racket_z_low_offset: float = -0.12,
        # End-effector workspace limits (optional, base frame, meters)
        ee_x_limits: Optional[Tuple[float, float]] = None,
        ee_y_limits: Optional[Tuple[float, float]] = None,
        ee_z_limits: Optional[Tuple[float, float]] = None,
        ee_psi_limits_rad: Optional[Tuple[float, float]] = None,
        ee_b_limits_rad: Optional[Tuple[float, float]] = None,
        ee_b_min_rad: Optional[float] = None,
        ee_b_max_rad: Optional[float] = None,
        ee_c_limits_rad: Optional[Tuple[float, float]] = None,
        ee_c_min_rad: Optional[float] = None,
        ee_c_max_rad: Optional[float] = None,
        ee_orientation_reference_xyzw: Optional[Tuple[float, float, float, float]] = None,
        ee_max_orientation_error_deg: float = 0.0,
        # Predictive stop: short-horizon EE position/orientation projection.
        enable_predictive_stop: bool = False,
        prediction_horizon_s: float = 0.0,
        prediction_dt_s: float = 0.01,
        prediction_margin_m: float = 0.0,
        prediction_orientation_margin_rad: float = 0.0,
    ):
        """Initialize done-condition thresholds.

        Args:
            ball_z_floor: Minimum ball height (m, sim frame z)
            ball_z_ceiling: Maximum ball height (m, sim frame z)
            ball_x_range: Max ball x distance from racket anchor (m)
            ball_y_range: Max ball y distance from racket anchor (m)
            base_x_limit: Max base x displacement (m, stationary => always OK)
            base_y_limit: Max base y displacement (m, stationary => always OK)
            racket_anchor_distance_limit: Max 3D racket distance from anchor (m)
            racket_z_high_offset: Max racket z above anchor (m)
            racket_z_low_offset: Max racket z below anchor (m, negative)
            ee_x_limits: Optional (min, max) end-effector x workspace (m, base frame)
            ee_y_limits: Optional (min, max) end-effector y workspace (m, base frame)
            ee_z_limits: Optional (min, max) end-effector z workspace (m, base frame)
            ee_psi_limits_rad: Optional wrapped (min, max) psi limits (rad)
            ee_b_limits_rad: Optional wrapped B-angle limits from XYZ Euler (rad)
            ee_b_min_rad: Optional one-sided B lower limit (rad)
            ee_b_max_rad: Optional one-sided B upper limit (rad)
            ee_c_limits_rad: Optional wrapped C-angle limits from XYZ Euler (rad)
            ee_c_min_rad: Optional one-sided C lower limit (rad)
            ee_c_max_rad: Optional one-sided C upper limit (rad)
            ee_orientation_reference_xyzw: Optional reference quaternion [x, y, z, w]
            ee_max_orientation_error_deg: Max quaternion angle error from reference;
                <=0 disables the quaternion-orientation condition
            enable_predictive_stop: If true, also evaluate short-horizon
                projected end-effector position/orientation before boundaries
                are crossed.
            prediction_horizon_s: Lookahead horizon for projected constraints.
            prediction_dt_s: Sampling period inside the lookahead horizon.
            prediction_margin_m: Inward workspace/racket-position margin used
                by predictive checks. Set to 0 to trigger at the boundary.
            prediction_orientation_margin_rad: Inward angular margin for
                predictive psi/B/C checks. Set to 0 to trigger at the boundary.
        """
        # Ball limits (sim frame)
        self.ball_z_floor = float(ball_z_floor)
        self.ball_z_ceiling = float(ball_z_ceiling)
        self.ball_x_range = float(ball_x_range)
        self.ball_y_range = float(ball_y_range)

        # Base limits (sim frame; stationary robot => always satisfied)
        self.base_x_limit = float(base_x_limit)
        self.base_y_limit = float(base_y_limit)

        # Racket limits relative to anchor (sim frame)
        self.racket_anchor_distance_limit = float(racket_anchor_distance_limit)
        self.racket_z_high_offset = float(racket_z_high_offset)
        self.racket_z_low_offset = float(racket_z_low_offset)

        # End-effector workspace limits (base frame, optional hard constraints)
        self.ee_x_limits = ee_x_limits
        self.ee_y_limits = ee_y_limits
        self.ee_z_limits = ee_z_limits
        self.ee_psi_limits_rad = ee_psi_limits_rad
        self.ee_b_limits_rad = ee_b_limits_rad
        self.ee_b_min_rad = (
            None if ee_b_min_rad is None else float(ee_b_min_rad))
        self.ee_b_max_rad = (
            None if ee_b_max_rad is None else float(ee_b_max_rad))
        self.ee_c_limits_rad = ee_c_limits_rad
        self.ee_c_min_rad = (
            None if ee_c_min_rad is None else float(ee_c_min_rad))
        self.ee_c_max_rad = (
            None if ee_c_max_rad is None else float(ee_c_max_rad))
        self.ee_orientation_reference_xyzw = _normalize_quat_xyzw(
            ee_orientation_reference_xyzw
            if ee_orientation_reference_xyzw is not None else [])
        self.ee_max_orientation_error_deg = float(ee_max_orientation_error_deg)
        self.enable_predictive_stop = bool(enable_predictive_stop)
        self.prediction_horizon_s = _finite_nonnegative(
            prediction_horizon_s, 0.0)
        self.prediction_dt_s = max(
            1e-4, _finite_nonnegative(prediction_dt_s, 0.01))
        self.prediction_margin_m = _finite_nonnegative(
            prediction_margin_m, 0.0)
        self.prediction_orientation_margin_rad = _finite_nonnegative(
            prediction_orientation_margin_rad, 0.0)
        self.last_prediction = {}

        # Counters for telemetry
        self.total_checks = 0
        self.total_violations = 0
        self.violation_counts = {}

    def check_done(
        self,
        ball_pos_m: np.ndarray,
        ball_vel_m_s: np.ndarray,
        racket_pos_m: Optional[np.ndarray],
        racket_anchor_m: np.ndarray,
        base_pos_m: Optional[np.ndarray] = None,
        ee_quat_xyzw: Optional[np.ndarray] = None,
        ee_psi_rad: Optional[float] = None,
        racket_vel_m_s: Optional[np.ndarray] = None,
        racket_acc_m_s2: Optional[np.ndarray] = None,
        ee_psi_vel_rad_s: Optional[float] = None,
        ee_abc_vel_rad_s: Optional[np.ndarray] = None,
        check_ball_conditions: bool = True,
    ) -> Tuple[bool, str, dict]:
        """Evaluate all done conditions.

        Args:
            ball_pos_m: Ball position (m) in sim frame [x, y, z_world]
            ball_vel_m_s: Ball velocity (m/s) in sim frame
            racket_pos_m: Current racket position (m) in sim frame, or None
            racket_anchor_m: Racket anchor position (m) at training initial pose
            base_pos_m: Base position (m) in sim frame [x, y, z], or None
                (stationary robot => [0, 0, z], base checks always pass)
            ee_quat_xyzw: Optional end-effector orientation quaternion [x, y, z, w]
            ee_psi_rad: Optional end-effector psi angle (rad)
            racket_vel_m_s: Optional current end-effector velocity in sim frame.
            racket_acc_m_s2: Optional current end-effector acceleration in sim frame.
            ee_psi_vel_rad_s: Optional current psi angular velocity.
            ee_abc_vel_rad_s: Optional XYZ Euler angular velocity [A, B, C].
            check_ball_conditions: If False, skip ball position conditions while
                still checking base/racket/EE safety constraints.

        Returns:
            (is_done, reason, conditions_dict)
            - is_done: True if ANY condition fires
            - reason: Human-readable string describing first active condition
            - conditions_dict: Dict mapping condition names to bool (True=violated)
        """
        self.total_checks += 1
        self.last_prediction = {}

        ax, ay, az = racket_anchor_m[:3]

        # Ball state conditions (computable from ball alone)
        conds = {
            "ball_too_low": False,
            "ball_too_high": False,
            "ball_x_out_of_bounds": False,
            "ball_y_out_of_bounds": False,
        }
        if check_ball_conditions:
            bx, by, bz = ball_pos_m[:3]
            conds["ball_too_low"] = bz < self.ball_z_floor
            conds["ball_too_high"] = bz > self.ball_z_ceiling
            conds["ball_x_out_of_bounds"] = abs(bx - ax) > self.ball_x_range
            conds["ball_y_out_of_bounds"] = abs(by - ay) > self.ball_y_range

        # Base position conditions (stationary real robot => always False)
        if base_pos_m is not None:
            base_x, base_y = base_pos_m[0], base_pos_m[1]
        else:
            base_x, base_y = 0.0, 0.0
        conds["base_x_out_of_bounds"] = abs(base_x) > self.base_x_limit
        conds["base_y_out_of_bounds"] = abs(base_y) > self.base_y_limit

        # Racket position conditions (require current racket pose from FK)
        if racket_pos_m is not None:
            rx, ry, rz = racket_pos_m[:3]
            racket_distance = float(np.linalg.norm(racket_pos_m - racket_anchor_m))
            racket_z_offset = rz - az

            conds["racket_too_far_from_anchor"] = (
                racket_distance > self.racket_anchor_distance_limit)
            conds["racket_too_high"] = racket_z_offset > self.racket_z_high_offset
            conds["racket_too_low"] = racket_z_offset < self.racket_z_low_offset

            # Optional: hard workspace limits on end-effector position (base frame)
            # These are ADDITIONAL constraints beyond the sim done-conditions,
            # useful for physical safety (e.g., collision avoidance, reachability).
            if self.ee_x_limits is not None:
                conds["ee_x_out_of_workspace"] = not (
                    self.ee_x_limits[0] <= rx <= self.ee_x_limits[1])
            if self.ee_y_limits is not None:
                conds["ee_y_out_of_workspace"] = not (
                    self.ee_y_limits[0] <= ry <= self.ee_y_limits[1])
            if self.ee_z_limits is not None:
                conds["ee_z_out_of_workspace"] = not (
                    self.ee_z_limits[0] <= rz <= self.ee_z_limits[1])
        else:
            # Racket FK unavailable; cannot check these conditions.
            conds["racket_too_far_from_anchor"] = False
            conds["racket_too_high"] = False
            conds["racket_too_low"] = False

        if self.ee_psi_limits_rad is not None:
            psi_ok = (
                ee_psi_rad is not None
                and np.isfinite(float(ee_psi_rad))
                and _angle_in_limits_rad(float(ee_psi_rad), self.ee_psi_limits_rad))
            conds["ee_psi_out_of_bounds"] = not bool(psi_ok)

        b_constraints_enabled = (
            self.ee_b_limits_rad is not None
            or self.ee_b_min_rad is not None
            or self.ee_b_max_rad is not None)
        c_constraints_enabled = (
            self.ee_c_limits_rad is not None
            or self.ee_c_min_rad is not None
            or self.ee_c_max_rad is not None)
        if b_constraints_enabled or c_constraints_enabled:
            abc = _quat_to_euler_xyz_rad(ee_quat_xyzw)
            if b_constraints_enabled:
                b_ok = (
                    abc is not None
                    and np.isfinite(float(abc[1]))
                    and (
                        self.ee_b_limits_rad is None
                        or _angle_in_limits_rad(
                            float(abc[1]), self.ee_b_limits_rad))
                    and (
                        self.ee_b_min_rad is None
                        or _angle_above_min_rad(
                            float(abc[1]), self.ee_b_min_rad))
                    and (
                        self.ee_b_max_rad is None
                        or _angle_below_max_rad(
                            float(abc[1]), self.ee_b_max_rad)))
                conds["ee_b_out_of_bounds"] = not bool(b_ok)
            if c_constraints_enabled:
                c_ok = (
                    abc is not None
                    and np.isfinite(float(abc[2]))
                    and (
                        self.ee_c_limits_rad is None
                        or _angle_in_limits_rad(
                            float(abc[2]), self.ee_c_limits_rad))
                    and (
                        self.ee_c_min_rad is None
                        or _angle_above_min_rad(
                            float(abc[2]), self.ee_c_min_rad))
                    and (
                        self.ee_c_max_rad is None
                        or _angle_below_max_rad(
                            float(abc[2]), self.ee_c_max_rad)))
                conds["ee_c_out_of_bounds"] = not bool(c_ok)

        if (self.ee_orientation_reference_xyzw is not None
                and self.ee_max_orientation_error_deg > 0.0):
            angle_error_deg = _quat_angle_error_deg(
                ee_quat_xyzw, self.ee_orientation_reference_xyzw)
            conds["ee_orientation_error_too_large"] = bool(
                angle_error_deg is not None
                and angle_error_deg > self.ee_max_orientation_error_deg)

        self._check_predictive_constraints(
            conds=conds,
            racket_pos_m=racket_pos_m,
            racket_anchor_m=racket_anchor_m,
            racket_vel_m_s=racket_vel_m_s,
            racket_acc_m_s2=racket_acc_m_s2,
            ee_psi_rad=ee_psi_rad,
            ee_psi_vel_rad_s=ee_psi_vel_rad_s,
            ee_quat_xyzw=ee_quat_xyzw,
            ee_abc_vel_rad_s=ee_abc_vel_rad_s,
        )

        # Aggregate
        is_done = any(conds.values())

        if is_done:
            self.total_violations += 1
            # Find first active condition for the reason string
            active = [k for k, v in conds.items() if v]
            reason = active[0] if active else "unknown"
            # Update per-condition counters
            for cond_name in active:
                self.violation_counts[cond_name] = (
                    self.violation_counts.get(cond_name, 0) + 1)
        else:
            reason = ""

        return is_done, reason, conds

    def _prediction_times(self) -> list[float]:
        if (not self.enable_predictive_stop
                or self.prediction_horizon_s <= 0.0):
            return []
        n = max(1, int(math.ceil(
            self.prediction_horizon_s / self.prediction_dt_s)))
        return [
            min(self.prediction_horizon_s, i * self.prediction_dt_s)
            for i in range(1, n + 1)
        ]

    def _mark_predicted(
            self, conds: dict, name: str, t_s: float, value=None) -> None:
        if conds.get(name, False):
            return
        conds[name] = True
        if not self.last_prediction:
            payload = {"condition": name, "time_s": float(t_s)}
            if value is not None:
                try:
                    arr = np.asarray(value, dtype=np.float64).reshape(-1)
                    payload["value"] = [float(x) for x in arr]
                except Exception:  # noqa: BLE001 - diagnostics only
                    payload["value"] = value
            self.last_prediction = payload

    @staticmethod
    def _inside_range_with_margin(
            value: float, limits: Tuple[float, float], margin: float) -> bool:
        lo, hi = float(limits[0]), float(limits[1])
        margin = max(0.0, min(float(margin), max(0.0, 0.49 * (hi - lo))))
        return bool((lo + margin) <= float(value) <= (hi - margin))

    def _check_predictive_constraints(
            self, *, conds: dict,
            racket_pos_m: Optional[np.ndarray],
            racket_anchor_m: np.ndarray,
            racket_vel_m_s: Optional[np.ndarray],
            racket_acc_m_s2: Optional[np.ndarray],
            ee_psi_rad: Optional[float],
            ee_psi_vel_rad_s: Optional[float],
            ee_quat_xyzw: Optional[np.ndarray],
            ee_abc_vel_rad_s: Optional[np.ndarray]) -> None:
        times = self._prediction_times()
        if not times:
            return

        pos = _finite_vec_or_none(racket_pos_m, 3)
        vel = _finite_vec_or_none(racket_vel_m_s, 3)
        if pos is not None and vel is not None:
            acc = _finite_vec_or_none(racket_acc_m_s2, 3)
            if acc is None:
                acc = np.zeros(3, dtype=np.float64)
            anchor = np.asarray(racket_anchor_m, dtype=np.float64).reshape(3)
            margin = self.prediction_margin_m
            for t_s in times:
                pred = pos + vel * t_s + 0.5 * acc * t_s * t_s
                dist = float(np.linalg.norm(pred - anchor))
                dz = float(pred[2] - anchor[2])
                if dist > self.racket_anchor_distance_limit - margin:
                    self._mark_predicted(
                        conds, "racket_predicted_too_far_from_anchor",
                        t_s, pred)
                if dz > self.racket_z_high_offset - margin:
                    self._mark_predicted(
                        conds, "racket_predicted_too_high", t_s, pred)
                if dz < self.racket_z_low_offset + margin:
                    self._mark_predicted(
                        conds, "racket_predicted_too_low", t_s, pred)
                if (self.ee_x_limits is not None
                        and not self._inside_range_with_margin(
                            pred[0], self.ee_x_limits, margin)):
                    self._mark_predicted(
                        conds, "ee_predicted_x_out_of_workspace", t_s, pred)
                if (self.ee_y_limits is not None
                        and not self._inside_range_with_margin(
                            pred[1], self.ee_y_limits, margin)):
                    self._mark_predicted(
                        conds, "ee_predicted_y_out_of_workspace", t_s, pred)
                if (self.ee_z_limits is not None
                        and not self._inside_range_with_margin(
                            pred[2], self.ee_z_limits, margin)):
                    self._mark_predicted(
                        conds, "ee_predicted_z_out_of_workspace", t_s, pred)

        orient_margin = self.prediction_orientation_margin_rad
        if (self.ee_psi_limits_rad is not None
                and ee_psi_rad is not None
                and ee_psi_vel_rad_s is not None
                and math.isfinite(float(ee_psi_rad))
                and math.isfinite(float(ee_psi_vel_rad_s))):
            for t_s in times:
                pred_psi = float(ee_psi_rad) + float(ee_psi_vel_rad_s) * t_s
                if not _angle_in_limits_rad_margin(
                        pred_psi, self.ee_psi_limits_rad, orient_margin):
                    self._mark_predicted(
                        conds, "ee_psi_predicted_out_of_bounds",
                        t_s, [pred_psi])
                    break

        abc = _quat_to_euler_xyz_rad(ee_quat_xyzw)
        abc_vel = _finite_vec_or_none(ee_abc_vel_rad_s, 3)
        if abc is None or abc_vel is None:
            return
        for t_s in times:
            pred_abc = abc + abc_vel * t_s
            if self.ee_b_limits_rad is not None and not _angle_in_limits_rad_margin(
                    float(pred_abc[1]), self.ee_b_limits_rad, orient_margin):
                self._mark_predicted(
                    conds, "ee_b_predicted_out_of_bounds",
                    t_s, pred_abc)
                break
            if (self.ee_b_min_rad is not None
                    and _wrap_angle_rad(float(pred_abc[1]))
                    < _wrap_angle_rad(self.ee_b_min_rad + orient_margin)):
                self._mark_predicted(
                    conds, "ee_b_predicted_out_of_bounds",
                    t_s, pred_abc)
                break
            if (self.ee_b_max_rad is not None
                    and _wrap_angle_rad(float(pred_abc[1]))
                    > _wrap_angle_rad(self.ee_b_max_rad - orient_margin)):
                self._mark_predicted(
                    conds, "ee_b_predicted_out_of_bounds",
                    t_s, pred_abc)
                break
            if self.ee_c_limits_rad is not None and not _angle_in_limits_rad_margin(
                    float(pred_abc[2]), self.ee_c_limits_rad, orient_margin):
                self._mark_predicted(
                    conds, "ee_c_predicted_out_of_bounds",
                    t_s, pred_abc)
                break
            if (self.ee_c_min_rad is not None
                    and _wrap_angle_rad(float(pred_abc[2]))
                    < _wrap_angle_rad(self.ee_c_min_rad + orient_margin)):
                self._mark_predicted(
                    conds, "ee_c_predicted_out_of_bounds",
                    t_s, pred_abc)
                break
            if (self.ee_c_max_rad is not None
                    and _wrap_angle_rad(float(pred_abc[2]))
                    > _wrap_angle_rad(self.ee_c_max_rad - orient_margin)):
                self._mark_predicted(
                    conds, "ee_c_predicted_out_of_bounds",
                    t_s, pred_abc)
                break

    def diagnostics(self) -> dict:
        """Return telemetry counters for logging."""
        return {
            "total_checks": self.total_checks,
            "total_violations": self.total_violations,
            "violation_rate": (
                self.total_violations / max(self.total_checks, 1)),
            "violation_counts": dict(self.violation_counts),
            "last_prediction": dict(self.last_prediction),
        }

    def reset_counters(self) -> None:
        """Clear telemetry counters (e.g., at the start of a new run)."""
        self.total_checks = 0
        self.total_violations = 0
        self.violation_counts.clear()
        self.last_prediction = {}
