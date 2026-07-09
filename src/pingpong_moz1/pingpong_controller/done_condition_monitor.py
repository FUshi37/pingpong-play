"""Done condition monitor for real-time safety checking.

Replicates the 9 termination conditions from the training environment
(mjx_juggle_env.py) to prevent the RL policy from operating outside its
trained safe boundaries. When any condition fires, the controller stops RL
and transitions to a safe fallback posture.
"""
from typing import Optional, Tuple
import numpy as np


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
    ) -> Tuple[bool, str, dict]:
        """Evaluate all done conditions.

        Args:
            ball_pos_m: Ball position (m) in sim frame [x, y, z_world]
            ball_vel_m_s: Ball velocity (m/s) in sim frame
            racket_pos_m: Current racket position (m) in sim frame, or None
            racket_anchor_m: Racket anchor position (m) at training initial pose
            base_pos_m: Base position (m) in sim frame [x, y, z], or None
                (stationary robot => [0, 0, z], base checks always pass)

        Returns:
            (is_done, reason, conditions_dict)
            - is_done: True if ANY condition fires
            - reason: Human-readable string describing first active condition
            - conditions_dict: Dict mapping condition names to bool (True=violated)
        """
        self.total_checks += 1

        bx, by, bz = ball_pos_m[:3]
        ax, ay, az = racket_anchor_m[:3]

        # Ball state conditions (computable from ball alone)
        conds = {
            "ball_too_low": bz < self.ball_z_floor,
            "ball_too_high": bz > self.ball_z_ceiling,
            "ball_x_out_of_bounds": abs(bx - ax) > self.ball_x_range,
            "ball_y_out_of_bounds": abs(by - ay) > self.ball_y_range,
        }

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

    def diagnostics(self) -> dict:
        """Return telemetry counters for logging."""
        return {
            "total_checks": self.total_checks,
            "total_violations": self.total_violations,
            "violation_rate": (
                self.total_violations / max(self.total_checks, 1)),
            "violation_counts": dict(self.violation_counts),
        }

    def reset_counters(self) -> None:
        """Clear telemetry counters (e.g., at the start of a new run)."""
        self.total_checks = 0
        self.total_violations = 0
        self.violation_counts.clear()
