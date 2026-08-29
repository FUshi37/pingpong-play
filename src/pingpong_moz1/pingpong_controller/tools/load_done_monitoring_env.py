#!/usr/bin/env python3
"""Translate supported ROS YAML safety parameters to probe environment keys."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


# Keep this intentionally limited to parameters understood by
# RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh.  In particular, do not import the ROS
# ``$(find-pkg-share ...)`` model paths: those are not shell-expanded probe
# paths and can point at a different policy artifact.
PARAMETER_TO_ENV = {
    "enable_done_monitoring": "ENABLE_DONE_MONITORING",
    "done_ball_z_floor_m": "DONE_Z_MIN_M",
    "done_ball_z_ceiling_m": "DONE_Z_MAX_M",
    "done_ball_x_range_m": "DONE_X_RANGE_M",
    "done_ball_y_range_m": "DONE_Y_RANGE_M",
    "done_ball_required_consecutive_ticks": (
        "DONE_BALL_REQUIRED_CONSECUTIVE_TICKS"),
    "done_racket_anchor_distance_m": "DONE_RACKET_ANCHOR_DISTANCE_M",
    "done_racket_z_high_offset_m": "DONE_RACKET_Z_HIGH_OFFSET_M",
    "done_racket_z_low_offset_m": "DONE_RACKET_Z_LOW_OFFSET_M",
    "done_ee_x_limits_m": "DONE_EE_X_LIMITS_M",
    "done_ee_y_limits_m": "DONE_EE_Y_LIMITS_M",
    "done_ee_z_limits_m": "DONE_EE_Z_LIMITS_M",
    "done_ee_psi_limits_rad": "DONE_EE_PSI_LIMITS_RAD",
    "done_ee_b_limits_rad": "DONE_EE_B_LIMITS_RAD",
    "done_ee_b_min_rad": "DONE_EE_B_MIN_RAD",
    "done_ee_b_max_rad": "DONE_EE_B_MAX_RAD",
    "done_ee_c_limits_rad": "DONE_EE_C_LIMITS_RAD",
    "done_ee_c_min_rad": "DONE_EE_C_MIN_RAD",
    "done_ee_c_max_rad": "DONE_EE_C_MAX_RAD",
    "done_ee_orientation_reference_xyzw": (
        "DONE_EE_ORIENTATION_REFERENCE_XYZW"),
    "done_ee_max_orientation_error_deg": (
        "DONE_EE_MAX_ORIENTATION_ERROR_DEG"),
    "done_enable_predictive_stop": "DONE_ENABLE_PREDICTIVE_STOP",
    "done_prediction_horizon_s": "DONE_PREDICTION_HORIZON_S",
    "done_prediction_dt_s": "DONE_PREDICTION_DT_S",
    "done_prediction_margin_m": "DONE_PREDICTION_MARGIN_M",
    "done_prediction_orientation_margin_rad": (
        "DONE_PREDICTION_ORIENTATION_MARGIN_RAD"),
    "enable_ball_capture_gate": "ENABLE_BALL_CAPTURE_GATE",
    "enable_ball_release_gate": "ENABLE_BALL_RELEASE_GATE",
    "enable_ball_lost_failsafe": "ENABLE_BALL_LOST_FAILSAFE",
    "ball_lost_failsafe_timeout_s": "BALL_LOST_FAILSAFE_TIMEOUT_S",
    "control_rate_hz": "CONTROL_RATE_HZ",
    "sim_base_height_m": "SIM_BASE_HEIGHT_M",
    # Optional probe-only keys are accepted when added to the same YAML.
    "workspace_safety_mode": "WORKSPACE_SAFETY_MODE",
    "workspace_hold_trigger_margin_m": "WORKSPACE_HOLD_TRIGGER_MARGIN_M",
    "workspace_soft_margin_m": "WORKSPACE_SOFT_MARGIN_M",
    "workspace_soft_pull_speed_m_s": "WORKSPACE_SOFT_PULL_SPEED_M_S",
    "workspace_soft_damping": "WORKSPACE_SOFT_DAMPING",
    "workspace_soft_max_correction_speed_m_s": (
        "WORKSPACE_SOFT_MAX_CORRECTION_SPEED_M_S"),
    "workspace_soft_dls_damping": "WORKSPACE_SOFT_DLS_DAMPING",
    "workspace_soft_nullspace_damping": (
        "WORKSPACE_SOFT_NULLSPACE_DAMPING"),
    "workspace_soft_posture_gain_s": "WORKSPACE_SOFT_POSTURE_GAIN_S",
    "workspace_soft_boundary_joint_speed_scale": (
        "WORKSPACE_SOFT_BOUNDARY_JOINT_SPEED_SCALE"),
    "safety_parking_velocity_scale": "SAFETY_PARKING_VELOCITY_SCALE",
    "safety_parking_acceleration_scale": (
        "SAFETY_PARKING_ACCELERATION_SCALE"),
}


def _ros_parameters(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("YAML root must be a mapping")
    preferred = document.get("pingpong_controller")
    if isinstance(preferred, dict):
        parameters = preferred.get("ros__parameters")
        if isinstance(parameters, dict):
            return parameters
    for node_config in document.values():
        if not isinstance(node_config, dict):
            continue
        parameters = node_config.get("ros__parameters")
        if isinstance(parameters, dict):
            return parameters
    raise ValueError("no '<node>.ros__parameters' mapping found")


def _shell_text(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ",".join(_shell_text(item) for item in value)
    if value is None:
        return ""
    if isinstance(value, dict):
        raise ValueError("mapping values are not supported")
    text = str(value)
    if "\n" in text or "\r" in text or "\t" in text:
        raise ValueError("parameter values may not contain tabs or newlines")
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("yaml_path", type=Path)
    args = parser.parse_args()
    with args.yaml_path.open("r", encoding="utf-8") as handle:
        parameters = _ros_parameters(yaml.safe_load(handle))
    for parameter_name, env_name in PARAMETER_TO_ENV.items():
        if parameter_name not in parameters:
            continue
        print(f"{env_name}\t{_shell_text(parameters[parameter_name])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
