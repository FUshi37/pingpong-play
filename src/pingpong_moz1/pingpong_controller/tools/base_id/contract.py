#!/usr/bin/env python3
"""Interface contract loading, CLI dot-override merging, hashing, and fail-closed validation.

All physical quantities in the contract use SI units (m, s, rad, m/s, rad/s, m/s^2, rad/s^2).
Any required field set to ``UNRESOLVED_REQUIRE_OPERATOR_FILL`` causes ``validate_contract``
to fail closed before any experiment execution.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import yaml

CONTRACT_SCHEMA_VERSION = "moz1_base_id_contract_v1"
UNRESOLVED_SENTINEL = "UNRESOLVED_REQUIRE_OPERATOR_FILL"

SUPPORTED_BASE_COMMAND_PROTOCOLS = ("vector3_vel", "mech_unit_value_array")
SUPPORTED_ARM_COMMAND_PROTOCOLS = ("mech_unit_value_array", "mech_unit_cmd_array")
SUPPORTED_BASE_POSE_SOURCES = ("integrate_base_twist", "cartesian_pose_world")
SUPPORTED_MOCAP_TRACK_MODES = ("marker", "rigid_body")

DEFAULT_TIER_LIMITS: Dict[str, Dict[str, float]] = {
    "low": {
        "v_max_m_s": 0.20,
        "w_max_rad_s": 0.30,
        "a_max_m_s2": 0.80,
        "alpha_max_rad_s2": 1.20,
        "max_disp_m": 0.30,
    },
    "med": {
        "v_max_m_s": 0.50,
        "w_max_rad_s": 0.80,
        "a_max_m_s2": 1.50,
        "alpha_max_rad_s2": 2.50,
        "max_disp_m": 0.60,
    },
    "high": {
        "v_max_m_s": 0.90,
        "w_max_rad_s": 1.40,
        "a_max_m_s2": 2.50,
        "alpha_max_rad_s2": 4.00,
        "max_disp_m": 0.90,
    },
}


class ContractValidationError(ValueError):
    """Raised when a base-ID interface contract is incomplete, malformed, or unsafe."""


def default_contract_path() -> Path:
    """Return the default contract YAML path beside this module."""
    return Path(__file__).resolve().parent / "base_id_interface_contract.yaml"


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` until a directory containing ``src/pingpong_moz1`` is found."""
    cur = (start or Path(__file__).resolve()).resolve()
    if cur.is_file():
        cur = cur.parent
    for candidate in (cur, *cur.parents):
        if (candidate / "src" / "pingpong_moz1").is_dir():
            return candidate
    return Path(__file__).resolve().parents[5]


def build_default_contract_template() -> Dict[str, Any]:
    """Return the canonical contract skeleton with conservative safety defaults.

    Fields that cannot be determined uniquely from repository inspection alone are
    initialized to ``UNRESOLVED_REQUIRE_OPERATOR_FILL`` so startup fails closed until
    resolved via YAML edit or CLI ``--set`` overrides.
    """
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "ros": {
            "domain_id": 33,
            "robot_host": "172.16.0.20",
        },
        "base_command": {
            "protocol": UNRESOLVED_SENTINEL,
            "topic": UNRESOLVED_SENTINEL,
            "msg_type": UNRESOLVED_SENTINEL,
            "qos_reliability": "best_effort",
            "mech_unit_name": "base",
            "mode": 1,
            "frame_type": 0,
            "field_mapping": {
                "vx_m_s": UNRESOLVED_SENTINEL,
                "vy_m_s": UNRESOLVED_SENTINEL,
                "wz_rad_s": UNRESOLVED_SENTINEL,
            },
            "publish_rate_hz": 50.0,
            "zero_brake_rate_hz": 50.0,
            "zero_brake_min_duration_s": 0.5,
            "allow_existing_publishers": False,
        },
        "base_feedback": {
            "primary_topic": UNRESOLVED_SENTINEL,
            "primary_msg_type": UNRESOLVED_SENTINEL,
            "secondary_joint_topic": "/joint_states",
            "secondary_joint_msg_type": "sensor_msgs/msg/JointState",
            "robot_status_topic": "/robot_states",
            "robot_status_msg_type": "mc_core_interface/msg/RobotStatus",
            "servo_feedback_topic": "/servo_data_msg",
            "servo_feedback_msg_type": "mc_core_interface/msg/ServoFeedback",
            "mech_unit_name": "base",
            "base_mu_idx": 0,
            "wheel_joints": ["Base-0", "Base-1", "Base-2", "Base-3"],
            "base_pose_source": "integrate_base_twist",
            "expected_kinematic_type": "omnidirectional_4wheel",
            "wheel_half_xy_m": [0.2423, 0.2423],
        },
        "imu_feedback": {
            "topic": "/imu_states",
            "msg_type": "geometry_msgs/msg/Vector3Stamped",
            "required": False,
            "angle_unit": "rad",
        },
        "arm_feedback": {
            "primary_topic": UNRESOLVED_SENTINEL,
            "primary_msg_type": UNRESOLVED_SENTINEL,
            "secondary_joint_topic": "/joint_states",
            "mech_unit_name": "right_arm",
            "right_arm_mu_idx": 3,
            "joint_names": [
                "RightArm-0",
                "RightArm-1",
                "RightArm-2",
                "RightArm-3",
                "RightArm-4",
                "RightArm-5",
                "RightArm-6",
            ],
            "fk_model_xml": "src/pingpong_moz1/pingpong_controller/models/moz1_pd.xml",
            "fk_ee_site": "right_ee_site",
            "fk_racket_normal_axis_index": 2,
        },
        "arm_concurrency_command": {
            "protocol": UNRESOLVED_SENTINEL,
            "topic": UNRESOLVED_SENTINEL,
            "msg_type": UNRESOLVED_SENTINEL,
            "qos_reliability": "best_effort",
            "mech_unit_name": "right_arm",
            "mode": 0,
            "right_arm_mu_idx": 3,
            "publish_rate_hz": 200.0,
            "hold_mode": "latched_current_pose",
            "max_joint_delta_rad": 0.05,
            "max_micro_sine_amp_rad": 0.02,
        },
        "mocap": {
            "enabled": False,
            "base_track_mode": "rigid_body",
            "bind_host": "0.0.0.0",
            "udp_port": 65501,
            "calibration_path": "mocap_calibration_auto.txt",
            "position_unit": "auto",
            "timestamp_unit": "s",
            "timeout_s": 0.5,
            "abort_on_dropout": False,
            "base_rigid_body_id": 1,
            "base_rigid_body_slot": 1,
            "base_rigid_body_name": None,
            "base_rigid_body_observation_bounds_m": None,
            "base_lever_arm_xy_m": [0.0, 0.0],
            "base_marker_id": None,
            "base_marker_observation_bounds_m": None,
            "racket_rigid_body_id": None,
            "racket_rigid_body_slot": None,
            "racket_rigid_body_name": None,
        },
        "operator_site_verification": {
            "base_frame_id": "base_link",
            "arm_mount_frame_id": "waist03",
            "base_plus_x_physical_direction": UNRESOLVED_SENTINEL,
            "base_plus_y_physical_direction": UNRESOLVED_SENTINEL,
            "base_plus_yaw_ccw_from_top_verified": UNRESOLVED_SENTINEL,
            "controller_base_vel_override_verified": UNRESOLVED_SENTINEL,
            "teleop_publisher_conflict_cleared": UNRESOLVED_SENTINEL,
        },
        "safety": {
            "geofence_radius_m": 1.0,
            "geofence_yaw_rad": 3.5,
            "feedback_stale_timeout_s": 0.100,
            "stop_vel_eps_m_s": 0.01,
            "stop_yaw_rate_eps_rad_s": 0.02,
            "brake_max_wait_s": 3.0,
            "require_robot_status_normal": True,
            "require_base_motor_power_on": True,
            "require_target_tracking_mode": True,
            "require_legwaist_impedance_mode": False,
            "legwaist_mu_idx": 1,
            "apply_command_slew_clamp": False,
            "tier_limits": copy.deepcopy(DEFAULT_TIER_LIMITS),
        },
    }


def _parse_scalar_override(raw: str) -> Any:
    """Parse a CLI ``--set key=value`` scalar with YAML typing rules."""
    loaded = yaml.safe_load(raw)
    return loaded


def apply_dot_overrides(
    contract: Mapping[str, Any],
    overrides: Sequence[str],
) -> Dict[str, Any]:
    """Return a deep copy of ``contract`` with ``dot.path=value`` overrides applied."""
    updated = copy.deepcopy(dict(contract))
    for item in overrides:
        if "=" not in item:
            raise ContractValidationError(
                f"Invalid --set override {item!r}: expected 'key.path=value'."
            )
        key_path, raw_value = item.split("=", 1)
        keys = [k.strip() for k in key_path.split(".") if k.strip()]
        if not keys:
            raise ContractValidationError(f"Empty key path in override {item!r}.")
        node: Any = updated
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = _parse_scalar_override(raw_value)
    return updated


def load_contract(
    path: Path | str,
    *,
    overrides: Sequence[str] = (),
) -> Dict[str, Any]:
    """Load a YAML contract file and apply optional CLI dot-path overrides."""
    contract_path = Path(path)
    if not contract_path.is_file():
        raise ContractValidationError(f"Contract file does not exist: {contract_path}")
    with contract_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ContractValidationError(
            f"Contract file must decode to a YAML mapping: {contract_path}"
        )
    if overrides:
        data = apply_dot_overrides(data, overrides)
    return data


def compute_contract_sha256(contract: Mapping[str, Any]) -> str:
    """Return a deterministic SHA-256 hex digest of a contract mapping."""
    canonical = json.dumps(contract, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def find_unresolved_fields(
    obj: Any,
    *,
    prefix: str = "",
    ignore_prefixes: Iterable[str] = (),
) -> List[str]:
    """Recursively collect dot-paths whose value equals or contains ``UNRESOLVED_SENTINEL``."""
    ignored = tuple(ignore_prefixes)
    found: List[str] = []
    if prefix and any(prefix == ig or prefix.startswith(ig + ".") for ig in ignored):
        return found
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            sub = f"{prefix}.{k}" if prefix else str(k)
            found.extend(find_unresolved_fields(v, prefix=sub, ignore_prefixes=ignored))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            sub = f"{prefix}[{idx}]"
            found.extend(find_unresolved_fields(item, prefix=sub, ignore_prefixes=ignored))
    elif isinstance(obj, str) and UNRESOLVED_SENTINEL in obj:
        found.append(prefix)
    return found


def _require_positive_finite(value: Any, name: str, errors: List[str]) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be a finite positive number, got {value!r}")
        return 0.0
    if not math.isfinite(val) or val <= 0.0:
        errors.append(f"{name} must be finite and > 0, got {value!r}")
        return 0.0
    return val


def validate_contract(
    contract: Mapping[str, Any],
    *,
    stage: str | None = None,
    repo_root: Path | None = None,
) -> List[str]:
    """Validate contract completeness and physical safety bounds; fail closed on errors.

    Returns a list of non-fatal informational warnings if valid, or raises
    ``ContractValidationError`` listing every failure.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if contract.get("schema_version") != CONTRACT_SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {CONTRACT_SCHEMA_VERSION!r}, "
            f"got {contract.get('schema_version')!r}"
        )

    # Ignore unresolved fields inside optional mocap section only when mocap is disabled.
    mocap_cfg = contract.get("mocap") or {}
    mocap_enabled = bool(isinstance(mocap_cfg, Mapping) and mocap_cfg.get("enabled", False))
    ignore_prefixes: List[str] = []
    if not mocap_enabled:
        ignore_prefixes.append("mocap")

    unresolved = find_unresolved_fields(contract, ignore_prefixes=ignore_prefixes)
    if unresolved:
        errors.append(
            "Unresolved operator fields must be filled in YAML or overridden via --set "
            f"before running: {', '.join(unresolved)}"
        )

    # Base command protocol & topic validation.
    base_cmd = contract.get("base_command")
    if not isinstance(base_cmd, Mapping):
        errors.append("Missing required mapping: base_command")
    else:
        proto = base_cmd.get("protocol")
        if proto not in SUPPORTED_BASE_COMMAND_PROTOCOLS:
            errors.append(
                f"base_command.protocol must be one of {SUPPORTED_BASE_COMMAND_PROTOCOLS}, "
                f"got {proto!r}"
            )
        topic = str(base_cmd.get("topic") or "").strip()
        if not topic.startswith("/"):
            errors.append(f"base_command.topic must be an absolute ROS topic, got {topic!r}")
        msg_type = str(base_cmd.get("msg_type") or "").strip()
        if proto == "vector3_vel" and msg_type != "geometry_msgs/msg/Vector3":
            errors.append(
                "base_command.protocol='vector3_vel' requires "
                "msg_type='geometry_msgs/msg/Vector3', "
                f"got {msg_type!r}"
            )
        if proto == "mech_unit_value_array":
            if msg_type != "mc_core_interface/msg/MechUnitValueArray":
                errors.append(
                    "base_command.protocol='mech_unit_value_array' requires "
                    "msg_type='mc_core_interface/msg/MechUnitValueArray', "
                    f"got {msg_type!r}"
                )
            if int(base_cmd.get("mode", -1)) != 1:
                errors.append(
                    "base_command.protocol='mech_unit_value_array' requires mode=1 "
                    "(MechUnitValueMode::Cartesian velocity), "
                    f"got {base_cmd.get('mode')!r}"
                )
            frame_type = int(base_cmd.get("frame_type", -1))
            if frame_type not in (0, 1):
                errors.append(
                    "base_command.frame_type for base velocity control must be 0 or 1 "
                    "(frame_type=2 is world position mode in mc_core), "
                    f"got {frame_type!r}"
                )
        pub_hz = _require_positive_finite(
            base_cmd.get("publish_rate_hz"), "base_command.publish_rate_hz", errors
        )
        if pub_hz > 250.0:
            errors.append(f"base_command.publish_rate_hz too high (>250 Hz): {pub_hz}")
        brake_hz = _require_positive_finite(
            base_cmd.get("zero_brake_rate_hz"), "base_command.zero_brake_rate_hz", errors
        )
        if brake_hz < 50.0:
            errors.append(
                f"base_command.zero_brake_rate_hz must be >= 50.0 Hz, got {brake_hz}"
            )
        brake_dur = _require_positive_finite(
            base_cmd.get("zero_brake_min_duration_s"),
            "base_command.zero_brake_min_duration_s",
            errors,
        )
        if brake_dur < 0.5:
            errors.append(
                f"base_command.zero_brake_min_duration_s must be >= 0.5 s, got {brake_dur}"
            )

    # Base feedback validation.
    base_fb = contract.get("base_feedback")
    if not isinstance(base_fb, Mapping):
        errors.append("Missing required mapping: base_feedback")
    else:
        for key in ("primary_topic", "secondary_joint_topic"):
            val = str(base_fb.get(key) or "").strip()
            if not val.startswith("/"):
                errors.append(f"base_feedback.{key} must start with '/', got {val!r}")
        pose_src = base_fb.get("base_pose_source")
        if pose_src not in SUPPORTED_BASE_POSE_SOURCES:
            errors.append(
                f"base_feedback.base_pose_source must be one of {SUPPORTED_BASE_POSE_SOURCES}, "
                f"got {pose_src!r}"
            )
        wheel_joints = base_fb.get("wheel_joints")
        if not isinstance(wheel_joints, list) or len(wheel_joints) < 2:
            errors.append("base_feedback.wheel_joints must be a non-empty list of wheel names")

    # Arm feedback & FK XML validation.
    arm_fb = contract.get("arm_feedback")
    if not isinstance(arm_fb, Mapping):
        errors.append("Missing required mapping: arm_feedback")
    else:
        jnames = arm_fb.get("joint_names")
        if not isinstance(jnames, list) or len(jnames) != 7:
            errors.append("arm_feedback.joint_names must list the 7 right-arm joints")
        xml_rel = str(arm_fb.get("fk_model_xml") or "").strip()
        if not xml_rel:
            errors.append("arm_feedback.fk_model_xml must be set")
        else:
            root = repo_root or find_repo_root()
            xml_path = Path(xml_rel)
            if not xml_path.is_absolute():
                xml_path = root / xml_path
            if not xml_path.is_file():
                errors.append(f"arm_feedback.fk_model_xml does not exist: {xml_path}")

    # Arm concurrency command validation (always validate structure; critical for s5).
    arm_cmd = contract.get("arm_concurrency_command")
    if not isinstance(arm_cmd, Mapping):
        errors.append("Missing required mapping: arm_concurrency_command")
    else:
        arm_proto = arm_cmd.get("protocol")
        if arm_proto not in SUPPORTED_ARM_COMMAND_PROTOCOLS:
            errors.append(
                f"arm_concurrency_command.protocol must be one of "
                f"{SUPPORTED_ARM_COMMAND_PROTOCOLS}, got {arm_proto!r}"
            )
        arm_topic = str(arm_cmd.get("topic") or "").strip()
        if not arm_topic.startswith("/"):
            errors.append(
                f"arm_concurrency_command.topic must start with '/', got {arm_topic!r}"
            )
        _require_positive_finite(
            arm_cmd.get("publish_rate_hz"), "arm_concurrency_command.publish_rate_hz", errors
        )
        max_delta = _require_positive_finite(
            arm_cmd.get("max_joint_delta_rad"),
            "arm_concurrency_command.max_joint_delta_rad",
            errors,
        )
        if max_delta > 0.15:
            errors.append(
                f"arm_concurrency_command.max_joint_delta_rad too large (>0.15 rad): {max_delta}"
            )

    # Mocap checks when enabled.
    if mocap_enabled:
        track_mode = str(mocap_cfg.get("base_track_mode") or "marker").strip().lower()
        if track_mode not in SUPPORTED_MOCAP_TRACK_MODES:
            errors.append(
                f"mocap.base_track_mode must be one of {SUPPORTED_MOCAP_TRACK_MODES}, "
                f"got {track_mode!r}"
            )
        elif track_mode == "marker":
            if (
                mocap_cfg.get("base_marker_id") is None
                and mocap_cfg.get("base_marker_observation_bounds_m") is None
            ):
                errors.append(
                    "mocap.enabled=true with base_track_mode='marker' requires "
                    "mocap.base_marker_id or mocap.base_marker_observation_bounds_m"
                )
        elif track_mode == "rigid_body":
            if (
                mocap_cfg.get("base_rigid_body_id") is None
                and mocap_cfg.get("base_rigid_body_slot") is None
                and mocap_cfg.get("base_rigid_body_observation_bounds_m") is None
            ):
                errors.append(
                    "mocap.enabled=true with base_track_mode='rigid_body' requires "
                    "base_rigid_body_slot, base_rigid_body_id, or "
                    "base_rigid_body_observation_bounds_m"
                )

    # Safety limits validation.
    safety = contract.get("safety")
    if not isinstance(safety, Mapping):
        errors.append("Missing required mapping: safety")
    else:
        gf_r = _require_positive_finite(
            safety.get("geofence_radius_m"), "safety.geofence_radius_m", errors
        )
        if gf_r > 1.5:
            errors.append(
                f"safety.geofence_radius_m={gf_r:.3f} m exceeds hard cap 1.5 m; "
                "do not blindly enlarge geofence"
            )
        elif gf_r > 1.0:
            warnings.append(
                f"safety.geofence_radius_m={gf_r:.3f} m exceeds default 1.0 m; confirm room clearance"
            )
        _require_positive_finite(
            safety.get("geofence_yaw_rad"), "safety.geofence_yaw_rad", errors
        )
        stale_s = _require_positive_finite(
            safety.get("feedback_stale_timeout_s"), "safety.feedback_stale_timeout_s", errors
        )
        if stale_s > 0.25:
            errors.append(
                f"safety.feedback_stale_timeout_s={stale_s:.3f} s exceeds max safe limit 0.25 s"
            )
        _require_positive_finite(
            safety.get("stop_vel_eps_m_s"), "safety.stop_vel_eps_m_s", errors
        )
        _require_positive_finite(
            safety.get("stop_yaw_rate_eps_rad_s"), "safety.stop_yaw_rate_eps_rad_s", errors
        )

        tier_limits = safety.get("tier_limits")
        if not isinstance(tier_limits, Mapping):
            errors.append("safety.tier_limits must be a mapping containing low/med/high")
        else:
            for tier_name in ("low", "med", "high"):
                tcfg = tier_limits.get(tier_name)
                if not isinstance(tcfg, Mapping):
                    errors.append(f"safety.tier_limits.{tier_name} is missing")
                    continue
                v_max = _require_positive_finite(
                    tcfg.get("v_max_m_s"), f"safety.tier_limits.{tier_name}.v_max_m_s", errors
                )
                w_max = _require_positive_finite(
                    tcfg.get("w_max_rad_s"), f"safety.tier_limits.{tier_name}.w_max_rad_s", errors
                )
                _require_positive_finite(
                    tcfg.get("a_max_m_s2"), f"safety.tier_limits.{tier_name}.a_max_m_s2", errors
                )
                _require_positive_finite(
                    tcfg.get("alpha_max_rad_s2"),
                    f"safety.tier_limits.{tier_name}.alpha_max_rad_s2",
                    errors,
                )
                disp = _require_positive_finite(
                    tcfg.get("max_disp_m"), f"safety.tier_limits.{tier_name}.max_disp_m", errors
                )
                if gf_r > 0.0 and disp > gf_r:
                    errors.append(
                        f"safety.tier_limits.{tier_name}.max_disp_m ({disp:.3f} m) exceeds "
                        f"safety.geofence_radius_m ({gf_r:.3f} m)"
                    )
                if v_max > 1.2 or w_max > 2.0:
                    errors.append(
                        f"safety.tier_limits.{tier_name} exceeds absolute ceiling "
                        f"(v_max={v_max}, w_max={w_max})"
                    )

    if stage == "s5_arm_concurrency" and isinstance(arm_cmd, Mapping):
        if float(arm_cmd.get("publish_rate_hz", 0.0)) < 100.0:
            warnings.append(
                f"s5_arm_concurrency configured with arm publish_rate_hz="
                f"{arm_cmd.get('publish_rate_hz')} (<200 Hz nominal)"
            )

    if errors:
        bullet_list = "\n".join(f"  - {e}" for e in errors)
        raise ContractValidationError(
            f"Base-ID interface contract validation failed ({len(errors)} issue(s)):\n{bullet_list}"
        )
    return warnings
