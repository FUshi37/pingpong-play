#!/usr/bin/env python3
"""Read-only interface audit tool for MOZ1 base dynamics identification.

Scans repository sources, Movax YAML configurations, and MJCF models, and optionally
queries a read-only ``ros2 topic list -t`` / ``ros2 topic info -v`` snapshot (gracefully
degrading to repository-only inspection when ROS 2 is unavailable).

NEVER creates ROS publishers or sends any command to hardware.
"""

from __future__ import annotations

import argparse
import copy
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

try:
    from .contract import (
        UNRESOLVED_SENTINEL,
        build_default_contract_template,
        default_contract_path,
        find_repo_root,
        find_unresolved_fields,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from contract import (  # type: ignore[no-redef]
        UNRESOLVED_SENTINEL,
        build_default_contract_template,
        default_contract_path,
        find_repo_root,
        find_unresolved_fields,
    )


@dataclass
class TopicEndpointSummary:
    """Read-only summary parsed from ``ros2 topic info -v``."""

    topic: str
    msg_type: str
    publisher_count: int = 0
    subscription_count: int = 0
    publisher_nodes: List[str] = field(default_factory=list)
    subscriber_nodes: List[str] = field(default_factory=list)
    qos_reliabilities: List[str] = field(default_factory=list)


@dataclass
class InspectionResult:
    """Structured output of read-only repo + optional ROS 2 audit."""

    contract: Dict[str, Any]
    confirmed_items: List[Tuple[str, str, str]]  # (field_path, value_str, evidence)
    unresolved_items: List[Tuple[str, str]]  # (field_path, reason)
    advisories: List[str]
    ros_snapshot_available: bool
    ros_topics: Dict[str, str]
    ros_topic_details: Dict[str, TopicEndpointSummary]


def _safe_load_yaml(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _parse_topic_list_output(stdout: str) -> Dict[str, str]:
    """Parse ``ros2 topic list -t`` lines like ``/mx_base_vel_command [geometry_msgs/msg/Vector3]``."""
    topics: Dict[str, str] = {}
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        match = re.match(r"^(/\S+)\s+\[([^\]]+)\]$", line)
        if match:
            topic_name = match.group(1)
            msg_types = [t.strip() for t in match.group(2).split(",") if t.strip()]
            if msg_types:
                topics[topic_name] = msg_types[0]
    return topics


def _parse_topic_info_output(topic: str, msg_type: str, stdout: str) -> TopicEndpointSummary:
    """Parse ``ros2 topic info -v <topic>`` into publisher/subscriber node lists and QoS."""
    summary = TopicEndpointSummary(topic=topic, msg_type=msg_type)
    current_node: Optional[str] = None
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("Type:"):
            summary.msg_type = line.split(":", 1)[1].strip() or summary.msg_type
        elif line.startswith("Publisher count:"):
            try:
                summary.publisher_count = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("Subscription count:"):
            try:
                summary.subscription_count = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif line.startswith("Node name:"):
            current_node = line.split(":", 1)[1].strip()
        elif line.startswith("Endpoint type:"):
            ep = line.split(":", 1)[1].strip().upper()
            if current_node:
                if ep == "PUBLISHER":
                    summary.publisher_nodes.append(current_node)
                elif ep == "SUBSCRIPTION":
                    summary.subscriber_nodes.append(current_node)
        elif line.startswith("Reliability:"):
            rel = line.split(":", 1)[1].strip()
            if rel and rel not in summary.qos_reliabilities:
                summary.qos_reliabilities.append(rel)
    return summary


def query_live_ros_snapshot(
    *,
    timeout_s: float = 4.0,
    topics_to_detail: Sequence[str] = (
        "/mx_base_vel_command",
        "/mx_base_cmd",
        "/mx_base_state",
        "/mx_robot_cmd",
        "/mx_mix_command",
        "/mx_robot_state",
        "/joint_states",
        "/imu_states",
        "/robot_states",
        "/servo_data_msg",
        "/cart_states",
    ),
) -> Tuple[bool, Dict[str, str], Dict[str, TopicEndpointSummary], List[str]]:
    """Run read-only ``ros2 topic list -t`` and ``ros2 topic info -v`` if ROS 2 is available."""
    notes: List[str] = []
    ros2_bin = shutil.which("ros2")
    if ros2_bin is None:
        notes.append("ROS 2 CLI (`ros2`) not found in PATH; degraded gracefully to repository scan.")
        return False, {}, {}, notes

    try:
        proc = subprocess.run(
            [ros2_bin, "topic", "list", "-t"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except Exception as exc:
        notes.append(f"Read-only `ros2 topic list -t` failed ({exc}); degraded to repository scan.")
        return False, {}, {}, notes

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip().splitlines()[:2]
        notes.append(
            f"Read-only `ros2 topic list -t` exited {proc.returncode} ({' | '.join(err)}); "
            "degraded to repository scan."
        )
        return False, {}, {}, notes

    topics = _parse_topic_list_output(proc.stdout)
    if not topics:
        notes.append("ROS 2 graph returned no topics; degraded to repository scan.")
        return False, {}, {}, notes

    details: Dict[str, TopicEndpointSummary] = {}
    for topic in topics_to_detail:
        if topic not in topics:
            continue
        try:
            info_proc = subprocess.run(
                [ros2_bin, "topic", "info", "-v", topic],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            if info_proc.returncode == 0:
                details[topic] = _parse_topic_info_output(topic, topics[topic], info_proc.stdout)
        except Exception:
            continue

    return True, topics, details, notes


def _scan_mjcf_model(xml_path: Path) -> Dict[str, Any]:
    """Extract right_ee_site, right-arm joints, and base wheel geometry from MJCF."""
    info: Dict[str, Any] = {
        "exists": xml_path.is_file(),
        "has_right_ee_site": False,
        "right_arm_joints": [],
        "wheel_positions_xy_m": [],
        "base_mass_kg": None,
    }
    if not xml_path.is_file():
        return info
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return info
    for site in root.iter("site"):
        if site.get("name") == "right_ee_site":
            info["has_right_ee_site"] = True
    for joint in root.iter("joint"):
        jname = joint.get("name") or ""
        if jname.startswith("RightArm-"):
            info["right_arm_joints"].append(jname)
    for body in root.iter("body"):
        bname = body.get("name") or ""
        if bname == "base":
            inertial = body.find("inertial")
            if inertial is not None and inertial.get("mass"):
                try:
                    info["base_mass_kg"] = float(inertial.get("mass", ""))
                except ValueError:
                    pass
        if bname.startswith("wheel") and body.get("pos"):
            parts = [float(v) for v in body.get("pos", "").split()]
            if len(parts) >= 2:
                info["wheel_positions_xy_m"].append([parts[0], parts[1]])
    return info


def _scan_repo_sources(repo_root: Path) -> Dict[str, Any]:
    """Scan local repository source and Movax YAML files (strictly read-only)."""
    findings: Dict[str, Any] = {
        "right_arm_fk_module": None,
        "right_arm_fk_path": None,
        "pingpong_command_topic": None,
        "pingpong_joint_topic": None,
        "mocap_udp_port": None,
        "mocap_calibration_path": None,
        "motion_yaml_path": None,
        "motion_config": {},
        "model_base_config": {},
        "system_config": {},
        "mjcf": {},
    }

    fk_path = repo_root / "src" / "pingpong_replay" / "pingpong_replay" / "cartesian_fk.py"
    if fk_path.is_file() and "class RightArmFK" in fk_path.read_text(encoding="utf-8"):
        findings["right_arm_fk_module"] = "pingpong_replay.cartesian_fk.RightArmFK"
        findings["right_arm_fk_path"] = str(fk_path.relative_to(repo_root))

    node_path = repo_root / "src" / "pingpong_moz1" / "pingpong_controller" / "pingpong_node.py"
    if node_path.is_file():
        text = node_path.read_text(encoding="utf-8", errors="ignore")
        m_cmd = re.search(r"declare_parameter\(\s*['\"]command_topic['\"]\s*,\s*['\"]([^'\"]+)['\"]", text)
        m_js = re.search(r"declare_parameter\(\s*['\"]joint_state_topic['\"]\s*,\s*['\"]([^'\"]+)['\"]", text)
        if m_cmd:
            findings["pingpong_command_topic"] = m_cmd.group(1)
        if m_js:
            findings["pingpong_joint_topic"] = m_js.group(1)

    probe_sh = repo_root / "RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh"
    if probe_sh.is_file():
        text = probe_sh.read_text(encoding="utf-8", errors="ignore")
        m_port = re.search(r'MOCAP_UDP_PORT="\$\{MOCAP_UDP_PORT:-(\d+)\}"', text)
        if m_port:
            findings["mocap_udp_port"] = int(m_port.group(1))
        if "mocap_calibration_auto.txt" in text:
            findings["mocap_calibration_path"] = "mocap_calibration_auto.txt"

    motion_yaml = repo_root / "movax" / "controller" / "configuration" / "motion" / "motion.yaml"
    motion_doc = _safe_load_yaml(motion_yaml)
    if motion_doc:
        findings["motion_yaml_path"] = str(motion_yaml.relative_to(repo_root))
        findings["motion_config"] = motion_doc.get("motion_config") or {}

    sys_yaml = repo_root / "movax" / "controller" / "configuration" / "basic" / "system.yaml"
    sys_doc = _safe_load_yaml(sys_yaml)
    if sys_doc:
        findings["system_config"] = sys_doc

    model_yaml = (
        repo_root
        / "movax"
        / "controller"
        / "configuration"
        / "model"
        / "user"
        / "spirit01_base_e_no_neck_hh.yaml"
    )
    model_doc = _safe_load_yaml(model_yaml)
    if model_doc:
        mech_units = model_doc.get("mech_units") or {}
        if isinstance(mech_units, Mapping) and "Base" in mech_units:
            findings["model_base_config"] = mech_units["Base"]

    xml_path = repo_root / "src" / "pingpong_moz1" / "pingpong_controller" / "models" / "moz1_pd.xml"
    findings["mjcf"] = _scan_mjcf_model(xml_path)
    return findings


def inspect_base_interface(
    *,
    repo_root: Optional[Path] = None,
    enable_ros_snapshot: bool = True,
    ros_snapshot_override: Optional[
        Tuple[bool, Dict[str, str], Dict[str, TopicEndpointSummary], List[str]]
    ] = None,
) -> InspectionResult:
    """Perform a complete read-only interface audit and populate a contract template."""
    root = (repo_root or find_repo_root()).resolve()
    contract = build_default_contract_template()
    confirmed: List[Tuple[str, str, str]] = []
    unresolved_reasons: Dict[str, str] = {}
    advisories: List[str] = []

    repo = _scan_repo_sources(root)

    # 1. FK & MJCF checks from repository.
    mjcf = repo["mjcf"]
    if repo["right_arm_fk_module"] and mjcf.get("has_right_ee_site"):
        confirmed.append((
            "arm_feedback.fk_model_xml",
            contract["arm_feedback"]["fk_model_xml"],
            f"Verified RightArmFK ({repo['right_arm_fk_path']}) + right_ee_site in moz1_pd.xml",
        ))
    else:
        contract["arm_feedback"]["fk_model_xml"] = UNRESOLVED_SENTINEL
        unresolved_reasons["arm_feedback.fk_model_xml"] = (
            "RightArmFK or right_ee_site not found in repository."
        )

    wheel_xy = mjcf.get("wheel_positions_xy_m") or []
    if len(wheel_xy) == 4:
        hx = abs(float(wheel_xy[0][0]))
        hy = abs(float(wheel_xy[0][1]))
        contract["base_feedback"]["wheel_half_xy_m"] = [round(hx, 6), round(hy, 6)]
        confirmed.append((
            "base_feedback.wheel_half_xy_m",
            str(contract["base_feedback"]["wheel_half_xy_m"]),
            "Parsed 4 symmetric 45-deg wheel mounts from models/moz1_pd.xml",
        ))

    if repo["mocap_udp_port"] is not None:
        contract["mocap"]["udp_port"] = int(repo["mocap_udp_port"])
        confirmed.append((
            "mocap.udp_port",
            str(contract["mocap"]["udp_port"]),
            "Confirmed in RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh",
        ))
    if repo["mocap_calibration_path"]:
        contract["mocap"]["calibration_path"] = repo["mocap_calibration_path"]
        confirmed.append((
            "mocap.calibration_path",
            str(contract["mocap"]["calibration_path"]),
            "Confirmed in RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh",
        ))

    # 2. Movax motion/model config inspection.
    motion_cfg = repo["motion_config"]
    spd_override = motion_cfg.get("speed_override_param") if isinstance(motion_cfg, Mapping) else None
    if isinstance(spd_override, Mapping):
        b_vel_ov = spd_override.get("base_vel_override")
        b_acc_ov = spd_override.get("base_acc_override")
        if b_vel_ov is not None or b_acc_ov is not None:
            advisories.append(
                f"Movax motion.yaml has base_vel_override={b_vel_ov}, "
                f"base_acc_override={b_acc_ov}. Verify active controller override on robot host "
                "before identifying velocity/acceleration envelopes (0.1 scales commands to 10% "
                "in WheelBaseJointVelocityPlanner)."
            )

    base_model_cfg = repo["model_base_config"]
    if isinstance(base_model_cfg, Mapping):
        rmp = base_model_cfg.get("rmp_params") or {}
        if isinstance(rmp, Mapping) and "linear_limits" in rmp:
            advisories.append(
                f"Movax Base rmp_params in spirit01_base_e_no_neck_hh.yaml: "
                f"linear_limits={rmp.get('linear_limits')}, "
                f"angular_limits={rmp.get('angular_limits')}, "
                f"filter_length={rmp.get('filter_length')}."
            )

    # 3. Optional read-only ROS 2 live snapshot.
    if ros_snapshot_override is not None:
        ros_ok, ros_topics, ros_details, ros_notes = ros_snapshot_override
    elif enable_ros_snapshot:
        ros_ok, ros_topics, ros_details, ros_notes = query_live_ros_snapshot()
    else:
        ros_ok, ros_topics, ros_details, ros_notes = (
            False,
            {},
            {},
            ["ROS 2 live snapshot skipped by CLI flag (--no-ros-snapshot)."],
        )
    advisories.extend(ros_notes)

    # Resolve base_command topic & protocol ONLY when uniquely confirmed by live ROS graph.
    if ros_ok:
        vel_info = ros_details.get("/mx_base_vel_command")
        cmd_info = ros_details.get("/mx_base_cmd")
        vel_mc_sub = bool(vel_info and any("MCControlNode" in n for n in vel_info.subscriber_nodes))
        cmd_mc_sub = bool(cmd_info and any("MCControlNode" in n for n in cmd_info.subscriber_nodes))

        if vel_mc_sub and not cmd_mc_sub:
            contract["base_command"].update({
                "protocol": "vector3_vel",
                "topic": "/mx_base_vel_command",
                "msg_type": "geometry_msgs/msg/Vector3",
                "qos_reliability": "best_effort",
                "field_mapping": {
                    "vx_m_s": "x",
                    "vy_m_s": "y",
                    "wz_rad_s": "z",
                },
            })
            confirmed.append((
                "base_command",
                "protocol=vector3_vel, topic=/mx_base_vel_command [geometry_msgs/msg/Vector3]",
                "Confirmed MCControlNode subscription in live ROS 2 graph "
                "(use_new_robot_base_cmd=false)",
            ))
            if vel_info and vel_info.publisher_count > 0:
                advisories.append(
                    f"CONFLICT CHECK: /mx_base_vel_command currently has "
                    f"{vel_info.publisher_count} publisher(s): {vel_info.publisher_nodes}. "
                    "Ensure MozTeleopRobotNode is idle/stopped before live excitation."
                )
        elif cmd_mc_sub and not vel_mc_sub:
            contract["base_command"].update({
                "protocol": "mech_unit_value_array",
                "topic": "/mx_base_cmd",
                "msg_type": "mc_core_interface/msg/MechUnitValueArray",
                "qos_reliability": "best_effort",
                "mech_unit_name": "base",
                "mode": 1,
                "frame_type": 0,
                "field_mapping": {
                    "vx_m_s": "cartesian.twist.linear.x",
                    "vy_m_s": "cartesian.twist.linear.y",
                    "wz_rad_s": "cartesian.twist.angular.z",
                },
            })
            confirmed.append((
                "base_command",
                "protocol=mech_unit_value_array, topic=/mx_base_cmd "
                "[mc_core_interface/msg/MechUnitValueArray, mode=1, twist]",
                "Confirmed MCControlNode subscription in live ROS 2 graph "
                "(use_new_robot_base_cmd=true)",
            ))
            if cmd_info and cmd_info.publisher_count > 0:
                advisories.append(
                    f"CONFLICT CHECK: /mx_base_cmd currently has "
                    f"{cmd_info.publisher_count} publisher(s): {cmd_info.publisher_nodes}."
                )
        else:
            unresolved_reasons["base_command.protocol"] = (
                "Could not uniquely determine whether MCControlNode subscribes to "
                "/mx_base_vel_command (Vector3) or /mx_base_cmd (MechUnitValueArray mode=1)."
            )

        # Primary 200 Hz state topic (/mx_robot_state).
        if "/mx_robot_state" in ros_topics:
            mtype = ros_topics["/mx_robot_state"]
            contract["base_feedback"]["primary_topic"] = "/mx_robot_state"
            contract["base_feedback"]["primary_msg_type"] = mtype
            contract["arm_feedback"]["primary_topic"] = "/mx_robot_state"
            contract["arm_feedback"]["primary_msg_type"] = mtype
            confirmed.append((
                "base_feedback.primary_topic",
                f"/mx_robot_state [{mtype}]",
                "Confirmed live publisher (MCMessageNode, 200 Hz base twist + wheel + right_arm)",
            ))
            confirmed.append((
                "arm_feedback.primary_topic",
                f"/mx_robot_state [{mtype}]",
                "Confirmed live publisher (MCMessageNode, 200 Hz right_arm q/dq/effort)",
            ))
            advisories.append(
                "Note: in mc_core (frame_type=0), /mx_robot_state base cartesian.pose is Identity; "
                "base_pose_source='integrate_base_twist' integrates body (vx, vy, wz) for geofence."
            )
        elif "/joint_states" in ros_topics:
            contract["base_feedback"]["primary_topic"] = "/joint_states"
            contract["base_feedback"]["primary_msg_type"] = ros_topics["/joint_states"]
            contract["arm_feedback"]["primary_topic"] = "/joint_states"
            contract["arm_feedback"]["primary_msg_type"] = ros_topics["/joint_states"]
            confirmed.append((
                "base_feedback.primary_topic",
                f"/joint_states [{ros_topics['/joint_states']}]",
                "Confirmed in live ROS 2 graph (/mx_robot_state absent)",
            ))

        # Arm concurrency command topic (/mx_robot_cmd vs /mx_mix_command).
        rcmd_info = ros_details.get("/mx_robot_cmd")
        mix_info = ros_details.get("/mx_mix_command")
        rcmd_mc_sub = bool(rcmd_info and any("MCControlNode" in n for n in rcmd_info.subscriber_nodes))
        mix_mc_sub = bool(mix_info and any("MCControlNode" in n for n in mix_info.subscriber_nodes))
        if rcmd_mc_sub and not mix_mc_sub:
            contract["arm_concurrency_command"].update({
                "protocol": "mech_unit_value_array",
                "topic": "/mx_robot_cmd",
                "msg_type": "mc_core_interface/msg/MechUnitValueArray",
                "mech_unit_name": "right_arm",
                "mode": 0,
            })
            confirmed.append((
                "arm_concurrency_command",
                "protocol=mech_unit_value_array, topic=/mx_robot_cmd "
                "[mc_core_interface/msg/MechUnitValueArray, name='right_arm', mode=0]",
                "Confirmed MCControlNode subscription in live ROS 2 graph (use_new_robot_cmd=true)",
            ))
        elif mix_mc_sub and not rcmd_mc_sub:
            contract["arm_concurrency_command"].update({
                "protocol": "mech_unit_cmd_array",
                "topic": "/mx_mix_command",
                "msg_type": "mc_core_interface/msg/MechUnitCmdArray",
                "right_arm_mu_idx": 3,
            })
            confirmed.append((
                "arm_concurrency_command",
                "protocol=mech_unit_cmd_array, topic=/mx_mix_command "
                "[mc_core_interface/msg/MechUnitCmdArray, mu_idx=3]",
                "Confirmed MCControlNode subscription in live ROS 2 graph (use_new_robot_cmd=false)",
            ))
        else:
            unresolved_reasons["arm_concurrency_command.protocol"] = (
                "Could not uniquely determine whether MCControlNode subscribes to "
                "/mx_robot_cmd or /mx_mix_command."
            )

        if "/imu_states" in ros_topics:
            contract["imu_feedback"]["topic"] = "/imu_states"
            contract["imu_feedback"]["msg_type"] = ros_topics["/imu_states"]
            confirmed.append((
                "imu_feedback.topic",
                f"/imu_states [{ros_topics['/imu_states']}]",
                "Confirmed topic advertisement (MCMessageNode; optional if base_imu serial offline)",
            ))
    else:
        # Repository-only fallback: do NOT guess base command topic!
        unresolved_reasons["base_command.protocol"] = (
            "No base control topic exists in pingpong_moz1 repo and live ROS 2 graph was not "
            "reachable. Set 'vector3_vel' (/mx_base_vel_command) or "
            "'mech_unit_value_array' (/mx_base_cmd, mode=1, twist)."
        )
        unresolved_reasons["base_feedback.primary_topic"] = (
            "Live ROS 2 graph unavailable; confirm '/mx_robot_state' (200 Hz) vs '/joint_states'."
        )
        if repo["pingpong_command_topic"]:
            advisories.append(
                f"Repository pingpong_node.py defaults to command_topic="
                f"{repo['pingpong_command_topic']!r}, but live robot may set "
                "use_new_robot_cmd=true (/mx_robot_cmd); verify live ROS graph before s5."
            )

    # Site-specific physical verification items always require explicit operator confirmation.
    default_site_reasons = {
        "operator_site_verification.base_plus_x_physical_direction": (
            "Operator must confirm physical travel direction of +vx on the robot floor "
            "(e.g., 'forward' or 'left' matching moz1_pd.xml)."
        ),
        "operator_site_verification.base_plus_y_physical_direction": (
            "Operator must confirm physical travel direction of +vy on the robot floor."
        ),
        "operator_site_verification.base_plus_yaw_ccw_from_top_verified": (
            "Operator must confirm +wz rotates CCW viewed from above (right-hand +Z)."
        ),
        "operator_site_verification.controller_base_vel_override_verified": (
            "Operator must confirm actual motion.yaml base_vel_override / base_acc_override "
            "on 172.16.0.20 (repo copy has 0.1)."
        ),
        "operator_site_verification.teleop_publisher_conflict_cleared": (
            "Operator must confirm MozTeleopRobotNode is not sending conflicting base commands."
        ),
    }
    unresolved_reasons.update(default_site_reasons)

    unresolved_paths = find_unresolved_fields(contract, ignore_prefixes=("mocap",))
    unresolved_items = [
        (path, unresolved_reasons.get(path, "Must be verified and filled by operator."))
        for path in unresolved_paths
    ]

    return InspectionResult(
        contract=contract,
        confirmed_items=confirmed,
        unresolved_items=unresolved_items,
        advisories=advisories,
        ros_snapshot_available=ros_ok,
        ros_topics=ros_topics,
        ros_topic_details=ros_details,
    )


def format_inspection_report(result: InspectionResult) -> str:
    """Format a human-readable Markdown audit report."""
    lines: List[str] = [
        "# MOZ1 Base-ID Read-Only Interface Audit Report",
        "",
        f"- Generated (UTC): `{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}`",
        f"- Live ROS 2 snapshot available: `{result.ros_snapshot_available}`",
        f"- Confirmed interface entries: `{len(result.confirmed_items)}`",
        f"- Unresolved (`{UNRESOLVED_SENTINEL}`) entries: `{len(result.unresolved_items)}`",
        "",
        "## 1. Confirmed Interfaces",
        "",
    ]
    if result.confirmed_items:
        for path, val, evidence in result.confirmed_items:
            lines.append(f"- **`{path}`** = `{val}`")
            lines.append(f"  - Evidence: {evidence}")
    else:
        lines.append("- _(None confirmed)_")

    lines.extend(["", f"## 2. Unresolved Fields (`{UNRESOLVED_SENTINEL}`)", ""])
    if result.unresolved_items:
        for path, reason in result.unresolved_items:
            lines.append(f"- **`{path}`**: {reason}")
    else:
        lines.append("- _(None — all required fields resolved)_")

    lines.extend(["", "## 3. Advisories & Safety Notes", ""])
    if result.advisories:
        for note in result.advisories:
            lines.append(f"- {note}")
    else:
        lines.append("- _(None)_")

    if result.ros_topic_details:
        lines.extend(["", "## 4. Live ROS 2 Topic Endpoint Snapshot (Read-Only)", ""])
        for topic, info in sorted(result.ros_topic_details.items()):
            lines.append(
                f"- `{topic}` `[{info.msg_type}]`: "
                f"pubs={info.publisher_count} {info.publisher_nodes}, "
                f"subs={info.subscription_count} {info.subscriber_nodes}, "
                f"qos={info.qos_reliabilities}"
            )
    return "\n".join(lines) + "\n"


def write_contract_yaml(contract: Mapping[str, Any], out_path: Path) -> None:
    """Serialize contract mapping to YAML with clear header comments."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# MOZ1 Base Dynamics Identification Interface Contract\n"
        "# Generated by inspect_base_interface.py (read-only audit).\n"
        "# Fill every UNRESOLVED_REQUIRE_OPERATOR_FILL entry (or pass --set overrides)\n"
        "# before running run_base_id_experiment.py.\n\n"
    )
    body = yaml.safe_dump(
        copy.deepcopy(dict(contract)),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    out_path.write_text(header + body, encoding="utf-8")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only ROS 2 and repository interface auditor for MOZ1 base ID."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root directory (auto-detected if omitted).",
    )
    parser.add_argument(
        "--no-ros-snapshot",
        action="store_true",
        help="Skip read-only `ros2 topic` queries and scan repository files only.",
    )
    parser.add_argument(
        "--output-contract",
        type=Path,
        default=default_contract_path(),
        help="Destination path for generated base_id_interface_contract.yaml.",
    )
    parser.add_argument(
        "--no-write-contract",
        action="store_true",
        help="Print audit report only without writing the YAML contract file.",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=None,
        help="Optional file path to save the Markdown audit report.",
    )
    args = parser.parse_args(argv)

    result = inspect_base_interface(
        repo_root=args.repo_root,
        enable_ros_snapshot=not args.no_ros_snapshot,
    )
    report = format_inspection_report(result)
    print(report)

    if not args.no_write_contract:
        write_contract_yaml(result.contract, args.output_contract)
        print(f"[inspect_base_interface] Wrote contract template to: {args.output_contract}")
    if args.output_report is not None:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(report, encoding="utf-8")
        print(f"[inspect_base_interface] Wrote audit report to: {args.output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
