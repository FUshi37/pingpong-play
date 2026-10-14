#!/usr/bin/env python3
"""Headless MuJoCo evaluation for the autonomous racket-juggle MPC.

The script runs the same high-level state machine used by the interactive
probe, then applies the deployment actuator path:

    nominal joint command -> inverse compensation -> delay/first-order actuator
    model -> MuJoCo position actuators.

It is intentionally command-line driven so parameter sweeps can be reproduced.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco as mj
import numpy as np


def _repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh").is_file():
            return parent
    return path.parents[4]


REPO_ROOT = _repo_root()
for module_root in (
        REPO_ROOT / "src" / "pingpong_moz1",
        REPO_ROOT / "src" / "pingpong_replay"):
    module_text = str(module_root)
    if module_text not in sys.path:
        sys.path.insert(0, module_text)

from pingpong_controller.racket_ball_tracker import (  # noqa: E402
    RacketBallTracker,
    RacketBallTrackerConfig,
    RacketMpcTuning,
)
from pingpong_controller.racket_juggle_mpc import (  # noqa: E402
    JUGGLE_INITIAL_Q_RAD,
    JugglePhysicsConfig,
    RacketJuggleMpc,
    racket_normal_from_quaternion_xyzw,
)
from pingpong_controller.rl_policy import (  # noqa: E402
    DEFAULT_ARM_ACC_LIMIT_DEG_S2,
    DEFAULT_ARM_VEL_LIMIT_DEG_S,
    RIGHT_ARM_JOINTS,
    TARGET_DEGREES,
)
from pingpong_replay.compensation import CommandCompensator  # noqa: E402
from pingpong_replay.delay_control import (  # noqa: E402
    command_buffer_length,
    delay_steps_from_tau,
    push_command_buffer,
)


DEFAULT_SIM_XML = (
    REPO_ROOT
    / "src"
    / "pingpong_moz1"
    / "pingpong_controller"
    / "tools"
    / "rl_sim"
    / "moz1_pd.xml"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "src"
    / "pingpong_moz1"
    / "pingpong_controller"
    / "outputs"
    / "mpc_juggle_mujoco"
)


@dataclass(frozen=True)
class EvalConfig:
    sim_xml_path: Path
    output_root: Path
    steps: int = 1200
    control_rate_hz: float = 200.0
    start_delay_s: float = 0.20
    ball_surface_gap_m: float = 0.007
    robot_settle_s: float = 0.70
    render_fps: float = 60.0
    render_width: int = 960
    render_height: int = 540
    record_video: bool = True
    allow_fallback_video: bool = True
    contact_ball_slide_friction: float = 0.20
    contact_racket_slide_friction: float = 0.35
    contact_spin_friction: float = 0.001
    contact_roll_friction: float = 0.0001
    contact_solref_time_s: float = 0.003
    contact_solref_damping: float = 0.80
    contact_margin_m: float = 0.001
    contact_condim: int = 3
    contact_racket_radius_m: float = 0.0
    sim_ball_drag_coefficient_per_m: float = 0.0
    enable_compensation: bool = True
    compensation_mode: str = "inverse_mpc"
    comp_delay_ms: float = 72.0
    comp_actuator_tau_s: float = 0.074
    comp_actuator_gain: float = 1.0
    comp_buffer_extra_steps: int = 4
    comp_delay_conditioning: bool = True
    comp_mpc_beta: float = 1.2
    comp_mpc_delay_scale: float = 1.05
    comp_mpc_tau_scale: float = 0.75
    comp_mpc_horizon_steps: int = 6
    comp_mpc_tracking_weight: float = 1.0
    comp_mpc_nominal_weight: float = 0.25
    comp_mpc_delta_weight: float = 0.05
    comp_mpc_max_delta_rad: float = 0.5235987755982988
    mpc_horizon_steps: int = 24
    mpc_position_weight: float = 1.0
    mpc_velocity_weight: float = 1.0
    mpc_effort_weight: float = 1.0e-4
    mpc_smooth_weight: float = 2.0e-4
    mpc_posture_weight: float = 1.0e-4
    max_cartesian_speed_m_s: float = 0.65
    max_cartesian_acceleration_m_s2: float = 16.0
    joint_velocity_scale: float = 0.25
    joint_acceleration_scale: float = 0.35
    posture_gain_s: float = 0.12
    orientation_gain_s: float = 0.0
    mpc_orientation_weight: float = 0.0
    target_boundary_margin_m: float = 0.005
    workspace_x_min_m: float = -0.298
    workspace_x_max_m: float = 0.163
    workspace_y_min_m: float = -0.502
    workspace_y_max_m: float = -0.290
    workspace_z_min_m: float = 0.692
    workspace_z_max_m: float = 1.462
    juggle_countdown_s: float = 0.0
    juggle_target_apex_height_m: float = 0.22
    juggle_seed_apex_height_m: float = 0.018
    juggle_apex_ramp_per_bounce_m: float = 0.030
    juggle_launch_peak_acceleration_m_s2: float = 16.0
    juggle_restitution: float = 0.58
    juggle_tangential_transfer: float = 0.20
    juggle_lateral_correction_gain: float = 1.0
    juggle_drag_coefficient_per_m: float = 0.0
    juggle_ball_radius_m: float = 0.020
    juggle_ready_xy_tolerance_m: float = 0.045
    juggle_ready_height_tolerance_m: float = 0.014
    juggle_ready_max_ball_speed_m_s: float = 0.12
    juggle_ready_max_racket_speed_m_s: float = 0.05
    juggle_ready_joint_tolerance_deg: float = 5.0
    juggle_ready_settle_s: float = 0.05
    juggle_max_ball_age_s: float = 0.08
    juggle_max_prediction_time_s: float = 1.0
    juggle_min_impact_lead_s: float = 0.035
    juggle_impact_refractory_s: float = 0.08
    juggle_contact_grace_s: float = 0.06
    juggle_contact_wait_lead_s: float = 0.0
    juggle_contact_wait_live_target_blend: float = 0.0
    juggle_min_bounce_speed_m_s: float = 0.12
    juggle_max_lateral_rebound_speed_m_s: float = 0.60
    juggle_control_normal_world_up_blend: float = 0.0
    juggle_allow_infeasible_mpc_fallback: bool = False
    juggle_intercept_lateral_score_weight: float = 0.0
    juggle_return_center_current_racket_blend: float = 0.0
    juggle_mpc_terminal_position_tolerance_m: float = 0.0
    juggle_mpc_terminal_velocity_tolerance_m_s: float = 0.05

    @property
    def dt(self) -> float:
        return 1.0 / max(float(self.control_rate_hz), 1.0e-9)

    @property
    def position_limits(self) -> tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float]]:
        return (
            (float(self.workspace_x_min_m), float(self.workspace_x_max_m)),
            (float(self.workspace_y_min_m), float(self.workspace_y_max_m)),
            (float(self.workspace_z_min_m), float(self.workspace_z_max_m)),
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _patch_xml_with_ball(xml_path: Path) -> Path:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    meshdir = compiler.get("meshdir")
    if meshdir:
        meshdir_path = Path(meshdir).expanduser()
        if meshdir_path.is_absolute():
            compiler.set("meshdir", str(meshdir_path))
        else:
            compiler.set("meshdir", str((xml_path.parent / meshdir).resolve()))
    else:
        compiler.set("meshdir", str((xml_path.parent / "meshes").resolve()))

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    ball_body = None
    for body in worldbody.findall("body"):
        if body.get("name") == "pingpong_ball":
            ball_body = body
            break
    if ball_body is None:
        ball_body = ET.SubElement(
            worldbody, "body", name="pingpong_ball", pos="0 0 1")
        ET.SubElement(ball_body, "freejoint", name="ball_free")
    if not any(geom.get("name") == "ball" for geom in ball_body.findall("geom")):
        ET.SubElement(
            ball_body,
            "geom",
            name="ball",
            type="sphere",
            size="0.02",
            mass="0.0027",
            rgba="1.0 0.55 0.1 1",
            contype="1",
            conaffinity="1",
            friction="0.20 0.001 0.0001",
            solref="0.003 0.80",
            solimp="0.90 0.995 0.001",
            margin="0.001",
            condim="3",
        )

    for body in worldbody.iter("body"):
        if body.get("name") in {"head21", "head22", "head23"}:
            for geom in body.findall("geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")

    tmp = tempfile.NamedTemporaryFile(
        prefix="mpc_juggle_mujoco_", suffix=".xml", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    tree.write(tmp_path, encoding="utf-8", xml_declaration=False)
    return tmp_path


def _ids(model: mj.MjModel, kind: mj.mjtObj, names: list[str]) -> list[int]:
    out = [mj.mj_name2id(model, kind, name) for name in names]
    missing = [name for name, idx in zip(names, out) if idx < 0]
    if missing:
        raise RuntimeError(f"MuJoCo XML missing names: {missing}")
    return out


def _quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return np.asarray([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _racket_pose(data: mj.MjData, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray(data.site_xpos[site_id], dtype=np.float64).reshape(3)
    mat = np.asarray(data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    quat_wxyz = np.zeros(4, dtype=np.float64)
    mj.mju_mat2Quat(quat_wxyz, mat.reshape(-1))
    return pos, _quat_wxyz_to_xyzw(quat_wxyz)


def _set_free_ball_state(
    model: mj.MjModel,
    data: mj.MjData,
    ball_qadr: int,
    ball_vadr: int,
    position_m: np.ndarray,
    velocity_m_s: np.ndarray,
) -> None:
    data.qpos[ball_qadr:ball_qadr + 3] = np.asarray(
        position_m, dtype=np.float64).reshape(3)
    data.qpos[ball_qadr + 3:ball_qadr + 7] = np.asarray(
        [1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    data.qvel[ball_vadr:ball_vadr + 3] = np.asarray(
        velocity_m_s, dtype=np.float64).reshape(3)
    data.qvel[ball_vadr + 3:ball_vadr + 6] = 0.0
    mj.mj_forward(model, data)


def _make_default_ctrl(model: mj.MjModel) -> np.ndarray:
    ctrl = np.zeros(model.nu, dtype=np.float64)
    for name, deg in TARGET_DEGREES.items():
        aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            ctrl[aid] = float(np.deg2rad(deg))
    for name in ("Base-X", "Base-Y", "Base-Yaw"):
        aid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, name)
        if aid >= 0:
            ctrl[aid] = 0.0
    return ctrl


def _seed_named_joint_qpos(model: mj.MjModel, data: mj.MjData) -> None:
    for name, deg in TARGET_DEGREES.items():
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            data.qpos[int(model.jnt_qposadr[jid])] = float(np.deg2rad(deg))


def _joint_ranges(model: mj.MjModel, joint_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    lower = []
    upper = []
    for jid in joint_ids:
        if bool(model.jnt_limited[jid]):
            lower.append(float(model.jnt_range[jid, 0]))
            upper.append(float(model.jnt_range[jid, 1]))
        else:
            lower.append(-math.pi)
            upper.append(math.pi)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


class MuJoCoJuggleHarness:
    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg
        self.xml_path = _patch_xml_with_ball(Path(cfg.sim_xml_path).resolve())
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)
        self.fk_data = mj.MjData(self.model)
        self.dt = float(cfg.dt)
        self.substeps = max(
            1, int(round(self.dt / max(float(self.model.opt.timestep), 1.0e-9))))

        self.arm_jids = _ids(self.model, mj.mjtObj.mjOBJ_JOINT, RIGHT_ARM_JOINTS)
        self.arm_aids = _ids(self.model, mj.mjtObj.mjOBJ_ACTUATOR, RIGHT_ARM_JOINTS)
        self.arm_qadr = np.asarray(
            [int(self.model.jnt_qposadr[jid]) for jid in self.arm_jids],
            dtype=np.int32,
        )
        self.arm_vadr = np.asarray(
            [int(self.model.jnt_dofadr[jid]) for jid in self.arm_jids],
            dtype=np.int32,
        )
        self.racket_site_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_SITE, "right_ee_site")
        self.racket_geom_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_GEOM, "racket_rubber_fore")
        self.ball_geom_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_GEOM, "ball")
        self.ball_joint_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_JOINT, "ball_free")
        if min(
                self.racket_site_id, self.racket_geom_id,
                self.ball_geom_id, self.ball_joint_id) < 0:
            raise RuntimeError("patched XML is missing racket site/geom or ball")
        self.ball_qadr = int(self.model.jnt_qposadr[self.ball_joint_id])
        self.ball_vadr = int(self.model.jnt_dofadr[self.ball_joint_id])
        self._apply_contact_parameters()

        self.default_ctrl = _make_default_ctrl(self.model)
        self.joint_lower, self.joint_upper = _joint_ranges(
            self.model, self.arm_jids)
        self.joint_velocity_limit = np.deg2rad(
            np.asarray(DEFAULT_ARM_VEL_LIMIT_DEG_S, dtype=np.float64))
        self.joint_acceleration_limit = np.deg2rad(
            np.asarray(DEFAULT_ARM_ACC_LIMIT_DEG_S2, dtype=np.float64))

        tracker_cfg = RacketBallTrackerConfig(
            dt=self.dt,
            position_gain_s=4.0,
            dls_damping=0.04,
            max_cartesian_speed_m_s=float(cfg.max_cartesian_speed_m_s),
            max_cartesian_acceleration_m_s2=float(
                cfg.max_cartesian_acceleration_m_s2),
            joint_velocity_scale=float(cfg.joint_velocity_scale),
            joint_acceleration_scale=float(cfg.joint_acceleration_scale),
            posture_gain_s=float(cfg.posture_gain_s),
            orientation_gain_s=float(cfg.orientation_gain_s),
            mpc_horizon_steps=int(cfg.mpc_horizon_steps),
            mpc_position_weight=float(cfg.mpc_position_weight),
            mpc_target_velocity_weight=float(cfg.mpc_velocity_weight),
            mpc_velocity_weight=float(cfg.mpc_effort_weight),
            mpc_smooth_weight=float(cfg.mpc_smooth_weight),
            mpc_posture_weight=float(cfg.mpc_posture_weight),
            mpc_orientation_weight=float(cfg.mpc_orientation_weight),
            target_boundary_margin_m=float(cfg.target_boundary_margin_m),
        )
        self.tuning = RacketMpcTuning(
            horizon_steps=int(cfg.mpc_horizon_steps),
            position_weight=float(cfg.mpc_position_weight),
            velocity_weight=float(cfg.mpc_velocity_weight),
            effort_weight=float(cfg.mpc_effort_weight),
            smooth_weight=float(cfg.mpc_smooth_weight),
            posture_weight=float(cfg.mpc_posture_weight),
            orientation_weight=float(cfg.mpc_orientation_weight),
            orientation_gain_s=float(cfg.orientation_gain_s),
            max_cartesian_speed_m_s=float(cfg.max_cartesian_speed_m_s),
            max_cartesian_acceleration_m_s2=float(
                cfg.max_cartesian_acceleration_m_s2),
            joint_velocity_scale=float(cfg.joint_velocity_scale),
            joint_acceleration_scale=float(cfg.joint_acceleration_scale),
        )
        self.tracker = RacketBallTracker(
            kinematics=self._kinematics,
            nominal_q_rad=JUGGLE_INITIAL_Q_RAD,
            joint_lower_rad=self.joint_lower,
            joint_upper_rad=self.joint_upper,
            joint_velocity_limit_rad_s=self.joint_velocity_limit,
            joint_acceleration_limit_rad_s2=self.joint_acceleration_limit,
            config=tracker_cfg,
        )
        self.juggler = RacketJuggleMpc(
            control_dt_s=self.dt,
            initial_q_rad=JUGGLE_INITIAL_Q_RAD,
        )
        self.physics = JugglePhysicsConfig(
            countdown_s=float(cfg.juggle_countdown_s),
            target_apex_height_m=float(cfg.juggle_target_apex_height_m),
            seed_apex_height_m=float(cfg.juggle_seed_apex_height_m),
            apex_ramp_per_bounce_m=float(cfg.juggle_apex_ramp_per_bounce_m),
            launch_peak_acceleration_m_s2=float(
                cfg.juggle_launch_peak_acceleration_m_s2),
            restitution=float(cfg.juggle_restitution),
            tangential_transfer=float(cfg.juggle_tangential_transfer),
            lateral_correction_gain=float(cfg.juggle_lateral_correction_gain),
            drag_coefficient_per_m=float(cfg.juggle_drag_coefficient_per_m),
            ball_radius_m=float(cfg.juggle_ball_radius_m),
            ready_xy_tolerance_m=float(cfg.juggle_ready_xy_tolerance_m),
            ready_height_tolerance_m=float(cfg.juggle_ready_height_tolerance_m),
            ready_max_ball_speed_m_s=float(cfg.juggle_ready_max_ball_speed_m_s),
            ready_max_racket_speed_m_s=float(
                cfg.juggle_ready_max_racket_speed_m_s),
            ready_joint_tolerance_rad=float(np.deg2rad(
                cfg.juggle_ready_joint_tolerance_deg)),
            ready_settle_s=float(cfg.juggle_ready_settle_s),
            max_ball_age_s=float(cfg.juggle_max_ball_age_s),
            max_prediction_time_s=float(cfg.juggle_max_prediction_time_s),
            min_impact_lead_s=float(cfg.juggle_min_impact_lead_s),
            impact_refractory_s=float(cfg.juggle_impact_refractory_s),
            contact_grace_s=float(cfg.juggle_contact_grace_s),
            contact_wait_lead_s=float(cfg.juggle_contact_wait_lead_s),
            contact_wait_live_target_blend=float(
                cfg.juggle_contact_wait_live_target_blend),
            min_bounce_speed_m_s=float(cfg.juggle_min_bounce_speed_m_s),
            max_lateral_rebound_speed_m_s=float(
                cfg.juggle_max_lateral_rebound_speed_m_s),
            control_normal_world_up_blend=float(
                cfg.juggle_control_normal_world_up_blend),
            allow_infeasible_mpc_fallback=bool(
                cfg.juggle_allow_infeasible_mpc_fallback),
            intercept_lateral_score_weight=float(
                cfg.juggle_intercept_lateral_score_weight),
            return_center_current_racket_blend=float(
                cfg.juggle_return_center_current_racket_blend),
            mpc_terminal_position_tolerance_m=float(
                cfg.juggle_mpc_terminal_position_tolerance_m),
            mpc_terminal_velocity_tolerance_m_s=float(
                cfg.juggle_mpc_terminal_velocity_tolerance_m_s),
        )
        self.compensator = None
        if bool(cfg.enable_compensation):
            self.compensator = CommandCompensator(
                n_joints=len(RIGHT_ARM_JOINTS),
                dt=self.dt,
                mode=str(cfg.compensation_mode),
                delay_ms=float(cfg.comp_delay_ms),
                actuator_tau_s=float(cfg.comp_actuator_tau_s),
                actuator_gain=float(cfg.comp_actuator_gain),
                command_buffer_extra_steps=int(cfg.comp_buffer_extra_steps),
                delay_conditioning=bool(cfg.comp_delay_conditioning),
                mpc_beta=float(cfg.comp_mpc_beta),
                mpc_delay_scale=float(cfg.comp_mpc_delay_scale),
                mpc_tau_scale=float(cfg.comp_mpc_tau_scale),
                mpc_horizon_steps=int(cfg.comp_mpc_horizon_steps),
                mpc_tracking_weight=float(cfg.comp_mpc_tracking_weight),
                mpc_nominal_weight=float(cfg.comp_mpc_nominal_weight),
                mpc_delta_weight=float(cfg.comp_mpc_delta_weight),
                mpc_max_delta_rad=float(cfg.comp_mpc_max_delta_rad),
                q_low=self.joint_lower,
                q_high=self.joint_upper,
                warm_q=JUGGLE_INITIAL_Q_RAD,
            )

        self.command_buffer_len = command_buffer_length(
            float(cfg.comp_delay_ms), self.dt, int(cfg.comp_buffer_extra_steps))
        self.delay_steps = int(np.clip(
            delay_steps_from_tau(float(cfg.comp_delay_ms) * 1.0e-3, self.dt),
            0,
            self.command_buffer_len - 1,
        ))
        self.command_buffer: np.ndarray | None = None
        self.actuator_command_buffer: np.ndarray | None = None
        self.arm_applied_q: np.ndarray | None = None
        self.previous_nominal_dq = np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float64)
        self.fk_seed_qpos = np.zeros(self.model.nq, dtype=np.float64)
        self._reset()

    def _apply_contact_parameters(self) -> None:
        condim = int(np.clip(int(self.cfg.contact_condim), 1, 6))
        solref = np.asarray([
            max(1.0e-5, float(self.cfg.contact_solref_time_s)),
            max(0.0, float(self.cfg.contact_solref_damping)),
        ], dtype=np.float64)
        margin = max(0.0, float(self.cfg.contact_margin_m))
        spin = max(0.0, float(self.cfg.contact_spin_friction))
        roll = max(0.0, float(self.cfg.contact_roll_friction))
        for geom_id, slide in (
                (self.ball_geom_id, self.cfg.contact_ball_slide_friction),
                (self.racket_geom_id, self.cfg.contact_racket_slide_friction),
        ):
            self.model.geom_friction[geom_id, :] = np.asarray([
                max(0.0, float(slide)),
                spin,
                roll,
            ], dtype=np.float64)
            self.model.geom_solref[geom_id, :] = solref
            self.model.geom_margin[geom_id] = margin
            self.model.geom_condim[geom_id] = condim
        if float(self.cfg.contact_racket_radius_m) > 0.0:
            self.model.geom_size[self.racket_geom_id, 0] = float(
                self.cfg.contact_racket_radius_m)

    def _apply_ball_drag_substep(self) -> None:
        drag = max(0.0, float(self.cfg.sim_ball_drag_coefficient_per_m))
        if drag <= 0.0:
            return
        velocity = self.data.qvel[self.ball_vadr:self.ball_vadr + 3]
        speed = float(np.linalg.norm(velocity))
        if speed <= 1.0e-12 or not np.isfinite(speed):
            return
        dt = float(self.model.opt.timestep)
        velocity[:] = velocity / (1.0 + drag * speed * dt)

    def _reset(self) -> None:
        self.data = mj.MjData(self.model)
        self.data.ctrl[:] = self.default_ctrl
        _seed_named_joint_qpos(self.model, self.data)
        for i, (qadr, aid) in enumerate(zip(self.arm_qadr, self.arm_aids)):
            self.data.qpos[qadr] = float(JUGGLE_INITIAL_Q_RAD[i])
            self.data.ctrl[aid] = float(JUGGLE_INITIAL_Q_RAD[i])
        _set_free_ball_state(
            self.model,
            self.data,
            self.ball_qadr,
            self.ball_vadr,
            np.asarray([10.0, 10.0, 2.0], dtype=np.float64),
            np.zeros(3, dtype=np.float64),
        )
        settle_steps = max(0, int(round(float(self.cfg.robot_settle_s) / self.dt)))
        for _ in range(settle_steps):
            self.data.ctrl[:] = self.default_ctrl
            for i, aid in enumerate(self.arm_aids):
                self.data.ctrl[aid] = float(JUGGLE_INITIAL_Q_RAD[i])
            for _sub in range(self.substeps):
                mj.mj_step(self.model, self.data)
                self._apply_ball_drag_substep()
        mj.mj_forward(self.model, self.data)

        q0 = self.data.qpos[self.arm_qadr].astype(np.float64)
        self.tracker.reset(q0, np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float64))
        self.juggler.enter_mode()
        if self.compensator is not None:
            self.compensator.reset(q0.astype(np.float32))

        shape = (self.command_buffer_len, len(RIGHT_ARM_JOINTS))
        self.command_buffer = np.broadcast_to(q0, shape).copy()
        self.actuator_command_buffer = np.broadcast_to(q0, shape).copy()
        self.arm_applied_q = q0.copy()
        self.previous_nominal_dq = np.zeros(
            len(RIGHT_ARM_JOINTS), dtype=np.float64)

        racket, quat = _racket_pose(self.data, self.racket_site_id)
        normal = racket_normal_from_quaternion_xyzw(quat)
        initial_ball = (
            racket
            + (float(self.cfg.juggle_ball_radius_m)
               + float(self.cfg.ball_surface_gap_m)) * normal)
        _set_free_ball_state(
            self.model,
            self.data,
            self.ball_qadr,
            self.ball_vadr,
            initial_ball,
            np.zeros(3, dtype=np.float64),
        )
        self.fk_seed_qpos = self.data.qpos.copy()

    def _kinematics(
        self, arm_q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        q = np.asarray(arm_q, dtype=np.float64).reshape(len(RIGHT_ARM_JOINTS))
        if not np.all(np.isfinite(q)):
            zeros = np.zeros(3, dtype=np.float64)
            zeros_j = np.zeros((3, len(RIGHT_ARM_JOINTS)), dtype=np.float64)
            return zeros, np.asarray([0.0, 0.0, 0.0, 1.0]), zeros_j, zeros_j, False
        self.fk_data.qpos[:] = self.fk_seed_qpos
        self.fk_data.qvel[:] = 0.0
        for i, qadr in enumerate(self.arm_qadr):
            self.fk_data.qpos[qadr] = float(q[i])
        mj.mj_forward(self.model, self.fk_data)
        pos, quat = _racket_pose(self.fk_data, self.racket_site_id)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros_like(jacp)
        mj.mj_jacSite(self.model, self.fk_data, jacp, jacr, self.racket_site_id)
        cols = [int(self.model.jnt_dofadr[jid]) for jid in self.arm_jids]
        return pos, quat, jacp[:, cols].copy(), jacr[:, cols].copy(), True

    def _contact_active(self) -> bool:
        pair = {int(self.ball_geom_id), int(self.racket_geom_id)}
        for idx in range(int(self.data.ncon)):
            contact = self.data.contact[idx]
            if {int(contact.geom1), int(contact.geom2)} == pair:
                return True
        return False

    def _run_actuator_model(
        self,
        q_nominal: np.ndarray,
        q_actuator_latest: np.ndarray,
        q_feedback: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.command_buffer is not None
        assert self.actuator_command_buffer is not None
        assert self.arm_applied_q is not None
        delay_steps = (
            self.delay_steps if bool(self.cfg.comp_delay_conditioning) else 0)
        self.command_buffer, q_ref_active = push_command_buffer(
            self.command_buffer,
            np.asarray(q_nominal, dtype=np.float64).reshape(len(RIGHT_ARM_JOINTS)),
            delay_steps,
        )
        self.actuator_command_buffer, q_actuator_active = push_command_buffer(
            self.actuator_command_buffer,
            np.asarray(q_actuator_latest, dtype=np.float64).reshape(
                len(RIGHT_ARM_JOINTS)),
            delay_steps,
        )
        tau_s = max(0.0, float(self.cfg.comp_actuator_tau_s))
        gain_target = (
            JUGGLE_INITIAL_Q_RAD
            + float(self.cfg.comp_actuator_gain)
            * (q_actuator_active - JUGGLE_INITIAL_Q_RAD))
        if tau_s <= 1.0e-6:
            applied = gain_target.astype(np.float64)
        else:
            alpha = self.dt / (tau_s + self.dt)
            applied = self.arm_applied_q + alpha * (gain_target - self.arm_applied_q)
        applied = np.clip(applied, self.joint_lower, self.joint_upper)
        if not np.all(np.isfinite(applied)):
            applied = np.asarray(q_feedback, dtype=np.float64).copy()
        self.arm_applied_q = applied.astype(np.float64)
        return q_ref_active, q_actuator_active, self.arm_applied_q.copy()

    def step_control(self, now_s: float, run_requested: bool, nonce: int) -> dict[str, Any]:
        self.fk_seed_qpos = self.data.qpos.copy()
        q = self.data.qpos[self.arm_qadr].astype(np.float64)
        dq = self.data.qvel[self.arm_vadr].astype(np.float64)
        ball_pos = self.data.qpos[self.ball_qadr:self.ball_qadr + 3].astype(np.float64)
        ball_vel = self.data.qvel[self.ball_vadr:self.ball_vadr + 3].astype(np.float64)

        racket, quat = _racket_pose(self.data, self.racket_site_id)
        normal = racket_normal_from_quaternion_xyzw(quat)
        _pos, _quat, jacp, _jacr, ok = self._kinematics(q)
        if not ok:
            raise RuntimeError("racket FK/Jacobian unavailable")
        racket_vel = jacp @ dq
        available_vertical_acceleration = float(np.sum(
            np.abs(normal @ jacp)
            * self.joint_acceleration_limit
            * float(self.tuning.joint_acceleration_scale)))

        outer_limits = self.tracker.inset_position_limits(self.cfg.position_limits)
        previous_juggle_state = self.juggler.state
        plan = self.juggler.step(
            wall_now_s=float(now_s),
            run_requested=bool(run_requested),
            start_nonce=int(nonce),
            feedback_q_rad=q,
            feedback_dq_rad_s=dq,
            racket_position_m=racket,
            racket_velocity_m_s=racket_vel,
            racket_normal=normal,
            ball_position_m=ball_pos,
            ball_velocity_m_s=ball_vel,
            ball_valid=True,
            ball_age_s=0.0,
            tuning=self.tuning,
            physics=self.physics,
            available_vertical_acceleration_m_s2=available_vertical_acceleration,
            position_limits_m=outer_limits,
        )

        if plan.control_kind == "joint":
            q_nominal, low_level = self.tracker.move_to_joint_position(
                feedback_q_rad=q,
                feedback_dq_rad_s=dq,
                target_q_rad=plan.joint_target_q_rad,
                mpc_tuning=self.tuning,
            )
        elif plan.control_kind == "cartesian":
            q_nominal, low_level = self.tracker.step(
                feedback_q_rad=q,
                feedback_dq_rad_s=dq,
                ball_position_m=plan.target_position_m,
                ball_velocity_m_s=plan.target_velocity_m_s,
                height_offset_m=0.0,
                use_mpc=True,
                mpc_tuning=self.tuning,
                position_limits_m=self.cfg.position_limits,
            )
        else:
            q_nominal, low_level = self.tracker.hold(
                q, reason=str(plan.diagnostics.get("state", "hold")))

        q_nominal = np.asarray(q_nominal, dtype=np.float64).reshape(
            len(RIGHT_ARM_JOINTS))
        qdot_nominal = np.asarray(self.tracker.command_dq, dtype=np.float64).reshape(
            len(RIGHT_ARM_JOINTS))
        qdd_nominal = (qdot_nominal - self.previous_nominal_dq) / self.dt
        self.previous_nominal_dq = qdot_nominal.copy()

        if self.compensator is None:
            q_actuator_latest = q_nominal.copy()
        else:
            q_actuator_latest = np.asarray(
                self.compensator.step(
                    q_cmd=q_nominal.astype(np.float32),
                    qdot_cmd=qdot_nominal.astype(np.float32),
                    qdd_cmd=qdd_nominal.astype(np.float32),
                    applied_q=q.astype(np.float32),
                ),
                dtype=np.float64,
            )
        _q_ref_active, q_actuator_active, q_applied = self._run_actuator_model(
            q_nominal, q_actuator_latest, q)

        self.data.ctrl[:] = self.default_ctrl
        for i, aid in enumerate(self.arm_aids):
            self.data.ctrl[aid] = float(q_applied[i])
        for _ in range(self.substeps):
            mj.mj_step(self.model, self.data)
            self._apply_ball_drag_substep()
        mj.mj_forward(self.model, self.data)

        next_ball = self.data.qpos[self.ball_qadr:self.ball_qadr + 3].astype(
            np.float64)
        next_ball_vel = self.data.qvel[self.ball_vadr:self.ball_vadr + 3].astype(
            np.float64)
        next_racket, next_quat = _racket_pose(self.data, self.racket_site_id)
        next_normal = racket_normal_from_quaternion_xyzw(next_quat)
        gap = float(next_normal @ (next_ball - next_racket))
        tangent = next_ball - next_racket - gap * next_normal
        return {
            "timeS": float(now_s),
            "runRequested": bool(run_requested),
            "controlKind": str(plan.control_kind),
            "state": str(plan.diagnostics.get("state", "")),
            "stateBefore": str(previous_juggle_state),
            "fault": plan.diagnostics.get("fault"),
            "ready": bool(plan.diagnostics.get("ready", False)),
            "bounceCount": int(plan.diagnostics.get("bounceCount", 0) or 0),
            "controllerBounceCount": int(self.juggler.bounce_count),
            "contactActive": self._contact_active(),
            "ball": next_ball.copy(),
            "ballVelocity": next_ball_vel.copy(),
            "racket": next_racket.copy(),
            "racketNormal": next_normal.copy(),
            "normalGapM": gap,
            "tangentDistanceM": float(np.linalg.norm(tangent)),
            "qFeedback": q.copy(),
            "dqFeedback": dq.copy(),
            "qNominal": q_nominal.copy(),
            "qActuatorLatest": q_actuator_latest.copy(),
            "qActuatorActive": np.asarray(q_actuator_active, dtype=np.float64),
            "qApplied": q_applied.copy(),
            "lowLevel": low_level,
            "juggle": dict(plan.diagnostics),
        }


class VideoRecorder:
    def __init__(self, cfg: EvalConfig, path: Path) -> None:
        self.cfg = cfg
        self.path = path
        self.writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            float(cfg.render_fps),
            (int(cfg.render_width), int(cfg.render_height)),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"could not open video writer: {path}")
        self.renderer = None
        self.renderer_failed = False
        self.camera = mj.MjvCamera()
        self.camera.azimuth = 140.0
        self.camera.elevation = -18.0
        self.camera.distance = 1.85
        self.camera.lookat[:] = np.asarray([-0.05, -0.40, 1.05], dtype=np.float64)

    def close(self) -> None:
        self.writer.release()
        if self.renderer is not None:
            self.renderer.close()

    def write(self, harness: MuJoCoJuggleHarness, sample: dict[str, Any]) -> None:
        frame = None
        if not self.renderer_failed:
            try:
                if self.renderer is None:
                    self.renderer = mj.Renderer(
                        harness.model,
                        width=int(self.cfg.render_width),
                        height=int(self.cfg.render_height),
                    )
                self.renderer.update_scene(harness.data, camera=self.camera)
                rgb = self.renderer.render()
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                self.renderer_failed = True
                if not bool(self.cfg.allow_fallback_video):
                    raise
        if frame is None:
            frame = self._fallback_frame(sample)
        self._draw_overlay(frame, sample)
        self.writer.write(frame)

    def _fallback_frame(self, sample: dict[str, Any]) -> np.ndarray:
        width = int(self.cfg.render_width)
        height = int(self.cfg.render_height)
        frame = np.full((height, width, 3), 245, dtype=np.uint8)
        x_min, x_max = -0.40, 0.25
        z_min, z_max = 0.60, 1.55

        def project(point: np.ndarray) -> tuple[int, int]:
            x = float(point[0])
            z = float(point[2])
            px = int(np.clip((x - x_min) / (x_max - x_min), 0.0, 1.0) * (width - 1))
            py = int((1.0 - np.clip((z - z_min) / (z_max - z_min), 0.0, 1.0)) * (height - 1))
            return px, py

        ball = np.asarray(sample["ball"], dtype=np.float64)
        racket = np.asarray(sample["racket"], dtype=np.float64)
        bx, bz = project(ball)
        rx, rz = project(racket)
        cv2.line(frame, (0, rz), (width - 1, rz), (215, 215, 215), 1)
        cv2.circle(frame, (rx, rz), 16, (40, 40, 180), 3)
        cv2.circle(frame, (bx, bz), 10, (0, 120, 255), -1)
        return frame

    @staticmethod
    def _draw_overlay(frame: np.ndarray, sample: dict[str, Any]) -> None:
        state = str(sample.get("state", ""))
        fault = sample.get("fault")
        lines = [
            f"t={float(sample.get('timeS', 0.0)):.3f}s "
            f"state={state} bounce={int(sample.get('controllerBounceCount', 0))}",
            f"ball z={float(np.asarray(sample.get('ball'))[2]):.3f} "
            f"vz={float(np.asarray(sample.get('ballVelocity'))[2]):.3f} "
            f"gap={float(sample.get('normalGapM', 0.0)):.3f}",
        ]
        if fault:
            lines.append(f"fault={fault}")
        for row, text in enumerate(lines):
            y = 26 + row * 24
            cv2.putText(
                frame,
                text,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                text,
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )


def _write_trace(path: Path, samples: list[dict[str, Any]]) -> None:
    fieldnames = [
        "timeS",
        "runRequested",
        "state",
        "controlKind",
        "controllerBounceCount",
        "contactActive",
        "ballX",
        "ballY",
        "ballZ",
        "ballVx",
        "ballVy",
        "ballVz",
        "racketX",
        "racketY",
        "racketZ",
        "normalGapM",
        "tangentDistanceM",
        "fault",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            ball = np.asarray(sample["ball"], dtype=np.float64)
            ball_vel = np.asarray(sample["ballVelocity"], dtype=np.float64)
            racket = np.asarray(sample["racket"], dtype=np.float64)
            writer.writerow({
                "timeS": f"{float(sample['timeS']):.6f}",
                "runRequested": int(bool(sample["runRequested"])),
                "state": sample["state"],
                "controlKind": sample["controlKind"],
                "controllerBounceCount": int(sample["controllerBounceCount"]),
                "contactActive": int(bool(sample["contactActive"])),
                "ballX": f"{ball[0]:.8f}",
                "ballY": f"{ball[1]:.8f}",
                "ballZ": f"{ball[2]:.8f}",
                "ballVx": f"{ball_vel[0]:.8f}",
                "ballVy": f"{ball_vel[1]:.8f}",
                "ballVz": f"{ball_vel[2]:.8f}",
                "racketX": f"{racket[0]:.8f}",
                "racketY": f"{racket[1]:.8f}",
                "racketZ": f"{racket[2]:.8f}",
                "normalGapM": f"{float(sample['normalGapM']):.8f}",
                "tangentDistanceM": f"{float(sample['tangentDistanceM']):.8f}",
                "fault": sample.get("fault") or "",
            })


def _count_contact_edges(samples: list[dict[str, Any]], min_gap_s: float = 0.05) -> int:
    count = 0
    last_t = -1.0e9
    prev = False
    for sample in samples:
        active = bool(sample.get("contactActive", False))
        now = float(sample.get("timeS", 0.0))
        if active and not prev and now - last_t >= min_gap_s:
            if float(np.asarray(sample.get("ballVelocity"))[2]) > 0.02:
                count += 1
                last_t = now
        prev = active
    return count


def _make_summary(
    cfg: EvalConfig,
    run_dir: Path,
    video_path: Path | None,
    trace_path: Path,
    samples: list[dict[str, Any]],
    elapsed_wall_s: float,
) -> dict[str, Any]:
    final = samples[-1] if samples else {}
    ball_z = np.asarray(
        [float(np.asarray(s["ball"])[2]) for s in samples], dtype=np.float64)
    bounce_count = int(final.get("controllerBounceCount", 0) or 0)
    return {
        "runDir": str(run_dir),
        "videoPath": None if video_path is None else str(video_path),
        "tracePath": str(trace_path),
        "steps": int(cfg.steps),
        "controlRateHz": float(cfg.control_rate_hz),
        "durationS": float(cfg.steps) * cfg.dt,
        "elapsedWallS": float(elapsed_wall_s),
        "controllerBounceCount": bounce_count,
        "contactEdgeCount": _count_contact_edges(samples),
        "targetBounceRangeHit": bool(13 <= bounce_count <= 15),
        "finalState": final.get("state"),
        "finalFault": final.get("fault"),
        "finalBall": _json_safe(final.get("ball")),
        "finalBallVelocity": _json_safe(final.get("ballVelocity")),
        "ballZMin": None if ball_z.size == 0 else float(np.min(ball_z)),
        "ballZMax": None if ball_z.size == 0 else float(np.max(ball_z)),
        "config": _json_safe(asdict(cfg)),
    }


def run_eval(cfg: EvalConfig) -> dict[str, Any]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_root).resolve() / f"mpc_juggle_mujoco_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    video_path = run_dir / "juggle_eval.mp4" if bool(cfg.record_video) else None
    trace_path = run_dir / "trace.csv"
    summary_path = run_dir / "summary.json"

    harness = MuJoCoJuggleHarness(cfg)
    recorder = VideoRecorder(cfg, video_path) if video_path is not None else None
    samples: list[dict[str, Any]] = []
    next_render_s = 0.0
    start_wall = time.time()
    try:
        for step in range(int(cfg.steps)):
            now_s = step * cfg.dt
            run_requested = now_s >= float(cfg.start_delay_s)
            nonce = 1 if run_requested else 0
            sample = harness.step_control(now_s, run_requested, nonce)
            samples.append(sample)
            if recorder is not None and now_s + 1.0e-12 >= next_render_s:
                recorder.write(harness, sample)
                next_render_s += 1.0 / max(float(cfg.render_fps), 1.0e-9)
    finally:
        if recorder is not None:
            recorder.close()

    elapsed_wall_s = time.time() - start_wall
    _write_trace(trace_path, samples)
    summary = _make_summary(
        cfg, run_dir, video_path, trace_path, samples, elapsed_wall_s)
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return summary


def _bool_arg(text: str) -> bool:
    normalized = str(text).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {text!r}")


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-xml-path", type=Path, default=DEFAULT_SIM_XML)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--control-rate-hz", type=float, default=200.0)
    parser.add_argument("--start-delay-s", type=float, default=0.20)
    parser.add_argument("--ball-surface-gap-m", type=float, default=0.007)
    parser.add_argument("--robot-settle-s", type=float, default=0.70)
    parser.add_argument("--render-fps", type=float, default=60.0)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=540)
    parser.add_argument("--record-video", type=_bool_arg, default=True)
    parser.add_argument("--allow-fallback-video", type=_bool_arg, default=True)
    parser.add_argument("--contact-ball-slide-friction", type=float, default=0.20)
    parser.add_argument("--contact-racket-slide-friction", type=float, default=0.35)
    parser.add_argument("--contact-spin-friction", type=float, default=0.001)
    parser.add_argument("--contact-roll-friction", type=float, default=0.0001)
    parser.add_argument("--contact-solref-time-s", type=float, default=0.003)
    parser.add_argument("--contact-solref-damping", type=float, default=0.80)
    parser.add_argument("--contact-margin-m", type=float, default=0.001)
    parser.add_argument("--contact-condim", type=int, default=3)
    parser.add_argument("--contact-racket-radius-m", type=float, default=0.0)
    parser.add_argument(
        "--sim-ball-drag-coefficient-per-m", type=float, default=0.0)
    parser.add_argument("--enable-compensation", type=_bool_arg, default=True)
    parser.add_argument("--compensation-mode", default="inverse_mpc")
    parser.add_argument("--comp-delay-ms", type=float, default=72.0)
    parser.add_argument("--comp-actuator-tau-s", type=float, default=0.074)
    parser.add_argument("--comp-actuator-gain", type=float, default=1.0)
    parser.add_argument("--comp-buffer-extra-steps", type=int, default=4)
    parser.add_argument("--comp-delay-conditioning", type=_bool_arg, default=True)
    parser.add_argument("--comp-mpc-beta", type=float, default=1.2)
    parser.add_argument("--comp-mpc-delay-scale", type=float, default=1.05)
    parser.add_argument("--comp-mpc-tau-scale", type=float, default=0.75)
    parser.add_argument("--comp-mpc-horizon-steps", type=int, default=6)
    parser.add_argument("--comp-mpc-tracking-weight", type=float, default=1.0)
    parser.add_argument("--comp-mpc-nominal-weight", type=float, default=0.25)
    parser.add_argument("--comp-mpc-delta-weight", type=float, default=0.05)
    parser.add_argument(
        "--comp-mpc-max-delta-rad", type=float, default=0.5235987755982988)
    parser.add_argument("--mpc-horizon-steps", type=int, default=24)
    parser.add_argument("--mpc-position-weight", type=float, default=1.0)
    parser.add_argument("--mpc-velocity-weight", type=float, default=1.0)
    parser.add_argument("--mpc-effort-weight", type=float, default=1.0e-4)
    parser.add_argument("--mpc-smooth-weight", type=float, default=2.0e-4)
    parser.add_argument("--mpc-posture-weight", type=float, default=1.0e-4)
    parser.add_argument("--max-cartesian-speed-m-s", type=float, default=0.65)
    parser.add_argument(
        "--max-cartesian-acceleration-m-s2", type=float, default=16.0)
    parser.add_argument("--joint-velocity-scale", type=float, default=0.25)
    parser.add_argument("--joint-acceleration-scale", type=float, default=0.35)
    parser.add_argument("--posture-gain-s", type=float, default=0.12)
    parser.add_argument("--orientation-gain-s", type=float, default=0.0)
    parser.add_argument("--mpc-orientation-weight", type=float, default=0.0)
    parser.add_argument("--target-boundary-margin-m", type=float, default=0.005)
    parser.add_argument("--workspace-x-min-m", type=float, default=-0.298)
    parser.add_argument("--workspace-x-max-m", type=float, default=0.163)
    parser.add_argument("--workspace-y-min-m", type=float, default=-0.502)
    parser.add_argument("--workspace-y-max-m", type=float, default=-0.290)
    parser.add_argument("--workspace-z-min-m", type=float, default=0.692)
    parser.add_argument("--workspace-z-max-m", type=float, default=1.462)
    parser.add_argument("--juggle-countdown-s", type=float, default=0.0)
    parser.add_argument("--juggle-target-apex-height-m", type=float, default=0.22)
    parser.add_argument("--juggle-seed-apex-height-m", type=float, default=0.018)
    parser.add_argument(
        "--juggle-apex-ramp-per-bounce-m", type=float, default=0.030)
    parser.add_argument(
        "--juggle-launch-peak-acceleration-m-s2", type=float, default=16.0)
    parser.add_argument("--juggle-restitution", type=float, default=0.58)
    parser.add_argument("--juggle-tangential-transfer", type=float, default=0.20)
    parser.add_argument("--juggle-lateral-correction-gain", type=float, default=1.0)
    parser.add_argument("--juggle-drag-coefficient-per-m", type=float, default=0.0)
    parser.add_argument("--juggle-ball-radius-m", type=float, default=0.020)
    parser.add_argument("--juggle-ready-xy-tolerance-m", type=float, default=0.045)
    parser.add_argument(
        "--juggle-ready-height-tolerance-m", type=float, default=0.014)
    parser.add_argument(
        "--juggle-ready-max-ball-speed-m-s", type=float, default=0.12)
    parser.add_argument(
        "--juggle-ready-max-racket-speed-m-s", type=float, default=0.05)
    parser.add_argument(
        "--juggle-ready-joint-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--juggle-ready-settle-s", type=float, default=0.05)
    parser.add_argument("--juggle-max-ball-age-s", type=float, default=0.08)
    parser.add_argument("--juggle-max-prediction-time-s", type=float, default=1.0)
    parser.add_argument("--juggle-min-impact-lead-s", type=float, default=0.035)
    parser.add_argument("--juggle-impact-refractory-s", type=float, default=0.08)
    parser.add_argument("--juggle-contact-grace-s", type=float, default=0.06)
    parser.add_argument("--juggle-contact-wait-lead-s", type=float, default=0.0)
    parser.add_argument(
        "--juggle-contact-wait-live-target-blend", type=float, default=0.0)
    parser.add_argument("--juggle-min-bounce-speed-m-s", type=float, default=0.12)
    parser.add_argument(
        "--juggle-max-lateral-rebound-speed-m-s", type=float, default=0.60)
    parser.add_argument(
        "--juggle-control-normal-world-up-blend", type=float, default=0.0)
    parser.add_argument(
        "--juggle-allow-infeasible-mpc-fallback", type=_bool_arg, default=False)
    parser.add_argument(
        "--juggle-intercept-lateral-score-weight", type=float, default=0.0)
    parser.add_argument(
        "--juggle-return-center-current-racket-blend", type=float, default=0.0)
    parser.add_argument(
        "--juggle-mpc-terminal-position-tolerance-m", type=float, default=0.0)
    parser.add_argument(
        "--juggle-mpc-terminal-velocity-tolerance-m-s", type=float, default=0.05)
    args = parser.parse_args()
    return EvalConfig(**vars(args))


def main() -> None:
    cfg = parse_args()
    run_eval(cfg)


if __name__ == "__main__":
    main()
