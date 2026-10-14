from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import gymnasium as gym
from gymnasium import spaces
import mujoco as mj
import numpy as np


@dataclass
class JuggleConfig:
    horizon_sec: float = 6.0
    frame_skip: int = 5
    action_scale_arm_rad: float = 0.03
    action_scale_base_xy: float = 0.020
    action_scale_base_yaw: float = 0.030
    ball_launch_height: float = 0.24
    ball_spawn_cube_size: float = 0.10
    ball_spawn_xy_jitter: float = 0.025
    ball_spawn_z_jitter: float = 0.035
    ball_init_vxy_max: float = 0.012
    ball_init_vz: float = -0.28
    ball_obs_rate_hz: float = 50.0
    ball_obs_pos_noise_std: float = 0.003
    ball_obs_vel_noise_std: float = 0.03
    total_training_steps: int = 10_000_000
    ball_obs_noise_warmup_ratio: float = 0.10
    ball_obs_noise_ramp_ratio: float = 0.20
    target_height: float = 0.42
    posture_weight: float = 0.85
    base_pose_weight: float = 0.35
    torque_penalty_weight: float = 0.0005
    hit_reward_base: float = 2.5
    hit_reward_combo: float = 1.2
    post_hit_survival_reward_weight: float = 1.4
    post_hit_ball_xy_sigma: float = 0.12
    post_hit_ball_vxy_penalty_weight: float = 0.18
    descending_intercept_reward_weight: float = 1.6
    descending_intercept_sigma: float = 0.10
    non_racket_ball_contact_penalty_weight: float = 1.5
    failed_hit_penalty_weight: float = 1.0
    sticky_contact_penalty_growth: float = 0.6
    low_hit_apex_margin: float = 0.06
    low_hit_penalty_weight: float = 10.0
    hit_center_local_sigma: float = 0.035
    hit_flatness_target_cos: float = 0.96
    hit_flatness_sigma: float = 0.08
    center_flat_hit_reward_weight: float = 1.8
    contact_flatness_penalty_weight: float = 0.45
    termination_miss_penalty_base: float = 2.5
    termination_miss_penalty_per_hit: float = 0.8
    hit_rearm_no_contact_steps: int = 2
    hit_rearm_distance: float = 0.035
    stick_contact_penalty_weight: float = 0.60
    stick_rel_speed_thresh: float = 0.25
    stick_rel_dist_thresh: float = 0.040
    stick_min_contact_steps: int = 4
    hit_confirm_rel_height: float = 0.06
    hit_confirm_abs_height: float = 1.00
    hit_confirm_max_steps: int = 70
    hit_confirm_use_spawn_cube_band: bool = False
    hit_confirm_spawn_band_margin: float = 0.0
    hit_height_center: float = 0.52
    hit_height_tolerance: float = 0.06
    hit_height_penalty_weight: float = 10.0
    racket_z_band_down: float = 0.00 # 0.06
    racket_z_band_up: float = 0.20 # 0.10
    racket_z_soft_penalty_weight: float = 1.2
    racket_up_drift_penalty_weight: float = 0.3
    racket_up_drift_vel_thresh: float = 0.02
    racket_z_hard_limit_down: float = 0.12
    racket_z_hard_limit_up: float = 0.24
    rel_height_center: float = 0.18
    rel_height_sigma: float = 0.06
    rel_height_bonus_weight: float = 0.45
    # Planar (XY) Gaussian anchor shaping for racket position.
    # For 2D isotropic Gaussian, 95% mass radius r95 ≈ 2.4477 * sigma.
    # Setting sigma ~= 0.041 gives r95 ~= 0.10 m.
    racket_xy_gauss_sigma: float = 0.041
    racket_xy_gauss_reward_weight: float = 0.50
    racket_xy_gauss_penalty_weight: float = 0.60
    # Keep interaction localized near chest-front striking region.
    racket_chest_xy_penalty_weight: float = 1.0
    racket_chest_z_penalty_weight: float = 0.8
    ball_anchor_xy_penalty_weight: float = 0.7
    ball_base_x_penalty_weight: float = 2.0
    ball_base_x_soft_limit: float = 0.04
    ball_base_vxy_penalty_weight: float = 0.60
    ball_vxy_penalty_weight: float = 0.40
    hit_center_sigma: float = 0.08
    apex_soft_limit_margin: float = 0.04
    apex_soft_penalty_weight: float = 5.0
    ball_xy_soft_limit_radius: float = 0.14
    ball_xy_soft_penalty_weight: float = 3.0
    domain_randomization: bool = True
    dr_randomize_ball: bool = True
    dr_randomize_contact: bool = True
    dr_randomize_actuator: bool = True
    dr_randomize_latency: bool = True
    dr_ball_mass_range: tuple[float, float] = (0.0024, 0.0030)
    dr_ball_friction_range: tuple[float, float] = (0.12, 0.35)
    dr_racket_friction_range: tuple[float, float] = (0.25, 0.55)
    dr_ball_solref_time_range: tuple[float, float] = (0.002, 0.006)
    dr_ball_solref_damping_range: tuple[float, float] = (0.70, 0.95)
    dr_gravity_z_range: tuple[float, float] = (-9.90, -9.70)
    dr_action_scale_mult_range: tuple[float, float] = (0.85, 1.15)
    dr_armature_mult_range: tuple[float, float] = (0.80, 1.20)
    dr_damping_mult_range: tuple[float, float] = (0.70, 1.30)
    dr_obs_latency_steps_range: tuple[int, int] = (0, 2)
    dr_action_latency_steps_range: tuple[int, int] = (0, 2)
    camera_visibility_mode: str = "off"  # "off", "box", "frustum", "pixel"
    virtual_camera_body_name: str = "head22"
    virtual_camera_mount_pos: tuple[float, float, float] = (0.0, -0.068, 0.062)
    virtual_camera_mount_quat: tuple[float, float, float, float] = (0.707107, 0.0, 0.0, -0.707107)
    virtual_camera_optical_pos: tuple[float, float, float] = (0.048, 0.0, 0.0)
    camera_image_width: int = 1280
    camera_image_height: int = 720
    camera_fx: float = 636.99
    camera_fy: float = 636.84
    camera_cx: float = 646.82
    camera_cy: float = 373.21
    camera_hfov_deg: float = 86.0
    camera_vfov_deg: float = 57.0
    camera_min_depth: float = 0.15
    camera_max_depth: float = 2.50
    camera_pixel_margin: float = 80.0
    camera_center_weight: float = 0.0
    camera_visibility_penalty_weight: float = 0.0
    camera_depth_penalty_weight: float = 0.0
    camera_box_penalty_weight: float = 0.0
    camera_visible_penalty_weight: float = 0.0
    camera_top_margin_penalty_weight: float = 0.0
    camera_dense_penalty_clip: float = 20.0
    camera_box_half_width: float = 0.35
    camera_box_half_height: float = 0.35
    camera_box_depth_min: float = 0.20
    camera_box_depth_max: float = 1.50
    arm_action_limiter: bool = False
    arm_vel_limit_penalty_weight: float = 0.0
    arm_acc_limit_penalty_weight: float = 0.0
    arm_limiter_penalty_weight: float = 0.0
    arm_vel_limit_deg_s: tuple[float, ...] = (210.0, 210.0, 240.0, 240.0, 300.0, 300.0, 300.0)
    arm_acc_limit_deg_s2: tuple[float, ...] = (1300.0, 1300.0, 1800.0, 3000.0, 3000.0, 3000.0, 3000.0)
    action_acc_scale: float = 1.0
    action_penalty_weight: float = 0.003
    action_delta_penalty_weight: float = 0.001
    dr_randomize_racket_mount: bool = False
    dr_racket_pos_offset_m: float = 0.0
    dr_racket_rot_offset_rad: float = 0.0
    dr_racket_radius_offset_m: float = 0.0
    hit_cadence_reward_weight: float = 0.0
    hit_cadence_target_interval: float = 0.65
    hit_cadence_sigma: float = 0.18
    hit_min_interval_penalty_weight: float = 0.0
    hit_min_interval: float = 0.40
    hit_min_count_interval: float = 0.0
    fast_hit_penalty_weight: float = 0.0
    hit_reward_cap_mode: str = "off"
    hit_reward_count_cap: int = 0
    hit_reward_cap_target_interval: float = 0.65
    ball_obs_dropout_prob: float = 0.0
    ball_obs_dropout_max_steps: int = 1
    ball_obs_dropout_burst_prob: float = 0.0
    ball_obs_dropout_burst_max_steps: int = 1
    ball_obs_age_clip: float = 0.20


RIGHT_ARM_JOINTS = ["RightArm-0", "RightArm-1", "RightArm-2", "RightArm-3", "RightArm-4", "RightArm-5", "RightArm-6"]
BASE_ACTS = ["Base-X", "Base-Y", "Base-Yaw"]

# TARGET_DEGREES = {
#     "RightArm-0": 9, "RightArm-1": -50, "RightArm-2": 20,
#     "RightArm-3": 90, "RightArm-4": 35, "RightArm-5": 8, "RightArm-6": 45,
#     "LeftArm-0": -9, "LeftArm-1": -50, "LeftArm-2": -20,
#     "LeftArm-3": -90, "LeftArm-4": -35, "LeftArm-5": 8, "LeftArm-6": -45,
#     "LegWaist-0": 0, "LegWaist-1": 60, "LegWaist-2": -90,
#     "LegWaist-3": 30, "LegWaist-4": 0, "LegWaist-5": 0,
#     "Head-0": 0, "Head-1": 40,
#     }
TARGET_DEGREES = {
    "RightArm-0": 9, "RightArm-1": -50, "RightArm-2": 20,
    "RightArm-3": 90, "RightArm-4": 45, "RightArm-5": -8, "RightArm-6": 45,
    "LeftArm-0": -9, "LeftArm-1": -50, "LeftArm-2": -20,
    "LeftArm-3": -90, "LeftArm-4": -35, "LeftArm-5": 8, "LeftArm-6": -45,
    "LegWaist-0": 0, "LegWaist-1": 60, "LegWaist-2": -90,
    "LegWaist-3": 30, "LegWaist-4": 0, "LegWaist-5": 0,
    "Head-0": 0, "Head-1": 40,
    }

def _deg_to_rad_map(deg_map: dict[str, float]) -> dict[str, float]:
    return {k: float(np.deg2rad(v)) for k, v in deg_map.items()}


def _quat_wxyz_to_mat(q) -> np.ndarray:
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array([
        [1.0 - yy - zz, xy - wz, xz + wy],
        [xy + wz, 1.0 - xx - zz, yz - wx],
        [xz - wy, yz + wx, 1.0 - xx - yy],
    ], dtype=np.float64)


def _build_temp_xml_with_ball(xml_path: Path) -> Path:
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    existing_meshdir = compiler.get("meshdir")
    if existing_meshdir:
        meshdir_path = Path(existing_meshdir).expanduser()
        if meshdir_path.is_absolute() and meshdir_path.exists():
            compiler.set("meshdir", str(meshdir_path))
        else:
            compiler.set("meshdir", str((xml_path.parent / existing_meshdir).resolve()))
    else:
        compiler.set("meshdir", str((xml_path.parent / "meshes").resolve()))

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")

    if not any((b.get("name") == "pingpong_ball") for b in worldbody.findall("body")):
        ball_body = ET.SubElement(worldbody, "body", name="pingpong_ball", pos="0.35 -0.12 1.05")
        ET.SubElement(ball_body, "freejoint", name="ball_free")
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

    # Normalize ping-pong ball contact parameters even if ball already exists.
    for b in worldbody.findall("body"):
        if b.get("name") == "pingpong_ball":
            for g in b.findall("geom"):
                if g.get("name") == "ball":
                    g.set("friction", "0.20 0.001 0.0001")
                    g.set("solref", "0.003 0.80")
                    g.set("solimp", "0.90 0.995 0.001")
                    g.set("margin", "0.001")
                    g.set("condim", "3")
                    g.set("contype", "1")
                    g.set("conaffinity", "1")
                    break
            break

    # User requested no table in this scene.
    for b in list(worldbody.findall("body")):
        if b.get("name") == "table":
            worldbody.remove(b)
    for g in list(worldbody.findall("geom")):
        if g.get("name") == "pingpong_table":
            worldbody.remove(g)

    # Keep head geometry visual-only for juggling. The real head/camera should
    # constrain visibility, but it should not create accidental ball contacts.
    for body in worldbody.iter("body"):
        if body.get("name") in {"head21", "head22", "head23"}:
            for g in body.findall("geom"):
                g.set("contype", "0")
                g.set("conaffinity", "0")

    # Ensure right hand has a paddle only when missing.
    # If user already edited moz1.xml (e.g. changed quat/pose), keep it unchanged.
    for body in worldbody.iter("body"):
        if body.get("name") == "right07":
            # Remove old blue helper sphere if present.
            for g in list(body.findall("geom")):
                if g.get("name") == "right_ee_ball":
                    body.remove(g)

            has_racket_body = any((c.tag == "body" and c.get("name") == "right_racket") for c in body)
            has_ee_site = any((s.get("name") == "right_ee_site") for s in body.iter("site"))

            if not has_racket_body:
                racket = ET.SubElement(body, "body", name="right_racket", pos="0 0 -0.215", quat="0.92388 0.38268 0 0")
                ET.SubElement(
                    racket,
                    "geom",
                    name="racket_wood",
                    type="cylinder",
                    size="0.075 0.0055",
                    pos="0 0 0",
                    rgba="0.5 0.2 0.1 1",
                    contype="0",
                    conaffinity="0",
                )
                ET.SubElement(
                    racket,
                    "geom",
                    name="racket_rubber_fore",
                    type="cylinder",
                    size="0.075 0.001",
                    pos="0 0 0.0065",
                    rgba="0.8 0.1 0.1 1",
                    solref="0.003 0.80",
                    solimp="0.90 0.995 0.001",
                    friction="0.35 0.002 0.0001",
                    margin="0.001",
                    condim="3",
                    contype="1",
                    conaffinity="1",
                )
                ET.SubElement(
                    racket,
                    "site",
                    name="right_ee_site",
                    pos="0 0 0.0065",
                    size="0.004",
                    rgba="0 0 0 0",
                )
            elif not has_ee_site:
                # Racket exists but missing ee site: add it to the racket body.
                for c in body:
                    if c.tag == "body" and c.get("name") == "right_racket":
                        ET.SubElement(
                            c,
                            "site",
                            name="right_ee_site",
                            pos="0 0 0.0065",
                            size="0.004",
                            rgba="0 0 0 0",
                        )
                        break

            # Normalize existing racket body/site/geoms to match measured paddle dimensions.
            for c in body:
                if c.tag == "body" and c.get("name") == "right_racket":
                    c.set("pos", "0 0 -0.215")
                    for g in c.findall("geom"):
                        gname = g.get("name")
                        if gname == "racket_wood":
                            g.set("size", "0.075 0.0055")
                            g.set("pos", "0 0 0")
                            g.set("contype", "0")
                            g.set("conaffinity", "0")
                        elif gname == "racket_rubber_fore":
                            g.set("size", "0.075 0.001")
                            g.set("pos", "0 0 0.0065")
                            g.set("contype", "1")
                            g.set("conaffinity", "1")
                            g.set("friction", "0.35 0.002 0.0001")
                            g.set("solref", "0.003 0.80")
                            g.set("solimp", "0.90 0.995 0.001")
                            g.set("margin", "0.001")
                            g.set("condim", "3")
                    for s in c.findall("site"):
                        if s.get("name") == "right_ee_site":
                            s.set("pos", "0 0 0.0065")
                            break
                    break
            break

    # Explicit contact pair for ball-rubber, to keep behavior consistent across edits.
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for p in list(contact.findall("pair")):
        if p.get("geom1") == "ball" and p.get("geom2") == "racket_rubber_fore":
            contact.remove(p)
        elif p.get("geom2") == "ball" and p.get("geom1") == "racket_rubber_fore":
            contact.remove(p)
    ET.SubElement(
        contact,
        "pair",
        geom1="ball",
        geom2="racket_rubber_fore",
        friction="0.35 0.002 0.0001",
        solref="0.003 0.80",
        solimp="0.90 0.995 0.001",
        condim="3",
    )

    tmp = tempfile.NamedTemporaryFile(prefix="moz1_pingpong_", suffix=".xml", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    ET.ElementTree(root).write(tmp_path, encoding="utf-8", xml_declaration=False)
    return tmp_path


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions in wxyz format and normalize."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    result = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float32)
    # Normalize to avoid quaternion drift
    norm = np.linalg.norm(result)
    if norm > 1e-8:
        result /= norm
    return result


def _euler_xyz_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert Euler angles (XYZ order) to quaternion (wxyz format) and normalize."""
    cr, sr = np.cos(roll/2), np.sin(roll/2)
    cp, sp = np.cos(pitch/2), np.sin(pitch/2)
    cy, sy = np.cos(yaw/2), np.sin(yaw/2)

    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy

    result = np.array([w, x, y, z], dtype=np.float32)
    # Normalize to avoid quaternion drift
    norm = np.linalg.norm(result)
    if norm > 1e-8:
        result /= norm
    return result


class JuggleEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, xml_path: str | None = None, cfg: JuggleConfig = JuggleConfig()) -> None:
        super().__init__()
        if xml_path is None:
            raise ValueError("xml_path is required")
        self.cfg = cfg

        patched = _build_temp_xml_with_ball(Path(xml_path).resolve())
        self.model = mj.MjModel.from_xml_path(str(patched))
        self.data = mj.MjData(self.model)

        self.dt = float(self.model.opt.timestep * cfg.frame_skip)
        self.max_steps = max(1, int(cfg.horizon_sec / self.dt))
        self.step_count = 0

        self.ball_joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "ball_free")
        self.racket_site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, "right_ee_site")
        self.ball_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "pingpong_ball")
        self.ball_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "ball")
        face_gid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "racket_rubber_fore")
        self.racket_geom_ids = [int(face_gid)] if face_gid >= 0 else []
        if not self.racket_geom_ids:
            gid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "racket_wood")
            if gid >= 0:
                self.racket_geom_ids = [int(gid)]

        self.arm_jids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, n) for n in RIGHT_ARM_JOINTS]
        self.arm_aids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, n) for n in RIGHT_ARM_JOINTS]
        self.base_aids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, n) for n in BASE_ACTS]
        self.arm_qadr = np.array([int(self.model.jnt_qposadr[j]) for j in self.arm_jids], dtype=np.int32)
        self.arm_vadr = np.array([int(self.model.jnt_dofadr[j]) for j in self.arm_jids], dtype=np.int32)
        self.arm_lo = np.array([self.model.jnt_range[j, 0] for j in self.arm_jids], dtype=np.float32)
        self.arm_hi = np.array([self.model.jnt_range[j, 1] for j in self.arm_jids], dtype=np.float32)

        bx = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "base_x")
        by = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "base_y")
        byaw = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "base_yaw")
        self.base_x_qadr = int(self.model.jnt_qposadr[bx])
        self.base_y_qadr = int(self.model.jnt_qposadr[by])
        self.base_yaw_qadr = int(self.model.jnt_qposadr[byaw])
        self.base_x_vadr = int(self.model.jnt_dofadr[bx])
        self.base_y_vadr = int(self.model.jnt_dofadr[by])
        self.base_yaw_vadr = int(self.model.jnt_dofadr[byaw])

        self.ball_qadr = int(self.model.jnt_qposadr[self.ball_joint_id])
        self.ball_vadr = int(self.model.jnt_dofadr[self.ball_joint_id])
        self.base_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "base_link")
        self.waist_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "waist03")
        head22_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "head22")
        head21_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "head21")
        self.head_body_id = int(head22_id if head22_id >= 0 else head21_id)
        self.head_front_local_offset = np.array([0.0, -0.068, 0.062], dtype=np.float32)
        self.head_front_local_normal = np.array([0.0, -1.0, 0.0], dtype=np.float32)

        vc_bid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, self.cfg.virtual_camera_body_name)
        self.virtual_camera_body_id = int(vc_bid) if vc_bid >= 0 else -1
        self._vc_mount_R = _quat_wxyz_to_mat(self.cfg.virtual_camera_mount_quat)
        self._vc_mount_pos = np.asarray(self.cfg.virtual_camera_mount_pos, dtype=np.float64)
        self._vc_optical_pos = np.asarray(self.cfg.virtual_camera_optical_pos, dtype=np.float64)
        # Reference mount frame uses +X as the optical axis. Convert to the
        # virtual camera frame used by projection: +X right, +Y up, +Z forward.
        self._vc_mount_to_camera_R = np.array(
            [
                [0.0, 0.0, 1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )

        self.ctrl_dim = len(RIGHT_ARM_JOINTS)
        self.prev_action = np.zeros(self.ctrl_dim, dtype=np.float32)
        self.prev_ball_pos = np.zeros(3, dtype=np.float32)
        self.prev_racket_pos = np.zeros(3, dtype=np.float32)
        self.prev_contact = False
        self.hit_armed = True
        self.no_contact_steps = 0
        self.contact_hold_steps = 0
        self.pending_hit = False
        self.pending_hit_steps = 0
        self.hit_confirm_abs_low = float(self.cfg.hit_confirm_abs_height)
        self.hit_confirm_abs_high = float("inf")
        self.hit_count = 0
        self.last_hit_time = -1.0
        self.last_hit_step = -1
        self.hit_intervals = []
        self.hit_cadence_rewards = []
        self.hit_min_interval_penalties = []
        self.last_counted_hit_time = -1.0
        self.last_count_gate_hit_time = -1.0
        self.confirmed_hit_count = 0
        self.ignored_fast_hit_count = 0
        self.counted_hit_intervals = []
        self.rewarded_hit_count = 0
        self.unrewarded_extra_hit_count = 0
        self.hit_reward_count_cap_active = self._get_hit_reward_count_cap()
        self.racket_anchor = np.zeros(3, dtype=np.float32)
        self.chest_target_offset = np.zeros(3, dtype=np.float32)
        self.chest_target_pos = np.zeros(3, dtype=np.float32)
        self.initial_base_pose = np.zeros(3, dtype=np.float32)
        self.ball_obs_every = 1
        if float(self.cfg.ball_obs_rate_hz) > 0.0:
            self.ball_obs_every = max(1, int(round(1.0 / (float(self.cfg.ball_obs_rate_hz) * self.dt))))
        self.cached_ball_obs_pos = np.zeros(3, dtype=np.float32)
        self.cached_ball_obs_vel = np.zeros(3, dtype=np.float32)
        self.last_ball_obs_step = -1
        self.total_env_steps = 0

        target_rad = _deg_to_rad_map(TARGET_DEGREES)
        names = list(target_rad.keys())
        posture_jids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, n) for n in names]
        posture_aids = [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, n) for n in names]
        self.posture_qadr = np.array([int(self.model.jnt_qposadr[j]) for j in posture_jids], dtype=np.int32)
        self.posture_targets = np.array([target_rad[n] for n in names], dtype=np.float32)

        self.default_ctrl = np.zeros(self.model.nu, dtype=np.float32)
        for n, aid in zip(names, posture_aids):
            if aid >= 0:
                self.default_ctrl[aid] = np.float32(target_rad[n])

        self.arm_q_target = np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float32)
        self.base_cmd = np.zeros(len(BASE_ACTS), dtype=np.float32)

        # Arm velocity/acceleration limits for hardware-friendly training
        self.arm_vel_limit_deg_s = np.array(self.cfg.arm_vel_limit_deg_s, dtype=np.float32)
        self.arm_acc_limit_deg_s2 = np.array(self.cfg.arm_acc_limit_deg_s2, dtype=np.float32)
        if len(self.arm_vel_limit_deg_s) != len(RIGHT_ARM_JOINTS):
            raise ValueError(f"arm_vel_limit_deg_s must have {len(RIGHT_ARM_JOINTS)} elements, got {len(self.arm_vel_limit_deg_s)}")
        if len(self.arm_acc_limit_deg_s2) != len(RIGHT_ARM_JOINTS):
            raise ValueError(f"arm_acc_limit_deg_s2 must have {len(RIGHT_ARM_JOINTS)} elements, got {len(self.arm_acc_limit_deg_s2)}")
        self.prev_arm_qvel = np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float32)

        # Command trajectory generator state
        self.arm_cmd_q = np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float32)
        self.arm_cmd_qvel = np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float32)

        # Racket mount DR: track racket body and geoms
        self.racket_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "right_racket")
        self.racket_wood_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "racket_wood")
        self.racket_rubber_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "racket_rubber_fore")

        # Save original racket mount parameters
        if self.racket_body_id >= 0:
            self.original_racket_body_pos = self.model.body_pos[self.racket_body_id].copy()
            self.original_racket_body_quat = self.model.body_quat[self.racket_body_id].copy()
        else:
            self.original_racket_body_pos = None
            self.original_racket_body_quat = None

        self.original_racket_geom_sizes = {}
        if self.racket_wood_geom_id >= 0:
            self.original_racket_geom_sizes[self.racket_wood_geom_id] = self.model.geom_size[self.racket_wood_geom_id].copy()
        if self.racket_rubber_geom_id >= 0:
            self.original_racket_geom_sizes[self.racket_rubber_geom_id] = self.model.geom_size[self.racket_rubber_geom_id].copy()

        # Current racket mount offsets (set in reset)
        self.dr_racket_pos_offset = np.zeros(3, dtype=np.float32)
        self.dr_racket_rot_offset = np.zeros(3, dtype=np.float32)
        self.dr_racket_radius_offset = 0.0


        # Ball obs dropout state
        self.ball_obs_dropout_remaining = 0
        self.ball_obs_valid_pos = np.zeros(3, dtype=np.float32)
        self.ball_obs_valid_vel = np.zeros(3, dtype=np.float32)
        self.ball_obs_age_seconds = 0.0
        self.ball_obs_dropout_steps_total = 0
        self.ball_obs_burst_count = 0

        n_arm = len(RIGHT_ARM_JOINTS)
        # +1 for ball_obs_age
        obs_dim = n_arm + n_arm + 3 + 3 + 3 + 3 + 3 + 3 + 3 + self.ctrl_dim + n_arm + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.ctrl_dim,), dtype=np.float32)

        self.original_ball_mass = None
        self.original_ball_friction = None
        self.original_ball_solref = None
        self.original_racket_friction = None
        self.original_gravity = self.model.opt.gravity.copy()
        self.original_dof_damping = self.model.dof_damping.copy()
        self.original_dof_armature = self.model.dof_armature.copy()
        if self.ball_body_id >= 0:
            self.original_ball_mass = float(self.model.body_mass[self.ball_body_id])
        if self.ball_geom_id >= 0:
            self.original_ball_friction = self.model.geom_friction[self.ball_geom_id].copy()
            self.original_ball_solref = self.model.geom_solref[self.ball_geom_id].copy()
        if self.racket_geom_ids:
            self.original_racket_friction = self.model.geom_friction[self.racket_geom_ids[0]].copy()

        self.action_scale_mult = 1.0
        self.dr_obs_latency_steps = 0
        self.dr_action_latency_steps = 0
        self.action_buffer = []
        self.obs_buffer = []
        self.dr_ball_mass = 0.0027
        self.dr_ball_friction = 0.20
        self.dr_racket_friction = 0.35
        self.dr_gravity_z = -9.81
        self.dr_ball_solref_time = 0.003
        self.dr_ball_solref_damping = 0.80


    def _get_hit_reward_count_cap(self) -> int:
        """Calculate active hit reward count cap based on mode."""
        if self.cfg.hit_reward_cap_mode == "off":
            return 0
        elif self.cfg.hit_reward_cap_mode == "fixed":
            return max(0, int(self.cfg.hit_reward_count_cap))
        elif self.cfg.hit_reward_cap_mode == "auto":
            episode_total_time = float(self.max_steps) * self.dt
            cap = int(np.floor(episode_total_time / max(1e-6, float(self.cfg.hit_reward_cap_target_interval))))
            return max(1, cap)
        else:
            raise ValueError(f"Invalid hit_reward_cap_mode: {self.cfg.hit_reward_cap_mode}. Must be 'off', 'auto', or 'fixed'.")

    def _ball_pos(self) -> np.ndarray:
        return np.array(self.data.xpos[self.ball_body_id], dtype=np.float32)

    def _racket_pos(self) -> np.ndarray:
        return np.array(self.data.site_xpos[self.racket_site_id], dtype=np.float32)

    def _ball_vel_est(self, cur: np.ndarray) -> np.ndarray:
        return (cur - self.prev_ball_pos) / max(self.dt, 1e-6)

    def _racket_vel_est(self, cur: np.ndarray) -> np.ndarray:
        return (cur - self.prev_racket_pos) / max(self.dt, 1e-6)

    def _racket_xmat(self) -> np.ndarray:
        return np.array(self.data.site_xmat[self.racket_site_id], dtype=np.float32).reshape(3, 3)

    def _base_pose_world(self) -> tuple[np.ndarray, np.ndarray]:
        if self.base_body_id >= 0:
            pos = np.asarray(self.data.xpos[self.base_body_id], dtype=np.float32)
            mat = np.asarray(self.data.xmat[self.base_body_id], dtype=np.float32).reshape(3, 3)
            return pos, mat

        yaw = float(self.data.qpos[self.base_yaw_qadr])
        c = float(np.cos(yaw))
        s = float(np.sin(yaw))
        pos = np.array(
            [self.data.qpos[self.base_x_qadr], self.data.qpos[self.base_y_qadr], 0.0],
            dtype=np.float32,
        )
        mat = np.array(
            [
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return pos, mat

    def _world_point_to_base(self, point_world: np.ndarray) -> np.ndarray:
        base_pos, base_R = self._base_pose_world()
        return (base_R.T @ (np.asarray(point_world, dtype=np.float32) - base_pos)).astype(np.float32)

    def _world_velocity_to_base_at_point(self, vel_world: np.ndarray, point_world: np.ndarray) -> np.ndarray:
        base_pos, base_R = self._base_pose_world()
        base_lin_vel = np.array(
            [self.data.qvel[self.base_x_vadr], self.data.qvel[self.base_y_vadr], 0.0],
            dtype=np.float32,
        )
        base_ang_vel = np.array([0.0, 0.0, self.data.qvel[self.base_yaw_vadr]], dtype=np.float32)
        rel_point = np.asarray(point_world, dtype=np.float32) - base_pos
        rel_vel_world = np.asarray(vel_world, dtype=np.float32) - base_lin_vel - np.cross(base_ang_vel, rel_point)
        return (base_R.T @ rel_vel_world).astype(np.float32)

    def _head_front_surface(self) -> tuple[np.ndarray, np.ndarray]:
        if self.head_body_id < 0:
            return np.zeros(3, dtype=np.float32), np.array([0.0, -1.0, 0.0], dtype=np.float32)
        head_pos = np.array(self.data.xpos[self.head_body_id], dtype=np.float32)
        head_rot = np.array(self.data.xmat[self.head_body_id], dtype=np.float32).reshape(3, 3)
        center = head_pos + head_rot @ self.head_front_local_offset
        normal = head_rot @ self.head_front_local_normal
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm > 1e-6:
            normal = normal / normal_norm
        else:
            normal = np.array([0.0, -1.0, 0.0], dtype=np.float32)
        return center.astype(np.float32), normal.astype(np.float32)

    def _virtual_camera_pose(self) -> tuple[np.ndarray, np.ndarray, bool]:
        if self.virtual_camera_body_id < 0:
            return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64), False
        body_pos = np.asarray(self.data.xpos[self.virtual_camera_body_id], dtype=np.float64)
        body_R = np.asarray(self.data.xmat[self.virtual_camera_body_id], dtype=np.float64).reshape(3, 3)
        cam_pos_world = body_pos + body_R @ (self._vc_mount_pos + self._vc_mount_R @ self._vc_optical_pos)
        cam_R_world = body_R @ self._vc_mount_R @ self._vc_mount_to_camera_R
        return cam_pos_world, cam_R_world, True

    def _ball_camera_metrics(self, bpos: np.ndarray) -> dict:
        metrics = {
            "camera_available": False,
            "ball_cam_x": 0.0,
            "ball_cam_y": 0.0,
            "ball_cam_z": 0.0,
            "ball_pixel_u": 0.0,
            "ball_pixel_v": 0.0,
            "camera_in_front": False,
            "camera_in_depth": False,
            "camera_in_frustum": False,
            "camera_in_image": False,
            "camera_in_margin": False,
            "camera_visible": False,
            "camera_pixel_center_pen": 0.0,
            "camera_pixel_margin_pen": 0.0,
            "camera_top_margin_pen": 0.0,
            "camera_frustum_pen": 0.0,
            "camera_depth_pen": 0.0,
            "camera_box_pen": 0.0,
        }

        cam_pos, cam_R, available = self._virtual_camera_pose()
        metrics["camera_available"] = bool(available)
        if not available:
            return metrics

        p_cam = cam_R.T @ (np.asarray(bpos, dtype=np.float64) - cam_pos)
        x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
        metrics["ball_cam_x"] = x
        metrics["ball_cam_y"] = y
        metrics["ball_cam_z"] = z
        if z <= 1e-6:
            return metrics

        width = float(self.cfg.camera_image_width)
        height = float(self.cfg.camera_image_height)
        fx = float(self.cfg.camera_fx)
        fy = float(self.cfg.camera_fy)
        cx = float(self.cfg.camera_cx)
        cy = float(self.cfg.camera_cy)
        margin = float(self.cfg.camera_pixel_margin)

        u = fx * (x / z) + cx
        v = cy - fy * (y / z)
        metrics["ball_pixel_u"] = float(u)
        metrics["ball_pixel_v"] = float(v)

        in_front = z > 0.0
        in_depth = float(self.cfg.camera_min_depth) <= z <= float(self.cfg.camera_max_depth)
        x_angle = float(np.arctan2(x, z))
        y_angle = float(np.arctan2(y, z))
        h_half = float(np.deg2rad(float(self.cfg.camera_hfov_deg) * 0.5))
        v_half = float(np.deg2rad(float(self.cfg.camera_vfov_deg) * 0.5))
        x_angle_excess = max(0.0, abs(x_angle) - h_half)
        y_angle_excess = max(0.0, abs(y_angle) - v_half)
        in_frustum = (x_angle_excess <= 0.0) and (y_angle_excess <= 0.0)
        in_image = 0.0 <= u < width and 0.0 <= v < height
        in_margin = margin <= u <= (width - margin) and margin <= v <= (height - margin)

        du = (u - cx) / max(0.5 * width, 1e-6)
        dv = (v - cy) / max(0.5 * height, 1e-6)
        center_pen = float(du * du + dv * dv)

        depth_pen = 0.0
        min_depth = float(self.cfg.camera_min_depth)
        max_depth = float(self.cfg.camera_max_depth)
        if z < min_depth:
            depth_pen = ((min_depth - z) / max(min_depth, 1e-6)) ** 2
        elif z > max_depth:
            depth_pen = ((z - max_depth) / max(max_depth, 1e-6)) ** 2

        u_low = max(0.0, margin - u)
        u_high = max(0.0, u - (width - margin))
        v_low = max(0.0, margin - v)
        v_high = max(0.0, v - (height - margin))
        margin_pen = float(
            (u_low / max(width, 1e-6)) ** 2
            + (u_high / max(width, 1e-6)) ** 2
            + (v_low / max(height, 1e-6)) ** 2
            + (v_high / max(height, 1e-6)) ** 2
        )
        frustum_pen = float(
            (x_angle_excess / max(h_half, 1e-6)) ** 2
            + (y_angle_excess / max(v_half, 1e-6)) ** 2
        )

        box_pen = 0.0
        x_excess = max(0.0, abs(x) - float(self.cfg.camera_box_half_width))
        y_excess = max(0.0, abs(y) - float(self.cfg.camera_box_half_height))
        z_low_excess = max(0.0, float(self.cfg.camera_box_depth_min) - z)
        z_high_excess = max(0.0, z - float(self.cfg.camera_box_depth_max))
        box_pen = float(x_excess * x_excess + y_excess * y_excess + z_low_excess * z_low_excess + z_high_excess * z_high_excess)

        top_excess = max(0.0, margin - v)
        top_margin_pen = float((top_excess / max(height, 1e-6)) ** 2)

        metrics.update({
            "camera_in_front": bool(in_front),
            "camera_in_depth": bool(in_depth),
            "camera_in_frustum": bool(in_frustum),
            "camera_in_image": bool(in_image),
            "camera_in_margin": bool(in_margin),
            "camera_visible": bool(in_front and in_depth and in_frustum and in_image),
            "camera_pixel_center_pen": center_pen,
            "camera_pixel_margin_pen": margin_pen,
            "camera_top_margin_pen": top_margin_pen,
            "camera_frustum_pen": frustum_pen,
            "camera_depth_pen": float(depth_pen),
            "camera_box_pen": box_pen,
        })
        return metrics

    def _ball_path_to_head_front_metrics(
        self,
        bpos: np.ndarray,
        bvel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
        center, normal = self._head_front_surface()
        plane_signed_dist = float(np.dot(bpos - center, normal))
        plane_dist = abs(plane_signed_dist)
        denom = float(np.dot(bvel, normal))
        path_dist = float("inf")
        time_to_plane = float("inf")
        if abs(denom) > 1e-6:
            t_hit = -plane_signed_dist / denom
            if t_hit >= 0.0:
                time_to_plane = float(t_hit)
                hit_pt = bpos + bvel * float(t_hit)
                tangent = hit_pt - center
                tangent -= float(np.dot(tangent, normal)) * normal
                path_dist = float(np.linalg.norm(tangent))
        speed_sq = float(np.dot(bvel, bvel))
        if speed_sq > 1e-8:
            t_closest = max(0.0, -float(np.dot(bpos - center, bvel)) / speed_sq)
            closest_pt = bpos + bvel * float(t_closest)
            ray_center_dist = float(np.linalg.norm(closest_pt - center))
        else:
            ray_center_dist = float(np.linalg.norm(bpos - center))
        return center, normal, plane_dist, path_dist, time_to_plane, ray_center_dist

    def _ball_contact_flags(self) -> tuple[bool, bool]:
        racket_contact = False
        other_ball_contact = False
        for i in range(int(self.data.ncon)):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 == self.ball_geom_id:
                if g2 in self.racket_geom_ids:
                    racket_contact = True
                else:
                    other_ball_contact = True
            elif g2 == self.ball_geom_id:
                if g1 in self.racket_geom_ids:
                    racket_contact = True
                else:
                    other_ball_contact = True
        return racket_contact, other_ball_contact

    def _sample_ball_obs(self, true_bpos: np.ndarray, true_bvel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        noisy_bpos = np.array(true_bpos, dtype=np.float32, copy=True)
        noisy_bvel = np.array(true_bvel, dtype=np.float32, copy=True)
        total_steps_cfg = max(1, int(self.cfg.total_training_steps))
        warmup = max(0, int(round(total_steps_cfg * float(self.cfg.ball_obs_noise_warmup_ratio))))
        ramp = max(1, int(round(total_steps_cfg * float(self.cfg.ball_obs_noise_ramp_ratio))))
        if self.total_env_steps <= warmup:
            noise_scale = 0.0
        else:
            noise_scale = min(1.0, float(self.total_env_steps - warmup) / float(ramp))
        pos_std = float(self.cfg.ball_obs_pos_noise_std) * noise_scale
        vel_std = float(self.cfg.ball_obs_vel_noise_std) * noise_scale
        if pos_std > 0.0:
            noisy_bpos += self.np_random.normal(0.0, pos_std, size=3).astype(np.float32)
        if vel_std > 0.0:
            noisy_bvel += self.np_random.normal(0.0, vel_std, size=3).astype(np.float32)
        return noisy_bpos, noisy_bvel

    def _get_ball_obs(self, true_bpos: np.ndarray, true_bvel: np.ndarray, *, force_refresh: bool = False) -> tuple[np.ndarray, np.ndarray]:
        refresh = force_refresh or self.last_ball_obs_step < 0
        if not refresh:
            refresh = (self.step_count - self.last_ball_obs_step) >= self.ball_obs_every
        if refresh:
            self.cached_ball_obs_pos, self.cached_ball_obs_vel = self._sample_ball_obs(true_bpos, true_bvel)
            self.last_ball_obs_step = int(self.step_count)
        return self.cached_ball_obs_pos.copy(), self.cached_ball_obs_vel.copy()

    def _apply_ball_obs_dropout(self, bpos: np.ndarray, bvel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply dropout to ball observation. Pipeline: obs_rate+noise -> dropout hold-last.
        Returns the (possibly held) ball obs and updates ball_obs_age_seconds."""
        if self.ball_obs_dropout_remaining > 0:
            self.ball_obs_dropout_remaining -= 1
            self.ball_obs_age_seconds += self.dt
            self.ball_obs_dropout_steps_total += 1
            return self.ball_obs_valid_pos.copy(), self.ball_obs_valid_vel.copy()

        # Check if new dropout starts
        dropout_prob = float(self.cfg.ball_obs_dropout_prob)
        burst_prob = float(self.cfg.ball_obs_dropout_burst_prob)
        if burst_prob > 0.0 and float(self.np_random.random()) < burst_prob:
            duration = int(self.np_random.integers(1, int(self.cfg.ball_obs_dropout_burst_max_steps) + 1))
            self.ball_obs_dropout_remaining = max(0, duration - 1)
            self.ball_obs_age_seconds += self.dt
            self.ball_obs_dropout_steps_total += 1
            self.ball_obs_burst_count += 1
            return self.ball_obs_valid_pos.copy(), self.ball_obs_valid_vel.copy()
        elif dropout_prob > 0.0 and float(self.np_random.random()) < dropout_prob:
            duration = int(self.np_random.integers(1, int(self.cfg.ball_obs_dropout_max_steps) + 1))
            self.ball_obs_dropout_remaining = max(0, duration - 1)
            self.ball_obs_age_seconds += self.dt
            self.ball_obs_dropout_steps_total += 1
            return self.ball_obs_valid_pos.copy(), self.ball_obs_valid_vel.copy()

        # No dropout: accept new observation
        self.ball_obs_valid_pos = bpos.copy()
        self.ball_obs_valid_vel = bvel.copy()
        self.ball_obs_age_seconds = 0.0
        return bpos.copy(), bvel.copy()

    def _get_obs(self) -> np.ndarray:
        q = np.array([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float32)
        dq = np.array([self.data.qvel[i] for i in self.arm_vadr], dtype=np.float32)
        base_q = np.array([self.data.qpos[self.base_x_qadr], self.data.qpos[self.base_y_qadr], self.data.qpos[self.base_yaw_qadr]], dtype=np.float32)
        base_dq = np.array([self.data.qvel[self.base_x_vadr], self.data.qvel[self.base_y_vadr], self.data.qvel[self.base_yaw_vadr]], dtype=np.float32)
        true_bpos = self._ball_pos()
        rpos = self._racket_pos()
        true_bvel = self._ball_vel_est(true_bpos)
        # Pipeline: true state -> obs_rate/hold -> noise -> dropout hold-last
        bpos_sampled, bvel_sampled = self._get_ball_obs(true_bpos, true_bvel)
        bpos, bvel = self._apply_ball_obs_dropout(bpos_sampled, bvel_sampled)
        rvel = self._racket_vel_est(rpos)
        bpos_base = self._world_point_to_base(bpos)
        bvel_base = self._world_velocity_to_base_at_point(bvel, bpos)
        rpos_base = self._world_point_to_base(rpos)
        rvel_base = self._world_velocity_to_base_at_point(rvel, rpos)
        rel_base = bpos_base - rpos_base
        arm_cmd_error = self.arm_cmd_q - np.array([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float32)
        age_clip = max(1e-6, float(self.cfg.ball_obs_age_clip))
        ball_obs_age_norm = np.float32(min(self.ball_obs_age_seconds / age_clip, 1.0))
        return np.concatenate([
            q,
            dq,
            base_q,
            base_dq,
            bpos_base,
            bvel_base,
            rpos_base,
            rvel_base,
            rel_base,
            self.prev_action,
            arm_cmd_error,
            np.array([ball_obs_age_norm], dtype=np.float32),
        ]).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if self.cfg.domain_randomization:
            # Always restore originals first to avoid cumulative effects
            self.model.opt.gravity[:] = self.original_gravity
            self.model.dof_damping[:] = self.original_dof_damping
            self.model.dof_armature[:] = self.original_dof_armature
            if self.ball_body_id >= 0 and self.original_ball_mass is not None:
                self.model.body_mass[self.ball_body_id] = self.original_ball_mass
            if self.ball_geom_id >= 0 and self.original_ball_friction is not None:
                self.model.geom_friction[self.ball_geom_id] = self.original_ball_friction
            if self.ball_geom_id >= 0 and self.original_ball_solref is not None:
                self.model.geom_solref[self.ball_geom_id] = self.original_ball_solref
            if self.racket_geom_ids and self.original_racket_friction is not None:
                self.model.geom_friction[self.racket_geom_ids[0]] = self.original_racket_friction

            # Ball DR: ball mass + gravity
            if self.cfg.dr_randomize_ball:
                self.dr_ball_mass = float(self.np_random.uniform(*self.cfg.dr_ball_mass_range))
                self.dr_gravity_z = float(self.np_random.uniform(*self.cfg.dr_gravity_z_range))
                if self.ball_body_id >= 0:
                    self.model.body_mass[self.ball_body_id] = self.dr_ball_mass
                self.model.opt.gravity[2] = self.dr_gravity_z
            else:
                self.dr_ball_mass = self.original_ball_mass if self.original_ball_mass is not None else 0.0027
                self.dr_gravity_z = float(self.original_gravity[2])

            # Contact DR: ball friction + racket friction + ball solref
            if self.cfg.dr_randomize_contact:
                self.dr_ball_friction = float(self.np_random.uniform(*self.cfg.dr_ball_friction_range))
                self.dr_racket_friction = float(self.np_random.uniform(*self.cfg.dr_racket_friction_range))
                self.dr_ball_solref_time = float(self.np_random.uniform(*self.cfg.dr_ball_solref_time_range))
                self.dr_ball_solref_damping = float(self.np_random.uniform(*self.cfg.dr_ball_solref_damping_range))
                if self.ball_geom_id >= 0:
                    self.model.geom_friction[self.ball_geom_id, 0] = self.dr_ball_friction
                    self.model.geom_solref[self.ball_geom_id, 0] = self.dr_ball_solref_time
                    self.model.geom_solref[self.ball_geom_id, 1] = self.dr_ball_solref_damping
                if self.racket_geom_ids:
                    self.model.geom_friction[self.racket_geom_ids[0], 0] = self.dr_racket_friction
            else:
                self.dr_ball_friction = float(self.original_ball_friction[0]) if self.original_ball_friction is not None else 0.20
                self.dr_racket_friction = float(self.original_racket_friction[0]) if self.original_racket_friction is not None else 0.35
                self.dr_ball_solref_time = float(self.original_ball_solref[0]) if self.original_ball_solref is not None else 0.003
                self.dr_ball_solref_damping = float(self.original_ball_solref[1]) if self.original_ball_solref is not None else 0.80

            # Actuator DR: action scale + damping + armature
            if self.cfg.dr_randomize_actuator:
                self.action_scale_mult = float(self.np_random.uniform(*self.cfg.dr_action_scale_mult_range))
                armature_mult = float(self.np_random.uniform(*self.cfg.dr_armature_mult_range))
                damping_mult = float(self.np_random.uniform(*self.cfg.dr_damping_mult_range))
                self.model.dof_damping[:] = self.original_dof_damping * damping_mult
                self.model.dof_armature[:] = self.original_dof_armature * armature_mult
            else:
                self.action_scale_mult = 1.0

            # Latency DR: obs latency + action latency
            if self.cfg.dr_randomize_latency:
                self.dr_obs_latency_steps = int(self.np_random.integers(*self.cfg.dr_obs_latency_steps_range, endpoint=True))
                self.dr_action_latency_steps = int(self.np_random.integers(*self.cfg.dr_action_latency_steps_range, endpoint=True))
            else:
                self.dr_obs_latency_steps = 0
                self.dr_action_latency_steps = 0
        else:
            self.action_scale_mult = 1.0
            self.dr_obs_latency_steps = 0
            self.dr_action_latency_steps = 0
            self.dr_ball_mass = self.original_ball_mass if self.original_ball_mass is not None else 0.0027
            self.dr_ball_friction = 0.20
            self.dr_racket_friction = 0.35
            self.dr_gravity_z = -9.81
            self.dr_ball_solref_time = 0.003
            self.dr_ball_solref_damping = 0.80

        # Racket mount DR: randomize or restore (before creating MjData)
        if self.cfg.domain_randomization and self.cfg.dr_randomize_racket_mount:
            # Sample random offsets
            self.dr_racket_pos_offset = self.np_random.uniform(
                -self.cfg.dr_racket_pos_offset_m,
                self.cfg.dr_racket_pos_offset_m,
                size=3
            ).astype(np.float32)
            self.dr_racket_rot_offset = self.np_random.uniform(
                -self.cfg.dr_racket_rot_offset_rad,
                self.cfg.dr_racket_rot_offset_rad,
                size=3
            ).astype(np.float32)
            self.dr_racket_radius_offset = float(self.np_random.uniform(
                -self.cfg.dr_racket_radius_offset_m,
                self.cfg.dr_racket_radius_offset_m
            ))

            # Apply position offset
            if self.racket_body_id >= 0 and self.original_racket_body_pos is not None:
                self.model.body_pos[self.racket_body_id] = self.original_racket_body_pos + self.dr_racket_pos_offset

            # Apply rotation offset
            if self.racket_body_id >= 0 and self.original_racket_body_quat is not None:
                rot_quat = _euler_xyz_to_quat_wxyz(
                    self.dr_racket_rot_offset[0],
                    self.dr_racket_rot_offset[1],
                    self.dr_racket_rot_offset[2]
                )
                self.model.body_quat[self.racket_body_id] = _quat_mul_wxyz(
                    self.original_racket_body_quat,
                    rot_quat
                )

            # Apply radius offset
            for geom_id, original_size in self.original_racket_geom_sizes.items():
                new_radius = max(0.03, original_size[0] + self.dr_racket_radius_offset)
                self.model.geom_size[geom_id][0] = new_radius
        else:
            # Restore original values
            self.dr_racket_pos_offset = np.zeros(3, dtype=np.float32)
            self.dr_racket_rot_offset = np.zeros(3, dtype=np.float32)
            self.dr_racket_radius_offset = 0.0

            if self.racket_body_id >= 0 and self.original_racket_body_pos is not None:
                self.model.body_pos[self.racket_body_id] = self.original_racket_body_pos.copy()
            if self.racket_body_id >= 0 and self.original_racket_body_quat is not None:
                self.model.body_quat[self.racket_body_id] = self.original_racket_body_quat.copy()
            for geom_id, original_size in self.original_racket_geom_sizes.items():
                self.model.geom_size[geom_id] = original_size.copy()

        self.action_buffer = []
        self.obs_buffer = []

        self.data = mj.MjData(self.model)
        self.base_cmd[:] = 0.0
        self.data.ctrl[:] = self.default_ctrl
        for aid in self.base_aids:
            self.data.ctrl[aid] = 0.0
        for _ in range(700):
            mj.mj_step(self.model, self.data)

        self.arm_q_target = np.array([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float32)
        self.arm_cmd_q = self.arm_q_target.copy()
        self.arm_cmd_qvel[:] = 0.0
        self.base_cmd[:] = 0.0
        self.initial_base_pose = np.array(
            [
                self.data.qpos[self.base_x_qadr],
                self.data.qpos[self.base_y_qadr],
                self.data.qpos[self.base_yaw_qadr],
            ],
            dtype=np.float32,
        )
        self.prev_action[:] = 0.0

        r0 = self._racket_pos()
        self.racket_anchor = r0.copy()
        spawn_center_xy = r0[:2].copy()

        if self.waist_body_id >= 0:
            waist0 = np.array(self.data.xpos[self.waist_body_id], dtype=np.float32)
            self.chest_target_offset = (self.racket_anchor - waist0).astype(np.float32)
            self.chest_target_pos = self.racket_anchor.copy()
        else:
            self.chest_target_offset[:] = 0.0
            self.chest_target_pos = self.racket_anchor.copy()
        spawn_jitter = np.array([
            self.np_random.uniform(-float(self.cfg.ball_spawn_xy_jitter), float(self.cfg.ball_spawn_xy_jitter)),
            self.np_random.uniform(-float(self.cfg.ball_spawn_xy_jitter), float(self.cfg.ball_spawn_xy_jitter)),
            self.np_random.uniform(-float(self.cfg.ball_spawn_z_jitter), float(self.cfg.ball_spawn_z_jitter)),
        ], dtype=np.float32)
        spawn_center = np.array([spawn_center_xy[0], spawn_center_xy[1], r0[2] + self.cfg.ball_launch_height], dtype=np.float32)
        ball_init = spawn_center + spawn_jitter
        self.data.qpos[self.ball_qadr : self.ball_qadr + 3] = ball_init
        self.data.qpos[self.ball_qadr + 3 : self.ball_qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.data.qvel[self.ball_vadr : self.ball_vadr + 6] = 0.0
        self.data.qvel[self.ball_vadr + 0] = float(self.np_random.uniform(-float(self.cfg.ball_init_vxy_max), float(self.cfg.ball_init_vxy_max)))
        self.data.qvel[self.ball_vadr + 1] = float(self.np_random.uniform(-float(self.cfg.ball_init_vxy_max), float(self.cfg.ball_init_vxy_max)))
        self.data.qvel[self.ball_vadr + 2] = float(self.cfg.ball_init_vz)
        mj.mj_forward(self.model, self.data)

        # Initialize prev_arm_qvel to avoid spurious acceleration on first step
        self.prev_arm_qvel = np.array([self.data.qvel[i] for i in self.arm_vadr], dtype=np.float32)

        self.prev_ball_pos = self._ball_pos()
        self.prev_racket_pos = self._racket_pos()
        self.prev_contact = False
        self.hit_armed = True
        self.no_contact_steps = 0
        self.contact_hold_steps = 0
        self.pending_hit = False
        self.pending_hit_steps = 0
        if bool(self.cfg.hit_confirm_use_spawn_cube_band):
            spawn_band_xy = float(self.cfg.ball_spawn_xy_jitter)
            spawn_band_z = float(self.cfg.ball_spawn_z_jitter)
            spawn_band = max(spawn_band_xy, spawn_band_z)
            self.hit_confirm_abs_low = float(self.racket_anchor[2] + float(self.cfg.ball_launch_height) - spawn_band - float(self.cfg.hit_confirm_spawn_band_margin))
            self.hit_confirm_abs_high = float(self.racket_anchor[2] + float(self.cfg.ball_launch_height) + spawn_band + float(self.cfg.hit_confirm_spawn_band_margin))
        else:
            self.hit_confirm_abs_low = float(self.cfg.hit_confirm_abs_height)
            self.hit_confirm_abs_high = float("inf")
        self.hit_count = 0
        self.last_hit_time = -1.0
        self.last_hit_step = -1
        self.hit_intervals = []
        self.hit_cadence_rewards = []
        self.hit_min_interval_penalties = []
        self.last_counted_hit_time = -1.0
        self.last_count_gate_hit_time = -1.0
        self.confirmed_hit_count = 0
        self.ignored_fast_hit_count = 0
        self.counted_hit_intervals = []
        self.rewarded_hit_count = 0
        self.unrewarded_extra_hit_count = 0
        self.hit_reward_count_cap_active = self._get_hit_reward_count_cap()
        self.step_count = 0
        true_bpos = self._ball_pos()
        true_bvel = self._ball_vel_est(true_bpos)
        self.cached_ball_obs_pos, self.cached_ball_obs_vel = self._sample_ball_obs(true_bpos, true_bvel)
        self.last_ball_obs_step = 0
        # Initialize dropout state with true ball state as first valid observation
        self.ball_obs_dropout_remaining = 0
        self.ball_obs_valid_pos = self.cached_ball_obs_pos.copy()
        self.ball_obs_valid_vel = self.cached_ball_obs_vel.copy()
        self.ball_obs_age_seconds = 0.0
        self.ball_obs_dropout_steps_total = 0
        self.ball_obs_burst_count = 0

        initial_obs = self._get_obs()
        if self.cfg.domain_randomization and self.dr_obs_latency_steps > 0:
            self.obs_buffer = [initial_obs.copy() for _ in range(self.dr_obs_latency_steps + 1)]
        else:
            self.obs_buffer = []

        if self.cfg.domain_randomization and self.dr_action_latency_steps > 0:
            zero_action = np.zeros(self.ctrl_dim, dtype=np.float32)
            self.action_buffer = [zero_action.copy() for _ in range(self.dr_action_latency_steps + 1)]
        else:
            self.action_buffer = []


        return initial_obs, {}

    def step(self, action: np.ndarray):
        a = np.asarray(action, dtype=np.float32).reshape(self.ctrl_dim)
        a = np.clip(a, -1.0, 1.0)

        if self.cfg.domain_randomization and self.dr_action_latency_steps > 0:
            self.action_buffer.append(a.copy())
            if len(self.action_buffer) > self.dr_action_latency_steps + 1:
                self.action_buffer.pop(0)
            a_delayed = self.action_buffer[0].copy()
        else:
            a_delayed = a.copy()

        da = a_delayed - self.prev_action

        a_arm = a_delayed[: len(RIGHT_ARM_JOINTS)]

        # Command trajectory generator
        arm_acc_limit_rad_s2 = np.deg2rad(self.arm_acc_limit_deg_s2)
        arm_vel_limit_rad_s = np.deg2rad(self.arm_vel_limit_deg_s)

        effective_action_acc_scale = self.cfg.action_acc_scale * self.action_scale_mult
        desired_qdd_raw = a_arm * arm_acc_limit_rad_s2 * effective_action_acc_scale

        if self.cfg.arm_action_limiter:
            desired_qdd = np.clip(desired_qdd_raw, -arm_acc_limit_rad_s2, +arm_acc_limit_rad_s2)
            raw_cmd_qvel = self.arm_cmd_qvel + desired_qdd * self.dt
            cmd_qvel = np.clip(raw_cmd_qvel, -arm_vel_limit_rad_s, +arm_vel_limit_rad_s)

            acc_clipped = ~np.isclose(desired_qdd, desired_qdd_raw, atol=1e-8)
            vel_clipped = ~np.isclose(cmd_qvel, raw_cmd_qvel, atol=1e-8)
            arm_cmd_acc_clip_frac = float(np.sum(acc_clipped)) / len(RIGHT_ARM_JOINTS)
            arm_cmd_vel_clip_frac = float(np.sum(vel_clipped)) / len(RIGHT_ARM_JOINTS)

            acc_clip_diff = desired_qdd_raw - desired_qdd
            vel_clip_diff = raw_cmd_qvel - cmd_qvel
            arm_limiter_pen = float(
                np.mean(vel_clip_diff ** 2 / (arm_vel_limit_rad_s ** 2 + 1e-8))
                + np.mean(acc_clip_diff ** 2 / (arm_acc_limit_rad_s2 ** 2 + 1e-8))
            )
        else:
            desired_qdd = desired_qdd_raw
            raw_cmd_qvel = self.arm_cmd_qvel + desired_qdd * self.dt
            cmd_qvel = raw_cmd_qvel
            arm_cmd_acc_clip_frac = 0.0
            arm_cmd_vel_clip_frac = 0.0
            arm_limiter_pen = 0.0

        self.arm_cmd_q = np.clip(self.arm_cmd_q + cmd_qvel * self.dt, self.arm_lo, self.arm_hi)
        self.arm_cmd_qvel = cmd_qvel.copy()
        self.arm_q_target = self.arm_cmd_q.copy()

        arm_q_current = np.array([self.data.qpos[i] for i in self.arm_qadr], dtype=np.float32)
        arm_cmd_tracking_error = self.arm_cmd_q - arm_q_current
        arm_cmd_tracking_error_norm = float(np.linalg.norm(arm_cmd_tracking_error))
        arm_cmd_tracking_error_max_abs = float(np.max(np.abs(arm_cmd_tracking_error)))

        arm_cmd_vel_ratio = np.abs(self.arm_cmd_qvel) / np.maximum(arm_vel_limit_rad_s, 1e-8)
        arm_cmd_vel_ratio_mean = float(np.mean(arm_cmd_vel_ratio))
        arm_cmd_vel_ratio_max = float(np.max(arm_cmd_vel_ratio))
        arm_cmd_acc_ratio = np.abs(desired_qdd) / np.maximum(arm_acc_limit_rad_s2, 1e-8)
        arm_cmd_acc_ratio_mean = float(np.mean(arm_cmd_acc_ratio))
        arm_cmd_acc_ratio_max = float(np.max(arm_cmd_acc_ratio))

        # Backward-compat aliases
        arm_limiter_clip_frac = arm_cmd_vel_clip_frac
        arm_limiter_acc_clip_frac = arm_cmd_acc_clip_frac

        # Base command remains zero (right-arm-only)
        self.base_cmd[:] = 0.0

        for _ in range(self.cfg.frame_skip):
            self.data.ctrl[:] = self.default_ctrl
            for i, aid in enumerate(self.arm_aids):
                self.data.ctrl[aid] = self.arm_q_target[i]
            self.base_cmd[:] = 0.0
            for i, aid in enumerate(self.base_aids):
                self.data.ctrl[aid] = 0.0
            mj.mj_step(self.model, self.data)

        # Compute arm velocity/acceleration limit penalties
        arm_qvel_rad_s = np.array([self.data.qvel[i] for i in self.arm_vadr], dtype=np.float32)
        arm_qvel_deg_s = np.rad2deg(arm_qvel_rad_s)
        arm_qacc_deg_s2 = np.rad2deg((arm_qvel_rad_s - self.prev_arm_qvel) / max(self.dt, 1e-6))
        arm_vel_ratio = np.abs(arm_qvel_deg_s) / np.maximum(self.arm_vel_limit_deg_s, 1e-6)
        arm_acc_ratio = np.abs(arm_qacc_deg_s2) / np.maximum(self.arm_acc_limit_deg_s2, 1e-6)
        arm_vel_exceed = np.maximum(arm_vel_ratio - 1.0, 0.0)
        arm_acc_exceed = np.maximum(arm_acc_ratio - 1.0, 0.0)
        arm_vel_limit_pen = float(np.mean(arm_vel_exceed ** 2))
        arm_acc_limit_pen = float(np.mean(arm_acc_exceed ** 2))
        arm_vel_ratio_mean = float(np.mean(arm_vel_ratio))
        arm_vel_ratio_max = float(np.max(arm_vel_ratio))
        arm_acc_ratio_mean = float(np.mean(arm_acc_ratio))
        arm_acc_ratio_max = float(np.max(arm_acc_ratio))

        self.step_count += 1
        self.total_env_steps += 1
        bpos = self._ball_pos()
        rpos = self._racket_pos()
        rmat = self._racket_xmat()
        racket_normal = rmat[:, 2]
        bvel = self._ball_vel_est(bpos)
        rvel = self._racket_vel_est(rpos)
        rel = bpos - rpos
        rel_local = rmat.T @ rel
        racket_z_rel = float(rpos[2] - self.racket_anchor[2])
        gravity_mag = max(1e-6, float(np.linalg.norm(self.model.opt.gravity)))
        upward_vz = max(0.0, float(bvel[2]))
        predicted_apex_z = float(bpos[2] + (upward_vz * upward_vz) / (2.0 * gravity_mag))
        head_front_center, head_front_normal, head_front_plane_dist, head_front_path_dist, head_front_time_to_plane, head_front_ray_center_dist = self._ball_path_to_head_front_metrics(
            bpos, bvel
        )
        descending_intercept_xy_err = 0.0
        descending_intercept_reward = 0.0
        racket_up_cos = max(0.0, float(np.dot(racket_normal, np.array([0.0, 0.0, 1.0], dtype=np.float32))))
        flatness_err = max(0.0, float(self.cfg.hit_flatness_target_cos) - racket_up_cos)
        flatness_score = float(np.exp(-0.5 * (flatness_err / max(1e-6, float(self.cfg.hit_flatness_sigma))) ** 2))
        contact_center_dist = float(np.linalg.norm(rel_local[:2]))
        contact_center_score = float(
            np.exp(-0.5 * (contact_center_dist / max(1e-6, float(self.cfg.hit_center_local_sigma))) ** 2)
        )

        if self.waist_body_id >= 0:
            waist_now = np.array(self.data.xpos[self.waist_body_id], dtype=np.float32)
            chest_target = waist_now + self.chest_target_offset
        else:
            chest_target = self.racket_anchor
        self.chest_target_pos = chest_target.astype(np.float32)

        in_contact, other_ball_contact = self._ball_contact_flags()
        sep_dist = float(np.linalg.norm(rel))
        rel_speed = float(np.linalg.norm(bvel - rvel))
        if not in_contact:
            self.no_contact_steps += 1
            self.contact_hold_steps = 0
            if self.no_contact_steps >= int(self.cfg.hit_rearm_no_contact_steps) and sep_dist >= float(self.cfg.hit_rearm_distance):
                self.hit_armed = True
        else:
            self.no_contact_steps = 0
            self.contact_hold_steps += 1

        # Start a candidate hit only on re-contact edge, then confirm it only
        # when ball rises sufficiently above the paddle.
        hit_edge = bool(in_contact and (not self.prev_contact) and self.hit_armed and (not self.pending_hit))
        if hit_edge:
            self.pending_hit = True
            self.pending_hit_steps = 0
            self.hit_armed = False

        new_hit = False
        counted_hit = False
        ignored_fast_hit = False
        rewardable_hit = False
        fast_hit_pen = 0.0
        counted_hit_interval = 0.0
        count_gate_hit_interval = 0.0
        hit_cadence_rew = 0.0
        hit_min_interval_pen = 0.0
        hit_height_pen = 0.0
        failed_hit_pen = 0.0
        low_hit_pen = 0.0
        center_flat_hit_reward = 0.0
        if self.pending_hit:
            self.pending_hit_steps += 1
            min_launch_rel_z = max(float(self.cfg.hit_confirm_rel_height), 0.04)
            min_launch_apex_z = float(
                self.racket_anchor[2] + max(0.70 * float(self.cfg.target_height), min_launch_rel_z + 0.06)
            )
            launched_upward = bool(
                (not in_contact)
                and (float(rel[2]) >= min_launch_rel_z)
                and (float(bvel[2]) > 0.0)
                and (predicted_apex_z >= min_launch_apex_z)
            )
            if launched_upward:
                # Confirmed launch event (hit detection unchanged)
                self.confirmed_hit_count += 1
                self.pending_hit = False
                self.pending_hit_steps = 0
                current_time = float(self.step_count) * self.dt

                # Calculate count gate hit interval (for all confirmed hits)
                if self.last_count_gate_hit_time >= 0.0:
                    count_gate_hit_interval = current_time - self.last_count_gate_hit_time
                else:
                    count_gate_hit_interval = 0.0

                # Determine whether this confirmed hit should be counted (min count interval gate)
                # Use last_count_gate_hit_time instead of last_counted_hit_time to prevent
                # fast hits from bypassing the gate by inserting ignored hits
                if float(self.cfg.hit_min_count_interval) <= 0.0:
                    counted_hit = True
                elif self.last_count_gate_hit_time < 0.0:
                    counted_hit = True
                elif (current_time - self.last_count_gate_hit_time) >= float(self.cfg.hit_min_count_interval):
                    counted_hit = True
                else:
                    counted_hit = False

                if counted_hit:
                    self.hit_count += 1
                    new_hit = True

                    # Check if this counted hit should receive hit-specific rewards (reward cap)
                    cap = self.hit_reward_count_cap_active
                    if cap <= 0:
                        rewardable_hit = True
                    elif self.rewarded_hit_count < cap:
                        rewardable_hit = True
                    else:
                        rewardable_hit = False

                    if rewardable_hit:
                        self.rewarded_hit_count += 1
                        center_flat_hit_reward = float(self.cfg.center_flat_hit_reward_weight) * contact_center_score * flatness_score
                    else:
                        self.unrewarded_extra_hit_count += 1
                        center_flat_hit_reward = 0.0

                    if self.last_counted_hit_time >= 0.0:
                        counted_hit_interval = current_time - self.last_counted_hit_time
                        self.counted_hit_intervals.append(counted_hit_interval)
                    self.last_counted_hit_time = current_time
                else:
                    ignored_fast_hit = True
                    self.ignored_fast_hit_count += 1
                    if float(self.cfg.fast_hit_penalty_weight) > 0.0 and float(self.cfg.hit_min_count_interval) > 0.0:
                        dt_since_gate_hit = current_time - self.last_count_gate_hit_time
                        deficit = float(self.cfg.hit_min_count_interval) - dt_since_gate_hit
                        fast_hit_pen = float(self.cfg.fast_hit_penalty_weight) * (
                            (deficit / max(1e-6, float(self.cfg.hit_min_count_interval))) ** 2
                        )

                # Update count gate time for ALL confirmed hits (counted or ignored)
                # This prevents fast consecutive hits from bypassing the gate
                self.last_count_gate_hit_time = current_time

                # Hit cadence reward/penalty (based on confirmed hits, unchanged statistics)
                if self.last_hit_time >= 0.0:
                    # Not the first hit
                    hit_interval = current_time - self.last_hit_time
                    self.hit_intervals.append(hit_interval)

                    # Cadence reward (Gaussian around target interval)
                    if self.cfg.hit_cadence_reward_weight > 0.0:
                        dt_err = hit_interval - self.cfg.hit_cadence_target_interval
                        hit_cadence_rew = self.cfg.hit_cadence_reward_weight * float(
                            np.exp(-0.5 * (dt_err / max(1e-6, self.cfg.hit_cadence_sigma)) ** 2)
                        )
                    else:
                        hit_cadence_rew = 0.0

                    # Short interval penalty
                    if self.cfg.hit_min_interval_penalty_weight > 0.0 and hit_interval < self.cfg.hit_min_interval:
                        interval_deficit = self.cfg.hit_min_interval - hit_interval
                        hit_min_interval_pen = self.cfg.hit_min_interval_penalty_weight * (
                            (interval_deficit / max(1e-6, self.cfg.hit_min_interval)) ** 2)
                    else:
                        hit_min_interval_pen = 0.0

                    # Only log cadence reward/penalty to means when they actually contribute to
                    # reward (counted_hit and rewardable_hit). Ignored fast hits and unrewarded extra hits
                    # do not add to reward, so excluding them from means keeps TensorBoard stats consistent.
                    if counted_hit and rewardable_hit:
                        self.hit_cadence_rewards.append(hit_cadence_rew)
                        self.hit_min_interval_penalties.append(hit_min_interval_pen)
                    else:
                        # Ensure ignored fast hits and unrewarded extra hits do not leak cadence reward/penalty into reward
                        hit_cadence_rew = 0.0
                        hit_min_interval_pen = 0.0

                self.last_hit_time = current_time
                self.last_hit_step = self.step_count
            elif self.pending_hit_steps >= int(self.cfg.hit_confirm_max_steps):
                # Contact happened but did not produce a meaningful lift.
                self.pending_hit = False
                self.pending_hit_steps = 0
                failed_hit_pen = float(self.cfg.failed_hit_penalty_weight)

        posture_q = np.array([self.data.qpos[i] for i in self.posture_qadr], dtype=np.float32)
        posture_pen = float(np.mean(np.square(np.abs(posture_q - self.posture_targets))))
        base_pose = np.array(
            [
                self.data.qpos[self.base_x_qadr],
                self.data.qpos[self.base_y_qadr],
                self.data.qpos[self.base_yaw_qadr],
            ],
            dtype=np.float32,
        )
        base_pose_err = base_pose - self.initial_base_pose
        base_pose_err[2] = (base_pose_err[2] + np.pi) % (2.0 * np.pi) - np.pi
        base_pose_pen = float(np.sum(np.square(base_pose_err)))

        ball_height_reward = 0.0
        target_ball_z = float(self.racket_anchor[2] + float(self.cfg.target_height))
        target_hit_apex_z = float(self.racket_anchor[2] + float(self.cfg.hit_height_center))
        if upward_vz > 0.0:
            dz = float(predicted_apex_z - target_ball_z)
            ball_height_reward = float(np.exp(-7.0 * dz * dz))
        elif rel[2] > 0.05:
            dz = float(bpos[2] - target_ball_z)
            ball_height_reward = 0.25 * float(np.exp(-10.0 * dz * dz))

        racket_center_pen = float(np.sum(np.square(rpos - self.racket_anchor)))
        racket_xy_dist = float(np.linalg.norm((rpos - self.racket_anchor)[:2]))
        racket_xy_gauss = float(np.exp(-0.5 * (racket_xy_dist / max(1e-6, float(self.cfg.racket_xy_gauss_sigma))) ** 2))
        racket_xy_gauss_pen = 1.0 - racket_xy_gauss
        racket_chest_xy_pen = float(np.sum(np.square((rpos - chest_target)[:2])))
        racket_chest_z_pen = float((rpos[2] - chest_target[2]) ** 2)
        ball_anchor_xy_pen = float(np.sum(np.square((bpos - chest_target)[:2])))
        base_yaw = float(base_pose[2])
        base_to_ball_world = np.array([bpos[0] - base_pose[0], bpos[1] - base_pose[1]], dtype=np.float32)
        ball_base_x = float(np.cos(base_yaw) * base_to_ball_world[0] + np.sin(base_yaw) * base_to_ball_world[1])
        ball_base_x_excess = max(0.0, abs(ball_base_x) - float(self.cfg.ball_base_x_soft_limit))
        ball_base_x_pen = ball_base_x_excess * ball_base_x_excess
        c_yaw = float(np.cos(base_yaw))
        s_yaw = float(np.sin(base_yaw))
        ball_base_vx = float(c_yaw * bvel[0] + s_yaw * bvel[1])
        ball_base_vy = float(-s_yaw * bvel[0] + c_yaw * bvel[1])
        ball_base_vxy_pen = ball_base_vx * ball_base_vx + ball_base_vy * ball_base_vy
        ball_vxy_pen = float(np.sum(np.square(bvel[:2])))
        xy_track_pen = float(np.sum(np.square(rel[:2])))
        post_hit_ball_xy_dist = float(np.linalg.norm((bpos - chest_target)[:2]))
        apex_soft_excess = max(0.0, predicted_apex_z - (target_ball_z + float(self.cfg.apex_soft_limit_margin)))
        apex_soft_pen = float(self.cfg.apex_soft_penalty_weight) * apex_soft_excess * apex_soft_excess
        ball_xy_soft_excess = max(0.0, post_hit_ball_xy_dist - float(self.cfg.ball_xy_soft_limit_radius))
        ball_xy_soft_pen = 0.0
        if self.hit_count > 0 or upward_vz > 0.0 or float(bpos[2]) > float(self.racket_anchor[2]):
            ball_xy_soft_pen = float(self.cfg.ball_xy_soft_penalty_weight) * ball_xy_soft_excess * ball_xy_soft_excess
        post_hit_ball_xy_score = float(
            np.exp(-0.5 * (post_hit_ball_xy_dist / max(1e-6, float(self.cfg.post_hit_ball_xy_sigma))) ** 2)
        )
        post_hit_survival_reward = 0.0
        if self.hit_count > 0 and (bpos[2] >= self.racket_anchor[2] - 0.02):
            post_hit_survival_reward = (
                post_hit_ball_xy_score
                - float(self.cfg.post_hit_ball_vxy_penalty_weight) * float(np.sum(np.square(bvel[:2])))
            )
        if self.hit_count > 0 and float(bvel[2]) < -1e-4 and float(bpos[2]) > float(rpos[2]):
            drop_dist = float(bpos[2] - rpos[2])
            vz_abs = max(1e-5, -float(bvel[2]))
            time_to_racket = drop_dist / vz_abs
            projected_ball_xy = bpos[:2] + bvel[:2] * time_to_racket
            descending_intercept_xy_err = float(np.linalg.norm(projected_ball_xy - rpos[:2]))
            descending_intercept_reward = float(
                np.exp(
                    -0.5
                    * (
                        descending_intercept_xy_err
                        / max(1e-6, float(self.cfg.descending_intercept_sigma))
                    )
                    ** 2
                )
            )
        torque_pen = float(np.mean(np.square(self.data.actuator_force[self.arm_aids])))
        sticky_contact = bool(
            in_contact
            and self.contact_hold_steps >= int(self.cfg.stick_min_contact_steps)
            and sep_dist <= float(self.cfg.stick_rel_dist_thresh)
            and rel_speed <= float(self.cfg.stick_rel_speed_thresh)
        )
        stick_pen = 0.0
        if sticky_contact:
            hold_excess = 1.0 + float(max(0, self.contact_hold_steps - int(self.cfg.stick_min_contact_steps)))
            stick_pen = hold_excess * float(self.cfg.sticky_contact_penalty_growth)

        non_racket_contact_pen = float(self.cfg.non_racket_ball_contact_penalty_weight) if other_ball_contact else 0.0

        camera_metrics = self._ball_camera_metrics(bpos)

        flat_contact_pen = 0.0
        if in_contact:
            flat_contact_pen = float(self.cfg.contact_flatness_penalty_weight) * float(max(0.0, 1.0 - flatness_score))

        z_excess_up = max(0.0, racket_z_rel - float(self.cfg.racket_z_band_up))
        z_excess_down = max(0.0, -float(self.cfg.racket_z_band_down) - racket_z_rel)
        racket_z_band_pen = z_excess_up * z_excess_up + z_excess_down * z_excess_down

        up_drift_pen = 0.0
        if racket_z_rel > 0.0 and float(rvel[2]) > float(self.cfg.racket_up_drift_vel_thresh):
            up_drift_pen = racket_z_rel * float(max(0.0, rvel[2]))

        rel_height_bonus = float(
            np.exp(-0.5 * ((float(rel[2]) - float(self.cfg.rel_height_center)) / max(1e-6, float(self.cfg.rel_height_sigma))) ** 2)
        )

        dense_reward = 0.0
        dense_reward += 1.2 * ball_height_reward
        dense_reward += self.cfg.rel_height_bonus_weight * rel_height_bonus
        dense_reward -= 1.4 * xy_track_pen
        dense_reward -= 0.35 * racket_center_pen
        dense_reward -= self.cfg.posture_weight * posture_pen
        dense_reward -= self.cfg.base_pose_weight * base_pose_pen
        dense_reward -= self.cfg.torque_penalty_weight * torque_pen
        dense_reward -= self.cfg.stick_contact_penalty_weight * stick_pen
        dense_reward -= non_racket_contact_pen
        dense_reward -= self.cfg.racket_chest_xy_penalty_weight * racket_chest_xy_pen
        dense_reward -= self.cfg.racket_chest_z_penalty_weight * racket_chest_z_pen
        dense_reward -= self.cfg.ball_anchor_xy_penalty_weight * ball_anchor_xy_pen
        dense_reward -= self.cfg.ball_base_x_penalty_weight * ball_base_x_pen
        dense_reward -= self.cfg.ball_base_vxy_penalty_weight * ball_base_vxy_pen
        dense_reward -= self.cfg.ball_vxy_penalty_weight * ball_vxy_pen
        dense_reward -= apex_soft_pen
        dense_reward -= ball_xy_soft_pen
        dense_reward += self.cfg.post_hit_survival_reward_weight * post_hit_survival_reward
        dense_reward += self.cfg.descending_intercept_reward_weight * descending_intercept_reward
        dense_reward += self.cfg.racket_xy_gauss_reward_weight * racket_xy_gauss
        dense_reward -= self.cfg.racket_xy_gauss_penalty_weight * racket_xy_gauss_pen
        dense_reward -= self.cfg.racket_z_soft_penalty_weight * racket_z_band_pen
        dense_reward -= self.cfg.racket_up_drift_penalty_weight * up_drift_pen
        dense_reward -= flat_contact_pen
        dense_reward -= self.cfg.action_penalty_weight * float(np.sum(np.square(a)))
        dense_reward -= self.cfg.action_delta_penalty_weight * float(np.sum(np.square(da)))
        dense_reward -= self.cfg.arm_vel_limit_penalty_weight * arm_vel_limit_pen
        dense_reward -= self.cfg.arm_acc_limit_penalty_weight * arm_acc_limit_pen
        dense_reward -= self.cfg.arm_limiter_penalty_weight * arm_limiter_pen

        camera_dense_penalty = 0.0
        _cam_mode = self.cfg.camera_visibility_mode
        if _cam_mode == "box":
            camera_dense_penalty += self.cfg.camera_box_penalty_weight * camera_metrics["camera_box_pen"]
        elif _cam_mode == "frustum":
            camera_dense_penalty += self.cfg.camera_visibility_penalty_weight * camera_metrics["camera_frustum_pen"]
            camera_dense_penalty += self.cfg.camera_depth_penalty_weight * camera_metrics["camera_depth_pen"]
        elif _cam_mode == "pixel":
            camera_dense_penalty += self.cfg.camera_center_weight * camera_metrics["camera_pixel_center_pen"]
            camera_dense_penalty += self.cfg.camera_visibility_penalty_weight * camera_metrics["camera_pixel_margin_pen"]
            camera_dense_penalty += self.cfg.camera_depth_penalty_weight * camera_metrics["camera_depth_pen"]
            camera_dense_penalty += self.cfg.camera_top_margin_penalty_weight * camera_metrics["camera_top_margin_pen"]
            if self.cfg.camera_visible_penalty_weight > 0.0 and not camera_metrics["camera_visible"]:
                camera_dense_penalty += self.cfg.camera_visible_penalty_weight
        if self.cfg.camera_dense_penalty_clip > 0.0:
            camera_dense_penalty = min(camera_dense_penalty, float(self.cfg.camera_dense_penalty_clip))
        dense_reward -= camera_dense_penalty

        reward = dense_reward * self.dt
        if new_hit and rewardable_hit:
            hit_bonus = self.cfg.hit_reward_base + self.cfg.hit_reward_combo * float(min(self.hit_count, 12))
            center_gain = float(np.exp(-0.5 * (contact_center_dist / max(1e-6, float(self.cfg.hit_center_sigma))) ** 2))
            hit_bonus *= max(0.2, center_gain * flatness_score)
            hit_height_err = abs(float(predicted_apex_z) - target_hit_apex_z)
            hit_height_excess = max(0.0, hit_height_err - float(self.cfg.hit_height_tolerance))
            hit_height_pen = float(self.cfg.hit_height_penalty_weight) * hit_height_excess * hit_height_excess
            low_hit_deficit = max(0.0, (target_ball_z - float(self.cfg.low_hit_apex_margin)) - predicted_apex_z)
            low_hit_pen = float(self.cfg.low_hit_penalty_weight) * low_hit_deficit * low_hit_deficit
            reward += hit_bonus
            reward += center_flat_hit_reward
            reward += hit_cadence_rew
            reward -= hit_min_interval_pen
            if predicted_apex_z >= target_ball_z:
                height_bonus = float(np.exp(-10.0 * (predicted_apex_z - target_ball_z) * (predicted_apex_z - target_ball_z)))
                reward += 0.35 * height_bonus
            reward -= hit_height_pen
            reward -= low_hit_pen
        reward -= failed_hit_pen
        reward -= fast_hit_pen

        base_x = float(self.data.qpos[self.base_x_qadr])
        base_y = float(self.data.qpos[self.base_y_qadr])
        terminated = False
        termination_reasons: list[str] = []
        if (not np.isfinite(self.data.qpos).all()) or (not np.isfinite(self.data.qvel).all()):
            termination_reasons.append("nonfinite_state")
        if bpos[2] < 0.8:
            termination_reasons.append("ball_too_low")
        if bpos[2] > 1.9:
            termination_reasons.append("ball_too_high")
        if abs(bpos[0] - self.racket_anchor[0]) > 0.5:
            termination_reasons.append("ball_x_out_of_bounds")
        if abs(bpos[1] - self.racket_anchor[1]) > 0.5:
            termination_reasons.append("ball_y_out_of_bounds")
        if abs(base_x) > 2.6:
            termination_reasons.append("base_x_out_of_bounds")
        if abs(base_y) > 2.6:
            termination_reasons.append("base_y_out_of_bounds")
        if np.linalg.norm(rpos - self.racket_anchor) > 1.1:
            termination_reasons.append("racket_too_far_from_anchor")
        if racket_z_rel > float(self.cfg.racket_z_hard_limit_up):
            termination_reasons.append("racket_too_high")
        if racket_z_rel < -float(self.cfg.racket_z_hard_limit_down):
            termination_reasons.append("racket_too_low")
        if termination_reasons:
            terminated = True
            if self.hit_count > 0 and any(
                reason in termination_reasons
                for reason in ("ball_too_low", "ball_too_high", "ball_x_out_of_bounds", "ball_y_out_of_bounds")
            ):
                reward -= float(
                    self.cfg.termination_miss_penalty_base
                    + self.cfg.termination_miss_penalty_per_hit * float(self.hit_count)
                )
        truncated = self.step_count >= self.max_steps

        raw_obs = self._get_obs()
        if self.cfg.domain_randomization and self.dr_obs_latency_steps > 0:
            self.obs_buffer.append(raw_obs.copy())
            if len(self.obs_buffer) > self.dr_obs_latency_steps + 1:
                self.obs_buffer.pop(0)
            obs = self.obs_buffer[0].copy()
        else:
            obs = raw_obs

        self.prev_ball_pos = bpos.copy()
        self.prev_racket_pos = rpos.copy()
        self.prev_action = a_delayed.copy()
        self.prev_contact = in_contact

        info = {
            "ball_pos": bpos,
            "racket_pos": rpos,
            "ball_vel": bvel,
            "racket_vel": rvel,
            "xy_err": float(np.linalg.norm(rel[:2])),
            "z_err": float(abs(rel[2] - self.cfg.target_height)),
            "ball_rel_z": float(rel[2]),
            "ball_vz": float(bvel[2]),
            "ball_vxy": float(np.linalg.norm(bvel[:2])),
            "predicted_apex_z": float(predicted_apex_z),
            "predicted_apex_rel_z": float(predicted_apex_z - self.racket_anchor[2]),
            "new_hit": new_hit,
            "hit_count": int(self.hit_count),
            "hit_armed": bool(self.hit_armed),
            "no_contact_steps": int(self.no_contact_steps),
            "contact_hold_steps": int(self.contact_hold_steps),
            "pending_hit": bool(self.pending_hit),
            "pending_hit_steps": int(self.pending_hit_steps),
            "hit_confirm_abs_low": float(self.hit_confirm_abs_low),
            "hit_confirm_abs_high": float(self.hit_confirm_abs_high),
            "sticky_contact": bool(sticky_contact),
            "other_ball_contact": bool(other_ball_contact),
            "hit_height_pen": float(hit_height_pen),
            "failed_hit_pen": float(failed_hit_pen),
            "low_hit_pen": float(low_hit_pen),
            "contact_center_dist": float(contact_center_dist),
            "contact_center_score": float(contact_center_score),
            "racket_up_cos": float(racket_up_cos),
            "flatness_score": float(flatness_score),
            "center_flat_hit_reward": float(center_flat_hit_reward),
            "non_racket_contact_pen": float(non_racket_contact_pen),
            "flat_contact_pen": float(flat_contact_pen),
            "racket_z_rel": racket_z_rel,
            "racket_xy_dist": float(racket_xy_dist),
            "racket_xy_gauss": float(racket_xy_gauss),
            "racket_xy_gauss_pen": float(racket_xy_gauss_pen),
            "racket_chest_xy_pen": float(racket_chest_xy_pen),
            "racket_chest_z_pen": float(racket_chest_z_pen),
            "ball_anchor_xy_pen": float(ball_anchor_xy_pen),
            "ball_base_x": float(ball_base_x),
            "ball_base_x_excess": float(ball_base_x_excess),
            "ball_base_x_pen": float(ball_base_x_pen),
            "ball_base_vx": float(ball_base_vx),
            "ball_base_vy": float(ball_base_vy),
            "ball_base_vxy_pen": float(ball_base_vxy_pen),
            "ball_vxy_pen": float(ball_vxy_pen),
            "apex_soft_excess": float(apex_soft_excess),
            "apex_soft_pen": float(apex_soft_pen),
            "ball_xy_soft_excess": float(ball_xy_soft_excess),
            "ball_xy_soft_pen": float(ball_xy_soft_pen),
            "post_hit_ball_xy_score": float(post_hit_ball_xy_score),
            "post_hit_survival_reward": float(post_hit_survival_reward),
            "descending_intercept_xy_err": float(descending_intercept_xy_err),
            "descending_intercept_reward": float(descending_intercept_reward),
            "chest_target_pos": np.array(self.chest_target_pos, dtype=np.float32),
            "head_front_center": np.array(head_front_center, dtype=np.float32),
            "head_front_normal": np.array(head_front_normal, dtype=np.float32),
            "head_front_plane_dist": float(head_front_plane_dist),
            "head_front_path_dist": float(head_front_path_dist),
            "head_front_time_to_plane": float(head_front_time_to_plane),
            "head_front_ray_center_dist": float(head_front_ray_center_dist),
            "racket_z_band_pen": float(racket_z_band_pen),
            "up_drift_pen": float(up_drift_pen),
            "rel_height_bonus": float(rel_height_bonus),
            "posture_pen": posture_pen,
            "base_pose_pen": base_pose_pen,
            "ball_height_reward": ball_height_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "termination_reasons": tuple(termination_reasons),
            "termination_reason": termination_reasons[0] if termination_reasons else "",
            "dr_ball_mass": float(self.dr_ball_mass),
            "dr_ball_friction": float(self.dr_ball_friction),
            "dr_racket_friction": float(self.dr_racket_friction),
            "dr_gravity_z": float(self.dr_gravity_z),
            "dr_action_scale_mult": float(self.action_scale_mult),
            "dr_obs_latency_steps": int(self.dr_obs_latency_steps),
            "dr_action_latency_steps": int(self.dr_action_latency_steps),
            "camera_available": bool(camera_metrics["camera_available"]),
            "camera_visibility_mode": str(self.cfg.camera_visibility_mode),
            "ball_cam_x": float(camera_metrics["ball_cam_x"]),
            "ball_cam_y": float(camera_metrics["ball_cam_y"]),
            "ball_cam_z": float(camera_metrics["ball_cam_z"]),
            "ball_pixel_u": float(camera_metrics["ball_pixel_u"]),
            "ball_pixel_v": float(camera_metrics["ball_pixel_v"]),
            "camera_in_front": bool(camera_metrics["camera_in_front"]),
            "camera_in_depth": bool(camera_metrics["camera_in_depth"]),
            "camera_in_frustum": bool(camera_metrics["camera_in_frustum"]),
            "camera_in_image": bool(camera_metrics["camera_in_image"]),
            "camera_in_margin": bool(camera_metrics["camera_in_margin"]),
            "camera_visible": bool(camera_metrics["camera_visible"]),
            "camera_reward_dense": float(-camera_dense_penalty * self.dt),
            "camera_pixel_center_pen": float(camera_metrics["camera_pixel_center_pen"]),
            "camera_pixel_margin_pen": float(camera_metrics["camera_pixel_margin_pen"]),
            "camera_top_margin_pen": float(camera_metrics["camera_top_margin_pen"]),
            "camera_frustum_pen": float(camera_metrics["camera_frustum_pen"]),
            "camera_depth_pen": float(camera_metrics["camera_depth_pen"]),
            "camera_box_pen": float(camera_metrics["camera_box_pen"]),
            "arm_vel_limit_pen": float(arm_vel_limit_pen),
            "arm_acc_limit_pen": float(arm_acc_limit_pen),
            "arm_vel_ratio_mean": float(arm_vel_ratio_mean),
            "arm_vel_ratio_max": float(arm_vel_ratio_max),
            "arm_acc_ratio_mean": float(arm_acc_ratio_mean),
            "arm_acc_ratio_max": float(arm_acc_ratio_max),
            "arm_qvel_deg_s": arm_qvel_deg_s,
            "arm_qacc_deg_s2": arm_qacc_deg_s2,
            "arm_action_limiter_enabled": bool(self.cfg.arm_action_limiter),
            "arm_cmd_vel_ratio_mean": float(arm_cmd_vel_ratio_mean),
            "arm_cmd_vel_ratio_max": float(arm_cmd_vel_ratio_max),
            "arm_cmd_acc_ratio_mean": float(arm_cmd_acc_ratio_mean),
            "arm_cmd_acc_ratio_max": float(arm_cmd_acc_ratio_max),
            "arm_cmd_acc_clip_frac": float(arm_cmd_acc_clip_frac),
            "arm_cmd_vel_clip_frac": float(arm_cmd_vel_clip_frac),
            "arm_cmd_tracking_error_norm": float(arm_cmd_tracking_error_norm),
            "arm_cmd_tracking_error_max_abs": float(arm_cmd_tracking_error_max_abs),
            "arm_limiter_delta_clip_frac": float(arm_limiter_clip_frac),
            "arm_limiter_acc_clip_frac": float(arm_limiter_acc_clip_frac),
            "arm_limiter_clip_frac": float(arm_limiter_clip_frac),
            "arm_limiter_pen": float(arm_limiter_pen),
            "arm_limiter_penalty": float(self.cfg.arm_limiter_penalty_weight * arm_limiter_pen),
            "hit_cadence_rew": float(hit_cadence_rew) if new_hit else 0.0,
            "hit_min_interval_pen": float(hit_min_interval_pen) if new_hit else 0.0,
            "hit_interval": float(self.hit_intervals[-1]) if len(self.hit_intervals) > 0 and new_hit else 0.0,
            "hit_interval_mean": float(np.mean(self.hit_intervals)) if len(self.hit_intervals) > 0 else 0.0,
            "hit_interval_min": float(np.min(self.hit_intervals)) if len(self.hit_intervals) > 0 else 0.0,
            "short_hit_interval_frac": float(np.mean([1.0 if dt < self.cfg.hit_min_interval else 0.0 for dt in self.hit_intervals])) if len(self.hit_intervals) > 0 else 0.0,
            "hit_rate_hz_mean": float(self.hit_count) / float(self.step_count * self.dt) if self.step_count > 0 else 0.0,
            "hit_cadence_rew_mean": float(np.mean(self.hit_cadence_rewards)) if len(self.hit_cadence_rewards) > 0 else 0.0,
            "hit_min_interval_pen_mean": float(np.mean(self.hit_min_interval_penalties)) if len(self.hit_min_interval_penalties) > 0 else 0.0,
            "counted_hit": bool(counted_hit),
            "ignored_fast_hit": bool(ignored_fast_hit),
            "fast_hit_pen": float(fast_hit_pen),
            "counted_hit_interval": float(counted_hit_interval),
            "count_gate_hit_interval": float(count_gate_hit_interval),
            "confirmed_hit_count": int(self.confirmed_hit_count),
            "ignored_fast_hit_count": int(self.ignored_fast_hit_count),
            "ignored_fast_hit_frac": float(self.ignored_fast_hit_count) / float(max(1, self.confirmed_hit_count)),
            "counted_hit_interval_mean": float(np.mean(self.counted_hit_intervals)) if len(self.counted_hit_intervals) > 0 else 0.0,
            "counted_hit_interval_min": float(np.min(self.counted_hit_intervals)) if len(self.counted_hit_intervals) > 0 else 0.0,
            "hit_reward_count_cap": int(self.hit_reward_count_cap_active),
            "rewarded_hit_count": int(self.rewarded_hit_count),
            "unrewarded_extra_hit_count": int(self.unrewarded_extra_hit_count),
            "hit_reward_cap_reached": bool(self.hit_reward_count_cap_active > 0 and self.rewarded_hit_count >= self.hit_reward_count_cap_active),
            "rewardable_hit": bool(rewardable_hit),
            "dr_racket_mount_enabled": bool(self.cfg.domain_randomization and self.cfg.dr_randomize_racket_mount),
            "dr_racket_pos_offset": np.array(self.dr_racket_pos_offset, dtype=np.float32),
            "dr_racket_rot_offset": np.array(self.dr_racket_rot_offset, dtype=np.float32),
            "dr_racket_radius_offset": float(self.dr_racket_radius_offset),
            "ball_obs_age": float(self.ball_obs_age_seconds),
            "ball_obs_dropout_active": bool(self.ball_obs_dropout_remaining > 0 or self.ball_obs_age_seconds > 0),
            "ball_obs_dropout_steps_total": int(self.ball_obs_dropout_steps_total),
            "ball_obs_burst_count": int(self.ball_obs_burst_count),
        }

        # Update prev_arm_qvel for next step's acceleration calculation
        self.prev_arm_qvel = arm_qvel_rad_s.copy()

        return obs, float(reward), terminated, truncated, info
