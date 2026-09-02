"""Real-ball driven simulation mirror for sim2real analysis.

This module intentionally keeps the mirror independent from the real control
path.  The mirror consumes the same policy-facing ball state as the robot
controller, but all robot joint/racket state comes from its own MuJoCo model
with position actuators enabled.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mujoco as mj
    _MUJOCO_AVAILABLE = True
except Exception as exc:  # noqa: BLE001 - surfaced in diagnostics.
    mj = None
    _MUJOCO_AVAILABLE = False
    _MUJOCO_IMPORT_ERROR = exc
else:
    _MUJOCO_IMPORT_ERROR = None

try:
    import cv2
    _CV2_AVAILABLE = True
except Exception as exc:  # noqa: BLE001 - rendering is optional.
    cv2 = None
    _CV2_AVAILABLE = False
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

from pingpong_controller.mjx_rl_policy import (
    Gpu1QaccFtV153PolicyController,
    Gpu0QvelRealRmpV85PolicyController,
    MjxRLPolicyController,
    Simcode2IdealPd67PolicyController,
    Simcode3Actuator67PolicyController,
    Simcode4Servo67PolicyController,
    Simcode5PureActuator67PolicyController,
    Simcode6Qvel67PolicyController,
    inspect_mjx_checkpoint_metadata,
)
from pingpong_controller.rl_policy import (
    OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85,
    OBSERVATION_PROFILE_GPU1_QACC_FT_V153,
    OBSERVATION_PROFILE_LEGACY_BUG_COMPAT,
    OBSERVATION_PROFILE_SIMCODE2_IDEALPD67,
    OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67,
    OBSERVATION_PROFILE_SIMCODE4_SERVO67,
    OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67,
    OBSERVATION_PROFILE_SIMCODE6_QVEL67,
    RIGHT_ARM_JOINTS,
    TARGET_DEGREES,
    normalize_observation_profile,
    uses_corrected_67d_observation,
)

try:
    from pingpong_replay.compensation import CommandCompensator
    _COMPENSATION_AVAILABLE = True
except Exception as exc:  # noqa: BLE001 - simulation still runs without it.
    CommandCompensator = None
    _COMPENSATION_AVAILABLE = False
    _COMPENSATION_IMPORT_ERROR = exc
else:
    _COMPENSATION_IMPORT_ERROR = None

try:
    from pingpong_replay.delay_control import (
        command_buffer_length as _command_buffer_length,
        delay_steps_from_tau as _delay_steps_from_tau,
        push_command_buffer as _push_command_buffer,
    )
except Exception:  # noqa: BLE001 - keep the mirror usable without replay pkg.
    def _delay_steps_from_tau(tau_s: float, dt: float) -> int:
        if not np.isfinite(tau_s) or not np.isfinite(dt) or dt <= 0.0:
            return 0
        return max(0, int(round(float(tau_s) / float(dt))))

    def _command_buffer_length(
        delay_max_ms: float,
        dt: float,
        extra_steps: int = 0,
    ) -> int:
        max_steps = _delay_steps_from_tau(
            max(0.0, float(delay_max_ms)) * 1e-3, dt)
        return max(1, max_steps + max(0, int(extra_steps)) + 1)

    def _push_command_buffer(
        command_buffer: np.ndarray,
        q_ref_latest: np.ndarray,
        delay_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        buf = np.asarray(command_buffer, dtype=np.float32)
        q = np.asarray(q_ref_latest, dtype=np.float32).reshape(buf.shape[1])
        new_buf = np.concatenate([buf[1:], q[None, :]], axis=0)
        steps = int(np.clip(delay_steps, 0, new_buf.shape[0] - 1))
        return new_buf, new_buf[new_buf.shape[0] - 1 - steps].copy()


@dataclass(frozen=True)
class SimMirrorConfig:
    model_path: Path
    policy_xml_path: Path
    sim_xml_path: Path
    policy_obs_dim: int
    control_dt: float
    backend: str = "mujoco"
    frame_rate_hz: float = 30.0
    render_width: int = 640
    render_height: int = 480
    enable_video_frames: bool = True
    use_mujoco_renderer: bool = True
    enable_compensation: bool = True
    compensation_mode: str = "inverse_mpc"
    comp_delay_ms: float = 72.0
    comp_actuator_tau_s: float = 0.074
    comp_actuator_gain: float = 1.0
    comp_mpc_beta: float = 1.2
    comp_mpc_delay_scale: float = 1.05
    comp_mpc_tau_scale: float = 0.75
    comp_mpc_horizon_steps: int = 6
    comp_mpc_tracking_weight: float = 1.0
    comp_mpc_nominal_weight: float = 0.25
    comp_mpc_delta_weight: float = 0.05
    comp_mpc_max_delta_rad: float = 0.0
    rl_use_valid_ball_obs_age: bool = True
    mjx_domain_randomization: bool = False
    mjx_zero_ball_obs_noise: bool = True
    policy_profile: str = OBSERVATION_PROFILE_LEGACY_BUG_COMPAT
    policy_model: str = "v85"
    simcode2_model_path: Path | None = None
    simcode2_policy_xml_path: Path | None = None
    simcode2_sim_xml_path: Path | None = None
    simcode3_model_path: Path | None = None
    simcode3_policy_xml_path: Path | None = None
    simcode3_sim_xml_path: Path | None = None
    simcode4_model_path: Path | None = None
    simcode4_policy_xml_path: Path | None = None
    simcode4_sim_xml_path: Path | None = None
    simcode5_model_path: Path | None = None
    simcode5_policy_xml_path: Path | None = None
    simcode5_sim_xml_path: Path | None = None
    simcode6_model_path: Path | None = None
    simcode6_policy_xml_path: Path | None = None
    simcode6_sim_xml_path: Path | None = None
    simcode7_model_path: Path | None = None
    simcode7_v85_model_path: Path | None = None
    simcode7_v91_model_path: Path | None = None
    simcode7_v97_model_path: Path | None = None
    simcode7_policy_xml_path: Path | None = None
    simcode7_sim_xml_path: Path | None = None
    simcode8_policy_xml_path: Path | None = None
    simcode8_sim_xml_path: Path | None = None
    gpu1_qacc_ft_model_path: Path | None = None
    gpu1_qacc_ft_policy_xml_path: Path | None = None
    gpu1_qacc_ft_sim_xml_path: Path | None = None


def normalize_policy_profile(value: object) -> str:
    """Return the canonical policy profile shared with real deployment."""
    return normalize_observation_profile(value)


def sim_mirror_config_for_policy_profile(
    base_cfg: SimMirrorConfig,
    profile: object,
) -> SimMirrorConfig:
    """Resolve a mirror config without mutating the legacy path defaults.

    The primary model/XML fields remain the legacy profile.  Selecting the
    simcode2 profile explicitly swaps all three together so a new policy can
    never be paired with the legacy policy FK XML or simulation plant XML.
    """
    canonical = normalize_policy_profile(profile)
    if canonical == OBSERVATION_PROFILE_LEGACY_BUG_COMPAT:
        return replace(base_cfg, policy_profile=canonical)

    if canonical == OBSERVATION_PROFILE_SIMCODE2_IDEALPD67:
        alternate_paths = {
            "model_path": base_cfg.simcode2_model_path,
            "policy_xml_path": base_cfg.simcode2_policy_xml_path,
            "sim_xml_path": base_cfg.simcode2_sim_xml_path,
        }
    elif canonical == OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67:
        alternate_paths = {
            "model_path": base_cfg.simcode3_model_path,
            "policy_xml_path": base_cfg.simcode3_policy_xml_path,
            "sim_xml_path": base_cfg.simcode3_sim_xml_path,
        }
    elif canonical == OBSERVATION_PROFILE_SIMCODE4_SERVO67:
        alternate_paths = {
            "model_path": base_cfg.simcode4_model_path,
            "policy_xml_path": base_cfg.simcode4_policy_xml_path,
            "sim_xml_path": base_cfg.simcode4_sim_xml_path,
        }
    elif canonical == OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67:
        alternate_paths = {
            "model_path": base_cfg.simcode5_model_path,
            "policy_xml_path": base_cfg.simcode5_policy_xml_path,
            "sim_xml_path": base_cfg.simcode5_sim_xml_path,
        }
    elif canonical == OBSERVATION_PROFILE_SIMCODE6_QVEL67:
        alternate_paths = {
            "model_path": base_cfg.simcode6_model_path,
            "policy_xml_path": base_cfg.simcode6_policy_xml_path,
            "sim_xml_path": base_cfg.simcode6_sim_xml_path,
        }
    elif canonical == OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85:
        policy_model = str(getattr(
            base_cfg, "policy_model", "v85") or "v85").strip().lower()
        if policy_model not in {
                "v85", "v91", "v97", "v97_source_clock"}:
            raise ValueError(
                f"unknown GPU0-QVEL/REAL-RMP mirror model: {policy_model!r}")
        selected_model_path = {
            "v85": base_cfg.simcode7_v85_model_path,
            "v91": base_cfg.simcode7_v91_model_path,
            "v97": base_cfg.simcode7_v97_model_path,
            # The source clock is a real-observation adapter.  The independent
            # simulator retains the checkpoint's one-new-state-per-5-ms
            # training clock while using the same authenticated V97 assets.
            "v97_source_clock": base_cfg.simcode7_v97_model_path,
        }[policy_model]
        alternate_paths = {
            "model_path": (
                selected_model_path or base_cfg.simcode7_model_path),
            "policy_xml_path": (
                base_cfg.simcode8_policy_xml_path
                if policy_model in {"v97", "v97_source_clock"}
                else base_cfg.simcode7_policy_xml_path),
            "sim_xml_path": (
                base_cfg.simcode8_sim_xml_path
                if policy_model in {"v97", "v97_source_clock"}
                else base_cfg.simcode7_sim_xml_path),
        }
    elif canonical == OBSERVATION_PROFILE_GPU1_QACC_FT_V153:
        alternate_paths = {
            "model_path": base_cfg.gpu1_qacc_ft_model_path,
            "policy_xml_path": base_cfg.gpu1_qacc_ft_policy_xml_path,
            "sim_xml_path": base_cfg.gpu1_qacc_ft_sim_xml_path,
        }
    else:
        raise ValueError(f"unsupported sim mirror policy profile: {canonical}")
    missing = [name for name, value in alternate_paths.items() if value is None]
    if missing:
        raise ValueError(
            f"{canonical} mirror is missing dedicated paths for: "
            + ", ".join(missing))
    return replace(
        base_cfg,
        policy_profile=canonical,
        policy_obs_dim=(
            57
            if canonical == OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85
            else 67),
        rl_use_valid_ball_obs_age=(canonical not in {
            OBSERVATION_PROFILE_SIMCODE4_SERVO67,
            OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67,
            OBSERVATION_PROFILE_SIMCODE6_QVEL67,
        }),
        # Simcode4 still requires its original MJX-only planner ordering.
        # Simcode5/6 and GPU0 RMP reconstruct their checkpoint plants on CPU.
        backend=(
            "mjx"
            if canonical == OBSERVATION_PROFILE_SIMCODE4_SERVO67
            else "mujoco"
            if canonical == OBSERVATION_PROFILE_GPU1_QACC_FT_V153
            else base_cfg.backend),
        model_path=Path(alternate_paths["model_path"]),
        policy_xml_path=Path(alternate_paths["policy_xml_path"]),
        sim_xml_path=Path(alternate_paths["sim_xml_path"]),
    )


class _LoggerShim:
    def info(self, msg: str) -> None:
        print(f"[SimMirror][INFO] {msg}")

    def warn(self, msg: str) -> None:
        print(f"[SimMirror][WARN] {msg}")

    def error(self, msg: str) -> None:
        print(f"[SimMirror][ERROR] {msg}")


def _finite_vec(value: Any, length: int, default: float = 0.0) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=np.float32).reshape(length)
    except Exception:
        arr = np.full((length,), float(default), dtype=np.float32)
    if not np.all(np.isfinite(arr)):
        arr = np.full((length,), float(default), dtype=np.float32)
    return arr


def _rounded(values: Any, digits: int = 6) -> list[float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return [round(float(v), digits) for v in arr]


def _patch_robot_xml_with_ball(xml_path: Path) -> Path:
    """Create a temporary XML with the ping-pong ball body added.

    This mirrors the training helper, but avoids importing the full training
    environment from the interactive backend.
    """
    root = ET.parse(xml_path).getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(root, "compiler")
    meshdir = compiler.get("meshdir")
    if meshdir:
        meshdir_path = Path(meshdir).expanduser()
        if meshdir_path.is_absolute() and meshdir_path.exists():
            compiler.set("meshdir", str(meshdir_path))
        else:
            compiler.set("meshdir", str((xml_path.parent / meshdir).resolve()))
    else:
        compiler.set("meshdir", str((xml_path.parent / "meshes").resolve()))

    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    if not any(b.get("name") == "pingpong_ball"
               for b in worldbody.findall("body")):
        ball_body = ET.SubElement(
            worldbody, "body", name="pingpong_ball", pos="0.35 -0.12 1.05")
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
    for body in worldbody.iter("body"):
        if body.get("name") in {"head21", "head22", "head23"}:
            for geom in body.findall("geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")

    tmp = tempfile.NamedTemporaryFile(
        prefix="pingpong_sim_mirror_", suffix=".xml", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    ET.ElementTree(root).write(tmp_path, encoding="utf-8", xml_declaration=False)
    return tmp_path


def _quat_wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float32).reshape(4)
    return np.asarray([q[1], q[2], q[3], q[0]], dtype=np.float32)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_simcode_mjx_env(
    policy_profile: object = OBSERVATION_PROFILE_LEGACY_BUG_COMPAT,
) -> tuple[Any, Any]:
    canonical = normalize_policy_profile(policy_profile)
    source_dir = {
        OBSERVATION_PROFILE_LEGACY_BUG_COMPAT: "simcode",
        OBSERVATION_PROFILE_SIMCODE2_IDEALPD67: "simcode2",
        OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67: "simcode3",
        OBSERVATION_PROFILE_SIMCODE4_SERVO67: "simcode4",
        OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67: "simcode5",
        OBSERVATION_PROFILE_SIMCODE6_QVEL67: "simcode6",
        OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85: "simcode7",
    }[canonical]
    rl_sim_dir = (
        _repo_root()
        / source_dir
        / "pingpong_moz1-main"
        / "pingpong_controller"
        / "tools"
        / "rl_sim"
    )
    module_path = rl_sim_dir / "mjx_juggle_env.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"{source_dir} MJX env not found: {module_path}")
    rl_sim_text = str(rl_sim_dir)
    # The selected tree must take precedence over a path left by the other
    # profile in the long-lived mirror subprocess.
    if rl_sim_text in sys.path:
        sys.path.remove(rl_sim_text)
    sys.path.insert(0, rl_sim_text)
    # Both source trees use bare imports for their local helpers.  A profile
    # switch in the long-lived mirror subprocess must not reuse the other
    # tree's cached helper modules.
    local_module_names = (
        "camera_calibration",
        "delay_control",
        "mjx_smoke",
        "recovered_rmp_jax",
        "rmp_training_evidence",
        "rl_juggle_env_random",
        "sim2real_bridger",
    )
    repo_root = _repo_root().resolve()
    local_roots = (
        repo_root / "simcode",
        repo_root / "simcode2",
        repo_root / "simcode3",
        repo_root / "simcode4",
        repo_root / "simcode5",
        repo_root / "simcode6",
        repo_root / "simcode7",
    )
    for module_name in local_module_names:
        loaded = sys.modules.get(module_name)
        loaded_file = getattr(loaded, "__file__", None)
        if loaded_file is None:
            continue
        try:
            loaded_path = Path(loaded_file).resolve()
            is_local = any(
                loaded_path.is_relative_to(root) for root in local_roots)
        except (OSError, RuntimeError, ValueError):
            is_local = False
        if is_local:
            sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(
        f"_pingpong_{source_dir}_mjx_juggle_env", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load simcode MJX env: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.MjxJuggleConfig, module.MjxJuggleEnv


def _encode_fallback_frame(cfg: SimMirrorConfig, snap: dict[str, Any], label: str) -> bytes | None:
    if not _CV2_AVAILABLE:
        return None
    width = max(320, int(cfg.render_width))
    height = max(240, int(cfg.render_height))
    img = np.full((height, width, 3), 245, dtype=np.uint8)
    margin = 48
    x_min, x_max = -0.55, 0.55
    z_min, z_max = 0.45, 1.75

    def project(p):
        x = float(p[0])
        z = float(p[2])
        u = margin + (x - x_min) / (x_max - x_min) * (width - 2 * margin)
        v = height - margin - (z - z_min) / (z_max - z_min) * (height - 2 * margin)
        return int(np.clip(u, 0, width - 1)), int(np.clip(v, 0, height - 1))

    cv2.rectangle(img, (margin, margin), (width - margin, height - margin), (190, 200, 210), 1)
    ball = _finite_vec(snap.get("ball"), 3)
    racket = _finite_vec(snap.get("racket"), 3)
    bu, bv = project(ball)
    ru, rv = project(racket)
    cv2.line(img, (ru, rv), (bu, bv), (170, 170, 170), 1)
    cv2.circle(img, (ru, rv), 8, (174, 121, 78), -1)
    cv2.circle(img, (bu, bv), 7, (43, 126, 242), -1)
    cv2.putText(
        img,
        label,
        (16, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (35, 45, 60),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"q0={float(_finite_vec(snap.get('jointPosition'), 7)[0]):+.2f} action={max(np.abs(_finite_vec(snap.get('policyAction'), 7))):.2f}",
        (16, height - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (70, 80, 95),
        1,
        cv2.LINE_AA,
    )
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return enc.tobytes() if ok else None


class RealBallSimMirror:
    """A single-env MuJoCo mirror driven by real policy-facing ball state."""

    def __init__(
        self,
        cfg: SimMirrorConfig,
        *,
        logger: Any | None = None,
    ) -> None:
        cfg = sim_mirror_config_for_policy_profile(cfg, cfg.policy_profile)
        self.cfg = cfg
        self.policy_profile = normalize_policy_profile(cfg.policy_profile)
        self._ideal_pd_profile = (
            self.policy_profile == OBSERVATION_PROFILE_SIMCODE2_IDEALPD67)
        self._servo_planner_profile = (
            self.policy_profile == OBSERVATION_PROFILE_SIMCODE4_SERVO67)
        self._pure_actuator_profile = (
            self.policy_profile in {
                OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67,
                OBSERVATION_PROFILE_SIMCODE6_QVEL67,
                OBSERVATION_PROFILE_GPU1_QACC_FT_V153,
            })
        self._real_rmp_profile = (
            self.policy_profile
            == OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85)
        self._corrected_67d_profile = uses_corrected_67d_observation(
            self.policy_profile)
        self._training_xml_profile = bool(
            self._corrected_67d_profile or self._real_rmp_profile)
        self._logger = logger or _LoggerShim()
        if not _MUJOCO_AVAILABLE:
            raise RuntimeError(f"mujoco unavailable: {_MUJOCO_IMPORT_ERROR}")
        self.sim_xml_path = Path(cfg.sim_xml_path).expanduser().resolve()
        self.checkpoint_env_cfg: dict[str, Any] = {}
        self.right_arm_pd_profile = "xml"
        if self._pure_actuator_profile or self._real_rmp_profile:
            self.checkpoint_env_cfg = dict(inspect_mjx_checkpoint_metadata(
                str(Path(cfg.model_path).expanduser().resolve())
            ).get("env_cfg") or {})
            self.right_arm_pd_profile = str(
                self.checkpoint_env_cfg.get("right_arm_pd_profile", "xml"))
        if self._training_xml_profile:
            from pingpong_controller.simcode2_xml import (
                build_simcode2_training_xml,
            )
            self.patched_xml_path = build_simcode2_training_xml(
                self.sim_xml_path,
                right_arm_pd_profile=self.right_arm_pd_profile,
            )
        else:
            self.patched_xml_path = _patch_robot_xml_with_ball(
                self.sim_xml_path)
        self.model = mj.MjModel.from_xml_path(str(self.patched_xml_path))
        self.data = mj.MjData(self.model)
        self.dt = float(cfg.control_dt)
        self.substeps = max(1, int(round(self.dt / max(self.model.opt.timestep, 1e-6))))
        policy_cls = {
            OBSERVATION_PROFILE_LEGACY_BUG_COMPAT: MjxRLPolicyController,
            OBSERVATION_PROFILE_SIMCODE2_IDEALPD67: (
                Simcode2IdealPd67PolicyController),
            OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67: (
                Simcode3Actuator67PolicyController),
            OBSERVATION_PROFILE_SIMCODE4_SERVO67: (
                Simcode4Servo67PolicyController),
            OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67: (
                Simcode5PureActuator67PolicyController),
            OBSERVATION_PROFILE_SIMCODE6_QVEL67: (
                Simcode6Qvel67PolicyController),
            OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85: (
                Gpu0QvelRealRmpV85PolicyController),
            OBSERVATION_PROFILE_GPU1_QACC_FT_V153: (
                Gpu1QaccFtV153PolicyController),
        }[self.policy_profile]
        self.policy = policy_cls(
            str(Path(cfg.model_path).expanduser().resolve()),
            dt=self.dt,
            robot_xml_path=str(Path(cfg.policy_xml_path).expanduser().resolve()),
            enable_delay_conditioning=(int(cfg.policy_obs_dim) == 67),
            use_valid_ball_obs_age=True,
            logger=self._logger,
        )

        self.pure_actuator_wn: np.ndarray | None = None
        self.pure_actuator_zeta: np.ndarray | None = None
        self.pure_actuator_gain: np.ndarray | None = None
        self.pure_actuator_delay_steps: np.ndarray | None = None
        if self._pure_actuator_profile:
            env_cfg = self.checkpoint_env_cfg
            model_name = str(env_cfg.get(
                "actuator_cmd_model", ""
            )).strip().lower().replace("-", "_")
            if model_name != "second_order":
                raise RuntimeError(
                    "pure-actuator MuJoCo mirror requires checkpoint "
                    f"actuator_cmd_model='second_order', got {model_name!r}")

            def _contract_vector(name: str) -> np.ndarray:
                values = np.asarray(env_cfg.get(name, ()), dtype=np.float32)
                if values.shape != (len(RIGHT_ARM_JOINTS),):
                    raise RuntimeError(
                        f"pure-actuator checkpoint {name} must contain "
                        f"{len(RIGHT_ARM_JOINTS)} values, got {values.shape}")
                if not np.all(np.isfinite(values)):
                    raise RuntimeError(
                        f"pure-actuator checkpoint {name} contains non-finite values")
                return values

            self.pure_actuator_wn = _contract_vector(
                "actuator_cmd_natural_frequency_rad_s")
            self.pure_actuator_zeta = _contract_vector(
                "actuator_cmd_damping_ratio")
            self.pure_actuator_gain = _contract_vector(
                "actuator_cmd_gain_per_joint")
            delay_ms = _contract_vector("actuator_cmd_delay_ms_per_joint")
            if (np.any(self.pure_actuator_wn <= 0.0)
                    or np.any(self.pure_actuator_zeta <= 0.0)
                    or np.any(self.pure_actuator_zeta >= 1.0)
                    or np.any(delay_ms < 0.0)):
                raise RuntimeError(
                    "invalid pure-actuator second-order checkpoint values")
            self.pure_actuator_delay_steps = np.rint(
                delay_ms * np.float32(1e-3 / self.dt)
            ).astype(np.int32)
            checkpoint_frame_skip = int(env_cfg.get(
                "frame_skip", self.substeps))
            checkpoint_dt = float(self.model.opt.timestep) * checkpoint_frame_skip
            if (checkpoint_frame_skip != self.substeps
                    or abs(checkpoint_dt - self.dt) > 1.0e-9):
                raise RuntimeError(
                    "pure-actuator CPU mirror control timing does not match the "
                    "checkpoint: "
                    f"checkpoint frame_skip={checkpoint_frame_skip}, "
                    f"dt={checkpoint_dt:.9f}s; mirror substeps={self.substeps}, "
                    f"dt={self.dt:.9f}s")

        self.arm_jids = self._ids(mj.mjtObj.mjOBJ_JOINT, RIGHT_ARM_JOINTS)
        self.arm_aids = self._ids(mj.mjtObj.mjOBJ_ACTUATOR, RIGHT_ARM_JOINTS)
        self.arm_qadr = np.asarray(
            [int(self.model.jnt_qposadr[jid]) for jid in self.arm_jids],
            dtype=np.int32,
        )
        self.arm_vadr = np.asarray(
            [int(self.model.jnt_dofadr[jid]) for jid in self.arm_jids],
            dtype=np.int32,
        )
        self.base_qadr = np.asarray([
            int(self.model.jnt_qposadr[mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_JOINT, name)])
            for name in ("base_x", "base_y", "base_yaw")
        ], dtype=np.int32)
        self.base_vadr = np.asarray([
            int(self.model.jnt_dofadr[mj.mj_name2id(
                self.model, mj.mjtObj.mjOBJ_JOINT, name)])
            for name in ("base_x", "base_y", "base_yaw")
        ], dtype=np.int32)
        self.racket_site_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_SITE, "right_ee_site")
        self.ball_joint_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_body_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, "pingpong_ball")
        self.ball_qadr = (
            int(self.model.jnt_qposadr[self.ball_joint_id])
            if self.ball_joint_id >= 0 else -1)
        self.ball_vadr = (
            int(self.model.jnt_dofadr[self.ball_joint_id])
            if self.ball_joint_id >= 0 else -1)
        if self.racket_site_id < 0:
            raise RuntimeError("simulation XML missing right_ee_site")

        self.rmp_planner: Any | None = None
        self.rmp_target_update_delay_steps = 0
        self.rmp_output_delay_steps: np.ndarray | None = None
        self.rmp_output_q_buffer: np.ndarray | None = None
        self.rmp_output_qd_buffer: np.ndarray | None = None
        self.rmp_output_qdd_buffer: np.ndarray | None = None
        self.rmp_controller_kp: np.ndarray | None = None
        self.rmp_controller_kv: np.ndarray | None = None
        self.rmp_acceleration_feedforward_s2: np.ndarray | None = None
        if self._real_rmp_profile:
            self._initialize_recovered_rmp()

        self.default_ctrl = self._make_default_ctrl()
        self.renderer = None
        self.render_error: str | None = None
        self.frame_seq = 0
        self.latest_frame: bytes | None = None
        self.latest_frame_info: dict[str, Any] = {"seq": 0, "ageS": None}
        self.next_frame_wall = 0.0
        self.compensator = None
        self.compensator_key = None
        self.compensation_error: str | None = None
        self.warm_arm_q: np.ndarray | None = None
        self.warm_arm_qvel: np.ndarray | None = None
        self.actuator_model_key: tuple[Any, ...] | None = None
        self.command_buffer: np.ndarray | None = None
        self.actuator_command_buffer: np.ndarray | None = None
        self.arm_q_ref_active: np.ndarray | None = None
        self.arm_actuator_q_ref_active: np.ndarray | None = None
        self.arm_applied_q: np.ndarray | None = None
        self.arm_applied_qvel: np.ndarray | None = None
        self.actuator_delay_steps = 0
        self.actuator_command_buffer_len = 1
        self.reset()
        self._logger.info(
            "simulation mirror ready | "
            f"policy_profile={self.policy_profile}, "
            f"xml={self.sim_xml_path.name}, substeps={self.substeps}, "
            f"dt={self.dt:.4f}s, renderer={bool(cfg.use_mujoco_renderer)}")

    def _ids(self, kind, names: list[str]) -> list[int]:
        out = [mj.mj_name2id(self.model, kind, name) for name in names]
        missing = [name for name, idx in zip(names, out) if idx < 0]
        if missing:
            raise RuntimeError(f"simulation XML missing names: {missing}")
        return out

    def _make_default_ctrl(self) -> np.ndarray:
        ctrl = np.zeros(self.model.nu, dtype=np.float32)
        for name, deg in TARGET_DEGREES.items():
            aid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                ctrl[aid] = float(np.deg2rad(deg))
        if self._training_xml_profile:
            # Corrected checkpoints carry their D455-aligned reset pose.
            # Seed the CPU plant at that pose before policy/history reset.
            for name, q_rad in zip(
                RIGHT_ARM_JOINTS, self.policy.initial_arm_q_rad,
            ):
                aid = mj.mj_name2id(
                    self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
                if aid >= 0:
                    ctrl[aid] = float(q_rad)
        for name in ("Base-X", "Base-Y", "Base-Yaw"):
            aid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
            if aid >= 0:
                ctrl[aid] = 0.0
        return ctrl

    def _checkpoint_vector(
        self,
        name: str,
        default: tuple[float, ...],
        *,
        dtype: Any = np.float64,
    ) -> np.ndarray:
        values = np.asarray(
            self.checkpoint_env_cfg.get(name, default), dtype=dtype)
        if values.shape != (len(RIGHT_ARM_JOINTS),) or not np.all(
                np.isfinite(values)):
            raise RuntimeError(
                f"GPU0 RMP checkpoint {name} must contain seven finite values")
        return values

    def _initialize_recovered_rmp(self) -> None:
        """Restore simcode7's nominal 1 kHz recovered-RMP plant on CPU."""
        try:
            from recovered_rmp import JointLimits, JointRMPPlanner, RMPParams
        except ModuleNotFoundError as exc:
            if exc.name != "recovered_rmp":
                raise
            # Source-tree launches do not necessarily put the workspace root
            # on sys.path (an absolute script path makes sys.path[0] the tools
            # directory). Resolve the bundled plant from the authenticated
            # checkpoint path instead of depending on the caller's cwd.
            checkpoint = Path(self.cfg.model_path).expanduser().resolve()
            candidates = (
                list(checkpoint.parents)
                + list(Path(__file__).resolve().parents)
                + [Path.cwd().resolve()]
            )
            workspace_root = next(
                (
                    root for root in candidates
                    if (root / "recovered_rmp" / "__init__.py").is_file()
                ),
                None,
            )
            if workspace_root is None:
                raise RuntimeError(
                    "GPU0 CPU mirror cannot locate bundled recovered_rmp"
                ) from exc
            root_text = str(workspace_root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            from recovered_rmp import JointLimits, JointRMPPlanner, RMPParams

        env_cfg = self.checkpoint_env_cfg
        if not bool(env_cfg.get("recovered_rmp_motion_mode", False)):
            raise RuntimeError(
                "GPU0 CPU mirror checkpoint does not enable recovered RMP")
        period = float(env_cfg.get("recovered_rmp_period_s", 0.001))
        checkpoint_substeps = int(env_cfg.get("frame_skip", self.substeps))
        if (checkpoint_substeps != self.substeps
                or not np.isclose(period, self.model.opt.timestep, rtol=0.0,
                                  atol=1.0e-12)
                or not np.isclose(
                    checkpoint_substeps * period, self.dt,
                    rtol=0.0, atol=1.0e-12)):
            raise RuntimeError(
                "GPU0 CPU mirror timing must match the checkpoint's "
                "5 x 1 ms recovered-RMP integration")

        kp_scale = self._checkpoint_vector(
            "recovered_rmp_jnt_kp_scale_per_joint", (1.0,) * 7)
        kd_scale = self._checkpoint_vector(
            "recovered_rmp_jnt_kd_scale_per_joint",
            (1.0, 1.0, 0.92, 1.0, 1.0, 1.0, 1.0))
        damping_scale = self._checkpoint_vector(
            "recovered_rmp_dof_damping_scale_per_joint",
            (0.6084, 0.9984, 0.9984, 1.92, 0.58, 1.17, 0.78))
        armature_scale = self._checkpoint_vector(
            "recovered_rmp_dof_armature_scale_per_joint",
            (0.4524, 1.65, 0.92, 1.22, 0.58, 0.5336, 0.58))
        self.model.dof_damping[self.arm_vadr] *= damping_scale
        self.model.dof_armature[self.arm_vadr] *= armature_scale

        velocity_deg_s = self._checkpoint_vector(
            "arm_vel_limit_deg_s",
            (210.0, 210.0, 240.0, 240.0, 300.0, 300.0, 300.0))
        acceleration_deg_s2 = self._checkpoint_vector(
            "arm_acc_limit_deg_s2",
            (1300.0, 1300.0, 1800.0, 3000.0, 3000.0, 3000.0, 3000.0))
        jerk_deg_s3 = self._checkpoint_vector(
            "recovered_rmp_jerk_limit_deg_s3",
            (35000.0, 35000.0, 45000.0, 60000.0,
             60000.0, 60000.0, 60000.0))
        params = RMPParams(
            jnt_kp=np.float64(400.0) * kp_scale,
            jnt_kd=np.float64(40.0) * kd_scale,
            acc_weight=float(env_cfg.get(
                "recovered_rmp_acceleration_weight", 0.001)),
            joint_vel_ffd=float(env_cfg.get(
                "recovered_rmp_joint_velocity_feedforward", 0.5)),
            joint_acc_ffd=float(env_cfg.get(
                "recovered_rmp_joint_acceleration_feedforward", 0.0)),
            filter_length=int(env_cfg.get(
                "recovered_rmp_output_filter_length", 35)),
            target_filter_length=int(env_cfg.get(
                "recovered_rmp_target_filter_length", 10)),
            estimator_joint_process_var=float(env_cfg.get(
                "recovered_rmp_estimator_joint_process_var", 50.0)),
            estimator_joint_measure_var=float(env_cfg.get(
                "recovered_rmp_estimator_joint_measure_var", 0.0001)),
        )
        limits = JointLimits(
            q_min=np.asarray(self.policy.arm_q_low, dtype=np.float64),
            q_max=np.asarray(self.policy.arm_q_high, dtype=np.float64),
            qd_max=np.deg2rad(velocity_deg_s),
            qdd_max=np.deg2rad(acceleration_deg_s2),
            qddd_max=np.deg2rad(jerk_deg_s3),
        )
        self.rmp_planner = JointRMPPlanner(
            limits,
            params,
            period,
            use_safety_bounds=bool(env_cfg.get(
                "recovered_rmp_use_safety_bounds", True)),
            use_jerk_limit=bool(env_cfg.get(
                "recovered_rmp_use_jerk_limit", False)),
        )
        self.rmp_target_update_delay_steps = int(env_cfg.get(
            "recovered_rmp_target_update_delay_steps", 3))
        if not 0 <= self.rmp_target_update_delay_steps < self.substeps:
            raise RuntimeError(
                "GPU0 RMP target-update delay lies outside one policy tick")
        delays = self._checkpoint_vector(
            "recovered_rmp_output_delay_steps_per_joint",
            (18.0, 19.0, 19.0, 18.0, 18.0, 18.0, 18.0),
            dtype=np.float64)
        if np.any(delays < 0.0):
            raise RuntimeError("GPU0 RMP output delays must be nonnegative")
        # Keep checkpoint integer delays bit-for-bit compatible while allowing
        # a mirror-only fractional-millisecond identification override.  The
        # recovered RMP advances at 1 kHz, so a fractional delay is represented
        # by linear interpolation between its two adjacent output samples.
        self.rmp_output_delay_steps = delays.astype(np.float64)
        actuator_kp_scale = self._checkpoint_vector(
            "recovered_rmp_actuator_kp_scale_per_joint", (1.0,) * 7)
        actuator_kv_scale = self._checkpoint_vector(
            "recovered_rmp_actuator_kv_scale_per_joint", (1.0,) * 7)
        if (np.any(actuator_kp_scale <= 0.0)
                or np.any(actuator_kv_scale <= 0.0)):
            raise RuntimeError(
                "GPU0 RMP mirror actuator Kp/Kv scales must be positive")
        self.model.actuator_gainprm[self.arm_aids, 0] *= actuator_kp_scale
        self.model.actuator_biasprm[self.arm_aids, 1] *= actuator_kp_scale
        self.model.actuator_biasprm[self.arm_aids, 2] *= actuator_kv_scale
        self.rmp_controller_kp = np.asarray(
            self.model.actuator_gainprm[self.arm_aids, 0], dtype=np.float64)
        self.rmp_controller_kv = -np.asarray(
            self.model.actuator_biasprm[self.arm_aids, 2], dtype=np.float64)
        if np.any(self.rmp_controller_kp <= 0.0):
            raise RuntimeError("GPU0 RMP position actuators require positive Kp")
        self.rmp_acceleration_feedforward_s2 = self._checkpoint_vector(
            "recovered_rmp_acceleration_feedforward_s2", (0.0,) * 7)

    def _reset_recovered_rmp(self, q0: np.ndarray) -> None:
        if self.rmp_planner is None or self.rmp_output_delay_steps is None:
            raise RuntimeError("GPU0 recovered RMP was not initialized")
        q = np.asarray(q0, dtype=np.float64).reshape(len(RIGHT_ARM_JOINTS))
        zeros = np.zeros_like(q)
        self.rmp_planner.reset(
            q, velocity=zeros, acceleration=zeros, target_position=q)
        buffer_length = max(
            1, int(np.ceil(np.max(self.rmp_output_delay_steps))))
        self.rmp_output_q_buffer = np.repeat(q[None, :], buffer_length, axis=0)
        self.rmp_output_qd_buffer = np.repeat(
            zeros[None, :], buffer_length, axis=0)
        self.rmp_output_qdd_buffer = np.repeat(
            zeros[None, :], buffer_length, axis=0)

    def _push_rmp_output_delay(
        self,
        buffer: np.ndarray,
        value: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.rmp_output_delay_steps is None:
            raise RuntimeError("GPU0 RMP output delays are unavailable")
        combined = np.concatenate(
            (buffer, np.asarray(value, dtype=np.float64)[None, :]), axis=0)
        delay = np.asarray(self.rmp_output_delay_steps, dtype=np.float64)
        delay_floor = np.floor(delay).astype(np.int64)
        delay_fraction = delay - delay_floor
        newest = combined.shape[0] - 1
        source_newer = newest - delay_floor
        source_older = np.maximum(0, source_newer - 1)
        joints = np.arange(len(RIGHT_ARM_JOINTS))
        delayed = (
            (1.0 - delay_fraction) * combined[source_newer, joints]
            + delay_fraction * combined[source_older, joints]
        )
        return combined[1:].copy(), delayed

    def _step_recovered_rmp(self, q_target: np.ndarray) -> np.ndarray:
        if (self.rmp_planner is None
                or self.rmp_output_q_buffer is None
                or self.rmp_output_qd_buffer is None
                or self.rmp_output_qdd_buffer is None
                or self.rmp_controller_kp is None
                or self.rmp_controller_kv is None
                or self.rmp_acceleration_feedforward_s2 is None):
            raise RuntimeError("GPU0 recovered RMP state is unavailable")
        delayed_q = np.asarray(q_target, dtype=np.float64)
        for servo_index in range(self.substeps):
            command = self.rmp_planner.step(
                q_target,
                target_updated=(
                    servo_index == self.rmp_target_update_delay_steps),
            )
            self.rmp_output_q_buffer, delayed_q = self._push_rmp_output_delay(
                self.rmp_output_q_buffer, command.position)
            self.rmp_output_qd_buffer, delayed_qd = self._push_rmp_output_delay(
                self.rmp_output_qd_buffer, command.velocity)
            self.rmp_output_qdd_buffer, delayed_qdd = self._push_rmp_output_delay(
                self.rmp_output_qdd_buffer, command.acceleration)
            low_level_ctrl_q = np.clip(
                delayed_q
                + self.rmp_controller_kv / self.rmp_controller_kp * delayed_qd
                + self.rmp_acceleration_feedforward_s2 * delayed_qdd,
                self.policy.arm_q_low,
                self.policy.arm_q_high,
            )
            self.data.ctrl[:] = self.default_ctrl
            self.data.ctrl[self.arm_aids] = low_level_ctrl_q
            mj.mj_step(self.model, self.data)
        # Match the MJX state contract: appliedCommand is the delayed Plan
        # position output; actuatorTarget separately exposes the qd-equivalent
        # XML position-actuator control value.
        return delayed_q.astype(np.float32)

    def reset(
        self,
        arm_q: np.ndarray | None = None,
        *,
        ball_pos: np.ndarray | None = None,
        ball_vel: np.ndarray | None = None,
        ball_valid: bool = False,
        ball_obs_age_s: float = 0.0,
    ) -> dict[str, Any]:
        self.data = mj.MjData(self.model)
        self.data.ctrl[:] = self.default_ctrl
        if arm_q is not None and not self._training_xml_profile:
            q = _finite_vec(arm_q, len(RIGHT_ARM_JOINTS))
            for i, (qadr, aid) in enumerate(zip(self.arm_qadr, self.arm_aids)):
                self.data.qpos[qadr] = float(q[i])
                self.data.ctrl[aid] = float(q[i])
        for _ in range(700):
            mj.mj_step(self.model, self.data)
        if arm_q is not None and self._training_xml_profile:
            # MJX reset first restores its canonical warm state, then the live
            # mirror's force-arm path overwrites q/dq/ctrl without an extra
            # settling interval.  Match that ordering exactly.
            q = _finite_vec(arm_q, len(RIGHT_ARM_JOINTS))
            self.data.qpos[self.arm_qadr] = q
            self.data.qvel[self.arm_vadr] = 0.0
            self.data.ctrl[self.arm_aids] = q
        mj.mj_forward(self.model, self.data)
        q0 = self.data.qpos[self.arm_qadr].astype(np.float32)
        self.policy.reset()
        self.policy.arm_cmd_q = q0.copy()
        self.policy.arm_cmd_qvel = (
            np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float32)
            if self._training_xml_profile
            else self.data.qvel[self.arm_vadr].astype(np.float32)
        )
        self.policy.last_cmd_q = q0.copy()
        self.policy.last_cmd_qvel = self.policy.arm_cmd_qvel.copy()
        self.policy.last_cmd_qacc = np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float32)
        self.policy.last_action = np.zeros(len(RIGHT_ARM_JOINTS), dtype=np.float32)
        if self._corrected_67d_profile:
            # reset() initially anchors the corrected observation path at the
            # checkpoint reset pose.  The mirror may instead be seeded from a
            # live arm pose, so re-anchor every state that is observable on the
            # first policy tick at the actual settled q0.
            command_buffer_len = max(
                1, int(getattr(self.policy, "_command_buffer_len", 1)))
            self.policy._cmd_history = [
                q0.copy() for _ in range(command_buffer_len)
            ]
            racket_pos_base, fk_ok = self.policy._compute_racket_pose_base(q0)
            if fk_ok:
                racket_anchor = np.asarray(
                    racket_pos_base, dtype=np.float32).reshape(3)
                self.policy._prev_racket_pos_base = racket_anchor.copy()
                self.policy._racket_anchor_pos_base = racket_anchor.copy()
        if self.warm_arm_q is None:
            self.warm_arm_q = q0.copy()
            self.warm_arm_qvel = self.policy.arm_cmd_qvel.copy()
        if self._ideal_pd_profile or self._real_rmp_profile:
            # simcode2's ideal-PD plant consumes the current nominal policy
            # command directly.  Keep the public command fields coherent while
            # leaving the legacy software delay/FOPDT state completely unused.
            self.arm_q_ref_active = q0.copy()
            self.arm_actuator_q_ref_active = q0.copy()
            self.arm_applied_q = q0.copy()
            self.actuator_model_key = None
            self.command_buffer = None
            self.actuator_command_buffer = None
            self.actuator_delay_steps = 0
            self.actuator_command_buffer_len = 1
            if self._real_rmp_profile:
                self._reset_recovered_rmp(q0)
        else:
            self._reset_actuator_model(q0)
        self.frame_seq = 0
        self.latest_frame = None
        self.latest_frame_info = {"seq": 0, "ageS": None}
        self.next_frame_wall = 0.0
        if self.compensator is not None:
            try:
                self.compensator.reset(q0.copy())
            except Exception:
                pass
        if ball_pos is not None:
            reset_ball_vel = (
                np.zeros(3, dtype=np.float32)
                if ball_vel is None else _finite_vec(ball_vel, 3)
            )
            self._set_ball_state(ball_pos, reset_ball_vel)
            mj.mj_forward(self.model, self.data)
        return self.snapshot(
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            ball_valid=bool(ball_valid),
            ball_obs_age_s=max(0.0, float(ball_obs_age_s)),
        )

    def _set_ball_state(self, ball_pos: np.ndarray, ball_vel: np.ndarray) -> None:
        if self.ball_qadr < 0 or self.ball_vadr < 0:
            return
        p = _finite_vec(ball_pos, 3)
        v = _finite_vec(ball_vel, 3)
        if self._training_xml_profile:
            # Probe/policy inputs are already in the base-yaw frame, whereas
            # the MuJoCo free joint is world-frame.  Invert simcode2's
            # _point_to_base/_vel_to_base before injecting the physical ball.
            base_q = self.data.qpos[self.base_qadr].astype(np.float64)
            base_dq = self.data.qvel[self.base_vadr].astype(np.float64)
            c = math.cos(float(base_q[2]))
            s = math.sin(float(base_q[2]))
            rel_x = c * float(p[0]) - s * float(p[1])
            rel_y = s * float(p[0]) + c * float(p[1])
            rel_vx = c * float(v[0]) - s * float(v[1])
            rel_vy = s * float(v[0]) + c * float(v[1])
            p = np.asarray([
                float(base_q[0]) + rel_x,
                float(base_q[1]) + rel_y,
                float(p[2]),
            ], dtype=np.float32)
            yaw_rate = float(base_dq[2])
            v = np.asarray([
                float(base_dq[0]) + rel_vx - yaw_rate * rel_y,
                float(base_dq[1]) + rel_vy + yaw_rate * rel_x,
                float(v[2]),
            ], dtype=np.float32)
        self.data.qpos[self.ball_qadr:self.ball_qadr + 3] = p
        self.data.qpos[self.ball_qadr + 3:self.ball_qadr + 7] = np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.data.qvel[self.ball_vadr:self.ball_vadr + 3] = v
        self.data.qvel[self.ball_vadr + 3:self.ball_vadr + 6] = 0.0

    def _actuator_model_key_from_values(
        self,
        *,
        delay_ms: float,
        tau_s: float,
        gain: float,
    ) -> tuple[Any, ...]:
        delay_conditioning = int(getattr(self.cfg, "policy_obs_dim", 0)) == 67
        return (
            round(float(max(0.0, delay_ms)), 6),
            round(float(max(0.0, tau_s)), 6),
            round(float(gain), 6),
            4,
            bool(delay_conditioning),
        )

    def _actuator_model_key(self, options: Any | None = None) -> tuple[Any, ...]:
        if (bool(getattr(self, "_pure_actuator_profile", False))
                and self.pure_actuator_wn is not None
                and self.pure_actuator_zeta is not None
                and self.pure_actuator_gain is not None
                and self.pure_actuator_delay_steps is not None):
            return (
                "checkpoint_second_order",
                tuple(float(v) for v in self.pure_actuator_wn),
                tuple(float(v) for v in self.pure_actuator_zeta),
                tuple(float(v) for v in self.pure_actuator_gain),
                tuple(int(v) for v in self.pure_actuator_delay_steps),
            )
        return self._actuator_model_key_from_values(
            delay_ms=float(
                getattr(options, "comp_delay_ms", self.cfg.comp_delay_ms)),
            tau_s=float(getattr(
                options, "comp_actuator_tau_s",
                self.cfg.comp_actuator_tau_s,
            )),
            gain=float(getattr(
                options, "comp_actuator_gain",
                self.cfg.comp_actuator_gain,
            )),
        )

    def _reset_actuator_model(
        self,
        q_seed: np.ndarray,
        options: Any | None = None,
    ) -> None:
        q0 = _finite_vec(q_seed, len(RIGHT_ARM_JOINTS))
        key = self._actuator_model_key(options)
        pure_second_order = (
            bool(getattr(self, "_pure_actuator_profile", False))
            and self.pure_actuator_delay_steps is not None)
        delay_ms = (
            float(np.max(self.pure_actuator_delay_steps)) * self.dt * 1e3
            if pure_second_order else float(key[0]))
        self.actuator_command_buffer_len = _command_buffer_length(
            delay_ms, self.dt, 4 if pure_second_order else int(key[3]))
        # The actor-facing active-command error uses the scalar tau_act delay
        # (45 ms for simcode6), while the physical second-order plant below
        # uses its distinct per-joint 35--55 ms delays.  Never source the
        # scalar history index from a legacy UI/profile-switch value.
        checkpoint_cfg = dict(getattr(self, "checkpoint_env_cfg", {}) or {})
        nominal_delay_ms = float(checkpoint_cfg.get(
            "delay_min_ms", getattr(self.cfg, "comp_delay_ms", 0.0)))
        self.actuator_delay_steps = int(np.clip(
            (int(round(nominal_delay_ms * 1e-3 / self.dt))
             if pure_second_order
             else _delay_steps_from_tau(delay_ms * 1e-3, self.dt)),
            0,
            self.actuator_command_buffer_len - 1,
        ))
        shape = (self.actuator_command_buffer_len, len(RIGHT_ARM_JOINTS))
        self.command_buffer = np.broadcast_to(q0, shape).copy()
        self.actuator_command_buffer = np.broadcast_to(q0, shape).copy()
        self.arm_q_ref_active = q0.copy()
        self.arm_actuator_q_ref_active = q0.copy()
        self.arm_applied_q = q0.copy()
        self.arm_applied_qvel = np.zeros_like(q0)
        self.actuator_model_key = key

    def _ensure_actuator_model(self, options: Any, seed_q: np.ndarray) -> None:
        key = self._actuator_model_key(options)
        if (self.actuator_model_key == key
                and self.command_buffer is not None
                and self.actuator_command_buffer is not None
                and self.arm_applied_q is not None):
            return
        self._reset_actuator_model(seed_q, options)

    def _run_actuator_model(
        self,
        q_ref_latest: np.ndarray,
        q_actuator_ref_latest: np.ndarray,
        options: Any,
        *,
        seed_q: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mirror simcode's command delay/filter before MuJoCo PD actuators."""
        q_ref = _finite_vec(q_ref_latest, len(RIGHT_ARM_JOINTS))
        q_actuator_ref = _finite_vec(q_actuator_ref_latest, len(RIGHT_ARM_JOINTS))
        self._ensure_actuator_model(options, seed_q)
        assert self.command_buffer is not None
        assert self.actuator_command_buffer is not None
        assert self.arm_applied_q is not None
        if self.arm_applied_qvel is None:
            self.arm_applied_qvel = np.zeros_like(self.arm_applied_q)

        if (bool(getattr(self, "_pure_actuator_profile", False))
                or bool(self._actuator_model_key(options)[4])):
            delay_steps = self.actuator_delay_steps
        else:
            delay_steps = 0
        self.command_buffer, q_ref_active = _push_command_buffer(
            self.command_buffer, q_ref, delay_steps)
        if (bool(getattr(self, "_pure_actuator_profile", False))
                and self.pure_actuator_wn is not None
                and self.pure_actuator_zeta is not None
                and self.pure_actuator_gain is not None
                and self.pure_actuator_delay_steps is not None):
            self.actuator_command_buffer = np.concatenate([
                self.actuator_command_buffer[1:], q_actuator_ref[None, :]
            ], axis=0)
            joint_steps = np.clip(
                self.pure_actuator_delay_steps,
                0,
                self.actuator_command_buffer.shape[0] - 1,
            )
            joint_indices = (
                self.actuator_command_buffer.shape[0] - 1 - joint_steps)
            q_actuator_active = self.actuator_command_buffer[
                joint_indices, np.arange(len(RIGHT_ARM_JOINTS))].copy()
        else:
            self.actuator_command_buffer, q_actuator_active = (
                _push_command_buffer(
                    self.actuator_command_buffer, q_actuator_ref, delay_steps))

        warm = (
            self.warm_arm_q
            if self.warm_arm_q is not None else np.zeros_like(q_actuator_active)
        )
        if (bool(getattr(self, "_pure_actuator_profile", False))
                and self.pure_actuator_wn is not None
                and self.pure_actuator_zeta is not None
                and self.pure_actuator_gain is not None):
            wn = self.pure_actuator_wn
            zeta = self.pure_actuator_zeta
            root = np.sqrt(np.maximum(1.0 - zeta * zeta, 1e-8))
            wd = wn * root
            decay = np.exp(-zeta * wn * self.dt)
            phase = wd * self.dt
            sin_phase = np.sin(phase)
            cos_phase = np.cos(phase)
            a11 = decay * (cos_phase + zeta / root * sin_phase)
            a12 = decay * sin_phase / np.maximum(wd, 1e-8)
            a21 = -decay * wn * wn * sin_phase / np.maximum(wd, 1e-8)
            a22 = decay * (cos_phase - zeta / root * sin_phase)
            gain_target = warm + self.pure_actuator_gain * (
                q_actuator_active - warm)
            current_q = self.arm_applied_q
            current_qvel = self.arm_applied_qvel
            applied_q = (
                a11 * current_q + a12 * current_qvel
                + (1.0 - a11) * gain_target
            ).astype(np.float32)
            self.arm_applied_qvel = (
                a21 * current_q + a22 * current_qvel
                - a21 * gain_target
            ).astype(np.float32)
        else:
            tau_s = max(0.0, float(getattr(
                options, "comp_actuator_tau_s", self.cfg.comp_actuator_tau_s)))
            gain = float(getattr(
                options, "comp_actuator_gain", self.cfg.comp_actuator_gain))
            gain_target = warm + gain * (q_actuator_active - warm)
            if tau_s <= 1e-6:
                applied_q = gain_target.astype(np.float32)
            else:
                alpha = self.dt / (tau_s + self.dt)
                applied_q = (
                    self.arm_applied_q
                    + np.float32(alpha) * (gain_target - self.arm_applied_q)
                ).astype(np.float32)
        applied_q = np.clip(
            applied_q, self.policy.arm_q_low, self.policy.arm_q_high,
        ).astype(np.float32)
        if bool(getattr(self, "_servo_planner_profile", False)):
            from pingpong_controller.safety_limiter import (
                project_target_tracking_command_step,
            )
            planned = project_target_tracking_command_step(
                applied_q,
                self.arm_applied_q,
                self.arm_applied_qvel,
                self.policy.arm_q_low,
                self.policy.arm_q_high,
                self.policy.arm_vel_limit_rad_s,
                self.policy.arm_acc_limit_rad_s2 * np.float32(0.8),
                self.dt,
            )
            applied_q = np.asarray(planned[0], dtype=np.float32)
            self.arm_applied_qvel = np.asarray(planned[1], dtype=np.float32)
        elif not bool(getattr(self, "_pure_actuator_profile", False)):
            self.arm_applied_qvel = (
                (applied_q - self.arm_applied_q) / max(self.dt, 1e-6)
            ).astype(np.float32)

        self.arm_q_ref_active = q_ref_active.astype(np.float32)
        self.arm_actuator_q_ref_active = q_actuator_active.astype(np.float32)
        self.arm_applied_q = applied_q
        return self.arm_q_ref_active, self.arm_actuator_q_ref_active, applied_q

    def _comp_key(self, options: Any) -> tuple[Any, ...]:
        return (
            str(getattr(options, "compensation_mode", "inverse_mpc")).lower(),
            round(float(getattr(options, "comp_delay_ms", 72.0)), 6),
            round(float(getattr(options, "comp_actuator_tau_s", 0.074)), 6),
            round(float(getattr(options, "comp_actuator_gain", 1.0)), 6),
            round(float(getattr(options, "comp_mpc_beta", 1.2)), 6),
            round(float(getattr(options, "comp_mpc_delay_scale", 1.05)), 6),
            round(float(getattr(options, "comp_mpc_tau_scale", 0.75)), 6),
            int(getattr(options, "comp_mpc_horizon_steps", 6)),
            round(float(getattr(options, "comp_mpc_tracking_weight", 1.0)), 6),
            round(float(getattr(options, "comp_mpc_nominal_weight", 0.25)), 6),
            round(float(getattr(options, "comp_mpc_delta_weight", 0.05)), 6),
            round(float(getattr(options, "comp_mpc_max_delta_rad", 0.0)), 6),
        )

    def _ensure_compensator(self, options: Any) -> None:
        if not bool(getattr(options, "enable_compensation", False)):
            return
        if not _COMPENSATION_AVAILABLE or CommandCompensator is None:
            self.compensation_error = (
                "CommandCompensator unavailable"
                if _COMPENSATION_IMPORT_ERROR is None
                else f"CommandCompensator unavailable: {_COMPENSATION_IMPORT_ERROR}"
            )
            return
        key = self._comp_key(options)
        if self.compensator is not None and self.compensator_key == key:
            return
        try:
            self.compensator = CommandCompensator(
                n_joints=len(RIGHT_ARM_JOINTS),
                dt=self.dt,
                mode=str(getattr(options, "compensation_mode", "inverse_mpc")),
                delay_ms=float(getattr(options, "comp_delay_ms", 72.0)),
                actuator_tau_s=float(getattr(options, "comp_actuator_tau_s", 0.074)),
                actuator_gain=float(getattr(options, "comp_actuator_gain", 1.0)),
                command_buffer_extra_steps=4,
                delay_conditioning=True,
                mpc_beta=float(getattr(options, "comp_mpc_beta", 1.2)),
                mpc_delay_scale=float(getattr(options, "comp_mpc_delay_scale", 1.05)),
                mpc_tau_scale=float(getattr(options, "comp_mpc_tau_scale", 0.75)),
                mpc_horizon_steps=int(getattr(options, "comp_mpc_horizon_steps", 6)),
                mpc_tracking_weight=float(getattr(options, "comp_mpc_tracking_weight", 1.0)),
                mpc_nominal_weight=float(getattr(options, "comp_mpc_nominal_weight", 0.25)),
                mpc_delta_weight=float(getattr(options, "comp_mpc_delta_weight", 0.05)),
                mpc_max_delta_rad=float(getattr(options, "comp_mpc_max_delta_rad", 0.0)),
                q_low=self.policy.arm_q_low,
                q_high=self.policy.arm_q_high,
                warm_q=self.warm_arm_q,
            )
            self.compensator.reset(self.data.qpos[self.arm_qadr].astype(np.float32))
            self.compensator_key = key
            self.compensation_error = None
        except Exception as exc:  # noqa: BLE001 - fall back to nominal.
            self.compensator = None
            self.compensator_key = None
            self.compensation_error = str(exc)

    def _apply_compensation(
        self,
        q_cmd: np.ndarray,
        options: Any,
        *,
        actual_q: np.ndarray | None = None,
    ) -> np.ndarray:
        if bool(getattr(self, "_pure_actuator_profile", False)):
            # The checkpoint plant consumes the actor's latest q-only target.
            # UI/API compensation flags must never insert a duplicate inverse
            # model into simcode5/6.
            return q_cmd.astype(np.float32)
        if not bool(getattr(options, "enable_compensation", False)):
            return q_cmd.astype(np.float32)
        self._ensure_compensator(options)
        if self.compensator is None:
            return q_cmd.astype(np.float32)
        try:
            cmd = self.policy.latest_command_state()
            qvel = np.asarray(cmd["velocity"], dtype=np.float32).reshape(-1)
            qacc = np.asarray(cmd["acceleration"], dtype=np.float32).reshape(-1)
            # simcode3 inverse-MPC was trained against the previous-tick
            # delayed/filter-modelled actuator target.  Simcode4 deliberately
            # changed that contract to actual joint feedback.  Keep the CPU
            # MuJoCo mirror consistent with the selected frozen plant instead
            # of silently applying simcode3 feedback semantics to simcode4.
            if self._servo_planner_profile and actual_q is not None:
                applied_q = _finite_vec(actual_q, len(RIGHT_ARM_JOINTS))
            else:
                applied_q = np.asarray(
                    self.arm_applied_q
                    if self.arm_applied_q is not None
                    else self.data.qpos[self.arm_qadr],
                    dtype=np.float32,
                )
            out = self.compensator.step(
                q_cmd=q_cmd.astype(np.float32),
                qdot_cmd=qvel,
                qdd_cmd=qacc,
                applied_q=applied_q,
            )
            self.compensation_error = None
            return np.asarray(out, dtype=np.float32).reshape(len(RIGHT_ARM_JOINTS))
        except Exception as exc:  # noqa: BLE001 - fall back to nominal.
            self.compensation_error = str(exc)
            return q_cmd.astype(np.float32)

    def step(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        *,
        ball_valid: bool,
        ball_obs_age_s: float,
        options: Any,
        num_steps: int = 1,
        confirmed_hit_count: int = 0,
    ) -> dict[str, Any]:
        q_nominal = np.asarray(self.policy.last_cmd_q, dtype=np.float32)
        q_actuator = self.data.ctrl[self.arm_aids].astype(np.float32)
        q_active = (
            self.arm_q_ref_active.copy()
            if self.arm_q_ref_active is not None else q_nominal.copy()
        )
        q_actuator_active = (
            self.arm_actuator_q_ref_active.copy()
            if self.arm_actuator_q_ref_active is not None else q_actuator.copy()
        )
        q_applied = (
            self.arm_applied_q.copy()
            if self.arm_applied_q is not None else q_actuator.copy()
        )
        steps = max(1, int(num_steps))
        for _ in range(steps):
            self._set_ball_state(ball_pos, ball_vel)
            mj.mj_forward(self.model, self.data)
            arm_q = self.data.qpos[self.arm_qadr].astype(np.float32)
            arm_dq = self.data.qvel[self.arm_vadr].astype(np.float32)
            self.policy.use_valid_ball_obs_age = bool(
                getattr(options, "rl_use_valid_ball_obs_age", True)
                and self.policy_profile not in {
                    OBSERVATION_PROFILE_SIMCODE4_SERVO67,
                    OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67,
                    OBSERVATION_PROFILE_SIMCODE6_QVEL67,
                })
            q_nominal = self.policy.predict(
                ball_pos,
                ball_vel,
                ball_valid=bool(ball_valid),
                ball_obs_age_s=float(max(0.0, ball_obs_age_s)),
                dt=self.dt,
                arm_q=arm_q,
                arm_dq=arm_dq,
                observation_scales=getattr(
                    options, "policy_observation_scales", ()),
                confirmed_hit_count=int(confirmed_hit_count),
            )
            if self._real_rmp_profile:
                # The actor performs the one and only qvel->q_ref integration.
                # The recovered planner then runs once per 1 ms MuJoCo step,
                # including the checkpoint's target and output delays.
                q_actuator = np.asarray(q_nominal, dtype=np.float32).copy()
                q_active = q_actuator.copy()
                q_applied = self._step_recovered_rmp(q_nominal)
                if self.rmp_output_q_buffer is None:
                    raise RuntimeError("GPU0 RMP output state is unavailable")
                q_actuator_active = self.rmp_output_q_buffer[-1].astype(
                    np.float32)
                self.arm_q_ref_active = q_active.copy()
                self.arm_actuator_q_ref_active = q_actuator_active.copy()
                self.arm_applied_q = q_applied.copy()
            elif self._ideal_pd_profile:
                # New simcode2 checkpoint contract: policy -> current nominal
                # target -> ideal XML position-actuator PD.  The 72 ms command
                # history is observation-only and must not delay plant input.
                q_actuator = np.asarray(q_nominal, dtype=np.float32).copy()
                q_active = q_actuator.copy()
                q_actuator_active = q_actuator.copy()
                q_applied = q_actuator.copy()
                self.arm_q_ref_active = q_active.copy()
                self.arm_actuator_q_ref_active = q_actuator_active.copy()
                self.arm_applied_q = q_applied.copy()
            else:
                q_actuator = self._apply_compensation(
                    q_nominal,
                    options,
                    actual_q=arm_q,
                )
                q_active, q_actuator_active, q_applied = self._run_actuator_model(
                    q_nominal,
                    q_actuator,
                    options,
                    seed_q=arm_q,
                )
            if not self._real_rmp_profile:
                self.data.ctrl[:] = self.default_ctrl
                for i, aid in enumerate(self.arm_aids):
                    self.data.ctrl[aid] = float(q_applied[i])
                for _ in range(self.substeps):
                    mj.mj_step(self.model, self.data)
        self._set_ball_state(ball_pos, ball_vel)
        mj.mj_forward(self.model, self.data)
        snap = self.snapshot(
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            ball_valid=ball_valid,
            ball_obs_age_s=ball_obs_age_s,
            q_nominal=q_nominal,
            q_actuator=q_actuator,
            q_active=q_active,
            q_actuator_active=q_actuator_active,
            q_applied=q_applied,
        )
        self._maybe_update_frame(snap)
        return snap

    def _racket_pose(self) -> tuple[np.ndarray, np.ndarray]:
        pos = np.asarray(
            self.data.site_xpos[self.racket_site_id], dtype=np.float32).reshape(3)
        mat = np.asarray(
            self.data.site_xmat[self.racket_site_id], dtype=np.float32).reshape(3, 3)
        quat_wxyz = np.zeros(4, dtype=np.float64)
        mj.mju_mat2Quat(quat_wxyz, mat.reshape(-1).astype(np.float64))
        return pos, _quat_wxyz_to_xyzw(quat_wxyz.astype(np.float32))

    def snapshot(
        self,
        *,
        ball_pos: np.ndarray | None = None,
        ball_vel: np.ndarray | None = None,
        ball_valid: bool = False,
        ball_obs_age_s: float = 0.0,
        q_nominal: np.ndarray | None = None,
        q_actuator: np.ndarray | None = None,
        q_active: np.ndarray | None = None,
        q_actuator_active: np.ndarray | None = None,
        q_applied: np.ndarray | None = None,
    ) -> dict[str, Any]:
        obs_action = self.policy.latest_obs_and_action()
        cmd = self.policy.latest_command_state()
        racket_pos, racket_quat = self._racket_pose()
        q = self.data.qpos[self.arm_qadr].astype(np.float32)
        dq = self.data.qvel[self.arm_vadr].astype(np.float32)
        ddq = self.data.qacc[self.arm_vadr].astype(np.float32)
        ctrl = self.data.ctrl[self.arm_aids].astype(np.float32)
        ball_p = _finite_vec(ball_pos, 3) if ball_pos is not None else (
            np.asarray(self.data.xpos[self.ball_body_id], dtype=np.float32)
            if self.ball_body_id >= 0 else np.zeros(3, dtype=np.float32))
        ball_v = _finite_vec(ball_vel, 3) if ball_vel is not None else np.zeros(3, dtype=np.float32)
        plant_contract = {
            "settingsSource": (
                "checkpoint_env_cfg"
                if (self._pure_actuator_profile or self._real_rmp_profile)
                else "configured_xml"),
            "rightArmPdProfile": str(self.right_arm_pd_profile),
            "controlDtS": round(float(self.dt), 9),
            "mujocoTimestepS": round(float(self.model.opt.timestep), 9),
            "frameSkip": int(self.substeps),
            "jointVelocitySource": "mujoco_data.qvel[jnt_dofadr]",
            "checkpointPlantSettingsAligned": bool(
                self._pure_actuator_profile or self._real_rmp_profile),
            "backendNumericallyIdenticalToMjx": False,
        }
        if self._pure_actuator_profile:
            plant_contract.update({
                "actuatorCmdModel": str(self.checkpoint_env_cfg.get(
                    "actuator_cmd_model", "")),
                "actuatorCompensationMode": "none",
                "delayMsPerJoint": list(self.checkpoint_env_cfg.get(
                    "actuator_cmd_delay_ms_per_joint", ())),
                "naturalFrequencyRadS": (
                    [] if self.pure_actuator_wn is None
                    else self.pure_actuator_wn.tolist()),
                "dampingRatio": (
                    [] if self.pure_actuator_zeta is None
                    else self.pure_actuator_zeta.tolist()),
                "gainPerJoint": (
                    [] if self.pure_actuator_gain is None
                    else self.pure_actuator_gain.tolist()),
            })
        if self._real_rmp_profile:
            plant_contract.update({
                "controlStack": "actor_qvel_integrator_then_recovered_rmp",
                "actuatorCompensationMode": "none",
                "rmpPeriodS": round(float(self.model.opt.timestep), 9),
                "rmpTargetUpdateDelaySteps": int(
                    self.rmp_target_update_delay_steps),
                "rmpOutputDelayStepsPerJoint": (
                    [] if self.rmp_output_delay_steps is None
                    else self.rmp_output_delay_steps.tolist()),
                "controllerKp": (
                    [] if self.rmp_controller_kp is None
                    else self.rmp_controller_kp.tolist()),
                "controllerKv": (
                    [] if self.rmp_controller_kv is None
                    else self.rmp_controller_kv.tolist()),
            })
        return {
            "enabled": True,
            "available": True,
            "backend": (
                "mujoco_recovered_rmp_mirror"
                if self._real_rmp_profile else (
                    "mujoco_ideal_pd_mirror"
                    if self._ideal_pd_profile else "mujoco_actuator_mirror")),
            "policyProfile": self.policy_profile,
            "policyModel": str(self.cfg.policy_model),
            "timeWallS": round(float(time.time()), 6),
            "valid": bool(ball_valid),
            "age": round(float(max(0.0, ball_obs_age_s)), 6),
            "ball": _rounded(ball_p),
            "velocity": _rounded(ball_v),
            "jointPosition": _rounded(q),
            "jointVelocity": _rounded(dq),
            "jointAcceleration": _rounded(ddq),
            "jointAccelerationSource": "mujoco_qacc",
            "plantContract": plant_contract,
            "actuatorTarget": _rounded(ctrl),
            "policyCommand": _rounded(q_nominal if q_nominal is not None else cmd["position"]),
            "compensatedCommand": _rounded(q_actuator if q_actuator is not None else ctrl),
            "activeCommand": _rounded(
                q_active if q_active is not None
                else (self.arm_q_ref_active if self.arm_q_ref_active is not None
                      else cmd["position"])),
            "activeCompensatedCommand": _rounded(
                q_actuator_active if q_actuator_active is not None
                else (self.arm_actuator_q_ref_active
                      if self.arm_actuator_q_ref_active is not None else ctrl)),
            "appliedCommand": _rounded(
                q_applied if q_applied is not None
                else (self.arm_applied_q if self.arm_applied_q is not None
                      else ctrl)),
            "racket": _rounded(racket_pos),
            "racketQuatXyzw": _rounded(racket_quat),
            "policyObs": _rounded(obs_action["obs"]),
            "policyObsUnscaled": _rounded(obs_action.get(
                "obs_unscaled", obs_action["obs"])),
            "policyObservationScaleIndices": list(obs_action.get(
                "observation_scale_indices", [])),
            "policyObservationScales": _rounded(obs_action.get(
                "observation_scales", np.ones_like(obs_action["obs"]))),
            "policyBallObservation": np.asarray(
                obs_action["obs"], dtype=np.float32,
            ).reshape(-1)[20:26].tolist(),
            "policyAction": _rounded(obs_action["action"]),
            "policyExecutedAction": _rounded(obs_action.get(
                "executed_action", obs_action["action"])),
            "command": {
                "position": _rounded(cmd["position"]),
                "velocity": _rounded(cmd["velocity"]),
                "acceleration": _rounded(cmd["acceleration"]),
            },
            "compensation": (
                {
                    "error": None,
                    "available": True,
                    "mode": "none",
                }
                if (self._ideal_pd_profile or self._pure_actuator_profile
                    or self._real_rmp_profile) else {
                    "error": self.compensation_error,
                    "available": bool(_COMPENSATION_AVAILABLE),
                }
            ),
            "servoPlanner": {
                "enabled": bool(self._servo_planner_profile),
                "feedbackSource": (
                    "actual_joint_state"
                    if self._servo_planner_profile
                    else "modeled_applied_target"),
                "velocityScale": 1.0,
                "accelerationScale": (
                    0.8 if self._servo_planner_profile else 1.0),
            },
            "frame": dict(self.latest_frame_info),
            "renderError": self.render_error,
        }

    def latest_visualization(self) -> tuple[bytes | None, dict[str, Any]]:
        return self.latest_frame, dict(self.latest_frame_info)

    def _maybe_update_frame(self, snap: dict[str, Any]) -> None:
        if not bool(self.cfg.enable_video_frames):
            return
        now = time.time()
        period = 1.0 / max(float(self.cfg.frame_rate_hz), 1e-6)
        if now < self.next_frame_wall:
            return
        frame = self._render_frame(snap)
        if frame is None:
            return
        self.frame_seq += 1
        self.latest_frame = frame
        self.latest_frame_info = {
            "seq": int(self.frame_seq),
            "wallTimeS": round(float(now), 6),
            "width": int(self.cfg.render_width),
            "height": int(self.cfg.render_height),
            "backend": "mujoco" if self.renderer is not None else "fallback_2d",
        }
        self.next_frame_wall = now + period

    def _render_frame(self, snap: dict[str, Any]) -> bytes | None:
        if (bool(self.cfg.use_mujoco_renderer)
                and self.render_error is None):
            try:
                if self.renderer is None:
                    self.renderer = mj.Renderer(
                        self.model,
                        width=int(self.cfg.render_width),
                        height=int(self.cfg.render_height),
                    )
                self.renderer.update_scene(self.data)
                rgb = self.renderer.render()
                if rgb is not None and _CV2_AVAILABLE:
                    ok, enc = cv2.imencode(
                        ".jpg",
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 78],
                    )
                    if ok:
                        return enc.tobytes()
            except Exception as exc:  # noqa: BLE001 - use fallback renderer.
                self.render_error = str(exc)
        return self._render_fallback_frame(snap)

    def _render_fallback_frame(self, snap: dict[str, Any]) -> bytes | None:
        if not _CV2_AVAILABLE:
            self.render_error = (
                "cv2 unavailable" if _CV2_IMPORT_ERROR is None
                else f"cv2 unavailable: {_CV2_IMPORT_ERROR}")
            return None
        return _encode_fallback_frame(self.cfg, snap, "sim mirror x/z")

    def close(self) -> None:
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None


class RealBallMjxJaxSimMirror:
    """A single-env MJX/JAX mirror driven by real policy-facing ball state."""

    def __init__(
        self,
        cfg: SimMirrorConfig,
        *,
        logger: Any | None = None,
    ) -> None:
        cfg = sim_mirror_config_for_policy_profile(cfg, cfg.policy_profile)
        self.cfg = cfg
        self.policy_profile = normalize_policy_profile(cfg.policy_profile)
        self._ideal_pd_profile = (
            self.policy_profile == OBSERVATION_PROFILE_SIMCODE2_IDEALPD67)
        self._servo_planner_profile = (
            self.policy_profile == OBSERVATION_PROFILE_SIMCODE4_SERVO67)
        self._pure_actuator_profile = (
            self.policy_profile in {
                OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67,
                OBSERVATION_PROFILE_SIMCODE6_QVEL67,
            })
        self._real_rmp_profile = (
            self.policy_profile
            == OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85)
        self._corrected_67d_profile = uses_corrected_67d_observation(
            self.policy_profile)
        self._logger = logger or _LoggerShim()
        if not _MUJOCO_AVAILABLE:
            raise RuntimeError(f"mujoco unavailable: {_MUJOCO_IMPORT_ERROR}")

        try:
            import jax  # noqa: PLC0415
            import jax.numpy as jnp  # noqa: PLC0415
            from mujoco import mjx as _mjx  # noqa: F401, PLC0415
        except Exception as exc:  # noqa: BLE001 - surfaced in diagnostics.
            raise RuntimeError(
                "MJX/JAX backend requested but jax or mujoco.mjx is unavailable. "
                "Install compatible jax/jaxlib and a MuJoCo build exposing mujoco.mjx "
                "inside the pingpong conda env."
            ) from exc

        self.jax = jax
        self.jnp = jnp
        MjxJuggleConfig, MjxJuggleEnv = _load_simcode_mjx_env(
            self.policy_profile)
        env_kwargs = self._mjx_env_kwargs(MjxJuggleConfig)
        self.env_cfg = MjxJuggleConfig(**env_kwargs)
        self.env = MjxJuggleEnv(
            Path(cfg.sim_xml_path).expanduser().resolve(),
            n_envs=1,
            cfg=self.env_cfg,
        )
        if int(getattr(self.env, "obs_dim", 0)) != int(cfg.policy_obs_dim):
            raise RuntimeError(
                f"MJX env obs_dim={self.env.obs_dim} does not match "
                f"policy_obs_dim={cfg.policy_obs_dim}. Check checkpoint env_cfg "
                "and POLICY_OBS_DIM."
            )
        self.obs_dim = int(cfg.policy_obs_dim)

        policy_cls = {
            OBSERVATION_PROFILE_LEGACY_BUG_COMPAT: MjxRLPolicyController,
            OBSERVATION_PROFILE_SIMCODE2_IDEALPD67: (
                Simcode2IdealPd67PolicyController),
            OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67: (
                Simcode3Actuator67PolicyController),
            OBSERVATION_PROFILE_SIMCODE4_SERVO67: (
                Simcode4Servo67PolicyController),
            OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67: (
                Simcode5PureActuator67PolicyController),
            OBSERVATION_PROFILE_SIMCODE6_QVEL67: (
                Simcode6Qvel67PolicyController),
            OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85: (
                Gpu0QvelRealRmpV85PolicyController),
        }[self.policy_profile]
        self.policy = policy_cls(
            str(Path(cfg.model_path).expanduser().resolve()),
            dt=float(self.env.dt),
            robot_xml_path=str(Path(cfg.policy_xml_path).expanduser().resolve()),
            enable_delay_conditioning=(int(cfg.policy_obs_dim) == 67),
            use_valid_ball_obs_age=True,
            logger=self._logger,
        )
        self.arm_qadr_np = np.asarray(
            self.jax.device_get(self.env.arm_qadr), dtype=np.int32).reshape(-1)
        self.arm_vadr_np = np.asarray(
            self.jax.device_get(self.env.arm_vadr), dtype=np.int32).reshape(-1)
        self.arm_aids_np = np.asarray(self.env.arm_aids, dtype=np.int32).reshape(-1)
        self.dt = float(self.env.dt)
        self._reset_jit = self.jax.jit(self.env.reset)
        self._step_jit = self.jax.jit(self.env.step)
        self._force_ball_jit = self.jax.jit(self._force_ball_state)
        self._force_arm_jit = self.jax.jit(self._force_arm_state)
        self.policy_weights_jax = {
            key: self.jnp.asarray(value, dtype=self.jnp.float32)
            for key, value in dict(self.policy._pi_weights or {}).items()
        }
        self._step_sequence_jit = self.jax.jit(self._step_sequence_jax)
        self.renderer = None
        self.render_data = None
        self.render_error: str | None = None
        self.frame_seq = 0
        self.latest_frame: bytes | None = None
        self.latest_frame_info: dict[str, Any] = {"seq": 0, "ageS": None}
        self.next_frame_wall = 0.0
        self.last_obs = np.zeros((int(cfg.policy_obs_dim),), dtype=np.float32)
        self.last_action = np.zeros((len(RIGHT_ARM_JOINTS),), dtype=np.float32)
        self.last_executed_action = np.zeros(
            (len(RIGHT_ARM_JOINTS),), dtype=np.float32)
        self.last_command_acceleration = np.zeros(
            (len(RIGHT_ARM_JOINTS),), dtype=np.float32)
        self.last_done = False
        self.last_reward = 0.0
        self.last_metrics: dict[str, Any] = {}
        self.state = None
        self.obs = None
        self.reset()
        self._logger.info(
            "MJX/JAX simulation mirror ready | "
            f"policy_profile={self.policy_profile}, "
            f"xml={Path(cfg.sim_xml_path).name}, dt={self.dt:.4f}s, "
            f"obs_dim={self.env.obs_dim}, renderer={bool(cfg.use_mujoco_renderer)}")

    def _mjx_env_kwargs(self, config_cls: Any) -> dict[str, Any]:
        field_names = {f.name for f in fields(config_cls)}
        meta = inspect_mjx_checkpoint_metadata(
            str(Path(self.cfg.model_path).expanduser().resolve()))
        env_cfg = dict(meta.get("env_cfg") or {})
        kwargs = {k: v for k, v in env_cfg.items() if k in field_names}

        fixed = {
            "domain_randomization": bool(self.cfg.mjx_domain_randomization),
            "enable_delay_conditioning": (int(self.cfg.policy_obs_dim) == 67),
            "ball_obs_rate_hz": 1.0 / max(float(self.cfg.control_dt), 1e-9),
            "ball_obs_fractional_rate": False,
            "ball_obs_dropout_prob": 0.0,
            "ball_obs_dropout_burst_prob": 0.0,
            "ball_obs_dropout_max_steps": 1,
            "ball_obs_dropout_burst_max_steps": 1,
            "ball_obs_age_tracks_stale": True,
            "ball_obs_dropout_on_refresh_only": False,
            "ball_obs_require_camera_visible": False,
            "dr_randomize_latency": False,
            "dr_randomize_actuator_cmd_filter": False,
        }
        if bool(self.cfg.mjx_zero_ball_obs_noise):
            fixed.update({
                "ball_obs_pos_noise_std": 0.0,
                "ball_obs_vel_noise_std": 0.0,
                "ball_obs_nominal_pos_bias_base": (0.0, 0.0, 0.0),
                "ball_obs_nominal_vel_bias_base": (0.0, 0.0, 0.0),
                "dr_randomize_ball_obs_frame": False,
            })
        if self._ideal_pd_profile:
            # Preserve the simcode2 checkpoint's ideal-PD plant contract.  In
            # particular, delay state exists only to form the final 17 policy
            # observations; it must not delay/filter the command sent to PD.
            fixed.update({
                "delay_min_ms": float(env_cfg.get("delay_min_ms", 72.0)),
                "delay_max_ms": float(env_cfg.get("delay_max_ms", 72.0)),
                "delay_jitter_ms": 0.0,
                "delay_sampling_mode": "uniform",
                "command_buffer_extra_steps": int(
                    env_cfg.get("command_buffer_extra_steps", 4)),
                "actuator_delay_observation_only": True,
                "actuator_cmd_filter": False,
                "actuator_cmd_tau": max(
                    0.0, float(env_cfg.get("actuator_cmd_tau", 0.074))),
                "actuator_cmd_gain": float(
                    env_cfg.get("actuator_cmd_gain", 1.0)),
                "actuator_compensation_mode": "none",
                "action_filter_tau_ms": 0.0,
                "action_jerk_limit": 0.0,
                "enable_anti_windup": False,
                "arm_post_compensation_limiter": False,
                "arm_servo_target_limiter": False,
                "arm_servo_target_tracking_planner": False,
                "arm_actual_state_limiter": False,
                "high_latency_obs": False,
                "dr_randomize_actuator": False,
                "dr_randomize_actuator_cmd_filter": False,
            })
        elif bool(getattr(self, "_pure_actuator_profile", False)):
            # Simcode5/6 are checkpoint-authoritative pure-plant profiles. The
            # real robot supplies these joint characteristics, but the mirror
            # must restore the per-joint delay + analytic second-order model
            # and sport PD profile from env_cfg.  Do not replace them with the
            # legacy UI's scalar FOPDT/compensation knobs.
            fixed.update({
                "delay_min_ms": float(env_cfg.get("delay_min_ms", 45.0)),
                "delay_max_ms": float(env_cfg.get("delay_max_ms", 45.0)),
                "delay_jitter_ms": 0.0,
                "delay_sampling_mode": "uniform",
                "command_buffer_extra_steps": int(
                    env_cfg.get("command_buffer_extra_steps", 4)),
                "actuator_delay_observation_only": False,
                "actuator_cmd_filter": True,
                "actuator_cmd_model": "second_order",
                "actuator_compensation_mode": "none",
                "arm_post_compensation_limiter": False,
                "arm_servo_target_limiter": False,
                "arm_servo_target_tracking_planner": False,
                "arm_servo_planner_before_actuator_model": False,
                "arm_actual_state_limiter": False,
                "ball_obs_age_tracks_stale": False,
                "dr_randomize_actuator": False,
                "dr_randomize_actuator_cmd_filter": False,
                "dr_randomize_second_order_actuator": False,
            })
        elif bool(getattr(self, "_real_rmp_profile", False)):
            # Restore the checkpoint's recovered-RMP + XML-PD simulator plant.
            # The external probe already supplies the exact held 90 Hz ball
            # state/age, so only the environment's internal camera sampler is
            # bypassed. No recovered-RMP component crosses into the real path.
            fixed.update({
                "action_filter_tau_ms": 0.0,
                "action_jerk_limit": 0.0,
                "enable_anti_windup": False,
                "actuator_cmd_filter": False,
                "actuator_compensation_mode": "none",
                "arm_servo_target_tracking_planner": False,
                "ball_obs_age_tracks_stale": True,
                "dr_randomize_actuator": False,
                "dr_randomize_actuator_cmd_filter": False,
                "dr_randomize_second_order_actuator": False,
            })
        else:
            if int(self.cfg.policy_obs_dim) == 67:
                fixed.update({
                    "delay_min_ms": float(
                        env_cfg.get("delay_min_ms", self.cfg.comp_delay_ms)
                        if self._servo_planner_profile
                        else self.cfg.comp_delay_ms),
                    "delay_max_ms": float(
                        env_cfg.get("delay_max_ms", self.cfg.comp_delay_ms)
                        if self._servo_planner_profile
                        else self.cfg.comp_delay_ms),
                    "delay_jitter_ms": 0.0,
                    "delay_sampling_mode": "uniform",
                })
            # Keep the deployed legacy mirror's compensation + software
            # actuator model exactly as configured by its existing UI knobs.
            fixed.update({
                "actuator_cmd_filter": True,
                "actuator_cmd_tau": max(
                    0.0, float(self.cfg.comp_actuator_tau_s)),
                "actuator_cmd_gain": float(self.cfg.comp_actuator_gain),
                "actuator_compensation_mode": (
                    str(self.cfg.compensation_mode)
                    if bool(self.cfg.enable_compensation) else "none"
                ),
                "actuator_mpc_beta": float(self.cfg.comp_mpc_beta),
                "actuator_mpc_delay_scale": float(
                    self.cfg.comp_mpc_delay_scale),
                "actuator_mpc_tau_scale": float(self.cfg.comp_mpc_tau_scale),
                "actuator_mpc_horizon_steps": max(
                    1, int(self.cfg.comp_mpc_horizon_steps)),
                "actuator_mpc_tracking_weight": float(
                    self.cfg.comp_mpc_tracking_weight),
                "actuator_mpc_nominal_weight": float(
                    self.cfg.comp_mpc_nominal_weight),
                "actuator_mpc_delta_weight": float(
                    self.cfg.comp_mpc_delta_weight),
                "actuator_mpc_max_delta_rad": max(
                    0.0, float(self.cfg.comp_mpc_max_delta_rad)),
            })
            if bool(getattr(self, "_servo_planner_profile", False)):
                # The planner is part of the frozen simcode4 plant.  Keep its
                # checkpoint-declared actual-feedback inverse-MPC contract and
                # drive-side 1.0/0.8 trajectory limits intact.
                fixed.update({
                    "actuator_cmd_filter": bool(env_cfg.get(
                        "actuator_cmd_filter", True)),
                    "actuator_cmd_tau": float(env_cfg.get(
                        "actuator_cmd_tau", 0.074)),
                    "actuator_cmd_gain": float(env_cfg.get(
                        "actuator_cmd_gain", 1.0)),
                    "actuator_compensation_mode": str(env_cfg.get(
                        "actuator_compensation_mode", "inverse_mpc")),
                    "actuator_mpc_beta": float(env_cfg.get(
                        "actuator_mpc_beta", 1.2)),
                    "actuator_mpc_delay_scale": float(env_cfg.get(
                        "actuator_mpc_delay_scale", 1.05)),
                    "actuator_mpc_tau_scale": float(env_cfg.get(
                        "actuator_mpc_tau_scale", 0.75)),
                    "actuator_mpc_horizon_steps": int(env_cfg.get(
                        "actuator_mpc_horizon_steps", 6)),
                    "actuator_mpc_tracking_weight": float(env_cfg.get(
                        "actuator_mpc_tracking_weight", 1.0)),
                    "actuator_mpc_nominal_weight": float(env_cfg.get(
                        "actuator_mpc_nominal_weight", 0.25)),
                    "actuator_mpc_delta_weight": float(env_cfg.get(
                        "actuator_mpc_delta_weight", 0.05)),
                    "actuator_mpc_max_delta_rad": float(env_cfg.get(
                        "actuator_mpc_max_delta_rad", math.pi / 6.0)),
                    "actuator_mpc_feedback_source": "actual",
                    "arm_servo_target_tracking_planner": True,
                    # Frozen submission order is authoritative for the mirror.
                    # The deployable pre-actuator order belongs to a retrained
                    # checkpoint/profile, never this frozen policy.
                    "arm_servo_target_planner_before_actuator": False,
                    "arm_servo_target_limiter": False,
                    "arm_servo_target_velocity_scale": 1.0,
                    "arm_servo_target_acceleration_scale": 0.8,
                })
        for key, value in fixed.items():
            if key in field_names:
                kwargs[key] = value
        return kwargs

    def _force_ball_state(
        self,
        state: Any,
        ball_pos: Any,
        ball_vel: Any,
        ball_valid: Any,
        age_s: Any,
    ) -> tuple[Any, Any]:
        jnp = self.jnp
        bpos_base = jnp.asarray(ball_pos, dtype=jnp.float32).reshape((1, 3))
        bvel_base = jnp.asarray(ball_vel, dtype=jnp.float32).reshape((1, 3))
        valid = jnp.asarray(ball_valid, dtype=bool).reshape((1,))
        age = jnp.maximum(
            jnp.asarray(age_s, dtype=jnp.float32).reshape((1,)),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        qpos = state.data.qpos
        qvel = state.data.qvel
        base_q = jnp.stack([
            qpos[:, self.env.base_x_qadr],
            qpos[:, self.env.base_y_qadr],
            qpos[:, self.env.base_yaw_qadr],
        ], axis=-1)
        base_dq = jnp.stack([
            qvel[:, self.env.base_x_vadr],
            qvel[:, self.env.base_y_vadr],
            qvel[:, self.env.base_yaw_vadr],
        ], axis=-1)
        c = jnp.cos(base_q[:, 2])
        s = jnp.sin(base_q[:, 2])
        rel_x = c * bpos_base[:, 0] - s * bpos_base[:, 1]
        rel_y = s * bpos_base[:, 0] + c * bpos_base[:, 1]
        rel_vx = c * bvel_base[:, 0] - s * bvel_base[:, 1]
        rel_vy = s * bvel_base[:, 0] + c * bvel_base[:, 1]
        bpos = jnp.stack([
            base_q[:, 0] + rel_x,
            base_q[:, 1] + rel_y,
            bpos_base[:, 2],
        ], axis=-1)
        bvel = jnp.stack([
            base_dq[:, 0] + rel_vx - base_dq[:, 2] * rel_y,
            base_dq[:, 1] + rel_vy + base_dq[:, 2] * rel_x,
            bvel_base[:, 2],
        ], axis=-1)
        qpos = qpos.at[:, self.env.ball_qadr:self.env.ball_qadr + 3].set(bpos)
        qpos = qpos.at[:, self.env.ball_qadr + 3:self.env.ball_qadr + 7].set(
            jnp.broadcast_to(
                jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
                (1, 4),
            )
        )
        qvel = qvel.at[:, self.env.ball_vadr:self.env.ball_vadr + 3].set(bvel)
        qvel = qvel.at[:, self.env.ball_vadr + 3:self.env.ball_vadr + 6].set(0.0)
        data = state.data.replace(qpos=qpos, qvel=qvel)
        data = self.env.batched_forward(state.model, data)

        cached_pos = jnp.where(valid[:, None], bpos, state.cached_ball_obs_pos)
        cached_vel = jnp.where(valid[:, None], bvel, state.cached_ball_obs_vel)
        valid_pos = jnp.where(valid[:, None], bpos, state.ball_obs_valid_pos)
        valid_vel = jnp.where(valid[:, None], bvel, state.ball_obs_valid_vel)
        age = jnp.where(valid, age, jnp.maximum(state.ball_obs_age_seconds, age))
        last_step = jnp.where(valid, state.step_count, state.last_ball_obs_step)
        state = state._replace(
            data=data,
            prev_ball_pos=bpos - bvel * float(self.env.dt),
            cached_ball_obs_pos=cached_pos,
            cached_ball_obs_vel=cached_vel,
            last_ball_obs_step=last_step,
            ball_obs_valid_pos=valid_pos,
            ball_obs_valid_vel=valid_vel,
            ball_obs_age_seconds=age,
            ball_obs_dropout_remaining=jnp.zeros_like(state.ball_obs_dropout_remaining),
        )
        obs = self.env.observe(state)
        return state, obs

    def _force_arm_state(self, state: Any, arm_q: Any) -> tuple[Any, Any]:
        jnp = self.jnp
        q = jnp.asarray(arm_q, dtype=jnp.float32).reshape((1, len(RIGHT_ARM_JOINTS)))
        zeros = jnp.zeros_like(q)
        qpos = state.data.qpos.at[:, self.env.arm_qadr].set(q)
        qvel = state.data.qvel.at[:, self.env.arm_vadr].set(zeros)
        if self._corrected_67d_profile or self._real_rmp_profile:
            ctrl = state.data.ctrl.at[:, self.env.arm_aids_j].set(q)
            data = state.data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)
        else:
            data = state.data.replace(qpos=qpos, qvel=qvel)
        data = self.env.batched_forward(state.model, data)
        updates = {
            "data": data,
            "arm_cmd_q": q,
            "arm_cmd_qvel": zeros,
            "arm_q_ref_latest": q,
            "arm_q_ref_active": q,
            "arm_actuator_q_ref_latest": q,
            "arm_actuator_q_ref_active": q,
            "arm_applied_q": q,
        }
        if self._corrected_67d_profile or self._real_rmp_profile:
            racket_pos = data.site_xpos[:, self.env.racket_site_id]
            command_buffer = jnp.broadcast_to(
                q[:, None, :], state.command_buffer.shape)
            updates.update({
                "racket_anchor": racket_pos,
                "prev_racket_pos": racket_pos,
                "arm_safe_q_ref_latest": q,
                "arm_safe_qvel": zeros,
                "arm_applied_qvel": zeros,
                "prev_action": zeros,
                "prev_arm_qvel": zeros,
                "command_buffer": command_buffer,
                "actuator_command_buffer": command_buffer,
                "action_buffer": jnp.zeros_like(state.action_buffer),
                "action_history": jnp.zeros_like(state.action_history),
                "anti_windup_scale": jnp.ones_like(state.anti_windup_scale),
            })
        state = state._replace(**updates)
        obs = self.env.observe(state)
        return state, obs

    def _policy_action_jax(self, obs: Any) -> Any:
        jnp = self.jnp
        w = self.policy_weights_jax
        x = jnp.asarray(obs, dtype=jnp.float32)
        x = jnp.tanh(x @ w["l1_w"] + w["l1_b"])
        x = jnp.tanh(x @ w["l2_w"] + w["l2_b"])
        return x @ w["out_w"] + w["out_b"]

    def _step_sequence_jax(
        self,
        state: Any,
        ball_pos_seq: Any,
        ball_vel_seq: Any,
        ball_valid_seq: Any,
        ball_age_seq: Any,
        observation_scale_vector: Any,
    ) -> tuple[Any, dict[str, Any]]:
        jnp = self.jnp

        def one_step(carry: Any, item: tuple[Any, Any, Any, Any]):
            step_state = carry
            bpos, bvel, valid, age = item
            step_state, obs = self._force_ball_state(
                step_state, bpos, bvel, valid, age)
            observation_joint_velocity = step_state.data.qvel[
                0, self.env.arm_vadr]
            observation_dq_max_abs_error = jnp.max(jnp.abs(
                obs[0, 7:14] - observation_joint_velocity))
            actor_obs = obs * observation_scale_vector[None, :]
            action = jnp.clip(
                self._policy_action_jax(actor_obs), -1.0, 1.0)
            previous_command_velocity = step_state.arm_cmd_qvel[0]
            step_state, _next_obs, reward, done, _metrics = self.env.step(
                step_state, action)
            data = step_state.data
            # This derivative is correct for both legacy acceleration actors
            # and simcode6's velocity actor after LPF/slew/qacc limiting.
            command_acceleration = (
                step_state.arm_cmd_qvel[0] - previous_command_velocity
            ) / float(self.env.dt)
            # Keep the scan output intentionally compact.  Returning only the
            # state needed by sim2real logging avoids copying the (large) full
            # MJX state to the host while still preserving a complete trace for
            # every policy tick in a fixed-size batch.
            sample = {
                "policy_obs": actor_obs[0],
                "policy_obs_unscaled": obs[0],
                "observation_joint_velocity": observation_joint_velocity,
                "observation_dq_max_abs_error": (
                    observation_dq_max_abs_error),
                "policy_action": action[0],
                "executed_action": step_state.prev_action[0],
                "joint_position": data.qpos[0, self.env.arm_qadr],
                "joint_velocity": data.qvel[0, self.env.arm_vadr],
                "joint_acceleration": data.qacc[0, self.env.arm_vadr],
                "actuator_target": data.ctrl[0, self.env.arm_aids_j],
                "policy_command": step_state.arm_q_ref_latest[0],
                "active_command": step_state.arm_q_ref_active[0],
                "compensated_command": step_state.arm_actuator_q_ref_latest[0],
                "active_compensated_command": step_state.arm_actuator_q_ref_active[0],
                "applied_command": step_state.arm_applied_q[0],
                "command_position": step_state.arm_cmd_q[0],
                "command_velocity": step_state.arm_cmd_qvel[0],
                "command_acceleration": command_acceleration,
                "racket_position": data.site_xpos[0, self.env.racket_site_id],
                "racket_matrix": data.site_xmat[0, self.env.racket_site_id],
                "reward": reward[0],
                "done": done[0],
                "ball": bpos,
                "velocity": bvel,
                "valid": valid,
                "age": jnp.maximum(age, jnp.asarray(0.0, dtype=jnp.float32)),
            }
            return step_state, sample

        return self.jax.lax.scan(
            one_step,
            state,
            (
                jnp.asarray(ball_pos_seq, dtype=jnp.float32),
                jnp.asarray(ball_vel_seq, dtype=jnp.float32),
                jnp.asarray(ball_valid_seq, dtype=bool),
                jnp.asarray(ball_age_seq, dtype=jnp.float32),
            ),
        )

    def _batch_samples_from_outputs(
        self,
        outputs: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Convert one device-fetched scan pytree into JSON-friendly samples."""
        reward_seq = np.asarray(outputs["reward"], dtype=np.float32).reshape(-1)
        done_seq = np.asarray(outputs["done"], dtype=bool).reshape(-1)
        samples: list[dict[str, Any]] = []
        for index in range(reward_seq.size):
            racket_mat = np.asarray(
                outputs["racket_matrix"][index], dtype=np.float32).reshape(3, 3)
            quat_wxyz = np.zeros(4, dtype=np.float64)
            mj.mju_mat2Quat(
                quat_wxyz, racket_mat.reshape(-1).astype(np.float64))
            racket_quat = _quat_wxyz_to_xyzw(quat_wxyz.astype(np.float32))
            reward = float(reward_seq[index])
            done = bool(done_seq[index])
            samples.append({
                "policyProfile": self.policy_profile,
                "policyModel": str(self.cfg.policy_model),
                "valid": bool(np.asarray(outputs["valid"][index])),
                "age": round(float(max(
                    0.0, np.asarray(outputs["age"][index], dtype=np.float32).item())), 6),
                "ball": _rounded(outputs["ball"][index]),
                "velocity": _rounded(outputs["velocity"][index]),
                "jointPosition": _rounded(outputs["joint_position"][index]),
                "jointVelocity": _rounded(outputs["joint_velocity"][index]),
                "jointAcceleration": _rounded(
                    outputs["joint_acceleration"][index]),
                "jointAccelerationSource": "mjx_qacc",
                "actuatorTarget": _rounded(outputs["actuator_target"][index]),
                "policyCommand": _rounded(outputs["policy_command"][index]),
                "compensatedCommand": _rounded(
                    outputs["compensated_command"][index]),
                "activeCommand": _rounded(outputs["active_command"][index]),
                "activeCompensatedCommand": _rounded(
                    outputs["active_compensated_command"][index]),
                "appliedCommand": _rounded(outputs["applied_command"][index]),
                "racket": _rounded(outputs["racket_position"][index]),
                "racketQuatXyzw": _rounded(racket_quat),
                "policyObs": _rounded(outputs["policy_obs"][index]),
                "policyObsUnscaled": _rounded(
                    outputs["policy_obs_unscaled"][index]),
                "policyJointVelocity": _rounded(
                    outputs["observation_joint_velocity"][index]),
                "dqObservationContract": {
                    "source": "MjxJuggleEnv.observe:data.qvel[:,arm_vadr]",
                    "maxAbsErrorRadS": float(np.asarray(
                        outputs["observation_dq_max_abs_error"][index],
                        dtype=np.float32,
                    )),
                    "matchesTrainingAndValidate": True,
                },
                "policyBallObservation": np.asarray(
                    outputs["policy_obs"][index], dtype=np.float32,
                ).reshape(-1)[20:26].tolist(),
                "policyAction": _rounded(outputs["policy_action"][index]),
                "policyExecutedAction": _rounded(
                    outputs["executed_action"][index]),
                "command": {
                    "position": _rounded(outputs["command_position"][index]),
                    "velocity": _rounded(outputs["command_velocity"][index]),
                    "acceleration": _rounded(
                        outputs["command_acceleration"][index]),
                },
                "servoPlanner": {
                    "enabled": bool(self._servo_planner_profile),
                    "feedbackSource": (
                        "actual_joint_state"
                        if self._servo_planner_profile
                        else "modeled_applied_target"),
                    "velocityScale": float(getattr(
                        self.env_cfg,
                        "arm_servo_target_velocity_scale",
                        1.0,
                    )),
                    "accelerationScale": float(getattr(
                        self.env_cfg,
                        "arm_servo_target_acceleration_scale",
                        1.0,
                    )),
                },
                "reward": round(reward, 6),
                "done": done,
                "mjx": {
                    "done": done,
                    "reward": round(reward, 6),
                    "metrics": {},
                    "dt": round(float(self.dt), 6),
                },
            })
        return samples

    def _snapshot_from_batch_sample(
        self,
        sample: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the normal public snapshot from the final scan sample."""
        return {
            "enabled": True,
            "available": True,
            "backend": "mjx_jax_real_ball",
            "policyProfile": self.policy_profile,
            "policyModel": str(self.cfg.policy_model),
            "timeWallS": round(float(time.time()), 6),
            "valid": bool(sample["valid"]),
            "age": sample["age"],
            "ball": sample["ball"],
            "velocity": sample["velocity"],
            "jointPosition": sample["jointPosition"],
            "jointVelocity": sample["jointVelocity"],
            "actuatorTarget": sample["actuatorTarget"],
            "policyCommand": sample["policyCommand"],
            "compensatedCommand": sample["compensatedCommand"],
            "activeCommand": sample["activeCommand"],
            "racket": sample["racket"],
            "racketQuatXyzw": sample["racketQuatXyzw"],
            "policyObs": sample["policyObs"],
            "policyObsUnscaled": sample.get(
                "policyObsUnscaled", sample["policyObs"]),
            "policyJointVelocity": sample.get("policyJointVelocity"),
            "dqObservationContract": dict(sample.get(
                "dqObservationContract", {})),
            "policyObservationScaleIndices": sample.get(
                "policyObservationScaleIndices", []),
            "policyObservationScales": sample.get(
                "policyObservationScales", []),
            "policyAction": sample["policyAction"],
            "policyExecutedAction": sample.get(
                "policyExecutedAction", sample["policyAction"]),
            "command": sample["command"],
            "servoPlanner": sample["servoPlanner"],
            "compensation": {
                "error": None,
                "available": True,
                "mode": str(self.env_cfg.actuator_compensation_mode),
            },
            "mjx": {
                "done": bool(sample["done"]),
                "reward": round(float(sample["reward"]), 6),
                "metrics": self.last_metrics,
                "dt": round(float(self.dt), 6),
            },
            "frame": dict(self.latest_frame_info),
            "renderError": self.render_error,
        }

    def prewarm(self, batch_size: int) -> None:
        """Compile the single-step and fixed-batch scan paths, then reset.

        Warm-up runs use a local reset state, so dummy policy ticks are never
        exposed as mirror samples or visualization frames.  The final reset
        leaves the mirror in the same externally visible state as a newly
        constructed instance.
        """
        requested_size = max(1, int(batch_size))
        key = self.jax.random.PRNGKey(0)[None, :]
        warm_state, _warm_obs = self._reset_jit(key)
        for size in dict.fromkeys((1, requested_size)):
            ball_pos = self.jnp.broadcast_to(
                warm_state.data.qpos[
                    0, self.env.ball_qadr:self.env.ball_qadr + 3],
                (size, 3),
            )
            ball_vel = self.jnp.broadcast_to(
                warm_state.data.qvel[
                    0, self.env.ball_vadr:self.env.ball_vadr + 3],
                (size, 3),
            )
            ball_valid = self.jnp.ones((size,), dtype=bool)
            ball_age = self.jnp.zeros((size,), dtype=self.jnp.float32)
            warm_state, outputs = self._step_sequence_jit(
                warm_state,
                ball_pos,
                ball_vel,
                ball_valid,
                ball_age,
                self.jnp.ones(
                    (self.obs_dim,), dtype=self.jnp.float32),
            )
            warm_state, warm_obs = self._force_ball_jit(
                warm_state,
                ball_pos[-1],
                ball_vel[-1],
                ball_valid[-1:],
                ball_age[-1:],
            )
            self.jax.block_until_ready(outputs["done"])
            self.jax.block_until_ready(warm_obs)
        self.reset()

    def reset(
        self,
        arm_q: np.ndarray | None = None,
        *,
        ball_pos: np.ndarray | None = None,
        ball_vel: np.ndarray | None = None,
        ball_valid: bool = False,
        ball_obs_age_s: float = 0.0,
    ) -> dict[str, Any]:
        key = self.jax.random.PRNGKey(0)[None, :]
        self.state, self.obs = self._reset_jit(key)
        if arm_q is not None:
            q = _finite_vec(arm_q, len(RIGHT_ARM_JOINTS))
            self.state, self.obs = self._force_arm_jit(self.state, q)
        if ball_pos is not None:
            bpos = _finite_vec(ball_pos, 3)
            bvel = (
                np.zeros(3, dtype=np.float32)
                if ball_vel is None else _finite_vec(ball_vel, 3)
            )
            valid = np.asarray([bool(ball_valid)], dtype=bool)
            age = np.asarray(
                [max(0.0, float(ball_obs_age_s))], dtype=np.float32)
            self.state, self.obs = self._force_ball_jit(
                self.state, bpos, bvel, valid, age)
        self.last_obs = np.asarray(
            self.jax.device_get(self.obs[0]), dtype=np.float32).reshape(-1)
        observation_joint_velocity = np.asarray(self.jax.device_get(
            self.state.data.qvel[0, self.env.arm_vadr]),
            dtype=np.float32,
        ).reshape(-1)
        dq_contract_error = float(np.max(np.abs(
            self.last_obs[7:14] - observation_joint_velocity)))
        if dq_contract_error > 1.0e-6:
            raise RuntimeError(
                "MJX policy dq contract mismatch at reset: "
                f"obs[7:14] vs data.qvel[arm_vadr] max_abs="
                f"{dq_contract_error:.9g} rad/s")
        self.last_policy_joint_velocity = observation_joint_velocity.copy()
        self.last_dq_observation_contract = {
            "source": "MjxJuggleEnv.observe:data.qvel[:,arm_vadr]",
            "maxAbsErrorRadS": dq_contract_error,
            "matchesTrainingAndValidate": True,
        }
        self.last_action = np.zeros((len(RIGHT_ARM_JOINTS),), dtype=np.float32)
        self.last_executed_action = np.zeros_like(self.last_action)
        self.last_command_acceleration = np.zeros_like(self.last_action)
        self.policy.reset()
        self.policy.last_obs = self.last_obs.copy()
        self.policy.last_action = self.last_action.copy()
        self.policy.last_executed_action = self.last_executed_action.copy()
        self.frame_seq = 0
        self.latest_frame = None
        self.latest_frame_info = {"seq": 0, "ageS": None}
        self.next_frame_wall = 0.0
        self.last_done = False
        self.last_reward = 0.0
        self.last_metrics = {}
        return self.snapshot(
            ball_pos=ball_pos,
            ball_vel=ball_vel,
            ball_valid=bool(ball_valid),
            ball_obs_age_s=max(0.0, float(ball_obs_age_s)),
        )

    def step(
        self,
        ball_pos: np.ndarray,
        ball_vel: np.ndarray,
        *,
        ball_valid: bool,
        ball_obs_age_s: float,
        options: Any,
        num_steps: int = 1,
    ) -> dict[str, Any]:
        if int(num_steps) <= 1:
            return self.step_sequence(
                np.asarray(ball_pos, dtype=np.float32).reshape(1, 3),
                np.asarray(ball_vel, dtype=np.float32).reshape(1, 3),
                ball_valid_seq=np.asarray([bool(ball_valid)]),
                ball_obs_age_seq=np.asarray(
                    [max(0.0, float(ball_obs_age_s))], dtype=np.float32),
                options=options,
            )
        return self.step_sequence(
            np.repeat(np.asarray(ball_pos, dtype=np.float32).reshape(1, 3), int(num_steps), axis=0),
            np.repeat(np.asarray(ball_vel, dtype=np.float32).reshape(1, 3), int(num_steps), axis=0),
            ball_valid_seq=np.full((int(num_steps),), bool(ball_valid), dtype=bool),
            ball_obs_age_seq=np.full(
                (int(num_steps),), max(0.0, float(ball_obs_age_s)), dtype=np.float32),
            options=options,
        )

    def step_sequence(
        self,
        ball_pos_seq: np.ndarray,
        ball_vel_seq: np.ndarray,
        *,
        ball_valid_seq: np.ndarray,
        ball_obs_age_seq: np.ndarray,
        options: Any,
    ) -> dict[str, Any]:
        bpos_seq = np.asarray(ball_pos_seq, dtype=np.float32).reshape(-1, 3)
        bvel_seq = np.asarray(ball_vel_seq, dtype=np.float32).reshape(-1, 3)
        valid_seq = np.asarray(ball_valid_seq, dtype=bool).reshape(-1)
        age_seq = np.maximum(
            np.asarray(ball_obs_age_seq, dtype=np.float32).reshape(-1), 0.0)
        if not bool(self.cfg.rl_use_valid_ball_obs_age):
            age_seq = np.where(valid_seq, 0.0, age_seq).astype(np.float32)
        n = min(bpos_seq.shape[0], bvel_seq.shape[0], valid_seq.shape[0], age_seq.shape[0])
        if n <= 0:
            snap = self.snapshot()
            snap["_batchSamples"] = []
            return snap
        bpos_seq = bpos_seq[:n]
        bvel_seq = bvel_seq[:n]
        valid_seq = valid_seq[:n]
        age_seq = age_seq[:n]
        observation_scale_vector = np.ones(
            (self.obs_dim,), dtype=np.float32)
        if not self._real_rmp_profile:
            candidate_scales = np.asarray(getattr(
                options, "policy_observation_scales", ()),
                dtype=np.float32,
            ).reshape(-1)
            if candidate_scales.shape == observation_scale_vector.shape:
                observation_scale_vector = np.where(
                    np.isfinite(candidate_scales), candidate_scales, 1.0,
                ).astype(np.float32)
        scale_indices = tuple(np.flatnonzero(
            observation_scale_vector != np.float32(1.0)
        ).astype(int).tolist())
        self.state, outputs = self._step_sequence_jit(
            self.state,
            bpos_seq,
            bvel_seq,
            valid_seq,
            age_seq,
            observation_scale_vector,
        )
        bpos = _finite_vec(bpos_seq[-1], 3)
        bvel = _finite_vec(bvel_seq[-1], 3)
        valid = np.asarray([bool(valid_seq[-1])])
        age = np.asarray([max(0.0, float(age_seq[-1]))], dtype=np.float32)
        self.state, self.obs = self._force_ball_jit(self.state, bpos, bvel, valid, age)

        # One transfer for the complete scan trace avoids a device
        # synchronization per field (and per simulation tick).
        outputs_np = self.jax.device_get(outputs)
        dq_contract_errors = np.asarray(
            outputs_np["observation_dq_max_abs_error"],
            dtype=np.float32,
        ).reshape(-1)
        max_dq_contract_error = float(np.max(dq_contract_errors))
        if max_dq_contract_error > 1.0e-6:
            raise RuntimeError(
                "MJX policy dq contract mismatch during step: "
                f"obs[7:14] vs data.qvel[arm_vadr] max_abs="
                f"{max_dq_contract_error:.9g} rad/s")
        samples = self._batch_samples_from_outputs(outputs_np)
        for sample in samples:
            sample["policyObservationScaleIndices"] = list(scale_indices)
            sample["policyObservationScales"] = _rounded(
                observation_scale_vector)
        obs_np = np.asarray(
            outputs_np["policy_obs"][-1], dtype=np.float32).reshape(-1)
        action = np.asarray(
            outputs_np["policy_action"][-1], dtype=np.float32).reshape(-1)
        executed_action = np.asarray(
            outputs_np["executed_action"][-1],
            dtype=np.float32,
        ).reshape(-1)
        command_acceleration = np.asarray(
            outputs_np["command_acceleration"][-1],
            dtype=np.float32,
        ).reshape(-1)
        self.policy.last_obs = obs_np.copy()
        self.policy.last_obs_unscaled = np.asarray(
            outputs_np["policy_obs_unscaled"][-1],
            dtype=np.float32,
        ).reshape(-1).copy()
        self.policy.last_observation_scale_indices = scale_indices
        self.policy.last_observation_scales = observation_scale_vector.copy()
        self.policy.last_action = action.copy()
        self.policy.last_executed_action = executed_action.copy()
        self.last_obs = obs_np.copy()
        self.last_policy_joint_velocity = np.asarray(
            outputs_np["observation_joint_velocity"][-1],
            dtype=np.float32,
        ).reshape(-1).copy()
        self.last_dq_observation_contract = {
            "source": "MjxJuggleEnv.observe:data.qvel[:,arm_vadr]",
            "maxAbsErrorRadS": float(dq_contract_errors[-1]),
            "batchMaxAbsErrorRadS": max_dq_contract_error,
            "matchesTrainingAndValidate": True,
        }
        self.last_action = action.copy()
        self.last_executed_action = executed_action.copy()
        self.last_command_acceleration = command_acceleration.copy()
        self.last_reward = float(
            np.asarray(outputs_np["reward"][-1], dtype=np.float32))
        self.last_done = bool(np.asarray(outputs_np["done"][-1]))
        self.last_metrics = {}
        snap = self._snapshot_from_batch_sample(samples[-1])
        snap["_batchSamples"] = samples
        self._maybe_update_frame(snap)
        return snap

    def _np(self, value: Any) -> np.ndarray:
        return np.asarray(self.jax.device_get(value), dtype=np.float32)

    def _racket_pose(self) -> tuple[np.ndarray, np.ndarray]:
        data = self.state.data
        pos = self._np(data.site_xpos[0, self.env.racket_site_id]).reshape(3)
        mat = self._np(data.site_xmat[0, self.env.racket_site_id]).reshape(3, 3)
        quat_wxyz = np.zeros(4, dtype=np.float64)
        mj.mju_mat2Quat(quat_wxyz, mat.reshape(-1).astype(np.float64))
        return pos, _quat_wxyz_to_xyzw(quat_wxyz.astype(np.float32))

    def snapshot(
        self,
        *,
        ball_pos: np.ndarray | None = None,
        ball_vel: np.ndarray | None = None,
        ball_valid: bool = False,
        ball_obs_age_s: float = 0.0,
    ) -> dict[str, Any]:
        data = self.state.data
        q = self._np(data.qpos[0, self.arm_qadr_np]).reshape(-1)
        dq = self._np(data.qvel[0, self.arm_vadr_np]).reshape(-1)
        ddq = self._np(data.qacc[0, self.arm_vadr_np]).reshape(-1)
        ctrl = self._np(data.ctrl[0, self.arm_aids_np]).reshape(-1)
        racket_pos, racket_quat = self._racket_pose()
        ball_p = _finite_vec(ball_pos, 3) if ball_pos is not None else (
            self._np(data.xpos[0, self.env.ball_body_id]).reshape(3))
        ball_v = _finite_vec(ball_vel, 3) if ball_vel is not None else np.zeros(3, dtype=np.float32)
        cmd_q = self._np(self.state.arm_cmd_q[0]).reshape(-1)
        cmd_qvel = self._np(self.state.arm_cmd_qvel[0]).reshape(-1)
        q_ref = self._np(self.state.arm_q_ref_latest[0]).reshape(-1)
        q_ref_active = self._np(self.state.arm_q_ref_active[0]).reshape(-1)
        actuator_q_ref = self._np(self.state.arm_actuator_q_ref_latest[0]).reshape(-1)
        cmd_qacc = self.last_command_acceleration
        return {
            "enabled": True,
            "available": True,
            "backend": "mjx_jax_real_ball",
            "policyProfile": self.policy_profile,
            "policyModel": str(self.cfg.policy_model),
            "timeWallS": round(float(time.time()), 6),
            "valid": bool(ball_valid),
            "age": round(float(max(0.0, ball_obs_age_s)), 6),
            "ball": _rounded(ball_p),
            "velocity": _rounded(ball_v),
            "jointPosition": _rounded(q),
            "jointVelocity": _rounded(dq),
            "jointAcceleration": _rounded(ddq),
            "jointAccelerationSource": "mjx_qacc",
            "actuatorTarget": _rounded(ctrl),
            "policyCommand": _rounded(q_ref),
            "compensatedCommand": _rounded(actuator_q_ref),
            "activeCommand": _rounded(q_ref_active),
            "racket": _rounded(racket_pos),
            "racketQuatXyzw": _rounded(racket_quat),
            "policyObs": _rounded(self.last_obs),
            "policyObsUnscaled": _rounded(getattr(
                self.policy, "last_obs_unscaled", self.last_obs)),
            "policyJointVelocity": _rounded(getattr(
                self,
                "last_policy_joint_velocity",
                self.last_obs[7:14],
            )),
            "dqObservationContract": dict(getattr(
                self, "last_dq_observation_contract", {})),
            "policyObservationScaleIndices": list(getattr(
                self.policy, "last_observation_scale_indices", ())),
            "policyObservationScales": _rounded(getattr(
                self.policy,
                "last_observation_scales",
                np.ones_like(self.last_obs),
            )),
            "policyAction": _rounded(self.last_action),
            "policyExecutedAction": _rounded(self.last_executed_action),
            "command": {
                "position": _rounded(cmd_q),
                "velocity": _rounded(cmd_qvel),
                "acceleration": _rounded(cmd_qacc),
            },
            "compensation": {
                "error": None,
                "available": True,
                "mode": str(self.env_cfg.actuator_compensation_mode),
            },
            "mjx": {
                "done": bool(self.last_done),
                "reward": round(float(self.last_reward), 6),
                "metrics": self.last_metrics,
                "dt": round(float(self.dt), 6),
            },
            "frame": dict(self.latest_frame_info),
            "renderError": self.render_error,
        }

    def latest_visualization(self) -> tuple[bytes | None, dict[str, Any]]:
        return self.latest_frame, dict(self.latest_frame_info)

    def _maybe_update_frame(self, snap: dict[str, Any]) -> None:
        if not bool(self.cfg.enable_video_frames):
            return
        now = time.time()
        period = 1.0 / max(float(self.cfg.frame_rate_hz), 1e-6)
        if now < self.next_frame_wall:
            return
        frame = self._render_frame(snap)
        if frame is None:
            return
        self.frame_seq += 1
        self.latest_frame = frame
        self.latest_frame_info = {
            "seq": int(self.frame_seq),
            "wallTimeS": round(float(now), 6),
            "width": int(self.cfg.render_width),
            "height": int(self.cfg.render_height),
            "backend": "mjx_mujoco_renderer" if self.renderer is not None else "fallback_2d",
        }
        self.next_frame_wall = now + period

    def _copy_env0_to_render_data(self) -> None:
        if self.render_data is None:
            self.render_data = mj.MjData(self.env.mj_model)
        self.render_data.qpos[:] = np.asarray(
            self.jax.device_get(self.state.data.qpos[0]), dtype=np.float64)
        self.render_data.qvel[:] = np.asarray(
            self.jax.device_get(self.state.data.qvel[0]), dtype=np.float64)
        self.render_data.ctrl[:] = np.asarray(
            self.jax.device_get(self.state.data.ctrl[0]), dtype=np.float64)
        mj.mj_forward(self.env.mj_model, self.render_data)

    def _render_frame(self, snap: dict[str, Any]) -> bytes | None:
        if bool(self.cfg.use_mujoco_renderer) and self.render_error is None:
            try:
                if self.renderer is None:
                    self.renderer = mj.Renderer(
                        self.env.mj_model,
                        width=int(self.cfg.render_width),
                        height=int(self.cfg.render_height),
                    )
                self._copy_env0_to_render_data()
                self.renderer.update_scene(self.render_data)
                rgb = self.renderer.render()
                if rgb is not None and _CV2_AVAILABLE:
                    ok, enc = cv2.imencode(
                        ".jpg",
                        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [int(cv2.IMWRITE_JPEG_QUALITY), 78],
                    )
                    if ok:
                        return enc.tobytes()
            except Exception as exc:  # noqa: BLE001 - use fallback renderer.
                self.render_error = str(exc)
        return self._render_fallback_frame(snap)

    def _render_fallback_frame(self, snap: dict[str, Any]) -> bytes | None:
        if not _CV2_AVAILABLE:
            self.render_error = (
                "cv2 unavailable" if _CV2_IMPORT_ERROR is None
                else f"cv2 unavailable: {_CV2_IMPORT_ERROR}")
            return None
        return _encode_fallback_frame(self.cfg, snap, "mjx/jax mirror x/z")

    def close(self) -> None:
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None
