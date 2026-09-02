"""MJX/JAX PPO checkpoint adapter for the right-arm juggling controller.

The training pipeline in ``tools/rl_sim/train_juggle_mjx_curriculum.py``
saves checkpoints as pickled dicts (``*.pkl``) whose ``params`` entry is a
nested dict of plain NumPy ``float32`` arrays — the actor/critic MLP
weights — alongside the env config used during training. This is a
different format from the stable-baselines3 ``.zip`` that
:class:`rl_policy.RLPolicyController` loads via ``PPO.load``.

:class:`MjxRLPolicyController` loads such a ``.pkl`` and runs the policy
forward pass in pure NumPy (a 2-layer ``tanh`` MLP, deterministic =
policy mean), so deployment needs neither JAX nor MuJoCo. The observation
layout, FK, and action-to-joint integration are inherited unchanged from
``RLPolicyController`` because the MJX env builds its 50-D observation in
exactly the same order:

    [q, dq, base_q, base_dq, ball_pos_base, ball_vel_base,
     racket_pos_base, racket_vel_base, rel_base,
     prev_action, arm_cmd_error, age_norm]

The adapter additionally reads the checkpoint's ``env_cfg`` to align the
runtime integration with how the policy was trained:
``action_acc_scale``, ``ball_obs_age_clip``, the arm velocity/accel
limits, and the ``arm_action_limiter`` flag.

Loading deliberately avoids importing the training module: the pickle
references a few JAX/training globals (e.g. the optimizer state
``NamedTuple``) that would otherwise drag in JAX. A scoped ``Unpickler``
stubs out every class from the training/JAX/MuJoCo modules, so only the
NumPy ``params`` survive — which is all inference needs.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from pingpong_controller.rl_policy import (
    ACTION_COMMAND_MODE_VELOCITY,
    DEFAULT_ACTION_ACC_SCALE,
    DEFAULT_ARM_ACC_LIMIT_DEG_S2,
    DEFAULT_ARM_VEL_LIMIT_DEG_S,
    DEFAULT_BALL_OBS_AGE_CLIP_S,
    EXPECTED_OBS_DIM,
    EXPECTED_OBS_DIM_GPU0_REAL_RMP,
    EXPECTED_OBS_DIM_WITH_DELAY,
    OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85,
    OBSERVATION_PROFILE_GPU1_QACC_FT_V153,
    OBSERVATION_PROFILE_SIMCODE2_IDEALPD67,
    OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67,
    OBSERVATION_PROFILE_SIMCODE4_SERVO67,
    OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67,
    OBSERVATION_PROFILE_SIMCODE6_QVEL67,
    RLPolicyController,
    uses_corrected_67d_observation,
)


# Modules whose classes we never want to actually import while unpickling a
# checkpoint. Anything from these is replaced with a no-op stub so that only
# the plain-NumPy `params` tree is reconstructed.
_STUBBED_MODULE_PREFIXES = (
    "train_juggle_mjx_ppo",
    "train_juggle_mjx_curriculum",
    "jax",
    "jaxlib",
    "mjx",
    "mujoco",
    "flax",
    "optax",
    "chex",
)


class _StubType(tuple):
    """Swallows constructor args; stands in for any stubbed global.

    Subclasses tuple so that NamedTuple payloads (which pickle restores by
    calling the type with positional fields) reconstruct without error. We
    never read these objects — only ``params`` (a plain dict) is used.
    """

    def __new__(cls, *args):
        return super().__new__(cls, args if args else ())

    def __call__(self, *args, **kwargs):
        return self


class _CheckpointUnpickler(pickle.Unpickler):
    """Unpickler that stubs JAX/training globals so JAX need not be present."""

    def find_class(self, module: str, name: str):
        if any(module == p or module.startswith(p + ".")
               for p in _STUBBED_MODULE_PREFIXES):
            return _StubType
        return super().find_class(module, name)


def inspect_mjx_checkpoint_metadata(model_path: str) -> dict:
    """Read lightweight deployment metadata from an MJX checkpoint.

    The pickle still has to be loaded because the training checkpoint stores
    metadata and params in one object, but callers can use this before building
    the controller to choose the correct 50D/67D observation path.
    """
    with open(model_path, "rb") as handle:
        payload = _CheckpointUnpickler(handle).load()
    if not isinstance(payload, dict):
        raise RuntimeError(f"MJX checkpoint is not a dict: {model_path}")
    return {
        "obs_dim": int(payload.get("obs_dim", 0) or 0),
        "act_dim": int(payload.get("act_dim", 0) or 0),
        "env_cfg": dict(payload.get("env_cfg") or {}),
    }


def _flatten_params(params: dict) -> dict:
    """Flatten the nested param tree into 'pi/l1/w'-style keys."""
    flat: dict = {}

    def _walk(node, prefix):
        if isinstance(node, dict):
            for key, value in node.items():
                _walk(value, f"{prefix}{key}/")
        else:
            flat[prefix.rstrip("/")] = node

    _walk(params, "")
    return flat


class MjxRLPolicyController(RLPolicyController):
    """RL controller that loads an MJX/JAX PPO ``.pkl`` checkpoint.

    Drop-in for :class:`RLPolicyController`: same ``predict`` signature,
    same observation/FK pipeline. Only the model load and forward pass
    differ. Pass the same constructor kwargs; limits and scales are read
    from the checkpoint's ``env_cfg`` when ``align_with_checkpoint`` is
    True (default), overriding the constructor defaults so the deployed
    integration matches training.
    """

    def __init__(
        self,
        model_path: str,
        *,
        align_with_checkpoint: bool = True,
        **kwargs,
    ) -> None:
        # Stash before super().__init__ runs, because _load_model (called
        # by the base constructor) needs it and runs before we return here.
        self._align_with_checkpoint = bool(align_with_checkpoint)
        self._pi_weights: Optional[dict] = None
        self._checkpoint_env_cfg: dict = {}
        super().__init__(model_path, **kwargs)

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def _load_model(self, model_path: str, device: str) -> None:
        del device  # NumPy forward pass is CPU-only; device is ignored.
        self._logger.info(f"Loading PPO policy checkpoint from: {model_path}")
        with open(model_path, "rb") as handle:
            payload = _CheckpointUnpickler(handle).load()

        if not isinstance(payload, dict) or "params" not in payload:
            raise RuntimeError(
                f"MJX checkpoint missing 'params': {model_path}")

        obs_dim = int(payload.get("obs_dim", self._expected_obs_dim))
        act_dim = int(payload.get("act_dim", self.n_arm))

        # Validate against the expected dimension (50 or 67 based on enable_delay_conditioning)
        if obs_dim != self._expected_obs_dim or act_dim != self.n_arm:
            raise RuntimeError(
                f"MJX checkpoint obs_dim/act_dim ({obs_dim}, {act_dim}) "
                f"does not match expected ({self._expected_obs_dim}, {self.n_arm}). "
                f"Checkpoint requires delay_conditioning={obs_dim == EXPECTED_OBS_DIM_WITH_DELAY}, "
                f"but controller has enable_delay_conditioning={self.enable_delay_conditioning}")

        flat = _flatten_params(payload["params"])
        required = [
            "pi/l1/w", "pi/l1/b",
            "pi/l2/w", "pi/l2/b",
            "pi/out/w", "pi/out/b",
        ]
        missing = [k for k in required if k not in flat]
        if missing:
            raise RuntimeError(
                f"MJX checkpoint policy params missing keys: {missing}")

        # Cast to contiguous float32 NumPy. The actor MLP is all we need
        # for deterministic inference (mean action); value net is dropped.
        self._pi_weights = {
            "l1_w": np.ascontiguousarray(flat["pi/l1/w"], dtype=np.float32),
            "l1_b": np.ascontiguousarray(flat["pi/l1/b"], dtype=np.float32),
            "l2_w": np.ascontiguousarray(flat["pi/l2/w"], dtype=np.float32),
            "l2_b": np.ascontiguousarray(flat["pi/l2/b"], dtype=np.float32),
            "out_w": np.ascontiguousarray(flat["pi/out/w"], dtype=np.float32),
            "out_b": np.ascontiguousarray(flat["pi/out/b"], dtype=np.float32),
        }
        self._validate_weight_shapes()

        self._checkpoint_env_cfg = dict(payload.get("env_cfg") or {})
        if self._align_with_checkpoint:
            self._apply_checkpoint_env_cfg(self._checkpoint_env_cfg)
        self._validate_checkpoint_contract(payload)

        # `self.model` stays None: a lightweight shim provides the obs/act
        # shapes that `_validate_io_shapes` checks against.
        self.model = _MjxModelShim(self._expected_obs_dim, self.n_arm)
        if not self.deterministic:
            self._logger.warn(
                "MjxRLPolicyController only supports deterministic inference "
                "(policy mean); ignoring deterministic=False")

    def _validate_weight_shapes(self) -> None:
        w = self._pi_weights
        hidden = w["l1_w"].shape[1]
        checks = [
            (w["l1_w"].shape, (self._expected_obs_dim, hidden)),
            (w["l1_b"].shape, (hidden,)),
            (w["l2_w"].shape, (hidden, hidden)),
            (w["l2_b"].shape, (hidden,)),
            (w["out_w"].shape, (hidden, self.n_arm)),
            (w["out_b"].shape, (self.n_arm,)),
        ]
        for got, expected in checks:
            if tuple(got) != tuple(expected):
                raise RuntimeError(
                    f"MJX policy weight shape mismatch: got {got}, "
                    f"expected {expected}")

    def _apply_checkpoint_env_cfg(self, cfg: dict) -> None:
        """Align runtime integration knobs with the training env config."""
        action_mode = cfg.get("action_command_mode")
        if action_mode is not None:
            self.action_command_mode = str(action_mode).strip().lower()

        acc_scale = cfg.get("action_acc_scale")
        if acc_scale is not None:
            self.action_acc_scale = float(acc_scale)

        velocity_scale = cfg.get("action_velocity_scale")
        if velocity_scale is not None:
            self.action_velocity_scale = float(velocity_scale)

        filter_tau_ms = cfg.get("action_filter_tau_ms")
        if filter_tau_ms is not None:
            self.action_filter_tau_ms = max(0.0, float(filter_tau_ms))

        jerk_limit = cfg.get("action_jerk_limit")
        if jerk_limit is not None:
            self.action_jerk_limit = max(0.0, float(jerk_limit))

        action_acc_limit = cfg.get("action_acc_limit")
        if action_acc_limit is not None:
            self.action_acc_limit = float(action_acc_limit)

        age_clip = cfg.get("ball_obs_age_clip")
        if age_clip is not None:
            self.ball_obs_age_clip_s = max(1e-6, float(age_clip))

        limiter = cfg.get("arm_action_limiter")
        if limiter is not None:
            self.arm_action_limiter = bool(limiter)

        vel_limit = cfg.get("arm_vel_limit_deg_s")
        if vel_limit is not None and len(vel_limit) == self.n_arm:
            self.arm_vel_limit_rad_s = np.deg2rad(
                np.asarray(vel_limit, dtype=np.float32))

        acc_limit = cfg.get("arm_acc_limit_deg_s2")
        if acc_limit is not None and len(acc_limit) == self.n_arm:
            self.arm_acc_limit_rad_s2 = np.deg2rad(
                np.asarray(acc_limit, dtype=np.float32))

        # These fields define the new simcode2 observation/history contract.
        # Keep them off the legacy BUG-compatible path, whose reset pose,
        # buffer indexing and contact-time behaviour must remain unchanged.
        if uses_corrected_67d_observation(self.observation_profile):
            pd_profile = cfg.get("right_arm_pd_profile")
            if pd_profile is not None:
                self.right_arm_pd_profile = str(pd_profile).strip().lower()

            reset_degrees = cfg.get("right_arm_reset_degrees")
            if reset_degrees is not None and len(reset_degrees) == self.n_arm:
                self.initial_arm_q_rad = np.deg2rad(
                    np.asarray(reset_degrees, dtype=np.float32))

            delay_max_ms = cfg.get("delay_max_ms")
            if delay_max_ms is not None:
                self.delay_conditioning_max_s = max(
                    1e-6, float(delay_max_ms) * 1e-3)
            delay_min_ms = cfg.get("delay_min_ms", delay_max_ms)
            if delay_min_ms is not None:
                self.delay_conditioning_tau_s = max(
                    0.0, float(delay_min_ms) * 1e-3)
            buffer_extra = cfg.get("command_buffer_extra_steps")
            if buffer_extra is not None:
                self.command_buffer_extra_steps = max(0, int(buffer_extra))

            max_contact_time = cfg.get("max_contact_time")
            if max_contact_time is not None:
                self.max_contact_time_s = max(0.0, float(max_contact_time))
            contact_height_offset = cfg.get("contact_height_offset")
            if contact_height_offset is not None:
                self.contact_height_offset_m = float(contact_height_offset)
            lost_timeout_ms = cfg.get("lost_ball_timeout_ms")
            if lost_timeout_ms is not None:
                self.contact_lost_ball_timeout_s = max(
                    0.0, float(lost_timeout_ms) * 1e-3)

        self._logger.info(
            "Policy checkpoint env_cfg applied: "
            f"action_command_mode={self.action_command_mode}, "
            f"action_velocity_scale={self.action_velocity_scale}, "
            f"action_filter_tau_ms={self.action_filter_tau_ms}, "
            f"action_jerk_limit={self.action_jerk_limit}, "
            f"action_acc_scale={self.action_acc_scale}, "
            f"ball_obs_age_clip_s={self.ball_obs_age_clip_s}, "
            f"arm_action_limiter={self.arm_action_limiter}, "
            f"nominal_delay_s={self.delay_conditioning_tau_s}, "
            f"command_buffer_extra_steps={self.command_buffer_extra_steps}")

    def _validate_checkpoint_contract(self, payload: dict) -> None:
        """Hook for named adapters that enforce more than tensor dimensions."""
        del payload

    def _policy_action(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic forward pass of the actor MLP, in pure NumPy.

        Mirrors ``train_juggle_mjx_ppo.apply_mlp``:
            x = tanh(obs @ l1_w + l1_b)
            x = tanh(x   @ l2_w + l2_b)
            mean = x @ out_w + out_b
        The deterministic action is the policy mean (no log_std sampling).
        """
        w = self._pi_weights
        x = np.asarray(obs, dtype=np.float32).reshape(-1)
        x = np.tanh(x @ w["l1_w"] + w["l1_b"])
        x = np.tanh(x @ w["l2_w"] + w["l2_b"])
        mean = x @ w["out_w"] + w["out_b"]
        return np.asarray(mean, dtype=np.float32).reshape(self.n_arm)


class _MjxModelShim:
    """Minimal stand-in exposing obs/action shapes for shape validation."""

    class _Space:
        def __init__(self, shape):
            self.shape = shape

    def __init__(self, obs_dim: int, act_dim: int):
        self.observation_space = self._Space((obs_dim,))
        self.action_space = self._Space((act_dim,))


class Simcode2IdealPd67PolicyController(MjxRLPolicyController):
    """Dedicated real-runtime path for the corrected simcode2 ideal-PD67 actor.

    Keeping a named controller makes it impossible to select the new checkpoint
    while silently retaining the legacy BUG-compatible observation contract.
    """

    def __init__(self, model_path: str, **kwargs) -> None:
        kwargs["enable_delay_conditioning"] = True
        # Valid-frame age feeds obs[49] and the corrected causal contact phase;
        # zeroing it would no longer reproduce the frozen simcode2 actor input.
        kwargs["use_valid_ball_obs_age"] = True
        kwargs["observation_profile"] = OBSERVATION_PROFILE_SIMCODE2_IDEALPD67
        super().__init__(model_path, **kwargs)


class Simcode3Actuator67PolicyController(MjxRLPolicyController):
    """Real-runtime path for simcode3's actuator-model + inverse-MPC actor.

    The actor observation/history contract is the corrected 67D contract, but
    the distinct profile prevents its checkpoint and XML from being silently
    paired with simcode2.  The physical robot supplies the fitted actuator
    plant; this controller only builds nominal policy commands and observations.
    """

    def __init__(self, model_path: str, **kwargs) -> None:
        kwargs["enable_delay_conditioning"] = True
        kwargs["use_valid_ball_obs_age"] = True
        kwargs["observation_profile"] = OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67
        super().__init__(model_path, **kwargs)


class Simcode4Servo67PolicyController(MjxRLPolicyController):
    """Runtime adapter for simcode4's inverse-MPC + servo-planner actor.

    Simcode4 deliberately keeps observation age at zero while a non-missing
    60 Hz camera sample is held on the 200 Hz control clock.  The frozen MJX
    mirror keeps its original post-FOPDT planner; hardware without a drive-side
    planner uses the explicitly non-equivalent upstream software planner in the
    interactive probe and never replays FOPDT in software.
    """

    def __init__(self, model_path: str, **kwargs) -> None:
        kwargs["enable_delay_conditioning"] = True
        kwargs["use_valid_ball_obs_age"] = False
        kwargs["observation_profile"] = OBSERVATION_PROFILE_SIMCODE4_SERVO67
        super().__init__(model_path, **kwargs)


class Simcode5PureActuator67PolicyController(MjxRLPolicyController):
    """Runtime adapter for simcode5's direct pure-actuator checkpoint.

    The simulated plant owns the checkpoint-declared per-joint command delay
    and second-order actuator response.  Hardware already has those joint
    characteristics, so this controller retains delayed command history only
    for the actor's 67D observation and returns the latest policy command.
    Held, non-missing 60 Hz ball frames keep observation age at zero, matching
    ``ball_obs_age_tracks_stale=false`` in the checkpoint.
    """

    def __init__(self, model_path: str, **kwargs) -> None:
        kwargs["enable_delay_conditioning"] = True
        kwargs["use_valid_ball_obs_age"] = False
        kwargs["observation_profile"] = (
            OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67)
        super().__init__(model_path, **kwargs)


class Simcode6Qvel67PolicyController(MjxRLPolicyController):
    """Fail-closed adapter for simcode6's direct qvel actor and physical plant.

    Matching only the 67-by-7 tensor shapes is insufficient here: the same
    shapes were used by acceleration-command checkpoints.  This adapter also
    verifies every field that changes actor input semantics or command
    integration before allowing inference.
    """

    _REQUIRED_EXACT = {
        "action_command_mode": ACTION_COMMAND_MODE_VELOCITY,
        "arm_action_limiter": True,
        "enable_anti_windup": False,
        "enable_delay_conditioning": True,
        "include_tau_act_norm": True,
        "include_command_state": True,
        "include_active_command_error": True,
        "include_phase_features": True,
        "frame_skip": 5,
        "ball_obs_fractional_rate": True,
        "ball_obs_age_tracks_stale": False,
        "ball_obs_dropout_on_refresh_only": False,
        "ball_obs_velocity_observer_mode": "raw",
        "actor_mask_previous_action": False,
        "high_latency_obs": False,
        "use_delay_embedding": False,
        "delay_embedding_dim": 0,
        "actuator_cmd_model": "second_order",
        "actuator_cmd_filter": True,
        "actuator_delay_observation_only": False,
        "actuator_compensation_mode": "none",
        "arm_servo_target_tracking_planner": False,
        "right_arm_pd_profile": "sport_taskspace_fit_v1",
    }
    _REQUIRED_FLOAT = {
        "action_velocity_scale": 0.85,
        "action_filter_tau_ms": 15.0,
        "action_jerk_limit": 30.0,
        "action_acc_limit": 1.0,
        "actor_previous_action_scale": 1.0,
        "actor_action_dc_rejection": 0.0,
        "proprio_dq_obs_noise_std": 0.0,
        "proprio_racket_vel_obs_noise_std": 0.0,
        "proprio_obs_one_step_stale_probability": 0.0,
        "ball_obs_rate_hz": 60.0,
        "ball_obs_age_clip": 0.5,
        "ball_obs_velocity_observer_tau_ms": 0.0,
        "ball_obs_velocity_observer_max_innovation_m_s": 0.0,
        "delay_min_ms": 45.0,
        "delay_max_ms": 45.0,
    }
    _REQUIRED_VECTOR = {
        "right_arm_reset_degrees": (
            5.736, -44.399, 30.683, 97.142, 49.323, -12.269, 14.214),
        "arm_vel_limit_deg_s": (
            210.0, 210.0, 240.0, 240.0, 300.0, 300.0, 300.0),
        "arm_acc_limit_deg_s2": (
            1300.0, 1300.0, 1800.0, 3000.0, 3000.0, 3000.0, 3000.0),
        "actuator_cmd_delay_ms_per_joint": (
            45.0, 50.0, 45.0, 40.0, 35.0, 45.0, 55.0),
        "actuator_cmd_natural_frequency_rad_s": (
            20.36384925, 22.65597475, 22.6504705, 21.730533,
            19.66336425, 22.81552625, 23.64581725),
        "actuator_cmd_damping_ratio": (
            0.391768, 0.366169, 0.345738, 0.346,
            0.347896, 0.345814, 0.380713),
        "actuator_cmd_gain_per_joint": (
            0.99788351, 0.9966119, 0.99231151, 0.99094239,
            0.982326685, 0.992694585, 0.983194325),
    }

    def __init__(self, model_path: str, **kwargs) -> None:
        kwargs["enable_delay_conditioning"] = True
        kwargs["use_valid_ball_obs_age"] = False
        kwargs["observation_profile"] = OBSERVATION_PROFILE_SIMCODE6_QVEL67
        super().__init__(model_path, **kwargs)
        if not np.isclose(self.dt, 0.005, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                "simcode6_qvel67 requires the frozen 200 Hz/0.005 s "
                f"control step, got dt={self.dt!r}")

    def _validate_checkpoint_contract(self, payload: dict) -> None:
        cfg = dict(payload.get("env_cfg") or {})
        mismatches = []
        for key, expected in self._REQUIRED_EXACT.items():
            got = cfg.get(key, "<missing>")
            if got != expected:
                mismatches.append(f"{key}={got!r} (expected {expected!r})")
        for key, expected in self._REQUIRED_FLOAT.items():
            got = cfg.get(key)
            try:
                matches = bool(np.isclose(
                    float(got), expected, rtol=0.0, atol=1e-9))
            except (TypeError, ValueError):
                matches = False
            if not matches:
                mismatches.append(f"{key}={got!r} (expected {expected!r})")

        for key, expected in self._REQUIRED_VECTOR.items():
            try:
                actual = np.asarray(cfg.get(key), dtype=np.float64)
                valid = bool(
                    actual.shape == (self.n_arm,)
                    and np.all(np.isfinite(actual))
                    and np.allclose(
                        actual,
                        np.asarray(expected, dtype=np.float64),
                        rtol=0.0,
                        atol=1e-8,
                    )
                )
            except (TypeError, ValueError):
                valid = False
            if not valid:
                mismatches.append(
                    f"{key}={cfg.get(key)!r} (expected {expected!r})")

        if int(cfg.get("command_buffer_extra_steps", -1)) != 4:
            mismatches.append(
                "command_buffer_extra_steps must be 4 for the 45 ms history")
        if tuple(cfg.get("dr_action_latency_steps_range", ())) != (0, 0):
            mismatches.append(
                "dr_action_latency_steps_range must be (0, 0)")
        if mismatches:
            raise RuntimeError(
                "simcode6 checkpoint deployment contract mismatch: "
                + "; ".join(mismatches))


class Gpu1QaccFtV153PolicyController(MjxRLPolicyController):
    """Authenticated real/mirror adapter for GPU1-QACC-FT V153.

    The physical robot receives the latest command-state ``q_cmd``.  The
    checkpoint's delayed second-order actuator is deliberately *not* replayed
    here; the CPU simulation mirror reconstructs it from ``env_cfg``.  The
    actor's only changed observation dimension is obs[49]: zero before the
    first confirmed hit and represented-ball age afterwards.
    """

    CHECKPOINT_SHA256 = (
        "a02b0421029e808b84ea0ee5cf615b3e65c6df3bc28c0f4766ad9953c0e85176"
    )
    PROFILE_NAME = (
        "goal_d455_sport_taskspace_record_new3_sim2real_fixed_base_ball4g_"
        "dual_domain_homotopy_v153"
    )
    ZERO_OBSERVATION_ACTOR_MEAN = np.asarray(
        [0.264987797, 0.095902964, -0.141679987, 0.130324334,
         -0.013870809, -0.059728369, -0.005993952],
        dtype=np.float32,
    )
    LOST_BALL_TIMEOUT_S = 0.350
    MAXIMUM_JOINT_FEEDBACK_AGE_S = 0.010

    _REQUIRED_EXACT = {
        "action_command_mode": "acceleration",
        "action_acc_scale": 1.0,
        "policy_integration_feedback_source": "command",
        "recovered_rmp_motion_mode": False,
        "include_qvel_reference_error_obs": False,
        "enable_delay_conditioning": True,
        "high_latency_obs": False,
        "include_tau_act_norm": True,
        "include_command_state": True,
        "include_active_command_error": True,
        "include_phase_features": True,
        "use_delay_embedding": False,
        "ball_obs_rate_hz": 60.0,
        "ball_obs_fractional_rate": True,
        "ball_obs_age_tracks_stale": False,
        "ball_obs_age_clip": 0.5,
        "ball_obs_velocity_observer_mode": "raw",
        "ball_obs_camera_missing_prob": 0.0,
        "ball_obs_dropout_prob": 0.0,
        "ball_obs_dropout_burst_prob": 0.0,
        "dr_post_hit_ball_obs_latency": True,
        "dr_post_hit_ball_obs_latency_probability": 1.0,
        "dr_post_hit_ball_obs_latency_min_confirmed_hits": 1,
        "dr_post_hit_ball_obs_latency_timestamp_consistent_age": True,
        "arm_action_limiter": True,
        "actuator_cmd_filter": True,
        "actuator_cmd_model": "second_order",
        "actuator_compensation_mode": "none",
        "arm_servo_target_tracking_planner": False,
        "right_arm_pd_profile": "sport_taskspace_fit_v1",
        "action_filter_tau_ms": 0.0,
        "action_jerk_limit": 0.0,
        "enable_anti_windup": False,
        "delay_min_ms": 45.0,
        "delay_max_ms": 45.0,
        "simulation_base_mode": "aligned_fixed",
        "ball_reset_mode": "falling_contact",
        "dr_ball_normalized_inertia_range": (0.4, 0.4),
    }
    _REQUIRED_VECTOR = {
        "arm_vel_limit_deg_s": (
            210.0, 210.0, 240.0, 240.0, 300.0, 300.0, 300.0),
        "arm_acc_limit_deg_s2": (
            1300.0, 1300.0, 1800.0, 3000.0, 3000.0, 3000.0, 3000.0),
        "actuator_cmd_delay_ms_per_joint": (
            45.0, 50.0, 45.0, 40.0, 35.0, 45.0, 50.0),
        "actuator_cmd_natural_frequency_rad_s": (
            21.8911379437, 22.0895753812, 22.6504705, 21.730533,
            20.1549483562, 22.2451380938, 23.0546718187),
        "actuator_cmd_damping_ratio": (
            0.3330028, 0.3295521, 0.3111642, 0.3114,
            0.3131064, 0.3112326, 0.28553475),
        "actuator_cmd_gain_per_joint": (
            0.99788351, 0.9966119, 0.99231151, 0.99094239,
            0.982326685, 0.992694585, 0.983194325),
    }

    def __init__(self, model_path: str, **kwargs) -> None:
        kwargs["enable_delay_conditioning"] = True
        kwargs["use_valid_ball_obs_age"] = True
        kwargs["observation_profile"] = OBSERVATION_PROFILE_GPU1_QACC_FT_V153
        kwargs["expected_obs_dim"] = EXPECTED_OBS_DIM_WITH_DELAY
        kwargs["ball_obs_age_clip_s"] = 0.5
        super().__init__(model_path, **kwargs)
        self.policy_model_id = "v153"
        self.policy_model_label = "V153"
        # V153 was trained with simulation_base_mode=aligned_fixed.  Unlike
        # earlier floating-base profiles, obs[14:20] is exactly zero on both
        # hardware and the sim mirror; never reuse the XML warm-settle drift.
        self._base_obs_q = np.zeros(3, dtype=np.float32)
        self._base_obs_dq = np.zeros(3, dtype=np.float32)
        self.maximum_joint_feedback_age_s = self.MAXIMUM_JOINT_FEEDBACK_AGE_S
        self._gpu1_requires_measured_reset = True
        golden = self._policy_action(
            np.zeros(EXPECTED_OBS_DIM_WITH_DELAY, dtype=np.float32))
        if not np.allclose(
                golden, self.ZERO_OBSERVATION_ACTOR_MEAN,
                rtol=0.0, atol=2e-7):
            raise RuntimeError(
                "GPU1-QACC-FT V153 actor failed its zero-observation golden vector")

    @staticmethod
    def _sha256(path: str) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_model(self, model_path: str, device: str) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"GPU1-QACC-FT V153 checkpoint not found: {path}")
        actual = self._sha256(str(path))
        if actual != self.CHECKPOINT_SHA256:
            raise RuntimeError(
                "GPU1-QACC-FT V153 checkpoint SHA-256 mismatch: "
                f"{actual}")
        self._checkpoint_sha256 = actual
        super()._load_model(str(path), device)

    def _validate_checkpoint_contract(self, payload: dict) -> None:
        expected_payload = {
            "obs_dim": 67,
            "critic_obs_dim": 279,
            "act_dim": 7,
            "stage_name": "record_new3_sim2real_v153_b2_b3_energy_p025_60hz",
            "stage_index": 59,
            "stage_update": 441,
            "global_update": 1_092,
            "step": 14_841_511_936,
        }
        mismatches = [
            f"{key}={payload.get(key)!r} (expected {expected!r})"
            for key, expected in expected_payload.items()
            if payload.get(key) != expected
        ]
        args = payload.get("args")
        if not isinstance(args, dict):
            mismatches.append("args is missing")
        else:
            if args.get("curriculum_profile") != self.PROFILE_NAME:
                mismatches.append("args.curriculum_profile changed")
            if args.get("seed") != 82_731:
                mismatches.append("args.seed must be 82731")
        cfg = payload.get("env_cfg")
        if not isinstance(cfg, dict):
            mismatches.append("env_cfg is missing")
            cfg = {}
        for key, expected in self._REQUIRED_EXACT.items():
            actual = cfg.get(key)
            if key == "dr_ball_normalized_inertia_range":
                actual = tuple(actual or ())
            if actual != expected:
                mismatches.append(
                    f"{key}={cfg.get(key)!r} (expected {expected!r})")
        if tuple(cfg.get("dr_obs_latency_steps_range", ())) != (0, 2):
            mismatches.append("dr_obs_latency_steps_range must be (0, 2)")
        for key, expected in self._REQUIRED_VECTOR.items():
            try:
                actual = np.asarray(cfg.get(key), dtype=np.float64)
                valid = bool(
                    actual.shape == (7,)
                    and np.allclose(actual, expected, rtol=0.0, atol=1e-8))
            except (TypeError, ValueError):
                valid = False
            if not valid:
                mismatches.append(f"{key} does not match V153")
        try:
            ball_mass = np.asarray(
                cfg.get("dr_ball_mass_range"), dtype=np.float64)
            mass_valid = bool(
                ball_mass.shape == (2,)
                and np.allclose(
                    ball_mass, (0.0039, 0.0041), rtol=0.0, atol=1e-12))
        except (TypeError, ValueError):
            mass_valid = False
        if not mass_valid:
            mismatches.append("dr_ball_mass_range must be (0.0039, 0.0041)")
        if mismatches:
            raise RuntimeError(
                "GPU1-QACC-FT V153 checkpoint deployment contract mismatch: "
                + "; ".join(mismatches))

    def reset(self) -> None:
        super().reset()
        self._gpu1_requires_measured_reset = True

    def _seed_from_measurement(self, q: np.ndarray) -> None:
        self.arm_cmd_q = q.copy()
        self.arm_cmd_qvel = np.zeros(self.n_arm, dtype=np.float32)
        self.prev_action = np.zeros(self.n_arm, dtype=np.float32)
        self._cmd_history = [
            q.copy() for _ in range(self._command_buffer_len)]
        racket_pos, fk_ok = self._compute_racket_pose_base(q)
        if not fk_ok:
            raise RuntimeError(
                "GPU1-QACC-FT requires encoder FK for right_ee_site")
        self._prev_racket_pos_base = np.asarray(
            racket_pos, dtype=np.float32).reshape(3).copy()
        self._prev_racket_sample_pos_base = self._prev_racket_pos_base.copy()
        self._prev_racket_sample_stamp_s = None
        self._last_racket_sample_vel_base = np.zeros(3, dtype=np.float32)
        self._gpu1_requires_measured_reset = False

    def predict(
        self,
        ball_pos_m: np.ndarray,
        ball_vel_m_s: np.ndarray,
        *,
        ball_valid: bool,
        ball_obs_age_s: float = 0.0,
        dt: Optional[float] = None,
        arm_q: Optional[np.ndarray] = None,
        arm_dq: Optional[np.ndarray] = None,
        arm_sample_stamp_s: Optional[float] = None,
        observation_scales=None,
        confirmed_hit_count: int = 0,
    ) -> np.ndarray:
        step_dt = self.dt if dt is None else float(dt)
        if not np.isclose(step_dt, 0.005, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                "GPU1-QACC-FT requires an exact 0.005 s policy tick")
        if not bool(ball_valid):
            raise RuntimeError(
                "GPU1-QACC-FT refuses inference without a valid ball state")
        actual_age_s = float(ball_obs_age_s)
        if (not np.isfinite(actual_age_s) or actual_age_s < 0.0
                or actual_age_s > self.LOST_BALL_TIMEOUT_S):
            raise RuntimeError(
                "GPU1-QACC-FT represented ball age exceeds 350 ms")
        hit_count = int(confirmed_hit_count)
        if hit_count < 0:
            raise ValueError("confirmed_hit_count must be non-negative")
        if arm_q is None or arm_dq is None:
            raise RuntimeError(
                "GPU1-QACC-FT requires fresh measured q/dq feedback")
        q = np.asarray(arm_q, dtype=np.float32).reshape(self.n_arm)
        dq = np.asarray(arm_dq, dtype=np.float32).reshape(self.n_arm)
        if (not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq))
                or np.any(q < self.arm_q_low) or np.any(q > self.arm_q_high)):
            raise RuntimeError("GPU1-QACC-FT measured joint state is invalid")
        if observation_scales is not None:
            scales = np.asarray(observation_scales, dtype=np.float32).reshape(-1)
            if scales.size and (
                    scales.shape != (EXPECTED_OBS_DIM_WITH_DELAY,)
                    or not np.all(scales == np.float32(1.0))):
                raise RuntimeError(
                    "GPU1-QACC-FT forbids actor observation scaling")
        if self._gpu1_requires_measured_reset:
            self._seed_from_measurement(q)
        return super().predict(
            ball_pos_m,
            ball_vel_m_s,
            ball_valid=True,
            ball_obs_age_s=(actual_age_s if hit_count >= 1 else 0.0),
            dt=step_dt,
            arm_q=q,
            arm_dq=dq,
            arm_sample_stamp_s=arm_sample_stamp_s,
            observation_scales=observation_scales,
        )


class Gpu0QvelRealRmpV85PolicyController(MjxRLPolicyController):
    """Exact 57-D measured-feedback adapter for GPU0-QVEL/REAL-RMP.

    The returned position is the persistent bounded ``q_ref`` consumed by the
    robot's existing RMP.  This class deliberately contains no recovered-RMP,
    XML-PD, actuator, delay, compensation, planner, or publication-hold model.
    """

    CHECKPOINT_SHA256 = (
        "0381dc68a8ea02b2b3db171cc7edcc45291ff1b510d944b5ae2cddd89505c93a"
    )
    V91_CHECKPOINT_SHA256 = (
        "92b5d39592a81236ee5ee5806499a0a8d4b422f131b312ad87ea1efc90507c66"
    )
    V97_CHECKPOINT_SHA256 = (
        "f166a4b8adb6b5b6a846b14407baa5da32ba3535492d23cb0ad5978cece8568d"
    )
    PROFILE_NAME = "goal_d455_measured_qvel_rmp_vertical_v85"
    STAGE_NAME = "rmp85_complete_nonexecution_full_episode_polish"
    REFERENCE_ERROR_HORIZON_S = np.asarray(
        [0.018, 0.019, 0.019, 0.018, 0.018, 0.018, 0.018],
        dtype=np.float32,
    )
    JOINT_LOW_RAD = np.asarray(
        [-2.0944, -2.96706, -3.05433, -0.174533,
         -3.05433, -1.65806, -1.5708],
        dtype=np.float32,
    )
    JOINT_HIGH_RAD = np.asarray(
        [3.14159, 0.15708, 3.05433, 2.25147,
         3.05433, 1.65806, 1.5708],
        dtype=np.float32,
    )
    RESET_DEG = np.asarray(
        [5.736, -44.399, 30.683, 97.142, 49.323, -12.269, 14.214],
        dtype=np.float32,
    )
    ZERO_OBSERVATION_ACTOR_MEAN = np.asarray(
        [-0.013018085, 0.011029976, 0.019584768, 0.064141512,
         0.115628816, -0.122984409, 0.017757997],
        dtype=np.float32,
    )
    V91_ZERO_OBSERVATION_ACTOR_MEAN = np.asarray(
        [-0.01141354, 0.01237924, 0.01954993, 0.06344163,
         0.11612345, -0.12466098, 0.01851650],
        dtype=np.float32,
    )
    V97_ZERO_OBSERVATION_ACTOR_MEAN = np.asarray(
        [-0.013275788, 0.009566929, 0.019368205, 0.067597359,
         0.120253950, -0.124855369, 0.022645233],
        dtype=np.float32,
    )
    CHECKPOINT_CONTRACTS = {
        CHECKPOINT_SHA256: {
            "model_id": "v85",
            "label": "V85",
            "payload": {
                "obs_dim": EXPECTED_OBS_DIM_GPU0_REAL_RMP,
                "critic_obs_dim": 368,
                "act_dim": 7,
                "stage_name": STAGE_NAME,
                "stage_index": 24,
                "stage_update": 174,
                "global_update": 430,
                "step": 2_922_905_600,
            },
            "args": {
                "curriculum_profile": PROFILE_NAME,
                "seed": 20261004,
            },
            "zero_observation_actor_mean": ZERO_OBSERVATION_ACTOR_MEAN,
        },
        V91_CHECKPOINT_SHA256: {
            "model_id": "v91",
            "label": "V91",
            "payload": {
                "obs_dim": EXPECTED_OBS_DIM_GPU0_REAL_RMP,
                "critic_obs_dim": 368,
                "act_dim": 7,
                "step": 3_405_774_848,
            },
            "args": {
                "curriculum_profile": (
                    "goal_d455_measured_qvel_rmp_vertical_v91"),
                "seed": 20261018,
                "resume_start_stage": "rmp91_wide_xy_no_hold_margin",
            },
            "zero_observation_actor_mean": V91_ZERO_OBSERVATION_ACTOR_MEAN,
        },
        V97_CHECKPOINT_SHA256: {
            "model_id": "v97",
            "label": "V97",
            "payload": {
                "obs_dim": EXPECTED_OBS_DIM_GPU0_REAL_RMP,
                "critic_obs_dim": 368,
                "act_dim": 7,
                "stage_name": "rmp97_rmp_internal_commit_12p5",
                "stage_index": 32,
                "stage_update": 164,
                "global_update": 5_329,
                "step": 5_836_242_944,
            },
            "args": {
                "curriculum_profile": (
                    "goal_d455_measured_qvel_rmp_vertical_v97"),
                "seed": 20261018,
            },
            "zero_observation_actor_mean": V97_ZERO_OBSERVATION_ACTOR_MEAN,
        },
    }
    MAXIMUM_JOINT_FEEDBACK_AGE_S = 0.010
    LOST_BALL_TIMEOUT_S = 0.350

    _REQUIRED_EXACT = {
        "action_command_mode": "velocity",
        "action_velocity_scale": 1.0,
        "policy_integration_feedback_source": "measured",
        "recovered_rmp_motion_mode": True,
        "recovered_rmp_direct_qvel_target": True,
        "recovered_rmp_bounded_qvel_reference": True,
        "include_qvel_reference_error_obs": True,
        "enable_delay_conditioning": False,
        "high_latency_obs": False,
        "ball_obs_rate_hz": 90.0,
        "ball_obs_fractional_rate": True,
        "ball_obs_age_tracks_stale": True,
        "ball_obs_age_clip": 0.5,
        "ball_obs_velocity_observer_mode": "raw",
        "lost_ball_timeout_ms": 350.0,
        "actor_mask_previous_action": False,
        "actor_previous_action_scale": 1.0,
        "arm_action_limiter": True,
        "actuator_cmd_filter": False,
        "actuator_compensation_mode": "none",
        "arm_servo_target_tracking_planner": False,
        "right_arm_pd_profile": "recovered_rmp_rmpmd_v2",
        "recovered_rmp_period_s": 0.001,
        "frame_skip": 5,
        "action_filter_tau_ms": 0.0,
        "action_jerk_limit": 0.0,
        "enable_anti_windup": False,
        "dr_rmp_target_update_hold_probability": 0.08,
        "dr_rmp_target_update_hold_tail_probability": 0.022,
        "dr_rmp_target_update_hold_tail_steps_range": (2, 9),
    }

    def __init__(self, model_path: str, **kwargs) -> None:
        kwargs["enable_delay_conditioning"] = False
        kwargs["use_valid_ball_obs_age"] = True
        kwargs["observation_profile"] = (
            OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85)
        kwargs["expected_obs_dim"] = EXPECTED_OBS_DIM_GPU0_REAL_RMP
        kwargs["ball_obs_age_clip_s"] = 0.5
        super().__init__(model_path, **kwargs)
        self.arm_q_low = self.JOINT_LOW_RAD.copy()
        self.arm_q_high = self.JOINT_HIGH_RAD.copy()
        self.initial_arm_q_rad = np.deg2rad(self.RESET_DEG).astype(np.float32)
        self._base_obs_q = np.zeros(3, dtype=np.float32)
        self._base_obs_dq = np.zeros(3, dtype=np.float32)
        self.maximum_joint_feedback_age_s = (
            self.MAXIMUM_JOINT_FEEDBACK_AGE_S)
        self.reset()
        golden = self._policy_action(
            np.zeros(EXPECTED_OBS_DIM_GPU0_REAL_RMP, dtype=np.float32))
        expected_golden = np.asarray(
            self._checkpoint_contract["zero_observation_actor_mean"],
            dtype=np.float32,
        )
        if not np.allclose(
                golden, expected_golden,
                rtol=0.0, atol=2e-7):
            raise RuntimeError(
                "GPU0-QVEL/REAL-RMP actor failed its zero-observation golden vector")

    @staticmethod
    def _sha256(path: str) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_model(self, model_path: str, device: str) -> None:
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"GPU0-QVEL/REAL-RMP checkpoint not found: {path}")
        actual = self._sha256(str(path))
        contract = self.CHECKPOINT_CONTRACTS.get(actual)
        if contract is None:
            raise RuntimeError(
                "GPU0-QVEL/REAL-RMP checkpoint SHA-256 is not an approved "
                f"V85/V91/V97 model: {actual}")
        self._checkpoint_sha256 = actual
        self._checkpoint_contract = contract
        self.policy_model_id = str(contract["model_id"])
        self.policy_model_label = str(contract["label"])
        super()._load_model(str(path), device)

    def _validate_checkpoint_contract(self, payload: dict) -> None:
        expected_payload = dict(self._checkpoint_contract["payload"])
        mismatches = []
        for key, expected in expected_payload.items():
            if payload.get(key) != expected:
                mismatches.append(
                    f"{key}={payload.get(key)!r} (expected {expected!r})")
        args = payload.get("args")
        if not isinstance(args, dict):
            mismatches.append("args is missing")
        else:
            for key, expected in self._checkpoint_contract["args"].items():
                if args.get(key) != expected:
                    mismatches.append(
                        f"args.{key}={args.get(key)!r} (expected {expected!r})")
        cfg = payload.get("env_cfg")
        if not isinstance(cfg, dict):
            mismatches.append("env_cfg is missing")
            cfg = {}
        for key, expected in self._REQUIRED_EXACT.items():
            if cfg.get(key) != expected:
                mismatches.append(
                    f"{key}={cfg.get(key)!r} (expected {expected!r})")
        if float(cfg.get(
                "recovered_rmp_qvel_reference_error_horizon_s",
                float("nan"))) != 0.0:
            mismatches.append(
                "recovered_rmp_qvel_reference_error_horizon_s must be 0.0")
        vector_contracts = {
            "recovered_rmp_qvel_reference_error_horizon_s_per_joint": (
                self.REFERENCE_ERROR_HORIZON_S),
            "arm_vel_limit_deg_s": np.asarray(
                [210.0, 210.0, 240.0, 240.0, 300.0, 300.0, 300.0],
                dtype=np.float32),
            "arm_acc_limit_deg_s2": np.asarray(
                [1300.0, 1300.0, 1800.0, 3000.0, 3000.0, 3000.0, 3000.0],
                dtype=np.float32),
        }
        for key, expected in vector_contracts.items():
            try:
                actual = np.asarray(cfg.get(key, ()), dtype=np.float32)
                valid = actual.shape == (7,) and np.array_equal(actual, expected)
            except (TypeError, ValueError):
                valid = False
            if not valid:
                mismatches.append(
                    f"{key} does not match the {self.policy_model_label} contract")
        params = payload.get("params")
        actor = params.get("pi") if isinstance(params, dict) else None
        if (not isinstance(params, dict)
                or params.get("teacher_pi") is not None
                or not isinstance(actor, dict)):
            mismatches.append("checkpoint must contain one plain pi actor")
        else:
            shapes = {
                "l1": ((57, 256), (256,)),
                "l2": ((256, 256), (256,)),
                "out": ((256, 7), (7,)),
            }
            for name, (w_shape, b_shape) in shapes.items():
                layer = actor.get(name)
                if (not isinstance(layer, dict)
                        or np.asarray(layer.get("w")).shape != w_shape
                        or np.asarray(layer.get("b")).shape != b_shape):
                    mismatches.append(f"actor layer {name} shape changed")
        if mismatches:
            raise RuntimeError(
                "GPU0-QVEL/REAL-RMP checkpoint deployment contract mismatch: "
                + "; ".join(mismatches))

    def reset(self) -> None:
        super().reset()
        self._gpu0_requires_measured_reset = True
        self._gpu0_previous_measured_dq = np.zeros(
            self.n_arm, dtype=np.float32)

    def _seed_from_measurement(self, q: np.ndarray, dq: np.ndarray) -> None:
        self.arm_cmd_q = q.copy()
        self.arm_cmd_qvel = np.zeros(self.n_arm, dtype=np.float32)
        self.prev_action = np.zeros(self.n_arm, dtype=np.float32)
        self._gpu0_previous_measured_dq = dq.copy()
        racket_pos, fk_ok = self._compute_racket_pose_base(q)
        if not fk_ok:
            raise RuntimeError(
                "GPU0-QVEL/REAL-RMP requires encoder FK for right_ee_site")
        self._prev_racket_pos_base = np.asarray(
            racket_pos, dtype=np.float32).reshape(3).copy()
        self._prev_racket_sample_pos_base = self._prev_racket_pos_base.copy()
        self._prev_racket_sample_stamp_s = None
        self._last_racket_sample_vel_base = np.zeros(3, dtype=np.float32)
        self._gpu0_requires_measured_reset = False

    def _build_obs(
        self,
        ball_pos_m: np.ndarray,
        ball_vel_m_s: np.ndarray,
        ball_obs_age_s: float,
        step_dt: float,
        arm_q: Optional[np.ndarray] = None,
        arm_dq: Optional[np.ndarray] = None,
        arm_sample_stamp_s: Optional[float] = None,
    ) -> np.ndarray:
        if arm_q is None or arm_dq is None:
            raise RuntimeError(
                "GPU0-QVEL/REAL-RMP requires one atomic measured q/dq sample")
        q = np.asarray(arm_q, dtype=np.float32).reshape(self.n_arm)
        dq = np.asarray(arm_dq, dtype=np.float32).reshape(self.n_arm)
        ball_pos = np.asarray(ball_pos_m, dtype=np.float32).reshape(3)
        ball_vel = np.asarray(ball_vel_m_s, dtype=np.float32).reshape(3)
        if not all(np.all(np.isfinite(value)) for value in (
                q, dq, ball_pos, ball_vel)):
            raise RuntimeError(
                "GPU0-QVEL/REAL-RMP observation inputs must be finite")
        racket_pos, fk_ok = self._compute_racket_pose_base(q)
        if not fk_ok:
            raise RuntimeError(
                "GPU0-QVEL/REAL-RMP requires encoder FK for right_ee_site")
        racket_pos = np.asarray(racket_pos, dtype=np.float32).reshape(3)
        racket_vel = self._racket_velocity_observation(
            racket_pos,
            step_dt=step_dt,
            arm_sample_stamp_s=arm_sample_stamp_s,
        )
        age_norm = np.float32(np.clip(float(ball_obs_age_s) / 0.5, 0.0, 1.0))
        obs = np.concatenate((
            q,
            dq,
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            ball_pos,
            ball_vel,
            racket_pos,
            racket_vel,
            ball_pos - racket_pos,
            self.prev_action.astype(np.float32),
            dq - self._gpu0_previous_measured_dq,
            np.asarray([age_norm], dtype=np.float32),
            self.arm_cmd_q.astype(np.float32) - q,
        )).astype(np.float32)
        if obs.shape != (EXPECTED_OBS_DIM_GPU0_REAL_RMP,):
            raise RuntimeError("GPU0-QVEL/REAL-RMP observation is not 57-D")
        return obs

    def _racket_velocity_observation(
        self,
        racket_pos: np.ndarray,
        *,
        step_dt: float,
        arm_sample_stamp_s: Optional[float],
    ) -> np.ndarray:
        """Return the original fixed-policy-tick V85/V91/V97 observation."""
        del arm_sample_stamp_s
        racket_vel = (
            (racket_pos - self._prev_racket_pos_base)
            / np.float32(step_dt)
        ).astype(np.float32)
        self._prev_racket_pos_base = racket_pos.copy()
        return racket_vel

    def predict(
        self,
        ball_pos_m: np.ndarray,
        ball_vel_m_s: np.ndarray,
        *,
        ball_valid: bool,
        ball_obs_age_s: float = 0.0,
        dt: Optional[float] = None,
        arm_q: Optional[np.ndarray] = None,
        arm_dq: Optional[np.ndarray] = None,
        arm_sample_stamp_s: Optional[float] = None,
        observation_scales=None,
        confirmed_hit_count: int = 0,
    ) -> np.ndarray:
        del confirmed_hit_count
        step_dt = self.dt if dt is None else float(dt)
        if not np.isclose(step_dt, 0.005, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                "GPU0-QVEL/REAL-RMP requires an exact 0.005 s policy tick")
        if not bool(ball_valid):
            raise RuntimeError(
                "GPU0-QVEL/REAL-RMP refuses inference without a valid ball state")
        age_s = float(ball_obs_age_s)
        if (not np.isfinite(age_s) or age_s < 0.0
                or age_s > self.LOST_BALL_TIMEOUT_S):
            raise RuntimeError(
                "GPU0-QVEL/REAL-RMP represented ball age exceeds 350 ms")
        if arm_q is None or arm_dq is None:
            raise RuntimeError(
                "GPU0-QVEL/REAL-RMP requires fresh measured q/dq feedback")
        q = np.asarray(arm_q, dtype=np.float32).reshape(self.n_arm)
        dq = np.asarray(arm_dq, dtype=np.float32).reshape(self.n_arm)
        if (not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq))
                or np.any(q < self.arm_q_low) or np.any(q > self.arm_q_high)):
            raise RuntimeError(
                "GPU0-QVEL/REAL-RMP measured joint state is invalid")
        if observation_scales is not None:
            scales = np.asarray(observation_scales, dtype=np.float32).reshape(-1)
            if scales.size and (
                    scales.shape != (EXPECTED_OBS_DIM_GPU0_REAL_RMP,)
                    or not np.all(scales == np.float32(1.0))):
                raise RuntimeError(
                    "GPU0-QVEL/REAL-RMP forbids actor observation scaling")
        if self._gpu0_requires_measured_reset:
            self._seed_from_measurement(q, dq)
        obs = self._build_obs(
            ball_pos_m,
            ball_vel_m_s,
            age_s,
            step_dt,
            arm_q=q,
            arm_dq=dq,
            arm_sample_stamp_s=arm_sample_stamp_s,
        )
        raw_action = self._policy_action(obs)
        if not np.all(np.isfinite(raw_action)):
            raise RuntimeError(
                "GPU0-QVEL/REAL-RMP actor returned a non-finite action")
        action = np.clip(raw_action, -1.0, 1.0).astype(np.float32)
        desired_dq = (action * self.arm_vel_limit_rad_s).astype(np.float32)
        integrated_q_ref = (
            self.arm_cmd_q + desired_dq * np.float32(0.005))
        maximum_error = (
            self.arm_vel_limit_rad_s * self.REFERENCE_ERROR_HORIZON_S)
        lower = np.maximum(self.arm_q_low, q - maximum_error)
        upper = np.minimum(self.arm_q_high, q + maximum_error)
        q_ref = np.clip(integrated_q_ref, lower, upper).astype(np.float32)
        previous_desired_dq = self.arm_cmd_qvel.copy()
        self.arm_cmd_q = q_ref.copy()
        self.arm_cmd_qvel = desired_dq.copy()
        self.prev_action = action.copy()
        self._gpu0_previous_measured_dq = dq.copy()
        self.last_obs = obs.copy()
        self.last_obs_unscaled = obs.copy()
        self.last_observation_scales = np.ones(
            EXPECTED_OBS_DIM_GPU0_REAL_RMP, dtype=np.float32)
        self.last_observation_scale_indices = ()
        self.last_action = np.asarray(raw_action, dtype=np.float32).copy()
        self.last_executed_action = action.copy()
        self.last_cmd_q = q_ref.copy()
        self.last_cmd_qvel = desired_dq.copy()
        self.last_cmd_qacc = (
            (desired_dq - previous_desired_dq) / np.float32(0.005)
        ).astype(np.float32)
        return q_ref.copy()


class Gpu0QvelRealRmpV97SourceClockPolicyController(
        Gpu0QvelRealRmpV85PolicyController):
    """V97 deployment adapter with source-clock racket velocity.

    This is deliberately a separate runtime identity from the authenticated
    V97 checkpoint adapter above.  The actor, action integration, limits, and
    every observation slice except racket velocity ``obs[29:32]`` remain
    unchanged.  Hardware JointState samples may repeat across policy ticks or
    skip source frames, so the derivative is evaluated only on coherent new
    source samples and is held across repeats.
    """

    POLICY_MODEL_ID = "v97_source_clock"
    POLICY_MODEL_LABEL = "V97 source-clock v1"
    RACKET_VELOCITY_OBSERVATION_CONTRACT = "joint_state_source_clock_v1"
    MINIMUM_SOURCE_DT_S = 1e-4
    MAXIMUM_SOURCE_DT_S = 0.10

    def __init__(self, model_path: str, **kwargs) -> None:
        super().__init__(model_path, **kwargs)
        if self._checkpoint_sha256 != self.V97_CHECKPOINT_SHA256:
            raise RuntimeError(
                "GPU0 V97 source-clock adapter requires the approved V97 "
                "checkpoint")
        self.checkpoint_policy_model_id = self.policy_model_id
        self.policy_model_id = self.POLICY_MODEL_ID
        self.policy_model_label = self.POLICY_MODEL_LABEL
        self.racket_velocity_observation_contract = (
            self.RACKET_VELOCITY_OBSERVATION_CONTRACT)

    def _racket_velocity_observation(
        self,
        racket_pos: np.ndarray,
        *,
        step_dt: float,
        arm_sample_stamp_s: Optional[float],
    ) -> np.ndarray:
        del step_dt
        current_pos = np.asarray(racket_pos, dtype=np.float32).reshape(3)
        sample_stamp = None
        if arm_sample_stamp_s is not None:
            try:
                candidate_stamp = float(arm_sample_stamp_s)
            except (TypeError, ValueError):
                candidate_stamp = float("nan")
            if np.isfinite(candidate_stamp):
                sample_stamp = candidate_stamp

        previous_pos = self._prev_racket_sample_pos_base
        previous_stamp = self._prev_racket_sample_stamp_s
        if sample_stamp is None:
            # Source time is part of this adapter's contract.  Never fall back
            # to the 5 ms policy clock because that recreates the original
            # zero/spike defect.  Re-anchor and wait for a causal source pair.
            racket_vel = np.zeros(3, dtype=np.float32)
            self._prev_racket_sample_pos_base = current_pos.copy()
            self._prev_racket_sample_stamp_s = None
        elif previous_pos is None or previous_stamp is None:
            racket_vel = np.zeros(3, dtype=np.float32)
            self._prev_racket_sample_pos_base = current_pos.copy()
            self._prev_racket_sample_stamp_s = sample_stamp
        elif (sample_stamp == previous_stamp
              and np.array_equal(current_pos, previous_pos)):
            # The policy timer observed the same immutable JointState again.
            racket_vel = self._last_racket_sample_vel_base.copy()
        elif sample_stamp == previous_stamp:
            # A timestamp reused with different q/FK is not a coherent repeat.
            # Re-anchor position without carrying it into the next derivative.
            racket_vel = np.zeros(3, dtype=np.float32)
            self._prev_racket_sample_pos_base = current_pos.copy()
        else:
            source_dt = sample_stamp - previous_stamp
            if self.MINIMUM_SOURCE_DT_S <= source_dt <= self.MAXIMUM_SOURCE_DT_S:
                candidate_vel = (
                    (current_pos - previous_pos) / np.float32(source_dt)
                ).astype(np.float32)
                racket_vel = (
                    candidate_vel
                    if np.all(np.isfinite(candidate_vel))
                    else np.zeros(3, dtype=np.float32)
                )
            else:
                # Non-monotonic or stale source time cannot define a causal
                # derivative.  This sample becomes the new anchor.
                racket_vel = np.zeros(3, dtype=np.float32)
            self._prev_racket_sample_pos_base = current_pos.copy()
            self._prev_racket_sample_stamp_s = sample_stamp

        self._last_racket_sample_vel_base = racket_vel.copy()
        # Keep the legacy policy-clock anchor coherent for introspection and
        # for an explicit switch back to the original adapter.
        self._prev_racket_pos_base = current_pos.copy()
        return racket_vel
