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

import pickle
from typing import Optional

import numpy as np

from pingpong_controller.rl_policy import (
    ACTION_COMMAND_MODE_VELOCITY,
    DEFAULT_ACTION_ACC_SCALE,
    DEFAULT_ARM_ACC_LIMIT_DEG_S2,
    DEFAULT_ARM_VEL_LIMIT_DEG_S,
    DEFAULT_BALL_OBS_AGE_CLIP_S,
    EXPECTED_OBS_DIM,
    EXPECTED_OBS_DIM_WITH_DELAY,
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
        self._logger.info(f"Loading MJX/JAX PPO checkpoint from: {model_path}")
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
            "MJX checkpoint env_cfg applied: "
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
