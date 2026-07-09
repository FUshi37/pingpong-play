#!/usr/bin/env python3
"""Online actuator-delay compensation for right-arm trajectory replay.

This wraps the reference NumPy control law in :mod:`delay_control` (vendored
verbatim from ``pingpong_controller/tools/rl_sim/delay_control.py``) so the
replay node can pre-distort every nominal joint command *before* publishing it
to the real robot, exactly mirroring what the simulation policy's actuator
compensator did during training.

Why this is needed
------------------
In simulation the policy integrates a nominal joint command
``(q_cmd, qdot_cmd, qdd_cmd)`` and then ``compensate_q_ref`` turns it into a
pre-distorted command ``q_ref_latest`` that *leads* the nominal motion.  In the
sim, ``q_ref_latest`` is pushed into a command buffer; a pure transport delay
plus a first-order actuator filter then "digest" it, producing the joint
position the robot actually reaches.  Training the policy *with* this
compensator means the real robot's identical delay + first-order lag should be
cancelled if we feed it the same pre-distorted command.

On hardware the delay and lag are physically real, so we do NOT re-simulate
them.  Each control tick we:

  1. take the nominal ``(q_cmd, qdot_cmd, qdd_cmd)`` for this step,
  2. read the real joint feedback ``applied_q`` from ``/joint_states``,
  3. run the same ``compensate_q_ref`` (inverse_mpc / inverse_smith / lead) to
     get the pre-distorted ``q_ref_latest``,
  4. publish ``q_ref_latest`` and let the real actuator's physical delay + lag
     turn it back into the nominal motion.

The command buffer is still maintained internally because the inverse-Smith /
inverse-MPC predictors roll a first-order filter forward over the queued
commands (``predict_filtered_command_before_delay``) starting from the measured
``applied_q`` -- i.e. the compensation is closed-loop on the real tracking
error, which is the whole point.

Defaults below are the values the shipped ``inverse_mpc`` checkpoint was trained
with (extracted from ``18_stage4g_strong_contact_dr_inverse_mpc.pkl``'s
``env_cfg``): delay 72 ms, actuator filter tau 0.074 s, gain 1.0, buffer extra
4 steps, mpc beta 1.2 / delay_scale 1.05 / tau_scale 0.75 / horizon 6 /
weights (1.0, 0.25, 0.05), max delta 30 deg.
"""

from typing import Optional

import numpy as np

from .delay_control import (
    command_buffer_length,
    compensate_q_ref,
    delay_steps_from_tau,
    push_command_buffer,
)

# Modes that mean "do nothing" (publish the nominal command unchanged).
_OFF_MODES = {"none", "off", "false", "0", ""}
VALID_COMPENSATION_MODES = ("none", "lead", "inverse_smith", "inverse_mpc")


class CommandCompensator:
    """Maintain the command buffer and pre-distort each nominal joint command.

    One instance is created per replay run. The caller feeds the nominal
    ``(q_cmd, qdot_cmd, qdd_cmd)`` for the current step together with the latest
    measured joint position ``applied_q``; :meth:`step` returns the compensated
    command to publish and advances the internal command buffer exactly as
    ``MJXPolicyController.predict`` does on every tick.

    All defaults match the shipped ``inverse_mpc`` checkpoint's ``env_cfg``.
    Override any of them through the replay node's ROS parameters.
    """

    def __init__(
        self,
        *,
        n_joints: int,
        dt: float,
        mode: str = "none",
        # Plant model (the real actuator's delay + first-order lag).
        delay_ms: float = 72.0,
        actuator_tau_s: float = 0.074,
        actuator_gain: float = 1.0,
        command_buffer_extra_steps: int = 4,
        # When False (delay_conditioning off in sim) the active command is the
        # latest compensated command rather than the delay-selected one. The
        # shipped checkpoint trained with delay conditioning ON, so the default
        # is True; it only affects what `current_active_q()` reports, never the
        # published command (we always publish q_ref_latest -- see step()).
        delay_conditioning: bool = True,
        # lead-compensation gains.
        lead_beta: float = 0.0,
        lead_delay_scale: float = 1.0,
        lead_tau_scale: float = 1.0,
        lead_max_delta_rad: float = 0.0,
        # inverse-Smith gains.
        inverse_beta: float = 1.0,
        inverse_delay_scale: float = 1.0,
        inverse_tau_scale: float = 1.0,
        inverse_max_delta_rad: float = 0.0,
        # inverse-MPC gains.
        mpc_beta: float = 1.2,
        mpc_delay_scale: float = 1.05,
        mpc_tau_scale: float = 0.75,
        mpc_horizon_steps: int = 6,
        mpc_tracking_weight: float = 1.0,
        mpc_nominal_weight: float = 0.25,
        mpc_delta_weight: float = 0.05,
        mpc_max_delta_rad: float = 0.5235987755982988,  # 30 deg
        # Position clip applied to the compensated command (robot jnt_range).
        q_low: Optional[np.ndarray] = None,
        q_high: Optional[np.ndarray] = None,
        # Warm/base pose used by the inverse predictors (sim used the training
        # target posture). Defaults to the first commanded pose if not given.
        warm_q: Optional[np.ndarray] = None,
    ) -> None:
        self.n_joints = int(n_joints)
        self.dt = float(dt)
        if not np.isfinite(self.dt) or self.dt <= 0.0:
            raise ValueError(f"dt must be positive and finite, got {dt}")

        self.mode = str(mode or "none").strip().lower().replace("-", "_")
        if self.mode in _OFF_MODES:
            self.mode = "none"
        if self.mode not in (
            "none", "lead", "inverse_smith", "smith", "inverse",
            "inverse_mpc", "regularized_inverse_mpc", "mpc",
        ):
            raise ValueError(
                f"compensation_mode must be one of {VALID_COMPENSATION_MODES}, "
                f"got '{mode}'")

        self.delay_ms = max(0.0, float(delay_ms))
        self.actuator_tau_s = max(0.0, float(actuator_tau_s))
        self.actuator_gain = float(actuator_gain)
        self.delay_conditioning = bool(delay_conditioning)

        self.lead_beta = float(lead_beta)
        self.lead_delay_scale = float(lead_delay_scale)
        self.lead_tau_scale = float(lead_tau_scale)
        self.lead_max_delta_rad = float(lead_max_delta_rad)

        self.inverse_beta = float(inverse_beta)
        self.inverse_delay_scale = float(inverse_delay_scale)
        self.inverse_tau_scale = float(inverse_tau_scale)
        self.inverse_max_delta_rad = float(inverse_max_delta_rad)

        self.mpc_beta = float(mpc_beta)
        self.mpc_delay_scale = float(mpc_delay_scale)
        self.mpc_tau_scale = float(mpc_tau_scale)
        self.mpc_horizon_steps = int(mpc_horizon_steps)
        self.mpc_tracking_weight = float(mpc_tracking_weight)
        self.mpc_nominal_weight = float(mpc_nominal_weight)
        self.mpc_delta_weight = float(mpc_delta_weight)
        self.mpc_max_delta_rad = float(mpc_max_delta_rad)

        self.q_low = None if q_low is None else np.asarray(q_low, dtype=np.float32)
        self.q_high = None if q_high is None else np.asarray(q_high, dtype=np.float32)
        self._warm_q = None if warm_q is None else np.asarray(warm_q, dtype=np.float32)

        # Command-buffer geometry mirrors MJXPolicyController.__init__.
        self.command_buffer_len = command_buffer_length(
            self.delay_ms, self.dt, int(command_buffer_extra_steps))
        self.delay_steps = int(np.clip(
            delay_steps_from_tau(self.delay_ms * 1e-3, self.dt),
            0, self.command_buffer_len - 1))

        self.command_buffer: Optional[np.ndarray] = None
        self._initialized = False

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def reset(self, q_init: np.ndarray) -> None:
        """Seed the command buffer with the starting pose (sim: reset()).

        Call once with the real arm pose just before replay begins so the
        first compensated commands are computed against a consistent history
        instead of zeros.
        """
        q0 = np.asarray(q_init, dtype=np.float32).reshape(-1)
        if q0.shape != (self.n_joints,) or not np.all(np.isfinite(q0)):
            raise ValueError(f"q_init must be a finite length-{self.n_joints} array")
        self.command_buffer = np.broadcast_to(
            q0, (self.command_buffer_len, self.n_joints)).copy()
        if self._warm_q is None:
            self._warm_q = q0.copy()
        self._initialized = True

    def step(
        self,
        q_cmd: np.ndarray,
        qdot_cmd: np.ndarray,
        qdd_cmd: np.ndarray,
        applied_q: np.ndarray,
    ) -> np.ndarray:
        """Return the compensated joint command to publish for this tick.

        ``q_cmd`` / ``qdot_cmd`` / ``qdd_cmd`` are the nominal command state for
        the step (from the recorded trajectory); ``applied_q`` is the latest
        measured right-arm position. Mirrors the compensate->push sequence in
        ``MJXPolicyController.predict`` (lines 942-982). When ``mode == "none"``
        this returns ``q_cmd`` unchanged (only clipping to limits).
        """
        q = np.asarray(q_cmd, dtype=np.float32).reshape(-1)
        if not self._initialized:
            # Be forgiving: auto-seed from the first nominal command.
            self.reset(q)

        if self.mode == "none":
            return self._clip(q)

        qdot = np.asarray(qdot_cmd, dtype=np.float32).reshape(-1)
        qdd = np.asarray(qdd_cmd, dtype=np.float32).reshape(-1)
        y_applied = np.asarray(applied_q, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(y_applied)) or y_applied.shape != (self.n_joints,):
            # Without valid feedback the inverse predictors are meaningless;
            # fall back to the last compensated command in the buffer.
            y_applied = self.command_buffer[-1].copy()

        q_ref_latest = compensate_q_ref(
            self.mode,
            q,
            qdot,
            qdd,
            dt=self.dt,
            delay_steps=self.delay_steps,
            actuator_tau_s=self.actuator_tau_s,
            actuator_gain=self.actuator_gain,
            lead_beta=self.lead_beta,
            lead_delay_scale=self.lead_delay_scale,
            lead_tau_scale=self.lead_tau_scale,
            lead_max_delta_rad=self.lead_max_delta_rad,
            inverse_beta=self.inverse_beta,
            inverse_delay_scale=self.inverse_delay_scale,
            inverse_tau_scale=self.inverse_tau_scale,
            inverse_max_delta_rad=self.inverse_max_delta_rad,
            mpc_beta=self.mpc_beta,
            mpc_delay_scale=self.mpc_delay_scale,
            mpc_tau_scale=self.mpc_tau_scale,
            mpc_horizon_steps=self.mpc_horizon_steps,
            mpc_tracking_weight=self.mpc_tracking_weight,
            mpc_nominal_weight=self.mpc_nominal_weight,
            mpc_delta_weight=self.mpc_delta_weight,
            mpc_max_delta_rad=self.mpc_max_delta_rad,
            applied_q=y_applied,
            command_buffer=self.command_buffer,
            warm_q=self._warm_q,
            q_low=self.q_low,
            q_high=self.q_high,
        )

        # Advance the buffer just like predict(): the delay-selected command
        # becomes "active" when delay conditioning is on, else the latest one.
        if self.delay_conditioning:
            self.command_buffer, _active = push_command_buffer(
                self.command_buffer, q_ref_latest, self.delay_steps)
        else:
            self.command_buffer, _active = push_command_buffer(
                self.command_buffer, q_ref_latest, 0)

        # On the real robot we publish the pre-distorted *latest* command and
        # let the physical delay+lag realize the nominal motion -- NOT _active
        # (that is the sim's model of what the plant would output).
        return np.asarray(q_ref_latest, dtype=np.float32)

    def _clip(self, q: np.ndarray) -> np.ndarray:
        if self.q_low is None and self.q_high is None:
            return q
        lo = -np.inf if self.q_low is None else self.q_low
        hi = np.inf if self.q_high is None else self.q_high
        return np.clip(q, lo, hi).astype(np.float32)

    def describe(self) -> str:
        """One-line summary for logging at startup."""
        if self.mode == "none":
            return "compensation=none (publishing nominal commands)"
        return (
            f"compensation={self.mode} delay={self.delay_ms:.1f}ms "
            f"({self.delay_steps} steps) tau={self.actuator_tau_s*1e3:.1f}ms "
            f"gain={self.actuator_gain:.3g} buf_len={self.command_buffer_len} "
            f"mpc(beta={self.mpc_beta:.3g},h={self.mpc_horizon_steps},"
            f"delay_scale={self.mpc_delay_scale:.3g},tau_scale={self.mpc_tau_scale:.3g},"
            f"max_delta={np.rad2deg(self.mpc_max_delta_rad):.1f}deg)")
