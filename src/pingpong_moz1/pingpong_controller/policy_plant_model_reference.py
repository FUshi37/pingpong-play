"""Bounded model-reference shadow/apply candidate for policy joint commands.

The target plant is reconstructed from checkpoint ``env_cfg``.  The real
plant is a versioned, explicitly imperfect candidate fit.  This module never
sleeps or adds a publish-side delay: delay exists only inside the predictors.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from pingpong_controller.policy_joint_sample import PolicyJointSample


class PlantMatcherMode(str, Enum):
    OFF = "off"
    SHADOW = "shadow"
    APPLY = "apply"

    @classmethod
    def parse(cls, value: object) -> "PlantMatcherMode":
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError(
                "POLICY_PLANT_MATCHER_MODE must be off, shadow, or apply") from exc


class PlantMatcherFailClosed(RuntimeError):
    """The only valid response is a measured-pose hold."""


def _vector(value: object, name: str, *, positive: bool = False) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64).reshape(-1)
    if out.shape != (7,):
        raise ValueError(f"{name} must contain seven values, got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} contains non-finite values")
    if positive and np.any(out <= 0.0):
        raise ValueError(f"{name} must be positive")
    return out.copy()


@dataclass(frozen=True)
class JointPlantParameters:
    natural_frequency_rad_s: np.ndarray
    damping_ratio: np.ndarray
    gain_per_joint: np.ndarray
    delay_ms_per_joint: np.ndarray

    def __post_init__(self) -> None:
        wn = _vector(
            self.natural_frequency_rad_s, "natural_frequency_rad_s",
            positive=True)
        zeta = _vector(self.damping_ratio, "damping_ratio", positive=True)
        gain = _vector(self.gain_per_joint, "gain_per_joint", positive=True)
        delay = _vector(self.delay_ms_per_joint, "delay_ms_per_joint")
        if np.any(zeta >= 1.0):
            raise ValueError("damping_ratio must be below one")
        if np.any(delay < 0.0):
            raise ValueError("delay_ms_per_joint must be non-negative")
        for value in (wn, zeta, gain, delay):
            value.setflags(write=False)
        object.__setattr__(self, "natural_frequency_rad_s", wn)
        object.__setattr__(self, "damping_ratio", zeta)
        object.__setattr__(self, "gain_per_joint", gain)
        object.__setattr__(self, "delay_ms_per_joint", delay)

    @classmethod
    def from_checkpoint_env_cfg(
        cls, env_cfg: Mapping[str, Any],
    ) -> "JointPlantParameters":
        model = str(env_cfg.get(
            "actuator_cmd_model", "")).strip().lower().replace("-", "_")
        if model != "second_order":
            raise ValueError(
                "checkpoint target actuator_cmd_model must be second_order")
        return cls(
            natural_frequency_rad_s=env_cfg.get(
                "actuator_cmd_natural_frequency_rad_s", ()),
            damping_ratio=env_cfg.get("actuator_cmd_damping_ratio", ()),
            gain_per_joint=env_cfg.get("actuator_cmd_gain_per_joint", ()),
            delay_ms_per_joint=env_cfg.get(
                "actuator_cmd_delay_ms_per_joint", ()),
        )

    @classmethod
    def from_real_fit(cls, payload: Mapping[str, Any]) -> "JointPlantParameters":
        return cls(
            natural_frequency_rad_s=payload.get("natural_frequency_rad_s", ()),
            damping_ratio=payload.get("damping_ratio", ()),
            gain_per_joint=payload.get("gain_per_joint", ()),
            delay_ms_per_joint=payload.get("delay_ms_per_joint", ()),
        )


class DelayedSecondOrderJointPlant:
    """Exact under-damped ZOH transition used by the frozen training mirror."""

    def __init__(
        self,
        parameters: JointPlantParameters,
        dt_s: float,
        warm_q: np.ndarray,
    ) -> None:
        self.parameters = parameters
        self.dt_s = float(dt_s)
        if not math.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")
        self.warm_q = _vector(warm_q, "warm_q")
        # np.rint intentionally matches the training/mirror delay rounding.
        self.delay_steps = np.rint(
            parameters.delay_ms_per_joint * 1e-3 / self.dt_s,
        ).astype(np.int64)
        self.buffer_length = int(np.max(self.delay_steps)) + 1
        wn = parameters.natural_frequency_rad_s
        zeta = parameters.damping_ratio
        root = np.sqrt(np.maximum(1.0 - zeta * zeta, 1e-8))
        wd = wn * root
        decay = np.exp(-zeta * wn * self.dt_s)
        phase = wd * self.dt_s
        sin_phase = np.sin(phase)
        cos_phase = np.cos(phase)
        self.a11 = decay * (cos_phase + zeta / root * sin_phase)
        self.a12 = decay * sin_phase / np.maximum(wd, 1e-8)
        self.a21 = -decay * wn * wn * sin_phase / np.maximum(wd, 1e-8)
        self.a22 = decay * (cos_phase - zeta / root * sin_phase)
        self.q = self.warm_q.copy()
        self.dq = np.zeros(7, dtype=np.float64)
        self.command_buffer = np.repeat(
            self.warm_q[None, :], self.buffer_length, axis=0)

    def clone(self) -> "DelayedSecondOrderJointPlant":
        return copy.deepcopy(self)

    def reset(self, q: np.ndarray, dq: np.ndarray | None = None) -> None:
        self.q = _vector(q, "reset q")
        self.dq = (
            np.zeros(7, dtype=np.float64)
            if dq is None else _vector(dq, "reset dq"))
        self.command_buffer = np.repeat(
            self.q[None, :], self.buffer_length, axis=0)

    def reconcile_measured(self, q: np.ndarray, dq: np.ndarray) -> None:
        self.q = _vector(q, "measured q")
        self.dq = _vector(dq, "measured dq")

    def step(self, command_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        command = _vector(command_q, "command_q")
        self.command_buffer = np.concatenate(
            (self.command_buffer[1:], command[None, :]), axis=0)
        indices = self.buffer_length - 1 - np.clip(
            self.delay_steps, 0, self.buffer_length - 1)
        active = self.command_buffer[indices, np.arange(7)]
        target = self.warm_q + self.parameters.gain_per_joint * (
            active - self.warm_q)
        current_q = self.q.copy()
        current_dq = self.dq.copy()
        self.q = (
            self.a11 * current_q + self.a12 * current_dq
            + (1.0 - self.a11) * target)
        self.dq = (
            self.a21 * current_q + self.a22 * current_dq
            - self.a21 * target)
        if not (np.all(np.isfinite(self.q)) and np.all(np.isfinite(self.dq))):
            raise PlantMatcherFailClosed("plant prediction became non-finite")
        return self.q.copy(), self.dq.copy()


@dataclass(frozen=True)
class PlantMatcherSolverConfig:
    horizon_steps: int
    q_tracking_weight: float
    dq_tracking_weight: float
    command_deviation_weight: float
    command_slew_weight: float
    max_correction_rad: np.ndarray
    max_correction_rate_rad_s: np.ndarray
    residual_uncertainty_scale: float

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "PlantMatcherSolverConfig":
        return cls(
            horizon_steps=max(1, int(payload.get("horizon_steps", 16))),
            q_tracking_weight=max(0.0, float(payload.get("q_tracking_weight", 1.0))),
            dq_tracking_weight=max(0.0, float(payload.get("dq_tracking_weight", 0.04))),
            command_deviation_weight=max(
                0.0, float(payload.get("command_deviation_weight", 0.2))),
            command_slew_weight=max(
                0.0, float(payload.get("command_slew_weight", 0.1))),
            max_correction_rad=_vector(
                payload.get("max_correction_rad", ()), "max_correction_rad",
                positive=True),
            max_correction_rate_rad_s=_vector(
                payload.get("max_correction_rate_rad_s", ()),
                "max_correction_rate_rad_s", positive=True),
            residual_uncertainty_scale=max(
                1.0, float(payload.get("residual_uncertainty_scale", 1.0))),
        )


def load_real_plant_fit(path: str | Path) -> tuple[dict[str, Any], JointPlantParameters, PlantMatcherSolverConfig]:
    fit_path = Path(path).expanduser().resolve()
    if not fit_path.is_file():
        raise ValueError(f"real plant fit not found: {fit_path}")
    payload = json.loads(fit_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("real plant fit schema_version must be 1")
    for key in ("fit_id", "fit_version", "source_recordings",
                "source_recording_sha256", "holdout_metrics",
                "provenance_status", "apply_eligible"):
        if key not in payload:
            raise ValueError(f"real plant fit missing provenance field {key!r}")
    if not isinstance(payload["source_recordings"], list):
        raise ValueError("source_recordings must be a list")
    hashes = payload.get("source_recording_sha256")
    if hashes is not None and not isinstance(hashes, dict):
        raise ValueError("source_recording_sha256 must be an object or null")
    hashes_verified = False
    if isinstance(hashes, dict) and hashes:
        for label, digest in hashes.items():
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(char not in "0123456789abcdef"
                           for char in digest.lower())):
                raise ValueError(
                    f"invalid source recording SHA256 for {label!r}")
        root_text = payload.get("source_recording_root")
        if root_text:
            # Config lives at <repo>/config/*.json.
            root = fit_path.parent.parent / str(root_text)
            present = True
            for relative, expected in hashes.items():
                source = (root / relative).resolve()
                if not source.is_file():
                    present = False
                    continue
                digest = hashlib.sha256()
                with source.open("rb") as handle:
                    for block in iter(
                            lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                actual = digest.hexdigest()
                if actual != expected:
                    raise ValueError(
                        f"source recording SHA256 mismatch for {source}: "
                        f"expected={expected}, actual={actual}")
            hashes_verified = present
    payload = dict(payload)
    payload["source_hashes_verified"] = hashes_verified
    return (
        payload,
        JointPlantParameters.from_real_fit(payload),
        PlantMatcherSolverConfig.from_mapping(payload.get("solver", {})),
    )


@dataclass(frozen=True)
class PlantMatcherProposal:
    policy_qcmd: np.ndarray
    matcher_shadow_q: np.ndarray
    selected_pre_safety_q: np.ndarray
    target_q: np.ndarray
    target_dq: np.ndarray
    diagnostics: Mapping[str, Any]


class PolicyPlantModelReference:
    def __init__(
        self,
        *,
        target_parameters: JointPlantParameters,
        real_parameters: JointPlantParameters,
        solver: PlantMatcherSolverConfig,
        mode: str,
        dt_s: float,
        warm_q: np.ndarray,
        q_low: np.ndarray,
        q_high: np.ndarray,
        dq_limit: np.ndarray,
        allow_apply: bool,
        release_hash_match: bool,
        real_fit_metadata: Mapping[str, Any],
    ) -> None:
        self.mode = PlantMatcherMode.parse(mode)
        self.dt_s = float(dt_s)
        self.solver = solver
        self.q_low = _vector(q_low, "q_low")
        self.q_high = _vector(q_high, "q_high")
        self.dq_limit = _vector(dq_limit, "dq_limit", positive=True)
        self.allow_apply = bool(allow_apply)
        self.release_hash_match = bool(release_hash_match)
        self.real_fit_metadata = dict(real_fit_metadata)
        self.target = DelayedSecondOrderJointPlant(
            target_parameters, self.dt_s, warm_q)
        self.real = DelayedSecondOrderJointPlant(
            real_parameters, self.dt_s, warm_q)
        self.initialized = False
        self.previous_sample: PolicyJointSample | None = None
        self.previous_published_q = _vector(warm_q, "warm_q")
        self.previous_correction = np.zeros(7, dtype=np.float64)
        self.last_proposal: PlantMatcherProposal | None = None

    @property
    def apply_authorized(self) -> bool:
        source_hash = self.real_fit_metadata.get("source_recording_sha256")
        digests = (
            list(source_hash.values())
            if isinstance(source_hash, dict) else [source_hash])
        provenance_verified = bool(digests) and all(
            isinstance(digest, str)
            and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest.lower())
            for digest in digests)
        return bool(
            self.allow_apply
            and self.release_hash_match
            and self.real_fit_metadata.get("apply_eligible", False)
            and self.real_fit_metadata.get("source_hashes_verified", False)
            and provenance_verified)

    def reset(self, sample: PolicyJointSample) -> None:
        self.target.reset(sample.q_raw, sample.dq_raw)
        self.real.reset(sample.q_raw, sample.dq_raw)
        self.previous_published_q = sample.q_raw.copy()
        self.previous_correction = np.zeros(7, dtype=np.float64)
        self.previous_sample = None
        self.initialized = True

    def _target_future(self, policy_qcmd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        plant = self.target.clone()
        qs = []
        dqs = []
        for _ in range(self.solver.horizon_steps):
            q, dq = plant.step(policy_qcmd)  # hold-last; no future actor lookahead
            qs.append(q)
            dqs.append(dq)
        return np.asarray(qs), np.asarray(dqs)

    def _real_future(self, candidate_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        plant = self.real.clone()
        qs = []
        dqs = []
        for _ in range(self.solver.horizon_steps):
            q, dq = plant.step(candidate_q)
            qs.append(q)
            dqs.append(dq)
        return np.asarray(qs), np.asarray(dqs)

    def _solve(self, policy_qcmd: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        target_q, target_dq = self._target_future(policy_qcmd)
        base_q, base_dq = self._real_future(policy_qcmd)
        correction = np.zeros(7, dtype=np.float64)
        epsilon = 1e-4
        for joint in range(7):
            perturbed = policy_qcmd.copy()
            perturbed[joint] += epsilon
            pert_q, pert_dq = self._real_future(perturbed)
            sensitivity_q = (pert_q[:, joint] - base_q[:, joint]) / epsilon
            sensitivity_dq = (pert_dq[:, joint] - base_dq[:, joint]) / epsilon
            error_q = base_q[:, joint] - target_q[:, joint]
            error_dq = base_dq[:, joint] - target_dq[:, joint]
            numerator = (
                self.solver.q_tracking_weight
                * float(sensitivity_q @ error_q)
                + self.solver.dq_tracking_weight
                * float(sensitivity_dq @ error_dq)
                + self.solver.command_slew_weight
                * float(-self.previous_correction[joint]))
            denominator = (
                self.solver.q_tracking_weight
                * float(sensitivity_q @ sensitivity_q)
                + self.solver.dq_tracking_weight
                * float(sensitivity_dq @ sensitivity_dq)
                + self.solver.command_deviation_weight
                + self.solver.command_slew_weight)
            if denominator > 1e-12:
                correction[joint] = -numerator / denominator
        correction_limit = self.solver.max_correction_rad / (
            self.solver.residual_uncertainty_scale)
        correction = np.clip(correction, -correction_limit, correction_limit)
        correction_slew = self.solver.max_correction_rate_rad_s * self.dt_s / (
            self.solver.residual_uncertainty_scale)
        correction = np.clip(
            correction,
            self.previous_correction - correction_slew,
            self.previous_correction + correction_slew,
        )
        candidate = policy_qcmd + correction
        candidate = np.clip(candidate, self.q_low, self.q_high)
        predicted_q, predicted_dq = self._real_future(candidate)
        if (np.any(predicted_q < self.q_low[None, :] - 1e-9)
                or np.any(predicted_q > self.q_high[None, :] + 1e-9)
                or np.any(np.abs(predicted_dq) > self.dq_limit[None, :] + 1e-9)):
            raise PlantMatcherFailClosed(
                "matcher prediction exceeds existing q/dq envelope")
        direct_error = float(np.sqrt(np.mean(
            (base_q - target_q) ** 2 + 0.04 * (base_dq - target_dq) ** 2)))
        shadow_error = float(np.sqrt(np.mean(
            (predicted_q - target_q) ** 2
            + 0.04 * (predicted_dq - target_dq) ** 2)))
        return candidate, {
            "directPredictedTargetErrorRms": direct_error,
            "shadowPredictedTargetErrorRms": shadow_error,
            "predictedImprovement": direct_error - shadow_error,
            "correctionRad": (candidate - policy_qcmd).tolist(),
            "maxAbsCorrectionRad": float(np.max(np.abs(
                candidate - policy_qcmd))),
            "horizonSteps": self.solver.horizon_steps,
        }

    def propose(
        self,
        policy_qcmd: np.ndarray,
        sample: PolicyJointSample,
        *,
        feedback_max_age_s: float = 0.05,
    ) -> PlantMatcherProposal:
        nominal = _vector(policy_qcmd, "policy_qcmd")
        transition_error = sample.transition_error(
            self.previous_sample, max_receipt_age_s=feedback_max_age_s)
        if transition_error is not None:
            raise PlantMatcherFailClosed(transition_error)
        if not self.initialized:
            self.reset(sample)
        self.previous_sample = sample
        if self.mode is PlantMatcherMode.OFF:
            proposal = PlantMatcherProposal(
                policy_qcmd=nominal.copy(),
                matcher_shadow_q=nominal.copy(),
                selected_pre_safety_q=nominal.copy(),
                target_q=sample.q_raw.copy(),
                target_dq=sample.dq_raw.copy(),
                diagnostics={
                    "mode": "off",
                    "offEquivalent": True,
                    "applied": False,
                },
            )
            self.last_proposal = proposal
            return proposal
        self.real.reconcile_measured(sample.q_raw, sample.dq_raw)
        target_q, target_dq = self.target.step(nominal)
        candidate, diagnostics = self._solve(nominal)
        if self.mode is PlantMatcherMode.APPLY and not self.apply_authorized:
            raise PlantMatcherFailClosed(
                "apply authorization failed: allow/hash/provenance/apply_eligible gate")
        selected = candidate if self.mode is PlantMatcherMode.APPLY else nominal
        proposal = PlantMatcherProposal(
            policy_qcmd=nominal.copy(),
            matcher_shadow_q=candidate.copy(),
            selected_pre_safety_q=selected.copy(),
            target_q=target_q,
            target_dq=target_dq,
            diagnostics={
                **diagnostics,
                "mode": self.mode.value,
                "applied": self.mode is PlantMatcherMode.APPLY,
                "applyAuthorized": self.apply_authorized,
                "feedbackSourceStampS": sample.source_stamp_s,
                "feedbackSourceSequence": sample.source_sequence,
                "feedbackRepeated": sample.repeated_source_sample,
                "realFitId": self.real_fit_metadata.get("fit_id"),
                "realFitVersion": self.real_fit_metadata.get("fit_version"),
                "holdoutMetrics": self.real_fit_metadata.get("holdout_metrics"),
            },
        )
        self.previous_correction = candidate - nominal
        self.last_proposal = proposal
        return proposal

    def observe_published(self, published_q: np.ndarray) -> None:
        published = _vector(published_q, "published_q")
        if not self.initialized:
            raise PlantMatcherFailClosed("matcher state uninitialized")
        self.real.step(published)
        self.previous_published_q = published.copy()
