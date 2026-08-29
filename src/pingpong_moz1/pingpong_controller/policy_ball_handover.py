"""Causal 90 Hz predictor/camera handover policies and offline paired replay."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable

import numpy as np


class PolicyBallHandoverMode(str, Enum):
    LEGACY_FORCE_FRESH = "legacy_force_fresh"
    STRICT_FRACTIONAL_CONTINUOUS = "strict_fractional_continuous"

    @classmethod
    def parse(cls, value: object) -> "PolicyBallHandoverMode":
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError(
                "POLICY_BALL_HANDOVER_MODE must be legacy_force_fresh or "
                "strict_fractional_continuous") from exc


def is_fractional_refresh(step: int, rate_hz: float, dt_s: float) -> bool:
    if int(step) == 0:
        return True
    period = float(rate_hz) * float(dt_s)
    return math.floor(step * period) > math.floor((step - 1) * period)


@dataclass
class BoundedHandoverBlend:
    max_position_innovation_m: float
    max_velocity_innovation_m_s: float
    duration_s: float
    target_position: np.ndarray | None = None
    target_velocity: np.ndarray | None = None
    elapsed_s: float = 0.0
    active: bool = False

    def begin(
        self,
        held_position: np.ndarray,
        held_velocity: np.ndarray,
        measured_position: np.ndarray,
        measured_velocity: np.ndarray,
    ) -> dict[str, Any]:
        held_p = np.asarray(held_position, dtype=np.float64).reshape(3)
        held_v = np.asarray(held_velocity, dtype=np.float64).reshape(3)
        measured_p = np.asarray(measured_position, dtype=np.float64).reshape(3)
        measured_v = np.asarray(measured_velocity, dtype=np.float64).reshape(3)
        if not all(np.all(np.isfinite(value)) for value in (
                held_p, held_v, measured_p, measured_v)):
            self.active = False
            return {"accepted": False, "reason": "non_finite_innovation"}
        pos_innovation = measured_p - held_p
        vel_innovation = measured_v - held_v
        pos_norm = float(np.linalg.norm(pos_innovation))
        vel_norm = float(np.linalg.norm(vel_innovation))
        if (pos_norm > float(self.max_position_innovation_m)
                or vel_norm > float(self.max_velocity_innovation_m_s)):
            self.active = False
            return {
                "accepted": False,
                "reason": "innovation_gate",
                "positionInnovationM": pos_innovation.tolist(),
                "velocityInnovationMps": vel_innovation.tolist(),
                "positionInnovationNormM": pos_norm,
                "velocityInnovationNormMps": vel_norm,
            }
        self.target_position = measured_p.copy()
        self.target_velocity = measured_v.copy()
        self.elapsed_s = 0.0
        self.active = True
        return {
            "accepted": True,
            "positionInnovationM": pos_innovation.tolist(),
            "velocityInnovationMps": vel_innovation.tolist(),
            "positionInnovationNormM": pos_norm,
            "velocityInnovationNormMps": vel_norm,
        }

    def scheduled_step(
        self,
        held_position: np.ndarray,
        held_velocity: np.ndarray,
        refresh_interval_s: float,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        held_p = np.asarray(held_position, dtype=np.float64).reshape(3)
        held_v = np.asarray(held_velocity, dtype=np.float64).reshape(3)
        if (not self.active or self.target_position is None
                or self.target_velocity is None):
            return held_p.copy(), held_v.copy(), {
                "active": False, "alpha": 0.0}
        self.elapsed_s += max(0.0, float(refresh_interval_s))
        duration = max(float(self.duration_s), 1e-9)
        alpha = min(1.0, self.elapsed_s / duration)
        # Each accepted sample is consistent: position and velocity use the
        # same finite-duration interpolation coefficient.
        out_p = held_p + alpha * (self.target_position - held_p)
        out_v = held_v + alpha * (self.target_velocity - held_v)
        if alpha >= 1.0:
            self.active = False
        return out_p, out_v, {"active": True, "alpha": alpha}


def paired_handover_replay(
    rows: Iterable[dict[str, Any]],
    actor: Callable[[np.ndarray], np.ndarray],
    *,
    rate_hz: float = 90.0,
    control_dt_s: float = 0.005,
    blend_duration_s: float = 0.04,
    position_gate_m: float = 0.18,
    velocity_gate_m_s: float = 8.0,
) -> dict[str, Any]:
    """Run both modes on identical frozen inputs without loading an actor."""
    frozen = list(rows)

    def run(mode: PolicyBallHandoverMode) -> dict[str, Any]:
        held = np.zeros(6, dtype=np.float64)
        have_held = False
        blend = BoundedHandoverBlend(
            position_gate_m, velocity_gate_m_s, blend_duration_s)
        observations = []
        actions = []
        impact_index = None
        for step, row in enumerate(frozen):
            measured = np.concatenate((
                np.asarray(row["position"], dtype=np.float64).reshape(3),
                np.asarray(row["velocity"], dtype=np.float64).reshape(3),
            ))
            valid = bool(row.get("valid", True)) and np.all(np.isfinite(measured))
            scheduled = is_fractional_refresh(step, rate_hz, control_dt_s)
            handover = bool(row.get("handover", False))
            refresh = scheduled or (
                mode is PolicyBallHandoverMode.LEGACY_FORCE_FRESH and handover)
            if not have_held and valid:
                held = measured.copy()
                have_held = True
            elif mode is PolicyBallHandoverMode.STRICT_FRACTIONAL_CONTINUOUS:
                if handover and valid and have_held and not blend.active:
                    blend.begin(held[:3], held[3:], measured[:3], measured[3:])
                if scheduled and valid and have_held:
                    if blend.active:
                        p, v, _ = blend.scheduled_step(
                            held[:3], held[3:], 1.0 / rate_hz)
                        held = np.concatenate((p, v))
                    else:
                        held = measured.copy()
            elif refresh and valid:
                held = measured.copy()
            observations.append(held.copy())
            actions.append(np.asarray(actor(held.copy()), dtype=np.float64))
            if impact_index is None and bool(row.get("second_impact", False)):
                impact_index = step
        obs = np.asarray(observations)
        action = np.asarray(actions)
        return {
            "observation": obs,
            "action": action,
            "observationJump": np.linalg.norm(np.diff(obs, axis=0), axis=1),
            "actionJump": np.linalg.norm(np.diff(action, axis=0), axis=1),
            "secondImpactIndex": impact_index,
        }

    legacy = run(PolicyBallHandoverMode.LEGACY_FORCE_FRESH)
    strict = run(PolicyBallHandoverMode.STRICT_FRACTIONAL_CONTINUOUS)
    end = legacy["secondImpactIndex"]
    if end is None:
        end = len(frozen) - 1
    cumulative = float(np.sum(np.linalg.norm(
        strict["observation"][:end + 1] - legacy["observation"][:end + 1],
        axis=1)))
    return {
        "inputRows": len(frozen),
        "legacyObservationJumpMax": float(np.max(
            legacy["observationJump"], initial=0.0)),
        "strictObservationJumpMax": float(np.max(
            strict["observationJump"], initial=0.0)),
        "legacyActionJumpMax": float(np.max(
            legacy["actionJump"], initial=0.0)),
        "strictActionJumpMax": float(np.max(
            strict["actionJump"], initial=0.0)),
        "preSecondImpactCumulativeObservationDifference": cumulative,
        "causalClaim": "offline_paired_replay_only_not_real_ab",
    }
