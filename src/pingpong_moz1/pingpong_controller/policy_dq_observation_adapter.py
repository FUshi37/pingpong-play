"""Causal real-joint velocity adapter for policy observations.

The adapter deliberately changes only the velocity vector supplied to the
actor.  Callers must retain the original JointState velocity for safety,
tracking, compensation, and recording.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


SIMCODE5_DQ_ADAPTER_GAIN = (
    0.8229080313,
    0.7241145962,
    0.8223335217,
    0.8333474552,
    0.7505777228,
    0.8204823652,
    0.7738224499,
)
SIMCODE5_DQ_ADAPTER_TAU_S = (0.020, 0.0, 0.0, 0.010, 0.0, 0.010, 0.020)
SIMCODE5_DQ_ADAPTER_DELAY_STEPS = (0, 0, 3, 0, 2, 0, 0)


def _finite_joint_vector(values: Any, *, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (7,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain seven finite values")
    return vector


@dataclass(frozen=True)
class PolicyDqAdapterConfig:
    """Immutable parameters for ``gain * delay(EMA(dq_real))``."""

    enabled: bool
    gain: tuple[float, ...]
    tau_s: tuple[float, ...]
    delay_steps: tuple[int, ...]
    dt_s: float

    def __post_init__(self) -> None:
        gain = _finite_joint_vector(self.gain, name="gain")
        tau_s = _finite_joint_vector(self.tau_s, name="tau_s")
        delay_raw = np.asarray(self.delay_steps).reshape(-1)
        if delay_raw.shape != (7,):
            raise ValueError("delay_steps must contain seven integers")
        try:
            delay_numeric = delay_raw.astype(np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "delay_steps must contain seven integers") from exc
        if (not np.all(np.isfinite(delay_numeric))
                or not np.all(delay_numeric == np.floor(delay_numeric))):
            raise ValueError("delay_steps must contain seven integers")
        delay_steps = delay_numeric.astype(np.int64)
        if np.any(gain < 0.0):
            raise ValueError("gain values must be non-negative")
        if np.any(tau_s < 0.0):
            raise ValueError("tau_s values must be non-negative")
        if np.any(delay_steps < 0):
            raise ValueError("delay_steps values must be non-negative")
        if not math.isfinite(float(self.dt_s)) or float(self.dt_s) <= 0.0:
            raise ValueError("dt_s must be finite and positive")

        object.__setattr__(self, "gain", tuple(float(v) for v in gain))
        object.__setattr__(self, "tau_s", tuple(float(v) for v in tau_s))
        object.__setattr__(
            self, "delay_steps", tuple(int(v) for v in delay_steps))
        object.__setattr__(self, "dt_s", float(self.dt_s))

    def payload(self) -> dict[str, Any]:
        return {
            "configuredEnabled": bool(self.enabled),
            "formula": "gain * delay_steps(EMA(dq_real, tau_s))",
            "gain": list(self.gain),
            "tauS": list(self.tau_s),
            "delaySteps": list(self.delay_steps),
            "delayMs": [
                round(1000.0 * self.dt_s * value, 6)
                for value in self.delay_steps
            ],
            "dtS": self.dt_s,
        }


class PolicyDqObservationAdapter:
    """Stateful causal implementation of the policy-facing dq transform.

    One update corresponds to one fixed-rate policy tick.  This matches the
    fitting/replay contract and treats a repeated ROS sample as a zero-order
    held input to the continuous-time EMA.
    """

    def __init__(self, config: PolicyDqAdapterConfig) -> None:
        self.config = config
        self._gain = np.asarray(config.gain, dtype=np.float64)
        self._tau_s = np.asarray(config.tau_s, dtype=np.float64)
        self._delay_steps = np.asarray(config.delay_steps, dtype=np.int64)
        self._max_delay = int(np.max(self._delay_steps))
        self._alpha = np.ones(7, dtype=np.float64)
        positive_tau = self._tau_s > 0.0
        self._alpha[positive_tau] = 1.0 - np.exp(
            -float(config.dt_s) / self._tau_s[positive_tau])
        self._history: deque[np.ndarray] = deque(
            maxlen=self._max_delay + 1)
        self._filtered: np.ndarray | None = None
        self._last_source_stamp_s: float | None = None
        self._sample_index = -1
        self._last_payload = self._inactive_payload("not_initialized")

    def _inactive_payload(self, reason: str) -> dict[str, Any]:
        return {
            **self.config.payload(),
            "active": False,
            "reason": str(reason),
            "sampleIndex": int(self._sample_index),
        }

    def reset(self, reason: str = "reset") -> None:
        self._history.clear()
        self._filtered = None
        self._last_source_stamp_s = None
        self._sample_index = -1
        self._last_payload = self._inactive_payload(reason)

    def bypass_payload(
        self,
        reason: str,
        raw_dq: Any | None = None,
    ) -> dict[str, Any]:
        payload = self._inactive_payload(reason)
        if raw_dq is not None:
            raw = _finite_joint_vector(raw_dq, name="raw_dq")
            payload.update({
                "rawDqRadS": raw.astype(np.float32).tolist(),
                "policyDqRadS": raw.astype(np.float32).tolist(),
            })
        self._last_payload = payload
        return dict(payload)

    def update(
        self,
        raw_dq: Any,
        *,
        source_stamp_s: float | None = None,
        enabled: bool | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        raw = _finite_joint_vector(raw_dq, name="raw_dq")
        runtime_enabled = (
            bool(self.config.enabled) if enabled is None else bool(enabled))
        if not runtime_enabled:
            payload = self.bypass_payload("disabled", raw)
            payload["runtimeEnabled"] = False
            self._last_payload = dict(payload)
            return raw.astype(np.float32), payload

        if self._filtered is None:
            self._filtered = raw.copy()
            for _ in range(self._max_delay + 1):
                self._history.append(self._filtered.copy())
        else:
            self._filtered = (
                self._filtered + self._alpha * (raw - self._filtered))
            self._history.append(self._filtered.copy())

        self._sample_index += 1
        delayed = np.asarray([
            self._history[-1 - int(delay)][joint]
            for joint, delay in enumerate(self._delay_steps)
        ], dtype=np.float64)
        policy_dq = self._gain * delayed

        stamp: float | None = None
        source_repeated: bool | None = None
        if source_stamp_s is not None:
            try:
                candidate = float(source_stamp_s)
                if math.isfinite(candidate):
                    stamp = candidate
                    source_repeated = bool(
                        self._last_source_stamp_s is not None
                        and candidate <= self._last_source_stamp_s + 1.0e-9)
            except (TypeError, ValueError):
                pass
        if stamp is not None:
            self._last_source_stamp_s = stamp

        payload = {
            **self.config.payload(),
            "runtimeEnabled": True,
            "active": True,
            "reason": "applied",
            "sampleIndex": int(self._sample_index),
            "sourceSampleStampS": stamp,
            "sourceSampleRepeated": source_repeated,
            "emaAlpha": self._alpha.astype(np.float32).tolist(),
            "rawDqRadS": raw.astype(np.float32).tolist(),
            "filteredDqRadS": self._filtered.astype(np.float32).tolist(),
            "delayedDqRadS": delayed.astype(np.float32).tolist(),
            "policyDqRadS": policy_dq.astype(np.float32).tolist(),
        }
        self._last_payload = payload
        return policy_dq.astype(np.float32), dict(payload)

    def status(self) -> dict[str, Any]:
        return dict(self._last_payload)
