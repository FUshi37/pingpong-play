"""Atomic joint observation contract shared by policy, diagnostics and safety."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class PolicyDqMode(str, Enum):
    RAW_COHERENT = "raw_coherent"
    LEGACY_DQ_ONLY_ABLATION = "legacy_dq_only_ablation"

    @classmethod
    def parse(cls, value: object) -> "PolicyDqMode":
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise ValueError(
                "POLICY_DQ_MODE must be raw_coherent or "
                "legacy_dq_only_ablation") from exc


def _immutable_joint_vector(value: Any, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if vector.shape != (7,):
        raise ValueError(f"{name} must have shape (7,), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains non-finite values")
    vector.setflags(write=False)
    return vector


@dataclass(frozen=True, eq=False)
class PolicyJointSample:
    q_raw: np.ndarray
    dq_raw: np.ndarray
    source_stamp_s: float
    source_sequence: int
    receipt_age_s: float
    repeated_source_sample: bool
    velocity_source: str
    source_clock: str = "unknown"

    def __post_init__(self) -> None:
        object.__setattr__(self, "q_raw", _immutable_joint_vector(
            self.q_raw, "q_raw"))
        object.__setattr__(self, "dq_raw", _immutable_joint_vector(
            self.dq_raw, "dq_raw"))
        stamp = float(self.source_stamp_s)
        age = float(self.receipt_age_s)
        sequence = int(self.source_sequence)
        if not math.isfinite(stamp):
            raise ValueError("source_stamp_s must be finite")
        if not math.isfinite(age) or age < 0.0:
            raise ValueError("receipt_age_s must be finite and non-negative")
        if sequence < 0:
            raise ValueError("source_sequence must be non-negative")
        if not str(self.velocity_source).strip():
            raise ValueError("velocity_source must be non-empty")
        object.__setattr__(self, "source_stamp_s", stamp)
        object.__setattr__(self, "receipt_age_s", age)
        object.__setattr__(self, "source_sequence", sequence)

    def transition_error(
        self,
        previous: "PolicyJointSample | None",
        *,
        max_receipt_age_s: float,
        allow_source_sequence_gap: bool = False,
    ) -> str | None:
        if self.receipt_age_s > float(max_receipt_age_s):
            return (
                f"feedback_stale:{self.receipt_age_s:.6f}s>"
                f"{float(max_receipt_age_s):.6f}s")
        if previous is None:
            return None
        if self.source_sequence < previous.source_sequence:
            return "source_sequence_regression"
        if (self.source_sequence > previous.source_sequence + 1
                and not bool(allow_source_sequence_gap)):
            return "source_sequence_gap"
        if self.source_stamp_s < previous.source_stamp_s - 1e-12:
            return "source_time_regression"
        same_sequence = self.source_sequence == previous.source_sequence
        if same_sequence:
            if self.source_stamp_s != previous.source_stamp_s:
                return "same_sequence_stamp_changed"
            if not np.array_equal(self.q_raw, previous.q_raw):
                return "same_sequence_q_changed"
            if not np.array_equal(self.dq_raw, previous.dq_raw):
                return "same_sequence_dq_changed"
            if not self.repeated_source_sample:
                return "same_sequence_not_marked_repeated"
        elif self.repeated_source_sample:
            return "new_sequence_marked_repeated"
        return None

    def payload(self) -> dict[str, Any]:
        return {
            "qRaw": self.q_raw.tolist(),
            "dqRaw": self.dq_raw.tolist(),
            "sourceStampS": self.source_stamp_s,
            "sourceSequence": self.source_sequence,
            "receiptAgeS": self.receipt_age_s,
            "repeatedSourceSample": bool(self.repeated_source_sample),
            "velocitySource": str(self.velocity_source),
            "sourceClock": str(self.source_clock),
        }
