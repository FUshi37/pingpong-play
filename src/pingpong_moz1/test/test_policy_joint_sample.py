#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.policy_joint_sample import (  # noqa: E402
    PolicyDqMode,
    PolicyJointSample,
)


def _sample(seq=1, stamp=10.0, *, repeated=False, age=0.001):
    return PolicyJointSample(
        q_raw=np.arange(7) * 0.01,
        dq_raw=np.arange(7) * -0.02,
        source_stamp_s=stamp,
        source_sequence=seq,
        receipt_age_s=age,
        repeated_source_sample=repeated,
        velocity_source="joint_state",
    )


def test_joint_sample_vectors_are_immutable_and_atomic():
    sample = _sample()
    with pytest.raises(ValueError):
        sample.q_raw[0] = 4.0
    assert sample.payload()["sourceStampS"] == 10.0
    assert sample.payload()["sourceSequence"] == 1


def test_same_source_sequence_is_zoh_not_a_new_velocity_sample():
    first = _sample()
    held = _sample(repeated=True)
    assert held.transition_error(first, max_receipt_age_s=0.05) is None


@pytest.mark.parametrize(
    "current,expected",
    [
        (_sample(seq=3, stamp=10.01), "source_sequence_gap"),
        (_sample(seq=2, stamp=9.99), "source_time_regression"),
        (_sample(seq=2, stamp=10.01, age=0.051), "feedback_stale"),
    ],
)
def test_gap_time_regression_and_stale_feedback_fail_closed(current, expected):
    error = current.transition_error(_sample(), max_receipt_age_s=0.05)
    assert error is not None and error.startswith(expected)


def test_latest_atomic_sample_can_explicitly_allow_skipped_source_messages():
    previous = _sample(seq=10, stamp=10.0)
    latest = _sample(seq=12, stamp=10.01)
    assert latest.transition_error(
        previous,
        max_receipt_age_s=0.05,
        allow_source_sequence_gap=True,
    ) is None


def test_dq_mode_is_explicit():
    assert PolicyDqMode.parse("raw_coherent") is PolicyDqMode.RAW_COHERENT
    assert PolicyDqMode.parse("legacy_dq_only_ablation") is (
        PolicyDqMode.LEGACY_DQ_ONLY_ABLATION)
    with pytest.raises(ValueError, match="POLICY_DQ_MODE"):
        PolicyDqMode.parse("magic_alignment")
