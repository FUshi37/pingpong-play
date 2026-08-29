#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.policy_ball_handover import (  # noqa: E402
    BoundedHandoverBlend,
    is_fractional_refresh,
    paired_handover_replay,
)


def test_90hz_fractional_schedule_is_not_bypassed():
    refresh = [index for index in range(10)
               if is_fractional_refresh(index, 90.0, 0.005)]
    assert refresh == [0, 3, 5, 7, 9]


def test_bounded_blend_keeps_position_velocity_consistent_and_gates_outlier():
    blend = BoundedHandoverBlend(0.18, 8.0, 0.04)
    accepted = blend.begin(
        np.zeros(3), np.zeros(3), np.full(3, 0.03), np.full(3, 0.6))
    assert accepted["accepted"] is True
    pos, vel, info = blend.scheduled_step(
        np.zeros(3), np.zeros(3), 1.0 / 90.0)
    np.testing.assert_allclose(pos / 0.03, vel / 0.6)
    assert 0.0 < info["alpha"] < 1.0
    rejected = blend.begin(
        pos, vel, np.full(3, 1.0), np.full(3, 20.0))
    assert rejected["accepted"] is False
    assert rejected["reason"] == "innovation_gate"


def test_frozen_paired_replay_reports_jumps_and_preimpact_difference():
    rows = []
    for index in range(12):
        position = [0.0, 0.0, 1.0]
        velocity = [0.0, 0.0, -1.0]
        handover = False
        if index >= 4:
            position = [0.03, 0.0, 1.02]
            velocity = [0.4, 0.0, -0.5]
            handover = index == 4
        rows.append({
            "position": position,
            "velocity": velocity,
            "valid": True,
            "handover": handover,
            "second_impact": index == 10,
        })
    report = paired_handover_replay(
        rows, actor=lambda obs: np.tanh(obs[:3] + 0.2 * obs[3:]))
    assert report["inputRows"] == 12
    assert report["preSecondImpactCumulativeObservationDifference"] > 0.0
    assert report["causalClaim"] == "offline_paired_replay_only_not_real_ab"
