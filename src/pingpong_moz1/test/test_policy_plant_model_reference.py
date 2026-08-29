#!/usr/bin/env python3

import json
import sys
from pathlib import Path

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.policy_joint_sample import PolicyJointSample  # noqa: E402
from pingpong_controller.policy_plant_model_reference import (  # noqa: E402
    DelayedSecondOrderJointPlant,
    JointPlantParameters,
    PlantMatcherFailClosed,
    PlantMatcherSolverConfig,
    PolicyPlantModelReference,
    load_real_plant_fit,
)


TARGET = JointPlantParameters(
    natural_frequency_rad_s=np.array(
        [20.363849, 22.655975, 22.650471, 21.730533,
         19.663364, 22.815526, 23.645817]),
    damping_ratio=np.array(
        [0.391768, 0.366169, 0.345738, 0.346,
         0.347896, 0.345814, 0.380713]),
    gain_per_joint=np.array(
        [0.997884, 0.996612, 0.992312, 0.990942,
         0.982327, 0.992695, 0.983194]),
    delay_ms_per_joint=np.array([45, 50, 45, 40, 35, 45, 55]),
)
REAL = JointPlantParameters(
    natural_frequency_rad_s=np.array(
        [21.891138, 22.089575, 22.650471, 21.730533,
         20.154948, 22.245138, 23.054672]),
    damping_ratio=np.array(
        [0.333003, 0.329552, 0.311164, 0.3114,
         0.313106, 0.311233, 0.285535]),
    gain_per_joint=TARGET.gain_per_joint,
    delay_ms_per_joint=np.array([45, 50, 45, 40, 35, 45, 50]),
)
SOLVER = PlantMatcherSolverConfig.from_mapping({
    "horizon_steps": 16,
    "q_tracking_weight": 1.0,
    "dq_tracking_weight": 0.04,
    "command_deviation_weight": 0.2,
    "command_slew_weight": 0.1,
    "max_correction_rad": [0.08] * 7,
    "max_correction_rate_rad_s": [10.0] * 7,
    "residual_uncertainty_scale": 1.0,
})


def _sample(seq, stamp, q=None, dq=None, repeated=False, age=0.001):
    return PolicyJointSample(
        q_raw=np.zeros(7) if q is None else q,
        dq_raw=np.zeros(7) if dq is None else dq,
        source_stamp_s=stamp,
        source_sequence=seq,
        receipt_age_s=age,
        repeated_source_sample=repeated,
        velocity_source="joint_state",
    )


def _matcher(mode="shadow", *, allow=False, metadata=None):
    return PolicyPlantModelReference(
        target_parameters=TARGET,
        real_parameters=REAL,
        solver=SOLVER,
        mode=mode,
        dt_s=0.005,
        warm_q=np.zeros(7),
        q_low=np.full(7, -2.0),
        q_high=np.full(7, 2.0),
        dq_limit=np.full(7, 20.0),
        allow_apply=allow,
        release_hash_match=True,
        real_fit_metadata=metadata or {
            "fit_id": "fixture",
            "fit_version": "1",
            "apply_eligible": False,
            "source_recording_sha256": None,
            "holdout_metrics": {"status": "fixture"},
        },
    )


def test_target_discretization_delay_rounding_and_frozen_first_active_tick():
    plant = DelayedSecondOrderJointPlant(TARGET, 0.005, np.zeros(7))
    assert plant.delay_steps.tolist() == [9, 10, 9, 8, 7, 9, 11]
    command = np.ones(7)
    for _ in range(7):
        q, dq = plant.step(command)
        np.testing.assert_array_equal(q, np.zeros(7))
        np.testing.assert_array_equal(dq, np.zeros(7))
    q, dq = plant.step(command)
    # Joint 4 (35 ms / seven ticks) is the first active delayed input.
    assert q[4] == pytest.approx(0.004637529426657983, abs=1e-12)
    assert dq[4] == pytest.approx(1.8326179944091756, abs=1e-12)
    np.testing.assert_array_equal(q[:4], np.zeros(4))


def test_off_mode_is_numerically_identical_to_direct_path():
    matcher = _matcher("off")
    nominal = np.linspace(-0.2, 0.2, 7)
    proposal = matcher.propose(nominal, _sample(0, 1.0))
    np.testing.assert_array_equal(proposal.selected_pre_safety_q, nominal)
    assert proposal.diagnostics["offEquivalent"] is True


def test_apply_is_blocked_without_verified_source_hash_even_with_allow():
    matcher = _matcher("apply", allow=True)
    with pytest.raises(PlantMatcherFailClosed, match="apply authorization"):
        matcher.propose(np.full(7, 0.1), _sample(0, 1.0))


@pytest.mark.parametrize(
    "sample,reason",
    [
        (_sample(2, 1.01), "source_sequence_gap"),
        (_sample(1, 0.99), "source_time_regression"),
        (_sample(1, 1.01, age=0.051), "feedback_stale"),
    ],
)
def test_gap_time_regression_and_stale_feedback_fail_closed(sample, reason):
    matcher = _matcher("shadow")
    matcher.propose(np.zeros(7), _sample(0, 1.0))
    matcher.observe_published(np.zeros(7))
    with pytest.raises(PlantMatcherFailClosed, match=reason):
        matcher.propose(np.zeros(7), sample)


def test_shadow_suggestion_reduces_predicted_target_error_and_stays_bounded():
    matcher = _matcher("shadow")
    sample = _sample(0, 1.0)
    improvements = []
    for tick in range(40):
        command = np.full(7, min(0.4, tick * 0.02))
        if tick:
            sample = _sample(tick, 1.0 + tick * 0.005)
        proposal = matcher.propose(command, sample)
        improvements.append(proposal.diagnostics.get("predictedImprovement", 0.0))
        assert np.max(np.abs(
            proposal.matcher_shadow_q - command)) <= 0.0800001
        # Shadow observes the direct command actually published.
        matcher.observe_published(command)
    assert max(improvements[12:]) > 0.0


def test_real_fit_keeps_missing_provenance_explicit_and_apply_ineligible(tmp_path):
    config = {
        "schema_version": 1,
        "fit_id": "candidate",
        "fit_version": "1",
        "source_recordings": ["record_new3"],
        "source_recording_sha256": None,
        "provenance_status": "missing_raw_hash",
        "apply_eligible": False,
        "holdout_metrics": {"status": "known_bias"},
        "natural_frequency_rad_s": [20] * 7,
        "damping_ratio": [0.3] * 7,
        "gain_per_joint": [1] * 7,
        "delay_ms_per_joint": [45] * 7,
        "solver": {
            "max_correction_rad": [0.05] * 7,
            "max_correction_rate_rad_s": [0.5] * 7,
        },
    }
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    payload, _parameters, _solver = load_real_plant_fit(path)
    assert payload["source_recording_sha256"] is None
    assert payload["apply_eligible"] is False
