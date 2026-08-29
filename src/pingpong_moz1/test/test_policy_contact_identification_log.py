#!/usr/bin/env python3

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
PROBE_PATH = (
    PACKAGE_ROOT / "pingpong_controller" / "tools"
    / "interactive_policy_safety_probe.py")
SPEC = importlib.util.spec_from_file_location(
    "interactive_policy_safety_probe_contact_test", PROBE_PATH)
PROBE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PROBE
SPEC.loader.exec_module(PROBE)


class _Controller:
    def current_arm_cmd_q(self):
        return np.zeros(7)

    def _compute_racket_pose_jacobians_base(self, q):
        del q
        return (
            np.array([0.1, -0.4, 1.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
            np.arange(21, dtype=np.float64).reshape(3, 7) * 0.01,
            np.arange(21, 42, dtype=np.float64).reshape(3, 7) * 0.01,
            True,
        )


def test_contact_log_uses_one_atomic_sample_for_fk_and_both_jacobians():
    probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
    sample = PROBE.PolicyJointSample(
        q_raw=np.linspace(-0.3, 0.3, 7),
        dq_raw=np.linspace(-1.0, 1.0, 7),
        source_stamp_s=123.456,
        source_sequence=42,
        receipt_age_s=0.003,
        repeated_source_sample=False,
        velocity_source="joint_state",
    )
    probe.current_policy_joint_sample = sample
    probe.controller = _Controller()
    probe.last_estimator_info = {"state": "tracking"}
    probe.last_high_ball_occlusion_info = {"state": "camera"}
    probe.last_policy_nominal_reference = np.full(7, 0.1)
    probe.last_matcher_shadow_q = np.full(7, 0.11)
    probe.last_matcher_pre_safety_q = np.full(7, 0.1)
    probe.last_published_safe_reference = np.full(7, 0.09)

    payload = probe._contact_identification_payload(
        measured_ball_pos=np.array([0.11, -0.39, 1.02]),
        measured_ball_vel=np.array([0.2, 0.1, -1.0]),
        raw_valid=True,
        age_s=0.004,
        camera_measurement={"velocity_impact_detected": True},
        policy_action=np.linspace(-0.2, 0.2, 7),
        policy_executed_action=np.linspace(-0.1, 0.1, 7),
    )

    assert payload["inferredContact"] is True
    assert payload["sourceStampS"] == payload["kinematicsSourceStampS"]
    assert payload["sourceSequence"] == 42
    expected_linear = (
        np.arange(21, dtype=np.float64).reshape(3, 7) * 0.01
    ) @ sample.dq_raw
    expected_angular = (
        np.arange(21, 42, dtype=np.float64).reshape(3, 7) * 0.01
    ) @ sample.dq_raw
    np.testing.assert_allclose(
        payload["racketLinearVelocity"], expected_linear, atol=1e-6)
    np.testing.assert_allclose(
        payload["racketAngularVelocity"], expected_angular, atol=1e-6)
    assert payload["velocityEquation"] == "J(q_raw)@dq_raw"
    assert payload["wristForceTorque"]["available"] is False
    assert payload["highSpeedCamera"]["available"] is False
