"""Feedback races must not reject a fresh atomic sample or bypass a real fault."""

from types import SimpleNamespace
import dataclasses
import threading
from unittest.mock import Mock

import numpy as np
import pytest

from pingpong_controller.tools import interactive_policy_safety_probe as probe
from pingpong_controller.ball_trajectory_recording import compact_trajectory_sample
from test_simcode9_fplc_contract import probe as fplc_probe, _policy_options


def feedback_probe(monkeypatch, *, age=.001, source_fault="", wait_ok=False):
    monkeypatch.setattr(probe, "FPLC_JOINT_FEEDBACK_MAX_AGE_S", .015)
    p = probe.PolicySafetyProbe.__new__(probe.PolicySafetyProbe)
    p.active_policy_profile = "fplc_gpu1_v172"
    p.control_loop_tick = 10
    calls = []
    p.ros_bridge = SimpleNamespace(
        wait_for_joint_feedback=lambda **kw: calls.append("wait") or wait_ok,
        get_joint_feedback_coherent_with_timing=lambda: (
            np.zeros(7), np.zeros(7), age, "joint_state", 1.,
            dict(jointSourceSequence=1, jointSourceFault=source_fault)),
    )
    return p, calls


def test_new_feedback_between_wait_timeout_and_atomic_read_is_accepted(monkeypatch):
    p, calls = feedback_probe(monkeypatch)
    p._joint_feedback_for_policy()
    assert calls == ["wait"]
    assert p.policy_joint_sample_error is None
    assert p.current_policy_joint_sample.receipt_age_s == .001
    timing = p.current_policy_joint_feedback_timing[1]
    assert timing["jointFeedbackWaitSucceeded"] is False
    assert timing["jointFeedbackRecoveredBeforeRead"] is True


@pytest.mark.parametrize("age", [.015001, .030, .100])
@pytest.mark.parametrize("wait_ok", [False, True])
def test_atomic_read_still_rejects_stale_feedback_after_either_wait_result(monkeypatch, age, wait_ok):
    p, _ = feedback_probe(monkeypatch, age=age, wait_ok=wait_ok)
    p._joint_feedback_for_policy()
    assert p.policy_joint_sample_error.startswith("feedback_stale:")
    assert p.current_policy_joint_sample.receipt_age_s == age


def test_fresh_callback_after_timeout_does_not_mask_a_source_fault(monkeypatch):
    p, _ = feedback_probe(monkeypatch, source_fault="source_time_regression")
    p._joint_feedback_for_policy()
    assert p.policy_joint_sample_error == "source_time_regression"


@pytest.mark.parametrize("read_time,accepted", [(50.004, True), (50.006, False)])
def test_callback_recovery_still_obeys_control_tick_deadline(monkeypatch, read_time, accepted):
    p, _ = feedback_probe(monkeypatch)
    p.control_tick_deadline_perf = 50.005
    monkeypatch.setattr(probe.time, "perf_counter", lambda: read_time)
    p._joint_feedback_for_policy()
    assert (p.policy_joint_sample_error is None) is accepted
    if not accepted:
        assert p.policy_joint_sample_error == "joint_feedback_wait_deadline"


def test_fresh_feedback_after_timeout_runs_actor_and_publishes_its_exact_target(fplc_probe, monkeypatch):
    p = fplc_probe
    measured = p.controller.native_reset_arm_q.copy()
    published = []
    p.ros_bridge = SimpleNamespace(
        wait_for_joint_feedback=lambda **kw: False,
        get_joint_feedback_coherent_with_timing=lambda: (
            measured.copy(), np.zeros(7), .001, "joint_state", 1., dict(jointSourceSequence=1)),
        get_joint_feedback_with_age=lambda: (measured.copy(), np.zeros(7), .001, "joint_state"),
        get_joint_feedback=lambda: (measured.copy(), np.zeros(7)),
        get_cart_feedback=lambda: (None, None, None),
        publish_right_arm_command=lambda q: published.append(np.array(q)),
    )
    p.command_stream_active = True
    options = dataclasses.replace(_policy_options(p), use_ros_state=True, publish_commands=True)
    p.runtime_options = options
    p.control_tick_deadline_perf = 50.005
    monkeypatch.setattr(probe.time, "perf_counter", lambda: 50.004)
    mode, reason, _, _, executed = p._run_one_control_step(
        np.array([0., -.49, 1.16]), np.array([0., 0., -.3]), True, .001, options)
    assert mode == "policy" and executed, reason
    assert not p.publish_rearm_required
    target = p.controller.arm_cmd_q.copy()
    assert p._publish_control_command(target, options, profile="normal")
    assert len(published) == 1
    np.testing.assert_array_equal(published[0], target)


def test_changed_velocity_under_reused_header_is_a_new_atomic_sample(monkeypatch):
    node = probe.RosBridgeNode.__new__(probe.RosBridgeNode)
    node.lock = threading.Lock()
    node.latest_joint_q = None
    node.latest_joint_dq = node.latest_joint_ddq = None
    node.latest_joint_sample_stamp_s = node.latest_joint_header_stamp_s = None
    node.latest_joint_stamp_monotonic_s = None
    clock = [100.]
    monkeypatch.setattr(probe.time, "monotonic", lambda: clock[0])
    message = SimpleNamespace(
        name=probe.RIGHT_ARM_JOINTS, position=[0.] * 7, velocity=[.1] * 7,
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)))
    node._joint_state_callback(message)
    first_sequence = node.latest_joint_source_sequence
    first_stamp = node.latest_joint_sample_stamp_s
    clock[0] += .005
    message.velocity = [.2] * 7
    node._joint_state_callback(message)
    assert node.latest_joint_source_sequence == first_sequence + 1
    assert node.latest_joint_sample_stamp_s == pytest.approx(first_stamp + .005)
    assert not node.latest_joint_repeated_source_sample
    np.testing.assert_array_equal(node.latest_joint_dq, np.full(7, .2, np.float32))


def test_exact_duplicate_keeps_the_whole_atomic_velocity_sample(monkeypatch):
    node = probe.RosBridgeNode.__new__(probe.RosBridgeNode)
    node.lock = threading.Lock()
    node.latest_joint_q = None
    node.latest_joint_dq = node.latest_joint_ddq = None
    node.latest_joint_sample_stamp_s = node.latest_joint_header_stamp_s = None
    node.latest_joint_stamp_monotonic_s = None
    clock = [100.]
    monkeypatch.setattr(probe.time, "monotonic", lambda: clock[0])
    message = SimpleNamespace(
        name=probe.RIGHT_ARM_JOINTS, position=[0.] * 7, velocity=[.1] * 7,
        header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=0)))
    node._joint_state_callback(message)
    first = (node.latest_joint_source_sequence, node.latest_joint_sample_stamp_s)
    clock[0] += .005
    node._joint_state_callback(message)
    assert (node.latest_joint_source_sequence, node.latest_joint_sample_stamp_s) == first
    assert node.latest_joint_repeated_source_sample


def test_trajectory_recording_retains_exact_fault_and_effective_send_state_without_feedback_reads():
    rows = []
    controller = SimpleNamespace(
        last_obs=np.zeros(72),
        latest_command_state=Mock(side_effect=AssertionError("command telemetry requested")),
        latest_obs_and_action=Mock(side_effect=AssertionError("policy telemetry requested")),
    )
    backend = SimpleNamespace(
        recorder=SimpleNamespace(active=True, trajectory_only=True, record_control_sample=rows.append),
        controller=controller, ros_bridge=None,
        now_s=.1, control_loop_tick=11,
        cfg=SimpleNamespace(enable_camera_ball=False, enable_mocap_ball=False),
        active_policy_profile="fplc_gpu1_v172", active_policy_model="actual_model",
        runtime_options=SimpleNamespace(publish_commands=False),
        publish_rearm_required=True, publish_rearm_reason="policy_joint_feedback_fault",
        current_policy_joint_feedback_timing=(10, {
            "jointReceiptAgeS": .020, "jointSampleControlError": "feedback_stale",
            "jointFeedbackWaitSucceeded": False, "jointFeedbackRecoveredBeforeRead": False,
        }),
    )
    reason = "fresh q/dq feedback required: feedback_stale"
    probe.PolicySafetyProbe._record_control_sample(
        backend, source_seq=10, source_time_s=.1, ball_observation_seq=1,
        camera_stamp_s=.1, camera_measurement={}, mode="policy_feedback_hold", policy_executed=False,
        reason=reason, conditions={}, raw_ball_pos=np.zeros(3), measured_ball_pos=np.zeros(3),
        measured_ball_vel=np.zeros(3), observed_ball_pos=np.zeros(3), observed_ball_vel=np.zeros(3),
        raw_valid=True, control_valid=True, age_s=0., noise_std_m=0.,
        options=SimpleNamespace(publish_commands=True), ball_source="mocap", command_published=False,
    )
    slim = compact_trajectory_sample(compact_trajectory_sample(rows[0]))
    assert slim["reason"] == reason
    status = slim["controlStatus"]
    assert status["jointSampleControlError"] == "feedback_stale"
    assert status["jointFeedbackAgeS"] == .020
    assert status["activePolicyModel"] == "actual_model"
    assert status["rearmRequired"] and status["rearmReason"] == "policy_joint_feedback_fault"
    assert not status["publishCommands"] and not status["commandPublished"]
    assert not status["policyExecuted"]
    assert all(key not in slim for key in ("policyObs", "command", "ros", "runtime"))

