"""Receiver isolation and bounded jitter recovery without ROS actuation."""
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from pingpong_controller.tools import interactive_policy_safety_probe as probe
from pingpong_controller import control_runtime
from pingpong_controller.fplc_deployment import joint_feedback_max_age_s


@pytest.fixture(autouse=True)
def historical_10ms_configuration(monkeypatch):
    # Retain the original recorded-gap regression cases while the launcher
    # default changes to 15 ms. New cases below cover all supported limits.
    monkeypatch.setattr(probe, 'FPLC_JOINT_FEEDBACK_MAX_AGE_S', .010)
    monkeypatch.setattr(probe, 'FPLC_JOINT_FEEDBACK_MAX_GAP_S', .015)


def receiver(age_s):
    node = probe.RosBridgeNode.__new__(probe.RosBridgeNode)
    node.latest_joint_q = np.zeros(7)
    node.latest_joint_stamp_monotonic_s = 100.0
    node.joint_feedback_diagnostics = dict(waitCount=0, waitRecoveries=0, waitTimeouts=0)
    clock = SimpleNamespace(now=100.0 + age_s)
    return node, clock


class ScriptedCondition:
    def __init__(self, callback):
        self.callback = callback
        self.timeouts = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def wait(self, timeout):
        self.timeouts.append(timeout)
        self.callback(timeout)


def test_recorded_12ms_gap_recovers_only_after_actual_callback(monkeypatch):
    node, clock = receiver(.010539894)

    def deliver(_timeout):
        clock.now = 100.012099266
        node.latest_joint_stamp_monotonic_s = clock.now

    node.joint_feedback_condition = ScriptedCondition(deliver)
    monkeypatch.setattr(probe.time, 'monotonic', lambda: clock.now)
    assert node.wait_for_joint_feedback(timeout_s=.002)
    assert {k: node.joint_feedback_diagnostics[k] for k in
            ('waitCount','waitRecoveries','waitTimeouts')} == dict(waitCount=1, waitRecoveries=1, waitTimeouts=0)
    assert node.joint_feedback_diagnostics['lastWait']['recovered']
    assert node.joint_feedback_condition.timeouts[0] == pytest.approx(.002)


@pytest.mark.parametrize('age_s,timeout_s', [(.011, .001), (.014, .005)])
def test_missing_callback_expires_within_tick_and_receipt_gap_budget(monkeypatch, age_s, timeout_s):
    node, clock = receiver(age_s)
    initial_stamp = node.latest_joint_stamp_monotonic_s

    def expire(timeout):
        clock.now += timeout + 1e-8

    node.joint_feedback_condition = ScriptedCondition(expire)
    monkeypatch.setattr(probe.time, 'monotonic', lambda: clock.now)
    assert not node.wait_for_joint_feedback(timeout_s=timeout_s)
    assert node.latest_joint_stamp_monotonic_s == initial_stamp
    assert node.joint_feedback_condition.timeouts[0] <= timeout_s + 1e-12
    assert clock.now <= 100.0150001
    assert node.joint_feedback_diagnostics['waitTimeouts'] == 1


@pytest.mark.parametrize('age_s,expected', [(.001, True), (.016, False), (.1, False)])
def test_fresh_feedback_and_long_interruptions_never_wait(monkeypatch, age_s, expected):
    node, clock = receiver(age_s)
    node.joint_feedback_condition = ScriptedCondition(lambda _: pytest.fail('unexpected wait'))
    monkeypatch.setattr(probe.time, 'monotonic', lambda: clock.now)
    assert node.wait_for_joint_feedback() is expected
    assert node.joint_feedback_diagnostics['waitCount'] == 0


def test_spurious_wakeup_does_not_freshen_old_feedback(monkeypatch):
    node, clock = receiver(.011)
    wakes = []

    def wake(timeout):
        wakes.append(timeout)
        clock.now += .0002 if len(wakes) == 1 else timeout + 1e-8

    node.joint_feedback_condition = ScriptedCondition(wake)
    monkeypatch.setattr(probe.time, 'monotonic', lambda: clock.now)
    assert not node.wait_for_joint_feedback(timeout_s=.001)
    assert len(wakes) == 2
    assert node.latest_joint_stamp_monotonic_s == 100.0


def test_late_callback_cannot_extend_execution_deadline(monkeypatch):
    node, clock = receiver(.011)

    def late_delivery(timeout):
        clock.now += timeout + .001
        node.latest_joint_stamp_monotonic_s = clock.now

    node.joint_feedback_condition = ScriptedCondition(late_delivery)
    monkeypatch.setattr(probe.time, 'monotonic', lambda: clock.now)
    assert not node.wait_for_joint_feedback(timeout_s=.001)
    assert node.joint_feedback_diagnostics['waitTimeouts'] == 1


@pytest.mark.parametrize('remaining_s,reserve_s,expected', [
    (.001, .0005, .0005), (.002, .003, 0.), (.02, .003, .005)])
def test_feedback_wait_uses_only_remaining_execution_budget(monkeypatch, remaining_s, reserve_s, expected):
    calls = []
    p = probe.PolicySafetyProbe.__new__(probe.PolicySafetyProbe)
    p.ros_bridge = SimpleNamespace(wait_for_joint_feedback=lambda **kw: calls.append(kw) or True)
    p.control_tick_deadline_perf = 50.0 + remaining_s
    monkeypatch.setattr(probe.time, 'perf_counter', lambda: 50.0)
    assert p._wait_for_fplc_joint_feedback(reserve_s=reserve_s)
    assert calls[0]['timeout_s'] == pytest.approx(expected)


def test_joint_subscription_has_dedicated_node_executor_and_thread(monkeypatch):
    nodes, executors, threads = [], [], []

    class Node:
        def __init__(self, *args, **kw):
            self.context = object()
            self.subscriptions = []
            self._joint_state_callback = lambda msg: None
            nodes.append(self)

        def create_subscription(self, message_type, topic, callback, depth):
            self.subscriptions.append((topic, callback, depth))

    class Executor:
        def __init__(self, **kw):
            self.nodes = []
            self.kwargs = kw
            executors.append(self)

        def add_node(self, node):
            self.nodes.append(node)

    class Thread:
        def __init__(self, **kw):
            self.kwargs = kw
            threads.append(self)

        def start(self):
            pass

    monkeypatch.setattr(probe, '_ROS_AVAILABLE', True)
    monkeypatch.setattr(probe, 'rclpy', SimpleNamespace(ok=lambda: True))
    monkeypatch.setattr(probe, 'Node', Node)
    monkeypatch.setattr(probe, 'RosBridgeNode', Node)
    monkeypatch.setattr(probe, 'MultiThreadedExecutor', Executor)
    monkeypatch.setattr(probe, 'SingleThreadedExecutor', Executor)
    monkeypatch.setattr(probe.threading, 'Thread', Thread)
    monkeypatch.setattr(probe, 'QoSProfile', lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(probe, 'HistoryPolicy', SimpleNamespace(KEEP_LAST='keep_last'))
    monkeypatch.setattr(probe, 'ReliabilityPolicy', SimpleNamespace(BEST_EFFORT='best_effort'))
    bridge = probe.RosBridge(SimpleNamespace(ros_bridge=True, joint_state_topic='/joint_states'))
    assert bridge.executor.nodes == [bridge.node]
    assert bridge.joint_executor.nodes == [bridge.joint_node]
    assert bridge.joint_node is not bridge.node
    assert bridge.joint_node.subscriptions == [('/joint_states', bridge.node._joint_state_callback,
        SimpleNamespace(depth=1, history='keep_last', reliability='best_effort'))]
    assert bridge.node.subscriptions == []
    assert {t.kwargs['name'] for t in threads} == {
        'policy_probe_ros_spin', 'policy_probe_joint_feedback'}


def test_receiver_scheduling_does_not_change_control_or_general_executor(monkeypatch):
    calls = []
    monkeypatch.setattr(control_runtime, 'configure_joint_feedback_thread',
                        lambda: calls.append('receiver_schedule') or dict(realtimeApplied=True))
    bridge = probe.RosBridge.__new__(probe.RosBridge)
    bridge.node = SimpleNamespace()
    bridge.joint_executor = SimpleNamespace(spin=lambda: calls.append('receiver_spin'))
    bridge._spin_joint_feedback()
    assert calls == ['receiver_schedule', 'receiver_spin']
    assert bridge.node.joint_receiver_runtime['realtimeApplied']


def test_cached_joint_indices_follow_reordered_whole_robot_messages(monkeypatch):
    node, clock = receiver(.005)
    node.lock = threading.Lock()
    node.latest_joint_sample_stamp_s = node.latest_joint_header_stamp_s = None
    node.latest_joint_dq = node.latest_joint_ddq = None
    node.joint_feedback_diagnostics.update(
        callbackCount=0, lastReceiptIntervalS=None, maximumReceiptIntervalS=0.,
        receiptIntervalsOver10ms=0)
    notifications = []
    node.joint_feedback_condition = SimpleNamespace(notify_all=lambda: notifications.append(True))
    monkeypatch.setattr(probe.time, 'monotonic', lambda: clock.now)
    values = {name: i*.01 for i, name in enumerate(probe.RIGHT_ARM_JOINTS)}
    for reverse in (False, True):
        names = ['uncontrolled_joint'] + list(probe.RIGHT_ARM_JOINTS)
        if reverse:
            names.reverse()
        message = SimpleNamespace(
            name=names, position=[values.get(name, float('nan')) for name in names],
            velocity=[0.] * len(names),
            header=SimpleNamespace(stamp=SimpleNamespace(sec=1, nanosec=5_000_000 if reverse else 0)))
        node._joint_state_callback(message)
        np.testing.assert_allclose(node.latest_joint_q, list(values.values()))
        clock.now += .005
    assert notifications == [True, True]
    assert node.joint_feedback_diagnostics['callbackCount'] == 2
    assert node.joint_feedback_diagnostics['lastReceiptIntervalS'] == pytest.approx(.005)


@pytest.mark.parametrize('limit_ms', [10, 15, 20])
def test_configured_limit_accepts_samples_without_waiting_or_changing_timestamps(monkeypatch, limit_ms):
    limit = joint_feedback_max_age_s(limit_ms)
    monkeypatch.setattr(probe, 'FPLC_JOINT_FEEDBACK_MAX_AGE_S', limit)
    monkeypatch.setattr(probe, 'FPLC_JOINT_FEEDBACK_MAX_GAP_S', limit+.005)
    node, clock = receiver(limit-.000001)
    stamp = node.latest_joint_stamp_monotonic_s
    node.joint_feedback_condition = ScriptedCondition(lambda _: pytest.fail('accepted data must not block the control tick'))
    monkeypatch.setattr(probe.time, 'monotonic', lambda: clock.now)
    assert node.wait_for_joint_feedback(timeout_s=.005)
    assert node.latest_joint_stamp_monotonic_s == stamp
    assert node.joint_feedback_diagnostics['waitCount'] == 0


@pytest.mark.parametrize('limit_ms', [10, 15, 20])
def test_configured_limit_still_expires_without_new_callback(monkeypatch, limit_ms):
    limit = joint_feedback_max_age_s(limit_ms)
    monkeypatch.setattr(probe, 'FPLC_JOINT_FEEDBACK_MAX_AGE_S', limit)
    monkeypatch.setattr(probe, 'FPLC_JOINT_FEEDBACK_MAX_GAP_S', limit+.005)
    node, clock = receiver(limit+.001)
    node.joint_feedback_condition = ScriptedCondition(lambda timeout: setattr(clock, 'now', clock.now+timeout+1e-8))
    monkeypatch.setattr(probe.time, 'monotonic', lambda: clock.now)
    assert not node.wait_for_joint_feedback(timeout_s=.0005)
    wait = node.joint_feedback_diagnostics['lastWait']
    assert wait['budgetS'] == pytest.approx(.0005)
    assert wait['elapsedS'] < .000501
    assert not wait['recovered']


@pytest.mark.parametrize('invalid', ['nan', 'inf', 'invalid', 0, 9, 21, 100])
def test_feedback_limit_rejects_unbounded_or_invalid_configuration(invalid):
    with pytest.raises(ValueError, match='between 10 and 20'):
        joint_feedback_max_age_s(invalid)


def test_fplc_age_configuration_does_not_relax_gpu0_feedback(monkeypatch):
    monkeypatch.setattr(probe, 'FPLC_JOINT_FEEDBACK_MAX_AGE_S', .020)
    p = probe.PolicySafetyProbe.__new__(probe.PolicySafetyProbe)
    p.active_policy_profile = 'gpu0_qvel_real_rmp_v85'
    p.ros_bridge = SimpleNamespace(get_joint_feedback_coherent_with_timing=lambda: (
        np.zeros(7), np.zeros(7), .012, 'joint_state', 1., dict(jointSourceSequence=1)))
    p._joint_feedback_for_policy()
    assert p.policy_joint_sample_error == 'feedback_stale:0.012000s>0.010000s'


def test_parent_acknowledges_startup_config_before_accepting_http_commands():
    from queue import Queue
    p = probe.ControlSubprocessProxy.__new__(probe.ControlSubprocessProxy)
    p.latest_config = {'ready': True}
    p.command_q = Queue(maxsize=1)
    p._wait_for_initial_config()
    assert p.command_q.get_nowait() == {'kind': 'config_ack'}


def test_hold_attempt_and_success_are_distinct_and_failure_is_not_retried():
    p = probe.PolicySafetyProbe.__new__(probe.PolicySafetyProbe)
    p.compensation_was_enabled = False
    p.done_failsafe_reason = 'test'
    p.hard_workspace_stop_hold_sent = False
    attempts = []
    p._publish_measured_stop_hold = lambda: attempts.append('hold') or False
    for _ in range(2):
        assert not p._stop_hard_workspace_command_stream()
    assert attempts == ['hold']
    assert p._hard_workspace_stop_payload()['holdAttempted']
    assert not p._hard_workspace_stop_payload()['holdSent']
