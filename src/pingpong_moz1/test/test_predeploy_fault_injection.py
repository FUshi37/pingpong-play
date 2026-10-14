"""Fault injection into the real probe boundary, with no ROS actuator connection."""
import dataclasses
from types import SimpleNamespace
import numpy as np
import pytest
from test_simcode9_fplc_contract import probe, _policy_options, _predict, probe_module


def bridge_for(probe, q=None, dq=None, age=.001, issued=None):
    q = probe.controller.native_reset_arm_q.copy() if q is None else q
    dq = np.zeros(7) if dq is None else dq
    issued = [] if issued is None else issued
    return SimpleNamespace(
        get_joint_feedback_with_age=lambda: (q.copy(), dq.copy(), age, 'injected'),
        get_joint_feedback_coherent=lambda: (q.copy(), dq.copy(), age, 'injected', probe.now_s),
        get_joint_feedback=lambda: (q.copy(), dq.copy()),
        get_cart_feedback=lambda: (None, None, None),
        publish_right_arm_command=lambda value: issued.append(np.array(value))), issued


@pytest.mark.parametrize('fault', ['q_nan', 'q_inf', 'q_short', 'dq_nan', 'dq_inf', 'dq_short',
                                 'age_negative', 'age_nan', 'age_inf', 'age_old'])
@pytest.mark.parametrize('operation', ['arm', 'normal_publish', 'disable_hold', 'hard_stop_hold'])
def test_predeploy_bad_feedback_never_arms_or_publishes(probe, fault, operation):
    q = probe.controller.native_reset_arm_q.copy()
    dq = np.zeros(7)
    age = .001
    if fault.startswith('q_'):
        if fault == 'q_short':
            q = q[:-1]
        else:
            q[1] = np.nan if fault == 'q_nan' else np.inf
    elif fault.startswith('dq_'):
        if fault == 'dq_short':
            dq = dq[:-1]
        else:
            dq[1] = np.nan if fault == 'dq_nan' else np.inf
    else:
        age = {'age_negative': -.001, 'age_nan': np.nan, 'age_inf': np.inf, 'age_old': .01001}[fault]
    target = _predict(probe.controller, arm_q=probe.controller.native_reset_arm_q.copy())
    probe.ros_bridge, issued = bridge_for(probe, q=q, dq=dq, age=age)
    options = dataclasses.replace(_policy_options(probe), publish_commands=True)
    probe.runtime_options = options
    probe.command_stream_active = operation != 'arm'
    if operation == 'arm':
        assert not probe._arm_command_stream_from_feedback()
    elif operation == 'normal_publish':
        assert not probe._publish_control_command(target, options)
        assert probe.publish_rearm_required
    else:
        method = '_publish_hold_on_command_disable' if operation == 'disable_hold' else '_publish_measured_stop_hold'
        assert not getattr(probe, method)()
        assert not probe.command_stream_active
    assert not issued, 'Invalid atomic feedback must not supply any new position target'


def policy_tick(probe, monkeypatch):
    probe.ros_bridge, issued = bridge_for(probe)
    probe.command_stream_active = True
    probe.has_captured = probe.has_released = True
    options = dataclasses.replace(_policy_options(probe), publish_commands=True, use_ros_state=True,
                                  enable_done_monitoring=True)
    probe.runtime_options = options
    # Any hidden limiter access is a regression, including the fault path.
    class Forbidden:
        def __getattribute__(self, name):
            pytest.fail('fault path accessed safety_limiter.'+name)
        def __setattr__(self, name, value):
            pytest.fail('fault path wrote safety_limiter.'+name)
    probe.safety_limiter = Forbidden()
    return options, issued


@pytest.mark.parametrize('output', [[np.nan]*7, [np.inf]*7, [0.]*6, [0.]*8])
def test_predeploy_bad_actor_output_disarms(probe, monkeypatch, output):
    options, issued = policy_tick(probe, monkeypatch)
    monkeypatch.setattr(probe.controller, '_policy_action', lambda obs: np.array(output, np.float32))
    result = probe._run_one_control_step(np.array([0., -.49, 1.14]), np.array([0., 0., -.3]),
                                         True, .001, options)
    assert not result[4] and result[0] == 'policy_feedback_hold'
    assert probe.publish_rearm_required and not probe.runtime_options.publish_commands
    assert probe.controller._fplc_requires_measured_reset
    for command in issued:
        np.testing.assert_array_equal(command, probe.controller.native_reset_arm_q)


@pytest.mark.parametrize('error_type', [ValueError, RuntimeError, np.linalg.LinAlgError])
def test_predeploy_inference_exception_disarms(probe, monkeypatch, error_type):
    options, issued = policy_tick(probe, monkeypatch)
    def bad_actor(obs):
        raise error_type('injected inference failure')
    monkeypatch.setattr(probe.controller, '_policy_action', bad_actor)
    result = probe._run_one_control_step(np.array([0., -.49, 1.14]), np.array([0., 0., -.3]),
                                         True, .001, options)
    assert not result[4] and not probe.runtime_options.publish_commands
    assert probe.publish_rearm_reason == 'fplc_inference_fault'
    assert probe.controller._fplc_requires_measured_reset
    assert not probe.command_stream_active


@pytest.mark.parametrize('error_type', [KeyError, AttributeError, OverflowError])
def test_predeploy_backend_exception_disarms(probe, monkeypatch, error_type):
    options, issued = policy_tick(probe, monkeypatch)
    def bad_tick():
        raise error_type('injected backend failure')
    monkeypatch.setattr(probe, '_run_backend_control_tick', bad_tick)
    probe._run_backend_control_once()
    assert probe.control_loop_last_error is not None
    assert not probe.runtime_options.publish_commands
    assert probe.publish_rearm_required
    assert probe.controller._fplc_requires_measured_reset
    assert not probe.command_stream_active
    assert probe._hard_workspace_stop_active()
    # No subsequent old-tick packet can recover the control stream.
    assert not probe._publish_control_command(probe.controller.current_arm_cmd_q(), options)
    assert len(issued) <= 1


@pytest.mark.parametrize('age', [-.001, np.nan, np.inf, .0101])
def test_predeploy_invalid_atomic_age_never_reaches_actor(probe, monkeypatch, age):
    options, _issued = policy_tick(probe, monkeypatch)
    q = probe.controller.native_reset_arm_q.copy()
    monkeypatch.setattr(probe, '_joint_feedback_for_policy',
                        lambda: (q, np.zeros(7), age, 'bad_age', 1., {}))
    monkeypatch.setattr(probe.controller, '_policy_action',
                        lambda obs: pytest.fail('invalid atomic age reached actor'))
    result = probe._run_one_control_step(np.array([0., -.49, 1.14]), np.array([0., 0., -.3]),
                                         True, .001, options)
    assert result[0] == 'policy_feedback_hold' and not result[4]
    assert not probe.runtime_options.publish_commands
    assert probe.controller._fplc_requires_measured_reset


@pytest.mark.parametrize('key,value', [('ball', [np.nan, -.49, 1.14]),
    ('ball', [np.inf, -.49, 1.14]), ('ball', [0., -.49]),
    ('velocity', [0., None, -.3]), ('velocity', 'broken')])
def test_predeploy_invalid_web_ball_is_invalid_not_a_valid_default(probe, key, value):
    # Deferred backend path is the production web control path. There is no
    # thread/ROS bridge in this fixture, so the malformed input cannot actuate.
    probe.cfg.backend_control_loop = True
    probe.step({'ballSource': 'drag', 'ball': [0., -.49, 1.14], 'velocity': [0., 0., -.3],
                'valid': True, key: value, 'publishCommands': False})
    assert not probe.web_drag_valid


def test_predeploy_observer_request_does_not_refresh_owner_lease(probe):
    probe.command_owner_session_id = 'writer'
    probe.command_publish_lease_deadline_wall = 123.
    probe.runtime_options = dataclasses.replace(_policy_options(probe), publish_commands=True)
    probe.latest_snapshot_payload = {'runtime': {'publishCommands': True}, 'mode': 'policy'}
    result = probe.step({'clientSessionId': 'observer', 'publishCommands': False})
    assert result['commandRequestIgnored']
    assert not result['commandOwnership']['accepted']
    assert probe.command_owner_session_id == 'writer'
    assert probe.command_publish_lease_deadline_wall == 123.
    assert probe.runtime_options.publish_commands
