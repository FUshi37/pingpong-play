import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from pingpong_controller.camera_observation_continuity import CameraObservationContinuity


def advance(state, now, **kwargs):
    options = dict(enabled=True, source_key=('fplc', 'camera', 1), valid=True,
        observed=True, ready=True, reason='measured', measurement_stamp=10.+now)
    options.update(kwargs)
    return state.step(now, **options)


@pytest.mark.parametrize('reason', sorted(CameraObservationContinuity.TRANSIENT_REASONS))
def test_known_transient_never_renews_measurement_age_and_resumes_on_new_real_sample(reason):
    state = CameraObservationContinuity()
    advance(state, 0.)
    for t in np.arange(.005, .151, .005):
        info = advance(state, float(t), valid=False, observed=False, reason=reason)
        assert info['action'] == 'waiting' and not info['stateValid']
        assert state.last_observed_stamp == 10.
    info = advance(state, .155)
    assert info['action'] == 'resumed'


def test_prediction_or_repeated_precontact_sample_cannot_resume_a_wait():
    state = CameraObservationContinuity()
    advance(state, 0.)
    advance(state, .005, valid=False, observed=False, reason='contact_pending')
    assert advance(state, .010, observed=False)['action'] == 'waiting'
    assert advance(state, .015, measurement_stamp=10.)['action'] == 'waiting'
    assert advance(state, .020)['action'] == 'resumed'


def test_wait_budget_does_not_restart_on_repeated_missing_frames():
    state = CameraObservationContinuity()
    advance(state, 0.)
    for t in np.arange(.005, .206, .005):
        info = advance(state, float(t), valid=False, observed=False, reason='prediction_timeout')
        assert info['action'] == 'waiting'
    assert advance(state, .211, valid=False, observed=False, reason='prediction_timeout')['action'] == 'fault'
    assert advance(state, .215, valid=False, observed=False, reason='prediction_timeout')['action'] == 'fault'


@pytest.mark.parametrize('change', [dict(ready=False), dict(reason='malformed'), dict(observed=False, valid=True)])
def test_no_cold_start_recovery_without_an_acquired_observation(change):
    state = CameraObservationContinuity()
    options = dict(valid=False, observed=False, reason='prediction_timeout')
    options.update(change)
    first = advance(state, 0., **options)
    if change == dict(observed=False, valid=True):
        assert first['action'] == 'running'
        assert advance(state, .005, valid=False, observed=False, reason='prediction_timeout')['action'] == 'fault'
    else:
        assert first['action'] == 'fault'


def test_unknown_fault_and_clock_rollback_are_not_transient_misses():
    state = CameraObservationContinuity()
    advance(state, 1.)
    assert advance(state, .9)['action'] == 'fault'
    state.reset()
    advance(state, 0.)
    advance(state, .005, valid=False, observed=False, reason='prediction_timeout')
    assert advance(state, .010, valid=False, observed=False, reason='malformed')['action'] == 'fault'


def test_source_change_and_disabled_path_cannot_reuse_acquisition():
    state = CameraObservationContinuity()
    advance(state, 0.)
    assert advance(state, .005, source_key=('fplc', 'camera', 2), valid=False,
                   observed=False, reason='prediction_timeout')['action'] == 'fault'
    assert advance(state, .010, enabled=False)['action'] == 'inactive'
    assert not state.acquired


from test_simcode9_fplc_contract import probe, probe_module, _policy_options
from pingpong_controller.ball_observation_model import BallObservationProcessor


def test_online_scalar_packet_preserves_small_and_signed_covariance_entries():
    raw = dict(observation_position_variance=3.4e-8,
               observation_velocity_variance=7.87654321e-6,
               observation_cross_covariance=-8.5e-8,
               observation_position_x_m=.123456789)
    packet = probe_module._camera_measurement_payload(raw)
    for key in ('observation_position_variance', 'observation_velocity_variance', 'observation_cross_covariance'):
        assert packet[key] == raw[key]
    assert packet['observation_position_x_m'] == .123457


@pytest.mark.parametrize('rate', [90, 120, 200])
def test_adapter_checks_true_range_age_before_an_actor_can_receive_it(probe, rate):
    options = dataclasses.replace(_policy_options(probe), ball_source='camera',
                                  ball_observation_rate_hz=float(rate))
    probe.last_high_ball_occlusion_info = dict(model='quality_bearing_range_flight',
        reason='bearing_update', observationRangeAgeS=.355, positionPredicted=True)
    probe.fplc_direct_ball_input = dict(direct_ball_position_m=np.array([0., -.49, 1.1]),
        direct_ball_age_s=.6, direct_ball_stamp_s=10., ball_observed=False,
        ball_state_already_propagated=True)
    _, _, valid, age = probe._apply_policy_ball_sampling_contract(
        np.array([0., -.49, 1.1]), np.zeros(3), True, 0., options, source_timestamp_s=10.)
    assert not valid and age == pytest.approx(.355)
    assert probe.last_policy_ball_handover_info['state'] == 'expired_range'
    assert 'ball_history_position_m' not in probe.fplc_direct_ball_input


@pytest.mark.parametrize('rate', [90, 120, 200])
def test_quality_predictor_expiry_never_turns_into_a_valid_held_observation(probe, rate):
    options = dataclasses.replace(_policy_options(probe), ball_source='camera',
                                  ball_observation_rate_hz=float(rate))
    probe.last_high_ball_occlusion_info = dict(model='quality_bearing_range_flight',
        reason='measured', observationRangeAgeS=.001, positionPredicted=False)
    probe.fplc_direct_ball_input = dict(ball_observed=True, ball_state_already_propagated=False,
        direct_ball_position_m=np.array([0., -.49, 1.1]), direct_ball_age_s=.001,
        direct_ball_stamp_s=10.)
    p = np.array([0., -.49, 1.1])
    assert probe._apply_policy_ball_sampling_contract(p, np.zeros(3), True, .001, options,
                                                     source_timestamp_s=10.)[2]
    probe.last_high_ball_occlusion_info.update(reason='prediction_timeout', observationRangeAgeS=.4)
    probe.fplc_direct_ball_input.update(ball_observed=False)
    assert not probe._apply_policy_ball_sampling_contract(p, np.zeros(3), False, .6, options,
                                                         source_timestamp_s=10.4)[2]


@pytest.mark.parametrize('rate', [90, 120, 200])
def test_real_control_tick_keeps_send_during_contact_warmup_and_resumes(probe, monkeypatch, rate):
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(probe_module.time, 'monotonic', lambda: clock.now)
    probe.cfg.ball_observation_rate_hz = float(rate)
    q = probe.controller.native_reset_arm_q.copy()
    issued, records = [], []
    probe.ros_bridge = SimpleNamespace(now_sec=lambda: 10.+clock.now,
        get_joint_feedback_with_age=lambda: (q.copy(), np.zeros(7), .001, 'fixture'),
        get_joint_feedback_coherent=lambda: (q.copy(), np.zeros(7), .001, 'fixture', 10.+clock.now),
        get_joint_feedback=lambda: (q.copy(), np.zeros(7)),
        get_cart_feedback=lambda: (None, None, None),
        publish_right_arm_command=lambda target: issued.append(np.array(target)))
    probe.command_stream_active = True
    probe.has_captured = probe.has_released = True
    options = dataclasses.replace(_policy_options(probe), publish_commands=True, use_ros_state=True,
        ball_source='camera', high_ball_occlusion_prediction=True, sensor_depth_anomaly_prediction=True,
        enable_compensation=False)
    probe.runtime_options = options
    offset = np.array([probe.cfg.sim_cartesian_x_offset_m, probe.cfg.sim_cartesian_y_offset_m, probe.cfg.sim_base_height_m])
    processor = BallObservationProcessor(offset)
    position = np.array([0., -.49, 1.06])
    def measured(stamp):
        obs = dict(detected=True, position=(position-offset)*1000., velocity=np.zeros(3),
            raw_measurement=dict(ball_observation_version=1, velocity_motion_state='supported',
                                 velocity_ready=True, range_sigma_mm=3.))
        processor.process(obs, stamp)
        return obs['raw_measurement']
    for stamp in np.arange(10)/90.+9.899:
        raw = measured(stamp)
    camera = SimpleNamespace(raw=raw, detected=True, stamp=9.999)
    monkeypatch.setattr(probe, '_select_ball_source', lambda *_a, **_k: (
        position.copy(), np.zeros(3), camera.detected, 'camera', {}, camera.stamp, camera.raw))
    monkeypatch.setattr(probe, '_record_control_sample', lambda **r: records.append(r))
    monkeypatch.setattr(probe, '_step_sim_mirror', lambda *_a, **_k: None)
    def tick():
        probe._run_control_tick_locked(np.zeros(3), np.zeros(3), True, 0., probe.runtime_options, make_snapshot=False)
    tick()
    assert records[-1]['policy_executed'] and records[-1]['command_published']
    for t in [.005, .010]:
        clock.now = t
        camera.raw = dict(raw, observation_velocity_valid=False, observation_position_valid=False,
            observation_model_reason='contact_pending')
        camera.detected = False
        tick()
        assert records[-1]['mode'] == 'policy_observation_wait'
        assert records[-1]['command_published'] and not records[-1]['control_valid']
        assert probe.runtime_options.publish_commands and not probe.publish_rearm_required
        np.testing.assert_allclose(issued[-1], q)
    clock.now = .015
    camera.stamp, camera.detected = 10.014, True
    camera.raw = measured(camera.stamp)
    tick()
    assert records[-1]['policy_executed'] and records[-1]['command_published']
    assert probe.camera_observation_continuity.last_info['action'] == 'resumed'
    clock.now = .020
    camera.detected = False
    camera.raw = dict(camera.raw, observation_velocity_valid=False, observation_position_valid=False,
                      observation_model_reason='prediction_timeout')
    tick()
    clock.now = .225
    tick()
    assert probe.publish_rearm_required and not probe.runtime_options.publish_commands
    assert records[-1]['mode'] == 'policy_ball_state_hold'
