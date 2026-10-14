"""Bounded detector dropout and command-latch regressions; no ROS actuation."""
import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest

from test_simcode9_fplc_contract import controller, probe, _policy_options, probe_module
from pingpong_controller.fplc_deployment import FPLC_PROFILES, REPO_ROOT
from pingpong_controller.fplc_policy import FPLCPolicyController
from pingpong_controller.sim_mirror import RealBallSimMirror, SimMirrorConfig, sim_mirror_config_for_policy_profile
from pingpong_controller.fplc_deployment import fplc_assets


@pytest.fixture
def holder(monkeypatch):
    p = probe_module.PolicySafetyProbe.__new__(probe_module.PolicySafetyProbe)
    p.cfg = SimpleNamespace(control_dt=.005, policy_ball_handover_mode='legacy_force_fresh')
    p._reset_policy_ball_sampling_contract()
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(probe_module.time, 'monotonic', lambda: clock.now)
    return p, clock


def sample(p, *, profile='fplc_gpu1_v172', source='camera', valid=True,
           age=.01, stamp=1., pos=(.001, -.49, 1.06), vel=(0., 0., .3),
           observed=None, propagated=False, hold=True):
    p.fplc_direct_ball_input = {
        'ball_observed': valid if observed is None else observed,
        'ball_state_already_propagated': propagated,
    }
    options = SimpleNamespace(policy_profile=profile, control_mode='policy',
                              ball_source=source, use_ball_estimator=False,
                              enable_ball_dropout_reference_hold=hold)
    return p._apply_policy_ball_sampling_contract(
        np.asarray(pos), np.asarray(vel), valid, age, options,
        source_timestamp_s=stamp)


@pytest.mark.parametrize('profile', FPLC_PROFILES)
@pytest.mark.parametrize('source', ['camera', 'mocap'])
def test_detector_miss_holds_trusted_state_without_fabricating_observation(holder, profile, source):
    p, clock = holder
    trusted = sample(p, profile=profile, source=source)
    clock.now = .005
    held = sample(p, profile=profile, source=source, valid=False, age=.6,
                  stamp=1.011, pos=(99., 99., 99.), vel=(99., 99., 99.))
    np.testing.assert_array_equal(held[0], trusted[0])
    np.testing.assert_array_equal(held[1], trusted[1])
    assert held[2] and held[3] == pytest.approx(.015)
    assert not p.fplc_direct_ball_input['ball_observed']
    assert p.last_policy_ball_handover_info['state'] == 'held_dropout'
    assert not p.last_policy_ball_handover_info['sampleAccepted']
    assert p.policy_ball_sample_source_stamp_s == 1.
    clock.now = .012
    recovered = sample(p, profile=profile, source=source, age=.002, stamp=1.022,
                       pos=(.002, -.49, 1.07))
    assert recovered[2] and recovered[3] == .002
    assert recovered[0][0] == np.float32(.002)


def test_continuous_loss_expires_even_when_the_wall_clock_does_not_advance(holder):
    p, _clock = holder
    sample(p, age=0.)
    for _ in range(69):
        held = sample(p, valid=False, age=.6, stamp=2.)
        assert held[2]
    sample(p, valid=False, age=.6, stamp=2.)
    expired = sample(p, valid=False, age=.6, stamp=2.)
    assert not expired[2] and expired[3] > .35
    assert not p.policy_ball_sample_has_valid


def test_wall_clock_stall_cannot_extend_the_350ms_budget(holder):
    p, clock = holder
    sample(p)
    clock.now = .4
    expired = sample(p, valid=False, age=.6)
    assert not expired[2] and expired[3] == pytest.approx(.41)


@pytest.mark.parametrize('age', [-.001, np.nan, np.inf, .35001, .6])
def test_invalid_or_expired_observation_is_not_treated_as_a_dropout(holder, age):
    p, _clock = holder
    sample(p)
    assert not sample(p, valid=True, age=age)[2]
    assert not sample(p, valid=False)[2]


@pytest.mark.parametrize('field,value', [('pos', (np.nan, -.49, 1.06)),
                                       ('vel', (0., np.inf, .3)), ('pos', (0., -.49))])
def test_malformed_observation_cannot_reuse_the_hold_cache(holder, field, value):
    p, _clock = holder
    sample(p)
    assert not sample(p, **{field: value})[2]
    assert not sample(p, valid=False)[2]


def test_uninitialized_reset_profile_and_source_changes_require_a_new_observation(holder):
    p, _clock = holder
    assert not sample(p, valid=False)[2]
    sample(p)
    p._reset_policy_ball_sampling_contract()
    assert not sample(p, valid=False)[2]
    sample(p)
    assert not sample(p, profile='fplc_gpu0_v161', valid=False)[2]
    sample(p)
    assert not sample(p, source='mocap', valid=False)[2]


def test_predictor_expiry_does_not_gain_another_350ms_hold(holder):
    p, _clock = holder
    sample(p)
    assert sample(p, observed=False, propagated=True, age=0.)[2]
    assert not sample(p, valid=False, observed=False, propagated=False)[2]


@pytest.mark.parametrize('hold', [True, False])
def test_reference_mode_applies_to_valid_prediction_without_extending_prediction_expiry(holder, hold):
    p, _clock = holder
    sample(p)
    assert sample(p, observed=False, propagated=True, age=0., hold=hold)[2]
    assert p.fplc_direct_ball_input['hold_reference_on_dropout'] == hold
    assert not p.policy_ball_sample_has_valid
    assert not sample(p, valid=False, observed=False, propagated=False, hold=hold)[2]


def test_explicit_invalid_drag_input_still_stops_immediately(holder):
    p, _clock = holder
    sample(p, source='drag')
    assert not sample(p, source='drag', valid=False)[2]


def test_sensor_observation_without_a_known_age_cannot_seed_a_hold(holder):
    p, _clock = holder
    assert not sample(p, age=None)[2]
    assert not sample(p, valid=False)[2]


def test_valid_stream_preserves_actor_history_projection_and_commands(controller, holder):
    p, clock = holder
    reference = FPLCPolicyController(observation_profile=controller.fplc_profile)
    q = controller.native_reset_arm_q.copy()
    for index in range(30):
        clock.now = index * .005
        pos = np.array([.002*np.sin(clock.now), -.49, 1.06+.01*np.sin(10*clock.now)], np.float32)
        vel = np.array([.002*np.cos(clock.now), 0., .1*np.cos(10*clock.now)], np.float32)
        age = .002 + (index % 3)*.005
        stamp = 1. + (index // 3)*.015
        kwargs = dict(arm_q=q, arm_dq=np.zeros(7), direct_ball_position_m=pos,
                      direct_ball_age_s=age, direct_ball_stamp_s=stamp, ball_observed=True)
        reference.predict(pos, vel, ball_valid=True, ball_obs_age_s=age, **kwargs)
        held_p, held_v, valid, held_age = sample(p, profile=controller.fplc_profile,
                                               age=age, stamp=stamp, pos=pos, vel=vel)
        controller.predict(held_p, held_v, ball_valid=valid, ball_obs_age_s=held_age, **kwargs)
        for field in ('last_obs', 'last_action', 'arm_cmd_q', 'arm_cmd_qvel',
                      'prev_action', 'ball_history', 'racket_history', 'age_history'):
            np.testing.assert_array_equal(getattr(controller, field), getattr(reference, field))
        assert controller.last_projection == reference.last_projection


def test_both_actor_layouts_keep_the_direct_observed_flag_false_during_dropout(controller, holder):
    p, clock = holder
    q = controller.native_reset_arm_q.copy()
    for index in range(5):
        clock.now = index*.005
        observed = index == 0
        pos, vel, valid, age = sample(p, profile=controller.fplc_profile, valid=observed)
        controller.predict(pos, vel, ball_valid=valid, ball_obs_age_s=age, arm_q=q,
                           arm_dq=np.zeros(7), ball_observed=observed,
                           direct_ball_position_m=pos, direct_ball_age_s=age,
                           direct_ball_stamp_s=1.,
                           hold_reference_on_dropout=p.fplc_direct_ball_input['hold_reference_on_dropout'])
    observed_index = 71 if controller._expected_obs_dim == 72 else 135
    assert controller.last_obs[observed_index] == 0.
    assert np.all(controller.age_history[0] >= .01)


def test_dropout_never_calls_actor_or_moves_reference_and_waits_for_sampled_recovery(controller, holder, monkeypatch):
    p, clock = holder
    q = controller.native_reset_arm_q.copy()
    def tick(index, observed):
        clock.now = index*.005
        pos, vel, valid, age = sample(p, profile=controller.fplc_profile, valid=observed)
        return controller.predict(pos, vel, ball_valid=valid, ball_obs_age_s=age, arm_q=q,
                                  arm_dq=np.zeros(7), ball_observed=observed,
                                  direct_ball_position_m=pos, direct_ball_age_s=age,
                                  direct_ball_stamp_s=1.+clock.now,
                                  hold_reference_on_dropout=p.fplc_direct_ball_input['hold_reference_on_dropout'])
    tick(0, True)
    target = controller.arm_cmd_q.copy()
    original_actor = controller._policy_action
    monkeypatch.setattr(controller, '_policy_action', lambda _obs: pytest.fail('actor ran during dropout'))
    for index in [1, 2, 3]:
        np.testing.assert_array_equal(tick(index, False), target)
        np.testing.assert_array_equal(controller.arm_cmd_qvel, np.zeros(7))
        np.testing.assert_array_equal(controller.prev_action, np.zeros(7))
        assert controller.last_projection['ballDropoutHold']
    # Tick4 is not a 90 Hz refresh: a recovered source must not expose the
    # previously missing sampled observed flag to the actor for one more tick.
    np.testing.assert_array_equal(tick(4, True), target)
    assert not controller._direct_observed
    monkeypatch.setattr(controller, '_policy_action', original_actor)
    tick(5, True)
    assert controller._direct_observed and not controller._dropout_reference_hold_active
    assert 'ballDropoutHold' not in controller.last_projection


def test_recorded_182724_single_frame_miss_no_longer_invalidates_ball(holder):
    p, clock = holder
    recording = REPO_ROOT / 'src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings/probe_recording_20261012_182724/control_samples.jsonl'
    if not recording.is_file():
        pytest.skip('Local experiment recording is not part of the repository')
    with recording.open() as handle:
        rows = [json.loads(line) for index, line in enumerate(handle, 1) if 1149 <= index <= 1153]
    for row in rows:
        clock.now = row['wallTimeS'] - rows[0]['wallTimeS']
        result = sample(p, valid=row['valid'], age=row['age'], stamp=row['cameraStampS'],
                        pos=row['observedBall'], vel=row['observedVelocity'])
        assert result[2], row['sourceSeq']
        if not row['rawValid']:
            assert result[3] < .04
            assert not p.fplc_direct_ball_input['ball_observed']


@pytest.mark.parametrize('hold', [True, False])
def test_full_control_path_tolerates_a_miss_then_latches_on_timeout(probe, monkeypatch, hold):
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(probe_module.time, 'monotonic', lambda: clock.now)
    q = probe.controller.native_reset_arm_q.copy()
    issued, records, mirror_inputs = [], [], []
    probe.ros_bridge = SimpleNamespace(
        now_sec=lambda: 10. + clock.now,
        get_joint_feedback_with_age=lambda: (q.copy(), np.zeros(7), .001, 'injected'),
        get_joint_feedback_coherent=lambda: (q.copy(), np.zeros(7), .001, 'injected', 10. + clock.now),
        get_joint_feedback=lambda: (q.copy(), np.zeros(7)),
        get_cart_feedback=lambda: (None, None, None),
        publish_right_arm_command=lambda target: issued.append(np.array(target)),
    )
    probe.command_stream_active = True
    probe.has_captured = probe.has_released = True
    options = dataclasses.replace(_policy_options(probe), publish_commands=True, use_ros_state=True,
                                  ball_source='camera', enable_done_monitoring=True,
                                  enable_ball_dropout_reference_hold=hold,
                                  high_ball_occlusion_prediction=False, sensor_depth_anomaly_prediction=False,
                                  camera_reacquisition_smoothing=False, optimized_ball_prediction_initial_state=False)
    probe.runtime_options = options
    detection = SimpleNamespace(valid=True)
    monkeypatch.setattr(probe, '_select_ball_source', lambda *_args, **_kwargs: (
        np.array([0., -.49, 1.06], np.float32), np.array([0., 0., .3], np.float32),
        detection.valid, 'camera', {}, 10. + clock.now - .001, {}))
    monkeypatch.setattr(probe, '_record_control_sample', lambda **record: records.append(record))
    monkeypatch.setattr(probe, '_step_sim_mirror', lambda *args, **kwargs: mirror_inputs.append((args, kwargs)))
    class ForbiddenLimiter:
        def __getattribute__(self, name):
            pytest.fail('FPLC dropout path accessed safety_limiter.' + name)
        def __setattr__(self, name, value):
            pytest.fail('FPLC dropout path wrote safety_limiter.' + name)
    probe.safety_limiter = ForbiddenLimiter()

    def tick():
        probe._run_control_tick_locked(np.zeros(3), np.zeros(3), True, 0.,
                                       probe.runtime_options, make_snapshot=False)

    tick()
    assert records[-1]['policy_executed'] and records[-1]['command_published']
    target_before_loss = probe.controller.arm_cmd_q.copy()
    detection.valid = False
    for now in [.005, .010, .015]:
        clock.now = now
        tick()
        assert not records[-1]['raw_valid'] and records[-1]['control_valid']
        assert records[-1]['policy_executed'] and records[-1]['command_published']
        assert probe.runtime_options.publish_commands and not probe.publish_rearm_required
        assert not probe.fplc_direct_ball_input['ball_observed']
        assert mirror_inputs[-1][1]['ball_valid']
        if hold:
            np.testing.assert_array_equal(issued[-1], target_before_loss)
            assert probe.controller.last_projection['actorExecuted'] is False
        else:
            assert not probe.controller._dropout_reference_hold_active
        assert probe.fplc_direct_ball_input['enable_dropout_reference_hold'] == hold
    assert probe.controller.last_obs[71] == 0.
    clock.now = .020
    detection.valid = True
    tick()
    assert records[-1]['policy_executed'] and probe.runtime_options.publish_commands
    detection.valid = False
    clock.now = .4
    tick()
    assert records[-1]['mode'] == 'policy_ball_state_hold'
    assert probe.publish_rearm_reason == 'policy_ball_state_fault'
    assert not probe.runtime_options.publish_commands
    after_stop = len(issued)
    detection.valid = True
    clock.now = .405
    tick()
    assert len(issued) == after_stop and not records[-1]['command_published']
    assert probe.publish_rearm_required and not probe.runtime_options.publish_commands


def test_synchronous_web_step_also_preserves_short_camera_dropout(probe, monkeypatch):
    clock = SimpleNamespace(now=0.)
    monkeypatch.setattr(probe_module.time, 'monotonic', lambda: clock.now)
    options = dataclasses.replace(_policy_options(probe), publish_commands=False, use_ros_state=False,
                                  ball_source='camera', high_ball_occlusion_prediction=False,
                                  sensor_depth_anomaly_prediction=False, camera_reacquisition_smoothing=False,
                                  optimized_ball_prediction_initial_state=False)
    detection = SimpleNamespace(valid=True)
    records = []
    monkeypatch.setattr(probe, '_runtime_options_from_payload', lambda _payload: options)
    monkeypatch.setattr(probe, '_select_ball_source', lambda *_args, **_kwargs: (
        np.array([0., -.49, 1.06], np.float32), np.array([0., 0., .3], np.float32),
        detection.valid, 'camera', {}, 10. + clock.now - .001, {}))
    monkeypatch.setattr(probe, '_raw_measurement_age_s', lambda valid, *_args, **_kwargs: .001 if valid else .6)
    monkeypatch.setattr(probe, '_record_control_sample', lambda **record: records.append(record))
    monkeypatch.setattr(probe, '_step_sim_mirror', lambda *_args, **_kwargs: {})
    monkeypatch.setattr(probe, 'snapshot', lambda mode, *_args, **_kwargs: {'mode': mode})
    probe.step({'dt': .005})
    assert records[-1]['policy_executed']
    detection.valid = False
    for now in [.005, .010, .015]:
        clock.now = now
        result = probe.step({'dt': .005})
        assert result['mode'] == 'policy'
        assert records[-1]['policy_executed'] and records[-1]['control_valid']
        assert not records[-1]['raw_valid'] and not probe.fplc_direct_ball_input['ball_observed']
    assert probe.controller.last_obs[71] == 0.


@pytest.mark.parametrize('hold', [True, False])
def test_cpu_mirror_uses_the_same_reference_hold_and_observed_flag(hold):
    profile = 'fplc_gpu1_v172'
    model, xml = fplc_assets(profile)
    config = sim_mirror_config_for_policy_profile(SimMirrorConfig(
        model_path=model, policy_xml_path=xml, sim_xml_path=xml, policy_obs_dim=72,
        control_dt=.005, enable_video_frames=False), profile)
    mirror = RealBallSimMirror(config)
    pos, vel = np.array([0., -.49, 1.06], np.float32), np.array([0., 0., .3], np.float32)
    options = SimpleNamespace(policy_observation_scales=(), rl_use_valid_ball_obs_age=True)
    try:
        mirror.step(pos, vel, ball_valid=True, ball_obs_age_s=.001, options=options,
                    direct_ball_input=dict(direct_ball_position_m=pos, direct_ball_age_s=.001,
                                           ball_observed=True, hold_reference_on_dropout=False))
        target = mirror.policy.arm_cmd_q.copy()
        for index in [1, 2, 3]:
            mirror.step(pos, vel, ball_valid=True, ball_obs_age_s=.001+index*.005, options=options,
                        direct_ball_input=dict(direct_ball_position_m=pos, direct_ball_age_s=.6,
                                               ball_observed=False, hold_reference_on_dropout=hold,
                                               enable_dropout_reference_hold=hold))
            if hold:
                np.testing.assert_array_equal(mirror.policy.arm_cmd_q, target)
                assert mirror.policy.last_projection['actorExecuted'] is False
            else:
                assert not mirror.policy._dropout_reference_hold_active
        assert mirror.policy.last_obs[71] == 0.
    finally:
        mirror.close()


def test_web_and_cli_reference_hold_setting_round_trip(probe):
    assert probe.cfg.enable_ball_dropout_reference_hold
    assert probe.browser_config()['policy']['ballDropoutReferenceHoldEnabled']
    assert not probe._runtime_options_from_payload({'enableBallDropoutReferenceHold': False}).enable_ball_dropout_reference_hold
    assert probe._runtime_options_from_payload({'enableBallDropoutReferenceHold': True}).enable_ball_dropout_reference_hold
    parser = probe_module.build_arg_parser()
    args = parser.parse_args(['--policy-profile', 'fplc_gpu1_v172', '--policy-obs-dim', '72',
                              '--disable-ball-dropout-reference-hold'])
    assert not probe_module.config_from_args(args).enable_ball_dropout_reference_hold
    html = probe_module.HTML_TEMPLATE
    assert 'id="enableBallDropoutReferenceHold"' in html
    assert 'enableBallDropoutReferenceHold: flowEls.enableBallDropoutReferenceHold.checked' in html
    assert 'ballDropoutReferenceHoldEnabled !== false' in html


def test_turning_off_reference_hold_does_not_reset_ball_age_or_disable_timeout(holder):
    p, clock = holder
    sample(p)
    clock.now = .005
    held = sample(p, valid=False, hold=True)
    clock.now = .01
    normal = sample(p, valid=False, hold=False)
    assert normal[2] and normal[3] > held[3]
    assert not p.fplc_direct_ball_input['hold_reference_on_dropout']
    assert not p.fplc_direct_ball_input['enable_dropout_reference_hold']
    assert not p.fplc_direct_ball_input['ball_observed']
    clock.now = .4
    assert not sample(p, valid=False, hold=False)[2]


def test_switch_can_resume_actor_immediately_and_return_to_hold(controller, holder, monkeypatch):
    p, clock = holder
    q = controller.native_reset_arm_q.copy()
    actor = controller._policy_action
    calls = []
    def counted(obs):
        calls.append(np.array(obs))
        return actor(obs)
    monkeypatch.setattr(controller, '_policy_action', counted)
    def tick(index, observed, hold):
        clock.now = index*.005
        pos, vel, valid, age = sample(p, profile=controller.fplc_profile, valid=observed, hold=hold)
        return controller.predict(pos, vel, ball_valid=valid, ball_obs_age_s=age, arm_q=q,
                                  arm_dq=np.zeros(7), direct_ball_position_m=pos,
                                  direct_ball_age_s=age, direct_ball_stamp_s=1.+clock.now,
                                  **p.fplc_direct_ball_input)
    tick(0, True, True)
    tick(1, False, True)
    assert len(calls) == 1 and controller._dropout_reference_hold_active
    tick(2, False, False)
    tick(3, False, False)
    assert len(calls) == 3 and not controller._dropout_reference_hold_active
    observed_index = 71 if controller._expected_obs_dim == 72 else 135
    assert calls[-1][observed_index] == 0.
    target = controller.arm_cmd_q.copy()
    tick(4, False, True)
    assert len(calls) == 3 and controller._dropout_reference_hold_active
    np.testing.assert_array_equal(controller.arm_cmd_q, target)
