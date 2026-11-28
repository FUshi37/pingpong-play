"""Authenticated u416 source/snapshot parity and hardware-path regression checks."""
from __future__ import annotations
import ast
import dataclasses
import io
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from pingpong_controller.fplc_deployment import (
    FPLC_CONTRACTS, REPO_ROOT, fplc_assets, fplc_real_workspace_limits,
    prepare_fplc_checkpoint_payload,
)
from pingpong_controller.fplc_policy import FPLCPolicyController
from pingpong_controller.fplc_single5_numpy import (
    FixedWindAcquisition, acquisition_features, encode_single5,
    encode_high_wind, observed_force_features,
)
from pingpong_controller.mjx_rl_policy import _CheckpointUnpickler, inspect_mjx_checkpoint_metadata
from pingpong_controller.policy_release_manifest import sha256_file
from pingpong_controller.sim_mirror import RealBallSimMirror, SimMirrorConfig
from pingpong_controller.tools import interactive_policy_safety_probe as runtime
from test_simcode9_fplc_contract import probe, _predict, _policy_options

PROFILE = 'fplc_gpu0_v325_u416'
WIND_EARLY_PROFILE = 'fplc_gpu0_v325_wind_early_r5fast_u6247'
SIM2REAL_PROFILE = 'fplc_gpu0_v325_wind_early_s2r_u7447'
R6_PROFILE = 'fplc_gpu0_v325_wind_early_s2r_r6_u7832'
CONTRACT = FPLC_CONTRACTS[PROFILE]
BASE = REPO_ROOT / 'simcode11/pingpong_moz1-main/pingpong_controller/outputs/rl_sim'
SOURCE = BASE / 'v325_single5_windfeature_high_train_20261124_r1'


@pytest.fixture(scope='module')
def controller():
    return FPLCPolicyController(observation_profile=PROFILE)


def original(filename, names, namespace=None, *, source=SOURCE):
    tree = ast.parse((source / filename).read_text())
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == set(names)
    ns = dict(math=math, np=np)
    ns.update(namespace or {})
    exec(compile(ast.Module(body=definitions, type_ignores=[]), filename, 'exec'), ns)
    return ns


@pytest.fixture(scope='module')
def snapshot():
    path = fplc_assets(PROFILE)[0].parent / 'trainer_state.zip'
    assert sha256_file(path) == '788975ba322906a463af28cf75e79539d6daafbba5474a9080284fee90d731f1'
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read('manifest.json'))
        arrays = {entry['path']: np.load(io.BytesIO(archive.read(f'runner_{i:05d}.npy')))
                  for i, entry in enumerate(manifest['trees']['runner'])
                  if any(key in entry['path'] for key in (
                      '.obs', 'step_count', 'fifth_hit_s', 'wind_startup_hit_count'))}
    return arrays


def test_release_sources_and_metadata_are_authenticated(controller, monkeypatch):
    for path, digest in CONTRACT['source_references'].items():
        assert sha256_file(REPO_ROOT / path) == digest
    model, _ = fplc_assets(PROFILE)
    with model.open('rb') as stream:
        payload = _CheckpointUnpickler(stream).load()
    assert 'env_cfg' not in payload and set(payload['params']) == {'pi', 'v', 'log_std'}
    assert set(payload['params']['pi']) == {'l1', 'l2', 'l3', 'out'}
    meta = inspect_mjx_checkpoint_metadata(str(model))
    assert meta['obs_dim'] == 120
    assert meta['env_cfg'] == controller._checkpoint_env_cfg
    assert meta['env_cfg']['fixed_contact_ball_z_bounds_m'] == [.98, 1.5]
    assert not hasattr(controller, '_wind_teacher_weights')
    monkeypatch.setitem(CONTRACT, 'checkpoint_metadata_sha256', '0'*64)
    with pytest.raises(RuntimeError, match='metadata SHA'):
        prepare_fplc_checkpoint_payload(payload, str(model))


def test_complete_encoder_matches_1024_native_snapshot_rows(snapshot):
    obs = snapshot['.obs']
    state = SimpleNamespace(
        step_count=next(v for k, v in snapshot.items() if k.endswith('.step_count')),
        wind_startup_hit_count=next(v for k, v in snapshot.items() if k.endswith('.wind_startup_hit_count')),
        fifth_hit_s=snapshot['.env_state.fifth_hit_s'])
    encoded = encode_single5(obs[:, :112], acquisition_features(state, xp=np))
    np.testing.assert_allclose(encoded, obs, atol=2e-6, rtol=2e-6)
    assert encoded.shape == (1024, 120) and np.isfinite(encoded).all()


def test_estimator_matches_frozen_source_and_invalid_history(snapshot):
    source = original('observed_force.py', {'select_consistent_wind', 'observed_force_features'})
    rows = snapshot['.obs'].copy()
    rows[:10, 104:112] = 0
    rows[10:20, 72:96] = np.nan
    rows[20:30, 96:104] = np.inf
    with np.errstate(invalid='ignore', over='ignore'):
        expected = source['observed_force_features'](rows, xp=np)
        actual = observed_force_features(rows, xp=np)
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[:30], np.tile([0., 0., 1., 0.], (30, 1)))
    assert np.max(np.linalg.norm(actual[:, :2], axis=1)) <= 1.000001
    assert (actual[:, 1] <= 0.).all()


def test_phase_and_high_wind_features_match_source():
    src = original('single5_runtime.py', {'acquisition_features'}, {'acquisition_time': lambda s: s.fifth_hit_s})
    high = original('wind_features.py', {'encode'}, dict(OLD_DIM=118, DEADBAND_M_S=1.25, ACTIVATION_WIDTH_M_S=.35))
    for count in (0, 4, 5, 6):
        state = SimpleNamespace(step_count=np.arange(0, 901, dtype=np.float32),
                                fifth_hit_s=np.float32(.5), wind_startup_hit_count=count)
        actual = acquisition_features(state, xp=np)
        np.testing.assert_array_equal(actual, src['acquisition_features'](state, xp=np))
        raw = np.zeros((901, 118), np.float32)
        raw[:, 112:114] = actual
        for speed in (0., 1.24, 1.25, 1.425, 1.6):
            raw[:, 114:116] = np.array([0., -speed/1.6])
            np.testing.assert_array_equal(encode_high_wind(raw, xp=np), high['encode'](raw, xp=np))
            if count < 5 or speed <= 1.25:
                assert not encode_high_wind(raw, xp=np)[:, 118:].any()


@pytest.mark.parametrize('profile', [PROFILE, WIND_EARLY_PROFILE, SIM2REAL_PROFILE, R6_PROFILE])
def test_single_actor_matches_original_forward_snapshot_and_zero_suffix(controller, snapshot, profile):
    class Jnp:
        def __getattr__(self, name):
            return getattr(np, name)
        def matmul(self, a, b, **kwargs):
            return np.matmul(a, b)
    model = fplc_assets(profile)[0]
    selected = controller if profile == PROFILE else FPLCPolicyController(observation_profile=profile)
    training_source = model.parents[1] / SOURCE.name
    source = original('wind_features.py', {'apply_actor'}, dict(
        OLD_DIM=118, NEW_DIM=120, jnp=Jnp(),
        jax=SimpleNamespace(lax=SimpleNamespace(Precision=SimpleNamespace(HIGHEST=None)))),
        source=training_source)
    with model.open('rb') as stream:
        pi = _CheckpointUnpickler(stream).load()['params']['pi']
    rows = snapshot['.obs'].copy()
    for zero_suffix in (False, True):
        if zero_suffix:
            rows[:, 118:] = 0
        expected = source['apply_actor'](pi, rows)
        actual = selected._policy_action(rows)
        np.testing.assert_array_equal(actual, expected)
        # Exercise the actual single-observation control path, too.
        for index in (0, 45, 333, 1023):
            np.testing.assert_allclose(selected._policy_action(rows[index]), expected[index], atol=4e-6, rtol=1e-6)


@pytest.mark.parametrize(('profile', 'update'), [
    (WIND_EARLY_PROFILE, 6247), (SIM2REAL_PROFILE, 7447), (R6_PROFILE, 7832)])
def test_wind_early_resolution_and_physical_config(controller, monkeypatch, profile, update):
    contract = FPLC_CONTRACTS[profile]
    for path, digest in contract['source_references'].items():
        assert sha256_file(REPO_ROOT / path) == digest
    model, xml = fplc_assets(profile)
    assert (REPO_ROOT / 'simcode12') in model.parents and (REPO_ROOT / 'simcode12') in xml.parents
    metadata = inspect_mjx_checkpoint_metadata(str(model))
    assert (metadata['obs_dim'], metadata['act_dim']) == (120, 7)
    assert metadata['env_cfg'] == {
        **controller._checkpoint_env_cfg, **contract.get('physical_config_overrides', {})}
    with model.open('rb') as stream:
        payload = _CheckpointUnpickler(stream).load()
    assert payload['actual_updates'] == update
    assert payload['plan_sha256'] == contract['checkpoint_plan_sha256']
    assert f'<option value="{profile}">' in runtime.HTML_TEMPLATE
    if profile == SIM2REAL_PROFILE:
        with pytest.raises(RuntimeError, match='Sim2Real curriculum stage mismatch'):
            prepare_fplc_checkpoint_payload({**payload, 's2r_stage_id': 8}, str(model))
    if profile == R6_PROFILE:
        with pytest.raises(RuntimeError, match='high-wind checkpoint configuration mismatch'):
            prepare_fplc_checkpoint_payload({**payload, 'obs_wind_max_m_s': 1.6}, str(model))
    monkeypatch.setitem(contract, 'physical_metadata_sha256', '0' * 64)
    with pytest.raises(RuntimeError, match='physical metadata SHA'):
        prepare_fplc_checkpoint_payload(payload, str(model))


@pytest.mark.parametrize('profile', [SIM2REAL_PROFILE, R6_PROFILE])
def test_sim2real_mirror_restores_frozen_warm_target_covariance(profile):
    model, xml = fplc_assets(profile)
    source = next(REPO_ROOT / path for path in FPLC_CONTRACTS[profile]['source_references']
                  if Path(path).name == 'sim2real_r5.py')
    declaration = next(node for node in ast.parse(source.read_text()).body
                       if isinstance(node, ast.Assign) and any(
                           isinstance(target, ast.Name) and target.id == 'P_WARM_3X3'
                           for target in node.targets))
    expected = np.asarray(ast.literal_eval(declaration.value.args[0]), dtype=np.float32)
    mirror = RealBallSimMirror(SimMirrorConfig(
        model_path=model, policy_xml_path=xml, sim_xml_path=xml, policy_profile=profile,
        policy_obs_dim=120, control_dt=.005, enable_video_frames=False))
    try:
        np.testing.assert_array_equal(mirror.rmp_planner._estimator.P, np.broadcast_to(expected, (7, 3, 3)))
        mirror._step_recovered_rmp(mirror.policy.native_reset_arm_q)
        assert not np.array_equal(mirror.rmp_planner._estimator.P[0], expected)
        snapshot = mirror.reset()
        np.testing.assert_array_equal(mirror.rmp_planner._estimator.P, np.broadcast_to(expected, (7, 3, 3)))
        np.testing.assert_array_equal(snapshot['plantContract']['rmpInitialTargetCovariance'], expected)
    finally:
        mirror.close()


def test_u7832_encoder_matches_frozen_implementation_above_1p6():
    model, _ = fplc_assets(R6_PROFILE)
    source = model.parents[1] / 'v325_single5_early_stability_20261126_r1'
    select = original('observed_force.py', {'select_consistent_wind'}, source=model.parents[1] / SOURCE.name)
    r6 = original('high_wind_r6.py', {'observed_force_features_r6', 'encode_r6'},
                  {**select, 'DEFAULT_OBS_WIND_MAX_M_S': 4.5, 'DEADBAND_M_S': 1.25,
                   'ACTIVATION_WIDTH_M_S': .35, 'OLD_DIM': 118}, source=source)
    ages = np.array([0., .05, .10, .15, .20, .30, .40, .50], dtype=np.float32)
    prefix = np.zeros((100, 112), dtype=np.float32)
    positions = prefix[:, 72:96].reshape(100, 8, 3)
    positions[:, :, 0] = np.linspace(-4., 4., 100)[:, None] * ages**2
    positions[:, :, 2] = .6 * ages - .5 * 9.81 * ages**2
    prefix[:, 96:104] = ages / .6
    prefix[:, 104:112] = 1
    prefix[-1, 104:112] = 0
    phase = np.ones((100, 2), dtype=np.float32)
    phase[::2] = 0
    with np.errstate(invalid='ignore'):
        force = r6['observed_force_features_r6'](prefix, xp=np)
    expected = r6['encode_r6'](np.concatenate([prefix, phase, force], axis=-1), xp=np)
    actual = encode_single5(prefix, phase, max_wind_m_s=4.5)
    np.testing.assert_array_equal(actual, expected)
    assert np.max(abs(actual[:, 114])) == pytest.approx(1.)
    assert np.any((abs(actual[:, 114]) * 4.5 > 1.6) & (abs(actual[:, 114]) < .99))
    assert not actual[::2, 118:].any()
    np.testing.assert_array_equal(actual[-1, 114:118], [0., 0., 1., 0.])


def test_u7832_live_observation_uses_profile_wind_scale_and_fixed_pose(snapshot, monkeypatch):
    controller = FPLCPolicyController(observation_profile=R6_PROFILE)
    row = next(obs for obs in snapshot['.obs'] if .7 < np.linalg.norm(obs[114:116]) < .95)
    monkeypatch.setattr(controller.wind_history, 'advance', lambda *args: np.r_[row[72:112], [0., 0.]])
    _predict(controller)
    obs = controller.last_obs
    np.testing.assert_array_equal(obs, encode_single5(obs[:112], obs[112:114], max_wind_m_s=4.5))
    assert np.linalg.norm(obs[114:116]) < np.linalg.norm(row[114:116])
    for confirmed in (0, 4, 5, 9):
        controller.wind_startup.confirmed_hits = confirmed
        controller.wind_startup.started_tick = 0 if confirmed >= 5 else None
        outer, inner = np.rad2deg(controller._active_pose_limits())
        np.testing.assert_allclose(outer, [15., 15.], atol=2e-6)
        np.testing.assert_allclose(inner, [14.9, 14.9], atol=2e-6)


def test_u7832_recording_labels_follow_real_and_mirror_profiles(tmp_path):
    def sample(profile):
        return {'runtime': {'policyProfile': profile}, 'policyObs': [0.] * 120,
                'sim': {'policyProfile': profile, 'policyObs': [0.] * 120}}

    r6 = sample(R6_PROFILE)
    for rows, expected in (([r6], '4.5'), ([sample(SIM2REAL_PROFILE)], '1.6'),
                           ([r6, sample(SIM2REAL_PROFILE)], 'profile_wind_max')):
        data = runtime._make_analysis_data(tmp_path, [], rows, {}, {}, {})
        assert data['obsNames'][114] == f'estimated_wind_x/{expected}'
        assert data['simObsNames'][114] == f'estimated_wind_x/{expected}'


def test_fixed_pose_does_not_ramp_or_gate_actions(controller):
    startup = FixedWindAcquisition(controller._checkpoint_env_cfg)
    for count in (0, 4, 5, 9):
        startup.confirmed_hits = count
        startup.started_tick = 200 if count >= 5 else None
        for tick in (0, 199, 200, 201, 220, 500, 800, 1000):
            np.testing.assert_array_equal(startup.limits_deg(tick), [5., 5.])
            assert 'residualGate' not in startup.diagnostics(tick)
    assert startup.features(800).tolist() == [1., 1.]
    src = original('single5_runtime.py', {'constant_degrees'})
    state = SimpleNamespace(step_count=np.arange(200))
    np.testing.assert_array_equal(src['constant_degrees'](state, .005, None, xp=np), np.full(200, 5.))


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('angle', [-7.01, -7., -6.99, -5.01, -5., -4.99, 4.99, 5., 5.01, 6.99, 7., 7.01])
def test_real_and_training_attitude_boundary(controller, monkeypatch, axis, angle):
    pose = controller._pose
    def with_angle(q, **kwargs):
        parts = list(pose(q, **kwargs)); parts[3] = np.zeros(3, np.float32)
        parts[3][axis] = np.deg2rad(angle)
        return tuple(parts)
    monkeypatch.setattr(controller, '_pose', with_angle)
    assert controller.pose_is_safe(controller.native_reset_arm_q, fplc_real_workspace_limits(PROFILE)) == (abs(angle) <= 7.)
    assert controller.pose_is_safe(controller.native_reset_arm_q) == (abs(angle) <= 5.)


def test_history_reset_dropout_and_rate_contract(controller):
    for rate in (90., 120., 200.):
        controller.reset()
        for tick in range(105):
            _predict(controller, ball_observation_rate_hz=rate, direct_ball_stamp_s=tick*.005,
                     ball_observed=True, direct_ball_age_s=.012)
            assert controller.last_obs.shape == (120,)
            assert np.isfinite(controller.last_obs).all()
            np.testing.assert_allclose(np.rad2deg(controller._active_pose_limits()[0]), [5., 5.], atol=1e-6)
        assert controller.last_obs[104:112].all()
        with pytest.raises(RuntimeError, match='rate change requires an explicit policy reset'):
            _predict(controller, ball_observation_rate_hz=200. if rate != 200. else 90.)
        ref = controller.arm_cmd_q.copy()
        _predict(controller, ball_observed=False, hold_reference_on_dropout=True,
                 ball_observation_rate_hz=rate)
        np.testing.assert_array_equal(controller.arm_cmd_q, ref)
        assert controller.last_obs[71] == 0
        controller.invalidate_after_intervention()
        assert controller.wind_history.tick == 0
        assert controller.wind_startup.started_tick is None
        assert not controller.wind_history.valid.any()


def test_runtime_recording_schema_and_switch_back(probe):
    for profile, width, degrees in [('fplc_gpu0_v325', 114, 20), (PROFILE, 120, 7),
                                    (SIM2REAL_PROFILE, 120, 7), ('fplc_gpu0_v224', 117, 4),
                                    (R6_PROFILE, 120, 17), (SIM2REAL_PROFILE, 120, 7),
                                    (WIND_EARLY_PROFILE, 120, 7), (PROFILE, 120, 7),
                                    ('fplc_gpu1_v172', 72, 4)]:
        probe._select_policy_profile(profile)
        options = _policy_options(probe, profile)
        probe._run_one_control_step(np.array([0., -.49, 1.16]), np.array([0., 0., -.3]), True, .01, options)
        c = runtime._policy_runtime_contract(probe.controller)
        assert len(runtime._obs_names(width)) == width
        assert c['observationSegments'][-1]['stop'] == width
        assert c['fplc']['realRelativeRollPitchLimitDeg'] == pytest.approx([degrees, degrees])
        assert c['fplc']['actorNetworkCount'] == (2 if width == 114 else 1)
        assert probe.publish_rearm_required
        if width == 120:
            assert c['fplc']['fixedPose']
            assert c['fplc']['actorLayers'] == [120, 282, 282, 282, 7]
            assert c['fplc']['windTraining']['final1200s99accepted'] is False
            assert runtime._policy_observation_feature(list(range(width)), 'RacketPositionObs') == [52, 53, 54]
            assert runtime._policy_observation_feature(list(range(width)), 'BallObservationObs') == [28, 29, 30]
            expected_scale = 4.5 if profile == R6_PROFILE else 1.6
            assert c['fplc']['obsWindMaxMS'] == expected_scale
            names = probe.browser_config()['policy']['observationNames']
            assert names[114] == f'estimated_wind_x/{expected_scale:g}'
            wind_segment = next(s for s in c['observationSegments'] if s['start'] == 114)
            assert wind_segment['unit'] == f'm/s/{expected_scale:g}'
        if profile in (WIND_EARLY_PROFILE, SIM2REAL_PROFILE):
            selected = probe.controller
            assert c['checkpointSha256'] == FPLC_CONTRACTS[profile]['sha256']
            assert c['fplc']['physicalMetadataSha256'] == selected.contract['physical_metadata_sha256']
            assert c['fplc']['actorBallAgeMode'] == 'native_refresh_clock'
            assert c['fplc']['windTraining']['strict_60s_successes'] == 1280
        if profile == SIM2REAL_PROFILE:
            assert c['fplc']['windTraining']['s2r_curriculum_stage']['id'] == 9
            assert c['fplc']['windTraining']['real_twin_strict_60s_successes'] == 640


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('fault', ['measured', 'target'])
@pytest.mark.parametrize(('profile', 'limit'), [(PROFILE, 7.), (SIM2REAL_PROFILE, 7.), (R6_PROFILE, 17.)])
def test_single5_publish_checks_measured_and_target(probe, monkeypatch, axis, fault, profile, limit):
    p = probe
    p._select_policy_profile(profile)
    options = dataclasses.replace(_policy_options(p, profile), publish_commands=True)
    measured = p.controller.native_reset_arm_q.copy()
    target = measured.copy()
    target[0] += .0001
    p.controller.last_cmd_q = target.copy()
    issued = []
    p.ros_bridge = SimpleNamespace(
        get_joint_feedback_with_age=lambda: (measured, np.zeros(7), .001, 'test'),
        get_joint_feedback=lambda: (measured, np.zeros(7)),
        publish_right_arm_command=lambda q: issued.append(np.array(q)))
    p.publish_rearm_required = False
    p.command_stream_active = True
    angle = [limit - .01]
    original_pose = p.controller._pose
    def selected_pose(q, **kwargs):
        parts = list(original_pose(q, **kwargs))
        parts[3] = np.zeros(3, np.float32)
        is_measured = np.array_equal(q, measured)
        if is_measured == (fault == 'measured'):
            parts[3][axis] = np.deg2rad(angle[0])
        return tuple(parts)
    monkeypatch.setattr(p.controller, '_pose', selected_pose)
    assert p._publish_control_command(target, options)
    np.testing.assert_array_equal(issued[-1], target)
    angle[0] = limit + .01
    assert not p._publish_control_command(target, options)
    assert p._hard_workspace_stop_active() and p.publish_rearm_required
    expected_fault = 'measured_pose_outside_envelope' if fault == 'measured' else 'target_pose_outside_envelope'
    assert expected_fault in p.publish_rearm_details['faults']
    np.testing.assert_allclose(p.publish_rearm_details['relativeRollPitchLimitDeg'], [limit, limit], atol=2e-6)



def test_210353_real_pitch_overshoot_has_authorized_outer_margin(controller):
    # Actual stop frame from probe_recording_20261124_210353. Target stayed
    # inside training5; measured pitch5.065 was stopped by the old outer5.
    measured = np.array([.30653464794, -.46256592870, .51955568790,
        1.56372177601, 1.04659509659, .09811413288, .56466746330], np.float32)
    target = np.array([.281599, -.384398, .41008, 1.634062,
        1.091529, .152973, .645296], np.float32)
    assert np.rad2deg(controller._pose(measured)[3][1]) == pytest.approx(5.065251, abs=1e-5)
    assert controller.pose_is_safe(measured, fplc_real_workspace_limits(PROFILE))
    assert not controller.pose_is_safe(measured)
    assert controller.pose_is_safe(target)
    np.testing.assert_allclose(np.rad2deg(controller._active_pose_limits()[0]), [5., 5.], atol=1e-6)
