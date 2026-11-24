"""V325 source parity, measured history and release integration without actuation."""
from __future__ import annotations

import ast
import dataclasses
import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from pingpong_controller.fplc_deployment import (
    FPLC_CONTRACTS, REPO_ROOT, fplc_assets, fplc_real_roll_pitch_limit_deg,
    fplc_workspace_intersection,
)
from pingpong_controller.fplc_policy import FPLCPolicyController
from pingpong_controller.fplc_wind_numpy import WindBallHistory, WindStartupAuthority, WIND_HISTORY_LAGS
from pingpong_controller.mjx_rl_policy import _CheckpointUnpickler
from pingpong_controller.tools import interactive_policy_safety_probe as probe
from test_simcode9_fplc_contract import probe as runtime_probe, _policy_options

PROFILE = 'fplc_gpu0_v325'
SOURCE = REPO_ROOT / 'simcode11/pingpong_moz1-main/pingpong_controller/tools/rl_sim'
WIND_SOURCE = next((SOURCE.parents[1] / 'outputs/rl_sim').glob(
    'v325_goal_serial*/retired_ram_snapshots/*/source'))


def source_definitions(filename, names, namespace, directory=SOURCE):
    """Run original formulas with NumPy, omitting only local JAX imports."""
    class Imports(ast.NodeTransformer):
        def visit_Import(self, node):
            return ast.Pass()
        def visit_ImportFrom(self, node):
            return ast.Pass()
    original = ast.parse((directory / filename).read_text())
    definitions = [Imports().visit(node) for node in original.body
                   if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in definitions} == set(names)
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[
        ast.alias(name='annotations')], level=0), *definitions], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), filename, 'exec'), namespace)


@pytest.fixture(scope='module')
def controller():
    return FPLCPolicyController(observation_profile=PROFILE)


def test_actor_matches_original_training_policy_and_authority_gate(controller):
    with fplc_assets(PROFILE)[0].open('rb') as handle:
        payload = _CheckpointUnpickler(handle).load()
    namespace = dict(jnp=np, jax=SimpleNamespace(lax=SimpleNamespace(stop_gradient=lambda x: x)))
    source_definitions('fixed_plane_wind_actor.py', ['apply_bound_gate'], namespace)
    source_definitions('train_juggle_mjx_ppo.py', ['apply_mlp', 'policy_components'], namespace)
    rng = np.random.default_rng(325)
    # Exercise acquisition, partial authority and full authority independently.
    for authority in (0., 2., 2.125, 2.25, 4., 20.):
        for _ in range(6):
            observation = rng.normal(size=114).astype(np.float32)
            observation[-2:] = authority / 20.
            expected, _, _ = namespace['policy_components'](payload['params'], observation[None])
            np.testing.assert_allclose(controller._policy_action(observation), expected[0],
                                       atol=3e-6, rtol=1e-6)


def test_history_matches_original_500ms_lags_age_valid_and_missing_past():
    namespace = dict(jnp=np, LAGS=WIND_HISTORY_LAGS)
    source_definitions('fixed_plane_wind_history.py', ['history_suffix'], namespace)
    history = WindBallHistory()
    # Build the independent training state using integer sample timestamps.
    state = SimpleNamespace(ball_positions=np.zeros((1, 101, 3), np.float32),
                            ball_sample_steps=np.zeros((1, 101), np.int32),
                            ball_valid=np.zeros((1, 101), bool))
    for tick in range(160):
        observed = tick % 19 != 7
        position = np.array([tick * .001, -.49, 1.15], np.float32)
        sample_step = tick - tick % 3
        age = (tick - sample_step) * .005
        actual = history.advance(position, age, observed, [20., 20.])
        state.ball_positions = np.concatenate((
            (position if observed else np.zeros(3))[None, None], state.ball_positions[:, :-1]), axis=1)
        state.ball_sample_steps = np.concatenate((np.array([[sample_step]]), state.ball_sample_steps[:, :-1]), axis=1)
        state.ball_valid = np.concatenate((np.array([[observed]]), state.ball_valid[:, :-1]), axis=1)
        state.core = SimpleNamespace(step_count=np.array([tick]))
        expected = namespace['history_suffix'](state, np.array([[20., 20.]], np.float32))[0]
        np.testing.assert_allclose(actual, expected, atol=1e-7, rtol=1e-6)
        if tick == 0:
            np.testing.assert_array_equal(actual[25:32], np.zeros(7))
            np.testing.assert_array_equal(actual[33:40], np.zeros(7))
    expired = history.suffix(2., [20., 20.])
    np.testing.assert_array_equal(expired[:40], np.zeros(40))


def test_history_age_boundary_uses_training_float32_precision():
    history = WindBallHistory()
    history.advance([0., -.49, 1.15], 0., True, [20., 20.])
    history.sample_times[0] = .3
    # Python subtraction gives 0.6000000000000001; training float32 keeps it valid.
    at_boundary = history.suffix(.9, [20., 20.])
    assert at_boundary[24] == 1. and at_boundary[32] == 1.
    expired = history.suffix(.900001, [20., 20.])
    assert expired[24] == 0. and expired[32] == 0.


def test_startup_ramp_matches_authenticated_training_source(controller):
    namespace = dict(math=__import__('math'))
    source_definitions('fixed_plane_wind_pose.py', [
        'shelter_gain', 'pose_clock_steps', 'schedule_fraction',
        'scheduled_degrees', 'effective_degrees'], namespace, WIND_SOURCE)
    cfg = SimpleNamespace(**controller._checkpoint_env_cfg['wind_pose'])
    startup = WindStartupAuthority(controller._checkpoint_env_cfg)
    for hits in (0, 4, 5, 8):
        startup.confirmed_hits = hits
        startup.started_tick = 123 if hits >= 5 else None
        for tick in (0, 122, 123, 124, 133, 143, 173, 223, 322, 323, 500):
            state = SimpleNamespace(step_count=np.array([tick], np.int32),
                wind_startup_hit_count=np.array([hits]),
                wind_startup_started_s=np.array([123*.005 if hits >= 5 else -1.]),
                total_env_steps=np.array([1000000]))
            expected = namespace['effective_degrees'](state, .005, cfg, xp=np)[0]
            np.testing.assert_allclose(startup.limits_deg(tick), [expected, expected], atol=2e-6)


def test_startup_counts_qualified_flights_and_rejects_low_bounces(controller):
    startup = WindStartupAuthority(controller._checkpoint_env_cfg)
    racket = np.array([0., -.49, 1.025])
    # A reversal near the racket alone is insufficient, even repeated many times.
    for tick in range(100):
        startup.observe(tick, racket + [0., 0., .02], [0., 0., .3], racket, observed=True)
    assert startup.confirmed_hits == 0
    for i in range(5):
        tick = 200 + i * 60
        startup.observe(tick, racket + [0., 0., .02], [0., 0., -.5], racket, observed=True)
        startup.observe(tick+5, racket + [0., 0., .06], [0., 0., 3.], racket, observed=True)
        assert startup.confirmed_hits == i+1
        # Repeated held samples cannot count another launch.
        startup.observe(tick+6, racket + [0., 0., .06], [0., 0., 3.], racket, observed=True)
        assert startup.confirmed_hits == i+1
    np.testing.assert_allclose(startup.limits_deg(startup.started_tick), [2., 2.])
    np.testing.assert_allclose(startup.limits_deg(startup.started_tick+100), [11., 11.])
    np.testing.assert_allclose(startup.limits_deg(startup.started_tick+200), [20., 20.])


def test_startup_missing_measurement_does_not_confirm_pending_launch(controller):
    startup = WindStartupAuthority(controller._checkpoint_env_cfg)
    racket = np.array([0., -.49, 1.025])
    startup.observe(0, racket + [0., 0., .02], [0., 0., 0.], racket, observed=True)
    startup.observe(1, racket + [0., 0., .06], [0., 0., 3.], racket, observed=False)
    startup.observe(2, racket + [0., 0., .06], [0., 0., 3.], racket, observed=True)
    assert startup.confirmed_hits == 0


def test_startup_confirmation_thresholds_match_native_helpers(controller):
    namespace = dict(jnp=np, math=__import__('math'))
    source_definitions('fixed_plane_v107.py', ['first_launch_confirmation_limits'], namespace)
    source_definitions('fixed_plane_limit_cycle_v1.py', ['limit_cycle_hit_apex_thresholds_m'], namespace)
    cfg = controller._checkpoint_env_cfg
    for hit in range(6):
        clearance, deadline = namespace['first_launch_confirmation_limits'](
            hit_count=np.array([hit]), legacy_clearance_m=cfg['hit_confirm_rel_height'],
            legacy_deadline_steps=cfg['hit_confirm_max_steps'],
            first_clearance_m=cfg['fixed_plane_v107_first_confirm_rel_height_m'],
            first_deadline_steps=cfg['fixed_plane_v107_first_confirm_max_steps'])
        target = cfg['fixed_plane_limit_cycle_apex_schedule_z_m'][min(hit, 3)]
        _, floor = namespace['limit_cycle_hit_apex_thresholds_m'](np.array([target]),
            contact_ball_center_z_m=np.array([cfg['fixed_contact_surface_z_m']+.02]),
            quality_margin_m=cfg['fixed_plane_limit_cycle_hit_quality_apex_margin_m'],
            survival_margin_m=cfg['fixed_plane_limit_cycle_hit_survival_apex_margin_m'])
        for side in (-1, 1):
            startup = WindStartupAuthority(cfg); startup.confirmed_hits = hit
            racket = np.array([0., -.49, 1.025])
            startup.observe(0, racket+[0., 0., .02], [0., 0., -1.], racket, observed=True)
            position = racket + [0., 0., float(clearance[0])+.001]
            vz = np.sqrt(2.*startup.gravity_m_s2*(float(floor[0]) + side*.001-position[2]))
            startup.observe(int(deadline[0])-1, position, [0., 0., vz], racket, observed=True)
            assert startup.confirmed_hits == hit+(side == 1)


def test_controller_qualifies_with_direct_velocity(controller):
    controller.reset()
    for height in (1.045, 1.09):
        controller.predict([0., -.49, height], [0., 0., 20.], ball_valid=True,
            arm_q=controller.native_reset_arm_q, arm_dq=np.zeros(7),
            direct_ball_position_m=[0., -.49, height], direct_ball_velocity_m_s=[0., 0., .1])
    assert controller.wind_startup.confirmed_hits == 0


def test_controller_uses_same_startup_bounds_in_obs_and_projection(controller):
    controller.reset()
    controller.predict([0., -.49, 1.045], [0., 0., 0.], ball_valid=True,
                       arm_q=controller.native_reset_arm_q, arm_dq=np.zeros(7),
                       confirmed_hit_count=999)  # Unqualified external counter is ignored.
    np.testing.assert_array_equal(controller.last_obs[-2:], np.array([.1, .1], np.float32))
    np.testing.assert_allclose(np.rad2deg(controller._active_pose_limits()[0]), [2., 2.], atol=2e-6)
    assert controller.last_projection['windStartup']['residualGate'] == 0.
    # The physical stop envelope stays at the user's requested20 degrees.
    np.testing.assert_allclose(np.rad2deg(controller.real_rp_outer), [20., 20.], atol=2e-6)
    controller.wind_startup.confirmed_hits = 5
    controller.wind_startup.started_tick = 0
    controller._fplc_tick = 201
    np.testing.assert_allclose(np.rad2deg(controller._active_pose_limits()[0]), [20., 20.], atol=2e-6)
    controller.invalidate_after_intervention()
    assert controller.wind_startup.confirmed_hits == 0 and controller.wind_startup.started_tick is None
    np.testing.assert_allclose(np.rad2deg(controller._active_pose_limits()[0]), [2., 2.], atol=2e-6)


@pytest.mark.parametrize('rate', [90., 120., 200.])
def test_controller_history_uses_direct_samples_and_clears_on_intervention(controller, rate):
    controller.reset()
    direct = np.array([.01, -.49, 1.15], np.float32)
    predicted = direct + np.array([.03, 0., .01], np.float32)
    for tick in range(105):
        controller.predict(predicted, np.zeros(3), ball_valid=True, ball_observed=True,
                           ball_obs_age_s=.01, arm_q=controller.native_reset_arm_q,
                           arm_dq=np.zeros(7), direct_ball_position_m=direct,
                           direct_ball_age_s=.01, ball_observation_rate_hz=rate,
                           ball_history_position_m=predicted,
                           ball_history_age_s=0., ball_history_measurement_age_s=.01)
        np.testing.assert_array_equal(controller.wind_history.positions[0], direct)
        assert controller.last_obs.shape == (114,)
    np.testing.assert_array_equal(controller.last_obs[72:96], np.zeros(24))
    controller.invalidate_after_intervention()
    assert not controller.wind_history.valid.any() and controller.wind_history.tick == 0
    assert controller.ball_history is None


def test_v325_browser_names_schema_and_existing_physical_limits(controller):
    names = probe._obs_names(114)
    assert len(names) == 114 and names[:72] == probe._obs_names(72)
    assert names[-2:] == ['roll_authority/20deg', 'pitch_authority/20deg']
    contract = probe._policy_runtime_contract(controller)
    assert contract['ioShapeValid'] and not contract['armActionLimiter']
    assert contract['observationSegments'][-1]['stop'] == 114
    np.testing.assert_allclose(contract['fplc']['relativeRollPitchLimitDeg'], [20, 20], atol=2e-6)
    np.testing.assert_allclose(contract['fplc']['realRelativeRollPitchLimitDeg'], [20, 20], atol=2e-6)
    html = probe.HTML_TEMPLATE
    assert f'<option value="{PROFILE}">' in html
    assert '1.6 m/s' in html and 'fplcWindNote' in html


@pytest.mark.parametrize('rate', [90, 120, 200])
def test_v325_launcher_config_only_authenticates_114d_and_20deg(rate):
    result = subprocess.run(['bash', str(REPO_ROOT / 'RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh')],
                            cwd=REPO_ROOT, env={**os.environ, 'PROBE_CONFIG_ONLY': 'true',
                                              'POLICY_RELEASE_ID': PROFILE,
                                              'BALL_OBSERVATION_RATE_HZ': str(rate)},
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'policy observation=114D' in result.stdout
    assert 'training_roll_pitch=20.0,20.0deg' in result.stdout
    assert 'real_roll_pitch=20.0,20.0deg' in result.stdout
    assert str(fplc_assets(PROFILE)[0]) in result.stdout


def test_v325_does_not_replace_or_relocate_old_releases():
    manifest = json.loads((REPO_ROOT / 'config/policy_release_manifest.json').read_text())
    for release in manifest['releases']:
        if release['release_id'] in (PROFILE, 'fplc_gpu0_v325_u416'):
            assert release['checkpoint_path'].startswith('simcode11/')
        else:
            assert not release['checkpoint_path'].startswith('simcode11/')
    assert FPLC_CONTRACTS[PROFILE]['training_acceptance']['wind_max_speed_m_s'] == 1.6
    assert (REPO_ROOT / 'simcode11/COLCON_IGNORE').is_file()


def test_real_attitude_limits_are_selected_per_model():
    for profile in FPLC_CONTRACTS:
        expected = (20., 20.) if profile == PROFILE else ((7., 7.) if profile == 'fplc_gpu0_v325_u416' else (4., 4.))
        assert fplc_real_roll_pitch_limit_deg(profile) == expected


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('angle_deg', [-20.01, -20., -19.99, -4.5, 4.5, 19.99, 20., 20.01])
def test_v325_real_attitude_boundary(controller, monkeypatch, axis, angle_deg):
    original_pose = controller._pose
    reference = controller.reference_rotation.copy()
    def pose_with_angle(q, **kwargs):
        parts = list(original_pose(q, **kwargs))
        parts[3] = np.zeros(3, np.float32)
        parts[3][axis] = np.deg2rad(angle_deg)
        return tuple(parts)
    monkeypatch.setattr(controller, '_pose', pose_with_angle)
    limits = fplc_workspace_intersection(((-.348, .263), (-.702, -.230), (.842, 1.462)))
    assert controller.pose_is_safe(controller.native_reset_arm_q, limits) == (abs(angle_deg) <= 20.)
    np.testing.assert_array_equal(controller.reference_rotation, reference)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('fault', ['measured', 'target'])
def test_v325_publish_checks_measured_and_target_at_20deg(runtime_probe, monkeypatch, axis, fault):
    p = runtime_probe
    p._select_policy_profile(PROFILE)
    options = dataclasses.replace(_policy_options(p, PROFILE), publish_commands=True)
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
    angle = [19.99]
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
    angle[0] = 20.01
    assert not p._publish_control_command(target, options)
    assert p._hard_workspace_stop_active() and p.publish_rearm_required
    expected_fault = 'measured_pose_outside_envelope' if fault == 'measured' else 'target_pose_outside_envelope'
    assert expected_fault in p.publish_rearm_details['faults']
    np.testing.assert_allclose(p.publish_rearm_details['relativeRollPitchLimitDeg'], [20., 20.], atol=2e-6)


def test_v325_cannot_load_72d_parent_weights():
    with pytest.raises(RuntimeError, match='checkpoint SHA'):
        FPLCPolicyController(fplc_assets('fplc_gpu1_v172')[0], observation_profile=PROFILE)


def test_browser_switches_72_114_117_and_wind_note(tmp_path):
    functions = '\n'.join(re.search(r'function ' + name + r'\(\) \{.*?\n\}',
                                    probe.HTML_TEMPLATE, re.S).group()
                          for name in ('syncPolicyObservationControls', 'syncPolicyModelControl'))
    profiles = {'fplc_gpu1_v172': 72, PROFILE: 114, 'fplc_gpu0_v325_u416': 120, 'fplc_gpu0_v224': 117}
    config = {'policy': {'profileContracts': {
        p: {'observationNames': probe._obs_names(dim)} for p, dim in profiles.items()}}}
    script = tmp_path / 'v325_switch.js'
    script.write_text('const CONFIG = ' + json.dumps(config) + ';\n' + '''
const note = {hidden: true}, single5Note = {hidden: true};
const document = {createElement() {return {};}, getElementById(id) {return id === 'fplcSingle5Note' ? single5Note : note;}};
const policyProfileEl = {value: 'fplc_gpu1_v172'};
const policyObservationScaleTargetEl = {value: '0', options: [],
  replaceChildren() {this.options = [];}, appendChild(o) {this.options.push(o);}};
const policyObservationScaleResetEl = {}, policyModelEl = {};
const policyModelRowEl = {classList: {toggle() {}}};
const resetInitialPoseEl = {options: [], value: 'simcode9_native'};
const inputs = {obsHz: {}, workspaceSafetyMode: {}, policyObservationScaleFactor: {}};
let policyObservationScales = [];
function setControlDisabled(el, value) {el.disabled = value;}
''' + functions + '''
for (const [profile, width] of [['fplc_gpu1_v172',72], ['fplc_gpu0_v325',114],
                               ['fplc_gpu0_v325_u416',120], ['fplc_gpu0_v224',117], ['fplc_gpu1_v172',72]]) {
  policyProfileEl.value = profile;
  syncPolicyModelControl(); syncPolicyObservationControls();
  if (policyObservationScaleTargetEl.options.length !== width ||
      policyObservationScales.length !== width ||
      policyObservationScales.some(v => v !== 1) ||
      note.hidden !== (profile !== 'fplc_gpu0_v325') ||
      single5Note.hidden !== (profile !== 'fplc_gpu0_v325_u416')) process.exit(1);
}
''')
    subprocess.run(['node', str(script)], check=True, capture_output=True)
