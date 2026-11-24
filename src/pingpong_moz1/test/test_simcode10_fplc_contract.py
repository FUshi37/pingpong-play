"""Independent simcode10 identities and causal observation deployment state."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import numpy as np
import pytest

from pingpong_controller.fplc_deployment import (
    FPLC_CONTRACTS, SIMCODE10_PROFILES, REPO_ROOT, fplc_assets,
)
from pingpong_controller.fplc_numpy import make_v224_history_context
from pingpong_controller.fplc_policy import FPLCPolicyController
from pingpong_controller.mjx_rl_policy import _CheckpointUnpickler
from pingpong_controller.policy_release_manifest import ReleaseContractError, validate_policy_release
from pingpong_controller.tools import interactive_policy_safety_probe as probe


EXPECTED_DIMS = {
    'fplc_gpu1_v223': (72, 393),
    'fplc_gpu0_v224': (117, 438),
    'fplc_gpu1_v224': (117, 438),
    'fplc_gpu1_v274_s9': (72, 398),
    'fplc_gpu1_v274_s11': (72, 398),
    'fplc_gpu0_v319': (72, 393),
    'fplc_v321_s4_u2097_20261029_r1': (72, 393),
    'fplc_gpu1_v322': (72, 393),
    'fplc_gpu0_v323': (72, 396),
    'fplc_gpu1_v324': (72, 396),
}
V224_PROFILES = ('fplc_gpu0_v224', 'fplc_gpu1_v224')
V321_PROFILE = 'fplc_v321_s4_u2097_20261029_r1'
NEW_MODELS = {
    'fplc_gpu1_v322': ('d35ea68508cbdfbaddfb3766f21c3f4e82f6f127e3caab8262d9e69a9ad307bc',
                       31361, 'fplc_gpu0_v319'),
    'fplc_gpu0_v323': ('126fd19f6e9f93ccb6b57b675c6f7a6ed55fb3e22134ce556a5711bd40e57d55',
                       3075, V321_PROFILE),
    'fplc_gpu1_v324': ('d1197b4024377c3e73927e75d91fff47411db5cdb1088c30a6c2cda86b4a9679',
                       2947, 'fplc_gpu1_v322'),
}


def test_model_archive_identity_and_dimensions():
    archive = json.loads((REPO_ROOT / 'simcode10/pingpong_moz1-main/pingpong_controller/tools/'
                         'rl_2real/FPLC_SELECTED_MODEL_CONTRACTS_20261026.json').read_text())
    candidate = json.loads((REPO_ROOT / FPLC_CONTRACTS[V321_PROFILE]['archive_contract_path']).read_text())
    sources = [*archive['models'], dict(
        id=candidate['release_id'], sha256=candidate['policy_sha256'],
        size_bytes=candidate['size_bytes'],
        path='pingpong_controller/outputs/rl_sim/releases/' + V321_PROFILE + '/policy.pkl',
        xml_sha256=candidate['xml_sha256'])]
    assert tuple(EXPECTED_DIMS) == SIMCODE10_PROFILES
    assert len({FPLC_CONTRACTS[p]['sha256'] for p in SIMCODE10_PROFILES}) == len(SIMCODE10_PROFILES)
    archive_profiles = tuple(p for p in SIMCODE10_PROFILES if p not in NEW_MODELS)
    for profile, source in zip(archive_profiles, sources, strict=True):
        contract = FPLC_CONTRACTS[profile]
        assert (contract['actor_dim'], contract['critic_dim']) == EXPECTED_DIMS[profile]
        assert contract['label'] == source['id']
        assert (contract['sha256'], contract['size_bytes']) == (source['sha256'], source['size_bytes'])
        assert contract['path'] == source['path']
        model, xml = fplc_assets(profile)
        assert model.is_relative_to(REPO_ROOT / 'simcode10')
        assert hashlib.sha256(xml.read_bytes()).hexdigest() == source['xml_sha256']
        if profile not in ('fplc_gpu1_v223', V321_PROFILE):
            assert contract['env_contract']['cartesian_pose_position_lower_m'] == [-.145, -.595, 1.]
            assert contract['env_contract']['cartesian_pose_position_upper_m'] == [.145, -.355, 1.15]


@pytest.mark.parametrize('profile', NEW_MODELS)
def test_new_checkpoint_identity_metadata_and_training_scope(profile):
    expected_sha, update, parent = NEW_MODELS[profile]
    contract = FPLC_CONTRACTS[profile]
    model, xml = fplc_assets(profile)
    assert model.is_relative_to(REPO_ROOT / 'simcode10')
    assert hashlib.sha256(model.read_bytes()).hexdigest() == contract['sha256'] == expected_sha
    assert model.stat().st_size == contract['size_bytes']
    assert hashlib.sha256(xml.read_bytes()).hexdigest() == contract['policy_xml_sha256']
    # These actual archives contain NumPy 2 globals; loading must also work in
    # the NumPy 1.26 deployment environment without importing training or JAX.
    with model.open('rb') as handle:
        payload = _CheckpointUnpickler(handle).load()
    assert (payload['obs_dim'], payload['critic_obs_dim']) == EXPECTED_DIMS[profile]
    assert payload['act_dim'] == contract['action_dim'] == 7
    assert payload['global_update'] == contract['global_update'] == update
    assert payload['stage_update'] == contract['stage_update']
    assert payload['step'] == contract['true_step']
    assert payload['opt'][-1] == contract['adam_t']
    assert payload['stage_name'] == contract['stage_name']
    assert payload['stage_converged'] is contract['stage_converged'] is False
    source = payload[contract['checkpoint_metadata_contract']]
    assert source['profile'] == contract['profile']
    assert source['parent_sha256'] == contract['parent_sha256'] == FPLC_CONTRACTS[parent]['sha256']
    assert payload['env_cfg']['horizon_sec'] == contract['env_contract']['horizon_sec'] == 1200.
    assert contract['training_acceptance']['qualified'] is True
    assert payload['independent_evaluation'] is None
    if profile != 'fplc_gpu1_v322':
        assert payload['training_acceptance'] == contract['training_acceptance']
        assert contract['final_acceptance_passed'] is payload['final_acceptance_passed'] is False
        assert contract['deterministic_certified'] is payload['deterministic_certified'] is False
        assert contract['training_acceptance']['hardware_coverage'] is False
        assert json.loads(json.dumps(source['noise'])) == contract['ball_observation_noise_contract']
        assert source['noise']['ball_position_noise_ar1_enabled'] is True
    validated = validate_policy_release(
        REPO_ROOT / 'config/policy_release_manifest.json', profile, REPO_ROOT, include_git=False)
    assert validated.checkpoint_sha256 == expected_sha
    assert validated.release['training_workspace_lower_m'] == contract['env_contract']['cartesian_pose_position_lower_m']
    assert validated.release['training_workspace_upper_m'] == contract['env_contract']['cartesian_pose_position_upper_m']


@pytest.mark.parametrize('rate', [90., 120., 200.])
@pytest.mark.parametrize('profile', NEW_MODELS)
def test_new_model_observation_projection_and_reentry_match_parent(profile, rate):
    candidate = FPLCPolicyController(observation_profile=profile)
    parent = FPLCPolicyController(observation_profile=NEW_MODELS[profile][2])
    np.testing.assert_array_equal(candidate.native_reset_arm_q, parent.native_reset_arm_q)
    np.testing.assert_array_equal(candidate.position_lower, parent.position_lower)
    np.testing.assert_array_equal(candidate.position_upper, parent.position_upper)
    for controller in (candidate, parent):
        controller._policy_action = lambda obs: np.linspace(-.02, .02, 7, dtype=np.float32)
    q = candidate.native_reset_arm_q.copy()
    for tick in range(50):
        ball = np.array([tick * .0001, -.49, 1.15], np.float32)
        options = dict(arm_q=q, arm_dq=np.zeros(7), ball_valid=True,
                       ball_observed=True, ball_obs_age_s=.003,
                       direct_ball_position_m=ball, direct_ball_age_s=.003,
                       direct_ball_stamp_s=tick * .005, ball_observation_rate_hz=rate)
        output = candidate.predict(ball, np.zeros(3), **options)
        np.testing.assert_allclose(output, parent.predict(ball, np.zeros(3), **options), atol=2e-7, rtol=1e-6)
        # Training AR1 noise changes the critic; it must not add synthetic
        # noise to the real measured actor position or its causal history.
        for field in ('last_obs', 'prev_action', 'ball_history', 'age_history'):
            np.testing.assert_allclose(getattr(candidate, field), getattr(parent, field), atol=2e-7, rtol=1e-6)
        q = q + np.float32(.1) * (output - q)
    for controller in (candidate, parent):
        controller.invalidate_after_intervention()
        assert controller.ball_history is None and controller._fplc_tick == 0
    np.testing.assert_allclose(candidate.predict(ball, np.zeros(3), **options),
                               parent.predict(ball, np.zeros(3), **options), atol=2e-7, rtol=1e-6)


@pytest.mark.parametrize('profile', NEW_MODELS)
def test_new_models_cannot_load_parent_or_other_model_weights(profile):
    for other in (NEW_MODELS[profile][2], *(p for p in NEW_MODELS if p != profile)):
        with pytest.raises(RuntimeError, match='checkpoint SHA'):
            FPLCPolicyController(fplc_assets(other)[0], observation_profile=profile)


@pytest.mark.parametrize('profile', NEW_MODELS)
@pytest.mark.parametrize('override,expected', [
    ({'BALL_OBSERVATION_RATE_HZ': '120'}, None),
    ({'BALL_OBSERVATION_RATE_HZ': '200'}, None),
    ({'POLICY_OBS_DIM': '117'}, 'observation_dim mismatch'),
    ({'CONTROL_RATE_HZ': '100'}, 'control_rate_hz mismatch'),
])
def test_new_model_launcher_rates_and_invalid_overrides(profile, override, expected):
    result = subprocess.run(['bash', str(REPO_ROOT / 'RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh')],
                            cwd=REPO_ROOT, env={**os.environ, 'PROBE_CONFIG_ONLY': 'true',
                                              'POLICY_RELEASE_ID': profile, **override},
                            text=True, capture_output=True, timeout=30)
    if expected:
        assert result.returncode != 0
        assert expected in result.stdout + result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert f'release_id={profile} profile={profile}' in result.stdout
        assert 'policy observation=72D' in result.stdout
        assert str(fplc_assets(profile)[0]) in result.stdout


def test_v321_archived_runtime_and_m04_observation_parity():
    contract = FPLC_CONTRACTS[V321_PROFILE]
    archive = json.loads((REPO_ROOT / contract['archive_contract_path']).read_text())
    candidate = FPLCPolicyController(observation_profile=V321_PROFILE)
    parent = FPLCPolicyController(observation_profile='fplc_gpu1_v172')
    assert archive['parent']['sha256'] == parent._checkpoint_sha256
    assert contract['sha256'] != parent._checkpoint_sha256
    assert contract['global_update'] == archive['global_update'] == 2097
    assert contract['stage_update'] == archive['stage_update'] == 400
    assert contract['stage_converged'] is archive['stage_converged'] is False
    assert candidate._checkpoint_env_cfg['horizon_sec'] == 1200
    for key, value in archive['runtime_reference']['cartesian_projection'].items():
        np.testing.assert_array_equal(candidate._checkpoint_env_cfg[key], value)
    np.testing.assert_array_equal(candidate.native_reset_arm_q, parent.native_reset_arm_q)
    # Identical executed actions isolate the observation, history, q_ref and
    # projection behavior from the deliberately different learned networks.
    for controller in (candidate, parent):
        controller._policy_action = lambda obs: np.linspace(-.02, .02, 7, dtype=np.float32)
    q = candidate.native_reset_arm_q.copy()
    for tick in range(50):
        ball = np.array([tick * .0001, -.49, 1.15], np.float32)
        options = dict(arm_q=q, arm_dq=np.zeros(7), ball_valid=True,
                       ball_observed=True, ball_obs_age_s=.003,
                       direct_ball_position_m=ball, direct_ball_age_s=.003,
                       direct_ball_stamp_s=tick * .005, ball_observation_rate_hz=90.)
        output = candidate.predict(ball, np.zeros(3), **options)
        np.testing.assert_allclose(output, parent.predict(ball, np.zeros(3), **options), atol=2e-7, rtol=1e-6)
        for field in ('last_obs', 'prev_action', 'ball_history', 'age_history'):
            np.testing.assert_allclose(getattr(candidate, field), getattr(parent, field), atol=2e-7, rtol=1e-6)
        if tick == 0:
            np.testing.assert_array_equal(candidate.last_obs[14:28], np.zeros(14))
            np.testing.assert_array_equal(candidate.last_obs[62:69], np.ones(7))
        q = q + np.float32(.1) * (output - q)


@pytest.mark.parametrize('profile,other', [
    (V321_PROFILE, 'fplc_gpu1_v172'), ('fplc_gpu1_v172', V321_PROFILE),
])
def test_v321_and_parent_cannot_exchange_checkpoints(profile, other):
    with pytest.raises(RuntimeError, match='checkpoint SHA'):
        FPLCPolicyController(fplc_assets(other)[0], observation_profile=profile)


@pytest.mark.parametrize('step', [0, 9, 10, 19, 20, 39, 40, 80])
@pytest.mark.parametrize('invalid', ['none', 'now', 'past', 'masked'])
def test_v224_local_history_formula_and_validity(step, invalid):
    # Independent geometric oracle: use full rotation matrices to express
    # differences in the current observed frame, including nontrivial tilt.
    rng = np.random.default_rng(123)
    history = rng.normal(size=(2, 41, 72)).astype(np.float32)
    rotations = np.empty((2, 41, 3, 3), dtype=np.float32)
    for batch in range(2):
        for tick in range(41):
            q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            q[:, 2] *= np.linalg.det(q)
            rotations[batch, tick] = q
            history[batch, tick, 55:61] = q[:, :2].T.ravel()
    history[:, :, 71] = 1.
    if invalid == 'now':
        history[0, 0, 71] = 0.
    if invalid == 'past':
        history[0, 20, 71] = 0.
    actual = make_v224_history_context(history, step_count=step, masked=invalid == 'masked')
    expected = np.zeros((2, 45), np.float32)
    for batch in range(2):
        for block, lag in enumerate((10, 20, 40)):
            if (step < lag or invalid == 'masked' or history[batch, 0, 71] <= .5
                    or history[batch, lag, 71] <= .5):
                continue
            now, past = history[batch, 0], history[batch, lag]
            rotation = rotations[batch, 0]
            expected[batch, block * 15:(block + 1) * 15] = np.r_[
                rotation.T @ (now[28:31] - past[28:31]),
                rotation.T @ (rotation[:, 2] - rotations[batch, lag, :, 2]),
                past[14:21], past[61], 1.]
    np.testing.assert_allclose(actual, expected, atol=5e-7, rtol=1e-6)
    assert actual.dtype == np.float32


@pytest.mark.parametrize('profile', V224_PROFILES)
def test_v224_real_tick_history_reset_and_reentry(profile):
    controller = FPLCPolicyController(observation_profile=profile)
    observed_prefixes = []
    q = controller.native_reset_arm_q.copy()
    controller._policy_action = lambda obs: np.zeros(7, np.float32)
    for tick in range(65):
        ball = np.array([tick * .0001, -.49, 1.15], np.float32)
        controller.predict(ball, np.zeros(3), arm_q=q, arm_dq=np.zeros(7),
                           ball_valid=True, ball_observed=True, ball_obs_age_s=.003,
                           direct_ball_position_m=ball, direct_ball_stamp_s=tick * .005)
        prefix = controller.last_obs[:72].copy()
        observed_prefixes.append(prefix)
        for block, lag in enumerate((10, 20, 40)):
            suffix = controller.last_obs[72 + block * 15:87 + block * 15]
            if tick < lag:
                np.testing.assert_array_equal(suffix, np.zeros(15))
            else:
                past = observed_prefixes[tick - lag]
                rotation = controller._pose(q, with_jacobian=False)[1]
                expected = np.r_[rotation.T @ (prefix[28:31] - past[28:31]),
                                 np.zeros(3), past[14:21], past[61], 1.]
                np.testing.assert_allclose(suffix, expected, atol=2e-7, rtol=1e-6)
        assert controller.prefix_history.shape == (1, 41, 72)
        assert controller._fplc_tick == tick + 1
    controller.invalidate_after_intervention()
    assert controller.prefix_history is None
    controller.predict(ball, np.zeros(3), arm_q=q, arm_dq=np.zeros(7),
                       ball_valid=True, ball_observed=True)
    np.testing.assert_array_equal(controller.last_obs[72:], np.zeros(45))


@pytest.mark.parametrize('profile', SIMCODE10_PROFILES)
def test_model_contract_reports_frontend_segments_and_rejects_wrong_schema(profile):
    controller = FPLCPolicyController(observation_profile=profile)
    width = EXPECTED_DIMS[profile][0]
    contract = probe._policy_runtime_contract(controller)
    assert contract['ioShapeValid'] and contract['fplc']['projectionRuntime']['compiled']
    assert contract['fplc']['actorSchema'] == FPLC_CONTRACTS[profile]['actor_schema']
    assert not contract['armActionLimiter'] and contract['controlDtS'] == .005
    assert len(probe._obs_names(width)) == width
    segments = contract['observationSegments']
    assert segments[0]['start'] == 0 and segments[-1]['stop'] == width
    assert all(a['stop'] == b['start'] for a, b in zip(segments, segments[1:]))
    values = list(range(width))
    assert probe._policy_observation_feature(values, 'BallObservationObs') == [28, 29, 30]
    assert probe._policy_observation_feature(values, 'RacketVelocityObs') == []
    assert f'value="{profile}"' in probe.HTML_TEMPLATE
    model, xml = fplc_assets(profile)
    with pytest.raises(ReleaseContractError, match='observation_dim mismatch'):
        validate_policy_release(REPO_ROOT / 'config/policy_release_manifest.json', profile, REPO_ROOT,
                                observation_dim=156, checkpoint_path=model, policy_xml_path=xml,
                                include_git=False)


@pytest.mark.parametrize('profile', V224_PROFILES)
def test_same_dimension_v224_branches_cannot_exchange_checkpoints(profile):
    other = next(p for p in V224_PROFILES if p != profile)
    with pytest.raises(RuntimeError, match='checkpoint SHA'):
        FPLCPolicyController(fplc_assets(other)[0], observation_profile=profile)


def test_frontend_dimension_selector_follows_72_117_72_switches(tmp_path):
    function = re.search(r'function syncPolicyObservationControls\(\) \{.*?\n\}',
                         probe.HTML_TEMPLATE, re.S).group()
    configs = {profile: dict(observationNames=probe._obs_names(dim[0]))
               for profile, dim in EXPECTED_DIMS.items()}
    script = tmp_path / 'observation_switch.js'
    script.write_text('const CONFIG = ' + json.dumps(dict(policy=dict(profileContracts=configs))) + ';\n' + '''
const policyProfileEl = {value: 'fplc_gpu1_v223'};
const policyObservationScaleTargetEl = {value: '0', options: [],
  replaceChildren() { this.options = []; },
  appendChild(option) { this.options.push(option); }};
const policyObservationScaleResetEl = {};
const inputs = {policyObservationScaleFactor: {value: '0.5'}};
let policyObservationScales = Array(72).fill(0.5);
const document = {createElement() { return {}; }};
''' + function + '''
for (const [profile, dim] of [['fplc_gpu1_v223', 72], ['fplc_gpu0_v224', 117],
                              ['fplc_gpu1_v224', 117], ['fplc_gpu1_v274_s11', 72],
                              ['fplc_gpu1_v322', 72], ['fplc_gpu0_v323', 72],
                              ['fplc_gpu1_v324', 72]]) {
  policyProfileEl.value = profile;
  syncPolicyObservationControls();
  if (policyObservationScales.length !== dim ||
      policyObservationScaleTargetEl.options.length !== dim ||
      policyObservationScales.some(value => value !== 1) ||
      !inputs.policyObservationScaleFactor.disabled ||
      !policyObservationScaleResetEl.disabled) process.exit(1);
  if (dim === 117 && !policyObservationScaleTargetEl.options[116].textContent.includes('200ms.valid'))
    process.exit(2);
  policyObservationScaleTargetEl.value = String(dim - 1);
}
''')
    subprocess.run(['node', str(script)], capture_output=True, check=True)
