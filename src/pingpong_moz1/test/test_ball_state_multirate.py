"""Selected state rate, truthful prediction history and unchanged 5 ms actor."""
import dataclasses
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from pingpong_controller.fplc_policy import FPLCPolicyController
from pingpong_controller.fplc_deployment import REPO_ROOT, FPLC_PROFILES
from pingpong_controller.policy_ball_handover import is_fractional_refresh
from pingpong_controller.policy_release_manifest import validate_policy_release, ReleaseContractError
from test_simcode9_fplc_contract import probe, _policy_options, probe_module


@pytest.fixture
def actor():
    actor=FPLCPolicyController(observation_profile='fplc_gpu1_v172')
    actor._policy_action=lambda observation: np.zeros(7,np.float32)
    return actor


def tick(actor,rate,*,direct=None,predicted=None,age=.01,observed=True,history_age=None,measurement_age=None):
    direct=np.array([0.,-.49,1.1],np.float32) if direct is None else np.asarray(direct,np.float32)
    return actor.predict(direct,np.zeros(3),ball_valid=True,ball_obs_age_s=.001,
        arm_q=actor.native_reset_arm_q,arm_dq=np.zeros(7),ball_observed=observed,
        direct_ball_position_m=direct,direct_ball_age_s=age,direct_ball_stamp_s=10.,
        ball_observation_rate_hz=rate,ball_history_position_m=predicted,
        ball_history_age_s=history_age,ball_history_measurement_age_s=measurement_age,
        enable_dropout_reference_hold=False)


@pytest.mark.parametrize('rate',[90,120,200])
def test_native_source_rate_is_real_fractional_history_refresh(actor,rate):
    last=None
    for step in range(200):
        p=np.array([step*.00001,-.49,1.1],np.float32)
        tick(actor,rate,direct=p)
        refresh=is_fractional_refresh(step,rate,.005)
        assert actor.last_ball_state_refresh==refresh
        if refresh:last=p
        np.testing.assert_array_equal(actor.ball_history[0,0],last)
    assert actor.ball_state_refresh_count==rate
    assert actor._fplc_tick==200 and actor.dt==.005


@pytest.mark.parametrize('rate',[90,120,200])
def test_predicted_history_updates_on_repeated_real_frame_without_resetting_its_age(actor,rate):
    tick(actor,rate)
    last=actor.ball_history[0,0].copy();last_age=.01
    for step in range(1,40):
        prediction=np.array([step*.0001,-.49,1.1],np.float32)
        true_age=.01+step*.005
        tick(actor,rate,predicted=prediction,history_age=0.,measurement_age=true_age,observed=False)
        if is_fractional_refresh(step,rate,.005):
            last=prediction;last_age=0.
        else:last_age+=.005
        np.testing.assert_array_equal(actor.ball_history[0,0],last)
        assert actor.age_history[0,0]==pytest.approx(last_age)
        assert actor._held_state_measurement_age==pytest.approx(true_age)
        observed_refresh=any(is_fractional_refresh(s,rate,.005) for s in range(1,step+1))
        assert actor._direct_observed==(not observed_refresh)
        assert actor._held_direct_stamp==10.
        assert actor.last_obs[71]==float(actor._direct_observed)


@pytest.mark.parametrize('age',[.35001,np.nan,np.inf,-.01])
def test_high_rate_cannot_refresh_an_expired_prediction(actor,age):
    tick(actor,200)
    with pytest.raises(RuntimeError,match='measurement age'):
        tick(actor,200,predicted=[0.,-.49,1.1],history_age=0.,measurement_age=age)


def test_rate_change_requires_reset_and_never_changes_qref_silently(actor):
    tick(actor,90);before=actor.arm_cmd_q.copy()
    with pytest.raises(RuntimeError,match='explicit policy reset'):tick(actor,120)
    np.testing.assert_array_equal(actor.arm_cmd_q,before)
    actor.reset();tick(actor,120)
    assert actor._ball_state_rate_hz==120


@pytest.mark.parametrize('rate',[120,200])
def test_prediction_still_cannot_seed_actor_without_real_observation(actor,rate):
    with pytest.raises(RuntimeError,match='direct ball sample'):
        tick(actor,rate,predicted=[0.,-.49,1.1],history_age=.01,observed=False)


@pytest.mark.parametrize('rate',[90,120,200])
def test_web_options_use_startup_choice_even_when_browser_requests_another_rate(probe,rate):
    probe.cfg.ball_observation_rate_hz=float(rate)
    options=_policy_options(probe)
    assert options.ball_observation_rate_hz==rate
    assert probe.cfg.control_dt==.005
    assert options.policy_profile=='fplc_gpu1_v172'


@pytest.mark.parametrize('rate',[90,120,200])
def test_actual_probe_passes_quality_prediction_and_true_range_age_to_actor(probe,rate):
    probe.cfg.ball_observation_rate_hz=float(rate)
    options=dataclasses.replace(_policy_options(probe),ball_source='camera')
    probe.last_high_ball_occlusion_info=dict(model='quality_bearing_range_flight',
        observationRangeAgeS=.087,positionPredicted=True)
    probe.fplc_direct_ball_input=dict(direct_ball_position_m=np.array([0.,-.49,1.1]),
        direct_ball_age_s=.6,direct_ball_stamp_s=10.,ball_observed=False,
        ball_state_already_propagated=True)
    p=np.array([.01,-.49,1.15])
    probe._apply_policy_ball_sampling_contract(p,np.zeros(3),True,0.,options,source_timestamp_s=10.)
    payload=probe.fplc_direct_ball_input
    assert payload['ball_observation_rate_hz']==rate
    assert payload['ball_history_age_s']==0.
    assert payload['ball_history_measurement_age_s']==.087
    assert not payload['ball_observed']
    assert payload['direct_ball_stamp_s']==10.
    np.testing.assert_array_equal(payload['ball_history_position_m'],p.astype(np.float32))
    probe._apply_policy_ball_sampling_contract(p,np.zeros(3),False,.6,options,source_timestamp_s=10.)
    assert 'ball_history_position_m' not in probe.fplc_direct_ball_input


@pytest.mark.parametrize('rate',[90,120,200])
def test_launcher_and_release_validate_supported_rates_without_hardware(rate):
    env={k:v for k,v in os.environ.items() if k in ('PATH','LANG','LC_ALL','PYTHONPATH')}
    env['PROBE_CONFIG_ONLY']='true'
    run=subprocess.run(['bash',str(REPO_ROOT/'RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh'),
                        '--ball-state-rate-hz',str(rate)],env=env,cwd=REPO_ROOT,
                       capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stderr
    assert f'ball_state_rate={rate}Hz' in run.stdout
    assert 'camera=direct_realsense 848x480@90Hz' in run.stdout
    assert 'FPLC_running_max_age=15ms' in run.stdout
    verified=validate_policy_release(REPO_ROOT/'config/policy_release_manifest.json',
        'fplc_gpu1_v172',REPO_ROOT,deployment_ball_observation_rate_hz=rate,include_git=False)
    assert verified.as_dict()['deployment_ball_observation_rate_hz']==rate


@pytest.mark.parametrize('rate',[0,60,100,240,float('nan')])
def test_other_rates_remain_rejected_by_release(rate):
    with pytest.raises(ReleaseContractError):
        validate_policy_release(REPO_ROOT/'config/policy_release_manifest.json',
            'fplc_gpu1_v172',REPO_ROOT,deployment_ball_observation_rate_hz=rate,include_git=False)


@pytest.mark.parametrize('rate',[120,200])
def test_mirror_accepts_serialized_multirate_prediction_payload(rate):
    from pingpong_controller.sim_mirror import RealBallSimMirror, SimMirrorConfig, sim_mirror_config_for_policy_profile
    from pingpong_controller.fplc_deployment import fplc_assets
    profile='fplc_gpu1_v172';model,xml=fplc_assets(profile)
    config=sim_mirror_config_for_policy_profile(SimMirrorConfig(model_path=model,policy_xml_path=xml,
        sim_xml_path=xml,policy_obs_dim=72,control_dt=.005,enable_video_frames=False),profile)
    mirror=RealBallSimMirror(config)
    try:
        for i in range(5):
            payload=dict(direct_ball_position_m=[0.,-.49,1.1],direct_ball_age_s=.025,
                ball_observed=True,ball_observation_rate_hz=rate,
                ball_history_position_m=[i*.001,-.49,1.12],ball_history_age_s=0.,
                ball_history_measurement_age_s=.025)
            payload=json.loads(json.dumps(payload))
            mirror.step(np.array([0.,-.49,1.12]),np.zeros(3),ball_valid=True,ball_obs_age_s=0.,
                options=SimpleNamespace(policy_observation_scales=(),rl_use_valid_ball_obs_age=True),
                direct_ball_input=payload)
        assert mirror.policy._ball_state_rate_hz==rate
        assert mirror.policy._held_state_age==0.
        assert mirror.policy._held_state_measurement_age==.025
    finally:mirror.close()


@pytest.mark.parametrize('profile',FPLC_PROFILES)
@pytest.mark.parametrize('rate',[120,200])
def test_all_fplc_actor_shapes_accept_high_rate_history(profile,rate):
    actor=FPLCPolicyController(observation_profile=profile)
    for step in range(40):
        tick(actor,rate,predicted=[step*.0001,-.49,1.1],history_age=0.,measurement_age=.02)
    assert actor.last_obs.shape==(actor.contract['actor_dim'],)
    assert actor.ball_state_refresh_count==sum(is_fractional_refresh(i,rate,.005) for i in range(40))
    assert actor._fplc_tick==40
