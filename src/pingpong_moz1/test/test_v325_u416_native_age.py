"""Native refresh-clock regression; physical measurement expiry stays real."""
from pathlib import Path
import ast
from types import SimpleNamespace
import numpy as np
import pytest
from pingpong_controller.fplc_deployment import REPO_ROOT
from pingpong_controller.fplc_policy import FPLCPolicyController
from pingpong_controller.policy_ball_handover import is_fractional_refresh
from pingpong_controller.tools import interactive_policy_safety_probe as runtime
from test_simcode9_fplc_contract import _NumpyJax

PROFILE='fplc_gpu0_v325_u416'

@pytest.fixture(scope='module')
def controllers():
    return [FPLCPolicyController(observation_profile=PROFILE) for _ in range(2)]

def step(c,tick,raw_age=.0065,rate=90.,observed=True,**extra):
    ball=np.array([.002*np.sin(tick*.01),-.49,1.12],np.float32)
    kwargs=dict(ball_valid=True,ball_obs_age_s=raw_age,arm_q=c.native_reset_arm_q,
        arm_dq=np.zeros(7,np.float32),direct_ball_position_m=ball,
        direct_ball_velocity_m_s=np.zeros(3,np.float32),direct_ball_age_s=raw_age,
        direct_ball_stamp_s=100.+tick*.005-raw_age,ball_observed=observed,
        ball_observation_rate_hz=rate,enable_dropout_reference_hold=False)
    kwargs.update(extra)
    return c.predict(ball,np.zeros(3,np.float32),**kwargs)

@pytest.mark.parametrize('rate',[90.,120.,200.])
def test_raw_measurement_age_does_not_shift_native_actor_age(controllers,rate):
    a,b=controllers;a.reset();b.reset()
    last_refresh=0
    for tick in range(125):
        raw_age=.002+(tick%9)*.001
        step(a,tick,raw_age=0.,rate=rate);step(b,tick,raw_age=raw_age,rate=rate)
        if is_fractional_refresh(tick,rate,.005):last_refresh=tick
        np.testing.assert_array_equal(a.last_obs,b.last_obs)
        np.testing.assert_array_equal(a.last_action,b.last_action)
        np.testing.assert_array_equal(a.arm_cmd_q,b.arm_cmd_q)
        assert b.last_obs[61]*.25==pytest.approx((tick-last_refresh)*.005,abs=1e-8)
        assert b.last_projection['ballObservationAges']['mode']=='native_refresh_clock'
        assert b.last_projection['ballObservationAges']['inputRawMeasurementAgeS']==raw_age
    assert a._held_direct_age!=b._held_direct_age


def test_integer_step_history_ages_match_original_training_function(controllers,monkeypatch):
    c=controllers[0];c.reset();monkeypatch.setattr(c,'_policy_action',lambda obs:np.zeros(7,np.float32))
    path=REPO_ROOT/'simcode11/pingpong_moz1-main/pingpong_controller/tools/rl_sim/fixed_plane_wind_history.py'
    node,=[n for n in ast.parse(path.read_text()).body if getattr(n,'name',None)=='history_suffix']
    ns=dict(jnp=_NumpyJax(),jax=SimpleNamespace(Array=np.ndarray),LAGS=(0,10,20,30,40,60,80,100))
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node]
    exec(compile(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])),str(path),'exec'),ns)
    positions=np.zeros((1,101,3),np.float32);steps=np.zeros((1,101),np.int32);valid=np.zeros((1,101),bool)
    last_refresh=0;observed_flag=True
    for tick in range(125):
        # A dropout sampled at a camera refresh must retain a false observed
        # flag while its native refresh age returns to0, as in _observe_ball.
        observed=not 42<=tick<=47
        step(c,tick,raw_age=.009,observed=observed)
        if is_fractional_refresh(tick,90.,.005):last_refresh=tick;observed_flag=observed
        positions[:,1:]=positions[:,:-1].copy();steps[:,1:]=steps[:,:-1].copy();valid[:,1:]=valid[:,:-1].copy()
        positions[:,0]=c._held_direct_position if observed_flag else 0.
        steps[:,0]=last_refresh;valid[:,0]=observed_flag
        state=SimpleNamespace(core=SimpleNamespace(step_count=np.array([tick],np.int32)),ball_positions=positions,ball_sample_steps=steps,ball_valid=valid)
        expected=ns['history_suffix'](state,np.array([[5.,5.]],np.float32))[0]
        np.testing.assert_allclose(c.last_obs[72:112],expected[:40],rtol=0,atol=1e-7)
        assert c.last_obs[71]==float(observed_flag)

@pytest.mark.parametrize('field',['represented','direct'])
def test_native_zero_age_cannot_hide_expired_physical_measurement(controllers,field):
    c=controllers[0];c.reset();step(c,0)
    kwargs={'ball_obs_age_s':.351} if field=='represented' else {'direct_ball_age_s':.351}
    with pytest.raises(RuntimeError,match='350 ms'):step(c,1,**kwargs)
    assert c._fplc_requires_measured_reset


def test_native_age_reset_and_raw_age_recording_contract(controllers):
    c=controllers[0];c.reset();step(c,0,raw_age=.012)
    assert c.last_obs[61]==0 and c.last_obs[96]==0
    assert np.all(c.age_history[0,1:]==1)
    contract=runtime._policy_runtime_contract(c)['fplc']
    assert contract['actorBallAgeMode']=='native_refresh_clock'
    assert contract['deploymentRevision']=='u416_native_age_v2'
    assert contract['ballObservationAges']['inputRawMeasurementAgeS']==.012
    assert contract['realRelativeRollPitchLimitDeg']==pytest.approx([7.,7.])
    c.invalidate_after_intervention()
    assert c.last_ball_observation_ages=={} and c._held_actor_age==0


def test_older_fplc_keeps_its_existing_measurement_age_semantics():
    c=FPLCPolicyController(observation_profile='fplc_gpu1_v172')
    step(c,0,raw_age=.012)
    assert c.last_obs[61]*.25==pytest.approx(.012)
    step(c,1,raw_age=.020)
    assert c.last_obs[61]*.25==pytest.approx(.017)
    assert c.contract.get('actor_ball_age_mode','measurement_age')=='measurement_age'


def test_unknown_age_contract_is_rejected(monkeypatch):
    from pingpong_controller.fplc_deployment import FPLC_CONTRACTS
    monkeypatch.setitem(FPLC_CONTRACTS[PROFILE],'actor_ball_age_mode','unregistered_clock')
    with pytest.raises(RuntimeError,match='ball-age mode'):
        FPLCPolicyController(observation_profile=PROFILE)
