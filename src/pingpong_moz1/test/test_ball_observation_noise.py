"""Noise calibration must not invent measurements, change clocks, or lag hits."""
import copy
from pathlib import Path
import numpy as np
import pytest

from pingpong_controller.ball_observation_noise import ObservationNoiseProfile, default_profile
from pingpong_controller.ball_observation_model import BallObservationProcessor, query_observation_snapshot


def profile(**extra):
    return ObservationNoiseProfile(dict(enabled=True, name='test',
        measurement_covariance_m2=[[2e-6, .3e-6, 0.], [.3e-6, 1e-6, 0.], [0., 0., 9e-6]],
        robust_nu=6., **extra))


@pytest.mark.parametrize('settings',[
    dict(enabled='false'),dict(filter_enabled=1),dict(output_bias_m=[0.,float('nan'),0.]),
    dict(output_bias_m=[0.,.006,0.]),dict(measurement_covariance_m2=np.zeros((3,3))),
    dict(measurement_covariance_m2=[[1e-6,2e-6,0.],[2e-6,1e-6,0.],[0.,0.,1e-6]]),
    dict(measurement_covariance_m2=[[1e-6,1e-7,0.],[0.,1e-6,0.],[0.,0.,1e-6]]),
    dict(process_noise_diagonal_m2_s3=[-.01,.01,.02]),dict(robust_nu=1.),
    dict(quality_reference_sigma_m=0.),dict(maximum_robust_inflation=20.),
])
def test_invalid_profiles_cannot_enter_the_camera_producer(settings):
    with pytest.raises(ValueError):ObservationNoiseProfile(settings)


def test_outlier_weighting_is_bounded_positive_and_keeps_xyz_correlation():
    p=profile();P=np.diag([1e-6]*3+[.01]*3)
    normal,_=p.measurement_noise(.003,1.,P,np.zeros(3))
    noisy,inflation=p.measurement_noise(.003,1.,P,np.array([.018,0.,0.]))
    assert 1.<inflation<=4.
    assert noisy[0,0]>normal[0,0] and noisy[1,1]==normal[1,1]
    assert noisy[0,1]!=0. and np.linalg.eigvalsh(noisy).min()>0.
    np.testing.assert_allclose(noisy,noisy.T,rtol=0,atol=1e-20)
    gated,_=p.measurement_noise(.003,1.,P,np.array([.018,0.,0.]),robust_allowed=False)
    np.testing.assert_array_equal(gated,normal)
    poor,_=p.measurement_noise(.006,2.,P,np.zeros(3))
    assert poor[0,0]>normal[0,0] and poor[2,2]>normal[2,2]


def make_observation(t, *, axis=True, calibrated=True):
    return dict(detected=True,position=np.array([.01*t,-.49,1.3+t-4.9*t*t])*1000.,
        velocity=np.array([.01,0.,1.-9.8*t])*1000.,raw_measurement=dict(
            ball_observation_version=1,ball_calibration_applied=calibrated,
            ball_axis_refinement=axis,range_sigma_mm=3.,velocity_ready=True,
            velocity_motion_state='flight',ball_calibration_time_shift_s=0.))


def test_bias_is_applied_once_to_output_only_and_not_to_fit_history(monkeypatch):
    bias=np.array([-.00016,-.00043,-.000338])
    base=BallObservationProcessor(prediction_floor_m=0.,noise_profile=ObservationNoiseProfile())
    tuned=BallObservationProcessor(prediction_floor_m=0.,noise_profile=profile(filter_enabled=False,output_bias_m=bias))
    for t in np.arange(15)/90.:
        original=make_observation(t);changed=copy.deepcopy(original)
        base.process(original,t);tuned.process(changed,t)
    a=original['raw_measurement'];b=changed['raw_measurement']
    assert a['observation_velocity_valid'] and b['observation_velocity_valid']
    for name in ('observation_range_age_s','observation_history_samples','observation_segment',
                 'observation_source','observation_position_variance','observation_velocity_variance'):
        assert a[name]==b[name]
    np.testing.assert_array_equal(base.model.state,tuned.model.state)
    for old,new in zip(base.model.history,tuned.model.history):
        assert old[0]==new[0];np.testing.assert_array_equal(old[1],new[1])
    snapshot=copy.deepcopy(b);tuned.process(changed,t)
    assert b==snapshot
    # Control consumes scalar snapshots only, without reading files or fitting.
    monkeypatch.setattr(Path,'read_text',lambda *a,**k:pytest.fail('query read a config file'))
    for now in [t,t+.005,t+.01]:
        before=query_observation_snapshot(a,t,now,[0.,0.,0.],prediction_floor_m=0.)
        after=query_observation_snapshot(b,t,now,[0.,0.,0.],prediction_floor_m=0.)
        assert before['valid']==after['valid']
        np.testing.assert_allclose(after['position'],before['position']-bias,atol=1e-14)
        np.testing.assert_array_equal(after['velocity'],before['velocity'])
    future=query_observation_snapshot(b,t,t-.01,[0.,0.,0.],prediction_floor_m=0.)
    assert not future['valid'];np.testing.assert_array_equal(future['position'],np.zeros(3))


def test_constant_bias_corrects_bounce_branch_without_a_velocity_discontinuity():
    from test_ball_axis_observation import contact_snapshot
    raw=contact_snapshot();p=profile(output_bias_m=[-.0001,-.0004,-.0003])
    before=copy.deepcopy(raw);p.correct_snapshot(raw)
    for now in [.008,.012,.02]:
        old=query_observation_snapshot(before,0.,now,np.zeros(3),prediction_floor_m=1.)
        new=query_observation_snapshot(raw,0.,now,np.zeros(3),prediction_floor_m=1.)
        assert old['valid']==new['valid']
        np.testing.assert_allclose(new['position'],old['position']-p.bias,atol=1e-14)
        np.testing.assert_array_equal(new['velocity'],old['velocity'])


@pytest.mark.parametrize('axis,calibrated',[(False,True),(True,False)])
def test_tuning_does_not_change_unbound_or_legacy_geometry(axis,calibrated):
    a=BallObservationProcessor(prediction_floor_m=0.,noise_profile=ObservationNoiseProfile())
    b=BallObservationProcessor(prediction_floor_m=0.,noise_profile=profile(output_bias_m=[.001,0.,0.]))
    for t in np.arange(15)/90.:
        o=make_observation(t,axis=axis,calibrated=calibrated);q=copy.deepcopy(o)
        a.process(o,t);b.process(q,t)
        for k in 'xyz':
            assert o['raw_measurement']['observation_position_'+k+'_m']==q['raw_measurement']['observation_position_'+k+'_m']
    assert 'observation_noise_profile' not in q['raw_measurement']


def test_legacy_environment_disables_noise_and_bias_without_config_io(monkeypatch):
    monkeypatch.setenv('CAMERA_OBSERVATION_TUNING','legacy')
    monkeypatch.setattr(Path,'read_text',lambda *a,**k:pytest.fail('legacy mode read config'))
    assert not default_profile().enabled


def test_profile_owns_its_immutable_covariance():
    R=np.eye(3)*1e-6;p=ObservationNoiseProfile(dict(measurement_covariance_m2=R))
    R[0,0]=2e-6
    assert p.reference[0,0]==1e-6
    with pytest.raises(ValueError):p.reference[0,0]=3e-6
