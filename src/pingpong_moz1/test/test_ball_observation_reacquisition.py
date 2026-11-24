"""Physical and causal contracts for the calibrated reacquisition filter."""
import numpy as np
import pytest
from pingpong_controller.ball_observation_model import BallObservationModel


def trajectory(t):
    return np.array([.1+.2*t, -.4+.1*t, 1.35+1.8*t-4.9*t*t])


def seeded():
    model=BallObservationModel()
    model.reacquisition_filter_enabled=True
    for t in np.arange(10)/90.:
        result=model.step(trajectory(t),np.zeros(3),True,t,t)
    assert result['valid']
    return model


def test_reacquisition_uses_actual_recent_points_not_full_deque_length():
    model=seeded()
    for t,dx in ((.23,.002),(.23+1/90.,-.002)):
        result=model.step(trajectory(t)+[dx,0,0],np.zeros(3),True,t,t)
        assert result['valid']
        # Alternating 2 mm noise over 11 ms must not replace the established
        # 0.2 m/s velocity with the raw two-point -0.16 m/s difference.
        assert abs(result['velocity'][0]-.2)<.05
        assert np.linalg.norm(result['position']-trajectory(t))<.004
        assert model.filter_update=='reacquisition_kalman'
    assert len(model.history)==10
    assert model.fit_history_samples==2
    assert model.fit_span_s==pytest.approx(1/90.)


def test_real_contact_can_change_velocity_without_old_flight_smoothing():
    model=seeded()
    t=.14;p=np.array([.12,-.41,1.1]);velocity=np.array([.7,-.4,2.2])
    first=model.step(p,np.zeros(3),True,t,t,segment_reset=True)
    assert first['position_valid'] and not first['valid']
    dt=1/90.;second=p+dt*velocity+.5*dt**2*model.gravity
    result=model.step(second,np.zeros(3),True,t+dt,t+dt)
    assert result['valid']
    np.testing.assert_allclose(result['velocity'],velocity+dt*model.gravity,atol=1e-8)
    np.testing.assert_allclose(result['position'],second,atol=1e-8)


def test_duplicate_queries_do_not_reuse_a_measurement_or_reduce_covariance():
    model=seeded();t=model.last_frame_time
    covariance=model.covariance.copy();state=model.state.copy();history=list(model.history)
    for dt in (0.,1/360.,1/240.,1/200.,1/120.):
        r=model.step(trajectory(t),np.zeros(3),True,t,t+dt)
        np.testing.assert_allclose(r['position'],trajectory(t+dt),atol=1e-8)
    np.testing.assert_array_equal(model.covariance,covariance)
    np.testing.assert_array_equal(model.state,state)
    assert [x[0] for x in model.history]==[x[0] for x in history]


def test_bearing_then_range_preserves_positive_covariance_and_true_range_age():
    model=seeded();old_range=model.last_range_time;origin=np.array([0.,-1.,1.8])
    for t in np.arange(10,20)/90.:
        ray=trajectory(t)-origin;ray/=np.linalg.norm(ray)
        result=model.step(np.zeros(3),np.zeros(3),False,t,t,
                          bearing_origin=origin,bearing_direction=ray)
        assert result['valid'] and model.last_range_time==old_range
    for t in np.arange(20,35)/90.:
        result=model.step(trajectory(t)+np.array([.001,-.001,.001]),np.zeros(3),True,t,t)
        assert result['valid']
        np.testing.assert_allclose(model.covariance,model.covariance.T,atol=1e-15)
        assert np.linalg.eigvalsh(model.covariance).min()>0.
        assert len(model.history)<=10


def test_near_range_quality_weight_is_continuous_and_does_not_reject_measurement():
    velocities=[]
    for depth in (419.9,420.1):
        model=seeded();model.current_raw={'depth_mm':depth}
        t=10/90.
        result=model.step(trajectory(t)+np.array([.002,0,0]),np.zeros(3),True,t,t)
        assert result['valid'] and result['source']=='measured'
        velocities.append(result['velocity'])
    assert np.linalg.norm(velocities[0]-velocities[1])<1e-4


def test_prediction_still_expires_and_bad_ranges_do_not_refresh_it():
    model=seeded();last=model.last_range_time
    for t in (.12,.14):
        model.step(trajectory(t)+np.array([.15,0,0]),np.zeros(3),True,t,t,
                   allow_discontinuous_reacquisition=False)
        assert model.last_range_time==last
    t=last+.201
    result=model.step(np.zeros(3),np.zeros(3),False,t,t)
    assert not result['valid'] and not result['position_valid']
    assert result['reason']=='prediction_timeout'
