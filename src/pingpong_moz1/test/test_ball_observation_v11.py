"""Evidence and expiry boundaries for the v1.1 camera-only observation path."""
import json
from types import SimpleNamespace
import numpy as np
import pytest
from pingpong_controller.local_ball_surface import local_surface
from pingpong_controller.ball_radius_observation import update_radius_observation
from pingpong_controller.ball_observation_calibration import silhouette_range, load_calibration, geometry_fingerprint
from pingpong_controller.ball_observation_model import BallObservationModel, query_observation_snapshot
from pingpong_controller.ball_contact_observation import BallContactObservation


def images(plane=False):
    h,w=120,160;u,v=80.,60.;fx=430.;z=.4
    yy,xx=np.mgrid[:h,:w]
    rays=np.stack(((xx-u)/fx,(yy-v)/fx,np.ones((h,w))),axis=-1)
    aa=np.sum(rays*rays,axis=-1);bb=z;cc=z*z-.02**2
    disc=bb*bb-aa*cc
    depth=(bb-np.sqrt(np.maximum(disc,0)))/aa
    if plane:depth=.38+.03*(xx-u)/fx
    depth=np.where(disc>0,np.round(depth/.001),0).astype(np.uint16)
    ir=np.where(disc>0,200,0).astype(np.uint8)
    # Only half of the stereo points survive, spread over the visible sphere.
    depth[(xx+yy)%2==0]=0
    return ir,depth,u,v,fx*.02/z,fx,fx,u,v,.001


def test_sparse_local_sphere_recovers_center_but_flat_patch_is_rejected():
    found=local_surface(*images())
    assert found is not None
    np.testing.assert_allclose(found['center_m'],[0,0,.4],atol=.0015)
    assert local_surface(*images(plane=True)) is None


def range_model():
    m=BallObservationModel();m.state=np.array([0.,0.,.5,0.,0.,0.]);m.state_time=1.
    m.covariance=np.diag([.006**2]*3+[.03**2]*3);m.velocity_valid=True;m.last_range_time=.9
    return m


def evidence():
    return dict(monocular_range_m=.51,monocular_range_sigma_m=.012,range_quality_reason='sparse_stereo')


def test_radius_needs_consecutive_evidence_and_never_renews_stereo_or_history():
    m=range_model();raw=evidence();origin=np.zeros(3);ray=np.array([0.,0.,1.])
    assert not update_radius_observation(m,raw,1.,origin,ray)
    assert update_radius_observation(m,raw,1.011,origin,ray)
    assert m.last_range_time==.9 and len(m.history)==0
    assert m.last_radius_accepted_time==1.011
    assert np.linalg.eigvalsh(m.covariance).min()>0
    m._new_segment(1.02)
    assert m.last_radius_time is None and m.last_radius_accepted_time is None
    assert not update_radius_observation(m,raw,1.022,origin,ray)


@pytest.mark.parametrize('change',[dict(monocular_range_m=.8),dict(monocular_range_sigma_m=0.),dict(range_quality_reason='dense'),dict(monocular_range_m=float('nan'))])
def test_radius_outliers_do_not_change_the_state(change):
    m=range_model();m.last_radius_time=.99;raw=evidence();raw.update(change);state=m.state.copy()
    assert not update_radius_observation(m,raw,1.,np.zeros(3),np.array([0.,0.,1.]))
    np.testing.assert_array_equal(m.state,state)
    assert m.last_radius_accepted_time is None


def test_rejected_bearing_cannot_use_plausible_radius():
    m=range_model();m.last_radius_time=.99;m.current_raw=evidence()
    assert not m._bearing_update(1.,np.zeros(3),np.array([1.,0.,0.]),.003)
    assert m.last_radius_accepted_time is None


def test_profile_or_image_edge_changes_disable_monocular_range():
    profile=dict(silhouette_range=dict(a=9.5,b=-.35,sigma_m=.012,ir_params=dict(threshold=150)))
    raw=dict(geometry_quality_valid=True,range_quality_reason='sparse_stereo',geometry_radius_px=20.,cx=100.,cy=100.)
    assert silhouette_range(raw,profile,dict(threshold=150),1.,640,480) is not None
    assert silhouette_range(raw,profile,dict(threshold=151),1.,640,480) is None
    raw['cx']=10.
    assert silhouette_range(raw,profile,dict(threshold=150),1.,640,480) is None


def snapshot():
    raw=dict(observation_velocity_valid=True,observation_position_valid=True,observation_range_age_s=.33,
        observation_radius_age_s=0.,observation_prediction_budget_s=.36,observation_prediction_floor_m=0.)
    for k,v in zip('xyz',[0.,0.,1.]):raw['observation_position_'+k+'_m']=v;raw['observation_velocity_'+k+'_m_s']=0.
    return raw


def test_radius_budget_and_lower_floor_expire_when_camera_stalls():
    raw=snapshot()
    assert query_observation_snapshot(raw,1.,1.,np.zeros(3),prediction_floor_m=1.24,max_prediction_s=.45)['valid']
    raw['observation_range_age_s']=.2
    r=query_observation_snapshot(raw,1.,1.034,np.zeros(3),prediction_floor_m=1.24,max_prediction_s=.45)
    assert not r['valid'] and r['reason']=='prediction_below_contact_band'
    raw['observation_range_age_s']=.33
    r=query_observation_snapshot(raw,1.,1.034,np.zeros(3),prediction_floor_m=0.,max_prediction_s=.45)
    assert not r['valid'] and r['reason']=='prediction_timeout'


def test_real_post_contact_fit_replaces_stale_prior_without_waiting_for_expiry():
    m=range_model();m.fit_rmse=.001;m.state[5]=1.
    c=BallContactObservation();c.plan=dict(time=.98)
    m.history.extend([(.99,np.zeros(3),.003),(1.,np.ones(3)*.001,.003)])
    raw={};c.publish(m,raw,1.)
    assert c.plan is None and raw.get('observation_contact_horizon_s') is None
    c.plan=dict(time=.98);m.history.clear();c.publish(m,raw,1.)
    assert raw['observation_contact_horizon_s']==0.


@pytest.mark.parametrize('extra',[dict(rotation=[[1,0,0],[0,1,0],[0,0,-1]]),dict(silhouette_range=dict(a=9.5,b=0,sigma_m=0,ir_params=dict(threshold=150)))])
def test_invalid_rotation_and_range_uncertainty_are_rejected(tmp_path,extra):
    c=SimpleNamespace(T_base_camera=np.eye(4),T_ir_to_rgb=np.eye(4),ir_fx=430.,ir_fy=430.,ir_cx=320.,ir_cy=180.,depth_scale=.001,ball_diameter_mm=40.)
    profile=dict(schema_version=1,geometry_fingerprint=geometry_fingerprint(c),translation_m=[0.,0.,0.],relative_time_s=0.,**extra)
    path=tmp_path/'cal.json';path.write_text(json.dumps(profile))
    with pytest.raises(ValueError):load_calibration(c,path)


def test_weak_radius_profile_is_disabled_when_exposure_changes():
    profile=dict(silhouette_range=dict(a=9.5,b=-.35,sigma_m=.012,ir_params=dict(threshold=150),camera_settings=dict(exposure=1500,gain=16)))
    raw=dict(geometry_quality_valid=True,range_quality_reason='sparse_stereo',geometry_radius_px=20.,cx=100.,cy=100.)
    assert silhouette_range(raw,profile,dict(threshold=150),1.,640,480,dict(exposure=1500,gain=16)) is not None
    assert silhouette_range(raw,profile,dict(threshold=150),1.,640,480,dict(exposure=2000,gain=16)) is None


def test_calibration_transforms_position_velocity_and_bearing_in_same_frame():
    import sys,types
    stub=types.ModuleType('ultralytics');stub.YOLO=object
    sys.modules.setdefault('ultralytics',stub)
    from pingpong_controller.hybrid_ball_detector import HybridBallDetector
    angle=.1;r=np.array([[np.cos(angle),-np.sin(angle),0.],[np.sin(angle),np.cos(angle),0.],[0.,0.,1.]])
    delta=np.array([.002,.004,.003]);transform=np.eye(4);transform[:3,3]=[.1,.2,.3]
    result=dict(detected=True,position=[100.,200.,500.],velocity=[10.,20.,30.],raw_measurement=dict(cx=100.,cy=70.,velocity_ready=True))
    core=SimpleNamespace(depth_mode='adaptive',output_frame='base_link',observation_quality_enabled=True,T_base_camera=transform,T_ir_to_rgb=np.eye(4),
        ir_fx=430.,ir_fy=430.,ir_cx=80.,ir_cy=60.,image_width=160,image_height=120,ir_params={},last_depth_selection={},
        observation_calibration=dict(rotation=r.tolist(),translation_m=delta.tolist(),relative_time_s=0.))
    core._detect_core_impl=lambda *args:result
    actual=HybridBallDetector._detect_core(core,None,None,None,1.)
    np.testing.assert_allclose(actual['position'],r@np.array([100.,200.,500.])+delta*1000.)
    np.testing.assert_allclose(actual['velocity'],r@np.array([10.,20.,30.]))
    raw=actual['raw_measurement'];ray=np.array([20./430.,10./430.,1.]);ray/=np.linalg.norm(ray)
    np.testing.assert_allclose([raw['bearing_origin_'+k+'_m'] for k in 'xyz'],r@transform[:3,3]+delta)
    np.testing.assert_allclose([raw['bearing_direction_'+k] for k in 'xyz'],r@ray)


def test_small_change_at_prior_boundary_does_not_create_a_velocity_spike():
    velocities=[]
    for vx in (.799,.801):
        m=BallObservationModel();m.fast_segment=True;m.contact_velocity_prior=(0.,np.array([0.,0.,2.]))
        for t in (0.,1./90.):
            p=np.array([vx*t,0.,1.+2.*t-4.9*t*t]);m.history.append((t,p,.003))
        assert m._fit(1./90.)
        velocities.append(m.state[3:])
        assert m.state[3]<vx
    assert np.linalg.norm(velocities[1]-velocities[0])<.003
