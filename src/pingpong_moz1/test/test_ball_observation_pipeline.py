import json
from types import SimpleNamespace
import numpy as np

from pingpong_controller.ball_observation_calibration import geometry_fingerprint,load_calibration,depth_pair_quality
from pingpong_controller.partial_ball_surface import partial_ball_candidates
from pingpong_controller.ball_contact_observation import BallContactObservation
from pingpong_controller.ball_observation_model import BallObservationProcessor,query_observation_snapshot


def camera():
    return SimpleNamespace(T_base_camera=np.eye(4),T_ir_to_rgb=np.eye(4),ir_fx=430.,ir_fy=430.,ir_cx=320.,ir_cy=180.,depth_scale=.001,ball_diameter_mm=40.)


def test_calibration_is_bound_to_camera_geometry_and_loads_once(tmp_path):
    core=camera();path=tmp_path/'calibration.json'
    value=dict(schema_version=1,geometry_fingerprint=geometry_fingerprint(core),translation_m=[.001,.002,-.015],relative_time_s=.004)
    path.write_text(json.dumps(value))
    assert load_calibration(core,path)==value
    core.T_base_camera[0,3]=.001
    assert load_calibration(core,path) is None


def test_async_depth_is_rejected_without_requiring_another_frame():
    assert depth_pair_quality(1000.,1000.)==(True,0.)
    assert depth_pair_quality(1000.,988.88)==(False,988.88-1000.)
    assert depth_pair_quality(1000.,None)==(None,None)
    assert depth_pair_quality(1000.,np.nan)[0] is False


def sphere_image():
    core=camera();h,w=360,640;u,v,z=330.,8.,.38
    center=np.array([(u-core.ir_cx)*z/core.ir_fx,(v-core.ir_cy)*z/core.ir_fy,z])
    yy,xx=np.mgrid[:h,:w];rays=np.stack(((xx-core.ir_cx)/core.ir_fx,(yy-core.ir_cy)/core.ir_fy,np.ones((h,w))),axis=-1)
    aa=np.sum(rays*rays,axis=-1);bb=rays@center;cc=center@center-.020**2
    discriminant=bb*bb-aa*cc;visible=discriminant>0
    d=(bb-np.sqrt(np.maximum(0.,discriminant)))/aa
    depth=np.where(visible,np.round(d/.001),0).astype(np.uint16)
    ir=np.where(visible,200,0).astype(np.uint8)
    return ir,depth,center


def partial(ir,depth):
    return partial_ball_candidates(ir,depth,fx=430.,fy=430.,cx=320.,cy=180.,depth_scale=.001)


def test_partial_visible_sphere_recovers_geometric_center():
    ir,depth,center=sphere_image();found=partial(ir,depth)
    assert len(found)==1
    np.testing.assert_allclose(found[0]['center_m'],center,atol=.0015)
    assert found[0]['support_fraction']>.9


def test_sparse_depth_and_a_large_bright_plane_are_not_ball_measurements():
    ir,depth,_=sphere_image();ys,xs=np.nonzero(depth)
    sparse=np.zeros_like(depth);sparse[ys[:80],xs[:80]]=depth[ys[:80],xs[:80]]
    assert partial(ir,sparse)==[]
    assert partial(np.full_like(ir,200),np.full_like(depth,400))==[]


def test_repeated_racket_feedback_does_not_erase_pose_derived_velocity():
    contact=BallContactObservation()
    contact.racket_motion((1.,[0.,0.,1000.],[0.,0.,0.]),1.)
    p,v,_=contact.racket_motion((1.02,[0.,0.,1010.],[0.,0.,0.]),1.03)
    np.testing.assert_allclose(v,[0.,0.,.5])
    np.testing.assert_allclose(p,[0.,0.,1.015])
    contact.racket_motion((1.02,[0.,0.,1010.],[0.,0.,0.]),1.04)
    assert len(contact.rackets)==2


def observation(p,*,shift=0.):
    return dict(detected=True,position=np.asarray(p)*1000.,velocity=[0.,0.,0.],raw_measurement=dict(ball_observation_version=1,
        ball_calibration_applied=True,ball_calibration_time_shift_s=shift,range_sigma_mm=3.,velocity_motion_state='flight'))


def contact_position(t):
    contact=.125;dt=t-contact;z=1.+.5*contact+.0082
    vz=-2. if dt<0 else .5+.67*2.5
    return np.array([.1,-.4,z+dt*vz-4.9*dt*dt])


def contact_seed():
    processor=BallObservationProcessor(prediction_floor_m=0.)
    for t in np.arange(11)/90.:
        obs=observation(contact_position(t))
        processor.process(obs,t,racket_state=(t,[100.,-400.,(1.+.5*t)*1000.],[0.,0.,500.]))
    return processor,obs,t


def test_future_contact_query_is_causal_and_needs_no_fit_in_control(monkeypatch):
    processor,obs,t=contact_seed();raw=obs['raw_measurement']
    assert raw['observation_bounce_horizon_s'] is not None
    count=len(processor.model.history)
    def forbidden(*a,**k):raise AssertionError('consumer must not fit')
    for name in ('lstsq','solve','inv','eigvalsh'):monkeypatch.setattr(np.linalg,name,forbidden)
    for query in (.127,.132,.140):
        result=query_observation_snapshot(raw,t,query,np.zeros(3),prediction_floor_m=0.)
        assert result['valid'] and result['source']=='contact_prediction'
        np.testing.assert_allclose(result['position'],contact_position(query),atol=1e-6)
    assert len(processor.model.history)==count
    assert not query_observation_snapshot(raw,t,.180,np.zeros(3),prediction_floor_m=0.)['valid']


def test_contact_prior_without_real_post_contact_samples_expires():
    processor,obs,t=contact_seed()
    for t in np.arange(11,18)/90.:
        obs=observation(contact_position(t));obs['detected']=False
        processor.process(obs,t,racket_state=(t,[100.,-400.,(1.+.5*t)*1000.],[0.,0.,500.]))
    result=query_observation_snapshot(obs['raw_measurement'],t,t,np.zeros(3),prediction_floor_m=0.)
    assert not result['valid']
    assert result['reason']=='contact_confirmation_timeout'
    assert len(processor.model.history)==0


def test_fixed_timestamp_calibration_cannot_publish_a_future_sample():
    processor=BallObservationProcessor(prediction_floor_m=0.)
    for t in np.arange(10)/90.:
        obs=observation([0.,0.,1.5+t-4.9*t*t],shift=.004)
        processor.process(obs,t)
    r=query_observation_snapshot(obs['raw_measurement'],t,t+.002,np.zeros(3),prediction_floor_m=0.)
    assert not r['valid'] and r['reason']=='future_observation'
    count=len(processor.model.history);processor.process(obs,t)
    assert len(processor.model.history)==count


def test_rolling_ball_on_racket_is_not_extrapolated_as_free_fall():
    processor=BallObservationProcessor(prediction_floor_m=0.)
    for t in np.arange(14)/90.:
        p=np.array([.03,.05+.08*t,1.0082])
        obs=observation(p)
        processor.process(obs,t,racket_state=(t,[0.,0.,1000.],[0.,0.,0.]))
    r=query_observation_snapshot(obs['raw_measurement'],t,t+.020,np.zeros(3),prediction_floor_m=0.)
    assert r['valid'] and obs['raw_measurement']['observation_supported']
    np.testing.assert_allclose(r['position'],p+[0.,.08*.020,0.],atol=1e-7)


def test_stationary_bright_object_away_from_racket_cannot_initialize_identity():
    processor=BallObservationProcessor(prediction_floor_m=0.)
    for t in np.arange(25)/90.:
        obs=observation([.3,-.2,1.2])
        processor.process(obs,t,racket_state=(t,[0.,-500.,1000.],[0.,0.,0.]))
        assert not obs['raw_measurement']['observation_position_valid']
    assert len(processor.model.history)==0


def test_background_pair_does_not_replace_known_ball_track():
    processor=BallObservationProcessor(prediction_floor_m=0.)
    for t in np.arange(10)/90.:
        obs=observation([0.,0.,1.4+t-4.9*t*t]);processor.process(obs,t)
    old_segment=processor.model.segment
    for t in np.arange(10,30)/90.:
        obs=observation([.5,.5,1.3]);processor.process(obs,t)
        assert not obs['raw_measurement']['observation_position_valid']
    assert processor.model.segment==old_segment


def test_legacy_stationary_flag_cannot_bypass_ball_identity_gate():
    processor=BallObservationProcessor(prediction_floor_m=0.)
    for t in np.arange(10)/90.:
        obs=observation([0.,0.,1.4+t-4.9*t*t]);processor.process(obs,t)
    for t in np.arange(10,20)/90.:
        obs=observation([.5,.5,1.3])
        obs['raw_measurement'].update(velocity_motion_state='supported',velocity_ready=True)
        processor.process(obs,t)
        assert not obs['raw_measurement']['observation_position_valid']
        assert not obs['raw_measurement']['observation_supported']
