"""Causal clock/transport/FK checks without ROS, camera or robot commands."""
from types import SimpleNamespace
import pickle
import threading
import time

import numpy as np
import pytest

from pingpong_controller.ball_racket_context import FeedbackReader, RacketKinematics


def node():
    wall=time.time()
    return SimpleNamespace(lock=threading.Lock(),latest_joint_q=np.zeros(7),
        latest_joint_dq=np.ones(7),latest_joint_stamp_monotonic_s=time.monotonic()-.02,
        latest_joint_stamp_wall=wall-.02,latest_joint_header_stamp_s=wall-12.,
        latest_joint_source_sequence=42,latest_joint_source_fault='',
        latest_joint_sample_stamp_s=wall-12.,
        get_clock=lambda:SimpleNamespace(now=lambda:SimpleNamespace(nanoseconds=int(time.time()*1e9))))


def test_unsynchronised_header_keeps_original_receipt_age_and_duplicate_identity():
    n=node();r=FeedbackReader(n);sample=r()
    assert sample['clock']=='receipt'
    assert .018<time.time()-sample['stamp']<.030
    n.latest_joint_q[:]=9.
    n.latest_joint_stamp_monotonic_s=time.monotonic()
    n.latest_joint_stamp_wall=time.time()
    assert r() is sample
    np.testing.assert_array_equal(sample['q'],np.zeros(7))
    assert sample['header_stamp']==n.latest_joint_header_stamp_s


def test_reader_never_waits_for_receiver_lock_or_rejuvenates_stale_state():
    n=node();reader=FeedbackReader(n);sample=reader()
    with n.lock:
        started=time.perf_counter();assert reader() is sample
        assert time.perf_counter()-started<.02
    n.latest_joint_source_sequence+=1;n.latest_joint_source_fault='clock_backwards'
    assert reader() is None


def test_synchronised_header_uses_source_time_and_source_arrays_are_not_modified():
    n=node();n.latest_joint_header_stamp_s=n.latest_joint_stamp_wall-.003
    original=n.latest_joint_q.copy();s=FeedbackReader(n)()
    assert s['clock']=='synchronised_header'
    assert .021<time.time()-s['stamp']<.040
    s['q'][:]=100
    np.testing.assert_array_equal(n.latest_joint_q,original)


def simple_kinematics():
    mj=pytest.importorskip('mujoco')
    model=mj.MjModel.from_xml_string('''<mujoco><worldbody><body pos="0 0 1">
      <joint type="hinge" axis="0 1 0"/><geom type="sphere" size=".01"/>
      <site name="racket" pos=".2 0 0"/></body></worldbody></mujoco>''')
    spec=dict(id='synthetic',model=pickle.dumps(model),qpos=np.zeros(model.nq),
        joints=[0],site=0,base=-1,base_addresses=[-1,-1,-1],z_correction=0.)
    return model,RacketKinematics(spec,[0,0,.1])


def test_private_fk_matches_geometry_and_jacobian_velocity_without_touching_model():
    model,k=simple_kinematics()
    sample=dict(q=[0.],dq=[2.],sequence=1,stamp=1.,clock='receipt',header_stamp=.5)
    state,ctx=k.evaluate(sample)
    np.testing.assert_allclose(state[1],[200,0,900],atol=1e-8)
    np.testing.assert_allclose(state[2],[0,0,-400],atol=1e-8)
    np.testing.assert_allclose(ctx['normal'],[0,0,1],atol=1e-8)
    assert k.model is not model and ctx['velocity_valid']
    assert k.evaluate(sample)[0] is state
    assert k.evaluate(None)==(None,None)


def test_missing_joint_velocity_is_marked_unavailable():
    _,k=simple_kinematics()
    state,ctx=k.evaluate(dict(q=[0.],dq=None,sequence=1,stamp=1.,clock='receipt',header_stamp=None))
    assert not ctx['velocity_valid']
    np.testing.assert_array_equal(state[2],np.zeros(3))


def test_visualization_cannot_overwrite_authoritative_feedback():
    from pingpong_controller.tools.interactive_policy_safety_probe import CameraSource
    core=SimpleNamespace(latest_racket_state=(1.,[1,2,3],[0,0,0]),
        set_racket_state=lambda *args:pytest.fail('UI must not replace measured FK'))
    source=CameraSource();source.camera_lock=threading.Lock();source.camera_detector=core
    source.set_camera_racket_context((.9,[2,3,4],[0,0,0]),{'version':1})
    source.set_policy_viz_state(np.zeros(3),np.zeros(3),True,np.zeros(3),np.ones(3),stamp_wall=99.)
    assert source.camera_racket_observation[0][0]==.9
    assert core.latest_racket_state[0]==1. # installed only at the frame boundary


def test_measured_velocity_survives_sparse_pose_history():
    from pingpong_controller.ball_contact_observation import BallContactObservation
    c=BallContactObservation()
    c.racket_motion((1.,np.zeros(3),np.zeros(3)),1.)
    c.context=dict(stamp=1.01,velocity_valid=True)
    p,v,a=c.racket_motion((1.01,np.zeros(3),np.array([0,0,500.])),1.015)
    np.testing.assert_allclose(v,[0,0,.5]);np.testing.assert_allclose(p,[0,0,.0025])
