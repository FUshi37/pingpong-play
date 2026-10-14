"""Atomic state/clock/lifecycle tests without ROS, UDP or a robot publisher."""
import time

import numpy as np
import pytest

from pingpong_controller.camera_process import replace_latest
from pingpong_controller.mocap_process import MocapInputProcess


def frozen_worker(kwargs,input_q,state_q,stop,parent_pid,anchor):
    replace_latest(state_q,{'kind':'ready'})
    captured=time.monotonic()
    receiver_stamp=anchor[0]+captured-anchor[1]
    feedback={'markerId':kwargs.get('marker_id',1),'ageS':0.,'tracked':True,
        'measurementTimestampS':receiver_stamp, 'receiverTimestampS':receiver_stamp,
        'clockSync':{'ready':True,'status':'synchronized'},
        'sourceTimestampS':12345.25,'controlBall':[-.03,-.49,1.06]}
    replace_latest(state_q,dict(kind='sample',position=[-.03,-.49,1.06],velocity=[0.,0.,.12],
        valid=kwargs.get('valid',True),feedback=feedback,observedMonotonicS=captured))
    stop.wait(20)


def wait_sample(client):
    deadline=time.monotonic()+5
    while client._sample is None and time.monotonic()<deadline:time.sleep(.005)
    assert client._sample is not None


def test_source_timestamps_survive_polling_and_age_expires():
    client=MocapInputProcess(worker_target=frozen_worker,timeout_s=.3)
    try:
        assert client.start();wait_sample(client)
        p,v,valid,info=client.get_control_state()
        assert valid and info['sourceTimestampS']==12345.25
        np.testing.assert_array_equal(p,np.array([-.03,-.49,1.06],np.float32))
        np.testing.assert_array_equal(v,np.array([0.,0.,.12],np.float32))
        stamp=info['measurementTimestampS'];age=info['measurementAgeS']
        info['clockSync']['ready']=False
        assert client.get_control_state()[3]['clockSync']['ready']
        time.sleep(.04)
        updated=client.get_control_state()[3]
        assert updated['measurementTimestampS']==stamp
        assert updated['measurementAgeS']>age+.025
        time.sleep(.3)
        assert not client.get_control_state()[2]
        assert client.get_control_state()[3]['measurementTimestampS']==stamp
    finally:client.stop()
    assert not client.process.is_alive() and not client.thread.is_alive()


def test_consumer_clock_jump_fails_closed_without_refreshing_stamp():
    offset=[100000.]
    client=MocapInputProcess(worker_target=frozen_worker,timeout_s=1.,receiver_clock=lambda:time.time()+offset[0])
    try:
        assert client.start();wait_sample(client)
        info=client.get_control_state()[3];assert client.get_control_state()[2]
        offset[0]+=2.
        assert not client.get_control_state()[2]
        failed=client.get_control_state()[3]
        assert failed['clockSync']['status']=='receiver_clock_changed'
        assert failed['measurementTimestampS']==info['measurementTimestampS']
    finally:client.stop()


@pytest.mark.parametrize('failure',['marker','exit','invalid'])
def test_marker_change_worker_death_and_invalid_sample_cannot_rejuvenate(failure):
    client=MocapInputProcess(worker_target=frozen_worker,timeout_s=1.,valid=failure!='invalid')
    try:
        assert client.start();wait_sample(client)
        stamp=client.get_control_state()[3]['measurementTimestampS']
        if failure=='marker':client.set_marker_id(2)
        elif failure=='exit':client.process.terminate();client.process.join(timeout=2)
        assert not client.get_control_state()[2]
        assert client.get_control_state()[3]['measurementTimestampS']==stamp
    finally:client.stop()


def test_transport_has_no_robot_publisher():
    client=MocapInputProcess()
    assert not hasattr(client,'publish_right_arm_command')
    assert not hasattr(client,'create_publisher')


def test_resynchronize_immediately_fences_cached_ready_packet():
    client=MocapInputProcess(worker_target=frozen_worker,timeout_s=1.)
    try:
        assert client.start();wait_sample(client)
        old=client.get_control_state()[3]
        assert client.get_control_state()[2]
        client.resynchronize()
        assert not client.get_control_state()[2]
        info=client.get_control_state()[3]
        assert not info['clockSync']['ready']
        assert info['clockSync']['status']=='awaiting_resynchronization'
        assert info['measurementTimestampS']==old['measurementTimestampS']
    finally:client.stop()
