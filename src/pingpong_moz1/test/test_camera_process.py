"""Camera transport/lifecycle tests with a synthetic worker; no ROS or devices."""
import os
from queue import Empty
import time
from types import SimpleNamespace

import numpy as np

from pingpong_controller.camera_process import CameraProcess, replace_latest
from pingpong_controller.tools.interactive_policy_safety_probe import (
    CameraSource, CameraRuntimeOptions, build_arg_parser, config_from_args)


def fake_camera_worker(cfg, input_q, state_q, frame_q, stop, parent_pid):
    replace_latest(state_q, {'kind': 'ready'})
    stamp = 1791811426.308056
    wall = time.time()
    pending = {}
    while not stop.is_set():
        try:
            pending = input_q.get(timeout=.01)
        except Empty:
            pass
        if pending.get('options', {}).get('yolo_enabled'):
            return
        feedback = {'stampS': stamp, 'measurement': {'observation_state': 'tracking'},
                    'ready': True, 'detected': True, 'options': pending.get('options'),
                    'racketStamp': pending.get('viz', {}).get('stamp')}
        replace_latest(state_q, {'kind': 'sample', 'position': [0., -.49, 1.1],
            'velocity': [0., 0., -.5], 'valid': pending.get('options',{}).get('detect',True), 'feedback': feedback,
            'observationWallS': wall, 'frameStatus': (True, 'base_link'),
            'rotation': np.eye(3)})
        replace_latest(frame_q, {'frame': b'jpeg', 'info': {
            'seq': 1, 'wallTimeS': wall, 'enabled': True}})
        stop.wait(.01)


def wait_until(predicate, timeout=5.):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('camera cache did not reach expected state')


def test_transport_preserves_source_stamp_and_age_does_not_reset():
    client = CameraProcess(SimpleNamespace(), worker_target=fake_camera_worker)
    try:
        wait_until(lambda: client.measurement()[2])
        pos, vel, valid, stamp, measurement = client.measurement()
        assert valid and stamp == 1791811426.308056
        np.testing.assert_array_equal(pos, np.array([0., -.49, 1.1], np.float32))
        np.testing.assert_array_equal(vel, np.array([0., 0., -.5], np.float32))
        old_age = client.state()[3]['ageS']
        time.sleep(.04)
        assert client.state()[3]['ageS'] > old_age+.025
        assert client.measurement()[3] == stamp
        measurement['observation_state'] = 'mutated'
        assert client.measurement()[4]['observation_state'] == 'tracking'
        assert client.frame_status() == (True, 'base_link')
        np.testing.assert_array_equal(client.rotation(), np.eye(3))
        wait_until(lambda: client.visualization()[0] is not None)
        assert client.visualization()[0] == b'jpeg'
    finally:
        client.close()
    assert not client.process.is_alive() and not client.thread.is_alive()
    assert not client.measurement()[2]


def test_options_survive_visualization_updates_and_worker_death_invalidates():
    cfg = config_from_args(build_arg_parser().parse_args([]))
    source = CameraSource()
    source._initialize_camera_state(cfg)
    client = CameraProcess(SimpleNamespace(), worker_target=fake_camera_worker)
    try:
        options = source.camera_options
        client.set_options(options)
        stamp = time.time()-.2
        client.set_viz((np.zeros(3), np.zeros(3), True, np.zeros(3), np.zeros(3)), True, stamp)
        wait_until(lambda: client._sample is not None and client.state()[3].get('racketStamp') == stamp)
        assert client.state()[3]['options']['detect'] == options.detect
        assert client.state()[3]['racketStamp'] == stamp
        from dataclasses import replace
        client.set_options(replace(options, yolo_enabled=True))
        wait_until(lambda: not client.process.is_alive())
        assert not client.measurement()[2]
    finally:
        client.close()


def test_camera_only_source_has_no_robot_transport():
    source = CameraSource()
    assert not hasattr(source, 'publish_right_arm_command')
    assert not hasattr(source, 'create_publisher')
    assert not hasattr(source, '_joint_state_callback')
    cfg = config_from_args(build_arg_parser().parse_args([]))
    assert cfg.camera_subprocess
    cfg = config_from_args(build_arg_parser().parse_args(['--disable-camera-subprocess']))
    assert not cfg.camera_subprocess


def test_invalid_detector_sample_does_not_rejuvenate_source_timestamp():
    from dataclasses import replace
    source=CameraSource()
    source._initialize_camera_state(config_from_args(build_arg_parser().parse_args([])))
    client=CameraProcess(SimpleNamespace(),worker_target=fake_camera_worker)
    try:
        wait_until(lambda: client.measurement()[2])
        stamp=client.measurement()[3]
        age=client.state()[3]['ageS']
        client.set_options(replace(source.camera_options,detect=False))
        wait_until(lambda: not client.measurement()[2])
        assert not client.state()[2]
        assert client.measurement()[3]==stamp
        assert client.state()[3]['ageS']>=age
    finally:
        client.close()


def test_resynchronize_fences_old_cache_even_before_worker_ack():
    client=CameraProcess(SimpleNamespace(),worker_target=fake_camera_worker)
    try:
        wait_until(lambda:client.measurement()[2])
        old=client.measurement()[3]
        client.resynchronize()
        assert not client.measurement()[2]
        assert not client.measurement()[4]['clockSync']['ready']
        assert not client.state()[3]['clockSync']['ready']
        assert client.measurement()[3]==old
    finally:client.close()
