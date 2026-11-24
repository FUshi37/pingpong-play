"""Exercise Record -> real bridge -> camera IPC -> raw disk writer -> Stop.

Only the physical sensor and ordinary log recorder are substituted. No ROS,
camera device, robot publisher or policy model is started.
"""
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from pingpong_controller.camera_process import CameraProcess, camera_worker
from pingpong_controller.raw_camera_recording import RawCameraRecorder
from pingpong_controller.tools.interactive_policy_safety_probe import (
    CameraSource, PolicySafetyProbe, RosBridge, build_arg_parser, config_from_args,
)


def unpack(packet):
    return packet


def synthetic_sensor_start(self):
    self.raw_camera_metadata = {'format': 'raw-ir-depth-npz-v1', 'synthetic': True}
    self.raw_camera_recorder = RawCameraRecorder(slot_bytes=32*24*3, unpack=unpack)
    self.camera_direct_running = True
    def produce():
        arrays = dict(ir=np.full((24,32),220,np.uint8), depth=np.full((24,32),500,np.uint16))
        while self.camera_direct_running:
            self.camera_frame_count += 1
            self.raw_camera_recorder.offer((arrays,dict(
                timestampS=time.time(),sourceFrameNumber=self.camera_frame_count)))
            time.sleep(.011)
    self.camera_direct_thread = threading.Thread(target=produce,daemon=True)
    self.camera_direct_thread.start()


def synthetic_sensor_worker(*args):
    CameraSource._init_direct_realsense_camera_source = synthetic_sensor_start
    camera_worker(*args)


def wait_for(predicate):
    deadline = time.monotonic()+8.
    while time.monotonic()<deadline:
        if predicate():return
        time.sleep(.01)
    raise AssertionError('Record/Stop chain did not reach expected state')


def make_probe(path, cfg, camera):
    probe = PolicySafetyProbe.__new__(PolicySafetyProbe)
    probe.cfg = cfg
    probe.lock = threading.RLock()
    bridge = RosBridge.__new__(RosBridge)
    bridge.node = SimpleNamespace(camera_process=camera)
    probe.ros_bridge = bridge
    probe.browser_config = lambda: {}
    status = dict(active=True,runDir=str(path),error=None)
    probe.recorder = Mock()
    probe.recorder.start.return_value = status.copy()
    probe.recorder.status.side_effect = lambda: status.copy()
    probe.recorder.stop.return_value = status.copy()
    probe.recorder.start_analysis.return_value = status.copy()
    probe.control_loop_tick = 0
    probe.sim_mirror = None
    return probe


def test_record_to_disk_and_stop_through_real_bridge_and_camera_worker(tmp_path):
    cfg = config_from_args(build_arg_parser().parse_args(['--enable-raw-camera-recording']))
    client = CameraProcess(cfg,worker_target=synthetic_sensor_worker,startup_timeout_s=20.)
    probe = make_probe(tmp_path/'run',cfg,client)
    path = tmp_path/'run/raw_camera'
    try:
        # RosBridge deliberately has no camera_process attribute: it lives on node.
        assert not hasattr(probe.ros_bridge,'camera_process')
        status = probe.start_recording()
        assert status['rawCamera']['enabled']
        wait_for(lambda:probe.recording_status()['rawCamera'].get('writer',{}).get('written',0)>=10)
        probe.stop_recording()
        wait_for(lambda:(path/'summary.json').exists() and
                 json.loads((path/'summary.json').read_text()).get('state')=='complete')
        wait_for(lambda:probe.recording_status()['rawCamera']['state']=='complete')
        summary=json.loads((path/'summary.json').read_text())
        assert summary['written']>=10 and not summary['writeErrors']
        row=json.loads((path/'frames.jsonl').read_text().splitlines()[0])
        with np.load(path/row['path']) as frame:
            assert frame['depth'].dtype==np.uint16 and (frame['depth']==500).all()
            assert (frame['ir']==220).all()
    finally:
        client.close()


def test_disabled_and_missing_camera_are_explicit(tmp_path):
    cfg = config_from_args(build_arg_parser().parse_args([]))
    probe = make_probe(tmp_path,cfg,None)
    assert probe.start_recording()['rawCamera']=={'enabled':False,'state':'disabled'}
    cfg.enable_raw_camera_recording=True
    status=probe.start_recording()['rawCamera']
    assert status['state']=='unavailable' and status['error']
    probe.ros_bridge.node.camera_process=SimpleNamespace(
        _raw_directory='/tmp/new-run/raw_camera',
        raw_recording_status=lambda:{'state':'complete','directory':'/tmp/old-run/raw_camera',
                                     'writer':{'written':100}})
    status=probe.recording_status()['rawCamera']
    assert status['state']=='starting' and not status.get('writer')


def test_image_choice_can_be_disabled_then_reenabled_with_live_camera_worker(tmp_path):
    cfg = config_from_args(build_arg_parser().parse_args(['--enable-raw-camera-recording']))
    client = CameraProcess(cfg, worker_target=synthetic_sensor_worker, startup_timeout_s=20.)
    probe = make_probe(tmp_path / 'run', cfg, client)
    try:
        status = probe.start_recording({'saveCameraImages': False})
        assert status['rawCamera'] == {'enabled': False, 'state': 'disabled'}
        wait_for(lambda: client._sample is not None)
        assert client._raw_directory is None
        assert not (tmp_path / 'run/raw_camera').exists()
        probe.stop_recording()

        probe.start_recording({'saveCameraImages': True})
        wait_for(lambda: probe.recording_status()['rawCamera'].get('writer', {}).get('written', 0) >= 5)
        probe.stop_recording()
        wait_for(lambda: probe.recording_status()['rawCamera']['state'] == 'complete')
        assert (tmp_path / 'run/raw_camera/frames.jsonl').is_file()
        assert client.process.is_alive()
    finally:
        client.close()
