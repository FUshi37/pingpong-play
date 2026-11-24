import ast
import json
from pathlib import Path
import time

import numpy as np

from pingpong_controller.raw_camera_recording import RawCameraRecorder


def wait_for(predicate, timeout=6.):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate(): return
        time.sleep(.01)
    raise AssertionError('raw recorder did not reach expected state')


def unpack_arrays(packet):
    return packet


def frame(i=1):
    return dict(ir=np.full((24,32),i,np.uint8), depth=np.full((24,32),500+i,np.uint16)), dict(
        timestampS=10+i/90.,sourceFrameNumber=i,detected=False)


def test_lossless_raw_failure_frame_and_clean_session_end(tmp_path):
    recorder=RawCameraRecorder(slot_bytes=24*32*3,slots=2,unpack=unpack_arrays)
    path=tmp_path/'raw_camera'
    try:
        recorder.set_session(path,{'schemaVersion':1})
        wait_for(lambda:recorder.accepting)
        packet=frame(3);recorder.offer(packet)
        wait_for(lambda:(path/'frames.jsonl').exists() and (path/'frames.jsonl').stat().st_size>0)
        recorder.set_session(None)
        wait_for(lambda:json.loads((path/'summary.json').read_text())['state']=='complete')
        rows=[json.loads(x) for x in (path/'frames.jsonl').read_text().splitlines()]
        assert len(rows)==1 and rows[0]['detected'] is False
        assert rows[0]['timestampS']==packet[1]['timestampS']
        with np.load(path/rows[0]['path']) as saved:
            np.testing.assert_array_equal(saved['ir'],packet[0]['ir'])
            np.testing.assert_array_equal(saved['depth'],packet[0]['depth'])
            assert saved['depth'].dtype==np.uint16
    finally:recorder.close()
    assert not recorder.process.is_alive()


def test_slow_writer_drops_instead_of_blocking_producer(tmp_path):
    recorder=RawCameraRecorder(slot_bytes=24*32*3,slots=1,unpack=unpack_arrays,write_delay_s=.2)
    try:
        recorder.set_session(tmp_path/'raw',{})
        wait_for(lambda:recorder.accepting)
        for i in range(40):
            recorder.offer(frame(i))
            time.sleep(.005)
        wait_for(lambda:recorder.status()['poolFull']>0)
        assert recorder.shm.size==24*32*3
        packet=frame()
        started=time.perf_counter()
        for _ in range(10000):recorder.offer(packet)
        assert time.perf_counter()-started<.2
        assert recorder.offered>=10040
    finally:recorder.close()


def test_existing_recording_is_never_overwritten(tmp_path):
    path=tmp_path/'existing';path.mkdir()
    sentinel=path/'summary.json';sentinel.write_text('{"untouched":true}')
    recorder=RawCameraRecorder(slot_bytes=4096,unpack=unpack_arrays)
    try:
        recorder.set_session(path,{})
        wait_for(lambda:recorder.status().get('writer',{}).get('error'))
        recorder.set_session(None)
    finally:recorder.close()
    assert sentinel.read_text()=='{"untouched":true}'


def test_time_limit_stops_raw_only_and_does_not_restart(tmp_path):
    recorder=RawCameraRecorder(slot_bytes=4096,unpack=unpack_arrays,max_duration_s=.15)
    try:
        recorder.set_session(tmp_path/'raw',{})
        wait_for(lambda:recorder.accepting)
        recorder.offer(frame())
        wait_for(lambda:recorder.status()['state']=='duration_limit')
        count=recorder.offered
        for _ in range(10):recorder.offer(frame())
        time.sleep(.1)
        assert recorder.offered==count and not recorder.accepting
    finally:recorder.close()


def test_memory_size_budget_and_multiple_sessions(tmp_path):
    recorder=RawCameraRecorder(slot_bytes=24*32*3,slots=1,unpack=unpack_arrays,max_bytes=24*32*3)
    try:
        for name in ['first','second']:
            path=tmp_path/name
            recorder.set_session(path,{})
            wait_for(lambda:recorder.accepting)
            recorder.offer(frame())
            wait_for(lambda:recorder.status()['state']=='size_limit')
            wait_for(lambda:(path/'summary.json').is_file() and
                json.loads((path/'summary.json').read_text()).get('state')=='complete')
            result=json.loads((path/'summary.json').read_text())
            assert result['written']==1 and result['producer']['bytesSubmitted']<=24*32*3
    finally:recorder.close()


def test_local_sidecar_cannot_be_enabled_in_control_process():
    from pingpong_controller.tools.interactive_policy_safety_probe import build_arg_parser,config_from_args
    import pytest
    args=build_arg_parser().parse_args(['--enable-raw-camera-recording','--disable-camera-subprocess'])
    with pytest.raises(ValueError,match='camera subprocess'):config_from_args(args)


def test_writer_death_and_bad_frame_leave_live_producer_usable(tmp_path):
    recorder=RawCameraRecorder(slot_bytes=4096,unpack=unpack_arrays)
    try:
        recorder.set_session(tmp_path/'raw',{})
        wait_for(lambda:recorder.accepting)
        recorder.offer(({'ir':np.zeros((200,200),np.uint8)},{}))
        wait_for(lambda:recorder.status()['copyErrors']>0)
        recorder.offer(frame())
        wait_for(lambda:recorder.status()['submitted']>0)
        recorder.process.terminate();recorder.process.join()
        wait_for(lambda:recorder.status()['state']=='error')
        recorder.offer(frame())
        assert not recorder.accepting
    finally:recorder.close()


def test_capture_and_controller_do_not_call_raw_recorder():
    path=Path(__file__).parents[1]/'pingpong_controller/tools/interactive_policy_safety_probe.py'
    tree=ast.parse(path.read_text())
    capture=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_direct_realsense_capture_loop')
    assert 'raw_camera' not in ast.unparse(capture)
    from pingpong_controller.raw_camera_recording import RawCameraRecorder
    import inspect
    offer=inspect.getsource(RawCameraRecorder.offer)
    assert all(token not in offer for token in ('np.','put(', 'put_nowait(', 'time.', 'open(', '.acquire(', '.wait('))
