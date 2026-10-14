import math
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest

from pingpong_controller.observation_clock import (
    CameraTimestampAligner, sample_clock_valid, receiver_clock_anchor)
from pingpong_controller.control_runtime import performance_cpus
from pingpong_controller.tools.interactive_policy_safety_probe import (
    CameraSource, PolicySafetyProbe, build_arg_parser, config_from_args)


def feed(clock, *, source=123., receiver=1791870000., mono=100., domain='realsense/hardware_clock'):
    for i in range(8):
        mapped, info = clock.update(source+i*.011, receiver+i*.011+.002*(i%2),
                                    mono+i*.011+.002*(i%2), domain=domain, frame_idx=i)
        if i < 7:
            assert mapped is None
    return mapped, info


def test_device_epoch_maps_to_consumer_and_retains_capture_stamp():
    clock = CameraTimestampAligner()
    mapped, info = feed(clock)
    assert mapped == pytest.approx(1791870000.+.077, abs=1e-6)
    assert info['sourceTimestampS'] == pytest.approx(123.077)
    assert info['excess_delay_s'] == pytest.approx(.002)
    assert info['ready'] and not info['fixed_latency_calibrated']
    assert sample_clock_valid(info, mapped+.01, monotonic_now=100.087)
    assert not sample_clock_valid(info, mapped+1., monotonic_now=100.087)
    assert not sample_clock_valid(info, mapped+.6, monotonic_now=100.677)


def test_ros_headers_preserve_absolute_capture_delay():
    clock = CameraTimestampAligner()
    for i in range(8):
        mapped, info = clock.update(500.+i*.01, 500.012+i*.01, 100.+i*.01,
                                    domain='ros_header', frame_idx=i)
    assert mapped == pytest.approx(500.07)
    assert info['receiverTimestampS'] - mapped == pytest.approx(.012)
    assert info['method'] == 'shared_ROS_header_clock_validation'


@pytest.mark.parametrize('domain', ['global_time','system_time'])
def test_system_camera_capture_preserves_usb_latency_and_ros_epoch_offset(domain):
    clock=CameraTimestampAligner()
    for i in range(8):
        capture=1000.+i*.011
        mapped, info=clock.update(capture, 91000.012+i*.011, 100.+i*.011,
            system_s=1000.012+i*.011,domain='realsense/timestamp_domain.'+domain, frame_idx=i)
    assert mapped == pytest.approx(91000.077)
    assert info['receiverTimestampS']-mapped == pytest.approx(.012)
    assert info['sourceTimestampS'] == pytest.approx(1000.077)
    assert info['captureLatencyPreserved']
    assert info['method']=='system_capture_to_ROS_clock'


def test_slight_global_clock_lead_is_calibrated_toward_past_before_ready():
    clock=CameraTimestampAligner()
    for i in range(8):
        mapped, info=clock.update(1000.003+i*.011, 91000.+i*.011, 100.+i*.011,
            system_s=1000.+i*.011,domain='realsense/timestamp_domain.global_time',frame_idx=i)
        if i<7:assert mapped is None
    assert info['ready'] and info['reset_count']==0
    assert info['futureClockCorrectionS']==pytest.approx(-.003,abs=1e-6)
    assert mapped<=info['receiverTimestampS']
    assert sample_clock_valid(info,91000.087,monotonic_now=100.087)
    assert not sample_clock_valid(info,91000.077-1.,monotonic_now=100.087)


@pytest.mark.parametrize('source', [None, math.nan, 0., 1000.1, 999.])
def test_invalid_ros_clock_never_replaced_with_now(source):
    clock = CameraTimestampAligner()
    mapped, info = clock.update(source, 1000., 100., domain='ros_header')
    assert mapped is None and not info['ready']


def test_duplicates_and_old_frames_do_not_refresh_time():
    clock = CameraTimestampAligner()
    _, info = feed(clock)
    for source, idx in [(123.077,7),(123.070,6)]:
        mapped, result = clock.update(source,1791870000.09,100.09,
                                      domain='realsense/hardware_clock',frame_idx=idx)
        assert mapped is None and not result['accepted']
    assert clock.fit._previous_unwrapped == info['sourceTimestampS']


def test_switch_domain_reset_and_device_restart_require_new_warmup():
    clock = CameraTimestampAligner()
    _, old = feed(clock, source=86399.)
    clock.reset()
    assert not clock.latest['ready']
    assert clock.generation == old['generation']+1
    _, ready = feed(clock, source=86399., receiver=1791870001., mono=101.)
    mapped, info = clock.update(.01,1791870001.09,101.09,
                                domain='realsense/hardware_clock',frame_idx=0)
    assert mapped is None and not info['ready'] and clock.fit.rollover_count == 0
    assert info['epoch'] > ready['epoch']


def test_result_in_flight_cannot_cross_clock_resynchronization():
    cfg=config_from_args(build_arg_parser().parse_args([]))
    source=CameraSource();source.cfg=cfg;source.camera_lock=threading.RLock()
    source._initialize_camera_state(cfg)
    _, old=feed(source.camera_clock)
    source.resynchronize_camera_clock()
    source._store_camera_observation(dict(detected=True,position=[0.,0.,1000.],
        velocity=[0.,0.,0.],_captureClockSync=old),old['measurementTimestampS'])
    assert not source.latest_camera_valid and source.latest_camera_stamp_sec is None


def test_slow_clock_anchor_is_rejected(monkeypatch):
    times=iter([1.,1.02])
    monkeypatch.setattr('pingpong_controller.observation_clock.time.monotonic',lambda:next(times))
    with pytest.raises(ValueError,match='5 ms'):
        receiver_clock_anchor(lambda:123.)


def test_source_switch_clears_old_prediction_and_requires_rearm_if_sending():
    probe=PolicySafetyProbe.__new__(PolicySafetyProbe)
    probe.selected_ball_clock_source='camera';probe.command_stream_active=True
    events=[]
    probe.ros_bridge=SimpleNamespace(resynchronize_ball_clock=lambda s:events.append(('sync',s)),
        ball_clock_status=lambda s:dict(required=True,ready=False))
    probe._reset_policy_ball_sampling_contract=lambda:events.append(('history',))
    probe._require_publish_rearm=lambda reason,**_:events.append(('rearm',reason))
    probe._publish_hold_on_command_disable=lambda:events.append(('hold',))
    from dataclasses import make_dataclass
    Options=make_dataclass('Options',[('ball_source',str),('publish_commands',bool)])
    options=probe._select_ball_clock(Options('mocap',True))
    assert not options.publish_commands
    assert ('sync','mocap') in events and ('rearm','ball_source_changed') in events
    assert events.count(('hold',))==1
    assert probe.ball_estimator is None and probe.high_ball_occlusion_predictor is None
    assert probe.policy_needs_reentry_reset


def test_hybrid_cpu_selection_uses_allowed_fast_physical_cores(tmp_path):
    for cpu,freq,core in [(0,6000,0),(1,6000,0),(2,6000,1),(3,4400,2),
                          (5,5700,3),(6,5700,3)]:
        root=tmp_path/f'cpu{cpu}'
        (root/'cpufreq').mkdir(parents=True);(root/'topology').mkdir()
        (root/'cpufreq/cpuinfo_max_freq').write_text(str(freq))
        (root/'topology/core_id').write_text(str(core))
        (root/'topology/physical_package_id').write_text('0')
    assert performance_cpus({0,1,2,3},tmp_path)=={0,2}
    assert performance_cpus({1,2,3},tmp_path)=={1,2}
    assert performance_cpus({3},tmp_path)=={3}
    assert performance_cpus({0,4},tmp_path)=={0,4}
    assert performance_cpus({0,1,2,3,5,6},tmp_path)=={0,2,5}
    assert performance_cpus({0,1,3,6},tmp_path)=={0,6}


def test_pending_sync_does_not_block_independent_slow_reset():
    from dataclasses import make_dataclass
    probe=PolicySafetyProbe.__new__(PolicySafetyProbe)
    probe.selected_ball_clock_source='camera'
    probe.slow_reset_trajectory=SimpleNamespace(active=True)
    probe.ros_bridge=SimpleNamespace(ball_clock_status=lambda _:dict(required=True,ready=False))
    probe._require_publish_rearm=lambda *_,**__:pytest.fail('reset must not depend on ball clock')
    Options=make_dataclass('Options',[('ball_source',str),('publish_commands',bool)])
    assert probe._select_ball_clock(Options('camera',True)).publish_commands


def test_warming_clock_bypasses_actor_prediction_and_old_reference_hold():
    probe=PolicySafetyProbe.__new__(PolicySafetyProbe)
    probe.control_loop_tick=10;probe.now_s=1.;probe.cfg=SimpleNamespace(control_dt=.005)
    probe.command_stream_active=False
    sync=dict(required=True,ready=False,epoch=1,status='warming_up')
    probe._select_ball_source=lambda *_,**__: (
        np.zeros(3),np.zeros(3),False,'camera',None,None,dict(clockSync=sync))
    probe._reset_policy_ball_sampling_contract=lambda:None
    probe._resolve_high_ball_occlusion_state=lambda *_,**__:pytest.fail('old prediction must not run')
    probe._run_one_control_step=lambda *_,**__:pytest.fail('actor must not run')
    options=SimpleNamespace(publish_commands=False)
    assert probe._run_control_tick_locked(np.zeros(3),np.zeros(3),True,0.,options,make_snapshot=False) is None
    assert probe.control_loop_tick==11 and not probe.policy_tick_active
    assert probe.ball_estimator is None and probe.high_ball_occlusion_predictor is None


def test_web_snapshot_yields_to_control_period_budget(monkeypatch):
    probe=PolicySafetyProbe.__new__(PolicySafetyProbe)
    probe.cfg=SimpleNamespace(control_dt=.005)
    probe.control_tick_started_perf=10.
    monkeypatch.setattr('pingpong_controller.tools.interactive_policy_safety_probe.time.perf_counter',lambda:10.003)
    assert probe._defer_snapshot_for_control_budget()
    assert probe.control_snapshots_deferred==1
    monkeypatch.setattr('pingpong_controller.tools.interactive_policy_safety_probe.time.perf_counter',lambda:10.001)
    assert not probe._defer_snapshot_for_control_budget()
