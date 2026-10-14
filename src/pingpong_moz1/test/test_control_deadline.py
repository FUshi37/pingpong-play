"""Physical policy clocks and the probe's existing fail-closed intervention."""
from dataclasses import dataclass
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from pingpong_controller.control_deadline import PolicyDeadlineMonitor
from pingpong_controller.tools.interactive_policy_safety_probe import PolicySafetyProbe


def test_200hz_and_short_jitter_do_not_fault():
    monitor=PolicyDeadlineMonitor()
    t=0.
    assert monitor.start(t,True,'fplc_gpu1_v172') is None
    for interval in ([.005]*100 + [.009,.001] + [.005]*100):
        t+=interval
        assert monitor.start(t,True,'fplc_gpu1_v172') is None
    assert monitor.payload()['activePolicyHz']==pytest.approx(200.)


@pytest.mark.parametrize('interval',[.006,.008])
def test_sustained_slow_policy_faults(interval):
    monitor=PolicyDeadlineMonitor()
    monitor.start(0.,True,'fplc_gpu1_v172')
    for index in range(1,80):
        assert monitor.start(index*interval,True,'fplc_gpu1_v172') is None
    fault=monitor.start(80*interval,True,'fplc_gpu1_v172')
    assert fault['reason']=='policy_control_deadline_exceeded'
    assert fault['windowHz']==pytest.approx(1/interval)


def test_long_gap_profile_switch_and_disarmed_reset():
    monitor=PolicyDeadlineMonitor()
    monitor.start(1.,True,'fplc_gpu1_v172')
    assert monitor.start(1.030,True,'fplc_gpu1_v172')['intervalMs']==pytest.approx(30.)
    assert monitor.start(2.,True,'gpu0_qvel_real_rmp_v85') is None
    assert monitor.payload()['windowSamples']==0
    assert monitor.start(10.,False,'gpu0_qvel_real_rmp_v85') is None
    assert monitor.start(20.,True,'gpu0_qvel_real_rmp_v85') is None


def test_deadline_fault_disarms_before_inference_and_uses_measured_hold():
    @dataclass
    class Options:
        publish_commands: bool=True
        use_ros_state: bool=True
    probe=PolicySafetyProbe.__new__(PolicySafetyProbe)
    probe.lock=threading.RLock()
    probe.runtime_options=Options()
    probe.active_policy_profile='fplc_gpu1_v172'
    probe.policy_tick_active=True
    probe.policy_deadline_monitor=SimpleNamespace(start=lambda *args:{'reason':'test_gap'})
    events=[]
    probe._require_publish_rearm=lambda reason,details:events.append(('rearm',reason,details))
    probe._mark_policy_paused=lambda:events.append('pause')
    probe.controller=SimpleNamespace(invalidate_after_intervention=lambda:events.append('invalidate'))
    probe._publish_hold_on_command_disable=lambda:events.append('measured_hold') or True
    probe._run_control_tick_locked=lambda *a,**kw:pytest.fail('must not infer after timing fault')
    probe.snapshot=lambda *a,**kw:kw
    probe.last_ball_pos=probe.last_observed_ball_pos=probe.last_observed_ball_vel=np.zeros(3)
    probe._run_backend_control_tick()
    assert events==[('rearm','policy_control_timing_fault',{'reason':'test_gap'}),
                    'pause','invalidate','measured_hold']
    assert not probe.policy_tick_active
    assert probe.latest_snapshot_payload['runtime_options'].publish_commands is False
    assert probe.latest_snapshot_payload['publish_commands'] is False
