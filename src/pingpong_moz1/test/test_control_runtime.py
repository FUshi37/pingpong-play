"""Only the calling policy thread may receive its requested RT scheduler."""
from types import SimpleNamespace
import pytest
from pingpong_controller import control_runtime as runtime


def mock_scheduler(monkeypatch, *, denied=False):
    calls=[]
    monkeypatch.setattr(runtime.os,'sched_getaffinity',lambda pid:{0,2})
    monkeypatch.setattr(runtime,'performance_cpus',lambda allowed:allowed)
    def set_policy(pid, policy, param):
        calls.append((pid,policy,param.sched_priority))
        if denied:raise PermissionError('RLIMIT_RTPRIO unavailable')
    monkeypatch.setattr(runtime.os,'sched_setscheduler',set_policy)
    monkeypatch.setattr(runtime.os,'sched_getscheduler',lambda pid:runtime.os.SCHED_RR)
    monkeypatch.setattr(runtime.os,'sched_getparam',lambda pid:SimpleNamespace(sched_priority=1))
    return calls


def test_default_rt_scheduler_targets_only_calling_thread(monkeypatch):
    monkeypatch.delenv('PROBE_CONTROL_RT_PRIORITY',raising=False)
    calls=mock_scheduler(monkeypatch)
    info=runtime.configure_control_thread()
    assert calls==[(0,runtime.os.SCHED_RR | getattr(runtime.os,'SCHED_RESET_ON_FORK',0),1)]
    assert info['realtimeApplied'] and info['realtimePriority']==1


def test_missing_rt_permissions_report_failure(monkeypatch):
    calls=mock_scheduler(monkeypatch,denied=True)
    info=runtime.configure_control_thread(rt_priority=1)
    assert not info['realtimeApplied']
    assert 'RLIMIT_RTPRIO' in info['realtimeError']
    assert len(calls)==1


def test_explicit_benchmark_rt_optout_does_not_change_scheduler(monkeypatch):
    calls=mock_scheduler(monkeypatch)
    info=runtime.configure_control_thread(rt_priority=0)
    assert calls==[] and not info['realtimeApplied']


@pytest.mark.parametrize('priority',[-1,21])
def test_out_of_range_priority_rejected_before_syscalls(monkeypatch,priority):
    calls=mock_scheduler(monkeypatch)
    with pytest.raises(ValueError):runtime.configure_control_thread(rt_priority=priority)
    assert calls==[]


def test_real_command_gate_closes_when_rt_setup_failed():
    import threading
    from dataclasses import make_dataclass
    from pingpong_controller.tools.interactive_policy_safety_probe import PolicySafetyProbe
    probe=PolicySafetyProbe.__new__(PolicySafetyProbe)
    probe.lock=threading.RLock()
    probe.cfg=SimpleNamespace(ros_bridge=True)
    Options=make_dataclass('Options',[('publish_commands',bool),('use_ros_state',bool)])
    probe.runtime_options=Options(True,True)
    probe.control_thread_runtime=dict(requestedRealtimePriority=1,realtimeApplied=False)
    probe.last_ball_pos=probe.last_observed_ball_pos=probe.last_observed_ball_vel=[0.,0.,0.]
    calls=[]
    probe._require_publish_rearm=lambda reason,**kw:calls.append(reason)
    probe._mark_policy_paused=lambda:calls.append('paused')
    probe._publish_hold_on_command_disable=lambda:calls.append('hold') or False
    probe.snapshot=lambda *a,**kw:dict(mode=a[0],published=kw['publish_commands'])
    probe._run_backend_control_tick()
    assert calls==['control_realtime_unavailable','paused','hold']
    assert probe.latest_snapshot_payload==dict(mode='control_schedule_hold',published=False)
