"""Opt-in instrumentation must preserve control behavior in reset/fault states."""
import json
from types import SimpleNamespace
import pytest
from pingpong_controller.probe_timing import ProbeStageTimings

@pytest.mark.parametrize('mode,send', [(None,True),('reset',True),('policy',True),('policy',False)])
def test_stage_timing_preserves_return_and_handles_uninitialized_mode(tmp_path,mode,send):
    calls=[]
    probe=SimpleNamespace(last_control_mode=mode,command_stream_active=send,
        controller=SimpleNamespace(),recorder=SimpleNamespace(),
        _run_backend_control_tick=lambda: calls.append('tick') or 'unchanged')
    timing=ProbeStageTimings(probe,tmp_path/'stages.json')
    assert probe._run_backend_control_tick()=='unchanged'
    assert calls==['tick']
    timing.write()
    result=json.loads((tmp_path/'stages.json').read_text())
    assert len(result)==1 and next(iter(result.values()))['count']==1


def test_stage_timing_preserves_original_exception(tmp_path):
    fault=RuntimeError('original control fault')
    def control_fault():raise fault
    probe=SimpleNamespace(last_control_mode=None,command_stream_active=True,
        controller=SimpleNamespace(),recorder=SimpleNamespace(),
        _run_backend_control_tick=control_fault)
    ProbeStageTimings(probe,tmp_path/'stages.json')
    with pytest.raises(RuntimeError) as raised:probe._run_backend_control_tick()
    assert raised.value is fault
