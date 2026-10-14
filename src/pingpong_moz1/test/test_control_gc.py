"""Cyclic collection may never run while a policy or reset can publish."""
import threading
from types import SimpleNamespace

import pytest

from pingpong_controller import control_gc


@pytest.fixture
def collector(monkeypatch):
    calls=[]
    fake=SimpleNamespace(isenabled=lambda:True,
        disable=lambda:calls.append('disable'), enable=lambda:calls.append('enable'),
        collect=lambda:calls.append('collect'))
    monkeypatch.setattr(control_gc,'gc',fake)
    instance=control_gc.DeferredControlGC(idle_period_s=100)
    instance.start()
    return instance,calls


@pytest.mark.parametrize('publish,stream',[(True,False),(True,True),(False,True)])
def test_no_collection_during_policy_or_reset_publication(collector,publish,stream):
    instance,calls=collector
    probe=SimpleNamespace(lock=threading.RLock(),runtime_options=SimpleNamespace(
        publish_commands=publish),command_stream_active=stream)
    assert not instance.collect_if_idle(probe)
    assert calls==['disable']


def test_idle_collection_is_bounded_and_original_state_restored(collector):
    instance,calls=collector
    probe=SimpleNamespace(lock=threading.RLock(),runtime_options=None,command_stream_active=False)
    assert instance.collect_if_idle(probe)
    assert not instance.collect_if_idle(probe)
    assert instance.payload()['idleCollections']==1
    instance.close()
    assert calls==['disable','collect','enable']


def test_rechecks_arm_state_after_lock_acquisition(collector):
    instance,calls=collector
    probe=SimpleNamespace(runtime_options=SimpleNamespace(publish_commands=False),command_stream_active=False)
    class ArmOnLock:
        def __enter__(self):probe.runtime_options.publish_commands=True
        def __exit__(self,*args):pass
    probe.lock=ArmOnLock()
    assert not instance.collect_if_idle(probe)
    assert calls==['disable']


def test_preserves_previously_disabled_gc(monkeypatch):
    calls=[]
    monkeypatch.setattr(control_gc,'gc',SimpleNamespace(isenabled=lambda:False,
        disable=lambda:calls.append('disable'),enable=lambda:calls.append('enable')))
    instance=control_gc.DeferredControlGC();instance.start();instance.close()
    assert calls==['disable']


@pytest.mark.parametrize('backend',[None,lambda *args:None])
def test_live_projection_rejects_missing_or_disabled_jit(monkeypatch,backend):
    from pingpong_controller import fplc_numpy
    monkeypatch.setattr(fplc_numpy,'_dykstra_single_compiled',backend)
    with pytest.raises(RuntimeError,match='200 Hz requires the compiled Numba projection'):
        fplc_numpy.warm_fplc_projection(require_compiled=True)


def test_live_projection_warmup_reports_compiled_gil_free_backend():
    from pingpong_controller import fplc_numpy
    assert fplc_numpy.warm_fplc_projection(require_compiled=True)=={
        'backend':'numba_nopython','compiled':True,'releasesGil':True}
