"""Bound GIL contention and keep control off slower hybrid CPU cores."""
import os
from pathlib import Path
import threading


def performance_cpus(allowed, root=Path('/sys/devices/system/cpu')):
    frequencies = {}
    for cpu in allowed:
        try:
            frequencies[cpu] = int((root / f'cpu{cpu}/cpufreq/cpuinfo_max_freq').read_text())
        except (OSError, ValueError):
            return set(allowed)
    if not frequencies or max(frequencies.values()) < 1.1 * min(frequencies.values()):
        return set(allowed)
    # Hybrid performance cores may have different turbo ceilings (e.g.
    # 5.7/6.0 GHz). Restricting to only the absolute maximum leaves two cores
    # available on the local machine and produced measured runqueue delays.
    # Select the high-frequency group above the largest hardware frequency
    # gap, rather than excluding ordinary performance cores.
    ordered = sorted(set(frequencies.values()))
    _, threshold = max((upper-lower, upper) for lower, upper in zip(ordered, ordered[1:]))
    selected = set()
    physical = set()
    for cpu in sorted(allowed):
        if frequencies[cpu] < threshold:
            continue
        try:
            topology = root / f'cpu{cpu}/topology'
            core = (topology.joinpath('physical_package_id').read_text().strip(),
                    topology.joinpath('core_id').read_text().strip())
        except OSError:
            core = cpu
        if core not in physical:
            selected.add(cpu)
            physical.add(core)
    return selected or set(allowed)


def configure_control_thread(rt_priority=None):
    info = dict(status='unsupported', threadId=threading.get_native_id())
    if rt_priority is None:
        rt_priority = int(os.environ.get('PROBE_CONTROL_RT_PRIORITY', '1'))
    if not 0 <= rt_priority <= 20:
        raise ValueError('PROBE_CONTROL_RT_PRIORITY must be between 0 and 20')
    info.update(requestedRealtimePriority=rt_priority, realtimeApplied=False)
    if not hasattr(os, 'sched_getaffinity'):
        return info
    try:
        allowed = set(os.sched_getaffinity(0))
        selected = performance_cpus(allowed)
        if selected != allowed:
            os.sched_setaffinity(0, selected)  # This calling control thread only.
        info.update(status='applied' if selected != allowed else 'unchanged',
                    allowedCpus=sorted(allowed), controlCpus=sorted(os.sched_getaffinity(0)))
    except OSError as exc:
        info.update(status='unavailable', error=str(exc))
    if hasattr(os, 'sched_setscheduler'):
        try:
            if rt_priority:
                # Lowest RT round-robin priority: only this control thread,
                # never ROS/sensor workers or the native robot controller.
                # Linux RLIMIT_RTPRIO permits this without running as root.
                reset_flag = getattr(os, 'SCHED_RESET_ON_FORK', 0)
                os.sched_setscheduler(0, os.SCHED_RR | reset_flag, os.sched_param(rt_priority))
            policy = os.sched_getscheduler(0)
            priority = os.sched_getparam(0).sched_priority
            reset_flag = getattr(os, 'SCHED_RESET_ON_FORK', 0)
            info.update(schedulerPolicy=policy & ~reset_flag, schedulerRawPolicy=policy,
                        realtimePriority=priority, childSchedulerReset=bool(policy & reset_flag),
                        realtimeApplied=bool((policy & ~reset_flag) == os.SCHED_RR
                                             and priority == rt_priority and rt_priority))
        except OSError as exc:
            info.update(realtimeApplied=False, realtimeError=str(exc))
    return info
