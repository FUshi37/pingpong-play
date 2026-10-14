"""Run one trainer under verified cgroup limits and record host pressure.

For this 32 GiB dual-GPU host, each trainer is assigned an explicit throttling
and hard-cap contract with no swap. The legacy default is 10/12 GiB with a
9.5-GiB proactive stop; launchers may opt into a validated contract up to
13/14 GiB with a correspondingly lower proactive stop. The legacy host
runtime guard is 6 GiB with one-sample triggering. Launchers for the measured
dual-1024 workload may explicitly select a 4 GiB, three-sample guard after
serial probes demonstrate zero PSI and cgroup memory events at the 5--6 GiB
boundary. A new trainer still requires the
same 10 GiB startup headroom as the successful GPU1-QACC reference.
Memory-triggered interrupts allow a long
in-flight XLA call to return before checkpointing; the cgroup high/max limits
continue to isolate a genuinely growing process. Explicit terminal stops keep
the shorter escalation window.
"""

from __future__ import annotations

import argparse
import datetime
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

GIB = 1024 ** 3
SCOPE_MEMORY_HIGH_BYTES = 10 * GIB
SCOPE_MEMORY_MAX_BYTES = 12 * GIB
SCOPE_STOP_BYTES = int(9.5 * GIB)
MAX_ALLOWED_MEMORY_HIGH_BYTES = 13 * GIB
MAX_ALLOWED_MEMORY_MAX_BYTES = 14 * GIB
HOST_STOP_AVAILABLE_BYTES = 6 * GIB
HOST_START_AVAILABLE_BYTES = 10 * GIB
MIN_HOST_STOP_AVAILABLE_BYTES = 4 * GIB
MAX_HOST_STOP_AVAILABLE_BYTES = HOST_STOP_AVAILABLE_BYTES
HOST_STOP_STREAK_SAMPLES = 1
MIN_HOST_STOP_STREAK_SAMPLES = 1
MAX_HOST_STOP_STREAK_SAMPLES = 10
PRESSURE_SOME_AVG10_LIMIT = 10.0
PRESSURE_FULL_AVG10_LIMIT = 5.0
PRESSURE_STREAK_SAMPLES = 3
SCOPE_STOP_STREAK_SAMPLES = 3
SIGTERM_GRACE_S = 10.0
SIGKILL_GRACE_S = 20.0
MEMORY_CHECKPOINT_SIGTERM_GRACE_S = 120.0
MEMORY_CHECKPOINT_SIGKILL_GRACE_S = 180.0
SAMPLE_PERIOD_S = 1.0
OUTER_SCOPE_GRACE_S = 8.0


@dataclass(frozen=True)
class ScopeMemoryLimits:
    """Kernel and proactive-stop limits for one protected trainer."""

    high_bytes: int
    max_bytes: int
    stop_bytes: int

    @classmethod
    def from_gib(
        cls,
        *,
        high_gib: float,
        max_gib: float,
        stop_gib: float,
    ) -> "ScopeMemoryLimits":
        """Build and validate an explicit GiB memory contract."""

        limits = cls(
            high_bytes=int(float(high_gib) * GIB),
            max_bytes=int(float(max_gib) * GIB),
            stop_bytes=int(float(stop_gib) * GIB),
        )
        if not (
            0 < limits.stop_bytes < limits.high_bytes < limits.max_bytes
            and limits.high_bytes <= MAX_ALLOWED_MEMORY_HIGH_BYTES
            and limits.max_bytes <= MAX_ALLOWED_MEMORY_MAX_BYTES
        ):
            raise ValueError(
                "unsafe requested memory contract: require "
                "0 < stop < high <= 13 GiB and high < max <= 14 GiB"
            )
        return limits


DEFAULT_SCOPE_MEMORY_LIMITS = ScopeMemoryLimits(
    high_bytes=SCOPE_MEMORY_HIGH_BYTES,
    max_bytes=SCOPE_MEMORY_MAX_BYTES,
    stop_bytes=SCOPE_STOP_BYTES,
)

CHILD_ENVIRONMENT_NAMES = frozenset(
    {
        'ACCEPTANCE_TEXT',
        'ACKNOWLEDGE_INCOMPLETE_RMP_DR_EVIDENCE',
        'AUDIT_TEXT',
        'BALL_MASS_MEASUREMENT_MANIFEST',
        'CMDP_COST_MODE',
        'CMDP_ENABLED',
        'CMDP_MIGRATE_COST_CRITICS',
        'CMDP_RESET_COST_STATE',
        'CONFIRM_GPU0_READY',
        'CONFIRM_GPU1_READY',
        'COURSE_LABEL',
        'ENABLE_TRANSACTION_GUARD',
        'FEASIBILITY_VERDICT',
        'EXPECTED_SOURCE_SHA256',
        'GPU0_V67_PROGRESS_CSV',
        'GPU0_V69_PROGRESS_CSV',
        'GPU0_V72_PROGRESS_CSV',
        'GPU0_V74_PROGRESS_CSV',
        'GPU0_V76_PROGRESS_CSV',
        'GPU0_V78_PROGRESS_CSV',
        'GPU0_V81_PROGRESS_CSV',
        'GPU0_V85_PROGRESS_CSV',
        'GPU0_V88_PROGRESS_CSV',
        'GPU0_V89_PROGRESS_CSV',
        'GPU0_V93_PROGRESS_CSV',
        'GPU0_V95_PROGRESS_CSV',
        'GPU0_V97_PROGRESS_CSV',
        'GPU0_V99_PROGRESS_CSV',
        'GPU0_V102_PROGRESS_CSV',
        'GPU0_V104_PROGRESS_CSV',
        'GPU0_V104_BASE_STEP',
        'GPU0_V107_PROGRESS_CSV',
        'GPU0_V109_PROGRESS_CSV',
        'GPU0_V111_PROGRESS_CSV',
        'GPU0_V112_PROGRESS_CSV',
        'GPU0_V113_PROGRESS_CSV',
        'GPU0_V133_PROGRESS_CSV',
        'GPU0_LEARNING_RATE',
        'GPU0_TARGET_KL',
        'GPU0_MIN_LOG_STD',
        'GPU0_BASE_STEP',
        'MAX_GPU0_ORDER_PROJECTION_FALLBACK_RATE',
        'GPU0_REQUIRED_STAGE_PREFIX',
        'GPU0_ORDER_REQUIRED',
        'GPU0_STEPS_PER_UPDATE',
        'GPU1_BATCH_PRESET',
        'GPU1_LEARNING_RATE',
        'GPU1_RESUME_MODE',
        'LAUNCHER_PATH_OVERRIDE',
        'MAX_STAGES',
        'MAX_STAGE_UPDATES',
        'MAX_IN_MEMORY_STAGE_HISTORY_ROWS',
        'MINIBATCH_SIZE',
        'MIN_GPU0_V104_UPDATES',
        'MIN_GPU0_V107_UPDATES',
        'MIN_GPU0_V109_UPDATES',
        'MIN_GPU0_V111_UPDATES',
        'MIN_GPU0_V112_UPDATES',
        'MIN_GPU0_V113_UPDATES',
        'MIN_GPU0_V133_UPDATES',
        'N_ENVS',
        'N_STEPS',
        'PYTHON_BIN',
        'PROFILE',
        'RMP_OUTPUT_REPLAY_REPORT',
        'RECOVER_LOCAL_U50',
        'RECOVER_MID_U250',
        'RECOVER_NEXT_STAGE_REEXEC',
        'RECOVER_STAGE_REEXEC',
        'RUN_DATE',
        'RUN_DIR',
        'RUN_ID',
        'RUN_SEED',
        'RESUME_STAGE',
        'SAME_STAGE_REEXEC_EVERY_UPDATES',
        'SOURCE_CHECKPOINT',
        'SOURCE_LABEL',
        'SOURCE_PROGRESS_CSV',
        'STAGE_REEXEC_RESUME_FROM',
        'STAGE_REEXEC_EXPECTED_SHA256',
        'STAGE_REEXEC_START_STAGE',
        'STOP_AFTER_STAGE_NAME',
        'TRAINER_REPAIR_FLAG',
        'TRANSACTION_EVAL_N_ENVS',
        'TRANSACTION_EVAL_SEEDS',
        'TRANSACTION_FIXED_XY_SEED_AGGREGATION',
        'TRANSACTION_INTERVAL_UPDATES',
        'TRANSACTION_MAX_QACC_EXCEED_FRACTION',
        'TRANSACTION_BACKTRACK_REJECTED_CANDIDATE',
        'TRANSACTION_REVALIDATE_INCUMBENT',
        'TRANSACTION_MAX_QVEL_EXCEED_FRACTION',
        'TRANSACTION_SECONDARY_GAP_TOLERANCE',
        'WANDB_ID',
        'WANDB_ID_PREFIX',
        'WANDB_METHOD_TAG',
        'WANDB_MODE',
        'WANDB_NAME',
        'WANDB_RESET_TAG',
        'WANDB_RESUME',
        'WANDB_SOURCE_TAG',
        'WANDB_STEP_OFFSET',
        'WANDB_VERSION_TAG',
        'V79_EXPERIMENTAL_PROBE',
        'V80_EXPERIMENTAL_PROBE',
        'V81_EXPERIMENTAL_PROBE',
        'V82_EXPERIMENTAL_PROBE',
        'V96_EXPERIMENTAL_PROBE',
        'V97_EXPERIMENTAL_PROBE',
        'V98_EXPERIMENTAL_PROBE',
        'V99_EXPERIMENTAL_PROBE',
        'V100_EXPERIMENTAL_PROBE',
        'V101_EXPERIMENTAL_PROBE',
        'V102_EXPERIMENTAL_PROBE',
        'V103_EXPERIMENTAL_PROBE',
        'V104_EXPERIMENTAL_PROBE',
        'V104_PROBE_MAX_STAGES',
        'V105_EXPERIMENTAL_PROBE',
        'V106_EXPERIMENTAL_PROBE',
        'V132_EXPERIMENTAL_PROBE',
        'V134_EXPERIMENTAL_PROBE',
        'V143_EXPERIMENTAL_PROBE',
        'V144_EXPERIMENTAL_PROBE',
        'V145_EXPERIMENTAL_PROBE',
        'XML_PATH',
    }
)


def training_child_environment(
    environment: dict[str, str],
) -> tuple[str, ...]:
    """Select non-secret launcher controls lost at the service boundary."""

    return tuple(
        f'{name}={environment[name]}'
        for name in sorted(CHILD_ENVIRONMENT_NAMES)
        if name in environment
    )


def parse_child_environment(items: list[str]) -> dict[str, str]:
    """Parse the outer wrapper's declared training-control environment."""

    result: dict[str, str] = {}
    for item in items:
        name, separator, value = item.partition('=')
        if separator != '=' or name not in CHILD_ENVIRONMENT_NAMES:
            raise ValueError(f'unsupported child environment entry: {item!r}')
        result[name] = value
    return result


def scope_path() -> Path:
    """Resolve this process's unified cgroup, failing on unsupported hosts."""
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        if line.startswith('0::'):
            return Path('/sys/fs/cgroup') / line[3:].lstrip('/')
    raise ValueError('cgroup v2 is required; unbounded fallback is prohibited')


def verify_scope(
    path: Path,
    limits: ScopeMemoryLimits = DEFAULT_SCOPE_MEMORY_LIMITS,
) -> None:
    """Verify actual kernel limits rather than trusting an environment flag."""
    maximum = int((path / 'memory.max').read_text().strip())
    high = int((path / 'memory.high').read_text().strip())
    swap = int((path / 'memory.swap.max').read_text().strip())
    if not (
        0 < limits.stop_bytes < high <= limits.high_bytes
        < maximum <= limits.max_bytes
    ) or swap != 0:
        raise ValueError(f'unsafe scope limits: high={high}, max={maximum}, swap={swap}')


def parse_memory_events(raw: str) -> dict[str, int]:
    """Parse cgroup-v2 ``memory.events`` without trusting key order."""
    events: dict[str, int] = {}
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) == 2:
            events[fields[0]] = int(fields[1])
    return events


def parse_memory_pressure(raw: str) -> dict[str, dict[str, float]]:
    """Parse the ``some`` and ``full`` PSI records used by systemd-oomd."""
    pressure: dict[str, dict[str, float]] = {}
    for line in raw.splitlines():
        fields = line.split()
        if not fields:
            continue
        values: dict[str, float] = {}
        for field in fields[1:]:
            key, separator, value = field.partition('=')
            if separator:
                values[key] = float(value)
        pressure[fields[0]] = values
    return pressure


def memory_pressure_exceeded(
    pressure: dict[str, dict[str, float]],
) -> bool:
    """Return whether reclaim/stall pressure exceeds the early-stop bands."""
    return bool(
        pressure.get('some', {}).get('avg10', 0.0)
        >= PRESSURE_SOME_AVG10_LIMIT
        or pressure.get('full', {}).get('avg10', 0.0)
        >= PRESSURE_FULL_AVG10_LIMIT
    )


def pressure_reason(
    current_bytes: int,
    available_bytes: int,
    *,
    memory_pressure: dict[str, dict[str, float]] | None = None,
    pressure_streak: int = 0,
    scope_stop_streak: int = 0,
    host_stop_streak: int = HOST_STOP_STREAK_SAMPLES,
    memory_high_events_delta: int = 0,
    scope_stop_bytes: int = SCOPE_STOP_BYTES,
    host_stop_available_bytes: int = HOST_STOP_AVAILABLE_BYTES,
    host_stop_streak_samples: int = HOST_STOP_STREAK_SAMPLES,
) -> str | None:
    """Choose a stop cause with room left before the kernel hard cap."""
    if (
        available_bytes < host_stop_available_bytes
        and host_stop_streak >= host_stop_streak_samples
    ):
        return 'host_available_low'
    if (
        current_bytes >= scope_stop_bytes
        and scope_stop_streak >= SCOPE_STOP_STREAK_SAMPLES
    ):
        return 'scope_memory_stop'
    if memory_high_events_delta > 0:
        return 'scope_memory_high_event'
    if (
        memory_pressure is not None
        and memory_pressure_exceeded(memory_pressure)
        and pressure_streak >= PRESSURE_STREAK_SAMPLES
    ):
        return 'sustained_memory_pressure'
    return None


def memory_sample(path: Path) -> dict:
    """Read scope usage, host headroom, swap and pressure once per sample."""
    host = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, rest = line.split(':', 1)
        if key in ('MemAvailable', 'SwapTotal', 'SwapFree'):
            host[key] = int(rest.split()[0]) * 1024
    return {
        'time': datetime.datetime.now().astimezone().isoformat(),
        'scope_current_bytes': int((path / 'memory.current').read_text()),
        'host_available_bytes': host['MemAvailable'],
        'host_swap_used_bytes': host['SwapTotal'] - host['SwapFree'],
        'memory_events': (path / 'memory.events').read_text().strip(),
        'memory_pressure': (path / 'memory.pressure').read_text().strip(),
    }


def supervise(
    command: list[str],
    log_dir: Path,
    child_environment: dict[str, str] | None = None,
    limits: ScopeMemoryLimits = DEFAULT_SCOPE_MEMORY_LIMITS,
    host_stop_available_bytes: int = HOST_STOP_AVAILABLE_BYTES,
    host_stop_streak_samples: int = HOST_STOP_STREAK_SAMPLES,
) -> int:
    """Signal only our child process group; never find or kill other trainers."""
    path = scope_path()
    if limits == DEFAULT_SCOPE_MEMORY_LIMITS:
        verify_scope(path)
    else:
        verify_scope(path, limits)
    with (log_dir / 'memory.jsonl').open('x', buffering=1) as stream:
        initial = memory_sample(path)
        stream.write(json.dumps({'event': 'preflight', 'cgroup': str(path),
                                 'command': command, **initial}) + '\n')
        if initial['host_available_bytes'] < HOST_START_AVAILABLE_BYTES:
            raise ValueError('less than 10 GiB host headroom; trainer was not started')
        env = dict(os.environ)
        env.update(child_environment or {})
        env['PP_MEMORY_SCOPE_ACTIVE'] = 'YES'
        child = subprocess.Popen(command, env=env, start_new_session=True)
        stop_at = None
        stop_reason = None
        term_sent = False
        kill_sent = False
        external_signal = None
        previous_events = parse_memory_events(initial.get('memory_events', ''))
        pressure_streak = 0
        scope_stop_streak = 0
        host_stop_streak = 0

        def request_stop(signum, _frame):
            nonlocal external_signal
            external_signal = signum

        previous = {sig: signal.signal(sig, request_stop)
                    for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            while child.poll() is None:
                sample = memory_sample(path)
                pressure = parse_memory_pressure(sample.get('memory_pressure', ''))
                if memory_pressure_exceeded(pressure):
                    pressure_streak += 1
                else:
                    pressure_streak = 0
                if sample['scope_current_bytes'] >= limits.stop_bytes:
                    scope_stop_streak += 1
                else:
                    scope_stop_streak = 0
                if sample['host_available_bytes'] < host_stop_available_bytes:
                    host_stop_streak += 1
                else:
                    host_stop_streak = 0
                events = parse_memory_events(sample.get('memory_events', ''))
                high_delta = max(
                    0,
                    events.get('high', 0) - previous_events.get('high', 0),
                )
                previous_events = events
                sample['memory_pressure_streak'] = pressure_streak
                sample['scope_stop_streak'] = scope_stop_streak
                sample['host_stop_streak'] = host_stop_streak
                sample['memory_high_events_delta'] = high_delta
                reason = pressure_reason(
                    sample['scope_current_bytes'],
                    sample['host_available_bytes'],
                    memory_pressure=pressure,
                    pressure_streak=pressure_streak,
                    scope_stop_streak=scope_stop_streak,
                    host_stop_streak=host_stop_streak,
                    memory_high_events_delta=high_delta,
                    scope_stop_bytes=limits.stop_bytes,
                    host_stop_available_bytes=host_stop_available_bytes,
                    host_stop_streak_samples=host_stop_streak_samples,
                )
                if stop_at is None and (reason or external_signal):
                    stop_at = time.monotonic()
                    stop_reason = reason or 'external_signal'
                    sample['stop_reason'] = stop_reason
                    sample['signal'] = 'SIGINT'
                    # Signal only the trainer first. Its launcher uses a tee
                    # process substitution in the same process group; killing
                    # that pipe together with Python can turn the checkpoint
                    # handler into a BrokenPipeError before it saves state.
                    os.kill(child.pid, signal.SIGINT)
                elif stop_at is not None:
                    elapsed = time.monotonic() - stop_at
                    if stop_reason == 'external_signal':
                        term_grace_s = SIGTERM_GRACE_S
                        kill_grace_s = SIGKILL_GRACE_S
                    else:
                        term_grace_s = MEMORY_CHECKPOINT_SIGTERM_GRACE_S
                        kill_grace_s = MEMORY_CHECKPOINT_SIGKILL_GRACE_S
                    if (
                        elapsed >= kill_grace_s
                        and not kill_sent
                    ):
                        sample['signal'] = 'SIGKILL'
                        os.killpg(child.pid, signal.SIGKILL)
                        kill_sent = True
                    elif (
                        elapsed >= term_grace_s
                        and not term_sent
                    ):
                        sample['signal'] = 'SIGTERM'
                        # Keep the launcher's tee alive for the same reason as
                        # the initial SIGINT. Only the final SIGKILL tears down
                        # the complete owned process group.
                        os.kill(child.pid, signal.SIGTERM)
                        term_sent = True
                stream.write(json.dumps(sample) + '\n')
                time.sleep(SAMPLE_PERIOD_S)
            code = child.wait()
            stream.write(json.dumps({'event': 'exit', 'returncode': code,
                                     'stop_requested': stop_at is not None}) + '\n')
            return 75 if stop_at is not None else (code if code >= 0 else 128 - code)
        finally:
            # Also remove orphan descendants (e.g. tee/W&B) in our own group.
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()
            for sig, handler in previous.items():
                signal.signal(sig, handler)


def _signal_scope(unit_name: str, signum: int) -> None:
    """Signal every process in one named transient scope."""

    subprocess.run(
        [
            'systemctl', '--user', 'kill', '--kill-whom=all',
            f'--signal={signal.Signals(signum).name}', unit_name,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_scope_process_tree(
    process: subprocess.Popen,
    unit_name: str,
    initial_signal: int = signal.SIGINT,
) -> int:
    """Stop the complete named cgroup even if ``systemd-run`` exits first."""

    _signal_scope(unit_name, initial_signal)
    try:
        return process.wait(timeout=SIGTERM_GRACE_S)
    except subprocess.TimeoutExpired:
        _signal_scope(unit_name, signal.SIGTERM)
    try:
        return process.wait(timeout=max(0.1, SIGKILL_GRACE_S - SIGTERM_GRACE_S))
    except subprocess.TimeoutExpired:
        _signal_scope(unit_name, signal.SIGKILL)
        try:
            return process.wait(timeout=OUTER_SCOPE_GRACE_S)
        except subprocess.TimeoutExpired:
            subprocess.run(
                ['systemctl', '--user', 'stop', unit_name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return process.wait()


def run_named_scope(systemd_command: list[str], unit_name: str) -> int:
    """Wait for a transient scope and forward terminal signals to its cgroup."""

    process = subprocess.Popen(systemd_command)
    external_signal: int | None = None

    def request_stop(signum, _frame):
        nonlocal external_signal
        external_signal = signum

    previous = {
        sig: signal.signal(sig, request_stop)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        while process.poll() is None and external_signal is None:
            time.sleep(0.1)
        if external_signal is not None:
            stop_scope_process_tree(process, unit_name, signal.SIGINT)
            return 128 + int(external_signal)
        return process.wait()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        # A dead systemd-run client does not prove its scope is empty. The
        # unit name is unique to this wrapper, so stopping it cannot affect a
        # different trainer.
        subprocess.run(
            ['systemctl', '--user', 'stop', unit_name],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--inside', action='store_true')
    parser.add_argument('--log-dir', type=Path)
    parser.add_argument('--child-env', action='append', default=[])
    parser.add_argument('--memory-high-gib', type=float, default=10.0)
    parser.add_argument('--memory-max-gib', type=float, default=12.0)
    parser.add_argument('--memory-stop-gib', type=float, default=9.5)
    parser.add_argument(
        '--host-stop-available-gib',
        type=float,
        default=HOST_STOP_AVAILABLE_BYTES / GIB,
    )
    parser.add_argument(
        '--host-stop-streak-samples',
        type=int,
        default=HOST_STOP_STREAK_SAMPLES,
    )
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    limits = ScopeMemoryLimits.from_gib(
        high_gib=args.memory_high_gib,
        max_gib=args.memory_max_gib,
        stop_gib=args.memory_stop_gib,
    )
    host_stop_available_bytes = int(float(args.host_stop_available_gib) * GIB)
    if not (
        MIN_HOST_STOP_AVAILABLE_BYTES
        <= host_stop_available_bytes
        <= MAX_HOST_STOP_AVAILABLE_BYTES
    ):
        parser.error('--host-stop-available-gib must be in [4, 6]')
    if not (
        MIN_HOST_STOP_STREAK_SAMPLES
        <= args.host_stop_streak_samples
        <= MAX_HOST_STOP_STREAK_SAMPLES
    ):
        parser.error('--host-stop-streak-samples must be in [1, 10]')
    if args.verify_only:
        if limits == DEFAULT_SCOPE_MEMORY_LIMITS:
            verify_scope(scope_path())
        else:
            verify_scope(scope_path(), limits)
        return 0
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.log_dir is None:
        parser.error('--log-dir and a command after -- are required')
    if args.inside:
        return supervise(
            command,
            args.log_dir,
            parse_child_environment(args.child_env),
            limits,
            host_stop_available_bytes,
            args.host_stop_streak_samples,
        )
    args.log_dir.mkdir(parents=True, exist_ok=False)
    unit_base = f'pp-training-{os.getpid()}-{time.time_ns()}'
    # ``systemd-run --scope`` cannot be combined with ``--wait`` on this
    # host. A transient Type=exec service gives the trainer the same dedicated
    # cgroup and memory-controller contract while allowing this wrapper to
    # wait for and propagate the real exit status.
    unit_name = f'{unit_base}.service'
    systemd_command = [
        'systemd-run', '--user', '--quiet', '--wait', '--pipe', '--collect',
        '--service-type=exec',
        f'--working-directory={Path.cwd()}',
        '--unit', unit_base,
        '-p', f'MemoryHigh={args.memory_high_gib:g}G',
        '-p', f'MemoryMax={args.memory_max_gib:g}G',
        '-p', 'MemorySwapMax=0', '-p', 'KillMode=control-group',
        sys.executable, str(Path(__file__).resolve()), '--inside',
        '--log-dir', str(args.log_dir.resolve()),
        '--memory-high-gib', f'{args.memory_high_gib:g}',
        '--memory-max-gib', f'{args.memory_max_gib:g}',
        '--memory-stop-gib', f'{args.memory_stop_gib:g}',
        '--host-stop-available-gib', f'{args.host_stop_available_gib:g}',
        '--host-stop-streak-samples', str(args.host_stop_streak_samples),
        '--', *command,
    ]
    forwarded_environment = training_child_environment(dict(os.environ))
    separator_index = systemd_command.index('--', systemd_command.index('--inside'))
    systemd_command[separator_index:separator_index] = [
        value
        for item in forwarded_environment
        for value in ('--child-env', item)
    ]
    (args.log_dir / 'scope_command.json').write_text(json.dumps(systemd_command))
    return run_named_scope(systemd_command, unit_name)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        raise SystemExit(f'[training_memory_scope] refusing unprotected training: {exc}')
