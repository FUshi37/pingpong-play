#!/usr/bin/env python3
"""Offline 200 Hz actor/history/projection timing; never initializes ROS/devices."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

# Match the production FPLC launch preset before NumPy initializes BLAS.
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/pingpong_moz1'))

import numpy as np

from pingpong_controller.fplc_deployment import SIMCODE10_PROFILES
from pingpong_controller.fplc_policy import FPLCPolicyController


def benchmark(profile: str, ticks: int) -> dict:
    controller = FPLCPolicyController(observation_profile=profile)
    q = controller.native_reset_arm_q.copy()
    dq = np.zeros(7, np.float32)
    # Compile/provision all paths before measuring the production tick budget.
    for tick in range(50):
        controller.predict(np.array([0., -.49, 1.15], np.float32), np.zeros(3, np.float32),
                           arm_q=q, arm_dq=dq, ball_valid=True, ball_observed=True)
    controller.reset()
    calls = dict(actor=0, projection=0)
    actor, projection = controller._policy_action, controller._project_reference

    def counted_actor(observation):
        calls['actor'] += 1
        return actor(observation)

    def counted_projection(measured, nominal):
        calls['projection'] += 1
        return projection(measured, nominal)

    controller._policy_action = counted_actor
    controller._project_reference = counted_projection
    starts, costs = [], []
    changed_references = 0
    previous = q.copy()
    origin = time.perf_counter()
    for tick in range(ticks):
        remaining = origin + tick * .005 - time.perf_counter()
        if remaining > 0.:
            time.sleep(remaining)
        t = tick * .005
        ball = np.array([.002 * np.sin(t), -.49, 1.13 + .05 * np.sin(7 * t)], np.float32)
        velocity = np.array([.002 * np.cos(t), 0., .35 * np.cos(7 * t)], np.float32)
        started = time.perf_counter()
        command = controller.predict(ball, velocity, arm_q=q, arm_dq=dq,
                                     ball_valid=True, ball_observed=True,
                                     direct_ball_position_m=ball, direct_ball_age_s=.003,
                                     direct_ball_stamp_s=t, ball_observation_rate_hz=90.)
        costs.append((time.perf_counter() - started) * 1000)
        starts.append(started)
        assert command.shape == (7,) and np.isfinite(command).all()
        changed_references += int(not np.array_equal(command, previous))
        previous = command.copy()
        next_q = q + np.float32(.1) * (command - q)
        dq, q = (next_q - q) / np.float32(.005), next_q
    elapsed = starts[-1] - starts[0]
    intervals = np.diff(starts) * 1000
    assert calls == dict(actor=ticks, projection=ticks)
    assert controller._fplc_tick == ticks
    assert controller.ball_state_refresh_count == round(ticks * 90 / 200)
    return dict(profile=profile, actorDim=controller._expected_obs_dim,
                checkpointSha256=controller._checkpoint_sha256,
                controlTicks=ticks, actorCalls=calls['actor'], projectionCalls=calls['projection'],
                observedControlHz=(ticks - 1) / elapsed,
                nominalBallStateHz=90., ballStateRefreshes=controller.ball_state_refresh_count,
                nominalControlHz=200., nominalHistoryAdvanceHz=200.,
                predictMs=dict(mean=float(np.mean(costs)), p99=float(np.percentile(costs, 99)),
                               maximum=float(np.max(costs))),
                controlIntervalMs=dict(p99=float(np.percentile(intervals, 99)),
                                       maximum=float(np.max(intervals))),
                changedReferenceCount=changed_references,
                unchangedReferenceCount=ticks - changed_references)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=5, choices=range(1, 61))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(
        scope='Offline real actor + observation/history + original 24-sweep projection; synthetic inputs',
        hardwareCommandsSent=0, rosInitialized=False, receiverFrequencyMeasured=False,
        fullCameraMocapRecordingLoadMeasured=False,
        models=[benchmark(profile, args.seconds * 200)
                for profile in ('fplc_gpu1_v172', *SIMCODE10_PROFILES)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    for result in report['models']:
        print(f"{result['profile']}: {result['observedControlHz']:.3f} Hz, "
              f"predict P99={result['predictMs']['p99']:.3f} ms")


if __name__ == '__main__':
    main()
