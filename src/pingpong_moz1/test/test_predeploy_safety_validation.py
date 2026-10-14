"""The acceptance gate itself must reject false confidence and missing evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'tools'/f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
gate = load('predeploy_safety_validation')
monitor = load('predeploy_ros_observer')
LIMITS = json.loads((ROOT/'config/predeploy_safety_cases.json').read_text())['thresholds']


def samples(moving=True, stopping=False):
    result = []
    for i in range(401):
        t = 100.+i*.005
        value = min(i, 200)*.0001 if moving else 0.
        dq = .02 if moving and (not stopping or i < 200) else 0.
        result.append(dict(kind='joint', receiptMonotonicS=t, stampS=t-9.384,
            q=[value]*7, dq=[dq]*7, positionM=[0., -.49, 1.025+value*.1], relativeRpDeg=[0., 0.]))
        result.append(dict(kind='command', receiptMonotonicS=t+.0001, stampS=t,
                           q=[i*.0001]*7))
    return result


def test_assessor_unresponsive_controller_cannot_pass():
    verdict = gate.assess_motion(samples(False), LIMITS, require_motion=True)
    assert verdict['status'] == 'BLOCKED'
    assert 'controller_did_not_execute_commands' in verdict['faults']


@pytest.mark.parametrize('key,value', [('q', [np.nan]*7), ('dq', [np.inf]*7),
    ('dq', [0.]*6), ('positionM', [0., -.49, None]), ('relativeRpDeg', [0.])])
def test_assessor_invalid_state_cannot_pass(key, value):
    rows = samples()
    rows[100][key] = value
    assert gate.assess_motion(rows, LIMITS)['status'] == 'FAIL'


@pytest.mark.parametrize('clock', ['stampS', 'receiptMonotonicS'])
def test_assessor_backwards_or_duplicate_clock_cannot_pass(clock):
    rows = samples()
    rows[100][clock] = rows[98][clock]
    assert gate.assess_motion(rows, LIMITS)['status'] == 'FAIL'


def test_assessor_real_envelope_still_applies_after_stop():
    rows = samples(stopping=True)
    rows[-2]['relativeRpDeg'] = [0., 5.74]
    verdict = gate.assess_motion(rows, LIMITS, stop_at=100.3)
    assert verdict['status'] == 'FAIL'
    assert 'measured_attitude_outside_real_envelope' in verdict['faults']


def test_assessor_stop_drift_or_rebound_cannot_pass():
    rows = samples(stopping=True)
    rows[-2]['positionM'][0] = .085
    verdict = gate.assess_motion(rows, LIMITS, stop_at=100.3)
    assert verdict['status'] == 'FAIL'
    assert 'stop_displacement_exceeded' in verdict['faults']
    rows = samples(stopping=True)
    for row in rows[-40::2]:
        row['dq'] = [.3]*7
    verdict = gate.assess_motion(rows, LIMITS, stop_at=100.3)
    assert 'actual_motion_did_not_settle_in_time' in verdict['faults']


def test_assessor_stationary_stop_does_not_prove_moving_stop():
    verdict = gate.assess_motion(samples(False), LIMITS, stop_at=100.3, require_moving_stop=True)
    assert verdict['status'] == 'BLOCKED'
    assert 'stop_not_triggered_during_actual_motion' in verdict['faults']


def test_assessor_constant_republished_commands_are_not_200hz_updates():
    rows = samples()
    for r in rows:
        if r['kind'] == 'command':
            r['q'] = [0.]*7
    verdict = gate.assess_motion(rows, LIMITS, require_motion=True)
    assert verdict['status'] == 'FAIL'
    assert verdict['metrics']['commandValueUpdateHz'] == 0


def test_assessor_missing_feedback_or_stop_tail_is_incomplete():
    assert gate.assess_motion([], LIMITS)['status'] == 'BLOCKED'
    assert gate.assess_motion(samples(), LIMITS, stop_at=101.8)['status'] == 'BLOCKED'
    assert gate.assess_motion(samples(), LIMITS, stop_at=99.)['status'] == 'BLOCKED'


def reset_samples():
    rows = []
    for i in range(601):
        t = i*.005
        u = min(t/2., 1.)
        blend = 3*u*u-2*u*u*u
        dq = .1*(6*u-6*u*u)/2 if t < 2 else 0.
        rows.append(dict(kind='joint', receiptMonotonicS=100.+t, stampS=90.616+t,
            q=[.1*blend]*7, dq=[dq]*7, positionM=[-.28*(1-blend), -.49, 1.025],
            relativeRpDeg=[12*(1-blend)]*2))
    return rows


def test_assessor_explicit_native_reset_can_prepare_from_boot_pose():
    verdict = gate.assess_reset(reset_samples(), LIMITS, [0.]*7, [.1]*7, started_inside=False)
    assert verdict['status'] == 'PASS', verdict
    assert not verdict['metrics']['resetStartedInsidePolicyEnvelope']
    # The same trajectory is never valid normal policy execution.
    assert gate.assess_motion(reset_samples(), LIMITS, need_commands=False)['status'] == 'FAIL'


def test_assessor_reset_never_hides_overspeed_corridor_or_bad_final_pose():
    for fault in ('speed', 'corridor', 'endpoint'):
        rows = reset_samples()
        if fault == 'speed':
            rows[100]['dq'] = [1.]*7
        elif fault == 'corridor':
            rows[100]['q'] = [1.]*7
        else:
            rows[-1]['relativeRpDeg'] = [12., 12.]
        assert gate.assess_reset(rows, LIMITS, [0.]*7, [.1]*7, started_inside=False)['status'] == 'FAIL'


def test_assessor_already_at_native_reset_may_correctly_stay_stationary():
    rows = samples(False)
    verdict = gate.assess_reset(rows, LIMITS, [0.]*7, [0.]*7, started_inside=True)
    assert verdict['status'] == 'PASS', verdict
    verdict = gate.assess_reset(rows, LIMITS, [0.]*7, [.1]*7, started_inside=True)
    assert verdict['status'] == 'FAIL'
    assert 'reset_did_not_reach_native_target' in verdict['faults']


def test_assessor_separate_clock_offset_does_not_create_fake_latency():
    limits = copy.deepcopy(LIMITS)
    limits['minimum_stop_entry_speed_deg_s'] = .5
    rows = samples(stopping=True)
    verdict = gate.assess_motion(rows, limits, stop_at=100.999, require_moving_stop=True)
    assert verdict['status'] == 'BLOCKED'  # the requested 1.5 s tail is not covered
    for r in rows:
        if r['kind'] == 'joint':
            r['receiptMonotonicS'] -= .8
    # Dedicated complete stop, moving > 5 deg/s until 100.3, then zero.
    rows = []
    for i in range(401):
        t = 100.+i*.005
        value = min(i, 59)*.0005
        rows.append(dict(kind='joint', receiptMonotonicS=t, stampS=t-9.384,
            q=[value]*7, dq=[.1 if i < 60 else 0.]*7,
            positionM=[0., -.49, 1.025+value*.1], relativeRpDeg=[0., 0.]))
        rows.append(dict(kind='command', receiptMonotonicS=t+.0001, stampS=t, q=[value]*7))
    verdict = gate.assess_motion(rows, LIMITS, stop_at=100.296, require_moving_stop=True)
    assert verdict['status'] == 'PASS', verdict
    assert verdict['metrics']['settledAfterS'] < .01


@pytest.mark.parametrize('status', ['FAIL', 'BLOCKED', 'NOT_COVERED', 'NOT_RUN'])
def test_aggregate_critical_missing_or_failed_cannot_pass(status):
    cases = [dict(id='a', stage='offline', critical=True), dict(id='b', stage='builtin', critical=True)]
    verdict = gate.aggregate(cases, {'a': {'status': 'PASS'}, 'b': {'status': status}}, 'builtin')
    assert not verdict['selectedStagePassed'] and not verdict['builtinSimulationGatePassed']
    assert not verdict['hardwareReleaseReady']


def test_aggregate_offline_pass_never_certifies_hardware_or_live():
    cases = [dict(id='a', stage='offline', critical=True), dict(id='b', stage='builtin', critical=True)]
    verdict = gate.aggregate(cases, {'a': {'status': 'PASS'}}, 'offline')
    assert verdict['selectedStagePassed'] and not verdict['builtinSimulationGatePassed']
    assert not verdict['hardwareReleaseReady']


def test_aggregate_unmapped_suite_failure_is_also_a_gate_failure():
    cases = [dict(id='a', stage='offline', critical=True)]
    verdict = gate.aggregate(cases, {'a': {'status': 'PASS'}, 'offline_suite': {'status': 'FAIL'}}, 'offline')
    assert not verdict['selectedStagePassed']
    assert verdict['failedCases'] == ['offline_suite']


@pytest.mark.parametrize('last', [None, np.nan, np.inf, 10.1, 9.8])
def test_heartbeat_loss_invalid_and_future_timestamp_trigger_stop(last):
    assert monitor.heartbeat_expired(10., last)


def test_heartbeat_fresh_monotonic_state_is_not_expired():
    assert not monitor.heartbeat_expired(10., 9.99)


def test_manifest_critical_gaps_are_explicit_and_limits_are_not_widened():
    data = json.loads((ROOT/'config/predeploy_safety_cases.json').read_text())
    cases = data['cases']
    assert len({c['id'] for c in cases}) == len(cases)
    assert len(cases) >= 40
    assert all(c['critical'] for c in cases)
    assert any(c['id'] == 'live_backend_crash' for c in cases)
    assert any(c['id'] == 'hw_emergency_stop' for c in cases)
    assert gate.FPLC_REAL_LOWER_M == (-.1, -.6, .95)
    assert gate.FPLC_REAL_UPPER_M == (.1, -.4, 1.2)
    assert gate.FPLC_REAL_ROLL_PITCH_LIMIT_DEG == (4., 4.)


def test_manifest_builtin_requires_explicit_simulation_confirmation():
    with pytest.raises(SystemExit) as error:
        gate.main(['--builtin-sim'])
    assert error.value.code == 2
