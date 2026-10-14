"""Evidence-producing simcode9 predeployment checks. Never certify hardware safety.

No ROS imports or commands in offline/plan/analyze modes. The live driver uses
the existing probe command path without rewriting policy references or limits.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src/pingpong_moz1'))
from pingpong_controller.fplc_deployment import (  # noqa: E402
    FPLC_CONTRACTS, FPLC_PROFILES, FPLC_REAL_LOWER_M, FPLC_REAL_UPPER_M,
    FPLC_REAL_ROLL_PITCH_LIMIT_DEG,
)

MANIFEST = REPO / 'config/predeploy_safety_cases.json'
CORE_FILES = (
    'RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh', 'RUN_PREDEPLOY_SAFETY_VALIDATION.sh',
    'config/predeploy_safety_cases.json', 'config/fplc_model_contracts.json',
    'config/policy_release_manifest.json', 'config_done_monitoring.yaml',
    'tools/predeploy_safety_validation.py', 'tools/predeploy_ros_observer.py',
    'src/pingpong_moz1/pingpong_controller/fplc_deployment.py',
    'src/pingpong_moz1/pingpong_controller/fplc_policy.py',
    'src/pingpong_moz1/pingpong_controller/fplc_numpy.py',
    'src/pingpong_moz1/pingpong_controller/mjx_rl_policy.py',
    'src/pingpong_moz1/pingpong_controller/tools/interactive_policy_safety_probe.py',
    'src/pingpong_moz1/test/test_simcode9_fplc_contract.py',
    'src/pingpong_moz1/test/test_predeploy_safety_validation.py',
    'src/pingpong_moz1/test/test_predeploy_fault_injection.py',
    'movax/controller/configuration/basic/system.yaml',
)
STATES = {'PASS', 'FAIL', 'BLOCKED', 'NOT_COVERED', 'NOT_RUN'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(profile):
    files = list(CORE_FILES)
    contract = FPLC_CONTRACTS[profile]
    files.extend((contract['checkpoint_path'], contract['policy_xml_path']))
    return {name: sha256(REPO / name) if (REPO / name).is_file() else None
            for name in files}


def read_jsonl(path):
    if not Path(path).is_file():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def finite_vector(value, size):
    return isinstance(value, (list, tuple)) and len(value) == size and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in value)


def assess_motion(rows, limits, *, require_motion=False, stop_at=None,
                  require_moving_stop=False, need_commands=True):
    """Use independent raw feedback/FK throughout the stop tail, never UI q_ref.

    Header differences measure dq derivatives within ONE source clock. Receipt
    monotonic time measures stop latency; never subtract robot/client headers.
    """
    import numpy as np
    faults, metrics = [], {}
    joints = [r for r in rows if r.get('kind') == 'joint']
    commands = [r for r in rows if r.get('kind') == 'command' and r.get('q') is not None]
    if len(joints) < limits['minimum_feedback_samples']:
        return {'status': 'BLOCKED', 'faults': ['insufficient_independent_joint_feedback'],
                'metrics': {'jointSamples': len(joints)}}
    required = [('q', 7), ('dq', 7), ('positionM', 3), ('relativeRpDeg', 2)]
    if any(not finite_vector(r.get(key), size) for r in joints for key, size in required):
        return {'status': 'FAIL', 'faults': ['nonfinite_or_incomplete_measured_state'], 'metrics': {}}
    times = np.array([r.get('receiptMonotonicS', float('nan')) for r in joints], float)
    stamps = np.array([r.get('stampS', float('nan')) for r in joints], float)
    if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        return {'status': 'FAIL', 'faults': ['invalid_feedback_receipt_clock'], 'metrics': {}}
    if not np.all(np.isfinite(stamps)) or np.any(np.diff(stamps) <= 0):
        return {'status': 'FAIL', 'faults': ['duplicate_or_backwards_feedback_stamp'], 'metrics': {}}
    q = np.array([r['q'] for r in joints])
    dq = np.array([r['dq'] for r in joints])
    xyz = np.array([r['positionM'] for r in joints])
    rp = np.array([r['relativeRpDeg'] for r in joints])
    gaps = np.diff(times)
    duration = times[-1] - times[0]
    metrics.update(jointSamples=len(joints), feedbackHz=float((len(joints)-1)/duration),
                   maximumFeedbackGapMs=float(gaps.max()*1000),
                   jointSpanDeg=np.rad2deg(np.ptp(q, axis=0)).tolist(),
                   maximumMeasuredSpeedDegS=np.rad2deg(np.abs(dq).max(axis=0)).tolist(),
                   maximumRelativeRpDeg=np.abs(rp).max(axis=0).tolist(),
                   positionMinM=xyz.min(axis=0).tolist(), positionMaxM=xyz.max(axis=0).tolist())
    if gaps.max() > limits['maximum_feedback_gap_s']:
        faults.append('feedback_gap_exceeded')
    if metrics['feedbackHz'] < limits['minimum_feedback_hz']:
        faults.append('feedback_rate_below_gate')
    lower, upper = np.array(FPLC_REAL_LOWER_M), np.array(FPLC_REAL_UPPER_M)
    if np.any(xyz < lower) or np.any(xyz > upper):
        faults.append('measured_position_outside_real_workspace')
    if np.any(np.abs(rp) > np.array(FPLC_REAL_ROLL_PITCH_LIMIT_DEG)):
        faults.append('measured_attitude_outside_real_envelope')
    if np.any(np.abs(dq) > np.deg2rad(limits['joint_velocity_limit_deg_s'])):
        faults.append('measured_joint_speed_exceeded')
    # q differentiation also catches frozen/bogus zero dq feedback.
    derived_dq = np.diff(q, axis=0) / np.diff(stamps)[:, None]
    derived_ddq = np.diff(dq, axis=0) / np.diff(stamps)[:, None]
    metrics['maximumDerivedSpeedDegS'] = np.rad2deg(np.abs(derived_dq).max(axis=0)).tolist()
    metrics['maximumDerivedAccelerationDegS2'] = np.rad2deg(np.abs(derived_ddq).max(axis=0)).tolist()
    if np.any(np.abs(derived_dq) > np.deg2rad(limits['joint_velocity_limit_deg_s'])):
        faults.append('measured_position_jump_or_speed_exceeded')
    if np.any(np.abs(derived_ddq) > np.deg2rad(limits['joint_acceleration_limit_deg_s2'])):
        faults.append('measured_acceleration_exceeded')
    moving = bool(np.max(np.ptp(q, axis=0)) >= math.radians(limits['minimum_motion_span_deg']))
    metrics['actualMotionObserved'] = moving
    if require_motion and not moving:
        return {'status': 'BLOCKED', 'faults': faults + ['controller_did_not_execute_commands'],
                'metrics': metrics}
    if commands:
        if any(not finite_vector(r.get('q'), 7) for r in commands):
            faults.append('invalid_actuator_command')
        else:
            cq = np.array([r['q'] for r in commands])
            ct = np.array([r['receiptMonotonicS'] for r in commands])
            changed = np.flatnonzero(np.max(np.abs(np.diff(cq, axis=0)), axis=1) > 1e-7) + 1
            metrics.update(commandPackets=len(commands), commandValueUpdates=len(changed),
                           commandValueUpdateHz=float(len(changed)/max(ct[-1]-ct[0], 1e-9)))
            if require_motion and metrics['commandValueUpdateHz'] < limits['minimum_command_update_hz']:
                faults.append('command_value_update_rate_below_gate')
            if require_motion and len(changed) > 1:
                update_gap = float(np.diff(ct[changed]).max())
                metrics['maximumCommandValueUpdateGapMs'] = update_gap*1000
                if update_gap > limits['maximum_command_update_gap_s']:
                    faults.append('command_value_update_gap_exceeded')
    elif need_commands:
        faults.append('missing_independent_command_evidence')
    if stop_at is not None:
        before = np.flatnonzero(times <= stop_at)
        after = np.flatnonzero(times >= stop_at)
        if not len(before) or not len(after):
            return {'status': 'BLOCKED', 'faults': faults + ['missing_stop_window'], 'metrics': metrics}
        initial = before[-1]
        tail = after
        observed = float(times[-1]-stop_at)
        metrics['stopObservationS'] = observed
        initial_speed = float(np.max(np.abs(dq[initial])))
        metrics['stopInitialMaximumSpeedDegS'] = math.degrees(initial_speed)
        if require_moving_stop and initial_speed < math.radians(limits['minimum_stop_entry_speed_deg_s']):
            return {'status': 'BLOCKED', 'faults': faults + ['stop_not_triggered_during_actual_motion'],
                    'metrics': metrics}
        if observed < limits['stop_observation_s'] - .03:
            return {'status': 'BLOCKED', 'faults': faults + ['stop_tail_too_short'], 'metrics': metrics}
        displacement = float(np.linalg.norm(xyz[tail]-xyz[initial], axis=1).max())
        metrics['maximumStopDisplacementMm'] = displacement*1000
        if displacement > limits['maximum_stop_displacement_m']:
            faults.append('stop_displacement_exceeded')
        # Earliest point after which ALL feedback stays stopped (no rebound).
        slow = np.max(np.abs(dq[tail]), axis=1) <= math.radians(limits['settled_speed_deg_s'])
        stays_slow = np.logical_and.accumulate(slow[::-1])[::-1]
        candidates = np.flatnonzero(stays_slow & (times[tail] <= times[-1]-limits['settled_quiet_s']))
        latency = float(times[tail[candidates[0]]]-stop_at) if len(candidates) else None
        metrics['settledAfterS'] = latency
        if latency is None or latency > limits['maximum_stop_latency_s']:
            faults.append('actual_motion_did_not_settle_in_time')
    return {'status': 'FAIL' if faults else 'PASS', 'faults': faults, 'metrics': metrics}


def aggregate(cases, results, mode):
    """Unknown/missing/skipped cases can never turn into PASS."""
    selected = [c for c in cases if c['stage'] in ('offline', 'builtin')]
    failures = [name for name, value in results.items() if value.get('status') == 'FAIL']
    complete = not failures and all(results.get(c['id'], {}).get('status') == 'PASS' for c in selected if c['critical'])
    missing = [c['id'] for c in selected if c['critical'] and
               results.get(c['id'], {}).get('status') != 'PASS']
    stages = ('offline',) if mode == 'offline' else ('offline', 'builtin')
    stage_ok = not failures and all(results.get(c['id'], {}).get('status') == 'PASS'
                   for c in cases if c['critical'] and c['stage'] in stages)
    manual = [c['id'] for c in cases if c['stage'] == 'hardware']
    return dict(status='FAIL' if failures else ('PASS_BUILTIN_ONLY' if complete else 'INCOMPLETE'),
                selectedStagePassed=stage_ok, builtinSimulationGatePassed=complete,
                hardwareReleaseReady=False, failedCases=failures, requiredUnpassedCases=missing,
                hardwareAcceptanceRequired=manual)


def assess_reset(rows, limits, start_q, target_q, *, started_inside):
    """Explicit bounded native-reset preparation, separate from policy motion.

    A controller may boot outside the policy box. The authorized slow recovery
    records that fact; it must stay in the start/target joint corridor, within
    reset rate limits, and end stopped inside the unchanged real envelope.
    """
    import numpy as np
    joints = [r for r in rows if r.get('kind') == 'joint']
    if len(joints) < 40 or not finite_vector(start_q, 7) or not finite_vector(target_q, 7):
        return dict(status='BLOCKED', faults=['reset_evidence_missing'], metrics={})
    if any(not finite_vector(r.get(k), n) for r in joints for k, n in
           [('q', 7), ('dq', 7), ('positionM', 3), ('relativeRpDeg', 2)]):
        return dict(status='FAIL', faults=['reset_state_invalid'], metrics={})
    q, dq = np.array([r['q'] for r in joints]), np.array([r['dq'] for r in joints])
    t = np.array([r['stampS'] for r in joints])
    gaps = np.diff(t)
    if np.any(gaps <= 0) or not np.all(np.isfinite(t)):
        return dict(status='FAIL', faults=['reset_feedback_clock_invalid'], metrics={})
    ddq = np.diff(dq, axis=0)/gaps[:, None]
    vmax, amax = float(np.rad2deg(np.abs(dq).max())), float(np.rad2deg(np.abs(ddq).max()))
    margin = math.radians(limits['reset_joint_corridor_margin_deg'])
    lower, upper = np.minimum(start_q, target_q)-margin, np.maximum(start_q, target_q)+margin
    faults = []
    feedback_check = assess_motion(rows, limits, need_commands=False)
    allowed_initial = () if started_inside else (
        'measured_position_outside_real_workspace', 'measured_attitude_outside_real_envelope')
    faults.extend(f for f in feedback_check['faults'] if f not in allowed_initial)
    if vmax > limits['reset_maximum_measured_speed_deg_s']:
        faults.append('reset_actual_speed_exceeded')
    if amax > limits['reset_maximum_measured_acceleration_deg_s2']:
        faults.append('reset_actual_acceleration_exceeded')
    if np.any(q < lower) or np.any(q > upper):
        faults.append('reset_left_start_target_joint_corridor')
    if np.max(np.abs(q[-1]-target_q)) > math.radians(limits['reset_final_joint_error_deg']):
        faults.append('reset_did_not_reach_native_target')
    tail = [r for r in joints if r['receiptMonotonicS'] >= joints[-1]['receiptMonotonicS']-.3]
    tail_pose = assess_motion(rows if started_inside else tail, dict(limits, minimum_feedback_samples=20),
                              need_commands=False)
    if tail_pose['status'] != 'PASS':
        faults.extend(tail_pose['faults'])
    if any(max(map(abs, r['dq'])) > math.radians(limits['settled_speed_deg_s']) for r in tail):
        faults.append('reset_not_stopped_at_native_target')
    return dict(status='FAIL' if faults else 'PASS', faults=faults, metrics=dict(
        resetStartedInsidePolicyEnvelope=started_inside, resetInitialPositionM=joints[0]['positionM'],
        resetInitialRelativeRpDeg=joints[0]['relativeRpDeg'], resetFinalPositionM=joints[-1]['positionM'],
        resetFinalRelativeRpDeg=joints[-1]['relativeRpDeg'],
        resetMaximumMeasuredSpeedDegS=vmax, resetMaximumDerivedAccelerationDegS2=amax))


class Run:
    def __init__(self, output, profile, mode):
        self.output, self.profile, self.mode = Path(output), profile, mode
        self.output.mkdir(parents=True, exist_ok=False)
        self.manifest = json.loads(MANIFEST.read_text())
        self.cases, self.results = self.manifest['cases'], {}
        self.started = dt.datetime.now(dt.timezone.utc).isoformat()
        self.initial_hashes = fingerprint(profile)
        self.write_json('source_sha256.json', self.initial_hashes)
        self.write_json('case_manifest.json', self.manifest)
        self.persist()

    def write_json(self, name, data):
        path = self.output / name
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
        temporary.replace(path)

    def record(self, case, status, reason, **data):
        assert status in STATES
        self.results[case] = dict(status=status, reason=reason, **data)
        print(f'[{status}] {case}: {reason}', flush=True)
        self.persist()

    def persist(self):
        summary = aggregate(self.cases, self.results, self.mode)
        self.write_json('report.json', dict(schemaVersion=1, startedUtc=self.started,
            mode=self.mode, profile=self.profile, thresholds=self.manifest['thresholds'],
            summary=summary, results=self.results,
            limits=dict(positionLowerM=FPLC_REAL_LOWER_M, positionUpperM=FPLC_REAL_UPPER_M,
                        relativeRpDeg=FPLC_REAL_ROLL_PITCH_LIMIT_DEG)))
        lines = ['# 实物部署前安全验证', '', f'时间：{self.started}；模型：`{self.profile}`；模式：`{self.mode}`。', '',
            f'判定：**{summary["status"]}**。当前阶段通过：{summary["selectedStagePassed"]}。',
            '此报告不签发实物部署许可；内置仿真无法覆盖真实碰撞、断电、制动及实体急停。', '',
            '| 用例 | 阶段 | 结果 | 原因 |', '|---|---|---|---|']
        for c in self.cases:
            r = self.results.get(c['id'], {'status': 'NOT_RUN', 'reason': '尚未执行'})
            reason = str(r['reason']).replace('|', '/').replace('\n', ' ')
            lines.append(f'| {c["id"]}：{c["title"]} | {c["stage"]} | {r["status"]} | {reason} |')
        lines += ['', '原始 ROS 数据、API 请求、JUnit、源码/模型/配置 SHA-256 均保存在本目录。',
                  'FAIL / BLOCKED / NOT_COVERED / NOT_RUN 不得当作已验证。停止判定使用实际反馈及完整停止尾段。', '']
        (self.output/'report.md').write_text('\n'.join(lines))

    def offline(self):
        junit, log = self.output/'offline.junit.xml', self.output/'offline.log'
        tests = [REPO/'src/pingpong_moz1/test'/name for name in (
            'test_simcode9_fplc_contract.py', 'test_probe_timing.py',
            'test_predeploy_safety_validation.py', 'test_predeploy_fault_injection.py')]
        env = dict(os.environ, PYTHONPATH=f'{REPO}/src/pingpong_moz1:{REPO}:'+os.environ.get('PYTHONPATH', ''),
                   PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
        with log.open('w') as stream:
            process = subprocess.Popen([sys.executable, '-m', 'pytest', '-q', *map(str, tests),
                f'--junitxml={junit}', '--tb=short'], cwd=REPO, env=env, stdout=stream, stderr=stream)
            try:
                while process.poll() is None:
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        print('[RUNNING] offline fault/contract regression; detailed log: '+str(log), flush=True)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
        parsed = []
        if junit.is_file():
            parsed = list(ET.parse(junit).iter('testcase'))
        for c in self.cases:
            if c['stage'] != 'offline':
                continue
            patterns = c.get('tests', [])
            matched = [node for node in parsed if any(fnmatch.fnmatchcase(node.get('name', ''), p)
                                                       for p in patterns)]
            if not matched or any(not any(fnmatch.fnmatchcase(n.get('name', ''), p) for n in matched)
                                  for p in patterns):
                self.record(c['id'], 'NOT_COVERED', '缺少对应测试或测试收集失败', evidence='offline.junit.xml')
            elif any(node.find('failure') is not None or node.find('error') is not None for node in matched):
                self.record(c['id'], 'FAIL', '离线故障回归失败', tests=len(matched), evidence='offline.junit.xml')
            elif any(node.find('skipped') is not None for node in matched):
                self.record(c['id'], 'NOT_COVERED', '对应测试被跳过', evidence='offline.junit.xml')
            else:
                self.record(c['id'], 'PASS', '隔离测试通过；不代表实际停止运动', tests=len(matched), evidence='offline.junit.xml')
        if process.returncode:
            self.record('offline_suite', 'FAIL', f'pytest exit={process.returncode}', evidence='offline.log')
        else:
            self.record('offline_suite', 'PASS', f'{len(parsed)} 项回归通过', evidence='offline.junit.xml')

    def finalize(self):
        final_hashes = fingerprint(self.profile)
        changed = [name for name, value in self.initial_hashes.items() if final_hashes.get(name) != value]
        self.record('configuration_unchanged', 'FAIL' if changed else 'PASS',
                    '测试期间文件变化' if changed else '源码、模型及底层配置哈希未变化', changedFiles=changed)
        for c in self.cases:
            if c['id'] not in self.results:
                self.record(c['id'], 'NOT_COVERED' if c['stage'] == 'hardware'
                            or c.get('implementation') == 'pending' else 'NOT_RUN',
                            c.get('acceptance', '当前阶段未执行'))
        self.persist()
        return aggregate(self.cases, self.results, self.mode)


class ProbeClient:
    def __init__(self, url, run):
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
            raise ValueError('动态测试只连接本机 HTTP 探针')
        self.url, self.run = url.rstrip('/'), run
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.owner, self.token = 'predeploy-'+uuid.uuid4().hex, uuid.uuid4().hex
        self.observer_owner = self.owner+'-observer'
        self.strict_observer = False
        self.base = dict(clientSessionId=self.owner, controlMode='policy', policyProfile=run.profile,
            policyModel=run.profile, ballSource='drag', useRosState=True, enableSimMirror=False,
            publishCommands=False, resetInitialPose='simcode9_native', valid=True, noiseStdM=0.,
            ball=[0., -.49, 1.32], velocity=[0., 0., 0.], enableCaptureGate=True,
            enableReleaseGate=True, enableLostFailsafe=True, enableDoneMonitoring=True,
            workspaceSafetyMode='position', enableContactActionShield=False, enableCompensation=False,
            cameraDetect=False, cameraVisualize=False, cameraYolo=False, commandLeaseMs=500, dt=1/60)

    def request(self, path, payload=None, phase=''):
        logical_path = path
        if path == '/state':
            # The existing probe exposes snapshots through POST /step. Once
            # our writer owns the session, the different observer session is
            # explicitly rejected by the existing single-writer contract and
            # returns its cached snapshot WITHOUT renewing the command lease.
            # No new endpoint or control-code patch is needed.
            path = '/step'
            payload = dict(clientSessionId=self.observer_owner, publishCommands=False)
        req = urllib.request.Request(self.url+path,
            data=None if payload is None else json.dumps(payload, allow_nan=False).encode(),
            headers={'Content-Type': 'application/json'})
        start = time.monotonic()
        with self.opener.open(req, timeout=2.) as response:
            result = json.load(response)
        with (self.run.output/'api_samples.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(phase=phase, path=logical_path, httpPath=path, request=payload,
                requestMonotonicS=start, receiptMonotonicS=time.monotonic(), snapshot=result))+'\n')
        if result.get('error'):
            raise RuntimeError(str(result['error']))
        if logical_path == '/state' and self.strict_observer and not result.get('commandRequestIgnored'):
            raise RuntimeError('观察请求未被所有权检查拒绝，不能当作无租约更新的只读观察')
        return result

    def step(self, publish=False, **overrides):
        return self.request('/step', dict(self.base, publishCommands=publish,
            publishArmToken=self.token, **overrides), 'step')

    def off(self):
        state = self.step(False)
        deadline = time.monotonic()+1.
        while state.get('runtime', {}).get('publishCommands') and time.monotonic() < deadline:
            time.sleep(.02)
            state = self.step(False)
        if state.get('runtime', {}).get('publishCommands') is not False:
            raise RuntimeError('Send 关闭未确认')
        return state

    @staticmethod
    def ball(t):
        speed = math.sqrt(2*9.81*(1.14-1.047))
        u = (t + speed/9.81 + .04) % (2*speed/9.81)
        return [ .005*math.sin(t), -.49+.004*math.sin(.7*t), 1.047+speed*u-4.905*u*u ], [
            .005*math.cos(t), .0028*math.cos(.7*t), speed-9.81*u]

    def stream(self, duration, *, reset=False, extra=None):
        start, next_tick = time.monotonic(), time.monotonic()
        state = {}
        while time.monotonic()-start < duration:
            if getattr(self, 'abort', lambda: False)():
                raise RuntimeError('独立监视器已触发停止')
            pos, vel = ([0., -.49, 1.32], [0., 0., 0.]) if reset else self.ball(time.monotonic()-start)
            state = self.step(True, **(dict(ball=pos, velocity=vel) | (extra or {})))
            if state.get('commandRearm', {}).get('required') or state.get('mode') == 'done_failsafe':
                break
            if reset and state.get('resetRecovery', {}).get('settled'):
                break
            next_tick += 1/60
            time.sleep(max(0., next_tick-time.monotonic()))
        return state

    def reset(self):
        self.off()
        before = self.request('/state', phase='before_native_reset')
        snapshot = self.request('/reset', self.base, 'native_reset')
        start_q = before.get('ros', {}).get('jointPosition')
        target_q = snapshot.get('runtime', {}).get('resetInitialJointPosition')
        if not finite_vector(start_q, 7) or not finite_vector(target_q, 7):
            raise RuntimeError('没有复位起点/原生目标关节证据')
        self.run.write_json('supervised_reset.json', dict(startQ=start_q, targetQ=target_q,
                            startedMonotonicS=time.monotonic()))
        (self.run.output/'monitor_watchdog_armed').touch()
        self.token = uuid.uuid4().hex
        state = self.stream(12., reset=True)
        self.off()
        if not state.get('resetRecovery', {}).get('settled'):
            raise RuntimeError('原生慢复位未收敛；禁止继续动态测试')
        time.sleep(.4)
        (self.run.output/'supervised_reset.json').unlink(missing_ok=True)
        self.token = uuid.uuid4().hex
        return start_q, target_q


def live(run, url):
    """Stop suite immediately on actual-motion, boundary, or stop failures."""
    client = ProbeClient(url, run)
    observer, heartbeat_thread, heartbeat_stop = None, None, threading.Event()
    acquired, completed = False, False
    raw = run.output/'ros_samples.jsonl'
    arm = run.output/'monitor_armed'
    monitor_failure = run.output/'monitor_failure.json'
    thresholds = run.manifest['thresholds']
    def beat():
        while not heartbeat_stop.wait(.025):
            path = run.output/'heartbeat.tmp'
            path.write_text(str(time.monotonic()))
            path.replace(run.output/'heartbeat')
    def rows(start, end=None):
        time.sleep(.04)  # observer flush, not simulated feedback
        return [r for r in read_jsonl(raw) if r.get('receiptMonotonicS', 0) >= start and
                (end is None or r.get('receiptMonotonicS', 0) <= end)]
    try:
        state = client.request('/state')
        if state.get('runtime', {}).get('publishCommands') is not False:
            raise RuntimeError('已有 Send 活跃会话；本测试不接管运动中的会话')
        state = client.request('/reset', client.base, 'acquire_idle_session')
        acquired = True
        client.strict_observer = True
        client.off()
        heartbeat_thread = threading.Thread(target=beat, daemon=True)
        heartbeat_thread.start()
        observer_log = (run.output/'observer.log').open('w')
        observer = subprocess.Popen([sys.executable, str(REPO/'tools/predeploy_ros_observer.py'),
            '--output', str(run.output), '--profile', run.profile],
            cwd=REPO, stdout=observer_log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic()+15.
        while not (run.output/'observer_ready').exists() and time.monotonic() < deadline:
            if observer.poll() is not None:
                raise RuntimeError('ROS 独立观察器启动失败；见 observer.log')
            time.sleep(.1)
        if not (run.output/'observer_ready').exists():
            raise RuntimeError('ROS 独立观察器启动超时')
        client.abort = monitor_failure.exists
        feedback_deadline = time.monotonic()+2.
        while time.monotonic() < feedback_deadline:
            measured_rows = rows(time.monotonic()-.5)
            if len([r for r in measured_rows if r.get('kind') == 'joint']) >= 40:
                break
        if not measured_rows:
            raise RuntimeError('没有独立原始 ROS 反馈，不能测试运动停止')
        preflight = assess_motion(measured_rows, thresholds, need_commands=False)
        nonpose_faults = [f for f in preflight['faults'] if f not in (
            'measured_position_outside_real_workspace', 'measured_attitude_outside_real_envelope')]
        if preflight['status'] == 'BLOCKED' or nonpose_faults:
            run.record('live_preflight', 'BLOCKED', '初始实测状态未满足策略边界/反馈要求：'+
                       ', '.join(preflight['faults']), **preflight['metrics'])
            raise RuntimeError('反馈/初始速度检查失败；禁止复位发令')
        reset_start = time.monotonic()
        try:
            start_q, target_q = client.reset()
        except Exception as exc:
            run.record('live_native_reset', 'FAIL', str(exc), evidence='ros_samples.jsonl')
            raise
        reset_verdict = assess_reset(rows(reset_start), thresholds, start_q, target_q,
                                     started_inside=preflight['status'] == 'PASS')
        run.record('live_native_reset', reset_verdict['status'], ', '.join(reset_verdict['faults']) or
                   '受监督原生慢复位实际反馈、速度和停稳通过', **reset_verdict['metrics'])
        if reset_verdict['status'] != 'PASS':
            raise RuntimeError('受监督复位没有通过实际安全验收')
        run.record('live_preflight', 'PASS', '独立反馈、复位后实测边界及停稳、监视器、空闲 Send 已确认')
        arm.touch()
        # A moving baseline is a mandatory prerequisite, including joint q.
        start = time.monotonic()
        client.stream(thresholds['baseline_duration_s'])
        end = time.monotonic()
        baseline = assess_motion(rows(start, end), thresholds, require_motion=True)
        run.record('live_baseline', baseline['status'], ', '.join(baseline['faults']) or
                   '真实关节运动及指令值更新已验证', **baseline['metrics'])
        if baseline['status'] != 'PASS':
            raise RuntimeError('真实运动基线未通过，停止后续故障测试')
        # Manual off is first: other faults depend on the same stop path.
        for case_id, fault in (
            ('live_manual_off', None), ('live_hidden_ball', {'valid': False}),
            ('live_low_ball_done', {'ball': [0., -.49, .20], 'velocity': [0., 0., -.5]}),
            ('live_high_ball_done', {'ball': [0., -.49, 2.2], 'velocity': [0., 0., 1.]}),
            ('live_outside_ball_done', {'ball': [1., -.49, 1.2], 'velocity': [0., 0., -.5]}),
            ('live_lease_loss', 'lease')):
            if case_id != 'live_manual_off':
                client.reset()
            start = time.monotonic()
            # Wait for a genuinely moving state, renewing the same arm token.
            for _ in range(120):
                client.stream(.025)
                recent = [r for r in rows(time.monotonic()-.04) if r.get('kind') == 'joint']
                if recent and finite_vector(recent[-1].get('dq'), 7) and max(map(abs, recent[-1]['dq'])) >= math.radians(thresholds['minimum_stop_entry_speed_deg_s']):
                    break
            else:
                run.record(case_id, 'BLOCKED', '未在真实运动中触发停止')
                raise RuntimeError('停止测试前置条件未满足')
            trigger = time.monotonic()
            if fault is None:
                stopped = client.off()
            elif fault == 'lease':
                time.sleep(.65)  # intentionally no heartbeat to the command lease
                stopped = client.request('/state', phase=case_id)
            else:
                stopped = client.stream(.45, extra=fault)
            # Keep the hardware monitor alive while Send is disabled. No reset.
            tail_deadline = trigger + thresholds['stop_observation_s'] + .12
            while time.monotonic() < tail_deadline:
                time.sleep(.04)
            final = client.request('/state', phase=case_id+'_tail')
            verdict = assess_motion(rows(start), thresholds, stop_at=trigger,
                require_moving_stop=True)
            if monitor_failure.exists():
                verdict['status'] = 'FAIL'
                verdict['faults'].append('independent_monitor_triggered')
            if final.get('runtime', {}).get('publishCommands') is not False:
                verdict['status'] = 'FAIL'
                verdict['faults'].append('send_not_disabled')
            if fault is not None:
                if not final.get('commandRearm', {}).get('required'):
                    verdict['status'] = 'FAIL'
                    verdict['faults'].append('missing_explicit_rearm_latch')
                count = final.get('ros', {}).get('publishCount')
                recovered = client.step(True, ball=[0., -.49, 1.14], velocity=[0., 0., -.5])
                time.sleep(.1)
                recovered = client.request('/state', phase=case_id+'_no_automatic_resume')
                if recovered.get('runtime', {}).get('publishCommands') or recovered.get('ros', {}).get('publishCount') != count:
                    verdict['status'] = 'FAIL'
                    verdict['faults'].append('old_token_or_restored_input_automatically_rearmed')
            run.record(case_id, verdict['status'], ', '.join(verdict['faults']) or
                       '运动中触发、实际停稳、边界及锁存通过', triggerMonotonicS=trigger,
                       stopReason=stopped.get('commandRearm', {}).get('reason'), **verdict['metrics'])
            if verdict['status'] != 'PASS':
                raise RuntimeError('停止验证失败；不得继续扩大异常注入')
        run.record('live_stop_regression', 'PASS', '全部自动运动停止用例实际反馈通过')
        completed = True
    except Exception as exc:
        run.record('live_runner', 'BLOCKED', str(exc))
        for c in run.cases:
            if c['stage'] == 'builtin' and c['id'] not in run.results:
                run.record(c['id'], 'BLOCKED', '前置检查或停止验证未通过；不继续发令')
    finally:
        # No automatic following enable, no reset after a failed stop.
        if acquired:
            try:
                final = client.off()
                run.write_json('final_snapshot.json', final)
            except Exception as exc:
                run.record('live_cleanup', 'FAIL', '关闭 Send 未确认：'+str(exc))
        if observer is not None and observer.poll() is None:
            # Request native stop if the suite was interrupted/faulted while
            # armed. The observer records the ACK AND keeps measuring motion.
            if not completed and (arm.exists() or (run.output/'monitor_watchdog_armed').exists()):
                (run.output/'request_native_stop').touch()
                time.sleep(.5)
            (run.output/'observer_finish').touch()
            try:
                observer.wait(timeout=5.)
            except subprocess.TimeoutExpired:
                observer.terminate()
                observer.wait(timeout=3.)
                run.record('live_observer_shutdown', 'FAIL', '观察器未正常退出')
        heartbeat_stop.set()
        if heartbeat_thread:
            heartbeat_thread.join(timeout=1.)
        if monitor_failure.exists():
            monitor_event = json.loads(monitor_failure.read_text())
            if monitor_event.get('reason') == 'test_suite_failed_request_native_stop':
                run.record('live_native_stop_request', 'PASS' if monitor_event.get(
                    'nativeStopAcknowledged') else 'FAIL',
                    '故障清理 StopAllMU 请求已确认；实际制动与恢复仍须专门验证',
                    evidence='monitor_failure.json')
            else:
                run.record('live_monitor', 'FAIL', '独立监视器发现异常并触发停止；ACK 不等于已停稳',
                           evidence='monitor_failure.json')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--offline', action='store_true', help='默认：无 ROS 发令的合约/故障回归')
    mode.add_argument('--builtin-sim', action='store_true', help='内置仿真下实际发送指令及异常输入')
    mode.add_argument('--plan', action='store_true', help='仅列出流程/覆盖项')
    parser.add_argument('--confirm-built-in-simulation', action='store_true',
                        help='操作者确认当前控制器为内置仿真；ROS 状态不能可靠证明这一点')
    parser.add_argument('--profile', choices=FPLC_PROFILES, default='fplc_gpu1_v172')
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--analyze-run', type=Path, help='只读重算本流程的原始数据，不签发新运行 PASS')
    args = parser.parse_args(argv)
    if args.builtin_sim and not args.confirm_built_in_simulation:
        parser.error('--builtin-sim requires --confirm-built-in-simulation')
    if args.analyze_run:
        old = json.loads((args.analyze_run/'report.json').read_text())
        verdict = assess_motion(read_jsonl(args.analyze_run/'ros_samples.jsonl'), old['thresholds'])
        print(json.dumps(dict(historicalAnalysis=True, hardwareReleaseReady=False, **verdict), indent=2))
        return 0 if verdict['status'] == 'PASS' else 1
    if args.plan:
        data = json.loads(MANIFEST.read_text())
        for c in data['cases']:
            print(f'{c["stage"]:8s} {c["id"]:35s} {c["title"]}')
        return 0
    output = args.output or REPO/'analysis_results'/('predeploy_safety_'+dt.datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    run = Run(output, args.profile, 'builtin' if args.builtin_sim else 'offline')
    print('证据目录：'+str(run.output), flush=True)
    interrupted = False
    try:
        run.offline()
        if args.builtin_sim:
            if aggregate(run.cases, run.results, 'offline')['selectedStagePassed']:
                live(run, args.url)
            else:
                for c in run.cases:
                    if c['stage'] == 'builtin':
                        run.record(c['id'], 'BLOCKED', '离线关键故障回归未通过')
    except (KeyboardInterrupt, SystemExit):
        interrupted = True
        run.record('runner_interrupted', 'FAIL', '测试被中断，不能签发通过')
    except Exception as exc:
        run.record('runner_error', 'FAIL', str(exc))
    summary = run.finalize()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print('报告：'+str(run.output/'report.md'), flush=True)
    return 130 if interrupted else (0 if summary['selectedStagePassed'] and not summary['failedCases'] else 1)


if __name__ == '__main__':
    sys.exit(main())
