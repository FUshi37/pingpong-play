"""Config/UI -> sensor filtering, cache fences and unchanged cadence."""
import json
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from pingpong_controller.mocap_ball_detector import MocapBallDetector, parse_mocap_packet
from pingpong_controller.mocap_input import MocapBallInput
from pingpong_controller.mocap_observation_range import parse_observation_bounds
from pingpong_controller.mocap_process import MocapInputProcess
from pingpong_controller.tools import interactive_policy_safety_probe as probe
from test_mocap_launcher_sync import launch_environment

ROOT = Path(__file__).resolve().parents[3]
BOX = ((-.4, .4), (.5, 1.5), (.2, 2.))
BALL = (.1, 1., 1.2)  # Outside the previous hardcoded Z <= 1 bound.
OTHER = (.1, 2., 1.2)


def frame(points, tick=0, *, scale=1., kind='flat'):
    payload = dict(marker_frmIdx=100 + tick, marker_fTimestamp=1000. + tick * .01)
    if kind == 'flat':
        for marker, point in enumerate(points, 1):
            for axis, value in zip('xyz', point):
                payload[f'marker_{marker}_p{axis}'] = value * scale
    else:
        payload['markers'] = [dict(id=i + 1, position=[v * scale for v in point])
                              for i, point in enumerate(points)]
    return json.dumps(payload).encode()


@pytest.mark.parametrize('kind', ['flat', 'nested'])
@pytest.mark.parametrize('unit,scale', [('m', 1.), ('mm', 1000.), ('cm', 100.), ('auto', 1000.)])
@pytest.mark.parametrize('slot', [0, 1, 2])
def test_xyz_selects_unique_ball_before_calibration_and_replaces_old_bounds(kind, unit, scale, slot):
    points = [OTHER, (3., 1., 1.)]
    points.insert(slot, BALL)
    packet = parse_mocap_packet(frame(points, scale=scale, kind=kind), marker_id=99,
                               position_unit=unit, marker_x_limits_m=(20., 30.),
                               marker_z_limits_m=(-1., 1.), observation_bounds_m=BOX)
    assert packet.tracked
    np.testing.assert_allclose(packet.position_m, BALL, atol=1e-7)
    assert packet.marker_selection_info['selected_marker_id'] == slot + 1
    assert packet.marker_selection_info['eligible_marker_count'] == 1
    assert packet.marker_selection_info['y_min_m'] == .5


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('edge', [0, 1])
def test_faces_inclusive_and_points_just_outside_rejected(axis, edge):
    point = list(BALL)
    point[axis] = BOX[axis][edge]
    assert parse_mocap_packet(frame([point]), observation_bounds_m=BOX).tracked
    point[axis] += -.0001 if edge == 0 else .0001
    assert not parse_mocap_packet(frame([point]), observation_bounds_m=BOX).tracked


@pytest.mark.parametrize('points,status', [([OTHER], 'no_candidate'),
    ([BALL, (.2, 1.1, 1.3)], 'ambiguous_candidates'),
    ([(float('nan'), 1., 1.)], 'no_candidate')])
def test_no_target_and_multiple_targets_do_not_guess(points, status):
    packet = parse_mocap_packet(frame(points), observation_bounds_m=BOX)
    assert not packet.tracked
    assert packet.marker_selection_info['status'] == status


@pytest.mark.parametrize('point,valid', [(BALL, True), (OTHER, False)])
def test_binary_udp_obeys_xyz_range(point, valid):
    packet = parse_mocap_packet(struct.pack('<d6f', 1000., *point, 0., 0., 0.),
                               position_unit='m', observation_bounds_m=BOX)
    assert packet.tracked is valid


def feed(source, points, tick):
    packet = parse_mocap_packet(frame(points, tick), observation_bounds_m=source.detector.observation_bounds_m)
    return source.detector.process_packet(packet, arrival_clock_s=1800000000. + tick * .01,
        arrival_monotonic_s=100. + tick * .01, arrival_wall_s=1800000000. + tick * .01)


def test_rejected_observations_never_refresh_position_stamp_or_velocity_history():
    source = MocapBallInput(observation_bounds_m=BOX, velocity_window_frames=1)
    for tick in range(12):
        feed(source, [BALL, OTHER], tick)
    detector = source.detector
    stamp, count = detector._latest_measurement_timestamp_s, detector._packet_count
    for tick in range(12, 20):
        assert not feed(source, [OTHER], tick)
        assert not detector._latest_observation_valid
        assert detector._latest_measurement_timestamp_s == stamp
        assert detector._packet_count == count
        assert not detector._velocity_history
        np.testing.assert_allclose(detector._latest_pos_m, BALL)
    assert feed(source, [BALL, OTHER], 20)
    np.testing.assert_array_equal(detector._latest_vel_m_s, np.zeros(3))


def test_changing_range_fences_old_cache_and_reselects_from_full_packet():
    source = MocapBallInput(observation_bounds_m=BOX)
    for tick in range(12):
        feed(source, [BALL, OTHER], tick)
    old = source.detector.observation_bounds_m
    packet_in_flight = parse_mocap_packet(frame([BALL, OTHER], 12), observation_bounds_m=old)
    revised = (BOX[0], (1.8, 2.2), BOX[2])
    epoch = source.detector.clock_sync.epoch
    source.set_observation_bounds(revised)
    assert source.detector.clock_sync.ready is True
    assert source.detector.clock_sync.epoch == epoch
    assert not source.detector._latest_observation_valid
    assert not source.detector._velocity_history
    source.detector.process_packet(packet_in_flight, arrival_clock_s=1800000000.12,
        arrival_monotonic_s=100.12, arrival_wall_s=1800000000.12)
    np.testing.assert_allclose(source.detector._latest_pos_m, OTHER)
    assert source.detector._latest_observation_valid
    assert source.detector._latest_measurement_timestamp_s == pytest.approx(1800000000.12)
    # Sending unchanged settings must never restart clock warm-up.
    epoch = source.detector.clock_sync.epoch
    source.set_observation_bounds(revised)
    assert source.detector.clock_sync.epoch == epoch
    source.set_observation_bounds(None)
    assert source.detector.observation_bounds_m == (None, None, None)
    assert source.detector.marker_x_limits_m is source.detector.marker_z_limits_m is None


def test_range_change_immediately_fences_subprocess_control_and_recording_cache():
    client = MocapInputProcess(observation_bounds_m=BOX)
    now = time.time()
    feedback = dict(clockSync=dict(ready=True), syncRequest=0, markerId=1,
                    ageS=0., measurementTimestampS=now, receiverTimestampS=now, tracked=True)
    client.process = SimpleNamespace(is_alive=lambda: True, pid=123)
    client._sample = dict(position=np.array(BALL), velocity=np.zeros(3), valid=True,
                         feedback=feedback, observedMonotonicS=time.monotonic(),
                         recording=('old packet', 0, time.monotonic()))
    assert client.get_control_state()[2]
    client.set_observation_bounds((None, (1.8, 2.2), None))
    assert not client.get_control_state()[2]
    assert client.get_control_state()[3]['clockSync']['status'] == 'awaiting_resynchronization'
    assert client.recording_snapshot()[3] == 1
    assert client._sample['recording'][1] == 0
    client.set_observation_bounds((None, (1.8, 2.2), None))
    assert client.sync_request == 1
    assert client.clock_sync_request == 0


@pytest.mark.parametrize('bounds', [[(-1, 1)], [(0, 0), None, None],
    [None, (2, 1), None], [None, None, (0, float('inf'))],
    [None, (float('nan'), 2), None], ['0,1', None, None]])
def test_malformed_bounds_never_change_active_receiver(bounds):
    detector = MocapBallDetector(observation_bounds_m=BOX)
    with pytest.raises(ValueError):
        detector.set_observation_bounds(bounds)
    assert detector.observation_bounds_m == BOX


@pytest.mark.parametrize('subprocess_mode', [True, False])
def test_real_launcher_passes_environment_xyz_to_both_receiver_modes(tmp_path, subprocess_mode):
    capture = tmp_path / 'args.json'
    wrapper = tmp_path / 'python3'
    wrapper.write_text(f'#!{sys.executable}\n' + '''
import json, os, sys
from pathlib import Path
if len(sys.argv) > 1 and Path(sys.argv[1]).name == 'interactive_policy_safety_probe.py':
    Path(os.environ['MOCAP_TEST_CAPTURE']).write_text(json.dumps(sys.argv[2:]))
    raise SystemExit(0)
os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
''')
    wrapper.chmod(0o755)
    (tmp_path / 'setup.bash').write_text('return 0\n')
    env = launch_environment(tmp_path, PROBE_CONFIG_ONLY='false',
        PATH=str(tmp_path) + os.pathsep + os.environ['PATH'], MOCAP_TEST_CAPTURE=str(capture),
        MOCAP_SUBPROCESS='true' if subprocess_mode else 'false',
        MOCAP_OBSERVATION_X_LIMITS_M='-0.4,0.4', MOCAP_OBSERVATION_Y_LIMITS_M='0.5,1.5',
        MOCAP_OBSERVATION_Z_LIMITS_M='0.2,2')
    result = subprocess.run(['bash', str(ROOT / 'RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh')],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    cfg = probe.config_from_args(probe.build_arg_parser().parse_args(json.loads(capture.read_text())))
    assert cfg.mocap_observation_bounds_m == BOX
    assert cfg.mocap_subprocess is subprocess_mode
    assert cfg.control_rate_hz == 200. and cfg.ball_observation_rate_hz == 90.
    node = probe.RosBridgeNode.__new__(probe.RosBridgeNode)
    node.cfg, node.now_sec = cfg, time.time
    node.get_logger = lambda: SimpleNamespace(info=Mock(), warn=Mock())
    cls = MocapInputProcess if subprocess_mode else MocapBallInput
    from unittest.mock import patch
    with patch.object(cls, 'start', return_value=True):
        node._init_mocap_ball_source()
    detector = (MocapBallInput(**node.mocap_input.kwargs).detector if subprocess_mode
                else node.mocap_input.detector)
    assert detector.observation_bounds_m == BOX
    assert detector._socket is None


@pytest.mark.parametrize('args,success', [
    (['--mocap-observation-x-limits-m=-0.4,0.4', '--mocap-observation-y-limits-m', '0.5,1.5'], True),
    (['--mocap-observation-z-limits-m='], True),
    (['--mocap-observation-x-limits-m=nan,1'], False),
    (['--mocap-observation-y-limits-m=2,1'], False),
    (['--mocap-observation-z-limits-m=1'], False),
    (['--mocap-observation-y-limits-m'], False),
])
def test_config_only_validates_cli_without_device_start(tmp_path, args, success):
    result = subprocess.run(['bash', str(ROOT / 'RUN_INTERACTIVE_POLICY_SAFETY_PROBE.sh'), *args],
        cwd=ROOT, env=launch_environment(tmp_path), capture_output=True, text=True, timeout=30)
    assert (result.returncode == 0) is success, result.stderr
    if success:
        assert 'legacy_filters=replaced' in result.stdout
    assert not (tmp_path / 'ros_log').exists()


def test_browser_payload_applies_only_changed_valid_bounds():
    backend = probe.PolicySafetyProbe.__new__(probe.PolicySafetyProbe)
    backend.cfg = SimpleNamespace(enable_mocap_ball=True, mocap_observation_bounds_m=None)
    backend.ros_bridge = SimpleNamespace(set_mocap_observation_bounds=Mock())
    payload = dict(mocapObservationBoundsM=json.loads(json.dumps(BOX)))
    for _ in range(20):
        backend._update_mocap_observation_bounds(payload)
    backend.ros_bridge.set_mocap_observation_bounds.assert_called_once_with(BOX)
    with pytest.raises(ValueError):
        backend._update_mocap_observation_bounds(dict(mocapObservationBoundsM=[[2, 1], None, None]))
    assert backend.cfg.mocap_observation_bounds_m == BOX


def test_browser_range_inputs_and_validation_use_explicit_apply(tmp_path):
    fn = re.search(r'function readMocapObservationBounds\(\) \{.*?\n\}', probe.HTML_TEMPLATE, re.S).group()
    script = '''
const mocapRangeEnabledEl = {checked: true};
const mocapRangeInputs = [[-.4,.4],[.5,1.5],[.2,2]].map(pair => pair.map(value => ({value: String(value)})));
''' + fn + '''
if (JSON.stringify(readMocapObservationBounds()) !== '[[-0.4,0.4],[0.5,1.5],[0.2,2]]') process.exit(1);
mocapRangeInputs[1][0].value = '';
try { readMocapObservationBounds(); process.exit(2); } catch (_) {}
mocapRangeInputs[1][1].value = '';
if (readMocapObservationBounds()[1] !== null) process.exit(3);
mocapRangeEnabledEl.checked = false;
if (JSON.stringify(readMocapObservationBounds()) !== '[null,null,null]') process.exit(4);
'''
    path = tmp_path / 'mocap_range.js'
    path.write_text(script)
    subprocess.run(['node', str(path)], check=True, capture_output=True)
    assert 'mocapObservationBoundsM: mocapObservationBounds' in probe.HTML_TEMPLATE
    for axis in 'XYZ':
        for edge in ('Min', 'Max'):
            assert f'id="mocapRange{axis}{edge}"' in probe.HTML_TEMPLATE


def test_new_options_replace_legacy_or_explicitly_disable_all_axes():
    assert parse_observation_bounds([None, None, None]) is None
    assert parse_observation_bounds(['', None, None]) == (None, None, None)
    cfg = probe.config_from_args(probe.build_arg_parser().parse_args([
        '--mocap-observation-y-limits-m=0.5,1.5']))
    assert cfg.mocap_observation_bounds_m == (None, (.5, 1.5), None)
    detector = MocapBallDetector(marker_x_limits_m=(-2, 2), marker_z_limits_m=(-1, 1),
                                observation_bounds_m=cfg.mocap_observation_bounds_m)
    assert detector.marker_x_limits_m is detector.marker_z_limits_m is None


def test_control_subprocess_browser_refresh_preserves_applied_range_and_detects_changes():
    proxy = probe.ControlSubprocessProxy.__new__(probe.ControlSubprocessProxy)
    proxy._drain_status = lambda: None
    proxy.lock = threading.Lock()
    proxy.latest_config = dict(mocapBall=dict(observationBoundsM=BOX))
    revised = [None, [1.8, 2.2], None]
    proxy.latest_snapshot = dict(mocap=dict(observationBoundsM=revised))
    assert proxy.browser_config()['mocapBall']['observationBoundsM'] == revised
    assert proxy.latest_config['mocapBall']['observationBoundsM'] == BOX
    assert proxy._step_payload_key(dict(mocapObservationBoundsM=BOX)) != proxy._step_payload_key(
        dict(mocapObservationBoundsM=revised))
