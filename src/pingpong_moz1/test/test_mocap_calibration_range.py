"""Rigid-body range selection and calibration resume, without motion or UDP."""
from dataclasses import asdict
from itertools import permutations
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src/pingpong_replay'))
from pingpong_controller.mocap_ball_detector import parse_mocap_packet
from pingpong_controller import mocap_auto_calibration as auto
from pingpong_controller.mocap_observation_range import parse_observation_bounds

BOX = ((-.4, .4), (.5, 1.5), (.2, 2.))
POINT = (.1, 1., 1.2)
FLAGS = [f'--mocap-observation-{axis}-limits-m={pair[0]},{pair[1]}'
         for axis, pair in zip('xyz', BOX)]


def datagram(bodies, *, scale=1.):
    payload = dict(marker_fTimestamp=1000., marker_1_px=.1,
                   marker_1_py=1., marker_1_pz=1.2)
    for slot, body_id, point, tracked in bodies:
        prefix = f'opti_rb_{slot}_'
        payload.update({prefix + 'id': body_id, prefix + 'btracked': tracked,
                        prefix + 'qw': 1., prefix + 'qx': 0., prefix + 'qy': 0., prefix + 'qz': 0.,
                        prefix + 'meanError': .0001 * scale, prefix + 'trackid': 1234})
        payload.update({prefix + 'p' + axis: value * scale for axis, value in zip('xyz', point)})
    return json.dumps(payload).encode()


@pytest.mark.parametrize('unit,scale', [('m', 1.), ('mm', 1000.), ('auto', 1000.)])
@pytest.mark.parametrize('slot', [1, 4, 19])
def test_range_selects_rigid_pose_regardless_of_configured_slot_and_scatter_points(unit, scale, slot):
    data = datagram([(0, 10, (3., 1., 1.2), 1), (slot, 42, POINT, 1)], scale=scale)
    packet = parse_mocap_packet(data, object_kind='opti_rb', rigid_body_slot=0, rigid_body_id=0,
                               position_unit=unit, rigid_body_observation_bounds_m=BOX)
    assert packet.tracked and packet.rigid_body_slot == slot and packet.rigid_body_id == 42
    np.testing.assert_allclose(packet.position_m, POINT, atol=1e-7)
    np.testing.assert_array_equal(packet.orientation_wxyz, [1., 0., 0., 0.])
    assert packet.mean_error_m == pytest.approx(.0001)
    assert packet.frame_idx == 1234 and packet.timestamp_s == 1000.


@pytest.mark.parametrize('axis', [0, 1, 2])
@pytest.mark.parametrize('edge', [0, 1])
def test_rigid_range_faces_inclusive(axis, edge):
    point = list(POINT)
    point[axis] = BOX[axis][edge]
    kwargs = dict(object_kind='opti_rb', rigid_body_observation_bounds_m=BOX)
    assert parse_mocap_packet(datagram([(7, 42, point, 1)]), **kwargs).tracked
    point[axis] += -.001 if edge == 0 else .001
    assert not parse_mocap_packet(datagram([(7, 42, point, 1)]), **kwargs).tracked


@pytest.mark.parametrize('bodies,status', [
    ([], 'no_candidate'),
    ([(7, 42, POINT, 0)], 'no_candidate'),
    ([(7, 42, (float('nan'), 1., 1.), 1)], 'no_candidate'),
    ([(7, 42, POINT, 1), (8, 43, (.2, 1., 1.3), 1)], 'ambiguous_candidates'),
])
def test_ambiguous_missing_untracked_or_nonfinite_rigid_body_is_invalid(bodies, status):
    packet = parse_mocap_packet(datagram(bodies), object_kind='opti_rb',
                               rigid_body_observation_bounds_m=BOX)
    assert not packet.tracked
    assert packet.marker_selection_info['status'] == status


def test_probe_ball_filter_keeps_fixed_rigid_body_parsing_independent():
    packet = parse_mocap_packet(datagram([(0, 10, (3., 1., 1.2), 1), (7, 42, POINT, 1)]),
        object_kind='opti_rb', rigid_body_slot=0, rigid_body_id=0, observation_bounds_m=BOX)
    assert packet.rigid_body_slot == 0
    np.testing.assert_allclose(packet.position_m, [3., 1., 1.2])


@pytest.fixture
def calibration():
    pytest.importorskip('rclpy')
    from pingpong_controller import mocap_calibration
    return mocap_calibration


@pytest.fixture
def node(calibration):
    node = calibration.MocapCalibrationNode.__new__(calibration.MocapCalibrationNode)
    node.calib_target = 'rigid_body'
    node.mocap_object_slot = node.mocap_object_id = 0
    node.position_unit, node.timestamp_unit = 'm', 's'
    node.mocap_observation_bounds_m = BOX
    node.selected_rigid_body = node.last_mocap_range_status = None
    node.get_logger = lambda: SimpleNamespace(info=lambda *args: None)
    return node


def test_range_tracks_renumbered_target_and_excludes_other_spatial_positions(node):
    assert node._parse_mocap_datagram(datagram([(7, 42, POINT, 1)])) is not None
    assert node.selected_rigid_body == dict(slot=7, id=42)
    for slot, body_id in [(8, 43), (7, 43), (0, 0), (3, 3)]:
        assert node._parse_mocap_datagram(datagram([(slot, body_id, POINT, 1)])) is not None
        assert node.selected_rigid_body == dict(slot=slot, id=body_id)
    for bodies in ([(7, 42, (3., 1., 1.), 1)],
                   [(7, 42, POINT, 1), (8, 43, (.2, 1.1, 1.3), 1)]):
        assert node._parse_mocap_datagram(datagram(bodies)) is None
        assert node.selected_rigid_body == dict(slot=3, id=3)
    assert node._parse_mocap_datagram(datagram([(7, 42, POINT, 1)])) is not None


@pytest.mark.parametrize('order', list(permutations(range(4))))
def test_current_four_body_split_udp_layout_tracks_x_largest_despite_renumbering(node, order):
    # Actual bridge layout: three bodies in one JSON UDP packet, the fourth
    # in another; numeric slot/id identify output order and change each frame.
    positions = [(-1.78518, .8935, -.07428), (-1.32258, .8159, -.01878),
                 (-.04709, 1.7415, .61009), (1.12221, 1.06896, .41774)]
    node.mocap_observation_bounds_m = ((.3, 2.), None, None)
    node.selected_rigid_body = dict(slot=0, id=0)
    bodies = [(slot, slot, positions[physical], 1) for slot, physical in enumerate(order)]
    accepted = [node._parse_mocap_datagram(datagram(part)) for part in (bodies[:3], bodies[3:])]
    accepted = [p for p in accepted if p is not None]
    assert len(accepted) == 1
    np.testing.assert_allclose(accepted[0].position_m, positions[3], atol=1e-7)
    target_slot = order.index(3)
    assert node.selected_rigid_body == dict(slot=target_slot, id=target_slot)


def test_calibration_receives_full_multi_body_datagram(node):
    bodies = [(slot, slot + 10, (3., 1., 1.2), 1) for slot in range(32)]
    bodies.append((32, 42, POINT, 1))
    data = datagram(bodies)
    assert len(data) > 4096
    def receive(buffer_size):
        assert buffer_size >= len(data)
        return data, ('127.0.0.1', 65501)
    node.socket = SimpleNamespace(recvfrom=receive)
    measurement = node.receive_mocap_measurement()
    assert measurement is not None
    np.testing.assert_allclose(measurement.position_m, POINT, atol=1e-7)
    assert node.selected_rigid_body == dict(slot=32, id=42)


@pytest.mark.parametrize('quaternion', [None, [0., 0., 0., 0.], [float('nan'), 0., 0., 0.]])
def test_bad_pose_cannot_lock_or_enter_calibration(node, quaternion):
    payload = json.loads(datagram([(7, 42, POINT, 1)]))
    for edge, value in zip('wxyz', quaternion or [None] * 4):
        if value is None:
            payload.pop('opti_rb_7_q' + edge)
        else:
            payload['opti_rb_7_q' + edge] = value
    assert node._parse_mocap_datagram(json.dumps(payload).encode()) is None
    assert node.selected_rigid_body is None
    assert node.last_mocap_range_status == 'invalid_orientation'


def test_stable_window_excludes_outside_and_accepts_renumbered_target(node, calibration, monkeypatch):
    clock = [0.]
    monkeypatch.setattr(calibration.time, 'monotonic', lambda: clock[0])
    rows = iter([[(7, 42, POINT, 1)], [(7, 42, (3., 1., 1.), 1)],
                 [(8, 43, POINT, 1)], [(7, 42, POINT, 1)],
                 [(7, 42, POINT, 1)], [(7, 42, POINT, 1)]])
    def receive():
        clock[0] += .03
        return node._parse_mocap_datagram(datagram(next(rows, [])))
    node.receive_mocap_measurement = receive
    node._discard_pending_mocap_datagrams = lambda: 0
    node.stability_window_s = .1
    node.max_mocap_std_m = .00025
    node.max_mocap_rotation_rms_deg = .2
    node.max_mocap_mean_error_m = .001
    stable, reason = node.receive_stable_mocap_pose(.5)
    assert stable is not None, reason
    assert stable.packet_count == 4
    np.testing.assert_allclose(stable.position_m, POINT, atol=1e-7)


def configured_args(flags=FLAGS):
    args = auto.build_parser().parse_args(['--max-supplemental-points=0', *flags])
    args.mocap_observation_bounds_m = parse_observation_bounds([
        args.mocap_observation_x_limits_m, args.mocap_observation_y_limits_m,
        args.mocap_observation_z_limits_m])
    return args


@pytest.fixture
def progress(tmp_path, calibration, monkeypatch):
    monkeypatch.chdir(ROOT)
    args = configured_args()
    from pingpong_replay.cartesian_fk import RightArmFK
    from pingpong_controller.mocap_transform import pose_wxyz_to_matrix
    fk = RightArmFK(args.robot_xml, joint_names=auto.JOINT_NAMES)
    q = auto.load_waypoints(args.waypoints)[1][0]
    pose = fk.ee_pose(q)
    sample = calibration.CalibrationSample(index=1, mocap_m=np.array(POINT),
        robot_m=pose[:3, 3], mocap_orientation_wxyz=np.array([1., 0., 0., 0.]),
        T_mocap_rigid=pose_wxyz_to_matrix(POINT, [1., 0., 0., 0.]),
        T_waist03_ee=pose, joint_positions_rad=q)
    saved = dict(schema_version=4, samples=[asdict(sample)], accepted_count=1, expected_count=50,
        waypoint_file=args.waypoints, options=vars(args), selected_rigid_body=dict(slot=7, id=42),
        completed_point=1, accepted_point_indices=[1], skipped_points=[])
    path = tmp_path / 'progress.json'
    def write():
        path.write_text(json.dumps(saved, default=lambda value: value.tolist()))
    write()
    return args, fk, saved, path, write


def test_resume_restores_range_and_accepts_new_udp_numbering(progress, node):
    args, fk, _, path, _ = progress
    saved = auto.load_resume(path, args, fk)
    auto.restore_session(node, saved)
    assert node.sample_count == 1
    assert node.selected_rigid_body == dict(slot=7, id=42)
    assert node._parse_mocap_datagram(datagram([(8, 43, POINT, 1)])) is not None
    assert node.selected_rigid_body == dict(slot=8, id=43)
    assert node._parse_mocap_datagram(datagram([(7, 42, (3., 1., 1.), 1)])) is None


def test_resume_rejects_changed_range_and_allows_missing_transient_number(progress):
    args, fk, saved, path, write = progress
    args.mocap_observation_bounds_m = (None, (.5, 1.5), None)
    with pytest.raises(ValueError, match='same mocap observation range'):
        auto.load_resume(path, args, fk)
    args.mocap_observation_bounds_m = BOX
    saved['selected_rigid_body'] = None
    write()
    assert auto.load_resume(path, args, fk)['selected_rigid_body'] is None


def test_progress_records_range_and_selected_body(tmp_path, node):
    args = configured_args()
    args.output = str(tmp_path / 'calibration.txt')
    from pingpong_controller.mocap_calibration import CalibrationSession
    node.session = CalibrationSession()
    node.selected_rigid_body = dict(slot=7, id=42)
    sequence = auto.LiveSequence(node, SimpleNamespace(diagnostics=lambda now: {}), args)
    sequence.save_progress()
    saved = json.loads(Path(args.output).with_suffix('.progress.json').read_text())
    assert saved['options']['mocap_observation_bounds_m'] == [list(pair) for pair in BOX]
    assert saved['selected_rigid_body'] == dict(slot=7, id=42)


@pytest.mark.parametrize('flag', ['--mocap-observation-x-limits-m=nan,1',
    '--mocap-observation-y-limits-m=2,1', '--mocap-observation-z-limits-m=1',
    '--mocap-observation-z-limits-m=0,inf'])
def test_invalid_range_fails_before_loading_route_or_ros(flag, monkeypatch):
    monkeypatch.setattr(auto, 'load_waypoints', lambda *args: pytest.fail('route loaded before validation'))
    with pytest.raises(SystemExit) as exc:
        auto.main(['--dry-run', flag])
    assert exc.value.code == 2


def test_environment_defaults_and_explicit_override(monkeypatch):
    for axis in 'XYZ':
        monkeypatch.delenv(f'MOCAP_OBSERVATION_{axis}_LIMITS_M', raising=False)
    monkeypatch.setenv('MOCAP_OBSERVATION_Y_LIMITS_M', '.5,1.5')
    assert configured_args([]).mocap_observation_bounds_m == (None, (.5, 1.5), None)
    assert configured_args(['--mocap-observation-y-limits-m=']).mocap_observation_bounds_m == (None, None, None)


def test_actual_launcher_dry_run_validates_all_50_postures_without_motion():
    env = {key: value for key, value in os.environ.items()
           if key not in {f'MOCAP_OBSERVATION_{a}_LIMITS_M' for a in 'XYZ'}}
    result = subprocess.run([str(ROOT / 'RUN_AUTO_MOCAP_CALIBRATION.sh'), '--dry-run',
        '--speed-deg-s=3', '--settle-seconds=5', '--max-attempts=50',
        '--max-supplemental-points=0', *FLAGS], cwd=ROOT, env=env,
        text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'frame=mocap_raw_m' in result.stdout
    assert 'unique in-range selection replaces fixed slot/id' in result.stdout
    assert '50/50:' in result.stdout and 'Dry-run: no ROS, UDP or commands.' in result.stdout
    assert 'Quintic motion: <= 3 deg/s' in result.stdout
