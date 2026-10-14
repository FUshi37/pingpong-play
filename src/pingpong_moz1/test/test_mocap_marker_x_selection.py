"""Ball selection regressions from reordered scatter points; no robot actuation."""
import json
import struct
from unittest.mock import patch

import numpy as np
import pytest

from pingpong_controller.mocap_ball_detector import MocapBallDetector, parse_mocap_packet
from pingpong_controller.mocap_input import MocapBallInput


BACKGROUND_A = (-2.5705, 1.4646, -.5221)
BACKGROUND_B = (-2.5844, 1.4650, -.4371)
BALL = (-.0908, 1.0378, .7976)


def frame(points, index=0, scale=1.):
    payload = dict(marker_frmIdx=100+index, marker_fTimestamp=1000.+index*.01)
    for marker, position in enumerate(points, 1):
        for axis, value in zip('xyz', position):
            payload[f'marker_{marker}_p{axis}'] = value*scale
    return json.dumps(payload).encode()


@pytest.mark.parametrize('ball_slot', [0, 1, 2])
@pytest.mark.parametrize('unit,scale', [('m', 1.), ('mm', 1000.), ('auto', 1000.)])
def test_unique_ball_survives_udp_field_reordering_and_units(ball_slot, unit, scale):
    points = [BACKGROUND_A, BACKGROUND_B]
    points.insert(ball_slot, BALL)
    packet = parse_mocap_packet(frame(points, scale=scale), marker_id=99,
                                position_unit=unit, marker_x_limits_m=(-2., 2.))
    assert packet.tracked
    np.testing.assert_allclose(packet.position_m, BALL, atol=1e-7)
    assert packet.marker_selection_info['selected_marker_id'] == ball_slot+1
    assert packet.frame_idx == 100 and packet.timestamp_s == 1000.


@pytest.mark.parametrize('x,valid', [(-2., True), (2., True), (-2.001, False), (2.001, False)])
def test_raw_x_boundaries_are_inclusive(x, valid):
    packet = parse_mocap_packet(frame([(x, 0., 0.)]), marker_x_limits_m=(-2., 2.))
    assert packet.tracked is valid


@pytest.mark.parametrize('points,status', [
    ([BACKGROUND_A, BACKGROUND_B], 'no_candidate'),
    ([], 'no_candidate'),
    ([BALL, (.2, 1., .8)], 'ambiguous_candidates'),
])
def test_absent_or_ambiguous_ball_is_an_explicit_invalid_observation(points, status):
    packet = parse_mocap_packet(frame(points), marker_x_limits_m=(-2., 2.))
    assert not packet.tracked
    assert packet.marker_selection_info['status'] == status
    assert packet.timestamp_s == 1000.


def test_nested_scatter_points_also_select_all_indices():
    data = json.dumps(dict(timestamp=12., markers=[
        dict(id=1, position=BACKGROUND_A), dict(id=7, position=BALL)])).encode()
    packet = parse_mocap_packet(data, marker_x_limits_m=(-2., 2.))
    np.testing.assert_allclose(packet.position_m, BALL, atol=1e-7)
    assert packet.marker_selection_info['selected_marker_id'] == 7


@pytest.mark.parametrize('x,valid', [(-2., True), (2., True), (3., False), (float('nan'), False)])
def test_binary_point_uses_the_same_raw_x_filter(x, valid):
    data = struct.pack('<d6f', 12., x, 1., .8, 0., 0., 0.)
    packet = parse_mocap_packet(data, position_unit='m', marker_x_limits_m=(-2., 2.))
    assert packet.tracked is valid


@pytest.mark.parametrize('limits', [(-2., -2.), (2., -2.), (float('nan'), 2.), (0., float('inf'))])
def test_invalid_limits_fail_before_any_receiver_starts(limits):
    with pytest.raises(ValueError, match='X limits'):
        MocapBallDetector(marker_x_limits_m=limits)


def test_loss_is_immediately_invalid_and_cannot_refresh_trusted_position_or_age():
    clock = dict(wall=1800000000., mono=100.)
    adapter = MocapBallInput(marker_x_limits_m=(-2., 2.),
                            receiver_clock=lambda: clock['wall'])

    def feed(index, points):
        clock.update(wall=1800000000.+index*.01, mono=100.+index*.01)
        packet = parse_mocap_packet(frame(points, index), marker_x_limits_m=(-2., 2.))
        return adapter.detector.process_packet(packet, arrival_clock_s=clock['wall'],
            arrival_monotonic_s=clock['mono'], arrival_wall_s=clock['wall'])

    with patch('pingpong_controller.mocap_ball_detector.time.monotonic', lambda: clock['mono']):
        for index in range(12):
            points = [BACKGROUND_A, BACKGROUND_B]
            points.insert(index%3, BALL)
            assert feed(index, points)
        previous_position, _, valid, previous = adapter.get_control_state()
        assert valid
        np.testing.assert_allclose(previous_position, [-.0908, 1.061383, 2.099996], atol=1e-6)
        for index in range(12, 20):
            assert not feed(index, [BACKGROUND_B, BACKGROUND_A])
            _, _, valid, info = adapter.get_control_state()
            assert not valid and not info['detected']
            assert info['measurementTimestampS'] == previous['measurementTimestampS']
            assert info['sourceTimestampS'] == previous['sourceTimestampS']
            assert info['packetCount'] == previous['packetCount']
            assert info['markerSelection']['status'] == 'no_candidate'
        assert info['measurementAgeS'] >= .079
        np.testing.assert_allclose(adapter.detector._latest_pos_m, BALL, atol=1e-7)
        assert feed(20, [BACKGROUND_B, BALL, BACKGROUND_A])
        _, velocity, valid, info = adapter.get_control_state()
        assert valid
        assert info['markerSelection']['selected_marker_id'] == 2
        assert info['frameIdx'] == 120
        np.testing.assert_array_equal(velocity, [0., 0., 0.])


def test_filter_is_before_robot_calibration_and_accepts_only_fresh_frames():
    clock = dict(wall=1800000000., mono=100.)
    detector = MocapBallDetector(marker_x_limits_m=(-2., 2.),
                                receiver_clock=lambda: clock['wall'], velocity_window_frames=1)
    detector.T_robot_mocap[0, 3] = 10.
    with patch('pingpong_controller.mocap_ball_detector.time.monotonic', lambda: clock['mono']):
        for index in range(12):
            clock.update(wall=1800000000.+index*.01, mono=100.+index*.01)
            # Direct process_packet callers using the legacy parser must also
            # select the true ball from the entire raw frame.
            packet = parse_mocap_packet(frame([BACKGROUND_A, BALL], index))
            assert detector.process_packet(packet, arrival_clock_s=clock['wall'],
                arrival_monotonic_s=clock['mono'], arrival_wall_s=clock['wall'])
        position, _, valid, before = detector.get_ball_state_with_diagnostics()
        assert valid and position[0] > 2.
        clock.update(wall=clock['wall']+.02, mono=clock['mono']+.02)
        duplicate = parse_mocap_packet(frame([BACKGROUND_A], 11))
        assert not detector.process_packet(duplicate, arrival_clock_s=clock['wall'],
            arrival_monotonic_s=clock['mono'], arrival_wall_s=clock['wall'])
        _, _, valid, after = detector.get_ball_state_with_diagnostics()
        assert valid and after['measurement_timestamp_s'] == before['measurement_timestamp_s']
        assert after['marker_selection']['status'] == 'selected'


def test_reordering_does_not_create_velocity_spikes():
    detector = MocapBallDetector(marker_x_limits_m=(-2., 2.), velocity_window_frames=1)
    for index in range(8):
        ball = (.01*index, 1., .8)
        points = [BACKGROUND_A, BACKGROUND_B]
        points.insert(index%3, ball)
        packet = parse_mocap_packet(frame(points, index))
        assert detector.process_packet(packet, arrival_clock_s=1800000000.+index*.01,
            arrival_monotonic_s=100.+index*.01, arrival_wall_s=1800000000.+index*.01)
    np.testing.assert_allclose(detector._latest_vel_m_s, [1., 0., 0.], atol=1e-6)
    assert detector._velocity_reject_count == 0
