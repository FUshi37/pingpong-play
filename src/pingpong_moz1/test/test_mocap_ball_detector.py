#!/usr/bin/env python3

import json
import socket
import struct
import sys
import time
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.mocap_ball_detector import (  # noqa: E402
    MocapBallDetector,
    _parse_optitrack_timestamp_string,
    parse_mocap_objects,
    parse_mocap_packet,
)


class MocapBallDetectorTest(unittest.TestCase):

    def test_flat_marker_json_from_mocap(self):
        payload = {
            'marker_frmIdx': 123,
            'marker_fTimestamp': 1.234,
            'marker_1_px': 1000.0,
            'marker_1_py': 2000.0,
            'marker_1_pz': 1500.0,
            'marker_1_vx': 100.0,
            'marker_1_vy': 200.0,
            'marker_1_vz': -50.0,
        }

        packet = parse_mocap_packet(
            json.dumps(payload).encode('utf-8'),
            marker_id=1,
            position_unit='mm',
            timestamp_unit='s',
        )

        self.assertIsNotNone(packet)
        self.assertEqual(packet.frame_idx, 123)
        self.assertAlmostEqual(packet.timestamp_s, 1.234)
        np.testing.assert_allclose(
            packet.position_m, [1.0, 2.0, 1.5], rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(
            packet.velocity_m_s, [0.1, 0.2, -0.05], rtol=0.0, atol=1e-7)

    def test_auto_unit_detects_millimeters(self):
        payload = {
            'marker_fTimestamp': 10,
            'marker_1_px': 500.0,
            'marker_1_py': -250.0,
            'marker_1_pz': 1200.0,
        }

        packet = parse_mocap_packet(
            json.dumps(payload).encode('utf-8'),
            marker_id=1,
            position_unit='auto',
        )

        self.assertIsNotNone(packet)
        np.testing.assert_allclose(
            packet.position_m, [0.5, -0.25, 1.2], rtol=0.0, atol=1e-7)
        self.assertIsNone(packet.velocity_m_s)

    def test_nested_marker_list_selects_marker_id(self):
        payload = {
            'timestamp': 1234.0,
            'markers': [
                {'id': 2, 'position': [9.0, 9.0, 9.0]},
                {
                    'id': 1,
                    'position': {'x': 1000.0, 'y': 2000.0, 'z': 1500.0},
                    'velocity': {'x': 100.0, 'y': 0.0, 'z': -50.0},
                },
            ],
        }

        packet = parse_mocap_packet(
            json.dumps(payload).encode('utf-8'),
            marker_id=1,
            position_unit='mm',
            timestamp_unit='ms',
        )

        self.assertIsNotNone(packet)
        self.assertAlmostEqual(packet.timestamp_s, 1.234)
        np.testing.assert_allclose(
            packet.position_m, [1.0, 2.0, 1.5], rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(
            packet.velocity_m_s, [0.1, 0.0, -0.05], rtol=0.0, atol=1e-7)

    def test_json_array_selects_marker_id(self):
        payload = [
            {'id': 2, 'x': 9.0, 'y': 9.0, 'z': 9.0},
            {'id': 1, 'x': 1.0, 'y': 2.0, 'z': 3.0},
        ]

        packet = parse_mocap_packet(
            json.dumps(payload).encode('utf-8'),
            marker_id=1,
            position_unit='m',
        )

        self.assertIsNotNone(packet)
        np.testing.assert_allclose(
            packet.position_m, [1.0, 2.0, 3.0], rtol=0.0, atol=1e-7)

    def test_legacy_binary_packet(self):
        data = struct.pack(
            '<d6f',
            2.5,
            1000.0, 2000.0, 3000.0,
            100.0, 200.0, 300.0,
        )

        packet = parse_mocap_packet(
            data,
            position_unit='mm',
            timestamp_unit='s',
        )

        self.assertIsNotNone(packet)
        self.assertEqual(packet.source_format, 'binary_d6f')
        self.assertAlmostEqual(packet.timestamp_s, 2.5)
        np.testing.assert_allclose(
            packet.position_m, [1.0, 2.0, 3.0], rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(
            packet.velocity_m_s, [0.1, 0.2, 0.3], rtol=0.0, atol=1e-7)

    def test_robot_transform_applies_translation_only_to_position(self):
        detector = MocapBallDetector()
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = np.array(
            [[0.0, -1.0, 0.0],
             [1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        transform[:3, 3] = [1.0, 2.0, 3.0]
        detector.T_robot_mocap = transform

        pos_robot = detector.transform_position(
            np.array([1.0, 0.0, 0.5], dtype=np.float32))
        vel_robot = detector.transform_velocity(
            np.array([1.0, 0.0, 0.5], dtype=np.float32))

        np.testing.assert_allclose(
            pos_robot, [1.0, 3.0, 3.5], rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(
            vel_robot, [0.0, 1.0, 0.5], rtol=0.0, atol=1e-7)

    def test_udp_receiver_updates_latest_state(self):
        detector = MocapBallDetector(
            udp_host='127.0.0.1',
            udp_port=0,
            timeout_s=1.0,
            marker_id=1,
            position_unit='mm',
        )
        if not detector.start():
            if isinstance(detector._last_start_error, PermissionError):
                self.skipTest('UDP sockets are blocked in this environment')
            self.fail(f'Failed to start UDP receiver: '
                      f'{detector._last_start_error}')
        try:
            port = detector._socket.getsockname()[1]
            payload = json.dumps({
                'marker_fTimestamp': 1.0,
                'marker_1_px': 1000.0,
                'marker_1_py': 2000.0,
                'marker_1_pz': 1500.0,
            }).encode('utf-8')
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.sendto(payload, ('127.0.0.1', port))
            finally:
                sock.close()

            deadline = time.time() + 1.0
            valid = False
            pos_m = None
            while time.time() < deadline:
                pos_m, _vel_m_s, valid = detector.get_ball_state()
                if valid:
                    break
                time.sleep(0.01)

            self.assertTrue(valid)
            np.testing.assert_allclose(
                pos_m, [1.0, 2.0, 1.5], rtol=0.0, atol=1e-7)
        finally:
            detector.stop()

    def test_optitrack_ball_marker_json(self):
        payload = {
            'marker_frmIdx': 42,
            'marker_fTimestamp': '00:00:01:500',
            'marker_2_px': 1000.0,
            'marker_2_py': 2000.0,
            'marker_2_pz': 1500.0,
        }
        packet = parse_mocap_packet(
            json.dumps(payload).encode('utf-8'),
            marker_id=2,
            object_kind='marker',
            position_unit='mm',
            timestamp_unit='s',
        )
        self.assertIsNotNone(packet)
        self.assertEqual(packet.frame_idx, 42)
        self.assertAlmostEqual(packet.timestamp_s, 1.5)
        np.testing.assert_allclose(
            packet.position_m, [1.0, 2.0, 1.5], rtol=0.0, atol=1e-7)
        self.assertIsNone(packet.velocity_m_s)

    def test_optitrack_rigid_body_json(self):
        payload = {
            'marker_fTimestamp': '00:00:02:000',
            'opti_rb_1_trackid': 1001,
            'opti_rb_1_id': 0,
            'opti_rb_1_tag': 0,
            'opti_rb_1_rigidtype': 1,
            'opti_rb_1_px': 1000.0,
            'opti_rb_1_py': 2000.0,
            'opti_rb_1_pz': 1500.0,
            'opti_rb_1_qw': 1.0,
            'opti_rb_1_qx': 0.0,
            'opti_rb_1_qy': 0.0,
            'opti_rb_1_qz': 0.0,
            'opti_rb_1_meanError': 0.5,
            'opti_rb_1_btracked': 1,
        }
        packet = parse_mocap_packet(
            json.dumps(payload).encode('utf-8'),
            object_kind='opti_rb',
            rigid_body_slot=1,
            rigid_body_id=1,
            position_unit='mm',
            timestamp_unit='s',
        )
        self.assertIsNotNone(packet)
        self.assertEqual(packet.source_format, 'optitrack_opti_rb')
        self.assertAlmostEqual(packet.timestamp_s, 2.0)
        np.testing.assert_allclose(
            packet.position_m, [1.0, 2.0, 1.5], rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(
            packet.orientation_wxyz, [1.0, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-7)
        self.assertTrue(packet.tracked)

    def test_optitrack_timestamp_string_parser(self):
        self.assertAlmostEqual(
            _parse_optitrack_timestamp_string('01:02:03:250'),
            3723.25)
        self.assertAlmostEqual(
            _parse_optitrack_timestamp_string('07:05:43.949'),
            25543.949)
        self.assertIsNone(_parse_optitrack_timestamp_string(''))

    def test_finite_difference_velocity_at_200hz(self):
        detector = MocapBallDetector(
            marker_id=2,
            velocity_alpha=1.0,
            max_velocity_m_s=50.0,
            velocity_window_frames=1,
        )
        with detector._lock:
            detector._estimate_velocity_locked(
                np.array([0.0, 0.0, 0.0], dtype=np.float32),
                0.0,
                0.0,
            )
            vel = detector._estimate_velocity_locked(
                np.array([0.1, 0.0, 0.0], dtype=np.float32),
                0.005,
                0.005,
            )
        np.testing.assert_allclose(vel, [20.0, 0.0, 0.0], rtol=0.0, atol=0.5)

    def test_three_frame_velocity_uses_source_time_without_position_average(self):
        detector = MocapBallDetector(
            marker_id=2,
            velocity_alpha=1.0,
            max_velocity_m_s=50.0,
            velocity_window_frames=3,
            velocity_window_max_span_s=0.05,
        )
        source_times = [10.000, 10.008, 10.017, 10.025]
        with detector._lock:
            for source_time_s in source_times:
                # Constant 4 m/s x motion sampled at non-uniform frame times.
                vel = detector._estimate_velocity_locked(
                    np.array([4.0 * (source_time_s - 10.0), 0.0, 0.0],
                             dtype=np.float32),
                    source_time_s,
                    1000.0 + source_time_s,
                )

        # p[3] - p[0] over t[3] - t[0], rather than an average of positions.
        np.testing.assert_allclose(vel, [4.0, 0.0, 0.0], rtol=0.0, atol=1e-5)
        diagnostics = detector.get_diagnostics()
        self.assertEqual(diagnostics['velocity_window_frames'], 3)
        self.assertEqual(diagnostics['velocity_time_source'], 'source')

    def test_three_frame_velocity_rejects_long_time_window(self):
        detector = MocapBallDetector(
            marker_id=2,
            velocity_alpha=1.0,
            max_velocity_m_s=50.0,
            velocity_window_frames=3,
            velocity_window_max_span_s=0.05,
        )
        with detector._lock:
            for source_time_s in (0.0, 0.008, 0.016, 0.080):
                vel = detector._estimate_velocity_locked(
                    np.array([source_time_s, 0.0, 0.0], dtype=np.float32),
                    source_time_s,
                    1000.0 + source_time_s,
                )

        np.testing.assert_allclose(vel, [0.0, 0.0, 0.0], rtol=0.0, atol=1e-7)
        self.assertEqual(
            detector.get_diagnostics()['velocity_window_reject_count'], 1)

    def test_optitrack_fractional_seconds_packet_uses_source_timestamp(self):
        payload = {
            'marker_fTimestamp': '07:05:43.949',
            'opti_rb_0_id': 5,
            'opti_rb_0_px': 1.0,
            'opti_rb_0_py': 2.0,
            'opti_rb_0_pz': 3.0,
            'opti_rb_0_btracked': 1,
        }
        packet = parse_mocap_packet(
            json.dumps(payload).encode('utf-8'),
            object_kind='opti_rb',
            rigid_body_slot=0,
            rigid_body_id=0,
            position_unit='m',
            timestamp_unit='s',
        )
        self.assertIsNotNone(packet)
        self.assertAlmostEqual(packet.timestamp_s, 25543.949)

    def test_parse_mocap_objects_optitrack_ball_and_rigid_body(self):
        payload = {
            'marker_fTimestamp': '00:00:00:000',
            'marker_2_px': 500.0,
            'marker_2_py': -250.0,
            'marker_2_pz': 1200.0,
            'opti_rb_1_id': 0,
            'opti_rb_1_px': 1000.0,
            'opti_rb_1_py': 2000.0,
            'opti_rb_1_pz': 1500.0,
            'opti_rb_1_btracked': 1,
        }
        objects = parse_mocap_objects(
            json.dumps(payload).encode('utf-8'),
            ball_marker_id=2,
            rigid_body_slot=1,
            rigid_body_id=1,
            position_unit='mm',
        )
        self.assertIsNotNone(objects['ball'])
        self.assertIsNotNone(objects['rigid_body'])
        np.testing.assert_allclose(
            objects['ball'].position_m, [0.5, -0.25, 1.2], rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(
            objects['rigid_body'].position_m, [1.0, 2.0, 1.5], rtol=0.0, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
