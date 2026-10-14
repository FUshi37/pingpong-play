"""Clock-domain and observation-timestamp regressions without robot actuation."""
import ast
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pingpong_controller.mocap_clock_sync import MocapClockSynchronizer
from pingpong_controller.mocap_ball_detector import parse_mocap_packet
from pingpong_controller.mocap_input import MocapBallInput


class MocapClockSyncTest(unittest.TestCase):
    def warm(self, sync, *, source=25543., receiver=1800000000., monotonic=100.):
        for i in range(12):
            result = sync.update(source + i*.01, receiver + i*.01,
                                 monotonic + i*.01, frame_idx=100+i)
        self.assertTrue(result.ready)
        return result

    def test_time_of_day_maps_to_receiver_epoch_after_warmup(self):
        sync = MocapClockSynchronizer()
        first = sync.update(25543., 1800000000., 100., frame_idx=1)
        self.assertFalse(first.ready)
        self.assertIsNone(first.receiver_timestamp_s)
        result = self.warm(MocapClockSynchronizer())
        self.assertAlmostEqual(result.receiver_timestamp_s, 1800000000.11)

    def test_variable_queue_delay_is_preserved_in_observation_age(self):
        sync = MocapClockSynchronizer()
        self.warm(sync)
        delayed = sync.update(25543.12, 1800000000.15, 100.15, frame_idx=112)
        self.assertTrue(delayed.ready)
        self.assertAlmostEqual(delayed.excess_delay_s, .03, places=6)
        self.assertAlmostEqual(1800000000.15-delayed.receiver_timestamp_s, .03, places=6)

    def test_duplicate_and_late_frame_do_not_refresh_clock(self):
        sync = MocapClockSynchronizer()
        previous = self.warm(sync)
        duplicate = sync.update(25543.11, 1800000000.12, 100.12, frame_idx=111)
        late = sync.update(25543.10, 1800000000.13, 100.13, frame_idx=110)
        self.assertFalse(duplicate.accepted)
        self.assertFalse(late.accepted)
        self.assertEqual(sync.epoch, previous.epoch)
        self.assertEqual(sync.duplicate_count, 1)
        self.assertEqual(sync.out_of_order_count, 1)

    def test_midnight_rollover_remains_continuous(self):
        sync = MocapClockSynchronizer()
        source_values = []
        for i in range(15):
            result = sync.update((86399.95+i*.01) % 86400,
                                 1800000000.+i*.01, 100.+i*.01, frame_idx=100+i)
            source_values.append(result.source_unwrapped_s)
        self.assertTrue(result.ready)
        self.assertEqual(sync.rollover_count, 1)
        self.assertEqual(sync.reset_count, 0)
        np.testing.assert_allclose(np.diff(source_values), .01, atol=1e-9)
        self.assertAlmostEqual(result.receiver_timestamp_s, 1800000000.14)
        late = sync.update(86399.99, 1800000000.15, 100.15, frame_idx=104)
        self.assertFalse(late.accepted)
        self.assertEqual(sync.reset_count, 0)

    def test_source_restart_requires_reacquisition(self):
        sync = MocapClockSynchronizer()
        self.warm(sync)
        result = sync.update(0., 1800000000.12, 100.12, frame_idx=0)
        self.assertFalse(result.ready)
        self.assertIsNone(result.receiver_timestamp_s)
        self.assertEqual(sync.reset_count, 1)
        for i in range(1, 12):
            result = sync.update(i*.01, 1800000000.12+i*.01,
                                 100.12+i*.01, frame_idx=i)
        self.assertTrue(result.ready)

    def test_receiver_clock_jump_requires_reacquisition(self):
        sync = MocapClockSynchronizer()
        self.warm(sync)
        result = sync.update(25543.12, 1800000060.12, 100.12, frame_idx=112)
        self.assertFalse(result.ready)
        self.assertIsNone(result.receiver_timestamp_s)
        self.assertEqual(sync.reset_count, 1)

    def test_long_receipt_gap_and_forward_source_jump_reacquire(self):
        for source, receiver, monotonic in (
                (25545.12, 1800000002.12, 102.12),
                (25603.12, 1800000000.12, 100.12)):
            with self.subTest(source=source):
                sync = MocapClockSynchronizer()
                self.warm(sync)
                result = sync.update(source, receiver, monotonic, frame_idx=112)
                self.assertFalse(result.ready)
                self.assertIsNone(result.receiver_timestamp_s)

    def test_missing_timestamp_cannot_become_synchronized(self):
        for missing in (None, float('nan'), float('inf'), -1.):
            with self.subTest(missing=missing):
                sync = MocapClockSynchronizer()
                self.warm(sync)
                result = sync.update(missing, 1800000000.12, 100.12)
                self.assertFalse(result.ready)
                self.assertIsNone(result.receiver_timestamp_s)

    def test_rolling_window_bounds_small_clock_drift(self):
        for rate in (.9999, 1.0001):
            with self.subTest(rate=rate):
                sync = MocapClockSynchronizer()
                for i in range(1201):
                    elapsed = i*.05
                    result = sync.update(1000.+elapsed*rate, 1800000000.+elapsed,
                                         100.+elapsed, frame_idx=i)
                self.assertTrue(result.ready)
                self.assertLess(1800000060.-result.receiver_timestamp_s, .002)
                self.assertLessEqual(result.receiver_timestamp_s, 1800000060.)

    def test_paused_receiver_clock_does_not_complete_sync(self):
        sync = MocapClockSynchronizer()
        for i in range(20):
            result = sync.update(i*.01, 10., 100.+i*.01, frame_idx=i)
        self.assertFalse(result.ready)
        self.assertIsNone(result.receiver_timestamp_s)

    def test_buffered_startup_burst_does_not_complete_sync(self):
        sync = MocapClockSynchronizer()
        for i in range(20):
            result = sync.update(i*.01, 1800000000.+i*.00001,
                                 100.+i*.00001, frame_idx=i)
        self.assertFalse(result.ready)


class MocapObservationTimingTest(unittest.TestCase):
    def setUp(self):
        self.clock = dict(receiver=1800000000., monotonic=100.)
        self.adapter = MocapBallInput(marker_id=2,
                                      receiver_clock=lambda: self.clock['receiver'])
        self.monotonic_patch = mock.patch(
            'pingpong_controller.mocap_ball_detector.time.monotonic',
            side_effect=lambda: self.clock['monotonic'])
        self.monotonic_patch.start()
        self.addCleanup(self.monotonic_patch.stop)

    def feed(self, i, *, delay=0., source=None, position=None):
        self.clock.update(receiver=1800000000.+i*.01+delay,
                          monotonic=100.+i*.01+delay)
        payload = dict(marker_frmIdx=100+i,
                       marker_fTimestamp=25543.+i*.01 if source is None else source,
                       marker_2_px=float(i if position is None else position),
                       marker_2_py=1., marker_2_pz=2.)
        packet = parse_mocap_packet(json.dumps(payload).encode(), marker_id=2,
                                    position_unit='m', object_kind='marker')
        return self.adapter.detector.process_packet(
            packet, arrival_clock_s=self.clock['receiver'],
            arrival_monotonic_s=self.clock['monotonic'],
            arrival_wall_s=self.clock['receiver'])

    def warm(self):
        for i in range(12):
            self.feed(i)

    def test_positions_are_gated_until_sync_and_keep_spatial_transform(self):
        self.feed(0)
        p, v, valid, feedback = self.adapter.get_control_state()
        self.assertFalse(valid)
        self.assertTrue(feedback['tracked'])
        self.assertFalse(feedback['clockSync']['ready'])
        self.assertIsNone(feedback['measurementTimestampS'])
        self.warm()
        p, v, valid, feedback = self.adapter.get_control_state()
        self.assertTrue(valid)
        np.testing.assert_allclose(p, [11., 1.023583, 3.302396], atol=1e-6)
        self.assertAlmostEqual(feedback['sourceTimestampS'], 25543.11)
        self.assertAlmostEqual(feedback['measurementTimestampS'], 1800000000.11)

    def test_configured_sync_span_holds_control_until_launch_criteria_pass(self):
        self.adapter = MocapBallInput(marker_id=2,
            receiver_clock=lambda: self.clock['receiver'],
            clock_sync_min_samples=16, clock_sync_min_span_s=.2, clock_sync_window_s=3.)
        for i in range(20):
            self.feed(i)
            self.assertFalse(self.adapter.get_control_state()[2])
        self.feed(20)
        self.assertTrue(self.adapter.get_control_state()[2])

    def test_repeated_packet_keeps_matching_position_stamp_and_receipt_age(self):
        self.warm()
        _, _, _, previous = self.adapter.get_control_state()
        self.assertFalse(self.feed(11, delay=.02, position=999.))
        p, _, valid, feedback = self.adapter.get_control_state()
        self.assertTrue(valid)
        self.assertEqual(float(p[0]), 11.)
        self.assertEqual(feedback['measurementTimestampS'], previous['measurementTimestampS'])
        self.assertEqual(feedback['packetCount'], previous['packetCount'])
        self.assertAlmostEqual(feedback['ageS'], .02)

    def test_receiver_clock_jump_invalidates_held_observation_before_next_packet(self):
        self.warm()
        self.clock['receiver'] += 60.
        _, _, valid, feedback = self.adapter.get_control_state()
        self.assertFalse(valid)
        self.assertEqual(feedback['clockSync']['status'], 'receiver_clock_changed')

    def test_snapshot_read_does_not_pair_position_with_a_separate_timestamp_read(self):
        self.warm()
        with mock.patch.object(self.adapter.detector, 'get_diagnostics',
                               side_effect=AssertionError('separate timestamp read')):
            p, _, valid, feedback = self.adapter.get_control_state()
        self.assertTrue(valid)
        self.assertEqual(float(p[0]), 11.)
        self.assertEqual(feedback['frameIdx'], 111)

    def test_bridge_and_policy_use_mapped_ros_time_and_measure_queue_age(self):
        from pingpong_controller.tools import interactive_policy_safety_probe as probe
        self.warm()
        self.feed(12, delay=.03)
        node = probe.RosBridgeNode.__new__(probe.RosBridgeNode)
        node.mocap_input = self.adapter
        node.mocap_lock = threading.Lock()
        node.mocap_marker_id = 2
        node.latest_mocap_feedback = {}
        node.cfg = SimpleNamespace(enable_mocap_ball=True, mocap_udp_host='0.0.0.0',
            mocap_udp_port=65501, mocap_rigid_body_slot=1, mocap_rigid_body_id=1,
            mocap_rigid_body_name='', mocap_calibration_path='',
            mocap_position_unit='m', mocap_timestamp_unit='s')
        p, v, valid, feedback = node.mocap_ball_state()
        mp, mv, measurement_valid, stamp, measurement = node.mocap_ball_measurement_with_diagnostics()
        self.assertTrue(valid and measurement_valid)
        np.testing.assert_array_equal(p, mp)
        np.testing.assert_array_equal(v, mv)
        self.assertEqual(stamp, feedback['stampS'])
        self.assertEqual(stamp, measurement['measurementTimestampS'])
        self.assertAlmostEqual(stamp, 1800000000.12)
        self.assertAlmostEqual(feedback['ageS'], 0.)
        backend = probe.PolicySafetyProbe.__new__(probe.PolicySafetyProbe)
        backend.cfg = SimpleNamespace(control_dt=.005, ball_lost_timeout_s=.6,
                                      racket_tracking_max_ball_age_s=.12)
        backend.ros_bridge = SimpleNamespace(now_sec=lambda: self.clock['receiver'])
        options = SimpleNamespace(policy_profile='fplc_gpu1_v172', control_mode='policy',
                                  ball_source='mocap')
        self.assertAlmostEqual(backend._raw_measurement_age_s(valid, stamp, options), .03, places=6)

    def test_live_packet_format_and_txt_calibration_produce_ground_height(self):
        calibration = Path(__file__).resolve().parents[3] / 'mocap_calibration.txt'
        adapter = MocapBallInput(marker_id=1, calibration_path=str(calibration),
                                 receiver_clock=lambda: self.clock['receiver'])
        for i in range(12):
            self.clock.update(receiver=1800000000.+i*.01, monotonic=100.+i*.01)
            # Actual 2026-10-12 packet shape and position; advance only its clock.
            payload = dict(marker_frmIdx=185162+i,
                marker_fTimestamp=f'12:21:{9.95+i*.01:06.3f}',
                marker_1_px=-.00357605, marker_1_py=.0170603, marker_1_pz=.00222578)
            packet = parse_mocap_packet(json.dumps(payload).encode(), marker_id=1,
                                        position_unit='auto', timestamp_unit='s')
            self.assertIsNotNone(packet)
            adapter.detector.process_packet(packet,
                arrival_clock_s=self.clock['receiver'],
                arrival_monotonic_s=self.clock['monotonic'],
                arrival_wall_s=self.clock['receiver'])
        p, _, valid, feedback = adapter.get_control_state()
        self.assertTrue(valid)
        self.assertAlmostEqual(feedback['sourceTimestampS'], 44470.06)
        self.assertAlmostEqual(feedback['measurementTimestampS'], 1800000000.11)
        np.testing.assert_allclose(p, [-.77950579, -.28227659, .01046574], atol=2e-7)

    def test_probe_estimator_recovers_after_synchronized_ros_clock_rewinds(self):
        from pingpong_controller.tools import interactive_policy_safety_probe as probe
        backend = probe.PolicySafetyProbe.__new__(probe.PolicySafetyProbe)
        backend.cfg = SimpleNamespace(control_dt=.005, ball_lost_timeout_s=.6,
                                      racket_tracking_max_ball_age_s=.12)
        backend.ros_bridge = SimpleNamespace(now_sec=lambda: 40.02)
        backend.now_s = 0.
        backend.last_ball_observation_stamp_s = 100.
        backend.next_ball_observation_s = 100.01
        backend.ball_observation_clock_domain = 'stamp'
        est = SimpleNamespace(reset=mock.Mock(), update=mock.Mock(),
            predict_to=mock.Mock(return_value=dict(valid=True, age_s=.02,
                position=[1.,2.,3.], velocity=[0.,0.,0.])))
        backend._ensure_ball_estimator = lambda options: est
        backend._estimator_payload = lambda options, out: out
        options = SimpleNamespace(policy_profile='fplc_gpu1_v172', control_mode='policy',
            ball_source='mocap', use_ball_estimator=True, estimator_hybrid_ekf9=False,
            ball_observation_rate_hz=90., estimator_control_max_extrapolation_s=.1)
        _, _, valid, _, _ = backend._resolve_control_ball_state(
            np.ones(3), np.zeros(3), True, options, measurement_stamp_s=40.)
        self.assertTrue(valid)
        est.reset.assert_called_once()
        self.assertEqual(est.update.call_args.args[1], 40.)
        self.assertEqual(backend.last_ball_observation_stamp_s, 40.)

    def test_runtime_node_uses_observation_stamp_and_resets_filter_on_clock_epoch(self):
        path = Path(__file__).resolve().parents[1] / 'pingpong_controller/pingpong_node.py'
        tree = ast.parse(path.read_text())
        method = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                      and node.name == '_get_mocap_ball_state')
        module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[
            ast.alias(name='annotations')], level=0), method], type_ignores=[])
        namespace = dict(np=np, BallState=SimpleNamespace, Time=SimpleNamespace)
        exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
        self.warm()
        node = SimpleNamespace(_mocap_detector=self.adapter.detector,
            _last_mocap_sync_epoch=None, _ball_estimator=SimpleNamespace(reset=mock.Mock()),
            _ball_estimator_lock=threading.Lock(), enable_ball_state_estimator=True,
            _last_mocap_estimator_packet_count=0, _last_mocap_recorded_packet_count=0,
            _feed_ball_state_estimator=mock.Mock(), _record_base_trajectory=mock.Mock(),
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(
                to_msg=lambda: SimpleNamespace(sec=0, nanosec=0))))
        result = namespace['_get_mocap_ball_state'](node)
        self.assertTrue(result.valid)
        self.assertAlmostEqual(result.stamp.sec + result.stamp.nanosec*1e-9, 1800000000.11)
        self.assertEqual(node._feed_ball_state_estimator.call_args.args[1],
                         result.raw_measurement['mocap']['measurement_timestamp_s'])
        # A source restart starts a new synchronization epoch and invalidates
        # control while preserving tracked position for calibration/diagnostics.
        self.feed(12, source=0.)
        result = namespace['_get_mocap_ball_state'](node)
        self.assertFalse(result.valid)
        node._ball_estimator.reset.assert_called_once()


if __name__ == '__main__':
    unittest.main()
