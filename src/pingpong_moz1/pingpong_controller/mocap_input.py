#!/usr/bin/env python3
"""Mocap ball input adapter shared by probe and runtime nodes."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple
import time

import numpy as np
try:
    from pingpong_controller.probe_ball_recording import make_sensor_recording_sample
except ImportError:
    from probe_ball_recording import make_sensor_recording_sample

try:
    from pingpong_controller.mocap_ball_detector import MocapBallDetector
except ImportError:
    from mocap_ball_detector import MocapBallDetector

# Matches the vision hand-eye output frame documented in pingpong_node:
# robot Cartesian / moz1_pd.xml waist03 (historically named base_link).
WAIST03_CARTESIAN_FRAME = 'waist03_cartesian'
SIM_FRAME = 'sim_frame'


def transform_waist03_to_sim(
        pos_waist03_m: np.ndarray,
        *,
        sim_cartesian_x_offset_m: float = 0.0,
        sim_cartesian_y_offset_m: float = 0.023583,
        sim_base_height_m: float = 1.302396) -> np.ndarray:
    """Apply the same waist03 -> sim offset used by the interactive probe."""
    pos = np.asarray(pos_waist03_m, dtype=np.float64).reshape(3).copy()
    pos[0] += float(sim_cartesian_x_offset_m)
    pos[1] += float(sim_cartesian_y_offset_m)
    pos[2] += float(sim_base_height_m)
    return pos.astype(np.float32)


class MocapBallInput:
    """Receive mocap ball UDP packets and expose probe-compatible coordinates."""

    def resynchronize(self, reason='source_selected'):
        with self.detector._lock:
            self._resynchronize_locked(reason)

    def _resynchronize_locked(self, reason):
        self._recording_sync_request += 1
        self.detector.clock_sync.reset(reason)
        self.detector._latest_measurement_timestamp_s = None
        self.detector._velocity_history.clear()
        self.detector._prev_vel_mocap_m_s = None
        self.detector._prev_sample_time_s = None
        self.detector._prev_sample_time_source = None

    def set_marker_id(self, marker_id):
        marker_id = int(marker_id)
        with self.detector._lock:
            if marker_id != self.marker_id:
                self.marker_id = self.detector.marker_id = marker_id
                self._resynchronize_locked('marker_changed')

    def __init__(self,
                 udp_host: str = '0.0.0.0',
                 udp_port: int = 65501,
                 calibration_path: Optional[str] = None,
                 marker_id: int = 1,
                 position_unit: str = 'auto',
                 timestamp_unit: str = 's',
                 velocity_alpha: float = 1.0,
                 max_velocity_m_s: float = 20.0,
                 velocity_window_frames: int = 3,
                 velocity_window_max_span_s: float = 0.05,
                 timeout_s: float = 0.5,
                 sim_cartesian_x_offset_m: float = 0.0,
                 sim_cartesian_y_offset_m: float = 0.023583,
                 sim_base_height_m: float = 1.302396,
                 receiver_clock: Optional[Callable[[], float]] = None,
                 clock_sync_min_samples: int = 8,
                 clock_sync_min_span_s: float = 0.05,
                 clock_sync_window_s: float = 10.0,
                 marker_x_limits_m: Optional[Tuple[float, float]] = None,
                 enable_recording_cache: bool = False,
                 marker_z_limits_m: Optional[Tuple[float, float]] = (-1.0, 1.0)):
        self.detector = MocapBallDetector(
            udp_host=udp_host,
            udp_port=udp_port,
            calibration_path=calibration_path,
            timeout_s=timeout_s,
            marker_id=int(marker_id),
            position_unit=str(position_unit),
            timestamp_unit=str(timestamp_unit),
            velocity_alpha=float(velocity_alpha),
            max_velocity_m_s=float(max_velocity_m_s),
            velocity_window_frames=int(velocity_window_frames),
            velocity_window_max_span_s=float(velocity_window_max_span_s),
            receiver_clock=receiver_clock,
            clock_sync_min_samples=clock_sync_min_samples,
            clock_sync_min_span_s=clock_sync_min_span_s,
            clock_sync_window_s=clock_sync_window_s,
            marker_x_limits_m=marker_x_limits_m,
            marker_z_limits_m=marker_z_limits_m,
        )
        self.marker_id = int(marker_id)
        self.position_unit = str(position_unit)
        self.timestamp_unit = str(timestamp_unit)
        self.sim_cartesian_x_offset_m = float(sim_cartesian_x_offset_m)
        self.sim_cartesian_y_offset_m = float(sim_cartesian_y_offset_m)
        self.sim_base_height_m = float(sim_base_height_m)
        self._last_start_error: Exception | None = None
        self._recording_packet = None
        self._recording_sync_request = 0
        self._recording_error = None
        self._recording_closed = False
        if enable_recording_cache:
            self.detector.recording_callback = self._update_recording_packet

    def _update_recording_packet(self):
        """Run from the UDP receiver, never from the 200 Hz control thread."""
        request = self._recording_sync_request
        try:
            p, v, valid, feedback = self.get_control_state()
            self._recording_packet = (
                make_sensor_recording_sample('mocap', p, v, valid,
                                             feedback.get('measurementTimestampS'), feedback),
                request, time.monotonic() - float(feedback.get('ageS') or 0.))
            self._recording_error = None
        except Exception as exc:
            self._recording_error = str(exc)

    def recording_snapshot(self):
        return (self._recording_packet, self._recording_error,
                self._recording_closed, self._recording_sync_request, self.detector.timeout_s)

    def start(self) -> bool:
        self._recording_closed = False
        ok = bool(self.detector.start())
        if not ok:
            self._last_start_error = getattr(
                self.detector, '_last_start_error', None)
        return ok

    def stop(self) -> None:
        self._recording_closed = True
        self.detector.stop()

    def get_waist03_state(
            self) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        return self.detector.get_ball_state_with_diagnostics()

    def get_control_state(
            self) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        pos_waist, vel_waist, valid, diagnostics = self.get_waist03_state()
        valid = bool(valid and diagnostics.get('synchronized_valid', False))
        feedback = self.build_feedback(
            pos_waist, vel_waist, valid, diagnostics)
        if not valid:
            return (
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                False,
                feedback,
            )
        pos_sim = transform_waist03_to_sim(
            pos_waist,
            sim_cartesian_x_offset_m=self.sim_cartesian_x_offset_m,
            sim_cartesian_y_offset_m=self.sim_cartesian_y_offset_m,
            sim_base_height_m=self.sim_base_height_m,
        )
        vel_sim = np.asarray(vel_waist, dtype=np.float32).reshape(3)
        return pos_sim, vel_sim, True, feedback

    def build_feedback(
            self,
            pos_waist: np.ndarray,
            vel_waist: np.ndarray,
            valid: bool,
            diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        pos_waist = np.asarray(pos_waist, dtype=np.float64).reshape(3)
        vel_waist = np.asarray(vel_waist, dtype=np.float64).reshape(3)
        pos_sim = None
        vel_sim = None
        if valid:
            pos_sim = transform_waist03_to_sim(
                pos_waist,
                sim_cartesian_x_offset_m=self.sim_cartesian_x_offset_m,
                sim_cartesian_y_offset_m=self.sim_cartesian_y_offset_m,
                sim_base_height_m=self.sim_base_height_m,
            )
            vel_sim = vel_waist.astype(np.float32)
        return {
            'markerId': self.marker_id,
            'positionUnit': self.position_unit,
            'timestampUnit': self.timestamp_unit,
            'detected': bool(valid),
            'frame': WAIST03_CARTESIAN_FRAME,
            'controlFrame': SIM_FRAME,
            'waist03Ball': [round(float(v), 6) for v in pos_waist],
            'waist03Velocity': [round(float(v), 6) for v in vel_waist],
            'controlBall': (
                None if pos_sim is None
                else [round(float(v), 6) for v in pos_sim]),
            'controlVelocity': (
                None if vel_sim is None
                else [round(float(v), 6) for v in vel_sim]),
            'sourceTimestampS': diagnostics.get('source_timestamp_s'),
            'measurementTimestampS': diagnostics.get('measurement_timestamp_s'),
            'receiverTimestampS': diagnostics.get('receiver_timestamp_s'),
            'measurementAgeS': diagnostics.get('measurement_age_s'),
            'clockSync': diagnostics.get('clock_sync', {}),
            'tracked': bool(diagnostics.get('valid', False)),
            'frameIdx': diagnostics.get('frame_idx'),
            'sourceFormat': diagnostics.get('source_format'),
            'packetCount': diagnostics.get('packet_count'),
            'udp': diagnostics.get('udp', {}),
            'parseErrorCount': diagnostics.get('parse_error_count'),
            'velocityRejectCount': diagnostics.get('velocity_reject_count'),
            'velocityWindowRejectCount': diagnostics.get(
                'velocity_window_reject_count'),
            'velocityWindowFrames': diagnostics.get('velocity_window_frames'),
            'velocityWindowMaxSpanS': diagnostics.get(
                'velocity_window_max_span_s'),
            'velocityAlpha': diagnostics.get('velocity_alpha'),
            'velocityTimeSource': diagnostics.get('velocity_time_source'),
            'ageS': diagnostics.get('age_s'),
            'addr': diagnostics.get('addr'),
            'raw': diagnostics.get('raw'),
            'markerSelection': diagnostics.get('marker_selection', {}),
        }
