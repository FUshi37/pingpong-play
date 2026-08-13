#!/usr/bin/env python3
"""Mocap ball input adapter shared by probe and runtime nodes."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

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

    def __init__(self,
                 udp_host: str = '0.0.0.0',
                 udp_port: int = 9870,
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
                 sim_base_height_m: float = 1.302396):
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
        )
        self.marker_id = int(marker_id)
        self.position_unit = str(position_unit)
        self.timestamp_unit = str(timestamp_unit)
        self.sim_cartesian_x_offset_m = float(sim_cartesian_x_offset_m)
        self.sim_cartesian_y_offset_m = float(sim_cartesian_y_offset_m)
        self.sim_base_height_m = float(sim_base_height_m)
        self._last_start_error: Exception | None = None

    def start(self) -> bool:
        ok = bool(self.detector.start())
        if not ok:
            self._last_start_error = getattr(
                self.detector, '_last_start_error', None)
        return ok

    def stop(self) -> None:
        self.detector.stop()

    def get_waist03_state(
            self) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        pos_m, vel_m_s, valid = self.detector.get_ball_state()
        diagnostics = self.detector.get_diagnostics()
        return pos_m, vel_m_s, valid, diagnostics

    def get_control_state(
            self) -> Tuple[np.ndarray, np.ndarray, bool, Dict[str, Any]]:
        pos_waist, vel_waist, valid, diagnostics = self.get_waist03_state()
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
            'frameIdx': diagnostics.get('frame_idx'),
            'sourceFormat': diagnostics.get('source_format'),
            'packetCount': diagnostics.get('packet_count'),
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
        }
