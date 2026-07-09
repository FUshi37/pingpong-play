#!/usr/bin/env python3
"""Motion capture ball detector - receives ball state via UDP."""

import json
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass
class MocapPacket:
    """One decoded mocap UDP packet in mocap coordinates."""

    position_m: np.ndarray
    velocity_m_s: Optional[np.ndarray]
    frame_idx: Optional[int]
    timestamp_s: Optional[float]
    source_format: str
    raw: Dict[str, Any]


def _to_float(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _to_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _scale_for_unit(values, unit: str) -> float:
    """Return position scale to meters."""
    unit = str(unit or 'auto').lower()
    if unit in ('m', 'meter', 'meters'):
        return 1.0
    if unit in ('mm', 'millimeter', 'millimeters'):
        return 1e-3
    if unit in ('cm', 'centimeter', 'centimeters'):
        return 1e-2
    if unit == 'auto':
        vals = np.asarray(values, dtype=float).reshape(-1)
        max_abs = float(np.nanmax(np.abs(vals))) if vals.size else 0.0
        # Mocap in meters is normally single-digit; millimeters are hundreds.
        return 1e-3 if max_abs > 20.0 else 1.0
    raise ValueError(
        f"Unsupported mocap position unit '{unit}', expected auto/m/mm/cm")


def _normalize_timestamp(value, unit: str) -> Optional[float]:
    t = _to_float(value)
    if t is None:
        return None
    unit = str(unit or 's').lower()
    if unit in ('s', 'sec', 'second', 'seconds'):
        return t
    if unit in ('ms', 'millisecond', 'milliseconds'):
        return t * 1e-3
    if unit in ('us', 'microsecond', 'microseconds'):
        return t * 1e-6
    if unit in ('ns', 'nanosecond', 'nanoseconds'):
        return t * 1e-9
    if unit == 'auto':
        at = abs(t)
        if at > 1e16:
            return t * 1e-9
        if at > 1e13:
            return t * 1e-6
        if at > 1e10:
            return t * 1e-3
        return t
    raise ValueError(
        f"Unsupported mocap timestamp unit '{unit}', expected auto/s/ms/us/ns")


def _first_number(payload: Dict[str, Any], keys) -> Optional[float]:
    for key in keys:
        if key in payload:
            value = _to_float(payload.get(key))
            if value is not None:
                return value
    return None


def _extract_xyz(payload: Dict[str, Any], marker_id: int, names) -> Optional[np.ndarray]:
    prefix = f'marker_{int(marker_id)}_'
    candidates = []
    for base in names:
        candidates.append((prefix + base[0], prefix + base[1], prefix + base[2]))
    for base in names:
        candidates.append(base)

    for keys in candidates:
        vals = [_first_number(payload, [key]) for key in keys]
        if all(v is not None for v in vals):
            return np.array(vals, dtype=np.float64)
    return None


def parse_mocap_packet(data: bytes,
                       marker_id: int = 1,
                       position_unit: str = 'auto',
                       timestamp_unit: str = 's') -> Optional[MocapPacket]:
    """Parse one UDP packet from the mocap system.

    Supported formats:
      - JSON:
        {"marker_frmIdx": 1, "marker_fTimestamp": 0.01,
         "marker_1_px": 0.1, "marker_1_py": 0.2, "marker_1_pz": 0.3}
      - Legacy binary little-endian: <d6f> timestamp, xyz, vxyz.
    """
    if not data:
        return None

    stripped = data.strip()
    if stripped[:1] in (b'{', b'['):
        try:
            payload = json.loads(stripped.decode('utf-8', errors='replace'))
        except json.JSONDecodeError:
            return None
        if isinstance(payload, list):
            if not payload or not isinstance(payload[0], dict):
                return None
            payload = payload[0]
        if not isinstance(payload, dict):
            return None

        pos = _extract_xyz(
            payload, marker_id,
            names=(('px', 'py', 'pz'), ('x', 'y', 'z'), ('pos_x', 'pos_y', 'pos_z')))
        if pos is None:
            return None
        scale = _scale_for_unit(pos, position_unit)
        pos_m = (pos * scale).astype(np.float32)

        vel = _extract_xyz(
            payload, marker_id,
            names=(('vx', 'vy', 'vz'), ('vel_x', 'vel_y', 'vel_z')))
        vel_m_s = (vel * scale).astype(np.float32) if vel is not None else None

        frame_idx = _to_int(payload.get('marker_frmIdx'))
        if frame_idx is None:
            frame_idx = _to_int(payload.get('frame_idx', payload.get('frame')))

        timestamp_value = None
        for key in (
                'marker_fTimestamp', 'timestamp', 'time', 't',
                f'marker_{int(marker_id)}_timestamp',
                f'marker_{int(marker_id)}_time'):
            if key in payload:
                timestamp_value = payload.get(key)
                break

        return MocapPacket(
            position_m=pos_m,
            velocity_m_s=vel_m_s,
            frame_idx=frame_idx,
            timestamp_s=_normalize_timestamp(timestamp_value, timestamp_unit),
            source_format='json',
            raw=payload)

    if len(data) >= 32:
        try:
            values = struct.unpack('<d6f', data[:32])
        except struct.error:
            return None
        pos = np.array([values[1], values[2], values[3]], dtype=np.float64)
        scale = _scale_for_unit(pos, position_unit)
        vel = np.array([values[4], values[5], values[6]], dtype=np.float64)
        return MocapPacket(
            position_m=(pos * scale).astype(np.float32),
            velocity_m_s=(vel * scale).astype(np.float32),
            frame_idx=None,
            timestamp_s=_normalize_timestamp(values[0], timestamp_unit),
            source_format='binary_d6f',
            raw={})

    return None


class MocapBallDetector:
    """Receives ball state from motion capture system via UDP.

    JSON packets with position-only markers are accepted; velocity is estimated
    from consecutive packets. The legacy binary <d6f> packet remains supported.
    """

    def __init__(self,
                 udp_host: str = '0.0.0.0',
                 udp_port: int = 9870,
                 calibration_path: Optional[str] = None,
                 timeout_s: float = 0.5,
                 marker_id: int = 1,
                 position_unit: str = 'auto',
                 timestamp_unit: str = 's',
                 velocity_alpha: float = 0.35,
                 max_velocity_m_s: float = 20.0):
        """Initialize mocap ball detector.

        Args:
            udp_host: UDP address to bind.
            udp_port: UDP port to listen on
            calibration_path: Path to calibration file (4x4 transform matrix).
                            If None, identity transform is used.
            timeout_s: Time after which ball state is considered invalid (seconds)
            marker_id: Marker index used by marker_<id>_px/py/pz JSON fields.
            position_unit: auto/m/mm/cm for incoming positions.
            timestamp_unit: s/ms/us/ns/auto for incoming timestamps.
            velocity_alpha: EMA smoothing factor for finite-difference velocity.
            max_velocity_m_s: Reject finite-difference velocity above this speed.
        """
        self.udp_host = str(udp_host)
        self.udp_port = udp_port
        self.timeout_s = timeout_s
        self.marker_id = int(marker_id)
        self.position_unit = str(position_unit or 'auto')
        self.timestamp_unit = str(timestamp_unit or 's')
        self.velocity_alpha = min(max(float(velocity_alpha), 0.0), 1.0)
        self.max_velocity_m_s = float(max_velocity_m_s)
        self._min_velocity_dt_s = 1e-4
        self._max_velocity_dt_s = 0.25

        # Calibration transform: T_robot_mocap (4x4 homogeneous matrix)
        # Transforms points from mocap frame to robot base_link frame
        self.T_robot_mocap = np.eye(4, dtype=np.float64)
        if calibration_path is not None:
            self.load_calibration(calibration_path)

        # Latest ball state (in robot frame)
        self._lock = threading.Lock()
        self._latest_pos_m = np.zeros(3, dtype=np.float32)  # meters
        self._latest_vel_m_s = np.zeros(3, dtype=np.float32)  # m/s
        self._latest_timestamp = 0.0  # wall time when last packet received
        self._latest_source_timestamp_s = None
        self._latest_frame_idx = None
        self._latest_format = None
        self._latest_addr = None
        self._latest_raw = {}
        self._packet_count = 0
        self._parse_error_count = 0
        self._velocity_reject_count = 0

        self._prev_pos_mocap_m = None
        self._prev_vel_mocap_m_s = None
        self._prev_sample_time_s = None
        self._prev_arrival_time_s = None

        # UDP receiver thread
        self._running = False
        self._thread = None
        self._socket = None

    def load_calibration(self, path: str) -> None:
        """Load calibration transform from file.

        File format: 4x4 homogeneous transform matrix (text, space-separated)
        """
        try:
            T = np.loadtxt(path, dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError(f"Calibration matrix must be 4x4, got {T.shape}")
            self.T_robot_mocap = T
            print(f"[MocapBallDetector] Loaded calibration from {path}")
            print(f"  T_robot_mocap:\n{self.T_robot_mocap}")
        except Exception as e:
            print(f"[MocapBallDetector] Failed to load calibration: {e}")
            print(f"  Using identity transform")

    def save_calibration(self, path: str) -> None:
        """Save calibration transform to file."""
        np.savetxt(path, self.T_robot_mocap, fmt='%.6f')
        print(f"[MocapBallDetector] Saved calibration to {path}")

    def start(self) -> None:
        """Start UDP receiver thread."""
        if self._running:
            return

        # Open UDP socket
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.bind((self.udp_host, self.udp_port))
            self._socket.settimeout(1.0)  # 1s timeout for blocking recv
            print(
                f"[MocapBallDetector] Listening on "
                f"{self.udp_host}:{self.udp_port} "
                f"(marker={self.marker_id}, unit={self.position_unit})")
        except Exception as e:
            print(f"[MocapBallDetector] Failed to open UDP socket: {e}")
            return

        self._running = True
        self._thread = threading.Thread(target=self._receiver_loop, daemon=True)
        self._thread.start()
        print(f"[MocapBallDetector] Receiver thread started")

    def stop(self) -> None:
        """Stop UDP receiver thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        print(f"[MocapBallDetector] Stopped")

    def _receiver_loop(self) -> None:
        """Background thread: receive UDP packets and update ball state."""
        while self._running:
            try:
                # Receive UDP packet (blocking with timeout)
                data, addr = self._socket.recvfrom(1024)
                arrival_time_s = time.time()

                packet = parse_mocap_packet(
                    data,
                    marker_id=self.marker_id,
                    position_unit=self.position_unit,
                    timestamp_unit=self.timestamp_unit)
                if packet is None:
                    with self._lock:
                        self._parse_error_count += 1
                    continue

                # Update latest state
                with self._lock:
                    vel_mocap = packet.velocity_m_s
                    if vel_mocap is None:
                        vel_mocap = self._estimate_velocity_locked(
                            packet.position_m,
                            packet.timestamp_s,
                            arrival_time_s)
                    else:
                        self._remember_velocity_sample_locked(
                            packet.position_m, vel_mocap,
                            packet.timestamp_s, arrival_time_s)

                    # Transform to robot frame
                    pos_robot = self.transform_position(packet.position_m)
                    vel_robot = self.transform_velocity(vel_mocap)

                    self._latest_pos_m = pos_robot
                    self._latest_vel_m_s = vel_robot
                    self._latest_timestamp = arrival_time_s
                    self._latest_source_timestamp_s = packet.timestamp_s
                    self._latest_frame_idx = packet.frame_idx
                    self._latest_format = packet.source_format
                    self._latest_addr = addr
                    self._latest_raw = packet.raw
                    self._packet_count += 1

            except socket.timeout:
                # No packet received within timeout - this is fine
                continue
            except Exception as e:
                if self._running:
                    print(f"[MocapBallDetector] Receiver error: {e}")
                continue

    def _sample_time_locked(self, source_timestamp_s, arrival_time_s) -> float:
        if source_timestamp_s is None:
            return float(arrival_time_s)
        t = float(source_timestamp_s)
        if self._prev_sample_time_s is None:
            return t
        dt = t - self._prev_sample_time_s
        if self._min_velocity_dt_s <= dt <= self._max_velocity_dt_s:
            return t
        # Fall back to arrival time when source timestamps are absent, duplicate,
        # in another clock domain, or configured with the wrong unit.
        if self._prev_arrival_time_s is not None:
            return float(arrival_time_s)
        return t

    def _remember_velocity_sample_locked(self, pos_mocap_m, vel_mocap_m_s,
                                         source_timestamp_s,
                                         arrival_time_s) -> None:
        sample_time_s = self._sample_time_locked(
            source_timestamp_s, arrival_time_s)
        self._prev_pos_mocap_m = np.asarray(pos_mocap_m, dtype=np.float32).copy()
        self._prev_vel_mocap_m_s = np.asarray(
            vel_mocap_m_s, dtype=np.float32).copy()
        self._prev_sample_time_s = sample_time_s
        self._prev_arrival_time_s = float(arrival_time_s)

    def _estimate_velocity_locked(self, pos_mocap_m, source_timestamp_s,
                                  arrival_time_s) -> np.ndarray:
        sample_time_s = self._sample_time_locked(
            source_timestamp_s, arrival_time_s)
        pos = np.asarray(pos_mocap_m, dtype=np.float32).reshape(3)

        if self._prev_pos_mocap_m is None or self._prev_sample_time_s is None:
            vel = np.zeros(3, dtype=np.float32)
        else:
            dt = sample_time_s - self._prev_sample_time_s
            if dt < self._min_velocity_dt_s or dt > self._max_velocity_dt_s:
                dt = (
                    float(arrival_time_s) - self._prev_arrival_time_s
                    if self._prev_arrival_time_s is not None else 0.0)
            if dt < self._min_velocity_dt_s or dt > self._max_velocity_dt_s:
                vel = (
                    self._prev_vel_mocap_m_s.copy()
                    if self._prev_vel_mocap_m_s is not None
                    else np.zeros(3, dtype=np.float32))
            else:
                raw_vel = (pos - self._prev_pos_mocap_m) / float(dt)
                speed = float(np.linalg.norm(raw_vel))
                if (self.max_velocity_m_s > 0.0
                        and speed > self.max_velocity_m_s):
                    self._velocity_reject_count += 1
                    vel = (
                        self._prev_vel_mocap_m_s.copy()
                        if self._prev_vel_mocap_m_s is not None
                        else np.zeros(3, dtype=np.float32))
                elif self._prev_vel_mocap_m_s is not None:
                    alpha = self.velocity_alpha
                    vel = (alpha * raw_vel
                           + (1.0 - alpha) * self._prev_vel_mocap_m_s)
                else:
                    vel = raw_vel
                vel = np.asarray(vel, dtype=np.float32)

        self._prev_pos_mocap_m = pos.copy()
        self._prev_vel_mocap_m_s = vel.copy()
        self._prev_sample_time_s = sample_time_s
        self._prev_arrival_time_s = float(arrival_time_s)
        return vel

    def transform_position(self, pos_mocap: np.ndarray) -> np.ndarray:
        """Transform position from mocap frame to robot frame.

        Args:
            pos_mocap: Position in mocap frame (3,) in meters

        Returns:
            Position in robot base_link frame (3,) in meters
        """
        # Convert to homogeneous coordinates
        p_mocap_h = np.array([pos_mocap[0], pos_mocap[1], pos_mocap[2], 1.0],
                             dtype=np.float64)
        # Transform
        p_robot_h = self.T_robot_mocap @ p_mocap_h
        # Extract xyz
        return p_robot_h[:3].astype(np.float32)

    def transform_velocity(self, vel_mocap: np.ndarray) -> np.ndarray:
        """Transform velocity from mocap frame to robot frame.

        Velocity transforms via rotation only (no translation).

        Args:
            vel_mocap: Velocity in mocap frame (3,) in m/s

        Returns:
            Velocity in robot base_link frame (3,) in m/s
        """
        # Extract rotation matrix from T_robot_mocap
        R_robot_mocap = self.T_robot_mocap[:3, :3]
        # Rotate velocity
        vel_robot = R_robot_mocap @ vel_mocap.astype(np.float64)
        return vel_robot.astype(np.float32)

    def get_ball_state(self) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Get latest ball state in robot frame.

        Returns:
            (position, velocity, valid) where:
                - position: (3,) array in meters (robot base_link frame)
                - velocity: (3,) array in m/s (robot base_link frame)
                - valid: True if data is fresh (within timeout)
        """
        with self._lock:
            pos = self._latest_pos_m.copy()
            vel = self._latest_vel_m_s.copy()
            age = time.time() - self._latest_timestamp
            valid = (self._packet_count > 0) and (age < self.timeout_s)

        return pos, vel, valid

    def get_diagnostics(self) -> dict:
        """Get diagnostic info."""
        with self._lock:
            age = time.time() - self._latest_timestamp
            return {
                'packet_count': self._packet_count,
                'age_s': age,
                'valid': (self._packet_count > 0) and (age < self.timeout_s),
                'parse_error_count': self._parse_error_count,
                'velocity_reject_count': self._velocity_reject_count,
                'frame_idx': self._latest_frame_idx,
                'source_timestamp_s': self._latest_source_timestamp_s,
                'arrival_time_s': self._latest_timestamp,
                'source_format': self._latest_format,
                'addr': self._latest_addr,
                'raw': dict(self._latest_raw) if isinstance(
                    self._latest_raw, dict) else self._latest_raw,
            }
