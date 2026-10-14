#!/usr/bin/env python3
"""Motion capture ball detector - receives ball state via UDP."""

from collections import deque
import json
import re
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

try:
    from pingpong_controller.mocap_clock_sync import MocapClockSynchronizer
except ImportError:
    from mocap_clock_sync import MocapClockSynchronizer

_BINARY_PACKET_FORMAT = '<d6f'
_BINARY_PACKET_SIZE = struct.calcsize(_BINARY_PACKET_FORMAT)


@dataclass
class MocapPacket:
    """One decoded mocap UDP packet in mocap coordinates."""

    position_m: np.ndarray
    velocity_m_s: Optional[np.ndarray]
    frame_idx: Optional[int]
    timestamp_s: Optional[float]
    source_format: str
    raw: Dict[str, Any]
    orientation_wxyz: Optional[np.ndarray] = None
    tracked: Optional[bool] = None
    mean_error_m: Optional[float] = None
    marker_selection_info: Optional[Dict[str, Any]] = None


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
    if value is None:
        return None
    if isinstance(value, str):
        parsed = _parse_optitrack_timestamp_string(value)
        if parsed is not None:
            return parsed
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


def _parse_optitrack_timestamp_string(value: str) -> Optional[float]:
    """Parse OptiTrack timestamps such as ``HH:MM:SS.mmm`` or ``HH:MM:SS:ms``."""
    text = str(value or '').strip()
    if not text:
        return None
    parts = text.split(':')
    if len(parts) == 3:
        try:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            if hours < 0.0 or not 0.0 <= minutes < 60.0 or not 0.0 <= seconds < 60.0:
                return None
            return hours * 3600.0 + minutes * 60.0 + seconds
        except (TypeError, ValueError):
            return None
    if len(parts) == 4:
        try:
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            tail = float(parts[3])
            if hours < 0.0 or not 0.0 <= minutes < 60.0 or not 0.0 <= seconds < 60.0:
                return None
            # The last field is usually milliseconds when it looks like an integer.
            frac_s = tail * 1e-3 if abs(tail) >= 1.0 else tail
            return hours * 3600.0 + minutes * 60.0 + seconds + frac_s
        except (TypeError, ValueError):
            return None
    return None


def _extract_opti_rb_position(payload: Dict[str, Any],
                              slot: int) -> Optional[np.ndarray]:
    prefix = f'opti_rb_{int(slot)}_'
    vals = [
        _first_number(payload, [prefix + 'px']),
        _first_number(payload, [prefix + 'py']),
        _first_number(payload, [prefix + 'pz']),
    ]
    if all(v is not None for v in vals):
        return np.array(vals, dtype=np.float64)
    return None


def _extract_opti_rb_orientation(payload: Dict[str, Any],
                                   slot: int) -> Optional[np.ndarray]:
    prefix = f'opti_rb_{int(slot)}_'
    vals = [
        _first_number(payload, [prefix + 'qw']),
        _first_number(payload, [prefix + 'qx']),
        _first_number(payload, [prefix + 'qy']),
        _first_number(payload, [prefix + 'qz']),
    ]
    if all(v is not None for v in vals):
        return np.array(vals, dtype=np.float64)
    return None


def _opti_rb_tracked(payload: Dict[str, Any], slot: int) -> Optional[bool]:
    prefix = f'opti_rb_{int(slot)}_'
    key = prefix + 'btracked'
    if key not in payload:
        return None
    parsed = _to_int(payload.get(key))
    if parsed is not None:
        return bool(parsed)
    value = payload.get(key)
    if isinstance(value, bool):
        return value
    return bool(value)


def _opti_rb_id_matches(payload: Dict[str, Any],
                        rigid_body_slot: int,
                        rigid_body_id: Optional[int]) -> bool:
    """Validate the id carried by the selected ``opti_rb_<slot>`` record.

    Some versions of the existing bridge use the numeric prefix itself as the
    rigid-body identifier and put an unrelated/legacy value in ``*_id``.  Keep
    that same-slot convention for compatibility, while correctly validating
    the field value when callers deliberately configure different slot/id
    values.
    """
    if rigid_body_id is None:
        return True
    slot = int(rigid_body_slot)
    target_id = int(rigid_body_id)
    key = f'opti_rb_{slot}_id'
    if key not in payload:
        return False
    field_id = _to_int(payload.get(key))
    return field_id == target_id or slot == target_id


def _parse_opti_rb_packet(payload: Dict[str, Any],
                          *,
                          rigid_body_slot: int,
                          rigid_body_id: Optional[int],
                          position_unit: str,
                          timestamp_unit: str) -> Optional[MocapPacket]:
    if not isinstance(payload, dict):
        return None
    if not _opti_rb_id_matches(payload, rigid_body_slot, rigid_body_id):
        return None
    tracked = _opti_rb_tracked(payload, rigid_body_slot)
    if tracked is False:
        return None
    pos = _extract_opti_rb_position(payload, rigid_body_slot)
    if pos is None:
        return None
    scale = _scale_for_unit(pos, position_unit)
    orientation = _extract_opti_rb_orientation(payload, rigid_body_slot)
    prefix = f'opti_rb_{int(rigid_body_slot)}_'
    mean_error = _first_number(payload, [prefix + 'meanError'])
    mean_error_m = (
        float(mean_error) * scale if mean_error is not None else None)
    frame_idx = _to_int(payload.get(f'opti_rb_{int(rigid_body_slot)}_trackid'))
    timestamp_value = _first_present(
        [payload],
        ('marker_fTimestamp', 'timestamp', 'time', 't'))
    return MocapPacket(
        position_m=(pos * scale).astype(np.float32),
        velocity_m_s=None,
        frame_idx=frame_idx,
        timestamp_s=_normalize_timestamp(timestamp_value, timestamp_unit),
        source_format='optitrack_opti_rb',
        raw=payload,
        orientation_wxyz=orientation,
        tracked=tracked,
        mean_error_m=mean_error_m,
    )


def _first_number(payload: Dict[str, Any], keys) -> Optional[float]:
    for key in keys:
        if key in payload:
            value = _to_float(payload.get(key))
            if value is not None:
                return value
    return None


def _extract_xyz(payload: Dict[str, Any],
                 marker_id: int, names) -> Optional[np.ndarray]:
    prefixes = [f'marker_{int(marker_id)}_', f'rigid_body_{int(marker_id)}_',
                f'body_{int(marker_id)}_']
    candidates = []
    for prefix in prefixes:
        for base in names:
            candidates.append((
                prefix + base[0], prefix + base[1], prefix + base[2]))
    for base in names:
        candidates.append(base)

    for keys in candidates:
        vals = [_first_number(payload, [key]) for key in keys]
        if all(v is not None for v in vals):
            return np.array(vals, dtype=np.float64)
    return None


def _object_matches(payload: Dict[str, Any],
                  object_id: Optional[int] = None,
                  object_name: Optional[str] = None) -> bool:
    """Return True when a nested marker/rigid-body dict matches id or name."""
    if object_name:
        target_name = str(object_name).strip().lower()
        for key in ('name', 'body_name', 'rigid_body_name', 'object_name',
                    'label', 'marker'):
            if key not in payload:
                continue
            value = payload.get(key)
            if value is None:
                continue
            if str(value).strip().lower() == target_name:
                return True

    if object_id is None:
        return False

    for key in ('id', 'marker_id', 'markerId', 'marker', 'body_id',
                'rigid_body_id', 'object_id'):
        if key not in payload:
            continue
        value = payload.get(key)
        if str(value) == str(object_id):
            return True
        parsed = _to_int(value)
        if parsed == int(object_id):
            return True
    return False


def _marker_matches(payload: Dict[str, Any], marker_id: int) -> bool:
    return _object_matches(payload, object_id=marker_id)


def _iter_marker_payloads(payload: Any,
                          marker_id: int,
                          object_name: Optional[str] = None):
    """Yield nested marker/rigid-body dictionaries, preferring matches."""
    if isinstance(payload, list):
        dicts = [item for item in payload if isinstance(item, dict)]
        matches = [
            item for item in dicts
            if _object_matches(item, marker_id, object_name)]
        for item in matches or dicts:
            yield item
        return

    if not isinstance(payload, dict):
        return

    for key in ('markers', 'rigid_bodies', 'bodies', 'objects', 'data'):
        value = payload.get(key)
        if isinstance(value, list):
            dicts = [item for item in value if isinstance(item, dict)]
            matches = [
                item for item in dicts
                if _object_matches(item, marker_id, object_name)]
            for item in matches or dicts:
                yield item
        elif isinstance(value, dict):
            if (_object_matches(value, marker_id, object_name)
                    or key == 'data'):
                yield value


def _coerce_vector(value: Any, component_names) -> Optional[np.ndarray]:
    """Convert a list/tuple/dict vector to a finite xyz numpy array."""
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        vals = [_to_float(value[0]), _to_float(value[1]), _to_float(value[2])]
        if all(v is not None for v in vals):
            return np.array(vals, dtype=np.float64)

    if isinstance(value, dict):
        for names in component_names:
            vals = [_first_number(value, [key]) for key in names]
            if all(v is not None for v in vals):
                return np.array(vals, dtype=np.float64)
    return None


def _extract_named_vector(payload: Dict[str, Any], marker_id: int,
                          keys, component_names) -> Optional[np.ndarray]:
    """Extract vector fields such as position:[x,y,z] or marker_1_pos:{...}."""
    prefix = f'marker_{int(marker_id)}_'
    for key in keys:
        for candidate in (prefix + key, key):
            if candidate in payload:
                vec = _coerce_vector(payload.get(candidate), component_names)
                if vec is not None:
                    return vec
    return None


def _extract_position(payload: Dict[str, Any],
                      marker_id: int) -> Optional[np.ndarray]:
    component_names = (
        ('px', 'py', 'pz'),
        ('x', 'y', 'z'),
        ('pos_x', 'pos_y', 'pos_z'),
    )
    pos = _extract_xyz(payload, marker_id, component_names)
    if pos is not None:
        return pos
    return _extract_named_vector(
        payload, marker_id,
        keys=('position', 'pos', 'translation', 'location', 'point', 'p'),
        component_names=component_names)


def _extract_velocity(payload: Dict[str, Any],
                      marker_id: int) -> Optional[np.ndarray]:
    flat_component_names = (
        ('vx', 'vy', 'vz'),
        ('vel_x', 'vel_y', 'vel_z'),
        ('velocity_x', 'velocity_y', 'velocity_z'),
    )
    vel = _extract_xyz(payload, marker_id, flat_component_names)
    if vel is not None:
        return vel
    named_component_names = flat_component_names + (('x', 'y', 'z'),)
    return _extract_named_vector(
        payload, marker_id,
        keys=('velocity', 'vel', 'v'),
        component_names=named_component_names)


def _first_present(payloads, keys) -> Any:
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key in keys:
            if key in payload:
                return payload.get(key)
    return None


def parse_mocap_objects(
        data: bytes,
        *,
        ball_marker_id: Optional[int] = None,
        rigid_body_slot: Optional[int] = None,
        rigid_body_id: Optional[int] = None,
        rigid_body_name: Optional[str] = None,
        position_unit: str = 'auto',
        timestamp_unit: str = 's') -> Dict[str, Optional['MocapPacket']]:
    """Parse one UDP packet and extract configured ball / rigid-body objects."""
    out: Dict[str, Optional[MocapPacket]] = {}
    if ball_marker_id is not None:
        out['ball'] = parse_mocap_packet(
            data,
            marker_id=int(ball_marker_id),
            object_kind='marker',
            position_unit=position_unit,
            timestamp_unit=timestamp_unit,
        )
    if rigid_body_slot is not None:
        out['rigid_body'] = parse_mocap_packet(
            data,
            object_kind='opti_rb',
            rigid_body_slot=int(rigid_body_slot),
            rigid_body_id=(
                int(rigid_body_id)
                if rigid_body_id is not None else int(rigid_body_slot)),
            position_unit=position_unit,
            timestamp_unit=timestamp_unit,
        )
    elif rigid_body_id is not None or rigid_body_name:
        out['rigid_body'] = parse_mocap_packet(
            data,
            marker_id=int(rigid_body_id if rigid_body_id is not None else 0),
            object_name=rigid_body_name,
            object_kind='marker',
            position_unit=position_unit,
            timestamp_unit=timestamp_unit,
        )
    return out


def _validate_marker_x_limits(limits):
    if limits is None:
        return None
    values = np.asarray(limits, dtype=np.float64)
    if (values.shape != (2,) or not np.all(np.isfinite(values))
            or values[0] >= values[1]):
        raise ValueError('Mocap marker X limits must be a finite increasing pair')
    return (float(values[0]), float(values[1]))


def _parse_x_filtered_marker_frame(payload, *, marker_id, limits,
                                   position_unit, timestamp_unit):
    """Select one raw mocap point, independently of its changing field index.

    A missing/ambiguous observation still carries frame timing, but its
    position must never enter the measurement or velocity history.
    """
    roots = payload if isinstance(payload, list) else [payload]
    roots = [item for item in roots if isinstance(item, dict)]
    candidates = []
    timing_payloads = list(roots)
    for root in roots:
        indices = sorted({int(match.group(1)) for key in root
                          if (match := re.fullmatch(r'marker_(\d+)_px', key))})
        if indices:
            candidates.extend((root, index) for index in indices)
            continue
        nested = []
        for key in ('markers', 'data'):
            value = root.get(key)
            if isinstance(value, list):
                nested.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, dict):
                nested.append(value)
        if nested:
            for item in nested:
                index = _to_int(_first_present([item], ('id', 'marker_id', 'markerId')))
                candidates.append((item, marker_id if index is None else index))
            timing_payloads.extend(nested)
        else:
            candidates.append((root, marker_id))

    frame_idx = _to_int(_first_present(
        timing_payloads, ('marker_frmIdx', 'frame_idx', 'frame', 'frame_id')))
    timestamp = _normalize_timestamp(_first_present(
        timing_payloads, ('marker_fTimestamp', 'timestamp', 'time', 't',
                          f'marker_{marker_id}_timestamp', f'marker_{marker_id}_time')),
        timestamp_unit)
    eligible = []
    total = 0
    for candidate, index in candidates:
        position = _extract_position(candidate, index)
        if position is None:
            continue
        total += 1
        scale = _scale_for_unit(position, position_unit)
        position_m = (position * scale).astype(np.float32)
        if not limits[0] <= float(position_m[0]) <= limits[1]:
            continue
        velocity = _extract_velocity(candidate, index)
        eligible.append((index, position_m,
                         None if velocity is None else (velocity * scale).astype(np.float32)))
    selected = len(eligible) == 1
    info = dict(enabled=True, frame='mocap_raw_m', x_min_m=limits[0], x_max_m=limits[1],
                total_marker_count=total, eligible_marker_count=len(eligible),
                selected_marker_id=eligible[0][0] if selected else None,
                status=('selected' if selected else
                        'no_candidate' if not eligible else 'ambiguous_candidates'),
                frame_idx=frame_idx)
    return MocapPacket(
        position_m=eligible[0][1] if selected else np.zeros(3, dtype=np.float32),
        velocity_m_s=eligible[0][2] if selected else None,
        frame_idx=frame_idx, timestamp_s=timestamp, source_format='json', raw=payload,
        tracked=selected, marker_selection_info=info)


def parse_mocap_packet(data: bytes,
                       marker_id: int = 1,
                       object_name: Optional[str] = None,
                       object_kind: str = 'auto',
                       rigid_body_slot: Optional[int] = None,
                       rigid_body_id: Optional[int] = None,
                       position_unit: str = 'auto',
                       timestamp_unit: str = 's',
                       marker_x_limits_m: Optional[Tuple[float, float]] = None) -> Optional[MocapPacket]:
    """Parse one UDP packet from the mocap system.

    Supported formats:
      - OptiTrack marker:
        {"marker_frmIdx": 1, "marker_fTimestamp": "0:0:1:200",
         "marker_1_px": 0.1, "marker_1_py": 0.2, "marker_1_pz": 0.3}
      - OptiTrack rigid body:
        {"opti_rb_0_px": ..., "opti_rb_0_qw": ..., "opti_rb_1_id": 1, ...}
      - Generic JSON marker lists and legacy binary <d6f>.

    When marker_x_limits_m is set, select the unique scatter point whose X
    lies in this inclusive range in meters, before robot-frame calibration.
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
            marker_payloads = list(
                _iter_marker_payloads(payload, marker_id, object_name))
            if not marker_payloads:
                return None
            root_payload = marker_payloads[0]
        else:
            root_payload = payload
        if not isinstance(root_payload, dict):
            return None

        kind = str(object_kind or 'auto').strip().lower()
        if kind in ('opti_rb', 'rigid_body'):
            if rigid_body_slot is None:
                return None
            return _parse_opti_rb_packet(
                root_payload,
                rigid_body_slot=int(rigid_body_slot),
                rigid_body_id=rigid_body_id,
                position_unit=position_unit,
                timestamp_unit=timestamp_unit,
            )
        if kind == 'auto' and rigid_body_slot is not None:
            packet = _parse_opti_rb_packet(
                root_payload,
                rigid_body_slot=int(rigid_body_slot),
                rigid_body_id=rigid_body_id,
                position_unit=position_unit,
                timestamp_unit=timestamp_unit,
            )
            if packet is not None:
                return packet

        if marker_x_limits_m is not None:
            return _parse_x_filtered_marker_frame(
                payload, marker_id=marker_id,
                limits=_validate_marker_x_limits(marker_x_limits_m),
                position_unit=position_unit, timestamp_unit=timestamp_unit)

        marker_payloads = list(
            _iter_marker_payloads(payload, marker_id, object_name))
        candidate_payloads = [root_payload]
        for marker_payload in marker_payloads:
            if marker_payload is not root_payload:
                candidate_payloads.append(marker_payload)

        pos_payload = None
        pos = None
        for candidate in candidate_payloads:
            pos = _extract_position(candidate, marker_id)
            if pos is not None:
                pos_payload = candidate
                break
        if pos is None:
            return None
        scale = _scale_for_unit(pos, position_unit)
        pos_m = (pos * scale).astype(np.float32)

        vel = None
        velocity_payloads = []
        if pos_payload is not None:
            velocity_payloads.append(pos_payload)
        velocity_payloads.extend(
            p for p in candidate_payloads if p is not pos_payload)
        for candidate in velocity_payloads:
            vel = _extract_velocity(candidate, marker_id)
            if vel is not None:
                break
        vel_m_s = (vel * scale).astype(np.float32) if vel is not None else None

        frame_idx = _to_int(_first_present(
            candidate_payloads,
            ('marker_frmIdx', 'frame_idx', 'frame', 'frame_id')))
        timestamp_value = _first_present(
            candidate_payloads,
            (
                'marker_fTimestamp', 'timestamp', 'time', 't',
                f'marker_{int(marker_id)}_timestamp',
                f'marker_{int(marker_id)}_time',
            ))

        return MocapPacket(
            position_m=pos_m,
            velocity_m_s=vel_m_s,
            frame_idx=frame_idx,
            timestamp_s=_normalize_timestamp(timestamp_value, timestamp_unit),
            source_format='json',
            raw=payload)

    if len(data) >= _BINARY_PACKET_SIZE:
        try:
            values = struct.unpack(
                _BINARY_PACKET_FORMAT, data[:_BINARY_PACKET_SIZE])
        except struct.error:
            return None
        pos = np.array([values[1], values[2], values[3]], dtype=np.float64)
        scale = _scale_for_unit(pos, position_unit)
        vel = np.array([values[4], values[5], values[6]], dtype=np.float64)
        packet = MocapPacket(
            position_m=(pos * scale).astype(np.float32),
            velocity_m_s=(vel * scale).astype(np.float32),
            frame_idx=None,
            timestamp_s=_normalize_timestamp(values[0], timestamp_unit),
            source_format='binary_d6f',
            raw={})
        if marker_x_limits_m is not None:
            limits = _validate_marker_x_limits(marker_x_limits_m)
            eligible = bool(np.all(np.isfinite(packet.position_m))
                            and limits[0] <= float(packet.position_m[0]) <= limits[1])
            packet.tracked = eligible
            packet.marker_selection_info = dict(
                enabled=True, frame='mocap_raw_m', x_min_m=limits[0], x_max_m=limits[1],
                total_marker_count=1, eligible_marker_count=int(eligible), selected_marker_id=None,
                status='selected' if eligible else 'no_candidate', frame_idx=None)
        return packet

    return None


class MocapBallDetector:
    """Receives ball state from motion capture system via UDP.

    JSON packets with position-only markers are accepted; velocity is estimated
    with a time-aware multi-frame difference. The legacy binary <d6f> packet
    remains supported.
    """

    def __init__(self,
                 udp_host: str = '0.0.0.0',
                 udp_port: int = 65501,
                 calibration_path: Optional[str] = None,
                 timeout_s: float = 0.5,
                 marker_id: int = 1,
                 object_kind: str = 'marker',
                 rigid_body_slot: Optional[int] = None,
                 rigid_body_id: Optional[int] = None,
                 position_unit: str = 'auto',
                 timestamp_unit: str = 's',
                 velocity_alpha: float = 1.0,
                 max_velocity_m_s: float = 20.0,
                 velocity_window_frames: int = 3,
                 velocity_window_max_span_s: float = 0.05,
                 receiver_clock: Optional[Callable[[], float]] = None,
                 clock_sync_min_samples: int = 8,
                 clock_sync_min_span_s: float = 0.05,
                 clock_sync_window_s: float = 10.0,
                 marker_x_limits_m: Optional[Tuple[float, float]] = None):
        """Initialize mocap ball detector.

        Args:
            udp_host: UDP address to bind.
            udp_port: UDP port to listen on
            calibration_path: Path to calibration file (4x4 transform matrix).
                            If None, identity transform is used.
            timeout_s: Time after which ball state is invalid, in seconds.
            marker_id: Marker index used by marker_<id>_px/py/pz JSON fields.
            marker_x_limits_m: Optional inclusive X range in raw mocap meters;
                selects the unique in-range point instead of a fixed index.
            object_kind: marker / opti_rb / auto.
            rigid_body_slot: OptiTrack slot Y in opti_rb_Y_* fields.
            rigid_body_id: OptiTrack id YY in opti_rb_YY_id.
            position_unit: auto/m/mm/cm for incoming positions.
            timestamp_unit: s/ms/us/ns/auto for incoming timestamps.
            velocity_alpha: EMA smoothing for finite-difference velocity.
                The default 1.0 disables EMA, so there is no additional
                low-pass delay after the multi-frame difference.
            max_velocity_m_s: Reject finite-difference velocity above this.
            velocity_window_frames: Number of mocap frame intervals used for
                each velocity difference. 3 at 120 Hz uses about 25 ms of
                position history; positions themselves are never averaged.
            velocity_window_max_span_s: Reject a velocity window whose source
                time span exceeds this value (for example after a long dropout).
            receiver_clock: Consumer clock in seconds; ROS nodes pass ROS time.
            clock_sync_min_samples: Unique observations required for clock alignment.
            clock_sync_min_span_s: Minimum source and receipt time span for alignment.
            clock_sync_window_s: Recent window used for the rolling offset estimate.
        """
        self.udp_host = str(udp_host)
        self.udp_port = udp_port
        self.timeout_s = timeout_s
        self.marker_id = int(marker_id)
        self.object_kind = str(object_kind or 'marker')
        self.rigid_body_slot = (
            None if rigid_body_slot is None else int(rigid_body_slot))
        self.rigid_body_id = (
            None if rigid_body_id is None else int(rigid_body_id))
        self.position_unit = str(position_unit or 'auto')
        self.timestamp_unit = str(timestamp_unit or 's')
        self.marker_x_limits_m = _validate_marker_x_limits(marker_x_limits_m)
        self.velocity_alpha = min(max(float(velocity_alpha), 0.0), 1.0)
        self.max_velocity_m_s = float(max_velocity_m_s)
        self.velocity_window_frames = max(1, int(velocity_window_frames))
        self.velocity_window_max_span_s = max(
            float(velocity_window_max_span_s), 1e-4)
        self._min_velocity_dt_s = 1e-4
        self._max_source_timestamp_step_s = 0.25
        self._receiver_clock = receiver_clock or time.time
        self.clock_sync = MocapClockSynchronizer(
            min_samples=clock_sync_min_samples, min_span_s=clock_sync_min_span_s,
            window_s=clock_sync_window_s)

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
        self._latest_arrival_monotonic_s = 0.0
        self._latest_measurement_timestamp_s = None
        self._latest_receiver_timestamp_s = None
        self._latest_unwrapped_source_timestamp_s = None
        self._latest_source_timestamp_s = None
        self._latest_frame_idx = None
        self._latest_format = None
        self._latest_addr = None
        self._latest_raw = {}
        self._latest_observation_valid = False
        self._last_marker_selection = {}
        self._packet_count = 0
        self._parse_error_count = 0
        self._velocity_reject_count = 0
        self._velocity_window_reject_count = 0
        self._velocity_time_source = None

        self._prev_vel_mocap_m_s = None
        self._prev_sample_time_s = None
        self._prev_sample_time_source = None
        self._prev_arrival_time_s = None
        self._velocity_history = deque(maxlen=self.velocity_window_frames)

        # UDP receiver thread
        self._running = False
        self._thread = None
        self._socket = None
        self._last_start_error = None

    def load_calibration(self, path: str) -> None:
        """Load calibration transform from file.

        File format: 4x4 homogeneous transform matrix (text, space-separated)
        """
        try:
            T = np.loadtxt(path, dtype=np.float64)
            if T.shape != (4, 4):
                raise ValueError(
                    f"Calibration matrix must be 4x4, got {T.shape}")
            self.T_robot_mocap = T
            print(f"[MocapBallDetector] Loaded calibration from {path}")
            print(f"  T_robot_mocap:\n{self.T_robot_mocap}")
        except Exception as e:
            print(f"[MocapBallDetector] Failed to load calibration: {e}")
            print("  Using identity transform")

    def save_calibration(self, path: str) -> None:
        """Save calibration transform to file."""
        np.savetxt(path, self.T_robot_mocap, fmt='%.6f')
        print(f"[MocapBallDetector] Saved calibration to {path}")

    def start(self) -> bool:
        """Start UDP receiver thread."""
        if self._running:
            return True

        # Open UDP socket
        try:
            self._last_start_error = None
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((self.udp_host, self.udp_port))
            self._socket.settimeout(1.0)  # 1s timeout for blocking recv
            print(
                f"[MocapBallDetector] Listening on "
                f"{self.udp_host}:{self.udp_port} "
                f"(marker={self.marker_id}, unit={self.position_unit})")
        except Exception as e:
            self._last_start_error = e
            print(f"[MocapBallDetector] Failed to open UDP socket: {e}")
            if self._socket is not None:
                self._socket.close()
                self._socket = None
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._receiver_loop, daemon=True)
        self._thread.start()
        print("[MocapBallDetector] Receiver thread started")
        return True

    def stop(self) -> None:
        """Stop UDP receiver thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        print("[MocapBallDetector] Stopped")

    def _receiver_loop(self) -> None:
        """Background thread: receive UDP packets and update ball state."""
        while self._running:
            try:
                # Receive UDP packet (blocking with timeout)
                data, addr = self._socket.recvfrom(65535)
                arrival_monotonic_s = time.monotonic()
                arrival_clock_s = self._receiver_clock()
                arrival_time_s = time.time()

                packet = parse_mocap_packet(
                    data,
                    marker_id=self.marker_id,
                    object_kind=self.object_kind,
                    rigid_body_slot=self.rigid_body_slot,
                    rigid_body_id=self.rigid_body_id,
                    position_unit=self.position_unit,
                    timestamp_unit=self.timestamp_unit,
                    marker_x_limits_m=self.marker_x_limits_m)
                if packet is None:
                    with self._lock:
                        self._parse_error_count += 1
                        if self.marker_x_limits_m is not None:
                            self._latest_observation_valid = False
                    continue

                self.process_packet(
                    packet, arrival_clock_s=arrival_clock_s,
                    arrival_monotonic_s=arrival_monotonic_s,
                    arrival_wall_s=arrival_time_s, addr=addr)

            except socket.timeout:
                # No packet received within timeout - this is fine
                continue
            except Exception as e:
                if self._running:
                    print(f"[MocapBallDetector] Receiver error: {e}")
                continue

    def process_packet(self, packet: MocapPacket, *, arrival_clock_s: float,
                       arrival_monotonic_s: float, arrival_wall_s: float,
                       addr=None) -> bool:
        """Accept one observation atomically; late/repeated frames stay held."""
        with self._lock:
            if self.marker_x_limits_m is not None and packet.marker_selection_info is None:
                if packet.source_format == 'json':
                    packet = _parse_x_filtered_marker_frame(
                        packet.raw, marker_id=self.marker_id, limits=self.marker_x_limits_m,
                        position_unit=self.position_unit, timestamp_unit=self.timestamp_unit)
                else:
                    eligible = bool(np.all(np.isfinite(packet.position_m))
                                    and self.marker_x_limits_m[0] <= float(packet.position_m[0])
                                    <= self.marker_x_limits_m[1])
                    packet.marker_selection_info = dict(
                        enabled=True, frame='mocap_raw_m',
                        x_min_m=self.marker_x_limits_m[0], x_max_m=self.marker_x_limits_m[1],
                        total_marker_count=1, eligible_marker_count=int(eligible),
                        selected_marker_id=None,
                        status='selected' if eligible else 'no_candidate', frame_idx=packet.frame_idx)
                    packet.tracked = eligible
            previous_epoch = self.clock_sync.epoch
            timing = self.clock_sync.update(
                packet.timestamp_s, arrival_clock_s, arrival_monotonic_s,
                frame_idx=packet.frame_idx)
            if not timing.accepted:
                return False
            if self.clock_sync.epoch != previous_epoch:
                self._velocity_history.clear()
                self._prev_vel_mocap_m_s = None
                self._prev_sample_time_s = None
                self._prev_sample_time_source = None
            self._last_marker_selection = dict(packet.marker_selection_info or {})
            if packet.tracked is False:
                self._latest_observation_valid = False
                self._velocity_history.clear()
                self._prev_vel_mocap_m_s = None
                self._prev_sample_time_s = None
                self._prev_sample_time_source = None
                return False
            vel_mocap = packet.velocity_m_s
            if vel_mocap is None:
                vel_mocap = self._estimate_velocity_locked(
                    packet.position_m, timing.source_unwrapped_s,
                    arrival_monotonic_s)
            else:
                self._remember_velocity_sample_locked(
                    packet.position_m, vel_mocap, timing.source_unwrapped_s,
                    arrival_monotonic_s)
            self._latest_pos_m = self.transform_position(packet.position_m)
            self._latest_vel_m_s = self.transform_velocity(vel_mocap)
            self._latest_timestamp = arrival_wall_s
            self._latest_arrival_monotonic_s = arrival_monotonic_s
            self._latest_receiver_timestamp_s = arrival_clock_s
            self._latest_measurement_timestamp_s = timing.receiver_timestamp_s
            self._latest_source_timestamp_s = packet.timestamp_s
            self._latest_unwrapped_source_timestamp_s = timing.source_unwrapped_s
            self._latest_frame_idx = packet.frame_idx
            self._latest_format = packet.source_format
            self._latest_addr = addr
            self._latest_raw = packet.raw
            self._latest_observation_valid = True
            self._packet_count += 1
            return True

    def _sample_time_locked(self, source_timestamp_s,
                            arrival_time_s) -> Tuple[float, str]:
        """Choose a monotonic timebase without mixing clock domains.

        Prefer OptiTrack's source timestamp.  If it is missing, duplicated,
        or discontinuous, temporarily use UDP-arrival time.  Switching clock
        domains clears the velocity history, avoiding an invalid difference
        between timestamps with unrelated origins.
        """
        source_time_s = _to_float(source_timestamp_s)
        if source_time_s is not None:
            if (self._prev_sample_time_s is None
                    or self._prev_sample_time_source != 'source'):
                return source_time_s, 'source'
            dt = source_time_s - self._prev_sample_time_s
            if self._min_velocity_dt_s <= dt <= self._max_source_timestamp_step_s:
                return source_time_s, 'source'
        return float(arrival_time_s), 'arrival'

    def _append_velocity_history_locked(self, pos_mocap_m, sample_time_s,
                                        time_source: str) -> None:
        if (self._velocity_history
                and self._velocity_history[-1][2] != time_source):
            self._velocity_history.clear()
        self._velocity_history.append((
            np.asarray(pos_mocap_m, dtype=np.float32).reshape(3).copy(),
            float(sample_time_s),
            str(time_source),
        ))

    def _hold_previous_velocity_locked(self) -> np.ndarray:
        if self._prev_vel_mocap_m_s is None:
            return np.zeros(3, dtype=np.float32)
        return self._prev_vel_mocap_m_s.copy()

    def _remember_velocity_sample_locked(self, pos_mocap_m, vel_mocap_m_s,
                                         source_timestamp_s,
                                         arrival_time_s) -> None:
        sample_time_s, time_source = self._sample_time_locked(
            source_timestamp_s, arrival_time_s)
        self._append_velocity_history_locked(
            pos_mocap_m, sample_time_s, time_source)
        self._prev_vel_mocap_m_s = np.asarray(
            vel_mocap_m_s, dtype=np.float32).copy()
        self._prev_sample_time_s = sample_time_s
        self._prev_sample_time_source = time_source
        self._prev_arrival_time_s = float(arrival_time_s)
        self._velocity_time_source = time_source

    def _estimate_velocity_locked(self, pos_mocap_m, source_timestamp_s,
                                  arrival_time_s) -> np.ndarray:
        sample_time_s, time_source = self._sample_time_locked(
            source_timestamp_s, arrival_time_s)
        pos = np.asarray(pos_mocap_m, dtype=np.float32).reshape(3)

        if (self._velocity_history
                and self._velocity_history[-1][2] != time_source):
            self._velocity_history.clear()

        if len(self._velocity_history) < self.velocity_window_frames:
            vel = self._hold_previous_velocity_locked()
        else:
            reference_pos, reference_time_s, _ = self._velocity_history[0]
            dt = sample_time_s - reference_time_s
            if (dt < self._min_velocity_dt_s
                    or dt > self.velocity_window_max_span_s):
                self._velocity_window_reject_count += 1
                vel = self._hold_previous_velocity_locked()
                # Do not bridge a long gap with stale pre-dropout samples.
                self._velocity_history.clear()
            else:
                raw_vel = (pos - reference_pos) / float(dt)
                speed = float(np.linalg.norm(raw_vel))
                if (self.max_velocity_m_s > 0.0
                        and speed > self.max_velocity_m_s):
                    self._velocity_reject_count += 1
                    vel = self._hold_previous_velocity_locked()
                elif self._prev_vel_mocap_m_s is not None:
                    alpha = self.velocity_alpha
                    vel = (alpha * raw_vel
                           + (1.0 - alpha) * self._prev_vel_mocap_m_s)
                else:
                    vel = raw_vel
                vel = np.asarray(vel, dtype=np.float32)

        self._append_velocity_history_locked(pos, sample_time_s, time_source)
        self._prev_vel_mocap_m_s = vel.copy()
        self._prev_sample_time_s = sample_time_s
        self._prev_sample_time_source = time_source
        self._prev_arrival_time_s = float(arrival_time_s)
        self._velocity_time_source = time_source
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
        pos, vel, valid, _ = self.get_ball_state_with_diagnostics()
        return pos, vel, valid

    def get_ball_state_with_diagnostics(self):
        """Return position, velocity, tracking validity and matching timestamps."""
        with self._lock:
            diagnostics = self._diagnostics_locked()
            return (self._latest_pos_m.copy(), self._latest_vel_m_s.copy(),
                    diagnostics['valid'], diagnostics)

    def get_diagnostics(self) -> dict:
        """Get diagnostic info."""
        with self._lock:
            return self._diagnostics_locked()

    def _diagnostics_locked(self):
        age = time.monotonic() - self._latest_arrival_monotonic_s
        valid = bool(self._latest_observation_valid and self._packet_count > 0
                     and 0 <= age < self.timeout_s)
        clock_sync = self.clock_sync.diagnostics()
        measurement_age = None
        if self._latest_measurement_timestamp_s is not None:
            now_receiver_s = self._receiver_clock()
            measurement_age = now_receiver_s - self._latest_measurement_timestamp_s
            receiver_elapsed = now_receiver_s - self._latest_receiver_timestamp_s
            if abs(receiver_elapsed - age) > self.clock_sync.reset_jump_s:
                clock_sync.update(ready=False, status='receiver_clock_changed')
        return {
            'packet_count': self._packet_count,
            'age_s': age,
            'valid': valid,
            'synchronized_valid': bool(
                valid and clock_sync['ready'] and measurement_age is not None
                and 0 <= measurement_age < self.timeout_s),
            'measurement_age_s': measurement_age,
            'measurement_timestamp_s': self._latest_measurement_timestamp_s,
            'receiver_timestamp_s': self._latest_receiver_timestamp_s,
            'unwrapped_source_timestamp_s': self._latest_unwrapped_source_timestamp_s,
            'clock_sync': clock_sync,
            'parse_error_count': self._parse_error_count,
            'velocity_reject_count': self._velocity_reject_count,
            'velocity_window_reject_count': self._velocity_window_reject_count,
            'velocity_window_frames': self.velocity_window_frames,
            'velocity_window_max_span_s': self.velocity_window_max_span_s,
            'velocity_alpha': self.velocity_alpha,
            'velocity_time_source': self._velocity_time_source,
            'frame_idx': self._latest_frame_idx,
            'source_timestamp_s': self._latest_source_timestamp_s,
            'arrival_time_s': self._latest_timestamp,
            'source_format': self._latest_format,
            'addr': self._latest_addr,
            'raw': dict(self._latest_raw) if isinstance(
                self._latest_raw, dict) else self._latest_raw,
            'marker_selection': dict(self._last_marker_selection),
        }
