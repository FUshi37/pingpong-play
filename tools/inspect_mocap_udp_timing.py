#!/usr/bin/env python3
"""Receive mocap timing evidence without ROS or robot command publishers."""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import selectors
import socket
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/pingpong_moz1'))
from pingpong_controller.mocap_ball_detector import (
    parse_mocap_packet, _normalize_timestamp, _to_int,
)
from pingpong_controller.mocap_input import MocapBallInput


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ports', type=int, nargs='+', default=[65501])
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--duration-s', type=float, default=15.0)
    parser.add_argument('--marker-id', type=int, default=1)
    parser.add_argument('--calibration-path', default=str(
        Path(__file__).resolve().parents[1] / 'mocap_calibration.txt'))
    parser.add_argument('--position-unit', default='auto')
    parser.add_argument('--timestamp-unit', default='s')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.duration_s <= 60:
        parser.error('--duration-s must be in (0, 60]')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    bind_errors = {}
    adapters = {port: MocapBallInput(
        udp_port=port, marker_id=args.marker_id, calibration_path=args.calibration_path,
        position_unit=args.position_unit, timestamp_unit=args.timestamp_unit)
        for port in args.ports}
    synchronizers = {port: adapter.detector.clock_sync
                     for port, adapter in adapters.items()}
    with selectors.DefaultSelector() as selector:
        for port in dict.fromkeys(args.ports):
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                # No SO_REUSEADDR: never divert packets from an active receiver.
                sock.bind((args.host, port))
                sock.setblocking(False)
                selector.register(sock, selectors.EVENT_READ, port)
                print(f'Listening on {args.host}:{port}', flush=True)
            except OSError as exc:
                bind_errors[str(port)] = str(exc)
                sock.close()
                print(f'Cannot listen on {port}: {exc}', flush=True)
        started = time.monotonic()
        try:
            with (args.output_dir / 'live_packets.jsonl').open('w') as output:
                while selector.get_map() and time.monotonic() - started < args.duration_s:
                    for key, _ in selector.select(timeout=.2):
                        data, addr = key.fileobj.recvfrom(65535)
                        arrival_monotonic_s = time.monotonic()
                        arrival_wall_s = time.time()
                        try:
                            payload = json.loads(data)
                        except (ValueError, UnicodeDecodeError):
                            payload = None
                        packet = parse_mocap_packet(
                            data, marker_id=args.marker_id, object_kind='marker',
                            position_unit=args.position_unit,
                            timestamp_unit=args.timestamp_unit)
                        row = dict(port=key.data, addr=list(addr), size_bytes=len(data),
                                   arrival_wall_s=arrival_wall_s,
                                   arrival_monotonic_s=arrival_monotonic_s,
                                   payload=payload)
                        if payload is None:
                            row['data_base64'] = base64.b64encode(data).decode('ascii')
                        if packet is not None:
                            row.update(source_timestamp_s=packet.timestamp_s,
                                       frame_idx=packet.frame_idx,
                                       position_m=packet.position_m.tolist())
                            adapter = adapters[key.data]
                            accepted = adapter.detector.process_packet(
                                packet, arrival_clock_s=arrival_wall_s,
                                arrival_monotonic_s=arrival_monotonic_s,
                                arrival_wall_s=arrival_wall_s, addr=addr)
                            pos, vel, valid, feedback = adapter.get_control_state()
                            row.update(accepted=accepted, control_valid=valid,
                                measurement_timestamp_s=feedback['measurementTimestampS'],
                                measurement_age_s=feedback['measurementAgeS'],
                                waist03_position_m=feedback['waist03Ball'],
                                control_position_m=pos.tolist() if valid else None,
                                control_velocity_m_s=vel.tolist() if valid else None)
                        elif isinstance(payload, dict):
                            raw_stamp = next((payload[key] for key in
                                ('marker_fTimestamp', 'timestamp', 'time', 't')
                                if key in payload), None)
                            row['source_timestamp_s'] = _normalize_timestamp(
                                raw_stamp, args.timestamp_unit)
                            row['frame_idx'] = _to_int(payload.get('marker_frmIdx'))
                        if packet is None:
                            timing = synchronizers[key.data].update(
                                row.get('source_timestamp_s'), arrival_wall_s,
                                arrival_monotonic_s, frame_idx=row.get('frame_idx'))
                            row['measurement_timestamp_s'] = timing.receiver_timestamp_s
                        row['clock_sync'] = synchronizers[key.data].diagnostics()
                        output.write(json.dumps(row, ensure_ascii=False) + '\n')
                        rows.append(row)
        finally:
            for key in list(selector.get_map().values()):
                selector.unregister(key.fileobj)
                key.fileobj.close()
    summary = dict(duration_s=time.monotonic() - started,
                   packet_count=len(rows), bind_errors=bind_errors, ports={})
    for port in args.ports:
        samples = [row for row in rows if row['port'] == port]
        parsed = [row for row in samples if row.get('source_timestamp_s') is not None]
        info = dict(packet_count=len(samples), source_timestamp_count=len(parsed))
        info['ball_observation_count'] = sum('position_m' in row for row in samples)
        info['synchronized_packet_count'] = sum(
            row['clock_sync']['ready'] for row in samples)
        info['clock_sync'] = synchronizers[port].diagnostics()
        controlled = [row for row in samples if row.get('control_valid')]
        info['control_valid_count'] = len(controlled)
        if controlled:
            info['control_position_median_m'] = np.median(
                [row['control_position_m'] for row in controlled], axis=0).tolist()
            info['observation_age_p50_ms'] = float(np.percentile(
                [row['measurement_age_s'] for row in controlled], 50)*1000)
            info['observation_age_p95_ms'] = float(np.percentile(
                [row['measurement_age_s'] for row in controlled], 95)*1000)
        if samples:
            info['max_packet_size_bytes'] = max(row['size_bytes'] for row in samples)
            info['example_payload'] = samples[0]['payload']
        if len(parsed) >= 2:
            source = np.array([row['source_timestamp_s'] for row in parsed])
            arrival = np.array([row['arrival_monotonic_s'] for row in parsed])
            ds, da = np.diff(source), np.diff(arrival)
            info.update(source_first_s=float(source[0]), source_last_s=float(source[-1]),
                        source_step_median_s=float(np.median(ds)),
                        arrival_step_median_s=float(np.median(da)),
                        repeated_timestamps=int(np.sum(ds == 0)),
                        backward_timestamps=int(np.sum(ds < 0)),
                        relative_arrival_jitter_p95_s=float(np.percentile(da-ds, 95)))
        summary['ports'][str(port)] = info
    (args.output_dir / 'live_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if rows else 1


if __name__ == '__main__':
    raise SystemExit(main())
