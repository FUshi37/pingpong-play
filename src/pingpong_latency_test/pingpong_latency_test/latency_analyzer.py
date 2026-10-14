#!/usr/bin/env python3
"""
Latency Analyzer

Analyzes flash detection logs from the robot side, and optionally aligns them
with the laptop's own white-flash timestamps to compute the full
"exposure + transport + detection" latency.

Usage:
    # Robot software-side breakdown only:
    ros2 run pingpong_latency_test latency_analyzer flash_events.json

    # End-to-end, aligning robot detections with laptop white timestamps:
    ros2 run pingpong_latency_test latency_analyzer flash_events.json \
        --laptop laptop_white_times.json

The laptop file must be a JSON list of white-flash epoch-second timestamps,
e.g.  [1782290527.123, 1782290529.456, ...]
Both clocks must share a common time base (NTP / chrony / PTP); see README.
"""

import json
import sys
import csv
import argparse
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False


def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def load_laptop_times(filepath, column='wall_time_s'):
    """Load laptop white-flash epoch-second timestamps from JSON list or CSV.

    CSV must have a header; uses the given column (default 'wall_time_s').
    """
    if filepath.lower().endswith('.csv'):
        times = []
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            if column not in reader.fieldnames:
                raise ValueError(
                    f'CSV has no column "{column}". Available: {reader.fieldnames}')
            for row in reader:
                if row[column]:
                    times.append(float(row[column]))
        return times
    return load_json(filepath)


def stat_line(name, values, unit='ms'):
    v = np.array(values, dtype=float)
    return (f'  {name:<28} mean={v.mean():7.2f}  median={np.median(v):7.2f}  '
            f'std={v.std():6.2f}  min={v.min():7.2f}  max={v.max():7.2f}  '
            f'p95={np.percentile(v, 95):7.2f} {unit}')


def print_software_breakdown(events):
    n = len(events)
    print(f'\n{"="*78}')
    print(f'ROBOT SOFTWARE-SIDE LATENCY ({n} white flashes)')
    print(f'{"="*78}')
    print(stat_line('transport (stamp->recv)', [e['transport_latency_ms'] for e in events]))
    print(stat_line('decode', [e['decode_latency_ms'] for e in events]))
    print(stat_line('detection (brightness)', [e['detection_latency_ms'] for e in events]))
    print()


def align_with_laptop(events, laptop_times, max_gap_s, clock_offset=0.0):
    """For each robot detection, find the nearest preceding laptop white time.

    clock_offset = laptop_clock - robot_clock (seconds). Laptop times are shifted
    onto the robot clock by subtracting it before comparing with detect_time.
    """
    laptop = np.sort(np.array(laptop_times, dtype=float) - clock_offset)
    pairs = []
    unmatched = 0
    for e in events:
        td = e['detect_time']
        # nearest laptop white time at or before detection
        idx = np.searchsorted(laptop, td, side='right') - 1
        if idx < 0:
            unmatched += 1
            continue
        gap = td - laptop[idx]
        if gap < 0 or gap > max_gap_s:
            unmatched += 1
            continue
        pairs.append({
            'flash_id': e['flash_id'],
            'laptop_white': float(laptop[idx]),
            'detect_time': td,
            'end_to_end_ms': gap * 1000.0,
            'transport_latency_ms': e['transport_latency_ms'],
            'decode_latency_ms': e['decode_latency_ms'],
            'detection_latency_ms': e['detection_latency_ms'],
        })
    return pairs, unmatched


def print_end_to_end(pairs, unmatched):
    print(f'\n{"="*78}')
    print(f'END-TO-END LATENCY  (laptop white -> robot detection)')
    print(f'  matched={len(pairs)}  unmatched={unmatched}')
    print(f'{"="*78}')
    if not pairs:
        print('  No matched pairs. Check clock sync and --max-gap.')
        return
    e2e = [p['end_to_end_ms'] for p in pairs]
    print(stat_line('END-TO-END', e2e))
    print()
    # exposure+transport-on-wire estimate = e2e - (transport + decode + detection)
    sw = [p['transport_latency_ms'] + p['decode_latency_ms'] + p['detection_latency_ms']
          for p in pairs]
    exposure = [a - b for a, b in zip(e2e, sw)]
    print(stat_line('  - sum software side', sw))
    print(stat_line('  = exposure + camera/driver', exposure))
    print('\n  (exposure+camera/driver = end-to-end minus measured software stages)')
    print()


def pair_single_clock(events, udp_events, net_delay_s, max_gap_s):
    """Pair each detection with its UDP trigger on the ROBOT clock.

    Both detect_time and arrival_time are robot-clock epoch seconds, so
        latency = detect_time - arrival_time - net_one_way_delay
    requires no clock synchronization.

    Strategy: prefer matching by seq when the detection order lines up; otherwise
    fall back to nearest preceding UDP arrival. We match each detection to the
    most recent UDP arrival at or before detect_time within max_gap_s.
    """
    arrivals = sorted(udp_events, key=lambda u: u['arrival_time'])
    arr_t = np.array([u['arrival_time'] for u in arrivals], dtype=float)
    pairs = []
    unmatched = 0
    for e in events:
        td = e['detect_time']
        idx = np.searchsorted(arr_t, td, side='right') - 1
        if idx < 0:
            unmatched += 1
            continue
        gap = td - arr_t[idx]
        if gap < 0 or gap > max_gap_s:
            unmatched += 1
            continue
        latency_ms = (gap - net_delay_s) * 1000.0
        pairs.append({
            'flash_id': e['flash_id'],
            'udp_seq': arrivals[idx]['seq'],
            'arrival_time': float(arr_t[idx]),
            'detect_time': td,
            'end_to_end_ms': latency_ms,
            'raw_gap_ms': gap * 1000.0,
            'transport_latency_ms': e['transport_latency_ms'],
            'decode_latency_ms': e['decode_latency_ms'],
            'detection_latency_ms': e['detection_latency_ms'],
        })
    return pairs, unmatched


def print_single_clock(pairs, unmatched, net_delay_s):
    print(f'\n{"="*78}')
    print(f'END-TO-END LATENCY  (single robot clock: detect - udp_arrival - net)')
    print(f'  net one-way delay subtracted = {net_delay_s*1000:.2f} ms')
    print(f'  matched={len(pairs)}  unmatched={unmatched}')
    print(f'{"="*78}')
    if not pairs:
        print('  No matched pairs. Was the laptop sending UDP to the right ip:port?')
        print('  Check flash_detector logged "UDP trigger: first packet ...".')
        return
    e2e = [p['end_to_end_ms'] for p in pairs]
    print(stat_line('END-TO-END (screen->state)', e2e))
    print()
    sw = [p['transport_latency_ms'] + p['decode_latency_ms'] + p['detection_latency_ms']
          for p in pairs]
    exposure = [a - b for a, b in zip(e2e, sw)]
    print(stat_line('  - sum software side', sw))
    print(stat_line('  = screen + exposure + cam/driver', exposure))
    print('\n  (the residual includes the LCD screen flip->photons delay, which is')
    print('   absent in the real ball scenario -- treat it as an upper bound)')
    print()


def make_plots(events, pairs, output_file):
    if not HAVE_PLT:
        print('matplotlib not available, skipping plots.')
        return
    ncols = 2 if pairs else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5), squeeze=False)

    ax = axes[0][0]
    ids = [e['flash_id'] for e in events]
    ax.plot(ids, [e['transport_latency_ms'] for e in events], 'o-', label='transport')
    ax.plot(ids, [e['decode_latency_ms'] for e in events], 's-', label='decode')
    ax.plot(ids, [e['detection_latency_ms'] for e in events], '^-', label='detection')
    ax.set_xlabel('flash id'); ax.set_ylabel('ms'); ax.set_title('Software-side latency')
    ax.legend(); ax.grid(alpha=0.3)

    if pairs:
        ax = axes[0][1]
        e2e = [p['end_to_end_ms'] for p in pairs]
        ax.hist(e2e, bins=min(20, len(e2e)), color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(e2e), color='r', linestyle='--', label=f'mean {np.mean(e2e):.1f}ms')
        ax.axvline(np.median(e2e), color='g', linestyle='--', label=f'median {np.median(e2e):.1f}ms')
        ax.set_xlabel('end-to-end (ms)'); ax.set_ylabel('count')
        ax.set_title('End-to-end distribution'); ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=120)
    print(f'Saved plot to {output_file}')


def main(args=None):
    parser = argparse.ArgumentParser(description='Analyze flash detection latency logs')
    parser.add_argument('robot_log', help='robot-side flash_events.json')
    parser.add_argument('--laptop', help='laptop white-flash timestamps: JSON list or CSV')
    parser.add_argument('--laptop-column', default='wall_time_s',
                        help='CSV column holding epoch-second white timestamps (default wall_time_s)')
    parser.add_argument('--max-gap', type=float, default=0.5,
                        help='max laptop->detection gap to count as a match (seconds, default 0.5)')
    parser.add_argument('--clock-offset', type=float, default=0.0,
                        help='laptop_clock - robot_clock (seconds), from clock_sync. '
                             'Subtracted from laptop times before alignment.')
    parser.add_argument('--udp', help='robot-side UDP arrivals JSON (flash_events_udp.json). '
                                      'Enables single-clock latency (no clock sync needed).')
    parser.add_argument('--net-delay', type=float, default=0.001,
                        help='one-way network delay in seconds to subtract in --udp mode '
                             '(default 0.001; use min-RTT/2 from clock_sync)')
    parser.add_argument('--plot', default=None, help='output PNG path (default: <robot_log>_analysis.png)')
    ns = parser.parse_args(args if args is not None else sys.argv[1:])

    events = load_json(ns.robot_log)
    if not events:
        print('No flash events in robot log.')
        return
    events = sorted(events, key=lambda e: e['flash_id'])

    print_software_breakdown(events)

    pairs = []
    if ns.udp:
        # Preferred: single-clock method, no clock sync.
        udp_events = load_json(ns.udp)
        pairs, unmatched = pair_single_clock(events, udp_events, ns.net_delay, ns.max_gap)
        print_single_clock(pairs, unmatched, ns.net_delay)
    elif ns.laptop:
        # Legacy: cross-clock subtraction with measured offset.
        laptop_times = load_laptop_times(ns.laptop, ns.laptop_column)
        pairs, unmatched = align_with_laptop(events, laptop_times, ns.max_gap, ns.clock_offset)
        print_end_to_end(pairs, unmatched)

    out = ns.plot or (ns.robot_log.rsplit('.', 1)[0] + '_analysis.png')
    make_plots(events, pairs, out)


if __name__ == '__main__':
    main()
