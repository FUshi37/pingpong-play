#!/usr/bin/env python3
"""
Clock Offset Measurement (NTP-style, plain UDP)

Measures the offset between two machines' system clocks (Python time.time(),
i.e. the same epoch clock that flash_detector and the laptop CSV use), so the
cross-machine end-to-end latency can be corrected.

This does NOT require ntp/chrony. It does a four-timestamp exchange:
    T0 = client send time      (client clock)
    T1 = server recv time      (server clock)
    T2 = server send time      (server clock)
    T3 = client recv time      (client clock)
    offset = ((T1 - T0) + (T2 - T3)) / 2     # server_clock - client_clock
    rtt    = (T3 - T0) - (T2 - T1)

Over many probes the sample with the SMALLEST rtt has the least queuing /
asymmetry, so its offset estimate is the most accurate -- this is the standard
trick that makes the method robust even on jittery WiFi.

Sign convention: offset = server_clock - client_clock.
  To convert a server-clock timestamp to the client clock:  t_client = t_server - offset
  To convert a client-clock timestamp to the server clock:  t_server = t_client + offset

Usage:
    # On machine A (e.g. the laptop), start the responder:
    python3 clock_sync.py --server --port 5005

    # On machine B (e.g. the robot), probe it:
    python3 clock_sync.py --client <A_ip> --port 5005 --count 500

The client prints offset (B->A) and RTT statistics, and the recommended
--clock-offset value to feed into latency_analyzer.
"""

import socket
import struct
import time
import argparse
import sys

PACKET_FMT = '!qdd'   # seq(int64), t1(double), t2(double)
PROBE_FMT = '!qd'     # seq(int64), t0(double)


def guess_local_ips():
    ips = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127."):
                ips.add(ip)
    except OSError:
        pass

    try:
        route_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        route_sock.connect(("8.8.8.8", 80))
        ip = route_sock.getsockname()[0]
        if not ip.startswith("127."):
            ips.add(ip)
    except OSError:
        pass
    finally:
        try:
            route_sock.close()
        except Exception:
            pass

    return sorted(ips)


def run_server(port, bind, verbose):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind, port))
    sock.settimeout(0.5)   # poll so Ctrl-C works (esp. on Windows where a
                           # blocking recvfrom does not get interrupted)
    print(f'[server] listening on udp/{bind}:{port}, Ctrl-C to stop', flush=True)
    local_ips = guess_local_ips()
    if local_ips:
        print(f'[server] candidate laptop IPv4 address(es): {", ".join(local_ips)}', flush=True)
        print('[server] robot must probe one of these reachable laptop IPs, not localhost/127.0.0.1',
              flush=True)
    expected_probe_size = struct.calcsize(PROBE_FMT)
    count = 0
    try:
        while True:
            try:
                data, addr = sock.recvfrom(64)
            except socket.timeout:
                continue
            if len(data) != expected_probe_size:
                print(f'[server] ignored malformed packet from {addr[0]}:{addr[1]} '
                      f'({len(data)} bytes, expected {expected_probe_size})',
                      flush=True)
                continue
            t1 = time.time()                      # recv time (server clock)
            seq, _t0 = struct.unpack(PROBE_FMT, data)
            t2 = time.time()                      # send time (server clock)
            sock.sendto(struct.pack(PACKET_FMT, seq, t1, t2), addr)
            count += 1
            if verbose:
                print(f'[server] probe seq={seq} from {addr[0]}:{addr[1]}', flush=True)
            if count == 1:
                print(f'[server] first probe received from {addr[0]} -- connectivity OK',
                      flush=True)
            if count % 100 == 0:
                print(f'[server] answered {count} probes (last from {addr[0]})', flush=True)
    except KeyboardInterrupt:
        print(f'\n[server] stopped after {count} probes', flush=True)
    finally:
        sock.close()


def run_client(host, port, count, interval):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(0.5)
    samples = []   # (rtt, offset)
    sent = 0
    lost = 0
    print(f'[client] probing {host}:{port} with {count} packets ...')
    try:
        for seq in range(count):
            t0 = time.time()
            sock.sendto(struct.pack(PROBE_FMT, seq, t0), (host, port))
            sent += 1
            try:
                data, _ = sock.recvfrom(64)
                t3 = time.time()
            except socket.timeout:
                lost += 1
                # Fail fast: if nothing comes back in the first 10 probes, the
                # path is almost certainly blocked -- don't wait out all 500.
                if sent >= 10 and not samples:
                    print(f'\n[client] {sent} probes sent, 0 replies. UDP path to '
                          f'{host}:{port} is blocked.')
                    print('  Check: server running? firewall allows UDP '
                          f'{port}? correct IP (not localhost)?')
                    sock.close()
                    sys.exit(1)
                continue
            rseq, t1, t2 = struct.unpack(PACKET_FMT, data)
            if rseq != seq:
                continue
            rtt = (t3 - t0) - (t2 - t1)
            offset = ((t1 - t0) + (t2 - t3)) / 2.0
            samples.append((rtt, offset))
            if len(samples) == 1:
                print('[client] first reply received -- connectivity OK, collecting ...')
            if sent % 100 == 0:
                print(f'[client] {sent}/{count} sent, {len(samples)} replies, {lost} lost')
            if interval > 0:
                time.sleep(interval)
    except KeyboardInterrupt:
        print('\n[client] interrupted, analyzing what was collected ...')
    finally:
        sock.close()

    if not samples:
        print('[client] no replies received. Check IP / port / firewall.')
        sys.exit(1)

    samples.sort(key=lambda s: s[0])              # sort by rtt ascending
    best_rtt, best_offset = samples[0]
    rtts = [s[0] for s in samples]
    # offset from the few lowest-rtt samples (more robust than single min)
    k = max(1, len(samples) // 20)                # lowest 5%
    low = samples[:k]
    import statistics
    off_low = statistics.median([s[1] for s in low])
    one_way_est = best_rtt / 2.0

    print('\n' + '=' * 70)
    print(f'  probes sent={sent} replied={len(samples)} lost={lost}')
    print(f'  RTT     min={min(rtts)*1000:.2f}ms  median={statistics.median(rtts)*1000:.2f}ms  '
          f'max={max(rtts)*1000:.2f}ms')
    print(f'  offset  (server - client)')
    print(f'    from min-RTT sample : {best_offset*1000:+.2f} ms   (rtt={best_rtt*1000:.2f}ms)')
    print(f'    median of lowest 5% : {off_low*1000:+.2f} ms')
    print(f'  est. one-way net delay (min-RTT/2): {one_way_est*1000:.2f} ms')
    print('-' * 70)
    print('  If SERVER = laptop and CLIENT = robot, the laptop CSV times are on')
    print('  the server clock. To compare with robot detect_time (client clock):')
    print(f'      pass to analyzer:  --clock-offset {best_offset:.6f}')
    print('      (analyzer subtracts (server-client) offset from laptop times)')
    print('=' * 70 + '\n')


def main(args=None):
    p = argparse.ArgumentParser(description='NTP-style UDP clock offset measurement')
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--server', action='store_true', help='run as responder')
    g.add_argument('--client', metavar='HOST', help='run as prober against HOST')
    p.add_argument('--port', type=int, default=5005)
    p.add_argument('--bind', default='0.0.0.0', help='server bind address, default all IPv4 interfaces')
    p.add_argument('--verbose', action='store_true', help='server: print every received valid probe')
    p.add_argument('--count', type=int, default=500, help='number of probes (client)')
    p.add_argument('--interval', type=float, default=0.005, help='seconds between probes (client)')
    ns = p.parse_args(args if args is not None else sys.argv[1:])

    if ns.server:
        run_server(ns.port, ns.bind, ns.verbose)
    else:
        run_client(ns.client, ns.port, ns.count, ns.interval)


if __name__ == '__main__':
    main()
