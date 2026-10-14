#!/usr/bin/env python3
"""
Flash Generator + UDP Trigger (Laptop Side, Windows-friendly)

Full-screen black/white flasher. On every flip to WHITE it sends a small UDP
packet to the robot so the robot can stamp the arrival with its OWN clock. This
is the single-clock latency method: it needs NO clock synchronization between
laptop and robot, sidestepping Windows clock drift / NTP stepping.

The robot computes per flash:
    latency = robot_detect_time - robot_udp_arrival_time - net_one_way_delay
where net_one_way_delay ~= min-RTT/2 from clock_sync.py (typically ~1 ms).

It also writes a CSV (one row per white flip) as a local backup / sanity log;
the CSV is NOT needed for the single-clock analysis, which uses the robot-side
files (flash_events.json + flash_events_udp.json).

Dependencies: standard library only (tkinter, socket, struct, time, csv).

Usage (laptop):
    python flash_generator.py --robot-ip 10.60.133.186 --udp-port 5006 \
        --interval 2.5 --count 30

Keys in the window:  q or ESC = quit early.
"""

import argparse
import socket
import struct
import time
import csv
import sys
import tkinter as tk


UDP_PACKET_FMT = '!q'   # must match flash_detector_node.UDP_PACKET_FMT (int64 seq)


class Flasher:
    def __init__(self, root, robot_ip, udp_port, interval, count,
                 white_ms, csv_path, warmup):
        self.root = root
        self.interval = interval          # seconds between white flips
        self.white_ms = white_ms          # how long the white screen stays on
        self.count = count                # number of WHITE flashes to emit (0 = endless)
        self.warmup = warmup              # warmup white flips not sent/recorded
        self.csv_path = csv_path

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest = (robot_ip, udp_port)

        self.seq = 0                      # recorded white-flash counter (post-warmup)
        self.flips = 0                    # total flips incl. warmup
        self.rows = []

        # Fullscreen canvas
        self.canvas = tk.Canvas(root, highlightthickness=0, bg='black')
        self.canvas.pack(fill='both', expand=True)
        root.attributes('-fullscreen', True)
        root.configure(bg='black')
        root.bind('<Escape>', lambda e: self.quit())
        root.bind('q', lambda e: self.quit())

        self.state = 'black'
        # Kick off after the window is actually mapped so the first white is real.
        root.after(800, self.flip_white)

    def _set_color(self, color):
        self.canvas.configure(bg=color)
        self.canvas.delete('all')
        # Force the draw to happen now so the photons are on their way before
        # we send the UDP packet (minimizes screen-vs-packet skew).
        self.canvas.update_idletasks()
        self.root.update_idletasks()

    def flip_white(self):
        self._set_color('white')
        # Timestamp + send UDP as close to the visual flip as possible.
        send_perf = time.perf_counter()
        wall = time.time()
        self.flips += 1
        is_warmup = self.flips <= self.warmup

        if not is_warmup:
            self.seq += 1
            try:
                self.sock.sendto(struct.pack(UDP_PACKET_FMT, self.seq), self.dest)
            except OSError as e:
                print(f'[warn] UDP send failed: {e}')
            self.rows.append({
                'seq': self.seq,
                'wall_time_s': f'{wall:.6f}',
                'perf_counter_s': f'{send_perf:.6f}',
            })
            print(f'WHITE flash seq={self.seq}  wall={wall:.6f}  sent to {self.dest[0]}:{self.dest[1]}')
        else:
            print(f'warmup white {self.flips}/{self.warmup} (not sent)')

        self.state = 'white'
        self.root.after(self.white_ms, self.flip_black)

    def flip_black(self):
        self._set_color('black')
        self.state = 'black'
        if self.count and self.seq >= self.count:
            # Done after the last white's black period.
            self.root.after(300, self.quit)
            return
        # Schedule next white so that white-flip period == interval.
        delay = max(1, int(self.interval * 1000) - self.white_ms)
        self.root.after(delay, self.flip_white)

    def quit(self):
        self.save_csv()
        try:
            self.sock.close()
        except OSError:
            pass
        print(f'\nDone. {self.seq} white flashes sent. CSV: {self.csv_path}')
        self.root.destroy()

    def save_csv(self):
        if not self.rows:
            return
        try:
            with open(self.csv_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=['seq', 'wall_time_s', 'perf_counter_s'])
                w.writeheader()
                w.writerows(self.rows)
        except Exception as e:
            print(f'[warn] failed to save CSV: {e}')


def main():
    p = argparse.ArgumentParser(description='Laptop flash generator + UDP trigger')
    p.add_argument('--robot-ip', required=True, help='robot IP to send UDP triggers to')
    p.add_argument('--udp-port', type=int, default=5006, help='must match flash_detector udp_port')
    p.add_argument('--interval', type=float, default=2.5, help='seconds between white flips')
    p.add_argument('--white-ms', type=int, default=600, help='white-on duration per flash (ms)')
    p.add_argument('--count', type=int, default=30, help='number of white flashes (0=endless)')
    p.add_argument('--warmup', type=int, default=2, help='warmup white flips, not sent/recorded')
    p.add_argument('--csv', default='laptop_flash_log.csv', help='backup CSV path')
    ns = p.parse_args()

    if ns.white_ms >= ns.interval * 1000:
        print('[error] --white-ms must be smaller than --interval*1000')
        sys.exit(2)

    print(f'Flashing to {ns.robot_ip}:{ns.udp_port}, interval={ns.interval}s, '
          f'count={ns.count}, warmup={ns.warmup}. Press q/ESC to stop.')

    root = tk.Tk()
    root.title('Flash Generator')
    Flasher(root, ns.robot_ip, ns.udp_port, ns.interval, ns.count,
            ns.white_ms, ns.csv, ns.warmup)
    root.mainloop()


if __name__ == '__main__':
    main()
