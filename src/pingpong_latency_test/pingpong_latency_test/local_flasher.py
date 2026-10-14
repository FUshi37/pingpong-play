#!/usr/bin/env python3
"""
Local Flasher (Same-Machine, Single-Clock Latency)

Runs on the SAME machine as flash_detector (e.g. the robot IPC with a screen
plugged in). Full-screen black/white flasher; on every flip to WHITE it stamps
time.time() and appends {seq, arrival_time} to a JSON file in the exact format
flash_detector writes for UDP arrivals (flash_events_udp.json).

Because the flip stamp and the camera detect_time are now taken on the SAME
clock of the SAME process tree, there is no network hop and no cross-machine
clock sync at all. Analyze with:

    ros2 run pingpong_latency_test latency_analyzer flash_events.json \
        --udp flash_events_udp.json --net-delay 0

vs the laptop+UDP method this removes the ~1ms network term entirely. It does
NOT remove the screen terms (see "Refresh rate" below) -- those are physics of
the display and are the same on any machine.

--------------------------------------------------------------------------------
Refresh rate (the term you must account for on a same-machine test)
--------------------------------------------------------------------------------
We stamp arrival_time the instant we *request* the white frame (after
update_idletasks). The pixels do not actually emit light until the next display
scan-out (vblank). Since our flip timer is uncorrelated with vblank, that wait
is ~uniform in [0, 1/refresh], i.e. a mean bias of HALF a refresh period that
INFLATES the measured latency:

    60 Hz  -> up to 16.7 ms, mean ~8.3 ms
   120 Hz  -> up to  8.3 ms, mean ~4.2 ms
   144 Hz  -> up to  6.9 ms, mean ~3.5 ms

On top of that sits the LCD pixel response (black->white rise), which no
software trick removes -- that is why a flash test is always an UPPER BOUND on
the real ball-detection pipeline (a real ball has no screen in front of it).

What to do about it:
  1. Set the monitor to its highest refresh rate (xrandr / display settings)
     before testing. This script auto-detects the rate via xrandr and prints it.
  2. Run many flashes; the per-flash quantization averages out in the mean.
  3. Optionally pass --compensate-refresh to add half a refresh period to each
     arrival_time, which subtracts the *known* scan-out bias and leaves only the
     LCD-response upper bound. Off by default so the raw numbers stay honest.
     (Equivalent: keep it off and pass --net-delay <half_period_s> to the
     analyzer.)

Dependencies: standard library only (tkinter, json, time). xrandr is optional
(used only to auto-detect the refresh rate).

Usage:
    ros2 run pingpong_latency_test local_flasher --interval 2.5 --count 30
    # or plain python on the IPC:
    python local_flasher.py --interval 2.5 --count 30

Keys in the window:  q or ESC = quit early.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
import tkinter as tk


def detect_refresh_hz():
    """Best-effort current refresh rate (Hz) of the active output via xrandr.

    Returns a float, or None if it cannot be determined.
    """
    try:
        out = subprocess.run(['xrandr', '--current'], capture_output=True,
                             text=True, timeout=2.0).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    # The active mode line marks the current rate with a trailing '*'.
    for line in out.splitlines():
        m = re.search(r'(\d+\.\d+)\*', line)
        if m:
            return float(m.group(1))
    return None


class LocalFlasher:
    def __init__(self, root, out_path, interval, count, white_ms, warmup,
                 refresh_hz, compensate):
        self.root = root
        self.out_path = out_path
        self.interval = interval          # seconds between white flips
        self.white_ms = white_ms          # how long the white screen stays on
        self.count = count                # number of WHITE flashes (0 = endless)
        self.warmup = warmup              # warmup flips not recorded
        # Half a refresh period: mean scan-out wait we optionally compensate for.
        self.half_period = (0.5 / refresh_hz) if (compensate and refresh_hz) else 0.0
        self.compensate = compensate and bool(refresh_hz)

        self.seq = 0                      # recorded white-flash counter (post-warmup)
        self.flips = 0                    # total flips incl. warmup
        self.events = []                  # [{'seq', 'arrival_time'}]

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
        # Force the redraw into the back buffer now so the photons are on their
        # way (modulo vblank) before we stamp the time.
        self.canvas.update_idletasks()
        self.root.update_idletasks()

    def flip_white(self):
        self._set_color('white')
        # Stamp on the robot clock as close to the visual flip as possible. The
        # detector's detect_time is on this same clock, so the difference is the
        # full screen->state latency with no network or clock-sync term.
        stamp = time.time() + self.half_period
        self.flips += 1
        is_warmup = self.flips <= self.warmup

        if not is_warmup:
            self.seq += 1
            self.events.append({'seq': self.seq, 'arrival_time': stamp})
            print(f'WHITE flash seq={self.seq}  arrival_time={stamp:.6f}'
                  + ('  (+half-period)' if self.compensate else ''))
        else:
            print(f'warmup white {self.flips}/{self.warmup} (not recorded)')

        self.state = 'white'
        self.root.after(self.white_ms, self.flip_black)

    def flip_black(self):
        self._set_color('black')
        self.state = 'black'
        if self.count and self.seq >= self.count:
            self.root.after(300, self.quit)
            return
        delay = max(1, int(self.interval * 1000) - self.white_ms)
        self.root.after(delay, self.flip_white)

    def quit(self):
        self.save()
        print(f'\nDone. {self.seq} white flashes recorded -> {self.out_path}')
        self.root.destroy()

    def save(self):
        try:
            with open(self.out_path, 'w') as f:
                json.dump(self.events, f, indent=2)
        except Exception as e:
            print(f'[warn] failed to save {self.out_path}: {e}')


def main():
    p = argparse.ArgumentParser(description='Same-machine flasher (single-clock latency)')
    p.add_argument('--out', default='flash_events_udp.json',
                   help='output JSON, same format flash_detector writes (default flash_events_udp.json)')
    p.add_argument('--interval', type=float, default=2.5, help='seconds between white flips')
    p.add_argument('--white-ms', type=int, default=600, help='white-on duration per flash (ms)')
    p.add_argument('--count', type=int, default=30, help='number of white flashes (0=endless)')
    p.add_argument('--warmup', type=int, default=2, help='warmup white flips, not recorded')
    p.add_argument('--refresh-hz', type=float, default=None,
                   help='display refresh rate (Hz); auto-detected via xrandr if omitted')
    p.add_argument('--compensate-refresh', action='store_true',
                   help='add half a refresh period to each stamp to remove the scan-out bias '
                        '(leaves only the LCD-response upper bound)')
    ns = p.parse_args()

    if ns.white_ms >= ns.interval * 1000:
        print('[error] --white-ms must be smaller than --interval*1000')
        sys.exit(2)

    refresh_hz = ns.refresh_hz if ns.refresh_hz is not None else detect_refresh_hz()
    if refresh_hz:
        half_ms = 0.5 / refresh_hz * 1000.0
        print(f'Refresh rate: {refresh_hz:.2f} Hz  '
              f'(scan-out adds 0..{1000.0/refresh_hz:.1f} ms, mean ~{half_ms:.1f} ms to measured latency)')
        if not ns.compensate_refresh:
            print(f'  tip: pass --compensate-refresh to subtract the ~{half_ms:.1f} ms scan-out bias, '
                  f'or analyze with --net-delay {0.5/refresh_hz:.4f}')
    else:
        print('Refresh rate: unknown (xrandr not available). '
              'Set the monitor to its highest rate and pass --refresh-hz to log the bias.')
        if ns.compensate_refresh:
            print('[warn] --compensate-refresh ignored: no refresh rate known.')

    print(f'Flashing -> {os.path.abspath(ns.out)}, interval={ns.interval}s, '
          f'count={ns.count}, warmup={ns.warmup}. Press q/ESC to stop.')

    root = tk.Tk()
    root.title('Local Flasher')
    LocalFlasher(root, ns.out, ns.interval, ns.count, ns.white_ms, ns.warmup,
                 refresh_hz, ns.compensate_refresh)
    root.mainloop()


if __name__ == '__main__':
    main()
