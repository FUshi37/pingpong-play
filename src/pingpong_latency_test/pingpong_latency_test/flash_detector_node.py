#!/usr/bin/env python3
"""
Flash Detector Node (Robot Side)

Subscribes to camera images and detects black->white transitions, recording a
timestamp for every detected white flash on the ROBOT clock.

In parallel it runs a UDP listener. The laptop sends a small UDP packet at the
instant it flips its screen to white. The robot stamps the packet's arrival
time with its OWN clock (time.time()). Because both the UDP-arrival time and the
camera-detection time are taken on the same robot clock, the end-to-end latency

    latency = detect_time - udp_arrival_time - net_one_way_delay

needs NO cross-machine clock sync. This sidesteps clock drift / NTP stepping on
the laptop, which otherwise dominates the error budget.

For each detected white transition it records (all epoch seconds, robot clock):
  - image_stamp:   sensor_msgs header.stamp (driver/camera capture time)
  - receive_time:  wall-clock time the image arrived in the callback
  - detect_time:   wall-clock time the brightness transition was confirmed
  - transport_latency_ms / decode_latency_ms / detection_latency_ms: SW breakdown

UDP arrivals are saved to a sibling file (<log_file stem>_udp.json) as a list of
{seq, arrival_time}. latency_analyzer pairs detections with arrivals on the
robot clock.

Usage:
    ros2 run pingpong_latency_test flash_detector --ros-args \
        -p image_topic:=/camera/camera/color/image_raw/compressed \
        -p roi:=x,y,w,h -p brightness_threshold:=0.6 \
        -p udp_port:=5006
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import json
import socket
import struct
import threading
import os
from dataclasses import dataclass, asdict


# UDP trigger packet: network byte order, int64 sequence number.
UDP_PACKET_FMT = '!q'
UDP_PACKET_SIZE = struct.calcsize(UDP_PACKET_FMT)


@dataclass
class FlashEvent:
    """A single detected black->white transition."""
    flash_id: int

    # Timestamps (epoch seconds)
    image_stamp: float    # sensor_msgs header.stamp (capture time from driver)
    receive_time: float   # wall-clock time image entered callback
    detect_time: float    # wall-clock time transition was confirmed

    # Latency breakdown (milliseconds) - software side only
    transport_latency_ms: float   # image_stamp -> receive_time
    decode_latency_ms: float      # decode / conversion cost
    detection_latency_ms: float   # brightness compute + logic cost

    # Detection info
    brightness: float             # normalized 0-1 brightness at confirmation
    prev_brightness: float        # brightness of previous frame


class FlashDetectorNode(Node):
    def __init__(self):
        super().__init__('flash_detector')

        # Parameters
        self.declare_parameter('image_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('brightness_threshold', 0.5)   # 0-1, white if above
        self.declare_parameter('transition_hysteresis', 0.15)  # gap below/above threshold
        self.declare_parameter('log_file', 'flash_events.json')
        self.declare_parameter('min_flash_interval', 0.3)     # debounce seconds
        self.declare_parameter('roi', '')                     # 'x,y,w,h' optional ROI
        self.declare_parameter('udp_port', 5006)              # UDP trigger port (0 disables)

        self.image_topic = self.get_parameter('image_topic').value
        self.brightness_threshold = self.get_parameter('brightness_threshold').value
        self.hysteresis = self.get_parameter('transition_hysteresis').value
        self.log_file = self.get_parameter('log_file').value
        self.min_flash_interval = self.get_parameter('min_flash_interval').value
        roi_str = self.get_parameter('roi').value
        self.udp_port = self.get_parameter('udp_port').value

        # Parse optional ROI
        self.roi = None
        if roi_str:
            try:
                x, y, w, h = (int(v) for v in roi_str.split(','))
                self.roi = (x, y, w, h)
            except ValueError:
                self.get_logger().warn(f'Invalid roi "{roi_str}", ignoring. Use "x,y,w,h".')

        # State
        self.bridge = CvBridge()
        self.current_state = 'black'      # 'black' or 'white'
        self.prev_brightness = 0.0
        self.last_transition_time = 0.0
        self.events = []
        self.flash_count = 0
        self.total_frames = 0

        # UDP trigger arrivals (robot clock). Guarded by a lock; written by the
        # listener thread, read at save time.
        self.udp_events = []              # list of {'seq': int, 'arrival_time': float}
        self.udp_lock = threading.Lock()
        self.udp_log_file = self._udp_log_path(self.log_file)
        self.udp_sock = None
        self.udp_thread = None
        self.udp_running = False

        # Rolling brightness diagnostics (reset each stats window)
        self.win_bright_min = 1.0
        self.win_bright_max = 0.0
        self.win_frames = 0
        self.last_brightness = 0.0

        # Thresholds with hysteresis
        self.white_thresh = self.brightness_threshold + self.hysteresis / 2.0
        self.black_thresh = self.brightness_threshold - self.hysteresis / 2.0

        # Subscriber
        self.is_compressed = 'compressed' in self.image_topic.lower()
        if self.is_compressed:
            self.image_sub = self.create_subscription(
                CompressedImage, self.image_topic, self.image_callback_compressed, 10
            )
        else:
            self.image_sub = self.create_subscription(
                Image, self.image_topic, self.image_callback_raw, 10
            )

        self.get_logger().info('Flash Detector started (record-only mode)')
        self.get_logger().info(f'  Image topic: {self.image_topic} ({"compressed" if self.is_compressed else "raw"})')
        self.get_logger().info(f'  White threshold: >{self.white_thresh:.2f}, Black threshold: <{self.black_thresh:.2f}')
        self.get_logger().info(f'  ROI: {self.roi if self.roi else "full frame"}')
        self.get_logger().info(f'  Log file: {self.log_file}')

        # Start UDP listener for laptop triggers (single-clock latency method)
        if self.udp_port and self.udp_port > 0:
            self._start_udp_listener()
            self.get_logger().info(f'  UDP trigger listener: port {self.udp_port} -> {self.udp_log_file}')
        else:
            self.get_logger().info('  UDP trigger listener: disabled (udp_port=0)')

        self.stats_timer = self.create_timer(5.0, self.print_stats)

    @staticmethod
    def _udp_log_path(log_file):
        stem, ext = os.path.splitext(log_file)
        return f'{stem}_udp{ext or ".json"}'

    def _start_udp_listener(self):
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.udp_sock.bind(('0.0.0.0', self.udp_port))
        self.udp_sock.settimeout(0.5)
        self.udp_running = True
        self.udp_thread = threading.Thread(target=self._udp_loop, daemon=True)
        self.udp_thread.start()

    def _udp_loop(self):
        first = True
        while self.udp_running:
            try:
                data, addr = self.udp_sock.recvfrom(64)
            except socket.timeout:
                continue
            except OSError:
                break
            arrival_time = time.time()          # ROBOT clock, same as detect_time
            if len(data) >= UDP_PACKET_SIZE:
                (seq,) = struct.unpack(UDP_PACKET_FMT, data[:UDP_PACKET_SIZE])
            else:
                seq = -1
            with self.udp_lock:
                self.udp_events.append({'seq': int(seq), 'arrival_time': arrival_time})
            if first:
                first = False
                self.get_logger().info(f'UDP trigger: first packet seq={seq} from {addr[0]}')
            self._save_udp_events()

    def _save_udp_events(self):
        # When UDP is disabled (udp_port=0) or no packets arrived, do NOT touch
        # the file: another producer (local_flasher, same-machine mode) may own
        # it, and writing an empty list here would clobber its data.
        if not self.udp_port:
            return
        try:
            with self.udp_lock:
                snapshot = list(self.udp_events)
            if not snapshot:
                return
            with open(self.udp_log_file, 'w') as f:
                json.dump(snapshot, f, indent=2)
        except Exception as e:
            self.get_logger().error(f'Failed to save UDP events: {e}')

    def _stop_udp_listener(self):
        self.udp_running = False
        if self.udp_thread is not None:
            self.udp_thread.join(timeout=1.0)
        if self.udp_sock is not None:
            self.udp_sock.close()

    def image_callback_compressed(self, msg: CompressedImage):
        receive_time = time.time()
        image_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

        t_decode_start = time.time()
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        decode_latency_ms = (time.time() - t_decode_start) * 1000.0

        if cv_image is not None:
            self.process_image(cv_image, image_stamp, receive_time, decode_latency_ms)

    def image_callback_raw(self, msg: Image):
        receive_time = time.time()
        image_stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9

        t_decode_start = time.time()
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        decode_latency_ms = (time.time() - t_decode_start) * 1000.0

        self.process_image(cv_image, image_stamp, receive_time, decode_latency_ms)

    def process_image(self, image, image_stamp, receive_time, decode_latency_ms):
        t_detect_start = time.time()
        self.total_frames += 1

        # Optional ROI crop
        if self.roi is not None:
            x, y, w, h = self.roi
            image = image[y:y + h, x:x + w]

        # Brightness 0-1
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray)) / 255.0

        # Rolling diagnostics
        self.last_brightness = brightness
        self.win_bright_min = min(self.win_bright_min, brightness)
        self.win_bright_max = max(self.win_bright_max, brightness)
        self.win_frames += 1

        # State machine with hysteresis
        transition_to_white = False
        if self.current_state == 'black' and brightness > self.white_thresh:
            self.current_state = 'white'
            transition_to_white = True
        elif self.current_state == 'white' and brightness < self.black_thresh:
            self.current_state = 'black'

        detect_time = time.time()
        detection_latency_ms = (detect_time - t_detect_start) * 1000.0

        if transition_to_white:
            # Debounce
            if detect_time - self.last_transition_time < self.min_flash_interval:
                self.prev_brightness = brightness
                return
            self.last_transition_time = detect_time
            self.flash_count += 1

            transport_latency_ms = (receive_time - image_stamp) * 1000.0

            event = FlashEvent(
                flash_id=self.flash_count,
                image_stamp=image_stamp,
                receive_time=receive_time,
                detect_time=detect_time,
                transport_latency_ms=transport_latency_ms,
                decode_latency_ms=decode_latency_ms,
                detection_latency_ms=detection_latency_ms,
                brightness=brightness,
                prev_brightness=self.prev_brightness,
            )
            self.events.append(event)

            self.get_logger().info(
                f'WHITE flash #{self.flash_count} | '
                f'detect_time={detect_time:.6f} image_stamp={image_stamp:.6f} | '
                f'transport={transport_latency_ms:.1f}ms decode={decode_latency_ms:.1f}ms '
                f'detect={detection_latency_ms:.1f}ms'
            )
            self.save_events()

        self.prev_brightness = brightness

    def save_events(self):
        try:
            with open(self.log_file, 'w') as f:
                json.dump([asdict(e) for e in self.events], f, indent=2)
        except Exception as e:
            self.get_logger().error(f'Failed to save events: {e}')

    def print_stats(self):
        # Brightness diagnostics every window - the key signal when nothing is detected.
        if self.win_frames == 0:
            self.get_logger().warn(
                f'No frames in last 5s (total={self.total_frames}). '
                f'Check topic name / ROS_DOMAIN_ID / camera driver. '
                f'Try: ros2 topic hz {self.image_topic}'
            )
        else:
            spans_white = self.win_bright_max > self.white_thresh
            spans_black = self.win_bright_min < self.black_thresh
            verdict = 'OK (spans both)' if (spans_white and spans_black) else (
                'never reaches WHITE thresh' if not spans_white else 'never drops to BLACK thresh'
            )
            self.get_logger().info(
                f'Brightness last 5s: min={self.win_bright_min:.3f} max={self.win_bright_max:.3f} '
                f'cur={self.last_brightness:.3f} | thresh white>{self.white_thresh:.3f} '
                f'black<{self.black_thresh:.3f} | {verdict} | '
                f'frames={self.win_frames} flashes={self.flash_count} state={self.current_state}'
            )

        # Reset window
        self.win_bright_min, self.win_bright_max, self.win_frames = 1.0, 0.0, 0

        with self.udp_lock:
            n_udp = len(self.udp_events)
        if self.udp_port:
            self.get_logger().info(
                f'  Detections={self.flash_count}  UDP triggers received={n_udp}'
            )

        if not self.events:
            return

        transport = [e.transport_latency_ms for e in self.events]
        decode = [e.decode_latency_ms for e in self.events]
        detection = [e.detection_latency_ms for e in self.events]

        self.get_logger().info(
            f'  Software latency (mean): transport={np.mean(transport):.2f}ms '
            f'decode={np.mean(decode):.2f}ms detect={np.mean(detection):.2f}ms'
        )


def main(args=None):
    rclpy.init(args=args)
    node = FlashDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_udp_listener()
        node.save_events()
        node._save_udp_events()
        node.print_stats()
        with node.udp_lock:
            n_udp = len(node.udp_events)
        node.get_logger().info(
            f'Saved {len(node.events)} flash events to {node.log_file}, '
            f'{n_udp} UDP triggers to {node.udp_log_file}')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
