#!/usr/bin/env python3
"""Replay 50 calibration postures slowly and replace rejected points with new poses.

Run RUN_AUTO_MOCAP_CALIBRATION.sh --dry-run to inspect the route without ROS.
The live command stream continues on a ROS timer during UDP sampling/retries.
No recorded mocap measurement is reused in the new calibration.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np

from pingpong_controller.reset_trajectory import SlowResetTrajectory

JOINT_NAMES = [f'RightArm-{i}' for i in range(7)]
DEFAULT_WAYPOINTS = Path(__file__).with_name(
    'mocap_calibration_waypoints_20261027.json')


class PointRejected(RuntimeError):
    """This finished motion produced no usable sample; continue the route."""


def load_waypoints(path: str | Path) -> tuple[dict, np.ndarray]:
    document = json.loads(Path(path).read_text())
    if document.get('joint_names') != JOINT_NAMES:
        raise ValueError('Waypoint joint_names must be RightArm-0 through RightArm-6')
    points = document['waypoints']
    if len(points) != 50:
        raise ValueError(f'Expected all 50 accepted samples, got {len(points)}')
    if [p['source_sample_index'] for p in points] != list(range(1, 51)):
        raise ValueError('Waypoint sample indices must preserve the original order 1..50')
    q = np.asarray([p['joint_positions_rad'] for p in points], dtype=float)
    if q.shape != (50, 7) or not np.all(np.isfinite(q)):
        raise ValueError('Waypoints must contain 50 finite seven-joint vectors in radians')
    return document, q


def load_resume(path, args, fk):
    """Validate a saved partial session before opening ROS or modifying progress."""
    path = Path(path).resolve()
    saved = json.loads(path.read_text())
    samples = saved.get('samples', [])
    count = len(samples)
    supplemental = saved.get('supplemental_waypoints', [])
    if not isinstance(supplemental, list):
        raise ValueError('Resume supplemental_waypoints must be a list')
    route_count = 50 + len(supplemental)
    if len(supplemental) > args.max_supplemental_points:
        raise ValueError('--max-supplemental-points is smaller than the saved supplemental route')
    if not 0 <= count <= route_count or saved.get('accepted_count') != count:
        raise ValueError('Resume accepted_count does not match the saved samples')
    if saved.get('expected_count') != route_count or saved.get('base_waypoint_count', 50) != 50:
        raise ValueError('Resume file does not match the original 50 points plus its supplemental route')
    if Path(saved.get('waypoint_file', '')).resolve() != Path(args.waypoints).resolve():
        raise ValueError('Resume file uses a different waypoint file')
    digest = hashlib.sha256(Path(args.waypoints).read_bytes()).hexdigest()
    if saved.get('waypoints_sha256', digest) != digest:
        raise ValueError('Waypoints changed since the progress file was saved')
    previous = saved.get('options', {})
    if supplemental and previous.get('target_samples', 50) != args.target_samples:
        raise ValueError('Resume requires the same --target-samples for a saved supplemental route')
    for name in ('rigid_body_slot', 'rigid_body_id', 'position_unit',
                 'timestamp_unit', 'right_arm_mu_idx'):
        if previous.get(name) != getattr(args, name):
            raise ValueError(f'Resume requires the same --{name.replace("_", "-")}')
    if Path(previous.get('robot_xml', '')).resolve() != Path(args.robot_xml).resolve():
        raise ValueError('Resume requires the same robot XML')
    accepted_points = list(saved.get('accepted_point_indices', range(1, count + 1)))
    skipped_points = list(saved.get('skipped_points', []))
    completed = saved.get('completed_point', count)
    # Older versions stopped on an exhausted/bad point. Mark that known
    # rejection as skipped on resume instead of spending another retry budget.
    if saved.get('schema_version', 1) < 3 and saved.get('status') == 'failed':
        current = saved.get('current_point')
        reason = saved.get('error') or ''
        recoverable = ('failed after ' in reason
                       or 'joint settling timed out' in reason
                       or 'feedback did not reach/stabilize at the target' in reason)
        if current == completed + 1 and current <= 50 and recoverable:
            attempts = [event.get('attempt', 0) for event in saved.get('attempts', [])
                        if event.get('point') == current]
            skipped_points.append(dict(point=current, reason=reason,
                                       attempts=max(attempts, default=0),
                                       recovered_from_previous_failure=True))
            completed = current
    if (any(type(point) is not int or not 1 <= point <= route_count for point in accepted_points)
            or len(accepted_points) != count
            or accepted_points != sorted(set(accepted_points))):
        raise ValueError('Resume accepted waypoint indices do not match the measured samples')
    skipped_indices = [point.get('point') for point in skipped_points]
    if (any(type(point) is not int or not 1 <= point <= route_count for point in skipped_indices)
            or len(skipped_indices) != len(set(skipped_indices))
            or type(completed) is not int or not 0 <= completed <= route_count
            or count + len(skipped_points) != completed
            or set(accepted_points).intersection(skipped_indices)
            or set(accepted_points).union(skipped_indices) != set(range(1, completed + 1))):
        raise ValueError('Resume completed/accepted/skipped waypoint records disagree')
    saved['accepted_point_indices'] = accepted_points
    saved['skipped_points'] = skipped_points
    saved['completed_point'] = completed
    saved['supplemental_waypoints'] = supplemental
    from pingpong_controller.mocap_calibration_supplement import SupplementalPlanner
    planner = SupplementalPlanner(fk, joint_limits(args.robot_xml), args)
    reference = (saved.get('supplemental_workspace') or {}).get('reference_rotation_sim')
    if reference is not None:
        planner.set_reference_rotation(reference)
    for index, row in enumerate(supplemental, 51):
        if not isinstance(row, dict) or row.get('point_index') != index:
            raise ValueError('Resume supplemental waypoint indices must be consecutive from 51')
        q = np.asarray(row.get('joint_positions_rad'), dtype=float)
        if not planner.contains(q, margin=planner.margin):
            raise ValueError(f'Resume supplemental point {index} is outside the V172 workspace')
    shapes = dict(mocap_m=(3,), robot_m=(3,), mocap_orientation_wxyz=(4,),
                  T_mocap_rigid=(4, 4), T_waist03_ee=(4, 4), joint_positions_rad=(7,))
    from pingpong_controller.mocap_transform import pose_wxyz_to_matrix
    for index, sample in enumerate(samples, 1):
        if sample.get('index') != index or sample.get('robot_source') != 'fk':
            raise ValueError('Saved samples must be consecutive FK pose pairs')
        arrays = {}
        for key, shape in shapes.items():
            value = np.asarray(sample.get(key), dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f'Saved sample {index} has an invalid {key}')
            arrays[key] = value
        if not np.allclose(fk.ee_pose(arrays['joint_positions_rad']), arrays['T_waist03_ee'], atol=1e-7, rtol=0):
            raise ValueError(f'Saved sample {index}: measured joints and FK pose disagree')
        if not np.allclose(arrays['robot_m'], arrays['T_waist03_ee'][:3, 3], atol=1e-7, rtol=0):
            raise ValueError(f'Saved sample {index}: robot position and pose disagree')
        mocap_pose = pose_wxyz_to_matrix(arrays['mocap_m'], arrays['mocap_orientation_wxyz'])
        if not np.allclose(mocap_pose, arrays['T_mocap_rigid'], atol=1e-7, rtol=0):
            raise ValueError(f'Saved sample {index}: mocap position/orientation and pose disagree')
    saved['_resume_path'] = str(path)
    return saved


def restore_session(node, saved):
    """Restore measured pose pairs, never substitute waypoint targets for feedback."""
    from pingpong_controller.mocap_calibration import CalibrationSample, CalibrationSession
    session = CalibrationSession()
    array_names = ('mocap_m', 'robot_m', 'mocap_orientation_wxyz',
                   'T_mocap_rigid', 'T_waist03_ee', 'joint_positions_rad')
    for row in saved['samples']:
        values = {key: value for key, value in row.items()
                  if key in CalibrationSample.__dataclass_fields__}
        for name in array_names:
            values[name] = np.asarray(row[name], dtype=float).copy()
        sample = CalibrationSample(**values)
        session.samples.append(sample)
        session.mocap_points.append(sample.mocap_m.copy())
        session.robot_points.append(sample.robot_m.copy())
        session.mocap_poses.append(sample.T_mocap_rigid.copy())
        session.robot_poses.append(sample.T_waist03_ee.copy())
    node.session = session


def joint_limits(xml_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(xml_path).getroot()
    compiler = root.find('compiler')
    degrees = compiler is None or compiler.get('angle', 'degree') == 'degree'
    limits = {}
    for joint in root.findall('.//worldbody//joint'):
        if joint.get('name') in JOINT_NAMES:
            limits[joint.get('name')] = [float(x) for x in joint.get('range', '').split()]
    if any(len(limits.get(name, [])) != 2 for name in JOINT_NAMES):
        raise ValueError('Robot XML must specify ranges for all seven right-arm joints')
    values = np.array([limits[name] for name in JOINT_NAMES])
    if degrees:
        values = np.deg2rad(values)
    if not np.all(np.isfinite(values)) or np.any(values[:, 0] >= values[:, 1]):
        raise ValueError('Invalid right-arm joint limits in robot XML')
    return values[:, 0], values[:, 1]


def segment_duration(start: np.ndarray, target: np.ndarray, args) -> float:
    distance = float(np.max(np.abs(target - start)))
    if distance <= 1e-7:
        return 0.0
    # Analytic extrema of the quintic trajectory used by SlowResetTrajectory.
    return max(args.min_move_seconds,
               1.875 * distance / np.deg2rad(args.speed_deg_s),
               np.sqrt((10 / np.sqrt(3)) * distance / np.deg2rad(args.accel_deg_s2)))


class ArmMotion:
    """Thread-protected feedback and bounded reference motion; independent of ROS."""

    def __init__(self, args, limits):
        self.args = args
        self.low, self.high = limits
        self.lock = threading.RLock()
        self.history = deque()
        self.q = None
        self.received_at = None
        self.command = None
        self.target = None
        self.fault = None
        self.armed = False
        self.last_tick = None
        self.workspace = None  # Enabled only for the supplemental route.
        self.trajectory = SlowResetTrajectory(
            dt_s=1 / args.publish_hz,
            velocity_limit_deg_s=args.speed_deg_s,
            acceleration_limit_deg_s2=args.accel_deg_s2,
            minimum_duration_s=args.min_move_seconds)

    def feedback(self, q, now):
        with self.lock:
            q = np.asarray(q, dtype=float)
            if q.shape != (7,) or not np.all(np.isfinite(q)):
                if self.armed:
                    self.fault = 'Non-finite or incomplete right-arm feedback'
                return
            self.q = q.copy()
            self.received_at = now
            self.history.append((now, q.copy()))
            while self.history and self.history[0][0] < now - max(5., self.args.joint_stable_seconds + 1):
                self.history.popleft()

    def check(self, now):
        with self.lock:
            if self.fault:
                raise RuntimeError(self.fault)
            if self.q is None or now - self.received_at > self.args.feedback_timeout:
                raise RuntimeError('Right-arm /joint_states feedback is missing or stale')
            if np.any(self.q < self.low) or np.any(self.q > self.high):
                raise RuntimeError('Measured joint position is outside model limits')
            if self.command is not None and np.max(np.abs(self.q - self.command)) > np.deg2rad(self.args.max_tracking_error_deg):
                raise RuntimeError('Joint tracking error exceeds the configured limit')
            if self.workspace is not None and not self.workspace.contains(self.q):
                raise RuntimeError('Measured end-effector left the V172 supplemental workspace')

    def arm(self, now):
        with self.lock:
            self.check(now)
            self.command = self.q.copy()
            self.target = self.q.copy()
            self.trajectory.start(self.command, self.target)
            self.last_tick = now
            self.armed = True

    def move_to(self, target, now):
        with self.lock:
            self.check(now)
            if not self.armed or self.trajectory.active:
                raise RuntimeError('Cannot start a new segment while unarmed or moving')
            target = np.asarray(target, dtype=float)
            if target.shape != (7,) or not np.all(np.isfinite(target)) or np.any(target < self.low) or np.any(target > self.high):
                raise ValueError('Target is invalid or outside model limits')
            if self.workspace is not None and not self.workspace.path_is_safe(self.command, target):
                raise PointRejected('Supplemental joint trajectory leaves the V172 workspace')
            duration = segment_duration(self.command, target, self.args)
            self.target = target.copy()
            self.trajectory.start(self.command, target)
            return duration

    def tick(self, now):
        with self.lock:
            if not self.armed or self.fault:
                return None
            try:
                self.check(now)
            except RuntimeError as exc:
                self.fault = str(exc)
                return None
            # Never catch up missed timer ticks with a burst or a position jump.
            elapsed = now - self.last_tick
            if elapsed <= 0:
                return None
            self.last_tick = now
            self.trajectory.dt_s = min(elapsed, 1 / self.args.publish_hz)
            command, _velocity, _done = self.trajectory.step()
            if self.workspace is not None and not self.workspace.contains(command):
                self.fault = 'Supplemental command reference left the V172 workspace'
                return None
            self.command = command
            return self.command.copy()

    def at_target(self, now):
        with self.lock:
            self.check(now)
            return (not self.trajectory.active and self.target is not None
                    and np.max(np.abs(self.q - self.target)) <= np.deg2rad(self.args.arrival_tolerance_deg))

    def ready_to_sample(self, now):
        with self.lock:
            self.check(now)
            if self.args.arrival_mode == 'target':
                return self.at_target(now)
            return not self.trajectory.active and self.target is not None

    def diagnostics(self, now):
        with self.lock:
            history = [q for t, q in self.history
                       if t >= now - self.args.joint_stable_seconds]
            error = (None if self.q is None or self.target is None
                     else np.rad2deg(self.q - self.target))
            span = None if not history else np.rad2deg(np.ptp(history, axis=0))
            return dict(
                arrival_mode=self.args.arrival_mode,
                feedback_age_s=None if self.received_at is None else now - self.received_at,
                measured_q_rad=None if self.q is None else self.q.tolist(),
                target_q_rad=None if self.target is None else self.target.tolist(),
                target_error_deg=None if error is None else error.tolist(),
                max_target_error_deg=None if error is None else float(np.max(np.abs(error))),
                joint_window_range_deg=None if span is None else span.tolist(),
                max_joint_window_range_deg=None if span is None else float(np.max(span)),
                moving=bool(self.trajectory.active), fault=self.fault)

    def stable_mean(self, now, since=None):
        with self.lock:
            if not self.ready_to_sample(now):
                return None
            start = now - self.args.joint_stable_seconds if since is None else since
            samples = [(t, q) for t, q in self.history if t >= start]
            required_span = self.args.joint_stable_seconds * .8 if since is None else min(.05, max(0., now - since) * .5)
            if len(samples) < 3 or samples[-1][0] - samples[0][0] < required_span:
                return None
            values = np.array([q for _, q in samples])
            if np.max(np.ptp(values, axis=0)) > np.deg2rad(self.args.joint_stable_range_deg):
                return None
            if (self.args.arrival_mode == 'target'
                    and np.max(np.abs(values - self.target)) > np.deg2rad(self.args.arrival_tolerance_deg)):
                return None
            return np.mean(values, axis=0)

    def disarm(self):
        with self.lock:
            self.armed = False
            self.trajectory.active = False


def collect_point(index, target, io, attempts):
    """Apply the same settling, measured-FK sampling and retries to every point."""
    attempt = 0
    try:
        io.move(index, target)
        for attempt in range(1, attempts + 1):
            io.settle()
            accepted = io.sample(index, attempt)
            io.checkpoint(index, attempt, accepted)
            if accepted:
                return
            if attempt == attempts:
                raise PointRejected(
                    f'No accepted sample after {attempts} attempts: {io.node.last_rejection}')
            io.retry_wait()
    except PointRejected as exc:
        io.skip(index, str(exc), attempt)


def collect_sequence(waypoints, io, attempts: int, start_index=1):
    """Retry the original route, recording an explicit skip on quality failures."""
    for index, target in enumerate(waypoints[start_index - 1:], start_index):
        collect_point(index, target, io, attempts)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    default=_json_default, allow_nan=False) + '\n')
    temporary.replace(path)


def create_node(args, motion):
    # Dry-run and unit tests do not import ROS, open sockets, or publish commands.
    from mc_core_interface.msg import MechUnitCmd, MechUnitCmdArray
    from pingpong_controller.mocap_calibration import MocapCalibrationNode

    class AutoNode(MocapCalibrationNode):
        def __init__(self):
            self.sampling_since = None
            self.last_rejection = ''
            super().__init__(
                udp_port=args.udp_port, robot_xml=args.robot_xml,
                rigid_body_slot=args.rigid_body_slot, rigid_body_id=args.rigid_body_id,
                ball_marker_id=1, robot_source='fk', position_unit=args.position_unit,
                timestamp_unit=args.timestamp_unit, stability_window_s=args.stability_window,
                max_mocap_std_m=args.max_mocap_std_mm / 1000,
                max_mocap_rotation_rms_deg=args.max_mocap_rotation_rms_deg,
                max_mocap_mean_error_m=args.max_mocap_mean_error_mm / 1000,
                handeye_method=args.handeye_method)
            self.command_pub = self.create_publisher(MechUnitCmdArray, args.command_topic, 1)
            self.command_timer = self.create_timer(1 / args.publish_hz, self._control_tick)
            self.graph_checked_at = 0.

        def _joint_state_callback(self, msg):
            mapping = dict(zip(msg.name, msg.position))
            if any(name not in mapping for name in JOINT_NAMES):
                return
            super()._joint_state_callback(msg)
            motion.feedback([mapping[name] for name in JOINT_NAMES], time.monotonic())

        def _extract_right_arm_joints(self):
            if self.sampling_since is not None:
                q = motion.stable_mean(time.monotonic(), self.sampling_since)
                if q is None:
                    self.last_rejection = 'Robot joint feedback moved during the mocap window'
                return q
            with motion.lock:
                return None if motion.q is None else motion.q.copy()

        def receive_stable_mocap_pose(self, timeout_s=.5):
            result, reason = super().receive_stable_mocap_pose(args.sample_timeout)
            self.last_rejection = reason
            return result, reason

        def _control_tick(self):
            try:
                now = time.monotonic()
                if now - self.graph_checked_at >= 1.:
                    self.graph_checked_at = now
                    if self.count_publishers(args.command_topic) > 1:
                        raise RuntimeError('Another publisher is using ' + args.command_topic)
                    if motion.armed and self.command_pub.get_subscription_count() == 0:
                        raise RuntimeError('Robot command subscriber disconnected')
                q = motion.tick(now)
                if q is None:
                    return
                cmd = MechUnitCmd()
                cmd.mu_idx = args.right_arm_mu_idx
                cmd.use_jnt = True
                cmd.jnt_pos = q.tolist()
                cmd.psi = 0.
                msg = MechUnitCmdArray()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.cmds = [cmd]
                self.command_pub.publish(msg)
            except Exception as exc:
                with motion.lock:
                    motion.fault = str(exc)

    return AutoNode()


class LiveSequence:
    def __init__(self, node, motion, args):
        self.node, self.motion, self.args = node, motion, args
        self.events = []
        self.arrived_since = None
        self.index = 0
        self.status = 'running'
        self.resume_history = []
        self.accepted_point_indices = []
        self.skipped_points = []
        self.completed_point = 0
        self.supplemental_waypoints = []
        self.supplemental_workspace = None
        self.supplemental_stop_reason = None

    @property
    def total_points(self):
        return 50 + len(self.supplemental_waypoints)

    def point_label(self, index=None):
        index = self.index if index is None else index
        return (f'Point {index}/50' if index <= 50 else
                f'Supplemental point {index - 50}/{self.args.max_supplemental_points} '
                f'(valid {self.node.sample_count}/{self.args.target_samples})')

    def restore(self, saved):
        restore_session(self.node, saved)
        self.events = list(saved.get('attempts', []))
        self.accepted_point_indices = list(saved['accepted_point_indices'])
        self.skipped_points = list(saved['skipped_points'])
        self.completed_point = saved['completed_point']
        self.supplemental_waypoints = list(saved['supplemental_waypoints'])
        self.supplemental_workspace = saved.get('supplemental_workspace')
        self.index = self.completed_point
        self.resume_history = list(saved.get('resume_history', []))
        self.resume_history.append(dict(source=saved['_resume_path'],
                                        restored_count=self.node.sample_count,
                                        restored_skipped_count=len(self.skipped_points),
                                        next_point=self.completed_point + 1,
                                        original_options=saved.get('options', {})))
        self.node.get_logger().info(
            f'Restored {self.node.sample_count} measured pose pairs, '
            f'{len(self.skipped_points)} skipped points; '
            f'continuing at route point {self.completed_point + 1}')
        self.save_progress()

    def check(self):
        self.motion.check(time.monotonic())

    def wait(self, seconds):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.check()
            time.sleep(.02)

    def move(self, index, target):
        self.check()
        self.index = index
        self.arrived_since = None
        duration = self.motion.move_to(target, time.monotonic())
        self.node.get_logger().info(f'{self.point_label(index)}: slow move, planned duration {duration:.1f} s')
        # Timer delays stretch the trajectory rather than skipping references.
        deadline = time.monotonic() + duration * 3 + self.args.arrival_timeout
        while True:
            now = time.monotonic()
            self.check()
            with self.motion.lock:
                moving = self.motion.trajectory.active
            if not moving:
                return
            if now > deadline:
                raise RuntimeError(f'Point {index}: motion did not finish before timeout')
            time.sleep(.02)

    def settle(self):
        deadline = time.monotonic() + self.args.arrival_timeout
        while time.monotonic() < deadline:
            now = time.monotonic()
            self.check()
            if self.motion.ready_to_sample(now):
                if self.arrived_since is None:
                    self.arrived_since = now
                if now - self.arrived_since >= self.args.settle_seconds and self.motion.stable_mean(now) is not None:
                    diagnostic = self.motion.diagnostics(now)
                    error = diagnostic['max_target_error_deg']
                    self.node.get_logger().info(
                        f'{self.point_label()}: joints stable, max target error '
                        f'{error:.3f} deg; recording measured joints/FK')
                    return
            else:
                self.arrived_since = None
            time.sleep(.02)
        diagnostic = self.motion.diagnostics(time.monotonic())
        raise PointRejected(
            f'Point {self.index}: joint settling timed out '
            f'(mode={self.args.arrival_mode}, '
            f'max target error={diagnostic["max_target_error_deg"]} deg, '
            f'joint window range={diagnostic["max_joint_window_range_deg"]} deg)')

    def sample(self, index, attempt):
        self.check()
        self.node.get_logger().info(f'{self.point_label(index)}: sampling attempt {attempt}/{self.args.max_attempts}')
        self.node.sampling_since = time.monotonic()
        previous_count = self.node.sample_count
        accepted = False
        try:
            accepted = self.node.record_sample()
            self.check()
            if accepted and self.motion.stable_mean(time.monotonic(), self.node.sampling_since) is None:
                accepted = False
                self.node.last_rejection = 'Robot joints moved during sampling'
            return accepted
        except BaseException:
            accepted = False
            raise
        finally:
            if not accepted:
                # Roll back all five paired lists, including interrupted attempts.
                for values in (self.node.session.samples, self.node.session.mocap_points,
                               self.node.session.robot_points, self.node.session.mocap_poses,
                               self.node.session.robot_poses):
                    del values[previous_count:]
            self.node.sampling_since = None

    def checkpoint(self, index, attempt, accepted):
        self.events.append(dict(point=index, attempt=attempt, accepted=accepted,
                                kind='original' if index <= 50 else 'supplemental',
                                reason='' if accepted else self.node.last_rejection))
        if accepted:
            self.accepted_point_indices.append(index)
            self.completed_point = index
        self.save_progress()

    def skip(self, index, reason, attempts):
        self.check()
        self.skipped_points.append(dict(point=index, reason=reason, attempts=attempts,
                                        kind='original' if index <= 50 else 'supplemental'))
        self.completed_point = index
        self.node.get_logger().warn(
            f'{self.point_label(index)} SKIPPED: {reason}; continuing to the next point')
        self.save_progress()

    def save_progress(self, error=None):
        write_json(Path(self.args.output).with_suffix('.progress.json'), dict(
            schema_version=4,
            status=self.status, error=error, waypoint_file=str(Path(self.args.waypoints).resolve()),
            waypoints_sha256=hashlib.sha256(Path(self.args.waypoints).read_bytes()).hexdigest(),
            current_point=self.index, accepted_count=self.node.sample_count,
            completed_point=self.completed_point, next_point=self.completed_point + 1,
            accepted_point_indices=self.accepted_point_indices,
            skipped_count=len(self.skipped_points), skipped_points=self.skipped_points,
            expected_count=self.total_points, base_waypoint_count=50,
            target_sample_count=self.args.target_samples,
            supplemental_waypoints=self.supplemental_waypoints,
            supplemental_workspace=self.supplemental_workspace,
            supplemental_stop_reason=self.supplemental_stop_reason,
            options=vars(self.args), attempts=self.events,
            resume_history=self.resume_history,
            motion=self.motion.diagnostics(time.monotonic()),
            samples=[asdict(sample) for sample in self.node.session.samples]))

    def retry_wait(self):
        self.wait(self.args.retry_seconds)

    def collect_supplemental(self, planner):
        if self.node.sample_count >= self.args.target_samples or self.args.max_supplemental_points == 0:
            return
        self.check()
        with self.motion.lock:
            reference = (self.supplemental_workspace or {}).get('reference_rotation_sim')
            planner.set_reference_rotation(
                planner.pose(self.motion.command)[:3, :3] if reference is None else reference)
            self.supplemental_workspace = planner.metadata()
            # Supplemental motion is restricted to the V172 box from its
            # starting posture onward. Do not invent a route into that box
            # from a posture outside the user's authorized supplemental area.
            if not planner.contains(self.motion.command) or not planner.contains(self.motion.q):
                self.supplemental_stop_reason = 'Starting measured/reference pose is outside the V172 workspace'
                self.node.get_logger().warn(self.supplemental_stop_reason + '; keeping collected samples')
                for row in self.supplemental_waypoints:
                    if row['point_index'] > self.completed_point:
                        self.skip(row['point_index'], self.supplemental_stop_reason, 0)
                self.save_progress()
                return
            self.motion.workspace = planner
        self.node.get_logger().info(
            f'Supplemental collection: {self.node.sample_count}/{self.args.target_samples} valid; '
            f'up to {self.args.max_supplemental_points} extra points in the V172 sim-frame workspace')
        while self.node.sample_count < self.args.target_samples:
            self.check()
            index = self.completed_point + 1
            if index <= self.total_points:
                # An interrupted supplemental motion keeps its exact target.
                row = self.supplemental_waypoints[index - 51]
            else:
                if len(self.supplemental_waypoints) >= self.args.max_supplemental_points:
                    self.supplemental_stop_reason = 'Supplemental point budget exhausted'
                    break
                with self.motion.lock:
                    start = self.motion.command.copy()
                row = planner.next_candidate(start, self.supplemental_waypoints,
                                             self.node.session.samples)
                if row is None:
                    self.supplemental_stop_reason = 'No distinct intermediate point with an in-workspace path remains'
                    break
                row['point_index'] = index
                self.supplemental_waypoints.append(row)
                # Persist the target before movement, so Ctrl-C cannot make
                # resume silently choose a different route or duplicate data.
                self.save_progress()
            collect_point(index, np.asarray(row['joint_positions_rad'], dtype=float),
                          self, self.args.max_attempts)
        if self.node.sample_count >= self.args.target_samples:
            self.supplemental_stop_reason = 'Target accepted sample count reached'
        else:
            self.node.get_logger().warn(
                f'Supplemental collection finished with {self.node.sample_count}/'
                f'{self.args.target_samples} valid: {self.supplemental_stop_reason}; '
                'solving with the accepted measured pose pairs')
        self.save_progress()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--waypoints', default=str(DEFAULT_WAYPOINTS))
    parser.add_argument('--robot-xml', default='src/pingpong_moz1/pingpong_controller/models/moz1_pd.xml')
    parser.add_argument('--dry-run', action='store_true', help='Validate all 50 points and print timings; no ROS or motion')
    parser.add_argument('--output', default='mocap_calibration_auto.txt')
    parser.add_argument('--resume', nargs='?', const='', default=None, metavar='PROGRESS_JSON',
                        help='Restore measured samples and continue; defaults to OUTPUT.progress.json')
    parser.add_argument('--udp-port', type=int, default=65501)
    parser.add_argument('--rigid-body-slot', type=int, default=0)
    parser.add_argument('--rigid-body-id', type=int, default=0)
    parser.add_argument('--command-topic', default='/mx_mix_command')
    parser.add_argument('--right-arm-mu-idx', type=int, default=3)
    parser.add_argument('--publish-hz', type=float, default=100.)
    parser.add_argument('--speed-deg-s', type=float, default=5.)
    parser.add_argument('--accel-deg-s2', type=float, default=5.)
    parser.add_argument('--min-move-seconds', type=float, default=6.)
    parser.add_argument('--settle-seconds', type=float, default=3.)
    parser.add_argument('--joint-stable-seconds', type=float, default=1.)
    parser.add_argument('--joint-stable-range-deg', type=float, default=.10)
    parser.add_argument('--arrival-mode', choices=('stable', 'target'), default='stable',
                        help='stable: sample measured FK once still; target: also require target tolerance')
    parser.add_argument('--arrival-tolerance-deg', type=float, default=.30,
                        help='Target tolerance used only with --arrival-mode target')
    parser.add_argument('--arrival-timeout', type=float, default=30.)
    parser.add_argument('--feedback-timeout', type=float, default=.25)
    parser.add_argument('--max-tracking-error-deg', type=float, default=5.)
    parser.add_argument('--startup-timeout', type=float, default=15.)
    parser.add_argument('--max-attempts', type=int, default=30,
                        help='Sampling attempts per point before skipping it and continuing')
    parser.add_argument('--target-samples', type=int, default=50,
                        help='Accepted pose-pair target after the original 50 points (minimum 50)')
    parser.add_argument('--max-supplemental-points', type=int, default=50,
                        help='Maximum extra intermediate points used to replace rejected samples; 0 disables')
    parser.add_argument('--supplemental-margin-mm', type=float, default=10.,
                        help='Inset of supplemental targets from the V172 position-box boundaries')
    parser.add_argument('--supplemental-joint-jitter-deg', type=float, default=1.,
                        help='Per-joint perturbation for intermediate points and local one/two-joint refinements')
    parser.add_argument('--supplemental-rotation-deg', type=float, default=2.,
                        help='Maximum end-effector orientation perturbation from an intermediate posture')
    parser.add_argument('--retry-seconds', type=float, default=1.)
    parser.add_argument('--sample-timeout', type=float, default=.5)
    parser.add_argument('--stability-window', type=float, default=.1)
    parser.add_argument('--max-mocap-std-mm', type=float, default=.25)
    parser.add_argument('--max-mocap-rotation-rms-deg', type=float, default=.20)
    parser.add_argument('--max-mocap-mean-error-mm', type=float, default=1.)
    parser.add_argument('--position-unit', choices=('auto', 'm', 'mm'), default='auto')
    parser.add_argument('--timestamp-unit', choices=('s', 'ms', 'us', 'ns'), default='s')
    parser.add_argument('--handeye-method', choices=('PARK', 'TSAI', 'HORAUD', 'ANDREFF', 'DANIILIDIS'), default='PARK')
    return parser


def main(argv=None):
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ['--']:
        arguments.pop(0)
    args = parser.parse_args(arguments)
    for name, value in vars(args).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not np.isfinite(value) or value < 0 or (value == 0 and name not in (
                    'rigid_body_slot', 'rigid_body_id', 'right_arm_mu_idx', 'max_supplemental_points')):
                parser.error(f'--{name.replace("_", "-")} must be positive and finite')
    if args.udp_port > 65535 or args.publish_hz < 10:
        parser.error('UDP port must be <=65535 and publish rate must be >=10 Hz')
    if args.target_samples < 50:
        parser.error('--target-samples must be at least 50; the original 50 points are always processed')
    if args.max_supplemental_points > 500:
        parser.error('--max-supplemental-points must be <=500')
    if args.target_samples > 50 + args.max_supplemental_points:
        parser.error('--target-samples exceeds the original route plus the supplemental point budget')
    if args.supplemental_joint_jitter_deg > 2. or args.supplemental_rotation_deg > 4.:
        parser.error('Supplemental joint jitter must be <=2 deg and rotation perturbation <=4 deg')
    if args.sample_timeout <= args.stability_window:
        parser.error('--sample-timeout must exceed --stability-window')
    if args.arrival_timeout <= max(args.settle_seconds, args.joint_stable_seconds):
        parser.error('--arrival-timeout must exceed both settling and stability windows')
    if Path(args.output).suffix.lower() != '.txt':
        parser.error('--output must end in .txt (metadata/progress use separate JSON files)')
    resume_data = None
    try:
        document, waypoints = load_waypoints(args.waypoints)
        limits = joint_limits(args.robot_xml)
        if np.any(waypoints < limits[0]) or np.any(waypoints > limits[1]):
            raise ValueError('One or more recorded waypoints exceed the robot joint limits')
        from pingpong_replay.cartesian_fk import RightArmFK
        fk = RightArmFK(args.robot_xml, joint_names=JOINT_NAMES)
        for row, q in zip(document['waypoints'], waypoints):
            if not np.allclose(fk.ee_pose(q), row['T_waist03_right_ee_site'], atol=1e-7, rtol=0):
                raise ValueError(f'Robot FK differs from recorded sample {row["source_sample_index"]}')
        from pingpong_controller.mocap_calibration_supplement import SupplementalPlanner
        planner = SupplementalPlanner(fk, limits, args)
        if args.max_supplemental_points:
            planner.prepare(waypoints, args.max_supplemental_points)
        if args.resume is not None:
            resume_path = args.resume or str(Path(args.output).with_suffix('.progress.json'))
            resume_data = load_resume(resume_path, args, fk)
    except (OSError, KeyError, ValueError, ImportError) as exc:
        parser.error(str(exc))
    print(f'Loaded 50 accepted postures from {document["source_created_at_utc"]}', flush=True)
    print(f'Quintic motion: <= {args.speed_deg_s:g} deg/s, <= {args.accel_deg_s2:g} deg/s^2; '
          f'minimum move {args.min_move_seconds:g} s, settle {args.settle_seconds:g} s; '
          f'up to {args.max_attempts} attempts/point', flush=True)
    print(f'Sampling mode: {args.arrival_mode}; calibration always uses measured joint feedback/FK.', flush=True)
    print(f'Supplemental collection: target {args.target_samples} accepted samples, '
          f'maximum {args.max_supplemental_points} extra intermediate points; '
          f'V172 sim-frame box {planner.lower.tolist()} .. {planner.upper.tolist()}, '
          f'target inset {args.supplemental_margin_mm:g} mm, '
          f'orientation perturbation <= {args.supplemental_rotation_deg:g} deg; '
          'supplemental attitude stays within 4 deg of its starting reference; '
          f'{len(planner.candidates)} candidates', flush=True)
    start_index = 1 if resume_data is None else resume_data['completed_point'] + 1
    if resume_data is not None:
        print(f'Resume: {len(resume_data["samples"])} accepted samples, '
              f'{len(resume_data["skipped_points"])} skipped; next route point {start_index}', flush=True)
    if args.dry_run:
        total = 0.
        for index, target in enumerate(waypoints[start_index - 1:], start_index):
            duration = None if index == start_index else segment_duration(waypoints[index - 2], target, args)
            print(f'{index:02d}/50: move=' + ('from live feedback' if duration is None else f'{duration:.2f}s')
                  + '  q(deg)=' + np.array2string(np.rad2deg(target), precision=2))
            total += 0 if duration is None else duration
        remaining_original = max(0, 51 - start_index)
        if resume_data is not None:
            for row in resume_data['supplemental_waypoints']:
                if row['point_index'] >= start_index:
                    print(f'Saved supplemental point {row["point_index"]}: q(deg)='
                          + np.array2string(np.rad2deg(row['joint_positions_rad']), precision=2))
        print(f'Original-route inter-point motion + settling: '
              f'{(total + remaining_original * args.settle_seconds) / 60:.1f} min; '
              'initial move, sampling, retries and supplemental points add time. '
              'Supplemental paths are selected from live feedback. Dry-run: no ROS, UDP or commands.')
        return 0
    return run_live(args, waypoints, resume_data, planner)


def run_live(args, waypoints, resume_data=None, planner=None):
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.signals import SignalHandlerOptions
    from pingpong_controller.mocap_transform import save_calibration_matrix

    motion = ArmMotion(args, joint_limits(args.robot_xml))
    node = executor = spin_thread = sequence = None
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    try:
        node = create_node(args, motion)
        executor = SingleThreadedExecutor()
        executor.add_node(node)

        def spin():
            try:
                executor.spin()
            except Exception as exc:
                with motion.lock:
                    motion.fault = f'ROS executor stopped: {exc}'

        spin_thread = threading.Thread(target=spin, daemon=True)
        spin_thread.start()
        sequence = LiveSequence(node, motion, args)
        if resume_data is not None:
            sequence.restore(resume_data)
        deadline = time.monotonic() + args.startup_timeout
        print('Waiting for fresh right-arm feedback, command subscriber and tracked rigid body...', flush=True)
        ready_since = None
        while True:
            now = time.monotonic()
            with motion.lock:
                if motion.fault:
                    raise RuntimeError(motion.fault)
                feedback_ready = motion.q is not None and now - motion.received_at <= args.feedback_timeout
                history = [(t, q) for t, q in motion.history if t >= now - args.joint_stable_seconds]
                feedback_still = (len(history) >= 3
                                  and history[-1][0] - history[0][0] >= .8 * args.joint_stable_seconds
                                  and np.max(np.ptp([q for _, q in history], axis=0)) <= np.deg2rad(args.joint_stable_range_deg))
            packet = node.receive_mocap_measurement()
            mocap_ready = packet is not None and packet.orientation_wxyz is not None
            if node.count_publishers(args.command_topic) > 1:
                raise RuntimeError('Stop other command publishers on ' + args.command_topic)
            if feedback_ready and feedback_still and mocap_ready and node.command_pub.get_subscription_count() > 0:
                ready_since = now if ready_since is None else ready_since
                if now - ready_since >= 1.:
                    break
            else:
                ready_since = None
            if time.monotonic() > deadline:
                raise RuntimeError('Startup timeout: check joint feedback, command subscriber and rigid-body UDP')
            time.sleep(.02)
        motion.arm(time.monotonic())
        collect_sequence(waypoints, sequence, args.max_attempts,
                         start_index=sequence.completed_point + 1)
        if planner is None:
            from pingpong_replay.cartesian_fk import RightArmFK
            from pingpong_controller.mocap_calibration_supplement import SupplementalPlanner
            planner = SupplementalPlanner(RightArmFK(args.robot_xml, joint_names=JOINT_NAMES),
                                          joint_limits(args.robot_xml), args)
            planner.prepare(waypoints, args.max_supplemental_points)
        sequence.collect_supplemental(planner)
        valid_count = node.sample_count
        skipped_count = len(sequence.skipped_points)
        if (sequence.completed_point != sequence.total_points
                or valid_count + skipped_count != sequence.total_points):
            raise RuntimeError('Waypoint accounting is incomplete; progress has been saved')
        node.get_logger().info(
            f'Route finished: {valid_count} valid measured pose pairs, '
            f'{skipped_count} skipped points, {len(sequence.supplemental_waypoints)} '
            f'supplemental points processed')
        if valid_count < 3:
            raise RuntimeError(
                f'Route finished but only {valid_count} valid pose pairs remain; '
                'calibration requires at least 3. Progress and skip reasons were saved.')
        sequence.check()
        transform, rmse, metrics = node.compute_calibration()
        if not np.all(np.isfinite(transform)) or not np.isfinite(rmse):
            raise RuntimeError('Calibration solver returned non-finite results')
        output = Path(args.output)
        metadata = node.build_metadata(transform, rmse, metrics, str(output), str(output.with_suffix('.json')))
        metadata['automatic_calibration'] = dict(waypoint_file=str(Path(args.waypoints).resolve()),
                                                  options=vars(args), attempts=sequence.events,
                                                  resume_history=sequence.resume_history,
                                                  waypoint_count=sequence.total_points,
                                                  base_waypoint_count=50, accepted_count=valid_count,
                                                  target_sample_count=args.target_samples,
                                                  target_reached=valid_count >= args.target_samples,
                                                  supplemental_waypoints=sequence.supplemental_waypoints,
                                                  supplemental_workspace=sequence.supplemental_workspace,
                                                  supplemental_stop_reason=sequence.supplemental_stop_reason,
                                                  completed_point=sequence.completed_point,
                                                  accepted_point_indices=sequence.accepted_point_indices,
                                                  skipped_count=skipped_count,
                                                  skipped_points=sequence.skipped_points)
        for sample, point in zip(metadata['samples'], sequence.accepted_point_indices):
            sample['waypoint_index'] = point
            sample['waypoint_kind'] = 'original' if point <= 50 else 'supplemental'
        write_json(output.with_suffix('.json'), metadata)
        save_calibration_matrix(output, transform,
                                header=f'T_robot_mocap: mocap -> waist03; {valid_count} valid automatic samples, {skipped_count} skipped waypoints')
        sequence.status = 'complete'
        sequence.save_progress()
        print(f'Calibration complete: {output.resolve()} '
              f'({valid_count}/{args.target_samples} valid, {skipped_count} skipped, '
              f'{sequence.total_points} waypoints processed; '
              f'RMSE {rmse * 1000:.2f} mm)', flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        motion.disarm()
        if sequence is not None:
            sequence.status = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
            sequence.save_progress(error=str(exc) or 'Ctrl-C')
        print(f'Automatic calibration stopped: {str(exc) or "Ctrl-C"}', file=sys.stderr, flush=True)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1
    finally:
        motion.disarm()
        if executor is not None:
            executor.shutdown(timeout_sec=2.)
        if spin_thread is not None:
            spin_thread.join(timeout=2.)
        if node is not None:
            node.socket.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
