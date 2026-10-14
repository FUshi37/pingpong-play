"""Independent ROS/FK monitor for the explicitly authorized BUILTIN test harness.

Never publishes position targets. May request StopAllMU when its test driver
dies/stalls, state becomes invalid, or actual motion crosses the real envelope.
The native service ACK is logged separately from physical stopping evidence.
This monitor does not install protection into the deployment controller.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import signal
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src/pingpong_moz1'))
from pingpong_controller.fplc_deployment import FPLC_PROFILES, FPLC_REAL_LOWER_M, FPLC_REAL_UPPER_M


def heartbeat_expired(now, last, timeout=.15):
    return not isinstance(last, (int, float)) or not math.isfinite(last) or last > now or now-last > timeout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--profile', choices=FPLC_PROFILES, required=True)
    args = parser.parse_args()
    # Heavy/ROS imports are only available in this explicitly launched child.
    import numpy as np
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from mc_core_interface.msg import MechUnitCmdArray, RobotStatus
    from mc_core_interface.srv import RobotCmdService
    from pingpong_controller.fplc_policy import FPLCPolicyController
    from pingpong_controller.rl_policy import RIGHT_ARM_JOINTS
    controller = FPLCPolicyController(observation_profile=args.profile, dt=.005)
    output = args.output
    thresholds = json.loads((output/'case_manifest.json').read_text())['thresholds']
    rclpy.init()
    node = Node('predeploy_independent_monitor_'+str(time.time_ns()))
    client = node.create_client(RobotCmdService, '/robot_cmd_service')
    stream = (output/'ros_samples.jsonl').open('a', buffering=1)
    last_feedback = time.monotonic()
    failure = None
    future = None
    stop_ack_written = False
    stopped_by_signal = False
    def signal_stop(signum, frame):
        nonlocal stopped_by_signal
        stopped_by_signal = True
    signal.signal(signal.SIGTERM, signal_stop)
    signal.signal(signal.SIGINT, signal_stop)
    def write(kind, **data):
        row = dict(kind=kind, receiptMonotonicS=time.monotonic(), receiptWallS=time.time(), **data)
        stream.write(json.dumps(row, allow_nan=False)+'\n')
    def fail(reason):
        nonlocal failure
        if failure is not None:
            return
        failure = dict(reason=reason, triggerMonotonicS=time.monotonic(), nativeStopAcknowledged=False)
        (output/'monitor_failure.json').write_text(json.dumps(failure, indent=2))
        write('monitor_fault', reason=reason)
    def joint(msg):
        nonlocal last_feedback
        try:
            indices = [list(msg.name).index(name) for name in RIGHT_ARM_JOINTS]
            q = np.array([msg.position[i] for i in indices], float)
            dq = np.array([msg.velocity[i] for i in indices], float)
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(dq)):
                raise ValueError('nonfinite q/dq')
            pose = controller._pose(q, with_jacobian=False)
            xyz, rp = pose[2], np.rad2deg(pose[3][:2])
            last_feedback = time.monotonic()
            write('joint', stampS=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
                  q=q.tolist(), dq=dq.tolist(), positionM=xyz.tolist(), relativeRpDeg=rp.tolist())
            if (output/'monitor_armed').exists():
                if not controller.pose_is_safe(q, tuple(zip(FPLC_REAL_LOWER_M, FPLC_REAL_UPPER_M))):
                    fail('measured_joints_pose_outside_real_envelope')
                elif np.any(np.abs(dq) > np.deg2rad(thresholds['joint_velocity_limit_deg_s'])):
                    fail('measured_joint_speed_exceeded')
            reset_path = output/'supervised_reset.json'
            try:
                phase = json.loads(reset_path.read_text())
            except FileNotFoundError:
                phase = None
            if phase is not None:
                margin = np.deg2rad(thresholds['reset_joint_corridor_margin_deg'])
                lower = np.minimum(phase['startQ'], phase['targetQ'])-margin
                upper = np.maximum(phase['startQ'], phase['targetQ'])+margin
                if np.any(q < controller.arm_q_low) or np.any(q > controller.arm_q_high):
                    fail('reset_measured_joint_limit_violation')
                elif np.any(q < lower) or np.any(q > upper):
                    fail('reset_left_start_target_joint_corridor')
                elif np.max(np.abs(dq)) > np.deg2rad(thresholds['reset_maximum_measured_speed_deg_s']):
                    fail('reset_measured_speed_exceeded')
        except Exception as exc:
            write('invalid_joint', error=str(exc))
            if (output/'monitor_armed').exists() or (output/'monitor_watchdog_armed').exists():
                fail('incomplete_or_nonfinite_joint_state')
    def command(msg):
        for c in msg.cmds:
            if c.mu_idx == 3 and c.use_jnt:
                q = list(c.jnt_pos)
                if len(q) != 7 or not all(math.isfinite(v) for v in q):
                    fail('nonfinite_or_malformed_actuator_command')
                    write('invalid_command')
                else:
                    write('command', stampS=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
                          q=q, muIdx=c.mu_idx)
    def status(msg):
        write('robot_status', stampS=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
            controlMode=list(msg.control_mode), taskMode=msg.task_mode,
            motionStatus=msg.motion_status, safetyStatus=msg.safety_status,
            motorStatus=list(msg.motor_status))
    node.create_subscription(JointState, '/joint_states', joint, 20)
    node.create_subscription(MechUnitCmdArray, '/mx_mix_command', command, 20)
    node.create_subscription(RobotStatus, '/robot_states', status, 20)
    (output/'observer_ready').touch()
    idle_started = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.005)
            now = time.monotonic()
            armed = (output/'monitor_armed').exists() or (output/'monitor_watchdog_armed').exists()
            if armed:
                try:
                    last = float((output/'heartbeat').read_text())
                except (ValueError, OSError):
                    last = None
                if heartbeat_expired(now, last):
                    fail('test_driver_heartbeat_missing_or_stalled')
                if now-last_feedback > thresholds['monitor_feedback_timeout_s']:
                    fail('raw_joint_feedback_missing')
            if (output/'request_native_stop').exists():
                fail('test_suite_failed_request_native_stop')
            if stopped_by_signal and armed:
                fail('monitor_interrupted_while_armed')
            if failure is not None and future is None:
                if client.service_is_ready():
                    req = RobotCmdService.Request(cmd='StopAllMU', data='{}')
                    future = client.call_async(req)
                    write('native_stop_request', cmd='StopAllMU')
                elif now-failure['triggerMonotonicS'] > .3:
                    failure['nativeStopError'] = 'robot_cmd_service unavailable'
                    (output/'monitor_failure.json').write_text(json.dumps(failure, indent=2))
            if future is not None and future.done() and not stop_ack_written:
                stop_ack_written = True
                try:
                    response = future.result()
                    failure['nativeStopAcknowledged'] = bool(response.success)
                    failure['nativeStopResponse'] = response.data
                    write('native_stop_response', success=bool(response.success), data=response.data)
                except Exception as exc:
                    failure['nativeStopError'] = str(exc)
                (output/'monitor_failure.json').write_text(json.dumps(failure, indent=2))
            # Survives parent death long enough to request stop and record tail.
            dead_parent_tail_done = bool(failure and failure['reason'] == 'test_driver_heartbeat_missing_or_stalled'
                                         and now-failure['triggerMonotonicS'] >= 3.)
            if (output/'observer_finish').exists() or dead_parent_tail_done or stopped_by_signal:
                break
            if not armed and now-idle_started > 60:
                fail('observer_never_armed')
                break
    finally:
        write('observer_exit', failure=failure)
        stream.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
