from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import threading
import time
from typing import Iterable

import numpy as np
import rclpy
from mc_core_interface.msg import MechUnitCmd, MechUnitCmdArray
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState


RIGHT_ARM_JOINTS = tuple(f"RightArm-{i}" for i in range(7))


class PlantSafetyError(RuntimeError):
    """Raised when continuing HIL control would violate a fail-closed check."""


@dataclass(frozen=True)
class JointSample:
    sequence: int
    source_stamp_ns: int
    receive_stamp_ns: int
    q: np.ndarray
    dq: np.ndarray
    velocity_source: str


@dataclass(frozen=True)
class PlantConfig:
    joint_topic: str = "/joint_states"
    command_topic: str = "/mx_mix_command"
    right_arm_mu_idx: int = 0
    control_hz: float = 200.0
    state_timeout_s: float = 0.050
    command_watchdog_s: float = 0.010
    causal_guard_samples: int = 1
    causal_timeout_retries: int = 0
    allow_existing_command_publisher: bool = False
    read_only: bool = False


class RosArmPlant(Node):
    """Atomic JointState reader and fail-closed MechUnit command publisher.

    Header stamps are used only for ordering/deltas. They are never compared to
    the local clock because the measured source clock has a large fixed offset.
    JointState.effort is deliberately ignored: it is load/torque, not qdd.
    """

    def __init__(self, cfg: PlantConfig = PlantConfig()) -> None:
        super().__init__("pingpong_hil_arm_plant")
        self.cfg = cfg
        self._condition = threading.Condition()
        self._latest: JointSample | None = None
        self._previous: JointSample | None = None
        self._recent: deque[JointSample] = deque(maxlen=20_000)
        self._sequence = 0
        self._last_publish_ns = 0
        self._last_command: np.ndarray | None = None
        self._fault: str | None = None
        self._causal_timeout_recoveries = 0
        self._closed = False

        state_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._sub = self.create_subscription(JointState, cfg.joint_topic, self._on_state, state_qos)

        self._publisher = None
        if not cfg.read_only:
            # Create the publisher only after the ownership audit, so this node
            # cannot accidentally count itself or emit a command before arming.
            deadline = time.monotonic() + 1.0
            infos = []
            while time.monotonic() < deadline:
                infos = self.get_publishers_info_by_topic(cfg.command_topic)
                if infos:
                    break
                time.sleep(0.05)
            other_names = sorted({str(info.node_name) for info in infos if str(info.node_name) != self.get_name()})
            if other_names and not cfg.allow_existing_command_publisher:
                raise PlantSafetyError(
                    f"command topic {cfg.command_topic} already has publisher(s): {other_names}; "
                    "stop them before HIL training"
                )
            command_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
            )
            self._publisher = self.create_publisher(MechUnitCmdArray, cfg.command_topic, command_qos)
            self._watchdog = self.create_timer(max(0.002, cfg.command_watchdog_s / 2.0), self._on_watchdog)

    def wait_for_command_subscriber(self, timeout_s: float) -> int:
        if self._publisher is None:
            raise PlantSafetyError("command publisher is not armed")
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            count = int(self._publisher.get_subscription_count())
            if count > 0:
                return count
            time.sleep(0.02)
        raise PlantSafetyError(f"no matched subscriber on {self.cfg.command_topic}")

    @staticmethod
    def _source_stamp_ns(msg: JointState) -> int:
        return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)

    def _on_state(self, msg: JointState) -> None:
        positions = {str(name): float(msg.position[i]) for i, name in enumerate(msg.name) if i < len(msg.position)}
        if not all(name in positions and np.isfinite(positions[name]) for name in RIGHT_ARM_JOINTS):
            return
        q = np.asarray([positions[name] for name in RIGHT_ARM_JOINTS], dtype=np.float64)
        velocities = {str(name): float(msg.velocity[i]) for i, name in enumerate(msg.name) if i < len(msg.velocity)}
        direct = np.asarray([velocities.get(name, np.nan) for name in RIGHT_ARM_JOINTS], dtype=np.float64)
        source_ns = self._source_stamp_ns(msg)
        receive_ns = time.monotonic_ns()

        with self._condition:
            previous = self._latest
            finite_difference = None
            if previous is not None:
                dt = (source_ns - previous.source_stamp_ns) * 1e-9
                if not (1e-4 <= dt <= 0.1):
                    dt = (receive_ns - previous.receive_stamp_ns) * 1e-9
                finite_difference = (q - previous.q) / max(dt, 1e-4)
            if np.all(np.isfinite(direct)):
                if (
                    finite_difference is not None
                    and float(np.max(np.abs(direct))) < 1e-7
                    and float(np.max(np.abs(finite_difference))) > 1e-4
                ):
                    dq = finite_difference
                    velocity_source = "finite_difference_inconsistent_zero"
                else:
                    dq = direct
                    velocity_source = "joint_state"
            elif finite_difference is not None:
                dq = finite_difference
                velocity_source = "finite_difference"
            else:
                dq = np.zeros(7, dtype=np.float64)
                velocity_source = "zero_first_sample"
            self._sequence += 1
            sample = JointSample(self._sequence, source_ns, receive_ns, q, dq, velocity_source)
            self._previous, self._latest = previous, sample
            self._recent.append(sample)
            self._condition.notify_all()

    def collect_samples(self, count: int, timeout_s: float) -> list[JointSample]:
        """Collect every callback sample in a ring buffer (no latest-only loss)."""
        count = int(count)
        if count < 2 or count > self._recent.maxlen:
            raise ValueError(f"count must be in [2, {self._recent.maxlen}]")
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            self._recent.clear()
            while len(self._recent) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise PlantSafetyError(f"only received {len(self._recent)}/{count} JointState samples")
                self._condition.wait(remaining)
            return [self._copy_sample(sample) for sample in list(self._recent)[-count:]]

    def wait_for_state(self, timeout_s: float | None = None) -> JointSample:
        timeout_s = self.cfg.state_timeout_s if timeout_s is None else float(timeout_s)
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._latest is None:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise PlantSafetyError(f"no complete right-arm sample on {self.cfg.joint_topic}")
                self._condition.wait(remaining)
            return self._copy_sample(self._latest)

    def publish_and_wait(self, target_q: Iterable[float]) -> tuple[JointSample, JointSample]:
        if self.cfg.read_only:
            raise PlantSafetyError("read-only plant cannot publish commands")
        self.assert_healthy()
        before = self.wait_for_state()
        self.publish(target_q)
        guard_samples = max(1, int(self.cfg.causal_guard_samples))
        required_sequence = before.sequence + guard_samples
        # ROS topics provide no command ACK. In addition to source ordering,
        # require a half-period local receipt guard. The baseline was captured
        # immediately before publish, so the next ordered source sample arrives
        # about one period later; a full-period guard would phase-skip it and
        # unintentionally reduce a 200Hz loop to 100Hz.
        earliest_receive_ns = self._last_publish_ns + int(
            0.5 * guard_samples * 1e9 / max(self.cfg.control_hz, 1.0)
        )
        deadline = time.monotonic() + self.cfg.state_timeout_s
        timeout_count = 0
        with self._condition:
            while (
                self._latest is None
                or self._latest.sequence < required_sequence
                or self._latest.receive_stamp_ns < earliest_receive_ns
            ):
                if self._fault is not None:
                    raise PlantSafetyError(self._fault)
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    if timeout_count < max(0, int(self.cfg.causal_timeout_retries)):
                        timeout_count += 1
                        self._causal_timeout_recoveries += 1
                        deadline = time.monotonic() + self.cfg.state_timeout_s
                        continue
                    self._fault = "JointState causal wait timed out"
                    raise PlantSafetyError(self._fault)
                self._condition.wait(remaining)
            after = self._copy_sample(self._latest)
        if before.source_stamp_ns > 0 and after.source_stamp_ns <= before.source_stamp_ns:
            raise PlantSafetyError("JointState source stamp did not advance after command")
        return before, after

    def publish(self, target_q: Iterable[float]) -> None:
        if self._publisher is None:
            raise PlantSafetyError("command publisher is not armed")
        q = np.asarray(tuple(target_q), dtype=np.float64).reshape(-1)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise PlantSafetyError(f"invalid right-arm command shape/value: {q}")
        cmd = MechUnitCmd()
        cmd.mu_idx = int(self.cfg.right_arm_mu_idx)
        cmd.jnt_pos = [float(v) for v in q]
        cmd.use_jnt = True
        cmd.psi = 0.0
        msg = MechUnitCmdArray()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.cmds = [cmd]
        self._publisher.publish(msg)
        self._last_command = q.copy()
        self._last_publish_ns = time.monotonic_ns()

    def move_to_pose(
        self,
        target_q: Iterable[float],
        *,
        speed_deg_s: float,
        tolerance_deg: float,
        timeout_s: float,
    ) -> JointSample:
        """Low-speed reset ramp driven only by measured JointState feedback."""
        target = np.asarray(tuple(target_q), dtype=np.float64).reshape(7)
        sample = self.wait_for_state()
        command = sample.q.copy()
        max_step = np.deg2rad(float(speed_deg_s)) / max(self.cfg.control_hz, 1.0)
        tolerance = np.deg2rad(float(tolerance_deg))
        deadline = time.monotonic() + float(timeout_s)
        settled = 0
        while time.monotonic() < deadline:
            command += np.clip(target - command, -max_step, max_step)
            _before, sample = self.publish_and_wait(command)
            at_command = float(np.max(np.abs(target - command))) <= 1e-7
            at_feedback = float(np.max(np.abs(target - sample.q))) <= tolerance
            settled = settled + 1 if at_command and at_feedback else 0
            if settled >= 10:
                return sample
        error_deg = float(np.rad2deg(np.max(np.abs(target - sample.q))))
        raise PlantSafetyError(f"reset ramp timed out with {error_deg:.2f}deg max error")

    def _on_watchdog(self) -> None:
        if self._publisher is None or self._last_command is None or self._closed:
            return
        now_ns = time.monotonic_ns()
        latest = self._latest
        age_s = np.inf if latest is None else (now_ns - latest.receive_stamp_ns) * 1e-9
        if latest is None or age_s > self.cfg.state_timeout_s:
            if latest is not None:
                self.publish(latest.q)
            retry_limit_s = self.cfg.state_timeout_s * (
                1.0 + max(0, int(self.cfg.causal_timeout_retries))
            )
            if latest is None or age_s > retry_limit_s:
                self._fault = "JointState stream stale; watchdog holding last observed position"
            return
        if (now_ns - self._last_publish_ns) * 1e-9 >= self.cfg.command_watchdog_s:
            self.publish(self._last_command)

    def assert_healthy(self) -> None:
        latest = self.wait_for_state()
        age_s = (time.monotonic_ns() - latest.receive_stamp_ns) * 1e-9
        if age_s > self.cfg.state_timeout_s:
            raise PlantSafetyError(f"JointState age {age_s:.4f}s exceeds {self.cfg.state_timeout_s:.4f}s")
        if self._fault is not None:
            raise PlantSafetyError(self._fault)

    @property
    def causal_timeout_recoveries(self) -> int:
        return int(self._causal_timeout_recoveries)

    @staticmethod
    def _copy_sample(sample: JointSample) -> JointSample:
        return JointSample(
            sample.sequence,
            sample.source_stamp_ns,
            sample.receive_stamp_ns,
            sample.q.copy(),
            sample.dq.copy(),
            sample.velocity_source,
        )

    def close_plant(self) -> None:
        if self._closed:
            return
        if self._publisher is not None and self._latest is not None:
            # Leave the plant at an actual-position hold, never at an untracked
            # final policy target.
            for _ in range(3):
                self.publish(self._latest.q)
                time.sleep(1.0 / max(self.cfg.control_hz, 1.0))
        self._closed = True


class RosPlantRuntime:
    """Own rclpy lifecycle and spin callbacks while Gym step blocks."""

    def __init__(self, cfg: PlantConfig) -> None:
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()
        try:
            self.plant = RosArmPlant(cfg)
        except BaseException:
            if self._owns_context and rclpy.ok():
                rclpy.shutdown()
            raise
        # The plant has only two short callbacks (JointState + watchdog), so a
        # single-threaded executor is sufficient at 200 Hz.  More importantly,
        # Humble's MultiThreadedExecutor owns a mutable ``_futures`` list that
        # has failed after long HIL runs (``TypeError: Task is not iterable``),
        # which then starves JointState and trips the causal safety timeout.
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.plant)
        self.thread = threading.Thread(target=self.executor.spin, name="hil_ros_executor", daemon=True)
        self.thread.start()

    def close(self) -> None:
        self.plant.close_plant()
        self.executor.shutdown(timeout_sec=1.0)
        self.thread.join(timeout=1.0)
        self.plant.destroy_node()
        if self._owns_context and rclpy.ok():
            rclpy.shutdown()
