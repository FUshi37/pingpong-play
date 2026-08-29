"""Stateful joint-space trajectory generation for Movax RMP motion mode.

The acceleration QP in :mod:`recovered_rmp.core` is only one part of the
runtime planner.  ``ManipulatorReactiveMotionPlanner::Plan`` retains a
position, velocity, and acceleration state and advances it at every servo
tick.  This module reconstructs that stateful layer so a simulator can receive
the same three reference signals as the real low-level controller.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import JointLimits, RMPResult, solve_joint_rmp
from .estimator import ConstantAccelerationKalman
from .params import RMPParams

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class JointRMPCommand:
    """One RMP motion-mode command in joint coordinates (SI units)."""

    position: FloatArray
    velocity: FloatArray
    acceleration: FloatArray
    target_position: FloatArray
    target_velocity: FloatArray
    target_acceleration: FloatArray
    qp: RMPResult


class JointRMPPlanner:
    """Recovered stateful joint-target RMP trajectory generator.

    A position-only target is first differentiated by the same
    constant-acceleration Kalman structure found in ``mc_core``.  The resulting
    q/qd/qdd target is averaged through the configured target ring buffer, fed
    into the recovered acceleration QP, and integrated with the exact update
    visible in ``ManipulatorReactiveMotionPlanner::Plan``::

        q_next  = q + (qd + 0.5 * dt * qdd) * dt
        qd_next = qd + qdd * dt

    ``dt`` is the servo/RMP period, not necessarily the slower ROS publish
    period.  On the copied robot stack this is normally 1 ms while recorded
    targets arrive at roughly 5 ms intervals.
    """

    def __init__(
        self,
        limits: JointLimits,
        params: RMPParams,
        dt: float,
        *,
        use_safety_bounds: bool = True,
        use_jerk_limit: bool = False,
    ) -> None:
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.limits = limits
        self.params = params
        self.dt = float(dt)
        self.use_safety_bounds = bool(use_safety_bounds)
        self.use_jerk_limit = bool(use_jerk_limit)
        self.size = limits.q_min.size
        self._estimator = ConstantAccelerationKalman(
            self.size,
            self.dt,
            params.estimator_joint_process_var,
            params.estimator_joint_measure_var,
        )
        length = max(1, int(params.target_filter_length))
        self._target_q_buffer: deque[FloatArray] = deque(maxlen=length)
        self._target_qd_buffer: deque[FloatArray] = deque(maxlen=length)
        self._target_qdd_buffer: deque[FloatArray] = deque(maxlen=length)
        output_length = max(1, int(params.filter_length))
        self._output_q_buffer: deque[FloatArray] = deque(maxlen=output_length)
        self._output_qd_buffer: deque[FloatArray] = deque(maxlen=output_length)
        self._output_qdd_buffer: deque[FloatArray] = deque(maxlen=output_length)
        self.accepted_target = np.zeros(self.size, dtype=float)
        self.ticks_since_target_update = 0
        self.q = np.zeros(self.size, dtype=float)
        self.qd = np.zeros(self.size, dtype=float)
        self.qdd = np.zeros(self.size, dtype=float)
        self.initialized = False

    def reset(
        self,
        position: ArrayLike,
        *,
        velocity: ArrayLike | None = None,
        acceleration: ArrayLike | None = None,
        target_position: ArrayLike | None = None,
    ) -> None:
        """Anchor the planned trajectory to a measured joint state."""

        q = np.asarray(position, dtype=float).reshape(-1)
        if q.size != self.size:
            raise ValueError(f"position must have length {self.size}")
        qd = (
            np.zeros(self.size, dtype=float)
            if velocity is None
            else np.asarray(velocity, dtype=float).reshape(self.size)
        )
        qdd = (
            np.zeros(self.size, dtype=float)
            if acceleration is None
            else np.asarray(acceleration, dtype=float).reshape(self.size)
        )
        target = q if target_position is None else np.asarray(target_position, dtype=float).reshape(self.size)
        self.q = q.copy()
        self.qd = qd.copy()
        self.qdd = qdd.copy()
        self.accepted_target = np.clip(target, self.limits.q_min, self.limits.q_max)
        self.ticks_since_target_update = 0
        self._estimator.reset(self.accepted_target)
        self._target_q_buffer.clear()
        self._target_qd_buffer.clear()
        self._target_qdd_buffer.clear()
        # Reset in the binary initializes the complete circular window.  Doing
        # the same avoids a start-up window whose denominator changes by tick.
        for _ in range(max(1, int(self.params.target_filter_length))):
            self._target_q_buffer.append(self.accepted_target.copy())
            self._target_qd_buffer.append(np.zeros(self.size, dtype=float))
            self._target_qdd_buffer.append(np.zeros(self.size, dtype=float))
        self._output_q_buffer.clear()
        self._output_qd_buffer.clear()
        self._output_qdd_buffer.clear()
        for _ in range(max(1, int(self.params.filter_length))):
            self._output_q_buffer.append(q.copy())
            self._output_qd_buffer.append(qd.copy())
            self._output_qdd_buffer.append(qdd.copy())
        self.initialized = True

    def _filtered_target(
        self,
        position_target: ArrayLike,
        *,
        target_updated: bool,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        measurement = np.asarray(position_target, dtype=float).reshape(-1)
        if measurement.size != self.size:
            raise ValueError(f"position_target must have length {self.size}")
        if target_updated:
            # UpdateJointTarget first rate-limits a newly received ROS target.
            # Its allowance is the joint speed limit multiplied by the number
            # of elapsed 1 kHz Plan ticks, capped at 100 ticks.
            elapsed = min(self.ticks_since_target_update, 100) * self.dt
            maximum_delta = self.limits.qd_max * elapsed
            self.accepted_target += np.clip(
                measurement - self.accepted_target,
                -maximum_delta,
                maximum_delta,
            )
            self.accepted_target = np.clip(
                self.accepted_target,
                self.limits.q_min,
                self.limits.q_max,
            )
            q_target, qd_target, qdd_target = self._estimator.update(
                self.accepted_target
            )
            self.ticks_since_target_update = 0
        else:
            # Plan corrects the estimator only once per newly received target;
            # it predicts for 20 held-target ticks, then actively decelerates
            # the estimated target velocity just like JntStatesEstimator.
            if self.ticks_since_target_update <= 20:
                self._estimator.predict()
            else:
                self._estimator.decelerate(self.limits.qdd_max)
            q_target = self._estimator.position
            qd_target = self._estimator.velocity
            qdd_target = self._estimator.acceleration
        self.ticks_since_target_update += 1
        # TargetCmdFilter clips the estimated joint target speed before writing
        # q/qd/qdd into its three circular matrices.
        qd_target = np.clip(qd_target, -self.limits.qd_max, self.limits.qd_max)
        self._target_q_buffer.append(q_target)
        self._target_qd_buffer.append(qd_target)
        self._target_qdd_buffer.append(qdd_target)
        return (
            np.mean(np.stack(tuple(self._target_q_buffer)), axis=0),
            np.mean(np.stack(tuple(self._target_qd_buffer)), axis=0),
            np.mean(np.stack(tuple(self._target_qdd_buffer)), axis=0),
        )

    def step(
        self,
        position_target: ArrayLike,
        *,
        target_updated: bool = True,
    ) -> JointRMPCommand:
        """Advance one servo tick from a position-only mechanical-unit target.

        Set ``target_updated=True`` only on the first servo tick after a new
        upstream command arrives.  Remaining held-target ticks must use
        ``False`` to match ``ManipulatorReactiveMotionPlanner::Plan``.
        """

        if not self.initialized:
            self.reset(position_target, target_position=position_target)
        q_target, qd_target, qdd_target = self._filtered_target(
            position_target,
            target_updated=target_updated,
        )
        previous = self.qdd if self.use_jerk_limit else None
        if self.use_safety_bounds:
            # ``strict=False`` matches the runtime's emergency-braking fallback
            # when a sampled/replayed state already lies outside the invariant.
            from .core import build_joint_qp, compute_joint_reference, joint_acceleration_bounds, solve_box_qp

            reference = compute_joint_reference(
                self.q,
                self.qd,
                q_target,
                qd_target,
                qdd_target,
                self.limits.qdd_max,
                self.params,
            )
            lower, upper = joint_acceleration_bounds(
                self.q,
                self.qd,
                self.limits,
                self.dt,
                angle_gap=self.params.angle_gap,
                qdd_previous=previous,
                use_jerk_limit=self.use_jerk_limit,
                strict=False,
            )
            problem = build_joint_qp(reference, self.params, lower, upper)
            result = RMPResult(solve_box_qp(problem), problem, reference)
        else:
            # Useful as a calibration ablation; the normal hardware-equivalent
            # path keeps safety bounds enabled.
            unbounded = JointLimits(
                q_min=np.full(self.size, -np.inf),
                q_max=np.full(self.size, np.inf),
                qd_max=np.full(self.size, np.inf),
                qdd_max=self.limits.qdd_max,
                qddd_max=None,
            )
            result = solve_joint_rmp(
                self.q,
                self.qd,
                q_target,
                qd_target,
                qdd_target,
                unbounded,
                self.params,
                self.dt,
            )

        qdd = result.qdd
        q_next = self.q + (self.qd + 0.5 * self.dt * qdd) * self.dt
        qd_next = self.qd + self.dt * qdd
        self.q = q_next
        self.qd = qd_next
        self.qdd = qdd.copy()
        # The hardware does not send the raw integrated state.  Plan writes
        # q/qd/qdd into three rings and averages the complete filter_length
        # window before filling SysAxisControlCommand.
        self._output_q_buffer.append(q_next.copy())
        self._output_qd_buffer.append(qd_next.copy())
        self._output_qdd_buffer.append(qdd.copy())
        command_q = np.mean(np.stack(tuple(self._output_q_buffer)), axis=0)
        command_qd = np.mean(np.stack(tuple(self._output_qd_buffer)), axis=0)
        command_qdd = np.mean(np.stack(tuple(self._output_qdd_buffer)), axis=0)
        return JointRMPCommand(
            position=command_q,
            velocity=command_qd,
            acceleration=command_qdd,
            target_position=q_target.copy(),
            target_velocity=qd_target.copy(),
            target_acceleration=qdd_target.copy(),
            qp=result,
        )
