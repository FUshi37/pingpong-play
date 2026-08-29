"""Optional JAX/MJX backend for batched training.

The QP matrices are identical to :mod:`recovered_rmp.core`.  The fixed-iteration
projected-gradient solver is a JIT-friendly numerical substitute for the C++
ProxQP dependency; use the NumPy active-set solver when bit-level convergence
of the small box QP matters more than batching.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from .params import RMPParams


class JointRMPStateJax(NamedTuple):
    """Batched state used by :func:`joint_rmp_step_jax`."""

    # Raw integrated planner state.  The command_* fields below are the
    # filter_length-window averages actually sent to the low-level controller.
    q: Any
    qd: Any
    qdd: Any
    command_q: Any
    command_qd: Any
    command_qdd: Any
    accepted_target: Any
    ticks_since_target_update: Any
    target_x: Any
    target_covariance: Any
    target_q_buffer: Any
    target_qd_buffer: Any
    target_qdd_buffer: Any
    target_buffer_index: Any
    output_q_buffer: Any
    output_qd_buffer: Any
    output_qdd_buffer: Any
    output_buffer_index: Any


def _jax() -> tuple[Any, Any]:
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - optional training environment
        raise ImportError("the JAX backend requires jax and jaxlib") from exc
    return jax, jnp


def _clip_norm(value: Any, maximum: float) -> Any:
    _, jnp = _jax()
    norm = jnp.linalg.norm(value, axis=-1, keepdims=True)
    scale = jnp.minimum(1.0, maximum / jnp.maximum(norm, 1.0e-12))
    return value * scale


def task_reference_jax(
    error: Any,
    current_twist: Any,
    target_twist: Any,
    target_acceleration: Any,
    params: RMPParams,
    dt: float,
) -> tuple[Any, Any]:
    """JIT-compatible six-dimensional form of the recovered task policy."""

    _, jnp = _jax()
    e = jnp.concatenate(
        (
            _clip_norm(error[..., :3], params.max_linear_vel * dt * 200.0),
            _clip_norm(error[..., 3:], params.max_angular_vel * dt * 200.0),
        ),
        axis=-1,
    )
    linear_speed = jnp.linalg.norm(current_twist[..., :3], axis=-1, keepdims=True)
    angular_speed = jnp.linalg.norm(current_twist[..., 3:], axis=-1, keepdims=True)
    linear_ffd = params.linear_vel_ffd * jnp.where(
        linear_speed < 0.05, 0.25 + 15.0 * linear_speed, 1.0
    )
    angular_ffd = params.angular_vel_ffd * jnp.where(
        angular_speed < 0.35, 0.25 + (0.75 / 0.35) * angular_speed, 1.0
    )
    linear = (
        params.kp * e[..., :3]
        + params.kd * (linear_ffd * target_twist[..., :3] - current_twist[..., :3])
        + params.linear_acc_ffd * target_acceleration[..., :3]
    )
    angular = (
        params.kp * e[..., 3:]
        + params.kd * (angular_ffd * target_twist[..., 3:] - current_twist[..., 3:])
        + params.angular_acc_ffd * target_acceleration[..., 3:]
    )
    reference = jnp.concatenate(
        (_clip_norm(linear, params.max_linear_acc), _clip_norm(angular, params.max_angular_acc)), axis=-1
    )
    return reference, e


def build_task_qp_jax(
    jacobian: Any,
    jacobian_dot_qd: Any,
    reference_acceleration: Any,
    q: Any,
    qd: Any,
    q_mid: Any,
    params: RMPParams,
    dt: float,
    *,
    psi_gain: float = 0.0,
) -> tuple[Any, Any]:
    """Return batched ``H, g``; leading dimensions are preserved."""

    _, jnp = _jax()
    residual = reference_acceleration - jacobian_dot_qd
    H = jnp.swapaxes(jacobian, -1, -2) @ jacobian
    identity = jnp.eye(jacobian.shape[-1], dtype=jacobian.dtype)
    H = H + (params.vel_weight * dt * dt + params.acc_weight) * identity
    g = -(jnp.swapaxes(jacobian, -1, -2) @ residual[..., None])[..., 0]
    g = g + params.vel_weight * dt * qd
    g = g + 4.0 * params.pos_limit_weight * abs(1.0 - psi_gain) * (q - q_mid)
    return H, g


def projected_box_qp_jax(
    H: Any,
    g: Any,
    lower: Any,
    upper: Any,
    *,
    iterations: int = 48,
) -> Any:
    """Solve batched box QPs with a fixed iteration count for ``jax.jit``."""

    jax, jnp = _jax()
    lipschitz_bound = jnp.max(jnp.sum(jnp.abs(H), axis=-1), axis=-1, keepdims=True)
    step = 0.95 / jnp.maximum(lipschitz_bound, 1.0e-9)
    initial = jnp.clip(jnp.linalg.solve(H, -g[..., None])[..., 0], lower, upper)

    def iteration(_: int, value: Any) -> Any:
        gradient = (H @ value[..., None])[..., 0] + g
        return jnp.clip(value - step * gradient, lower, upper)

    return jax.lax.fori_loop(0, iterations, iteration, initial)


def normalized_mjx_action(qdd: Any, acceleration_limit: Any, action_acc_scale: float = 1.5) -> Any:
    """Map recovered rad/s^2 commands to ``MjxJuggleEnv``'s [-1, 1] action."""

    _, jnp = _jax()
    return jnp.clip(qdd / (acceleration_limit * action_acc_scale), -1.0, 1.0)


def init_joint_rmp_state_jax(
    position: Any,
    *,
    velocity: Any | None = None,
    acceleration: Any | None = None,
    target_position: Any | None = None,
    target_filter_length: int = 10,
    output_filter_length: int = 35,
) -> JointRMPStateJax:
    """Initialize a batched recovered joint-RMP state.

    ``position`` has shape ``(..., joints)``.  All leading dimensions are
    preserved, making the state directly usable by a vectorized MJX env.
    """

    _, jnp = _jax()
    q = jnp.asarray(position)
    qd = jnp.zeros_like(q) if velocity is None else jnp.asarray(velocity, dtype=q.dtype)
    qdd = jnp.zeros_like(q) if acceleration is None else jnp.asarray(acceleration, dtype=q.dtype)
    target = q if target_position is None else jnp.asarray(target_position, dtype=q.dtype)
    target_x = jnp.stack((target, jnp.zeros_like(target), jnp.zeros_like(target)), axis=-1)
    covariance = jnp.zeros(target_x.shape + (3,), dtype=q.dtype)
    covariance = covariance.at[..., 0, 0].set(0.01)
    length = max(1, int(target_filter_length))
    q_buffer = jnp.broadcast_to(target[..., None, :], target.shape[:-1] + (length, target.shape[-1]))
    zero_buffer = jnp.zeros_like(q_buffer)
    output_length = max(1, int(output_filter_length))
    output_q_buffer = jnp.broadcast_to(
        q[..., None, :], q.shape[:-1] + (output_length, q.shape[-1])
    )
    output_qd_buffer = jnp.broadcast_to(
        qd[..., None, :], q.shape[:-1] + (output_length, q.shape[-1])
    )
    output_qdd_buffer = jnp.broadcast_to(
        qdd[..., None, :], q.shape[:-1] + (output_length, q.shape[-1])
    )
    return JointRMPStateJax(
        q=q,
        qd=qd,
        qdd=qdd,
        command_q=q,
        command_qd=qd,
        command_qdd=qdd,
        accepted_target=target,
        ticks_since_target_update=jnp.zeros(q.shape[:-1], dtype=jnp.int32),
        target_x=target_x,
        target_covariance=covariance,
        target_q_buffer=q_buffer,
        target_qd_buffer=zero_buffer,
        target_qdd_buffer=zero_buffer,
        target_buffer_index=jnp.asarray(0, dtype=jnp.int32),
        output_q_buffer=output_q_buffer,
        output_qd_buffer=output_qd_buffer,
        output_qdd_buffer=output_qdd_buffer,
        output_buffer_index=jnp.asarray(0, dtype=jnp.int32),
    )


def joint_rmp_step_jax(
    state: JointRMPStateJax,
    position_target: Any,
    q_min: Any,
    q_max: Any,
    qd_max: Any,
    qdd_max: Any,
    qddd_max: Any,
    params: RMPParams,
    dt: float,
    *,
    jnt_kp: Any | None = None,
    jnt_kd: Any | None = None,
    estimator_joint_process_var: Any | None = None,
    estimator_joint_measure_var: Any | None = None,
    joint_vel_ffd: Any | None = None,
    joint_acc_ffd: Any | None = None,
    acc_weight: Any | None = None,
    target_filter_length: Any | None = None,
    servo_steps: int = 1,
    target_update_delay_steps: int = 0,
    use_safety_bounds: bool = True,
    use_jerk_limit: bool = False,
    return_history: bool = False,
) -> JointRMPStateJax | tuple[JointRMPStateJax, Any, Any, Any]:
    """Advance the recovered position-target RMP inside JAX/MJX.

    The target is held for ``servo_steps``.  For a 200 Hz policy/ROS command
    and the recovered 1 kHz PlanTask period, use ``servo_steps=5`` and
    ``dt=0.001``.  ``target_update_delay_steps`` reproduces a measured
    command-transport delay before the new outer-loop target reaches PlanTask;
    it must remain within the current outer-loop interval.  ``q``, ``qd`` and
    ``qdd`` retain the raw integration state;
    ``command_q``, ``command_qd`` and ``command_qdd`` are the three filtered
    signals sent by real RMP motion mode to the low-level controller.  With
    ``return_history=True``, the returned histories contain those command
    signals so an MJX physics step can apply every 1 kHz reference.
    """

    if dt <= 0.0 or servo_steps <= 0:
        raise ValueError("dt and servo_steps must be positive")
    if not 0 <= int(target_update_delay_steps) < int(servo_steps):
        raise ValueError(
            "target_update_delay_steps must be in [0, servo_steps)"
        )
    jax, jnp = _jax()
    target = jnp.asarray(position_target, dtype=state.q.dtype)
    lower_position = jnp.asarray(q_min, dtype=state.q.dtype)
    upper_position = jnp.asarray(q_max, dtype=state.q.dtype)
    velocity_limit = jnp.asarray(qd_max, dtype=state.q.dtype)
    acceleration_limit = jnp.asarray(qdd_max, dtype=state.q.dtype)
    jerk_limit = jnp.asarray(qddd_max, dtype=state.q.dtype)
    angle_gap = jnp.asarray(params.angle_gap, dtype=state.q.dtype)
    identity3 = jnp.eye(3, dtype=state.q.dtype)
    period = float(dt)
    F = jnp.asarray(
        [[1.0, period, 0.5 * period * period], [0.0, 1.0, period], [0.0, 0.0, 1.0]],
        dtype=state.q.dtype,
    )
    dt2, dt3 = period**2, period**3
    dt4, dt5 = period**4, period**5
    base_Q = jnp.asarray(
        [[dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
         [dt4 / 8.0, dt3 / 3.0, dt2 / 2.0],
         [dt3 / 6.0, dt2 / 2.0, period]],
        dtype=state.q.dtype,
    )
    process_variance = jnp.asarray(
        params.estimator_joint_process_var
        if estimator_joint_process_var is None
        else estimator_joint_process_var,
        dtype=state.q.dtype,
    )
    measurement_variance = jnp.asarray(
        params.estimator_joint_measure_var
        if estimator_joint_measure_var is None
        else estimator_joint_measure_var,
        dtype=state.q.dtype,
    )
    Q = process_variance[..., None, None] * base_Q
    joint_kp = (
        jnp.asarray(params.jnt_kp, dtype=state.q.dtype)
        if jnt_kp is None
        else jnp.asarray(jnt_kp, dtype=state.q.dtype)
    )
    joint_kd = (
        jnp.asarray(params.jnt_kd, dtype=state.q.dtype)
        if jnt_kd is None
        else jnp.asarray(jnt_kd, dtype=state.q.dtype)
    )
    velocity_feedforward = jnp.asarray(
        params.joint_vel_ffd if joint_vel_ffd is None else joint_vel_ffd,
        dtype=state.q.dtype,
    )
    acceleration_feedforward = jnp.asarray(
        params.joint_acc_ffd if joint_acc_ffd is None else joint_acc_ffd,
        dtype=state.q.dtype,
    )
    acceleration_weight = jnp.asarray(
        params.acc_weight if acc_weight is None else acc_weight,
        dtype=state.q.dtype,
    )
    filter_length = state.target_q_buffer.shape[-2]
    selected_filter_length = None
    if target_filter_length is not None:
        selected_filter_length = jnp.clip(
            jnp.asarray(target_filter_length, dtype=jnp.int32),
            1,
            filter_length,
        )

    def one_servo_step(
        servo_index: int,
        value: JointRMPStateJax,
    ) -> JointRMPStateJax:
        is_target_update = servo_index == int(target_update_delay_steps)
        elapsed = (
            jnp.minimum(value.ticks_since_target_update, 100)[..., None]
            * period
        )
        maximum_target_delta = velocity_limit * elapsed
        new_accepted_target = jnp.clip(
            value.accepted_target
            + jnp.clip(
                target - value.accepted_target,
                -maximum_target_delta,
                maximum_target_delta,
            ),
            lower_position,
            upper_position,
        )
        accepted_target = jnp.where(
            is_target_update,
            new_accepted_target,
            value.accepted_target,
        )
        predicted_x = jnp.einsum("ij,...nj->...ni", F, value.target_x)
        predicted_covariance = (
            jnp.einsum("ij,...njk,lk->...nil", F, value.target_covariance, F)
            + Q
        )

        def correct_estimator(_: None) -> tuple[Any, Any]:
            innovation = accepted_target - predicted_x[..., 0]
            innovation_covariance = (
                predicted_covariance[..., 0, 0] + measurement_variance
            )
            gain = (
                predicted_covariance[..., :, 0]
                / innovation_covariance[..., None]
            )
            corrected_x = predicted_x + gain * innovation[..., None]
            # Joseph covariance update, algebraically equivalent to the
            # binary's scalar measurement update.
            kh = gain[..., :, None] * jnp.asarray(
                [1.0, 0.0, 0.0], dtype=state.q.dtype
            )
            left = identity3 - kh
            corrected_covariance = (
                left @ predicted_covariance @ jnp.swapaxes(left, -1, -2)
                + measurement_variance[..., None, None]
                * gain[..., :, None]
                * gain[..., None, :]
            )
            return corrected_x, corrected_covariance

        braking_acceleration = jnp.clip(
            -value.target_x[..., 1] / period,
            -acceleration_limit,
            acceleration_limit,
        )
        decelerating_x = value.target_x.at[..., 2].set(
            jnp.where(
                jnp.abs(value.target_x[..., 1]) <= 1.0e-6,
                0.0,
                braking_acceleration,
            )
        )
        decelerated_x = jnp.einsum("ij,...nj->...ni", F, decelerating_x)

        def held_target_prediction(_: None) -> tuple[Any, Any]:
            predict_mask = value.ticks_since_target_update <= 20
            return (
                jnp.where(predict_mask[..., None, None], predicted_x, decelerated_x),
                predicted_covariance,
            )

        target_x, target_covariance = jax.lax.cond(
            is_target_update,
            correct_estimator,
            held_target_prediction,
            operand=None,
        )
        estimated_q = target_x[..., 0]
        estimated_qd = jnp.clip(target_x[..., 1], -velocity_limit, velocity_limit)
        estimated_qdd = target_x[..., 2]
        index = value.target_buffer_index
        q_buffer = value.target_q_buffer.at[..., index, :].set(estimated_q)
        qd_buffer = value.target_qd_buffer.at[..., index, :].set(estimated_qd)
        qdd_buffer = value.target_qdd_buffer.at[..., index, :].set(estimated_qdd)
        if selected_filter_length is None:
            filtered_q = jnp.mean(q_buffer, axis=-2)
            filtered_qd = jnp.mean(qd_buffer, axis=-2)
            filtered_qdd = jnp.mean(qdd_buffer, axis=-2)
        else:
            slots = jnp.arange(filter_length, dtype=jnp.int32)
            ages = (index - slots) % filter_length
            filter_mask = ages < selected_filter_length[..., None]

            def recent_mean(buffer: Any) -> Any:
                recent = jnp.swapaxes(buffer, -2, -1)
                return jnp.sum(
                    recent * filter_mask, axis=-1
                ) / selected_filter_length

            filtered_q = recent_mean(q_buffer)
            filtered_qd = recent_mean(qd_buffer)
            filtered_qdd = recent_mean(qdd_buffer)

        reference = (
            joint_kp * (filtered_q - value.q)
            + joint_kd * (velocity_feedforward * filtered_qd - value.qd)
            + acceleration_feedforward * filtered_qdd
        )
        reference = jnp.clip(reference, -acceleration_limit, acceleration_limit)
        qdd = reference / (1.0 + acceleration_weight)
        if use_safety_bounds:
            safe_min = lower_position + angle_gap
            safe_max = upper_position - angle_gap
            braking = 0.998 * acceleration_limit
            safe_velocity_upper = jnp.minimum(
                velocity_limit,
                jnp.sqrt(2.0 * braking * jnp.maximum(safe_max - value.q, 0.0)),
            )
            safe_velocity_lower = jnp.maximum(
                -velocity_limit,
                -jnp.sqrt(2.0 * braking * jnp.maximum(value.q - safe_min, 0.0)),
            )
            lower = jnp.maximum(-acceleration_limit, (safe_velocity_lower - value.qd) / period)
            upper = jnp.minimum(acceleration_limit, (safe_velocity_upper - value.qd) / period)
            if use_jerk_limit:
                lower = jnp.maximum(lower, value.qdd - jerk_limit * period)
                upper = jnp.minimum(upper, value.qdd + jerk_limit * period)
            feasible = lower <= upper
            # Match ``joint_acceleration_bounds(strict=False)``: if the two
            # independently valid constraints no longer intersect, choose the
            # zero-nearest value in the gap bounded by [upper, lower].
            emergency_recovery = jnp.clip(jnp.zeros_like(qdd), upper, lower)
            qdd = jnp.where(
                feasible,
                jnp.clip(qdd, lower, upper),
                emergency_recovery,
            )
        q_next = value.q + (value.qd + 0.5 * period * qdd) * period
        qd_next = value.qd + period * qdd
        output_index = value.output_buffer_index
        output_q_buffer = value.output_q_buffer.at[..., output_index, :].set(
            q_next
        )
        output_qd_buffer = value.output_qd_buffer.at[..., output_index, :].set(
            qd_next
        )
        output_qdd_buffer = value.output_qdd_buffer.at[..., output_index, :].set(
            qdd
        )
        return JointRMPStateJax(
            q=q_next,
            qd=qd_next,
            qdd=qdd,
            command_q=jnp.mean(output_q_buffer, axis=-2),
            command_qd=jnp.mean(output_qd_buffer, axis=-2),
            command_qdd=jnp.mean(output_qdd_buffer, axis=-2),
            accepted_target=accepted_target,
            ticks_since_target_update=jnp.where(
                is_target_update,
                jnp.ones_like(value.ticks_since_target_update),
                value.ticks_since_target_update + 1,
            ),
            target_x=target_x,
            target_covariance=target_covariance,
            target_q_buffer=q_buffer,
            target_qd_buffer=qd_buffer,
            target_qdd_buffer=qdd_buffer,
            target_buffer_index=(index + 1) % filter_length,
            output_q_buffer=output_q_buffer,
            output_qd_buffer=output_qd_buffer,
            output_qdd_buffer=output_qdd_buffer,
            output_buffer_index=(output_index + 1)
            % value.output_q_buffer.shape[-2],
        )

    if return_history:
        def scan_servo_step(
            value: JointRMPStateJax,
            _: Any,
        ) -> tuple[JointRMPStateJax, tuple[Any, Any, Any]]:
            next_value = one_servo_step(_, value)
            return next_value, (
                next_value.command_q,
                next_value.command_qd,
                next_value.command_qdd,
            )

        final_state, history = jax.lax.scan(
            scan_servo_step,
            state,
            xs=jnp.arange(int(servo_steps), dtype=jnp.int32),
        )
        return final_state, history[0], history[1], history[2]
    return jax.lax.fori_loop(0, int(servo_steps), one_servo_step, state)
