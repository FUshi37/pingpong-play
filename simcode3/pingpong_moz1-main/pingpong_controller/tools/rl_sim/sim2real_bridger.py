"""Feasible-trajectory compensation primitives for Sim2Real Bridger.

The compensation state is ``(q, dq, ddq)``.  A model-based inverse supplies a
soft position/velocity target, while this module advances the transmitted
command inside the exact discrete q/dq/ddq viability interval and a causal
jerk interval.  The resulting command is constrained by construction; it is
not an unconstrained compensation angle followed by a post-hoc clip.

NumPy and JAX implementations intentionally share the same equations so the
offline/deployment controller can be checked against batched MJX execution.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class BridgerStep(NamedTuple):
    q: np.ndarray
    qvel: np.ndarray
    qacc: np.ndarray
    feasible: np.ndarray
    jerk_feasible: np.ndarray
    interval_low: np.ndarray
    interval_high: np.ndarray


def _stopping_velocity_limit_numpy(
    distance: np.ndarray,
    acc_limit: np.ndarray,
    dt: float,
) -> np.ndarray:
    distance = np.maximum(np.asarray(distance), 0.0)
    acc_limit = np.asarray(acc_limit, dtype=distance.dtype)
    dt_value = np.asarray(float(dt), dtype=distance.dtype)
    accel_step = acc_limit * dt_value
    scaled = 8.0 * distance / np.maximum(acc_limit * dt_value**2, 1e-12)
    brake_steps = np.floor(0.5 * (np.sqrt(1.0 + scaled) - 1.0))
    return distance / (dt_value * (brake_steps + 1.0)) + 0.5 * accel_step * brake_steps


def _jerk_braking_distance_numpy(
    velocity: np.ndarray,
    acceleration: np.ndarray,
    acc_limit: np.ndarray,
    jerk_limit: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Exact semi-implicit discrete travel under maximum negative jerk/braking."""

    dt_value = float(dt)
    jerk_step = jerk_limit * dt_value
    ramp_steps = np.floor(np.maximum((acceleration + acc_limit) / jerk_step, 0.0) + 1e-9)
    root_disc = (acceleration - 0.5 * jerk_step) ** 2 + 2.0 * jerk_step * velocity / dt_value
    later_root = (
        acceleration
        - 0.5 * jerk_step
        + np.sqrt(np.maximum(root_disc, 0.0))
    ) / jerk_step
    positive_ramp_steps = np.minimum(
        ramp_steps,
        np.maximum(np.floor(later_root), 0.0),
    )
    m = positive_ramp_steps
    ramp_distance = dt_value * (
        m * velocity
        + dt_value
        * (
            acceleration * m * (m + 1.0) / 2.0
            - jerk_step * m * (m + 1.0) * (m + 2.0) / 6.0
        )
    )
    v_after_ramp = velocity + dt_value * (
        ramp_steps * acceleration
        - jerk_step * ramp_steps * (ramp_steps + 1.0) / 2.0
    )
    constant_steps = np.maximum(np.floor(np.maximum(v_after_ramp, 0.0) / (acc_limit * dt_value)), 0.0)
    constant_distance = dt_value * (
        constant_steps * v_after_ramp
        - acc_limit * dt_value * constant_steps * (constant_steps + 1.0) / 2.0
    )
    reaches_end_of_ramp = positive_ramp_steps >= ramp_steps
    distance = ramp_distance + np.where(reaches_end_of_ramp, constant_distance, 0.0)
    return np.where(root_disc >= 0.0, np.maximum(distance, 0.0), 0.0)


def _max_safe_acceleration_numpy(
    q: np.ndarray,
    velocity: np.ndarray,
    boundary_high: np.ndarray,
    acc_limit: np.ndarray,
    jerk_limit: np.ndarray,
    dt: float,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Largest next acceleration whose next state can jerk-brake before a wall."""

    def residual(candidate_acc: np.ndarray) -> np.ndarray:
        next_velocity = velocity + candidate_acc * dt
        next_q = q + next_velocity * dt
        stop = _jerk_braking_distance_numpy(
            next_velocity, candidate_acc, acc_limit, jerk_limit, dt
        )
        return boundary_high - margin - next_q - stop

    lower = -acc_limit.copy()
    upper = acc_limit.copy()
    possible = residual(lower) >= -5e-8
    all_safe = residual(upper) >= 0.0
    for _ in range(16):
        middle = 0.5 * (lower + upper)
        safe = residual(middle) >= 0.0
        lower = np.where(safe, middle, lower)
        upper = np.where(safe, upper, middle)
    conservative_root = np.maximum(lower - 2e-3, -acc_limit)
    return np.where(all_safe, acc_limit, conservative_root), possible


def _max_safe_acceleration_for_velocity_numpy(
    velocity: np.ndarray,
    velocity_high: np.ndarray,
    acc_limit: np.ndarray,
    jerk_limit: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Largest next acceleration that can jerk-brake before a velocity wall."""

    jerk_step = jerk_limit * dt

    def residual(candidate_acc: np.ndarray) -> np.ndarray:
        next_velocity = velocity + candidate_acc * dt
        positive_steps = np.maximum(np.floor(np.maximum(candidate_acc, 0.0) / jerk_step), 0.0)
        extra_velocity = dt * (
            positive_steps * candidate_acc
            - jerk_step * positive_steps * (positive_steps + 1.0) / 2.0
        )
        return velocity_high - next_velocity - np.maximum(extra_velocity, 0.0)

    lower = -acc_limit.copy()
    upper = acc_limit.copy()
    possible = residual(lower) >= -5e-8
    all_safe = residual(upper) >= 0.0
    for _ in range(16):
        middle = 0.5 * (lower + upper)
        safe = residual(middle) >= 0.0
        lower = np.where(safe, middle, lower)
        upper = np.where(safe, upper, middle)
    conservative_root = np.maximum(lower - 2e-3, -acc_limit)
    return np.where(all_safe, acc_limit, conservative_root), possible


def constrained_compensation_step_numpy(
    target_q: np.ndarray,
    target_qvel: np.ndarray,
    current_q: np.ndarray,
    current_qvel: np.ndarray,
    current_qacc: np.ndarray,
    pos_low: np.ndarray,
    pos_high: np.ndarray,
    vel_limit: np.ndarray,
    acc_limit: np.ndarray,
    jerk_limit: np.ndarray,
    *,
    dt: float,
    natural_frequency_hz: float,
    damping_ratio: float = 1.0,
    target_qacc: np.ndarray | None = None,
) -> BridgerStep:
    """Advance one model-based compensation step inside hard motion bounds.

    The critically damped request tracks the inverse-model target.  It is
    projected once into the recursively viable q/dq/ddq velocity interval and
    the causal jerk interval.  ``jerk_feasible`` is reported per joint; a valid
    Bridger trajectory is required to keep it true rather than silently use
    the hard-safety fallback.
    """

    if target_qacc is None:
        target_qacc = np.zeros_like(np.asarray(target_q, dtype=np.float64))
    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (
            target_q,
            target_qvel,
            current_q,
            current_qvel,
            current_qacc,
            pos_low,
            pos_high,
            vel_limit,
            acc_limit,
            jerk_limit,
            target_qacc,
        )
    ]
    arrays = list(np.broadcast_arrays(*arrays))
    (
        target,
        target_vel,
        q,
        qvel,
        qacc,
        low,
        high,
        vmax,
        amax,
        jmax,
        target_acc,
    ) = arrays
    if q.ndim < 1:
        raise ValueError("constrained compensation inputs must include a joint axis")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if not np.isfinite(natural_frequency_hz) or natural_frequency_hz <= 0.0:
        raise ValueError("natural_frequency_hz must be positive and finite")
    if not np.isfinite(damping_ratio) or damping_ratio <= 0.0:
        raise ValueError("damping_ratio must be positive and finite")
    if any(np.any(~np.isfinite(value)) for value in arrays):
        raise ValueError("constrained compensation inputs must be finite")
    if np.any(low >= high) or np.any(vmax <= 0.0) or np.any(amax <= 0.0) or np.any(jmax <= 0.0):
        raise ValueError("position and motion limits must be valid")

    # The state variable is acceleration, so viability is built in acceleration
    # space.  The two wall constraints use the analytic jerk-limited stopping
    # distance and make the feasible set recursive without a downstream filter.
    position_margin = 2e-6
    safe_low_pos = low + position_margin
    safe_high_pos = high - position_margin
    effective_amax = amax * 0.9998
    effective_vmax = vmax * 0.99999
    dt_value = float(dt)
    upper_acc, upper_possible = _max_safe_acceleration_numpy(
        q, qvel, safe_high_pos, effective_amax, jmax, dt_value, position_margin
    )
    neg_upper_acc, lower_possible = _max_safe_acceleration_numpy(
        -q, -qvel, -safe_low_pos, effective_amax, jmax, dt_value, position_margin
    )
    lower_acc = -neg_upper_acc
    upper_vel_acc, upper_vel_possible = _max_safe_acceleration_for_velocity_numpy(
        qvel, effective_vmax, effective_amax, jmax, dt_value
    )
    neg_upper_vel_acc, lower_vel_possible = _max_safe_acceleration_for_velocity_numpy(
        -qvel, effective_vmax, effective_amax, jmax, dt_value
    )
    lower_vel_acc = -neg_upper_vel_acc
    jerk_delta = jmax * dt_value
    interval_low_acc = np.maximum.reduce(
        [
            -effective_amax,
            qacc - jerk_delta,
            (-effective_vmax - qvel) / dt_value,
            (safe_low_pos - q - qvel * dt_value) / (dt_value * dt_value),
            lower_acc,
            lower_vel_acc,
        ]
    )
    interval_high_acc = np.minimum.reduce(
        [
            effective_amax,
            qacc + jerk_delta,
            (effective_vmax - qvel) / dt_value,
            (safe_high_pos - q - qvel * dt_value) / (dt_value * dt_value),
            upper_acc,
            upper_vel_acc,
        ]
    )
    feasible_joint = (
        upper_possible
        & lower_possible
        & upper_vel_possible
        & lower_vel_possible
        & (interval_low_acc <= interval_high_acc + 5e-8)
    )

    target_clipped = np.minimum(np.maximum(target, safe_low_pos), safe_high_pos)
    omega = 2.0 * np.pi * float(natural_frequency_hz)
    desired_acc = target_acc + omega * omega * (target_clipped - q) + 2.0 * float(
        damping_ratio
    ) * omega * (target_vel - qvel)
    jerk_feasible = interval_low_acc <= interval_high_acc + 5e-8
    # Keep execution finite on a declared invalid initial state, but expose it.
    active_low = interval_low_acc
    active_high = interval_high_acc
    mid = 0.5 * (active_low + active_high)
    active_low = np.minimum(active_low, mid)
    active_high = np.maximum(active_high, mid)
    next_acc = np.minimum(np.maximum(desired_acc, active_low), active_high)
    next_vel = qvel + next_acc * dt_value
    next_q = q + next_vel * dt_value
    feasible = np.all(feasible_joint, axis=-1)
    return BridgerStep(
        q=next_q,
        qvel=next_vel,
        qacc=next_acc,
        feasible=feasible,
        jerk_feasible=jerk_feasible,
        interval_low=qvel + interval_low_acc * dt_value,
        interval_high=qvel + interval_high_acc * dt_value,
    )


def constrained_compensation_step_jax(
    target_q,
    target_qvel,
    current_q,
    current_qvel,
    current_qacc,
    pos_low,
    pos_high,
    vel_limit,
    acc_limit,
    jerk_limit,
    *,
    dt: float,
    natural_frequency_hz: float,
    damping_ratio: float = 1.0,
    target_qacc=None,
):
    """Batched JAX equivalent of :func:`constrained_compensation_step_numpy`."""

    import jax.numpy as jnp

    target = jnp.asarray(target_q)
    target_vel = jnp.asarray(target_qvel, dtype=target.dtype)
    target_acc = (
        jnp.zeros_like(target)
        if target_qacc is None
        else jnp.asarray(target_qacc, dtype=target.dtype)
    )
    q = jnp.asarray(current_q, dtype=target.dtype)
    qvel = jnp.asarray(current_qvel, dtype=target.dtype)
    qacc = jnp.asarray(current_qacc, dtype=target.dtype)
    low = jnp.asarray(pos_low, dtype=target.dtype)
    high = jnp.asarray(pos_high, dtype=target.dtype)
    vmax = jnp.asarray(vel_limit, dtype=target.dtype)
    amax = jnp.asarray(acc_limit, dtype=target.dtype)
    jmax = jnp.asarray(jerk_limit, dtype=target.dtype)
    safe_low_pos = low + jnp.asarray(2e-6, dtype=target.dtype)
    safe_high_pos = high - jnp.asarray(2e-6, dtype=target.dtype)
    effective_amax = amax * 0.9998
    effective_vmax = vmax * 0.99999
    dt_value = jnp.asarray(float(dt), dtype=target.dtype)

    def braking_distance(velocity, acceleration):
        jerk_step = jmax * dt_value
        ramp_steps = jnp.floor(jnp.maximum((acceleration + effective_amax) / jerk_step, 0.0) + 1e-9)
        root_disc = (acceleration - 0.5 * jerk_step) ** 2 + 2.0 * jerk_step * velocity / dt_value
        later_root = (
            acceleration - 0.5 * jerk_step + jnp.sqrt(jnp.maximum(root_disc, 0.0))
        ) / jerk_step
        positive_ramp_steps = jnp.minimum(
            ramp_steps, jnp.maximum(jnp.floor(later_root), 0.0)
        )
        m = positive_ramp_steps
        ramp_distance = dt_value * (
            m * velocity
            + dt_value
            * (
                acceleration * m * (m + 1.0) / 2.0
                - jerk_step * m * (m + 1.0) * (m + 2.0) / 6.0
            )
        )
        v_after_ramp = velocity + dt_value * (
            ramp_steps * acceleration
            - jerk_step * ramp_steps * (ramp_steps + 1.0) / 2.0
        )
        constant_steps = jnp.maximum(
            jnp.floor(jnp.maximum(v_after_ramp, 0.0) / (effective_amax * dt_value)), 0.0
        )
        constant_distance = dt_value * (
            constant_steps * v_after_ramp
            - effective_amax
            * dt_value
            * constant_steps
            * (constant_steps + 1.0)
            / 2.0
        )
        distance = ramp_distance + jnp.where(
            positive_ramp_steps >= ramp_steps, constant_distance, 0.0
        )
        return jnp.where(root_disc >= 0.0, jnp.maximum(distance, 0.0), 0.0)

    def max_safe_acceleration(position, velocity, boundary):
        def residual(candidate_acc):
            next_velocity = velocity + candidate_acc * dt_value
            next_position = position + next_velocity * dt_value
            return boundary - 2e-6 - next_position - braking_distance(next_velocity, candidate_acc)

        bisect_low = -effective_amax
        bisect_high = effective_amax
        possible = residual(bisect_low) >= -5e-8
        all_safe = residual(bisect_high) >= 0.0
        for _ in range(16):
            middle = 0.5 * (bisect_low + bisect_high)
            safe = residual(middle) >= 0.0
            bisect_low = jnp.where(safe, middle, bisect_low)
            bisect_high = jnp.where(safe, bisect_high, middle)
        conservative_root = jnp.maximum(bisect_low - 2e-3, -effective_amax)
        return jnp.where(all_safe, effective_amax, conservative_root), possible

    def max_safe_acceleration_for_velocity(velocity, velocity_high):
        jerk_step = jmax * dt_value

        def residual(candidate_acc):
            next_velocity = velocity + candidate_acc * dt_value
            positive_steps = jnp.maximum(
                jnp.floor(jnp.maximum(candidate_acc, 0.0) / jerk_step), 0.0
            )
            extra_velocity = dt_value * (
                positive_steps * candidate_acc
                - jerk_step * positive_steps * (positive_steps + 1.0) / 2.0
            )
            return velocity_high - next_velocity - jnp.maximum(extra_velocity, 0.0)

        bisect_low = -effective_amax
        bisect_high = effective_amax
        possible = residual(bisect_low) >= -5e-8
        all_safe = residual(bisect_high) >= 0.0
        for _ in range(16):
            middle = 0.5 * (bisect_low + bisect_high)
            safe = residual(middle) >= 0.0
            bisect_low = jnp.where(safe, middle, bisect_low)
            bisect_high = jnp.where(safe, bisect_high, middle)
        conservative_root = jnp.maximum(bisect_low - 2e-3, -effective_amax)
        return jnp.where(all_safe, effective_amax, conservative_root), possible

    upper_acc, upper_possible = max_safe_acceleration(q, qvel, safe_high_pos)
    neg_upper_acc, lower_possible = max_safe_acceleration(-q, -qvel, -safe_low_pos)
    lower_acc = -neg_upper_acc
    upper_vel_acc, upper_vel_possible = max_safe_acceleration_for_velocity(qvel, effective_vmax)
    neg_upper_vel_acc, lower_vel_possible = max_safe_acceleration_for_velocity(-qvel, effective_vmax)
    lower_vel_acc = -neg_upper_vel_acc
    jerk_delta = jmax * dt_value
    interval_low_acc = jnp.maximum(
        jnp.maximum(-effective_amax, qacc - jerk_delta),
        jnp.maximum(
            (-effective_vmax - qvel) / dt_value,
            jnp.maximum(
                (safe_low_pos - q - qvel * dt_value) / dt_value**2,
                jnp.maximum(lower_acc, lower_vel_acc),
            ),
        ),
    )
    interval_high_acc = jnp.minimum(
        jnp.minimum(effective_amax, qacc + jerk_delta),
        jnp.minimum(
            (effective_vmax - qvel) / dt_value,
            jnp.minimum(
                (safe_high_pos - q - qvel * dt_value) / dt_value**2,
                jnp.minimum(upper_acc, upper_vel_acc),
            ),
        ),
    )
    feasible_joint = (
        upper_possible
        & lower_possible
        & upper_vel_possible
        & lower_vel_possible
        & (interval_low_acc <= interval_high_acc + 5e-8)
    )

    target_clipped = jnp.clip(target, safe_low_pos, safe_high_pos)
    omega = 2.0 * jnp.pi * float(natural_frequency_hz)
    desired_acc = target_acc + omega * omega * (target_clipped - q) + 2.0 * float(
        damping_ratio
    ) * omega * (target_vel - qvel)
    jerk_feasible = interval_low_acc <= interval_high_acc + 5e-8
    active_low = interval_low_acc
    active_high = interval_high_acc
    mid = 0.5 * (active_low + active_high)
    active_low = jnp.minimum(active_low, mid)
    active_high = jnp.maximum(active_high, mid)
    next_acc = jnp.minimum(jnp.maximum(desired_acc, active_low), active_high)
    next_vel = qvel + next_acc * dt_value
    next_q = q + next_vel * dt_value
    feasible = jnp.all(feasible_joint, axis=-1)
    return (
        next_q,
        next_vel,
        next_acc,
        feasible,
        jerk_feasible,
        qvel + interval_low_acc * dt_value,
        qvel + interval_high_acc * dt_value,
    )
