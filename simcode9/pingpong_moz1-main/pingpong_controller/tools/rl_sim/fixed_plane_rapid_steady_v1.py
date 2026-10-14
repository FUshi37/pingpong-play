"""Physics-based apex schedule for the opt-in rapid-steady V1 course.

This is a goal generator, not a controller or contact-model override.  The
ideal equal-speed trajectory is a planning prior; native contact and measured
per-impact outcomes remain the training and validation sources of truth.
"""

from __future__ import annotations

import numpy as np
from collections.abc import Mapping
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class RapidSteadyApexSchedule:
    """Hit-indexed absolute apex and ideal racket speed, in SI units."""

    band_entry_hit_index: int
    apex_z_m: tuple[float, ...]
    required_racket_vz_m_s: tuple[float, ...]
    three_hit_required_racket_vz_m_s: float


def scheduled_apex_z_m(
    confirmed_hit_count: jax.Array,
    apex_schedule_z_m: tuple[float, ...],
) -> jax.Array:
    """Select the active goal without using future state or a phase teacher."""

    schedule = jnp.asarray(apex_schedule_z_m, dtype=jnp.float32)
    if schedule.ndim != 1 or schedule.size < 2:
        raise ValueError("explicit apex schedule must have at least two hits")
    index = jnp.clip(
        jnp.asarray(confirmed_hit_count, dtype=jnp.int32),
        0,
        schedule.size - 1,
    )
    return jnp.take(schedule, index)


def apex_progress_for_schedule(
    confirmed_hit_count: jax.Array,
    apex_schedule_z_m: tuple[float, ...],
) -> jax.Array:
    """Match the actor progress scalar to the actual scheduled apex."""

    schedule = tuple(float(value) for value in apex_schedule_z_m)
    if len(schedule) < 2 or schedule[-1] <= schedule[0]:
        raise ValueError("explicit apex schedule must rise to nominal")
    active = scheduled_apex_z_m(confirmed_hit_count, schedule)
    return jnp.clip(
        (active - schedule[0]) / (schedule[-1] - schedule[0]),
        0.0,
        1.0,
    )


def per_hit_apex_event_metrics(
    *,
    counted_hit: jax.Array,
    hit_count: jax.Array,
    predicted_apex_z_m: jax.Array,
    contact_surface_z_m: jax.Array | None = None,
    target_surface_z_m: float | None = None,
    racket_vxy_m_s: jax.Array | None = None,
    racket_omega_rad_s: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """Attribute observed apex to its true counted impact ordinal.

    Hit 4 is the first intended in-band contact; hit 5 and later should
    repeat the nominal orbit.  Missing events have a zero numerator and
    denominator, so no low-launch samples dilute the steady statistics.
    """

    event = jnp.asarray(counted_hit, dtype=bool)
    ordinal = jnp.asarray(hit_count, dtype=jnp.int32)
    apex = jnp.asarray(predicted_apex_z_m, dtype=jnp.float32)
    if event.shape != ordinal.shape or event.shape != apex.shape:
        raise ValueError("hit event, ordinal and apex shapes must match")
    result: dict[str, jax.Array] = {}
    for hit_index in range(1, 6):
        active = event & (ordinal == hit_index)
        result[f"metric/rapid_steady_hit{hit_index}_events"] = active.astype(
            jnp.float32
        )
        result[f"metric/rapid_steady_hit{hit_index}_apex_sum_z_m"] = jnp.where(
            active, apex, 0.0
        )
    steady = event & (ordinal >= 4)
    result["metric/rapid_steady_steady_events"] = steady.astype(jnp.float32)
    result["metric/rapid_steady_steady_apex_sum_z_m"] = jnp.where(
        steady, apex, 0.0
    )
    result["metric/rapid_steady_steady_band_events"] = (
        steady & (apex >= 1.40) & (apex <= 1.45)
    ).astype(jnp.float32)
    quality_inputs = (
        contact_surface_z_m,
        target_surface_z_m,
        racket_vxy_m_s,
        racket_omega_rad_s,
    )
    if any(value is not None for value in quality_inputs):
        if any(value is None for value in quality_inputs):
            raise ValueError("late-impact quality inputs must be complete")
        surface = jnp.asarray(contact_surface_z_m, dtype=jnp.float32)
        racket_vxy = jnp.asarray(racket_vxy_m_s, dtype=jnp.float32)
        racket_omega = jnp.asarray(racket_omega_rad_s, dtype=jnp.float32)
        if any(value.shape != event.shape for value in (
            surface, racket_vxy, racket_omega,
        )):
            raise ValueError("late-impact quality arrays must share shape")
        result["metric/rapid_steady_steady_surface_z_abs_error_sum_m"] = (
            jnp.where(
                steady,
                jnp.abs(surface - float(target_surface_z_m)),
                0.0,
            )
        )
        result["metric/rapid_steady_steady_racket_vxy_sq_sum"] = jnp.where(
            steady, racket_vxy * racket_vxy, 0.0
        )
        result["metric/rapid_steady_steady_racket_omega_sq_sum"] = jnp.where(
            steady, racket_omega * racket_omega, 0.0
        )
    return result


def summarize_apex_event_metrics(
    raw_metrics: Mapping[str, object],
) -> dict[str, float]:
    """Turn compact event sums into fail-closed per-hit means and band rate."""

    def scalar(name: str) -> float:
        value = raw_metrics.get(name, 0.0)
        result = float(np.asarray(value, dtype=np.float64).sum())
        if not math.isfinite(result) or result < 0.0:
            raise ValueError(f"invalid rapid-steady event metric {name}")
        return result

    summary: dict[str, float] = {}
    for index in range(1, 6):
        count = scalar(f"rapid_steady_hit{index}_events")
        apex_sum = scalar(f"rapid_steady_hit{index}_apex_sum_z_m")
        summary[f"rapid_steady_hit{index}_event_count"] = count
        summary[f"rapid_steady_hit{index}_mean_apex_z_m"] = (
            apex_sum / count if count > 0.0 else float("nan")
        )
    steady_count = scalar("rapid_steady_steady_events")
    steady_apex_sum = scalar("rapid_steady_steady_apex_sum_z_m")
    steady_band_count = scalar("rapid_steady_steady_band_events")
    summary["rapid_steady_steady_event_count"] = steady_count
    summary["rapid_steady_steady_mean_apex_z_m"] = (
        steady_apex_sum / steady_count
        if steady_count > 0.0 else float("nan")
    )
    summary["rapid_steady_steady_band_rate"] = (
        steady_band_count / steady_count
        if steady_count > 0.0 else float("nan")
    )
    return summary


def finite_apex_event_summary(
    raw_metrics: Mapping[str, object],
) -> dict[str, float]:
    """Keep count-zero evidence but omit undefined conditional means."""

    summary = summarize_apex_event_metrics(raw_metrics)
    return {key: value for key, value in summary.items() if math.isfinite(value)}


def rapid_steady_apex_event_reward(
    *,
    event: jax.Array,
    predicted_apex_z_m: jax.Array,
    scheduled_apex_z_m: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Score resolved contact height in the *actual* fixed-contact return.

    The broad Gaussian supplies a gradient over the observed low-bounce
    population; the narrow one rewards accurate steady cycles. A separate
    one-sided barrier penalizes both early-transition overshoot and the
    global 1.45 m ceiling. This is opt-in and never changes native contact.
    """

    active = jnp.asarray(event, dtype=bool)
    apex = jnp.asarray(predicted_apex_z_m, dtype=jnp.float32)
    target = jnp.asarray(scheduled_apex_z_m, dtype=jnp.float32)
    if active.shape != apex.shape or active.shape != target.shape:
        raise ValueError("rapid-steady reward inputs must share shape")
    error = apex - target
    positive = (
        jnp.exp(-0.5 * (error / 0.080) ** 2)
        + 2.0 * jnp.exp(-0.5 * (error / 0.025) ** 2)
    )
    upper_z = jnp.where(
        target >= 1.40 - 1.0e-6,
        1.45,
        jnp.minimum(target + 0.035, 1.45),
    )
    upper_excess = jnp.maximum(apex - upper_z, 0.0)
    upper_penalty = -2.0 * jnp.minimum((upper_excess / 0.050) ** 2, 4.0)
    positive = jnp.where(active, positive, 0.0)
    upper_penalty = jnp.where(active, upper_penalty, 0.0)
    return positive + upper_penalty, positive, upper_penalty


def rapid_steady_late_motion_bonus(
    *,
    event: jax.Array,
    confirmed_hit_count: jax.Array,
    racket_vxy_m_s: jax.Array,
    racket_omega_rad_s: jax.Array,
    vxy_weight: float,
    omega_weight: float,
) -> jax.Array:
    """Nonnegative third-and-later impact credit outside negative-event caps."""

    weights = (float(vxy_weight), float(omega_weight))
    if not all(math.isfinite(weight) and weight >= 0.0 for weight in weights):
        raise ValueError("late motion weights must be finite and nonnegative")
    active = jnp.asarray(event, dtype=bool) & (
        jnp.asarray(confirmed_hit_count) >= 3
    )
    vxy = jnp.abs(jnp.asarray(racket_vxy_m_s, dtype=jnp.float32))
    omega = jnp.abs(jnp.asarray(racket_omega_rad_s, dtype=jnp.float32))
    if active.shape != vxy.shape or active.shape != omega.shape:
        raise ValueError("late motion bonus inputs must share shape")
    vxy_score = 1.0 / (1.0 + (vxy / 0.18) ** 2)
    omega_score = 1.0 / (1.0 + (omega / 0.80) ** 2)
    return jnp.where(
        active,
        weights[0] * vxy_score + weights[1] * omega_score,
        0.0,
    )


def rapid_steady_preimpact_motion_dt(
    *,
    crossing_valid: jax.Array,
    reward_active: jax.Array,
    confirmed_hit_count: jax.Array,
    time_to_contact_s: jax.Array,
    tracking_xy_error_m: jax.Array,
    lead_s: float,
    control_dt_s: float,
) -> jax.Array:
    """Align first, then reward a quiet racket over the final 200 ms."""

    lead = float(lead_s)
    dt = float(control_dt_s)
    if not math.isfinite(lead) or not math.isfinite(dt) or lead <= 0.0 or dt <= 0.0:
        raise ValueError("pre-impact motion lead and control dt must be positive")
    ttc = jnp.asarray(time_to_contact_s, dtype=jnp.float32)
    xy_error = jnp.abs(jnp.asarray(tracking_xy_error_m, dtype=jnp.float32))
    active = (
        jnp.asarray(crossing_valid, dtype=bool)
        & jnp.asarray(reward_active, dtype=bool)
        & (jnp.asarray(confirmed_hit_count) >= 2)
        & (ttc > 0.0)
        & (ttc <= lead)
    )
    if active.shape != ttc.shape or active.shape != xy_error.shape:
        raise ValueError("pre-impact motion inputs must share shape")
    aligned = jnp.exp(-0.5 * (xy_error / 0.050) ** 2)
    return jnp.where(active, dt * aligned, 0.0)


def rapid_steady_launch_hold_motion_penalty(
    *,
    hold_active: jax.Array,
    racket_vxy_m_s: jax.Array,
    racket_omega_rad_s: jax.Array,
    vxy_weight: float,
    omega_weight: float,
    control_dt_s: float,
) -> jax.Array:
    """Discourage lateral/rotational racket motion while the ball is held.

    The racket-relative hold copies measured racket velocity into the ball.
    This dense pre-release term remains outside the saturated hit-event
    budget and is exactly zero for every released-ball/legacy step.
    """

    weights = (float(vxy_weight), float(omega_weight))
    dt = float(control_dt_s)
    if not all(math.isfinite(weight) and weight >= 0.0 for weight in weights):
        raise ValueError("hold motion weights must be finite and nonnegative")
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("control_dt_s must be finite and positive")
    active = jnp.asarray(hold_active, dtype=bool)
    vxy = jnp.abs(jnp.asarray(racket_vxy_m_s, dtype=jnp.float32))
    omega = jnp.abs(jnp.asarray(racket_omega_rad_s, dtype=jnp.float32))
    if active.shape != vxy.shape or active.shape != omega.shape:
        raise ValueError("hold motion inputs must share shape")
    return jnp.where(
        active,
        -dt * (weights[0] * vxy + weights[1] * omega),
        0.0,
    )


def rapid_steady_subfloor_bridge_reward(
    *,
    event: jax.Array,
    predicted_apex_z_m: jax.Array,
    scheduled_apex_z_m: jax.Array,
    contact_ball_center_z_m: jax.Array,
    survival_margin_m: float,
) -> jax.Array:
    """Give recurrent subfloor launches a continuous route to survival height.

    V23's Gaussian begins only at the survival threshold. This opt-in
    bridge starts at zero near contact height and joins that Gaussian at
    the threshold, while leaving the first two hit targets unchanged.
    """

    active = jnp.asarray(event, dtype=bool)
    apex = jnp.asarray(predicted_apex_z_m, dtype=jnp.float32)
    target = jnp.asarray(scheduled_apex_z_m, dtype=jnp.float32)
    contact = jnp.asarray(contact_ball_center_z_m, dtype=jnp.float32)
    if not (active.shape == apex.shape == target.shape == contact.shape):
        raise ValueError("subfloor bridge inputs must share shape")
    margin = float(survival_margin_m)
    if not math.isfinite(margin) or margin <= 0.0:
        raise ValueError("survival_margin_m must be finite and positive")
    survival = jnp.maximum(target - margin, contact + 0.015)
    boundary_reward, _, _ = rapid_steady_apex_event_reward(
        event=jnp.ones_like(active),
        predicted_apex_z_m=survival,
        scheduled_apex_z_m=target,
    )
    progress = jnp.clip(
        (apex - contact) / jnp.maximum(survival - contact, 1.0e-6),
        0.0,
        1.0,
    )
    return jnp.where(
        active & (target >= 1.40 - 1.0e-6),
        boundary_reward * progress,
        0.0,
    )


def design_apex_schedule(
    *,
    launch_apex_z_m: float,
    band_entry_apex_z_m: float,
    nominal_apex_z_m: float,
    contact_ball_center_z_m: float,
    gravity_m_s2: float,
    effective_restitution: float,
    comfortable_recurrent_racket_vz_m_s: float,
    max_band_entry_hit_index: int = 8,
) -> RapidSteadyApexSchedule:
    """Find the earliest band-entry hit under an explicit speed budget.

    For a same-height, vertical ballistic return, the ideal outgoing speed
    after successive impacts follows ``v[k+1] = (1+e)*u + e*v[k]`` when the
    same recurrent racket speed ``u`` is used.  This is a *lower-bound design*
    under nominal contact, not proof that a delayed RMP robot can realize the
    complete trajectory.  The nominal apex follows one hit after band entry.
    """

    values = (
        launch_apex_z_m,
        band_entry_apex_z_m,
        nominal_apex_z_m,
        contact_ball_center_z_m,
        gravity_m_s2,
        effective_restitution,
        comfortable_recurrent_racket_vz_m_s,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("rapid-steady schedule inputs must be finite")
    if not (
        contact_ball_center_z_m < launch_apex_z_m
        < band_entry_apex_z_m <= nominal_apex_z_m
    ):
        raise ValueError("rapid-steady apex heights must be ordered")
    if gravity_m_s2 <= 0.0 or not 0.0 < effective_restitution < 1.0:
        raise ValueError("gravity and effective restitution are invalid")
    if comfortable_recurrent_racket_vz_m_s <= 0.0:
        raise ValueError("recurrent racket speed budget must be positive")
    if max_band_entry_hit_index < 3:
        raise ValueError("max_band_entry_hit_index must be at least three")

    restitution = float(effective_restitution)
    gravity = float(gravity_m_s2)
    contact_z = float(contact_ball_center_z_m)

    def outgoing_speed(apex_z_m: float) -> float:
        return math.sqrt(2.0 * gravity * (float(apex_z_m) - contact_z))

    launch_vz = outgoing_speed(launch_apex_z_m)
    entry_vz = outgoing_speed(band_entry_apex_z_m)

    def uniform_recurrent_speed(entry_hit_index: int) -> float:
        recurrent_contacts = entry_hit_index - 1
        geometric_sum = sum(
            restitution**power for power in range(recurrent_contacts)
        )
        return (
            entry_vz - restitution**recurrent_contacts * launch_vz
        ) / ((1.0 + restitution) * geometric_sum)

    three_hit_speed = uniform_recurrent_speed(3)
    entry_hit_index = next(
        (
            index
            for index in range(3, max_band_entry_hit_index + 1)
            if uniform_recurrent_speed(index)
            <= comfortable_recurrent_racket_vz_m_s + 1.0e-12
        ),
        None,
    )
    if entry_hit_index is None:
        raise ValueError("band-entry apex is infeasible under speed budget")

    recurrent_speed = uniform_recurrent_speed(entry_hit_index)
    outgoing_vz = [launch_vz]
    for _ in range(entry_hit_index - 1):
        outgoing_vz.append(
            (1.0 + restitution) * recurrent_speed
            + restitution * outgoing_vz[-1]
        )
    nominal_vz = outgoing_speed(nominal_apex_z_m)
    required_vz = [launch_vz / (1.0 + restitution)]
    required_vz.extend([recurrent_speed] * (entry_hit_index - 1))
    required_vz.append(
        (nominal_vz - restitution * outgoing_vz[-1])
        / (1.0 + restitution)
    )
    if required_vz[-1] > comfortable_recurrent_racket_vz_m_s + 1.0e-12:
        raise ValueError("nominal apex handoff exceeds speed budget")
    apex_z = tuple(
        contact_z + velocity * velocity / (2.0 * gravity)
        for velocity in outgoing_vz
    ) + (float(nominal_apex_z_m),)
    return RapidSteadyApexSchedule(
        band_entry_hit_index=entry_hit_index,
        apex_z_m=apex_z,
        required_racket_vz_m_s=tuple(required_vz),
        three_hit_required_racket_vz_m_s=three_hit_speed,
    )

def design_hit3_hit4_apex_schedule(
    *,
    launch_apex_z_m: float,
    third_hit_apex_z_m: float,
    fourth_hit_apex_z_m: float,
    contact_ball_center_z_m: float,
    gravity_m_s2: float,
    effective_restitution: float,
) -> RapidSteadyApexSchedule:
    """Minimize peak hit-2/3 racket speed for fixed hit-3/4 goals.

    Equal hit-2 and hit-3 racket speeds minimize their maximum under the
    nominal vertical restitution recurrence. Hit 4 closes the apex gap.
    This is a motion-design target, not a contact-outcome guarantee.
    """

    values = (
        launch_apex_z_m, third_hit_apex_z_m, fourth_hit_apex_z_m,
        contact_ball_center_z_m, gravity_m_s2, effective_restitution,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("rapid apex schedule inputs must be finite")
    if not (
        contact_ball_center_z_m < launch_apex_z_m
        < third_hit_apex_z_m <= fourth_hit_apex_z_m
    ):
        raise ValueError("launch, third-hit and fourth-hit apex must be ordered")
    if gravity_m_s2 <= 0.0 or not 0.0 < effective_restitution < 1.0:
        raise ValueError("gravity and restitution are invalid")

    g = float(gravity_m_s2)
    e = float(effective_restitution)
    contact_z = float(contact_ball_center_z_m)

    def speed(apex_z_m: float) -> float:
        return math.sqrt(2.0 * g * (float(apex_z_m) - contact_z))

    launch_out = speed(launch_apex_z_m)
    third_out = speed(third_hit_apex_z_m)
    fourth_out = speed(fourth_hit_apex_z_m)
    recurrent_speed = (
        third_out - e * e * launch_out
    ) / ((1.0 + e) * (1.0 + e))
    second_out = e * launch_out + (1.0 + e) * recurrent_speed
    second_apex = contact_z + second_out * second_out / (2.0 * g)
    fourth_racket_speed = (fourth_out - e * third_out) / (1.0 + e)
    steady_racket_speed = fourth_out * (1.0 - e) / (1.0 + e)
    return RapidSteadyApexSchedule(
        band_entry_hit_index=3,
        apex_z_m=(
            float(launch_apex_z_m), second_apex,
            float(third_hit_apex_z_m), float(fourth_hit_apex_z_m),
            float(fourth_hit_apex_z_m),
        ),
        required_racket_vz_m_s=(
            launch_out / (1.0 + e), recurrent_speed, recurrent_speed,
            fourth_racket_speed, steady_racket_speed,
        ),
        three_hit_required_racket_vz_m_s=recurrent_speed,
    )
