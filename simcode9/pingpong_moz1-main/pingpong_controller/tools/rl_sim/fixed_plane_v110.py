"""Reward-only safety shaping for V110's measured joint acceleration."""

import jax.numpy as jnp


def soft_joint_acceleration_margin_penalty(
    arm_acc_ratio: jnp.ndarray,
    *,
    margin_ratio: float,
    weight: float,
    control_dt_s: float,
) -> jnp.ndarray:
    """Return per-lane cost when absolute joint qacc nears its hard limit."""

    excess = jnp.maximum(arm_acc_ratio - margin_ratio, 0.0)
    return -weight * control_dt_s * jnp.mean(excess, axis=-1)
