"""Explicit114-D authority-gated actor residual; legacy actors never enter here."""


def apply_bound_gate(params, obs, teacher_mean, correction):
    """Keep the original acquisition mean exactly; full authority has full QVEL."""
    import jax
    import jax.numpy as jnp
    if obs.ndim < 2 or obs.shape[-1] != 114:
        raise ValueError('Wind authority gate requires the registered114-D observation')
    span_deg = jax.lax.stop_gradient(params['wind_pose_bound_gate_v1'])
    authority_deg = jnp.min(obs[..., -2:], axis=-1) * 20.
    gate = jnp.clip((authority_deg - 2.) / jnp.maximum(span_deg, 1e-6), 0., 1.)[..., None]
    # At full authority a clipped frozen baseline plus a +/-2 correction can
    # reach every normalized action in [-1,1]. No additional pose limit exists.
    baseline = teacher_mean + gate * (jnp.clip(teacher_mean, -1., 1.) - teacher_mean)
    return baseline, gate * correction
