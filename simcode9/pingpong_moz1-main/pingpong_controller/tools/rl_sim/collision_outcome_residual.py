#!/usr/bin/env python3
"""Versioned, opt-in post-impact velocity outcome residual.

The nominal MuJoCo/MJX contact solver remains authoritative unless callers
explicitly select ``dynamic_impact_v1``.  The residual is intended to run once,
after the ball has fully separated from the racket contact constraint.  It is
not a replacement contact force and it must not be used for a held/static ball
launch.

The v1 physical population is deliberately narrow: stationary-racket,
near-normal first rebounds of the measured 4 g ball.  The interface already
contains racket pose/twist, contact offset, and ball/session identity so later
datasets can identify those effects without changing the schema.  Coefficients
that the current data cannot identify remain zero or are represented only as
uncertainty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


MODEL_VERSION = "collision_outcome_residual_v1"
CALIBRATION_EVIDENCE_ID = "collision_outcome_residual_v1_20260902_r5"
MODE_OFF = "off"
MODE_DYNAMIC_IMPACT_V1 = "dynamic_impact_v1"
SUPPORTED_MODES = frozenset({MODE_OFF, MODE_DYNAMIC_IMPACT_V1})


def collision_outcome_event_edges(
    *,
    in_contact: Any,
    previous_contact: Any,
    pending_valid: Any,
    held_or_static: Any,
) -> tuple[Any, Any]:
    """Return physical contact/separation edges without reward semantics.

    The function deliberately uses only contact state and the mechanical hold
    gate.  Reward debounce, hit arming, pending reward confirmation, and hit
    counts must never suppress or duplicate a physical collision outcome.
    NumPy and JAX boolean arrays are both supported.
    """

    contact_edge = in_contact & (~previous_contact) & (~held_or_static)
    release_edge = (
        pending_valid
        & previous_contact
        & (~in_contact)
        & (~held_or_static)
    )
    return contact_edge, release_edge


def _finite_vector(name: str, value: Any, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {size}-vector")
    return array


def _finite_rotation(value: Any) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("racket_rotation_world must be a finite 3x3 matrix")
    gram = rotation.T @ rotation
    if not np.allclose(gram, np.eye(3), atol=2.0e-4, rtol=0.0):
        raise ValueError("racket_rotation_world must be orthonormal")
    if np.linalg.det(rotation) < 0.999:
        raise ValueError("racket_rotation_world must be right handed")
    return rotation


@dataclass(frozen=True)
class ImpactInputV1:
    """Pure data schema for one candidate dynamic ball-racket impact.

    All vectors use world coordinates except ``impact_offset_m``, whose two
    components are along columns 0 and 1 of ``racket_rotation_world``.  Column
    2 is the outward racket contact normal.
    """

    pre_ball_position_m: Any
    pre_ball_velocity_m_s: Any
    native_post_ball_velocity_m_s: Any
    racket_position_m: Any
    racket_rotation_world: Any
    racket_linear_velocity_m_s: Any
    racket_angular_velocity_rad_s: Any
    impact_offset_m: Any
    ball_identity_index: int
    session_identity_index: int
    valid: bool
    residual_application_delay_s: float = 0.0
    gravity_world_m_s2: Any = (0.0, 0.0, -9.81)

    def validated(self) -> "ImpactInputV1":
        return ImpactInputV1(
            pre_ball_position_m=_finite_vector(
                "pre_ball_position_m", self.pre_ball_position_m, 3
            ),
            pre_ball_velocity_m_s=_finite_vector(
                "pre_ball_velocity_m_s", self.pre_ball_velocity_m_s, 3
            ),
            native_post_ball_velocity_m_s=_finite_vector(
                "native_post_ball_velocity_m_s",
                self.native_post_ball_velocity_m_s,
                3,
            ),
            racket_position_m=_finite_vector(
                "racket_position_m", self.racket_position_m, 3
            ),
            racket_rotation_world=_finite_rotation(
                self.racket_rotation_world
            ),
            racket_linear_velocity_m_s=_finite_vector(
                "racket_linear_velocity_m_s",
                self.racket_linear_velocity_m_s,
                3,
            ),
            racket_angular_velocity_rad_s=_finite_vector(
                "racket_angular_velocity_rad_s",
                self.racket_angular_velocity_rad_s,
                3,
            ),
            impact_offset_m=_finite_vector(
                "impact_offset_m", self.impact_offset_m, 2
            ),
            ball_identity_index=int(self.ball_identity_index),
            session_identity_index=int(self.session_identity_index),
            valid=bool(self.valid),
            residual_application_delay_s=float(
                self.residual_application_delay_s
            ),
            gravity_world_m_s2=_finite_vector(
                "gravity_world_m_s2", self.gravity_world_m_s2, 3
            ),
        )


@dataclass(frozen=True)
class ImpactOutcomeV1:
    """One residual evaluation with diagnostics in SI units."""

    post_ball_velocity_m_s: np.ndarray
    velocity_residual_m_s: np.ndarray
    velocity_residual_local_m_s: np.ndarray
    predicted_post_relative_velocity_local_m_s: np.ndarray
    predictive_std_local_m_s: np.ndarray
    pre_relative_velocity_local_m_s: np.ndarray
    native_post_relative_velocity_local_m_s: np.ndarray
    racket_contact_point_velocity_m_s: np.ndarray
    incoming_normal_speed_m_s: float
    applied: bool
    saturated: bool
    outside_calibrated_population: bool
    reason: str


@dataclass(frozen=True)
class CollisionOutcomeResidualV1Config:
    """Frozen v1 model and structured posterior-sampling contract."""

    mode: str = MODE_OFF
    model_version: str = MODEL_VERSION
    calibration_evidence_id: str = CALIBRATION_EVIDENCE_ID
    # Real first-bounce fit.  The default-off setting is the compatibility
    # contract; these values matter only for explicit diagnostic activation.
    normal_restitution_reference: float = 0.6158787369
    normal_restitution_speed_reference_m_s: float = 2.4241343454
    normal_restitution_speed_slope_s_m: float = 0.0
    normal_restitution_bounds: tuple[float, float] = (0.45, 0.80)
    nominal_solver_solref: tuple[float, float] = (0.003, 0.80)
    nominal_ball_friction: float = 0.20
    nominal_racket_friction: float = 0.35
    # target tangent relative velocity = gain * pre tangent + bias.  Offset,
    # spin, ball identity, and session coefficients remain absent in v1 because
    # the supplied vertical drops cannot identify them.
    tangential_target_gain: tuple[float, float] = (0.0, 0.0)
    tangential_target_bias_m_s: tuple[float, float] = (
        -0.03138641140123787,
        0.10448988571194456,
    )
    normal_velocity_residual_bound_m_s: float = 2.0
    tangential_velocity_residual_bound_m_s: tuple[float, float] = (1.0, 0.80)
    # Frozen observed population envelope.  Out-of-population inputs fail
    # closed to nominal contact instead of silently extrapolating.
    calibrated_incoming_normal_speed_range_m_s: tuple[float, float] = (
        1.75,
        2.75,
    )
    calibrated_pre_tangent_speed_max_m_s: float = 0.20
    calibrated_impact_offset_max_m: float = 0.040
    calibrated_racket_contact_speed_max_m_s: float = 0.10
    calibrated_racket_angular_speed_max_rad_s: float = 0.30
    # Posterior components are deliberately separate: one coherent episode
    # restitution latent and one independent event velocity residual.
    posterior_sampling: bool = False
    episode_restitution_std: float = 0.0
    event_velocity_std_local_m_s: tuple[float, float, float] = (
        0.07882617420170356,
        0.04827496066594485,
        0.02472083755972077,
    )
    predictive_model_std_local_m_s: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )

    def __post_init__(self) -> None:
        if self.model_version != MODEL_VERSION:
            raise ValueError(
                f"model_version must be {MODEL_VERSION!r}, got "
                f"{self.model_version!r}"
            )
        if self.calibration_evidence_id != CALIBRATION_EVIDENCE_ID:
            raise ValueError(
                "calibration_evidence_id must be "
                f"{CALIBRATION_EVIDENCE_ID!r}, got "
                f"{self.calibration_evidence_id!r}"
            )
        if self.mode not in SUPPORTED_MODES:
            if "static" in str(self.mode).lower() or "launch" in str(
                self.mode
            ).lower():
                raise ValueError(
                    "static/low-speed launch requires a separate model and "
                    "must not alias dynamic_impact_v1"
                )
            raise ValueError(
                f"unsupported collision outcome mode {self.mode!r}; "
                f"expected one of {sorted(SUPPORTED_MODES)}"
            )
        self._validate_range(
            "normal_restitution_bounds", self.normal_restitution_bounds
        )
        self._validate_range(
            "calibrated_incoming_normal_speed_range_m_s",
            self.calibrated_incoming_normal_speed_range_m_s,
        )
        for name, value in (
            ("normal_restitution_reference", self.normal_restitution_reference),
            (
                "normal_restitution_speed_reference_m_s",
                self.normal_restitution_speed_reference_m_s,
            ),
            (
                "normal_restitution_speed_slope_s_m",
                self.normal_restitution_speed_slope_s_m,
            ),
            (
                "normal_velocity_residual_bound_m_s",
                self.normal_velocity_residual_bound_m_s,
            ),
            ("nominal_ball_friction", self.nominal_ball_friction),
            ("nominal_racket_friction", self.nominal_racket_friction),
            (
                "calibrated_pre_tangent_speed_max_m_s",
                self.calibrated_pre_tangent_speed_max_m_s,
            ),
            (
                "calibrated_impact_offset_max_m",
                self.calibrated_impact_offset_max_m,
            ),
            (
                "calibrated_racket_contact_speed_max_m_s",
                self.calibrated_racket_contact_speed_max_m_s,
            ),
            (
                "calibrated_racket_angular_speed_max_rad_s",
                self.calibrated_racket_angular_speed_max_rad_s,
            ),
            ("episode_restitution_std", self.episode_restitution_std),
        ):
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        for name, value, size in (
            ("tangential_target_gain", self.tangential_target_gain, 2),
            (
                "tangential_target_bias_m_s",
                self.tangential_target_bias_m_s,
                2,
            ),
            (
                "tangential_velocity_residual_bound_m_s",
                self.tangential_velocity_residual_bound_m_s,
                2,
            ),
            (
                "event_velocity_std_local_m_s",
                self.event_velocity_std_local_m_s,
                3,
            ),
            (
                "predictive_model_std_local_m_s",
                self.predictive_model_std_local_m_s,
                3,
            ),
            ("nominal_solver_solref", self.nominal_solver_solref, 2),
        ):
            array = _finite_vector(name, value, size)
            if name in {
                "tangential_velocity_residual_bound_m_s",
                "event_velocity_std_local_m_s",
                "predictive_model_std_local_m_s",
                "nominal_solver_solref",
            } and np.any(array < 0.0):
                raise ValueError(f"{name} must be non-negative")
        if float(self.normal_velocity_residual_bound_m_s) <= 0.0:
            raise ValueError(
                "normal_velocity_residual_bound_m_s must be positive"
            )
        for name in (
            "calibrated_pre_tangent_speed_max_m_s",
            "calibrated_impact_offset_max_m",
            "calibrated_racket_contact_speed_max_m_s",
            "calibrated_racket_angular_speed_max_rad_s",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")

    @staticmethod
    def _validate_range(name: str, value: Any) -> None:
        bounds = _finite_vector(name, value, 2)
        if not bounds[0] < bounds[1]:
            raise ValueError(f"{name} must be strictly increasing")


def _numpy_seed(seed: int, stream: int, index: int = 0) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        [int(seed) & 0xFFFFFFFF, int(stream), int(index)]
    )
    return np.random.default_rng(sequence)


class CollisionOutcomeResidualV1:
    """NumPy facade used by calibration, manifests, and unit tests."""

    def __init__(self, config: CollisionOutcomeResidualV1Config) -> None:
        self.config = config

    def sample_episode_latent_numpy(self, seed: int) -> np.ndarray:
        if not self.config.posterior_sampling:
            return np.zeros(1, dtype=np.float64)
        return _numpy_seed(seed, 0xE9150DE).normal(
            0.0,
            float(self.config.episode_restitution_std),
            size=1,
        )

    def sample_event_noise_numpy(
        self, seed: int, impact_index: int
    ) -> np.ndarray:
        if not self.config.posterior_sampling:
            return np.zeros(3, dtype=np.float64)
        standard_deviation = np.asarray(
            self.config.event_velocity_std_local_m_s,
            dtype=np.float64,
        )
        return _numpy_seed(seed, 0xE7E170, impact_index).normal(
            0.0,
            standard_deviation,
            size=3,
        )

    def apply_numpy(
        self,
        impact: ImpactInputV1,
        *,
        event_gate: bool = True,
        episode_latent: Any | None = None,
        event_noise_local_m_s: Any | None = None,
    ) -> ImpactOutcomeV1:
        impact = impact.validated()
        if (
            not np.isfinite(impact.residual_application_delay_s)
            or impact.residual_application_delay_s < 0.0
            or impact.residual_application_delay_s > 0.050
        ):
            raise ValueError(
                "residual_application_delay_s must be finite and in [0, 0.050]"
            )
        native = np.asarray(
            impact.native_post_ball_velocity_m_s, dtype=np.float64
        )
        zeros = np.zeros(3, dtype=np.float64)
        rotation = np.asarray(impact.racket_rotation_world, dtype=np.float64)
        offset_world = rotation[:, :2] @ np.asarray(
            impact.impact_offset_m, dtype=np.float64
        )
        contact_point_velocity = np.asarray(
            impact.racket_linear_velocity_m_s, dtype=np.float64
        ) + np.cross(
            np.asarray(impact.racket_angular_velocity_rad_s, dtype=np.float64),
            offset_world,
        )
        pre_relative_local = rotation.T @ (
            np.asarray(impact.pre_ball_velocity_m_s, dtype=np.float64)
            - contact_point_velocity
        )
        native_relative_local = rotation.T @ (
            native - contact_point_velocity
        )
        incoming = max(0.0, -float(pre_relative_local[2]))

        def unchanged(reason: str, *, outside: bool = False) -> ImpactOutcomeV1:
            return ImpactOutcomeV1(
                post_ball_velocity_m_s=native.copy(),
                velocity_residual_m_s=zeros.copy(),
                velocity_residual_local_m_s=zeros.copy(),
                predicted_post_relative_velocity_local_m_s=(
                    native_relative_local.copy()
                ),
                predictive_std_local_m_s=zeros.copy(),
                pre_relative_velocity_local_m_s=pre_relative_local.copy(),
                native_post_relative_velocity_local_m_s=(
                    native_relative_local.copy()
                ),
                racket_contact_point_velocity_m_s=(
                    contact_point_velocity.copy()
                ),
                incoming_normal_speed_m_s=incoming,
                applied=False,
                saturated=False,
                outside_calibrated_population=outside,
                reason=reason,
            )

        if self.config.mode == MODE_OFF:
            return unchanged("mode_off")
        if not event_gate:
            return unchanged("event_gate_closed")
        if not impact.valid:
            return unchanged("invalid_impact")

        low, high = self.config.calibrated_incoming_normal_speed_range_m_s
        outside = (
            incoming < float(low)
            or incoming > float(high)
            or float(np.linalg.norm(pre_relative_local[:2]))
            > float(self.config.calibrated_pre_tangent_speed_max_m_s)
            or float(np.linalg.norm(impact.impact_offset_m))
            > float(self.config.calibrated_impact_offset_max_m)
            or float(np.linalg.norm(contact_point_velocity))
            > float(self.config.calibrated_racket_contact_speed_max_m_s)
            or float(np.linalg.norm(impact.racket_angular_velocity_rad_s))
            > float(self.config.calibrated_racket_angular_speed_max_rad_s)
        )
        if outside:
            return unchanged(
                "outside_calibrated_population", outside=True
            )

        if episode_latent is None:
            episode_delta = 0.0
        else:
            episode_delta = float(
                _finite_vector("episode_latent", episode_latent, 1)[0]
            )
        if event_noise_local_m_s is None:
            event_noise = zeros
        else:
            event_noise = _finite_vector(
                "event_noise_local_m_s", event_noise_local_m_s, 3
            )
        restitution = (
            float(self.config.normal_restitution_reference)
            + float(self.config.normal_restitution_speed_slope_s_m)
            * (
                incoming
                - float(
                    self.config.normal_restitution_speed_reference_m_s
                )
            )
            + episode_delta
        )
        restitution = float(
            np.clip(restitution, *self.config.normal_restitution_bounds)
        )
        target_relative_local = np.empty(3, dtype=np.float64)
        target_relative_local[:2] = (
            np.asarray(self.config.tangential_target_gain, dtype=np.float64)
            * pre_relative_local[:2]
            + np.asarray(
                self.config.tangential_target_bias_m_s, dtype=np.float64
            )
        )
        target_relative_local[2] = restitution * incoming
        target_relative_local += event_noise
        target_world_at_contact = (
            contact_point_velocity + rotation @ target_relative_local
        )
        target_world_at_application = (
            target_world_at_contact
            + np.asarray(impact.gravity_world_m_s2, dtype=np.float64)
            * float(impact.residual_application_delay_s)
        )
        target_relative_local = rotation.T @ (
            target_world_at_application - contact_point_velocity
        )

        raw_residual_local = target_relative_local - native_relative_local
        bounds = np.asarray(
            [
                *self.config.tangential_velocity_residual_bound_m_s,
                self.config.normal_velocity_residual_bound_m_s,
            ],
            dtype=np.float64,
        )
        residual_local = np.clip(raw_residual_local, -bounds, bounds)
        saturated = bool(
            np.any(np.abs(raw_residual_local - residual_local) > 1.0e-12)
        )
        residual_world = rotation @ residual_local
        post = native + residual_world
        model_std = np.asarray(
            self.config.predictive_model_std_local_m_s, dtype=np.float64
        )
        event_std = np.asarray(
            self.config.event_velocity_std_local_m_s, dtype=np.float64
        )
        predictive_std = np.sqrt(model_std**2 + event_std**2)
        predictive_std[2] = float(
            np.sqrt(
                predictive_std[2] ** 2
                + (incoming * self.config.episode_restitution_std) ** 2
            )
        )
        return ImpactOutcomeV1(
            post_ball_velocity_m_s=post,
            velocity_residual_m_s=residual_world,
            velocity_residual_local_m_s=residual_local,
            predicted_post_relative_velocity_local_m_s=(
                target_relative_local
            ),
            predictive_std_local_m_s=predictive_std,
            pre_relative_velocity_local_m_s=pre_relative_local,
            native_post_relative_velocity_local_m_s=native_relative_local,
            racket_contact_point_velocity_m_s=contact_point_velocity,
            incoming_normal_speed_m_s=incoming,
            applied=True,
            saturated=saturated,
            outside_calibrated_population=False,
            reason="applied",
        )


def apply_collision_outcome_residual_jax(
    config: CollisionOutcomeResidualV1Config,
    *,
    pre_ball_velocity_m_s: Any,
    native_post_ball_velocity_m_s: Any,
    racket_rotation_world: Any,
    racket_linear_velocity_m_s: Any,
    racket_angular_velocity_rad_s: Any,
    impact_offset_m: Any,
    episode_restitution_latent: Any,
    event_noise_local_m_s: Any,
    event_gate: Any,
    valid: Any,
    residual_application_delay_s: Any,
    gravity_world_m_s2: Any,
) -> tuple[Any, ...]:
    """JAX-compatible batched kernel used by :mod:`mjx_juggle_env`.

    The caller must bypass this function entirely in off mode.  That Python
    guard is what preserves the historical transition path without an extra
    forward pass or floating-point operation.
    """

    import jax.numpy as jnp

    rotation = jnp.asarray(racket_rotation_world)
    pre_velocity = jnp.asarray(pre_ball_velocity_m_s)
    native_velocity = jnp.asarray(native_post_ball_velocity_m_s)
    racket_velocity = jnp.asarray(racket_linear_velocity_m_s)
    racket_omega = jnp.asarray(racket_angular_velocity_rad_s)
    offset_local = jnp.concatenate(
        (
            jnp.asarray(impact_offset_m),
            jnp.zeros((*jnp.asarray(impact_offset_m).shape[:-1], 1)),
        ),
        axis=-1,
    )
    offset_world = jnp.einsum("...ij,...j->...i", rotation, offset_local)
    contact_velocity = racket_velocity + jnp.cross(
        racket_omega, offset_world
    )
    rotation_transpose = jnp.swapaxes(rotation, -1, -2)
    pre_relative_local = jnp.einsum(
        "...ij,...j->...i",
        rotation_transpose,
        pre_velocity - contact_velocity,
    )
    native_relative_local = jnp.einsum(
        "...ij,...j->...i",
        rotation_transpose,
        native_velocity - contact_velocity,
    )
    incoming = jnp.maximum(0.0, -pre_relative_local[..., 2])
    speed_bounds = config.calibrated_incoming_normal_speed_range_m_s
    in_population = (
        (incoming >= float(speed_bounds[0]))
        & (incoming <= float(speed_bounds[1]))
        & (
            jnp.linalg.norm(pre_relative_local[..., :2], axis=-1)
            <= float(config.calibrated_pre_tangent_speed_max_m_s)
        )
        & (
            jnp.linalg.norm(jnp.asarray(impact_offset_m), axis=-1)
            <= float(config.calibrated_impact_offset_max_m)
        )
        & (
            jnp.linalg.norm(contact_velocity, axis=-1)
            <= float(config.calibrated_racket_contact_speed_max_m_s)
        )
        & (
            jnp.linalg.norm(racket_omega, axis=-1)
            <= float(config.calibrated_racket_angular_speed_max_rad_s)
        )
    )
    apply = jnp.asarray(event_gate) & jnp.asarray(valid) & in_population
    restitution = (
        float(config.normal_restitution_reference)
        + float(config.normal_restitution_speed_slope_s_m)
        * (
            incoming
            - float(config.normal_restitution_speed_reference_m_s)
        )
        + jnp.asarray(episode_restitution_latent)
    )
    restitution = jnp.clip(
        restitution,
        float(config.normal_restitution_bounds[0]),
        float(config.normal_restitution_bounds[1]),
    )
    target_tangent = (
        jnp.asarray(config.tangential_target_gain)
        * pre_relative_local[..., :2]
        + jnp.asarray(config.tangential_target_bias_m_s)
    )
    target_local_at_contact = jnp.concatenate(
        (target_tangent, (restitution * incoming)[..., None]), axis=-1
    ) + jnp.asarray(event_noise_local_m_s)
    target_world_at_contact = contact_velocity + jnp.einsum(
        "...ij,...j->...i", rotation, target_local_at_contact
    )
    target_world_at_application = (
        target_world_at_contact
        + jnp.asarray(gravity_world_m_s2)
        * jnp.asarray(residual_application_delay_s)[..., None]
    )
    target_local = jnp.einsum(
        "...ij,...j->...i",
        rotation_transpose,
        target_world_at_application - contact_velocity,
    )
    raw_residual_local = target_local - native_relative_local
    residual_bounds = jnp.asarray(
        (
            *config.tangential_velocity_residual_bound_m_s,
            config.normal_velocity_residual_bound_m_s,
        )
    )
    clipped_residual_local = jnp.clip(
        raw_residual_local, -residual_bounds, residual_bounds
    )
    saturated = jnp.any(
        jnp.abs(raw_residual_local - clipped_residual_local) > 1.0e-7,
        axis=-1,
    )
    residual_local = jnp.where(
        apply[..., None], clipped_residual_local, 0.0
    )
    residual_world = jnp.einsum(
        "...ij,...j->...i", rotation, residual_local
    )
    post_velocity = native_velocity + residual_world
    model_std = jnp.asarray(config.predictive_model_std_local_m_s)
    event_std = jnp.asarray(config.event_velocity_std_local_m_s)
    predictive_std = jnp.sqrt(model_std * model_std + event_std * event_std)
    normal_std = jnp.sqrt(
        predictive_std[2] * predictive_std[2]
        + (
            incoming * float(config.episode_restitution_std)
        )
        ** 2
    )
    predictive_std = jnp.broadcast_to(
        predictive_std, pre_relative_local.shape
    ).at[..., 2].set(normal_std)
    predictive_std = jnp.where(apply[..., None], predictive_std, 0.0)
    return (
        post_velocity,
        residual_world,
        residual_local,
        target_local,
        predictive_std,
        pre_relative_local,
        native_relative_local,
        contact_velocity,
        incoming,
        apply,
        (~in_population) & jnp.asarray(event_gate) & jnp.asarray(valid),
        saturated & apply,
    )
