"""Recovered acceleration-level RMP/QP core.

The equations in ``build_joint_qp`` and ``build_task_qp`` are direct
translations of the corresponding optimized C++ routines.  Kinematics are an
input because the private ``rkdl`` implementation is not part of the copied
runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .params import RMPParams

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class JointLimits:
    q_min: FloatArray
    q_max: FloatArray
    qd_max: FloatArray
    qdd_max: FloatArray
    qddd_max: Optional[FloatArray] = None

    def __post_init__(self) -> None:
        arrays = (self.q_min, self.q_max, self.qd_max, self.qdd_max)
        converted = tuple(np.asarray(v, dtype=float).reshape(-1) for v in arrays)
        size = converted[0].size
        if any(v.size != size for v in converted):
            raise ValueError("all joint limit arrays must have the same length")
        object.__setattr__(self, "q_min", converted[0])
        object.__setattr__(self, "q_max", converted[1])
        object.__setattr__(self, "qd_max", converted[2])
        object.__setattr__(self, "qdd_max", converted[3])
        if self.qddd_max is not None:
            jerk = np.asarray(self.qddd_max, dtype=float).reshape(-1)
            if jerk.size != size:
                raise ValueError("qddd_max must match the other joint limits")
            object.__setattr__(self, "qddd_max", jerk)


@dataclass(frozen=True)
class QPProblem:
    """Box-constrained QP: min 0.5*x.T*H*x + g.T*x."""

    H: FloatArray
    g: FloatArray
    lower: FloatArray
    upper: FloatArray


@dataclass(frozen=True)
class RMPResult:
    qdd: FloatArray
    problem: QPProblem
    reference_acceleration: FloatArray
    task_error: Optional[FloatArray] = None
    psi_gain: float = 0.0


class JointBoundsInfeasible(RuntimeError):
    """The current state cannot satisfy the recovered safety envelope."""


def _vector(value: ArrayLike, name: str, size: Optional[int] = None) -> FloatArray:
    result = np.asarray(value, dtype=float).reshape(-1)
    if size is not None and result.size != size:
        raise ValueError(f"{name} must have length {size}, got {result.size}")
    return result


def wrap_to_pi(angle: ArrayLike) -> FloatArray | float:
    value = np.asarray(angle, dtype=float)
    wrapped = value - np.round(value / (2.0 * np.pi)) * (2.0 * np.pi)
    if wrapped.ndim == 0:
        return float(wrapped)
    return wrapped


def _clip_norm(value: FloatArray, maximum: float) -> FloatArray:
    norm = float(np.linalg.norm(value))
    if norm > maximum and norm > 0.0:
        return value * (maximum / norm)
    return value


def so3_log(rotation: ArrayLike) -> FloatArray:
    """Return the rotation vector of a 3x3 rotation matrix."""

    R = np.asarray(rotation, dtype=float).reshape(3, 3)
    cos_angle = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    skew = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if angle < 1.0e-8:
        return 0.5 * skew
    if np.pi - angle < 1.0e-5:
        eigvals, eigvecs = np.linalg.eigh((R + np.eye(3)) * 0.5)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] * skew[dominant] < 0.0:
            axis = -axis
        return angle * axis
    return angle / (2.0 * np.sin(angle)) * skew


def frame_error(
    current_position: ArrayLike,
    current_rotation: ArrayLike,
    target_position: ArrayLike,
    target_rotation: ArrayLike,
) -> FloatArray:
    """Construct ``[translation, rotation-vector]`` target error.

    The rotation error is expressed in the world/spatial frame.  A simulator
    using a body-frame Jacobian should rotate this error and all twists into
    that same frame before calling the QP builder.
    """

    current_R = np.asarray(current_rotation, dtype=float).reshape(3, 3)
    target_R = np.asarray(target_rotation, dtype=float).reshape(3, 3)
    translation = _vector(target_position, "target_position", 3) - _vector(
        current_position, "current_position", 3
    )
    return np.concatenate((translation, so3_log(target_R @ current_R.T)))


def _regulated_feedforward(configured: float, speed: float, threshold: float) -> float:
    """Low-speed feed-forward reduction recovered from assembly."""

    if speed < threshold:
        # The binary constants are 0.25 and (0.75 / threshold): 15 for
        # translation, about 2.1428 for rotation.
        return configured * (0.25 + 0.75 * speed / threshold)
    return configured


def compute_task_reference(
    error: ArrayLike,
    current_twist: ArrayLike,
    target_twist: ArrayLike,
    target_acceleration: ArrayLike,
    params: RMPParams,
    dt: float,
    *,
    psi: Optional[float] = None,
    psi_dot: float = 0.0,
    psi_target: float = 0.0,
    algorithmic_singularity: float = 1.0,
) -> tuple[FloatArray, FloatArray, float]:
    """Recover the desired Cartesian (and optional arm-angle) acceleration."""

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    e = _vector(error, "error", 6).copy()
    V = _vector(current_twist, "current_twist", 6)
    Vd = _vector(target_twist, "target_twist", 6)
    Vd_dot = _vector(target_acceleration, "target_acceleration", 6)

    # The literal factor 200.0 is present in both the independent and whole-body
    # planner.  It bounds the error admitted to the PD term per controller tick.
    e[:3] = _clip_norm(e[:3], params.max_linear_vel * dt * 200.0)
    e[3:] = _clip_norm(e[3:], params.max_angular_vel * dt * 200.0)

    linear_ffd = _regulated_feedforward(params.linear_vel_ffd, float(np.linalg.norm(V[:3])), 0.05)
    angular_ffd = _regulated_feedforward(params.angular_vel_ffd, float(np.linalg.norm(V[3:])), 0.35)

    reference = np.empty(6, dtype=float)
    reference[:3] = (
        params.kp * e[:3]
        + params.kd * (linear_ffd * Vd[:3] - V[:3])
        + params.linear_acc_ffd * Vd_dot[:3]
    )
    reference[3:] = (
        params.kp * e[3:]
        + params.kd * (angular_ffd * Vd[3:] - V[3:])
        + params.angular_acc_ffd * Vd_dot[3:]
    )
    reference[:3] = _clip_norm(reference[:3], params.max_linear_acc)
    reference[3:] = _clip_norm(reference[3:], params.max_angular_acc)

    psi_gain = 0.0
    if params.use_arm_angle and psi is not None:
        # The binary gates arm-angle control as it approaches an algorithmic
        # singularity: zero below 0.1, linear from 0.1 to 1.0, then one.
        singularity = float(algorithmic_singularity)
        psi_gain = 0.0 if singularity < 0.1 else min(singularity, 1.0)
        psi_acc = params.psi_kp * wrap_to_pi(psi_target - psi) - params.psi_kd * psi_dot
        psi_acc = float(np.clip(psi_acc, -params.max_psi_acc, params.max_psi_acc))
        reference = np.concatenate((reference, [psi_gain * psi_acc]))

    return reference, e, psi_gain


def compute_joint_reference(
    q: ArrayLike,
    qd: ArrayLike,
    q_target: ArrayLike,
    qd_target: ArrayLike,
    qdd_target: ArrayLike,
    qdd_max: ArrayLike,
    params: RMPParams,
) -> FloatArray:
    q = _vector(q, "q")
    size = q.size
    qd = _vector(qd, "qd", size)
    q_target = _vector(q_target, "q_target", size)
    qd_target = _vector(qd_target, "qd_target", size)
    qdd_target = _vector(qdd_target, "qdd_target", size)
    qdd_max = _vector(qdd_max, "qdd_max", size)
    reference = (
        params.jnt_kp * (q_target - q)
        + params.jnt_kd * (params.joint_vel_ffd * qd_target - qd)
        + params.joint_acc_ffd * qdd_target
    )
    return np.clip(reference, -qdd_max, qdd_max)


def joint_acceleration_bounds(
    q: ArrayLike,
    qd: ArrayLike,
    limits: JointLimits,
    dt: float,
    *,
    angle_gap: ArrayLike | float = 0.0,
    qdd_previous: Optional[ArrayLike] = None,
    use_jerk_limit: bool = True,
    strict: bool = True,
) -> tuple[FloatArray, FloatArray]:
    """Reconstruct the planner's predictive joint safety envelope.

    The binary uses a piecewise quadratic root calculation with a 0.998 safety
    factor.  This implementation expresses the same braking invariant in its
    closed form and intersects it with velocity, acceleration and optional jerk
    limits.  This routine is the one deliberately equivalent reconstruction in
    the core; see ``RECOVERY_EVIDENCE.md``.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    q = _vector(q, "q", limits.q_min.size)
    qd = _vector(qd, "qd", q.size)
    gap = np.broadcast_to(np.asarray(angle_gap, dtype=float), q.shape)
    safe_min = limits.q_min + gap
    safe_max = limits.q_max - gap
    amax = np.maximum(limits.qdd_max, 0.0)

    lower = -amax.copy()
    upper = amax.copy()

    finite_upper = np.isfinite(safe_max) & np.isfinite(amax)
    finite_lower = np.isfinite(safe_min) & np.isfinite(amax)
    braking = 0.998 * amax
    velocity_upper = limits.qd_max.copy()
    velocity_lower = -limits.qd_max.copy()
    velocity_upper[finite_upper] = np.minimum(
        velocity_upper[finite_upper],
        np.sqrt(2.0 * braking[finite_upper] * np.maximum(safe_max[finite_upper] - q[finite_upper], 0.0)),
    )
    velocity_lower[finite_lower] = np.maximum(
        velocity_lower[finite_lower],
        -np.sqrt(2.0 * braking[finite_lower] * np.maximum(q[finite_lower] - safe_min[finite_lower], 0.0)),
    )
    lower = np.maximum(lower, (velocity_lower - qd) / dt)
    upper = np.minimum(upper, (velocity_upper - qd) / dt)

    if use_jerk_limit and qdd_previous is not None and limits.qddd_max is not None:
        previous = _vector(qdd_previous, "qdd_previous", q.size)
        lower = np.maximum(lower, previous - limits.qddd_max * dt)
        upper = np.minimum(upper, previous + limits.qddd_max * dt)

    invalid = lower > upper
    if np.any(invalid):
        crossing = lower - upper
        tiny = invalid & (crossing < 1.0e-4)
        # Exact recovery of the numerical repair at C++ lines 821--824.
        center = 0.5 * (lower + upper)
        lower[tiny] = center[tiny] - 1.0e-8
        upper[tiny] = center[tiny] + 1.0e-8
        invalid = lower > upper
        if np.any(invalid) and strict:
            indices = np.flatnonzero(invalid).tolist()
            raise JointBoundsInfeasible(
                f"no safe acceleration interval for joints {indices}; "
                "the current position/velocity is outside the recoverable envelope"
            )
        if np.any(invalid):
            # Training-only recovery mode: command the strongest feasible brake.
            brake = np.clip(np.zeros_like(lower), upper, lower)
            lower[invalid] = brake[invalid]
            upper[invalid] = brake[invalid]
    return lower, upper


def build_joint_qp(reference_acceleration: ArrayLike, params: RMPParams, lower: ArrayLike, upper: ArrayLike) -> QPProblem:
    reference = _vector(reference_acceleration, "reference_acceleration")
    size = reference.size
    # Exact recovered construction: H starts as I, acc_weight is added to its
    # diagonal, and g is the negative clipped reference acceleration.
    H = np.eye(size, dtype=float) * (1.0 + params.acc_weight)
    return QPProblem(H, -reference, _vector(lower, "lower", size), _vector(upper, "upper", size))


def build_task_qp(
    jacobian: ArrayLike,
    jacobian_dot_qd: ArrayLike,
    reference_acceleration: ArrayLike,
    q: ArrayLike,
    qd: ArrayLike,
    q_mid: ArrayLike,
    params: RMPParams,
    dt: float,
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    psi_gain: float = 0.0,
) -> QPProblem:
    J = np.asarray(jacobian, dtype=float)
    if J.ndim != 2:
        raise ValueError("jacobian must be a matrix")
    task_size, joint_size = J.shape
    q = _vector(q, "q", joint_size)
    qd = _vector(qd, "qd", joint_size)
    q_mid = _vector(q_mid, "q_mid", joint_size)
    Jdqd = _vector(jacobian_dot_qd, "jacobian_dot_qd", task_size)
    reference = _vector(reference_acceleration, "reference_acceleration", task_size)

    residual = reference - Jdqd
    H = J.T @ J
    H.flat[:: joint_size + 1] += params.vel_weight * dt * dt + params.acc_weight
    g = -J.T @ residual
    g += params.vel_weight * dt * qd
    g += 4.0 * params.pos_limit_weight * abs(1.0 - float(psi_gain)) * (q - q_mid)
    return QPProblem(H, g, _vector(lower, "lower", joint_size), _vector(upper, "upper", joint_size))


def solve_box_qp(problem: QPProblem, *, tolerance: float = 1.0e-10, max_iterations: int = 100) -> FloatArray:
    """Solve a small strictly-convex box QP with an active-set method."""

    H = np.asarray(problem.H, dtype=float)
    g = _vector(problem.g, "g")
    lower = _vector(problem.lower, "lower", g.size)
    upper = _vector(problem.upper, "upper", g.size)
    if H.shape != (g.size, g.size):
        raise ValueError("H has the wrong shape")
    if np.any(lower > upper):
        raise ValueError("infeasible box bounds")
    H = 0.5 * (H + H.T)
    regularization = max(1.0, float(np.max(np.abs(np.diag(H))))) * 1.0e-12

    try:
        x = np.linalg.solve(H, -g)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(H + regularization * np.eye(g.size), -g, rcond=None)[0]
    x = np.clip(x, lower, upper)
    active = np.zeros(g.size, dtype=np.int8)  # -1 lower, +1 upper, 0 free

    for _ in range(max_iterations):
        gradient = H @ x + g
        at_lower = x <= lower + tolerance
        at_upper = x >= upper - tolerance
        active[(at_lower) & (gradient >= -tolerance)] = -1
        active[(at_upper) & (gradient <= tolerance)] = 1

        # Release a bound that violates the KKT sign condition.
        violations = np.zeros(g.size, dtype=float)
        violations[(active == -1)] = np.maximum(-gradient[(active == -1)], 0.0)
        violations[(active == 1)] = np.maximum(gradient[(active == 1)], 0.0)
        if float(np.max(violations, initial=0.0)) > tolerance:
            active[int(np.argmax(violations))] = 0

        fixed = active != 0
        free = ~fixed
        x[active == -1] = lower[active == -1]
        x[active == 1] = upper[active == 1]
        if np.any(free):
            rhs = -g[free]
            if np.any(fixed):
                rhs -= H[np.ix_(free, fixed)] @ x[fixed]
            Hff = H[np.ix_(free, free)]
            try:
                candidate = np.linalg.solve(Hff, rhs)
            except np.linalg.LinAlgError:
                candidate = np.linalg.lstsq(Hff + regularization * np.eye(Hff.shape[0]), rhs, rcond=None)[0]
            x[free] = candidate

        below = x < lower - tolerance
        above = x > upper + tolerance
        if np.any(below | above):
            x = np.clip(x, lower, upper)
            active[below] = -1
            active[above] = 1
            continue

        gradient = H @ x + g
        free_ok = not np.any(np.abs(gradient[active == 0]) > tolerance * 10.0)
        lower_ok = not np.any(gradient[active == -1] < -tolerance * 10.0)
        upper_ok = not np.any(gradient[active == 1] > tolerance * 10.0)
        if free_ok and lower_ok and upper_ok:
            return x

    raise RuntimeError("box QP active-set solver did not converge")


def solve_joint_rmp(
    q: ArrayLike,
    qd: ArrayLike,
    q_target: ArrayLike,
    qd_target: ArrayLike,
    qdd_target: ArrayLike,
    limits: JointLimits,
    params: RMPParams,
    dt: float,
    *,
    qdd_previous: Optional[ArrayLike] = None,
) -> RMPResult:
    reference = compute_joint_reference(q, qd, q_target, qd_target, qdd_target, limits.qdd_max, params)
    lower, upper = joint_acceleration_bounds(
        q, qd, limits, dt, angle_gap=params.angle_gap, qdd_previous=qdd_previous
    )
    problem = build_joint_qp(reference, params, lower, upper)
    return RMPResult(solve_box_qp(problem), problem, reference)


def solve_task_rmp(
    q: ArrayLike,
    qd: ArrayLike,
    q_mid: ArrayLike,
    limits: JointLimits,
    jacobian: ArrayLike,
    jacobian_dot_qd: ArrayLike,
    error: ArrayLike,
    current_twist: ArrayLike,
    target_twist: ArrayLike,
    target_acceleration: ArrayLike,
    params: RMPParams,
    dt: float,
    *,
    qdd_previous: Optional[ArrayLike] = None,
    psi: Optional[float] = None,
    psi_dot: float = 0.0,
    psi_target: float = 0.0,
    algorithmic_singularity: float = 1.0,
) -> RMPResult:
    reference, limited_error, psi_gain = compute_task_reference(
        error,
        current_twist,
        target_twist,
        target_acceleration,
        params,
        dt,
        psi=psi,
        psi_dot=psi_dot,
        psi_target=psi_target,
        algorithmic_singularity=algorithmic_singularity,
    )
    lower, upper = joint_acceleration_bounds(
        q, qd, limits, dt, angle_gap=params.angle_gap, qdd_previous=qdd_previous
    )
    problem = build_task_qp(
        jacobian,
        jacobian_dot_qd,
        reference,
        q,
        qd,
        q_mid,
        params,
        dt,
        lower,
        upper,
        psi_gain=psi_gain,
    )
    return RMPResult(solve_box_qp(problem), problem, reference, limited_error, psi_gain)
