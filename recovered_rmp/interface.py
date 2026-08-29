"""Interface layer for using the recovered RMP planner with a single JSON config.

This module is the "convenient API" for downstream simulation code.  It is
deliberately pure Python/NumPy: it does not import MuJoCo, MJX, or JAX, so it
can be used from any simulator.  The one JSON file owns every tunable that the
RMP path and the low-level XML PD consume, so experiments can be changed
without touching code.

Three usage levels are provided:

* :func:`load_rmp_config` -- read and validate one JSON file into
  :class:`RMPConfig` / :class:`RMPParams` / :class:`JointLimits` / XML PD
  fields.
* :func:`create_planner` / :func:`create_planner_from_dict` -- build a ready
  :class:`~recovered_rmp.trajectory.JointRMPPlanner` from that config.
* :func:`drive_joint_rmp` -- the all-in-one helper: create the planner and run
  a target trajectory, returning a :class:`JointRMPCommand` per servo tick.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .core import JointLimits
from .params import RMPParams
from .trajectory import JointRMPCommand, JointRMPPlanner

FloatArray = NDArray[np.float64]

_N_JOINTS = 7

#: RightArm limits and gains in the copied Movax YAML, in SI units.
#: Kept here so that a bare JSON carrying only scales still resolves to a
#: complete planner without needing the YAML on disk.
_YAML_RIGHT_ARM = {
    "kp": 324.0,
    "kd": 36.0,
    "jnt_kp": 400.0,
    "jnt_kd": 40.0,
    "q_min": [
        -2.094395,
        -2.967060,
        -3.054326,
        -0.174533,
        -3.054326,
        -1.658063,
        -1.570796,
    ],
    "q_max": [
        3.141593,
        0.157080,
        3.054326,
        2.251475,
        3.054326,
        1.658063,
        1.570796,
    ],
    "qd_max": [3.6652, 3.6652, 4.1888, 4.1888, 5.2360, 5.2360, 5.2360],
    "qdd_max": [
        22.6893,
        22.6893,
        31.4159,
        52.3599,
        52.3599,
        52.3599,
        52.3599,
    ],
    "qddd_max": [610.87, 610.87, 785.40, 1047.2, 1047.2, 1047.2, 1047.2],
}


@dataclass
class RMPConfig:
    """Resolved RMP configuration loaded from JSON.

    ``rmp`` is the recovered-YAML :class:`RMPParams` with the JSON scales
    already applied; ``limits`` are the planner's safety-bounds limits;
    ``xml_pd`` holds the low-level XML PD / dof / command-path values that an
    MJX-style environment consumes.  The latter exist for simulator
    integration only -- :func:`create_planner` never reads them.
    """

    rmp: RMPParams
    limits: JointLimits
    dt: float
    use_safety_bounds: bool
    use_jerk_limit: bool
    target_update_delay_steps: int
    output_delay_steps_per_joint: tuple[int, ...]
    xml_pd: dict[str, Any]
    meta: dict[str, Any]
    source: str
    raw: dict[str, Any] = field(repr=False)


def _vector(
    value: ArrayLike | None,
    name: str,
    size: int,
    *,
    default: ArrayLike | None = None,
) -> NDArray[np.float64]:
    if value is None or (isinstance(value, (list, tuple)) and len(value) == 0):
        if default is None:
            raise ValueError(f"{name} is required")
        return np.asarray(default, dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != size:
        raise ValueError(f"{name} must contain {size} values, got {arr.size}")
    return arr


def _scalar(value: Any, name: str, default: Any) -> float | int | bool:
    return default if value is None else value


def _scales(
    raw: Mapping[str, Any],
    key: str,
    size: int,
    default: float,
) -> tuple[float, ...]:
    values = raw.get(key, ())
    if values is None or len(values) == 0:
        return (float(default),) * size
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != size:
        raise ValueError(f"{key} must contain {size} values, got {arr.size}")
    return tuple(float(v) for v in arr)


def _planner_from_raw(
    raw: Mapping[str, Any],
    *,
    source: str,
    apply_per_joint_scales: bool = False,
) -> tuple[RMPParams, JointLimits]:
    """Resolve RMPParams and JointLimits from a bare JSON mapping.

    Per-joint scales are intentionally *not* applied to ``jnt_kp``/``jnt_kd``
    by default: in the original runtime those two values are scalar YAML
    constants shared by all joints.  The per-joint scale fields are kept in the
    JSON for MJX-style environments that apply them downstream; pass
    ``apply_per_joint_scales=True`` to fold them into the planner itself
    (useful when driving an environment that already scales, and the planner
    must agree).
    """

    rmp_data = raw.get("rmp", {})
    planning = raw.get("planning", {})
    limits_data = raw.get("joint_limits", {})
    yaml_defaults = _YAML_RIGHT_ARM

    kp = float(_scalar(rmp_data.get("kp"), "kp", yaml_defaults["kp"]))
    kd = float(_scalar(rmp_data.get("kd"), "kd", yaml_defaults["kd"]))
    jnt_kp = float(
        _scalar(rmp_data.get("jnt_kp"), "jnt_kp", yaml_defaults["jnt_kp"])
    )
    jnt_kd = float(
        _scalar(rmp_data.get("jnt_kd"), "jnt_kd", yaml_defaults["jnt_kd"])
    )
    jnt_kp *= float(_scalar(rmp_data.get("jnt_kp_scale"), "jnt_kp_scale", 1.0))
    jnt_kd *= float(_scalar(rmp_data.get("jnt_kd_scale"), "jnt_kd_scale", 1.0))
    if apply_per_joint_scales:
        jnt_kp = (
            np.asarray(
                _scales(rmp_data, "jnt_kp_scale_per_joint", _N_JOINTS, 1.0)
            )
            * jnt_kp
        )
        jnt_kd = (
            np.asarray(
                _scales(rmp_data, "jnt_kd_scale_per_joint", _N_JOINTS, 1.0)
            )
            * jnt_kd
        )

    angle_gap = _vector(
        rmp_data.get("angle_gap_rad"),
        "rmp.angle_gap_rad",
        _N_JOINTS,
        default=np.deg2rad(2.0) * np.ones(_N_JOINTS),
    )
    target_filter = int(
        _scalar(rmp_data.get("target_filter_length"), "target_filter_length", 10)
    )
    output_filter = int(
        _scalar(rmp_data.get("output_filter_length"), "output_filter_length", 35)
    )
    if target_filter < 1 or output_filter < 1:
        raise ValueError("filter lengths must be positive")
    process_var = float(
        _scalar(
            rmp_data.get("estimator_joint_process_var"),
            "estimator_joint_process_var",
            50.0,
        )
    )
    measure_var = float(
        _scalar(
            rmp_data.get("estimator_joint_measure_var"),
            "estimator_joint_measure_var",
            0.0001,
        )
    )

    params = RMPParams(
        kp=kp,
        kd=kd,
        jnt_kp=float(jnt_kp) if np.isscalar(jnt_kp) else jnt_kp,
        jnt_kd=float(jnt_kd) if np.isscalar(jnt_kd) else jnt_kd,
        angle_gap=tuple(float(v) for v in angle_gap),
        vel_weight=float(
            _scalar(rmp_data.get("vel_weight"), "vel_weight", 30.0)
        ),
        acc_weight=float(
            _scalar(rmp_data.get("acc_weight"), "acc_weight", 0.001)
        ),
        pos_limit_weight=float(
            _scalar(
                rmp_data.get("pos_limit_weight"), "pos_limit_weight", 0.002
            )
        ),
        joint_vel_ffd=float(
            _scalar(
                rmp_data.get("joint_velocity_feedforward"),
                "joint_velocity_feedforward",
                0.5,
            )
        ),
        joint_acc_ffd=float(
            _scalar(
                rmp_data.get("joint_acceleration_feedforward"),
                "joint_acceleration_feedforward",
                0.0,
            )
        ),
        filter_length=output_filter,
        target_filter_length=target_filter,
        estimator_joint_process_var=process_var,
        estimator_joint_measure_var=measure_var,
    )

    q_min = _vector(
        limits_data.get("q_min"), "joint_limits.q_min", _N_JOINTS,
        default=yaml_defaults["q_min"],
    )
    q_max = _vector(
        limits_data.get("q_max"), "joint_limits.q_max", _N_JOINTS,
        default=yaml_defaults["q_max"],
    )
    qd_max = _vector(
        limits_data.get("qd_max"), "joint_limits.qd_max", _N_JOINTS,
        default=yaml_defaults["qd_max"],
    )
    qdd_max = _vector(
        limits_data.get("qdd_max"), "joint_limits.qdd_max", _N_JOINTS,
        default=yaml_defaults["qdd_max"],
    )
    qddd_value = limits_data.get("qddd_max")
    qddd_max = (
        None
        if qddd_value is None
        else _vector(
            qddd_value, "joint_limits.qddd_max", _N_JOINTS,
            default=yaml_defaults["qddd_max"],
        )
    )
    limits = JointLimits(
        q_min=q_min,
        q_max=q_max,
        qd_max=qd_max,
        qdd_max=qdd_max,
        qddd_max=qddd_max,
    )
    return params, limits


def load_rmp_config(
    path: str | Path,
    *,
    apply_per_joint_scales: bool = False,
) -> RMPConfig:
    """Load and validate an RMP JSON configuration file.

    Returns an :class:`RMPConfig` whose ``rmp`` and ``limits`` are ready to
    pass to :func:`create_planner` or directly to
    :class:`~recovered_rmp.trajectory.JointRMPPlanner`.  Angles are already in
    SI units (radians).  The JSON file is the single source of truth: any key
    left out falls back to the recovered RightArm YAML defaults.
    """

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    planning = raw.get("planning", {})
    if not isinstance(planning, dict):
        raise ValueError("'planning' must be an object")
    dt = float(_scalar(planning.get("dt"), "dt", planning.get("period_s", 0.001)))
    if dt <= 0.0:
        raise ValueError("planning.dt must be positive")
    rmp_params, limits = _planner_from_raw(
        raw,
        source=str(config_path),
        apply_per_joint_scales=apply_per_joint_scales,
    )
    use_safety = bool(
        _scalar(planning.get("use_safety_bounds"), "use_safety_bounds", True)
    )
    use_jerk = bool(
        _scalar(planning.get("use_jerk_limit"), "use_jerk_limit", False)
    )
    delay_steps = tuple(
        int(v)
        for v in planning.get("output_delay_steps_per_joint", ()) or ()
    )
    if len(delay_steps) not in (0, _N_JOINTS):
        raise ValueError(
            "planning.output_delay_steps_per_joint must be empty or contain "
            f"{_N_JOINTS} values"
        )
    xml_pd = dict(raw.get("xml_pd", {}) or {})
    meta = dict(raw.get("meta", {}) or {})
    return RMPConfig(
        rmp=rmp_params,
        limits=limits,
        dt=dt,
        use_safety_bounds=use_safety,
        use_jerk_limit=use_jerk,
        target_update_delay_steps=int(
            _scalar(
                planning.get("target_update_delay_steps"),
                "target_update_delay_steps",
                3,
            )
        ),
        output_delay_steps_per_joint=delay_steps,
        xml_pd=xml_pd,
        meta=meta,
        source=str(config_path),
        raw=raw,
    )


def create_planner_from_dict(
    config: Mapping[str, Any],
    *,
    dt: float | None = None,
    apply_per_joint_scales: bool = False,
) -> JointRMPPlanner:
    """Build a :class:`JointRMPPlanner` from a bare JSON-like mapping."""

    params, limits = _planner_from_raw(
        dict(config),
        source="<mapping>",
        apply_per_joint_scales=apply_per_joint_scales,
    )
    planning = config.get("planning", {})
    step_dt = float(
        dt
        if dt is not None
        else _scalar(planning.get("dt"), "dt", planning.get("period_s", 0.001))
    )
    return JointRMPPlanner(
        limits,
        params,
        step_dt,
        use_safety_bounds=bool(
            _scalar(
                planning.get("use_safety_bounds"), "use_safety_bounds", True
            )
        ),
        use_jerk_limit=bool(
            _scalar(planning.get("use_jerk_limit"), "use_jerk_limit", False)
        ),
    )


def create_planner(
    config: RMPConfig | str | Path,
    *,
    dt: float | None = None,
    apply_per_joint_scales: bool = False,
) -> JointRMPPlanner:
    """Build a ready :class:`JointRMPPlanner` from a JSON file or RMPConfig.

    If ``config`` is a path and ``apply_per_joint_scales=True``, the JSON
    per-joint jnt_kp/jnt_kd scales are folded into the planner gains (see
    :func:`load_rmp_config` for why that is off by default).
    """

    resolved = (
        config
        if isinstance(config, RMPConfig)
        else load_rmp_config(
            config, apply_per_joint_scales=apply_per_joint_scales
        )
    )
    return JointRMPPlanner(
        resolved.limits,
        resolved.rmp,
        float(dt) if dt is not None else resolved.dt,
        use_safety_bounds=resolved.use_safety_bounds,
        use_jerk_limit=resolved.use_jerk_limit,
    )


def _validate_targets(
    targets: NDArray[np.float64],
    size: int,
) -> NDArray[np.float64]:
    arr = np.asarray(targets, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != size:
        raise ValueError(
            f"targets must have {size} columns (one per joint), "
            f"got {arr.shape[1]}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("targets contain non-finite values")
    return arr


def drive_joint_rmp(
    targets: ArrayLike,
    config: RMPConfig | str | Path,
    *,
    initial_position: ArrayLike | None = None,
    initial_velocity: ArrayLike | None = None,
    initial_acceleration: ArrayLike | None = None,
    control_period: float | None = None,
    dt: float | None = None,
    apply_per_joint_scales: bool = False,
) -> list[JointRMPCommand]:
    """Run a position-target trajectory through the planner.

    Each row of ``targets`` is a 7-vector position target (rad).  The planner
    advances at the config's 1 kHz ``dt``; ``control_period`` (default from
    the config's ``planning.control_period_s``) defines how many servo ticks a
    target is held, so a 200 Hz target stream becomes 5 ticks per target.
    ``target_updated=True`` is set only on the first tick of each held
    command, exactly like the runtime.  Returns one
    :class:`JointRMPCommand` per servo tick.
    """

    resolved = (
        config
        if isinstance(config, RMPConfig)
        else load_rmp_config(
            config, apply_per_joint_scales=apply_per_joint_scales
        )
    )
    planner = create_planner(resolved, dt=dt)
    arr = _validate_targets(targets, planner.size)
    hold_ticks = 1
    if control_period is None and resolved.raw.get("planning", {}).get(
        "control_period_s"
    ):
        control_period = float(
            resolved.raw["planning"]["control_period_s"]
        )
    if control_period is not None and control_period > planner.dt:
        hold_ticks = max(1, int(round(control_period / planner.dt)))

    commands: list[JointRMPCommand] = []
    for index in range(arr.shape[0]):
        target = arr[index]
        if index == 0:
            planner.reset(
                initial_position if initial_position is not None else target,
                velocity=initial_velocity,
                acceleration=initial_acceleration,
                target_position=target,
            )
        for tick in range(hold_ticks):
            commands.append(
                planner.step(target, target_updated=(tick == 0))
            )
    return commands


def commands_to_arrays(
    commands: Sequence[JointRMPCommand],
) -> dict[str, FloatArray]:
    """Stack a command list into plain arrays (n_ticks x 7)."""

    return {
        "position": np.stack([c.position for c in commands]),
        "velocity": np.stack([c.velocity for c in commands]),
        "acceleration": np.stack([c.acceleration for c in commands]),
        "target_position": np.stack([c.target_position for c in commands]),
        "target_velocity": np.stack([c.target_velocity for c in commands]),
        "target_acceleration": np.stack([c.target_acceleration for c in commands]),
    }


__all__ = [
    "RMPConfig",
    "commands_to_arrays",
    "create_planner",
    "create_planner_from_dict",
    "drive_joint_rmp",
    "load_rmp_config",
]
