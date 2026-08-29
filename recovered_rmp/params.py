"""Recovered RMP parameter layout and Movax YAML loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

if TYPE_CHECKING:
    from .core import JointLimits


@dataclass(frozen=True)
class RMPParams:
    """Parameters used by ``ManipulatorReactiveMotionPlanner``.

    The field names and grouping come from the DWARF type
    ``spirit_ai::base::RMPParams``.  Defaults are the RightArm values in the
    copied ``spirit01_base_e_no_neck_hh.yaml`` configuration, converted to SI.
    """

    kp: float = 324.0
    kd: float = 36.0
    jnt_kp: float = 400.0
    jnt_kd: float = 40.0
    psi_kp: float = 25.0
    psi_kd: float = 10.0
    angle_gap: tuple[float, ...] = (np.deg2rad(2.0),) * 7
    vel_weight: float = 30.0
    acc_weight: float = 0.001
    pos_limit_weight: float = 0.002
    max_linear_vel: float = 1.5
    max_angular_vel: float = np.deg2rad(400.0)
    max_linear_acc: float = 200.0
    max_angular_acc: float = np.deg2rad(4500.0)
    max_psi_acc: float = np.deg2rad(30.0)
    joint_vel_ffd: float = 0.5
    joint_acc_ffd: float = 0.0
    linear_vel_ffd: float = 0.6
    angular_vel_ffd: float = 0.5
    linear_acc_ffd: float = 0.0
    angular_acc_ffd: float = 0.0
    filter_length: int = 35
    target_filter_length: int = 10
    arm_weight_gain: float = 2.0
    estimator_linear_process_var: float = 2.0
    estimator_linear_measure_var: float = 0.0002
    estimator_angular_process_var: float = 2.0
    estimator_angular_measure_var: float = 1.0e-5
    estimator_joint_process_var: float = 50.0
    estimator_joint_measure_var: float = 0.0001
    close_estimator: bool = False
    use_arm_angle: bool = False
    dynamic_pos_regulation: bool = False
    check_limit_eps: float = np.deg2rad(10.0)
    damp_gain: float = 50.0
    stop_all_joints: bool = False
    enable_continuous_check: bool = False
    continuous_check_coeff: float = 2.0


def _pair(values: Any, name: str) -> tuple[float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    return float(values[0]), float(values[1])


def _params_from_mapping(data: Mapping[str, Any], angles_in_degrees: bool) -> RMPParams:
    angle_scale = np.pi / 180.0 if angles_in_degrees else 1.0
    linear_limits = _pair(data.get("linear_limits", (1.5, 200.0)), "linear_limits")
    angular_limits = _pair(data.get("angular_limits", (400.0, 4500.0)), "angular_limits")
    joint_ffd = _pair(data.get("jnt_ffd_coeff", (0.5, 0.0)), "jnt_ffd_coeff")
    linear_ffd = _pair(data.get("linear_ffd_coeff", (0.6, 0.0)), "linear_ffd_coeff")
    angular_ffd = _pair(data.get("angular_ffd_coeff", (0.5, 0.0)), "angular_ffd_coeff")
    linear_var = _pair(data.get("estimator_linear_variance", (2.0, 0.0002)), "estimator_linear_variance")
    angular_var = _pair(data.get("estimator_angular_variance", (2.0, 1.0e-5)), "estimator_angular_variance")
    joint_var = _pair(data.get("estimate_joint_variance", (50.0, 0.0001)), "estimate_joint_variance")

    angle_gap = tuple(float(v) * angle_scale for v in data.get("angle_gap", (2.0,) * 7))
    return RMPParams(
        kp=float(data.get("kp", 324.0)),
        kd=float(data.get("kd", 36.0)),
        jnt_kp=float(data.get("jnt_kp", 400.0)),
        jnt_kd=float(data.get("jnt_kd", 40.0)),
        psi_kp=float(data.get("psi_kp", 25.0)),
        psi_kd=float(data.get("psi_kd", 10.0)),
        angle_gap=angle_gap,
        vel_weight=float(data.get("vel_weight", 30.0)),
        acc_weight=float(data.get("acc_weight", 0.001)),
        pos_limit_weight=float(data.get("pos_limit_weight", 0.002)),
        max_linear_vel=linear_limits[0],
        max_angular_vel=angular_limits[0] * angle_scale,
        max_linear_acc=linear_limits[1],
        max_angular_acc=angular_limits[1] * angle_scale,
        max_psi_acc=float(data.get("max_psi_acc", 30.0)) * angle_scale,
        joint_vel_ffd=joint_ffd[0],
        joint_acc_ffd=joint_ffd[1],
        linear_vel_ffd=linear_ffd[0],
        angular_vel_ffd=angular_ffd[0],
        linear_acc_ffd=linear_ffd[1],
        angular_acc_ffd=angular_ffd[1],
        filter_length=int(data.get("filter_length", 35)),
        target_filter_length=int(data.get("target_filter_length", 10)),
        arm_weight_gain=float(data.get("arm_weight_gain", 2.0)),
        estimator_linear_process_var=linear_var[0],
        estimator_linear_measure_var=linear_var[1],
        estimator_angular_process_var=angular_var[0],
        estimator_angular_measure_var=angular_var[1],
        estimator_joint_process_var=joint_var[0],
        estimator_joint_measure_var=joint_var[1],
        close_estimator=bool(data.get("close_estimator", False)),
        use_arm_angle=bool(data.get("use_arm_angle", False)),
        dynamic_pos_regulation=bool(data.get("dynamic_pos_regulation", False)),
        check_limit_eps=float(data.get("check_limit_eps", 10.0)) * angle_scale,
        damp_gain=float(data.get("damp_gain", 50.0)),
        stop_all_joints=bool(data.get("stop_all_joints", False)),
        enable_continuous_check=bool(data.get("enable_continuous_check", False)),
        continuous_check_coeff=float(data.get("continuous_check_coeff", 2.0)),
    )


def load_movax_rmp_config(
    path: str | Path,
    mech_unit: str = "RightArm",
    *,
    angles_in_degrees: bool = True,
) -> tuple[RMPParams, "JointLimits"]:
    """Load one mechanical unit from a copied Movax model YAML.

    Returns both RMP gains and joint position/velocity/acceleration/jerk limits.
    Values in the shipped YAML are degrees-based, so conversion is enabled by
    default.  Importing :class:`JointLimits` locally avoids a module cycle.
    """

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - depends on host environment
        raise ImportError("PyYAML is required to load the Movax configuration") from exc

    from .core import JointLimits

    with Path(path).open("r", encoding="utf-8") as stream:
        root = yaml.safe_load(stream)
    try:
        unit = root["mech_units"][mech_unit]
        rmp_data = unit["rmp_params"]
        joint_confs = unit["joint_confs"]
    except (KeyError, TypeError) as exc:
        raise KeyError(f"missing mech_units.{mech_unit} configuration") from exc

    def joint_key(name: str) -> tuple[int, str]:
        suffix = name[1:] if name.startswith("J") else name
        return (int(suffix), name)

    ordered = [joint_confs[key] for _, key in sorted(joint_key(k) for k in joint_confs)]
    scale = np.pi / 180.0 if angles_in_degrees else 1.0

    def values(field: str, default: float) -> np.ndarray:
        return np.asarray(
            [float(conf.get("motion_limits", {}).get(field, default)) * scale for conf in ordered],
            dtype=float,
        )

    limits = JointLimits(
        q_min=values("range_min", -np.inf),
        q_max=values("range_max", np.inf),
        qd_max=values("max_vel", np.inf),
        qdd_max=values("max_acc", np.inf),
        qddd_max=values("max_jerk", np.inf),
    )
    return _params_from_mapping(rmp_data, angles_in_degrees), limits
