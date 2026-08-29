"""Simulation-oriented recovery of Movax's reactive motion planner.

The package exposes the acceleration-level QP recovered from the shipped
``mc_core`` binary.  It is deliberately independent of ROS and the private
``rkdl`` kinematics library: callers provide joint states, task errors and
Jacobians from their simulator.
"""

from .core import (
    JointLimits,
    JointBoundsInfeasible,
    QPProblem,
    RMPResult,
    build_joint_qp,
    build_task_qp,
    compute_joint_reference,
    compute_task_reference,
    frame_error,
    joint_acceleration_bounds,
    solve_box_qp,
    solve_joint_rmp,
    solve_task_rmp,
    wrap_to_pi,
)
from .estimator import ConstantAccelerationKalman, TargetDifferentiator
from .interface import (
    RMPConfig,
    commands_to_arrays,
    create_planner,
    create_planner_from_dict,
    drive_joint_rmp,
    load_rmp_config,
)
from .params import RMPParams, load_movax_rmp_config
from .trajectory import JointRMPCommand, JointRMPPlanner
from .whole_body import WeightedTask, combine_qps, solve_weighted_tasks

__all__ = [
    "ConstantAccelerationKalman",
    "JointLimits",
    "JointRMPCommand",
    "JointRMPPlanner",
    "JointBoundsInfeasible",
    "QPProblem",
    "RMPConfig",
    "RMPParams",
    "RMPResult",
    "TargetDifferentiator",
    "WeightedTask",
    "build_joint_qp",
    "build_task_qp",
    "combine_qps",
    "commands_to_arrays",
    "compute_joint_reference",
    "compute_task_reference",
    "create_planner",
    "create_planner_from_dict",
    "drive_joint_rmp",
    "frame_error",
    "joint_acceleration_bounds",
    "load_movax_rmp_config",
    "load_rmp_config",
    "solve_box_qp",
    "solve_joint_rmp",
    "solve_task_rmp",
    "solve_weighted_tasks",
    "wrap_to_pi",
]
