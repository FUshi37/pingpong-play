"""One recovered-RMP control step for a MuJoCo right arm.

Copy ``right_arm_rmp_step`` into a CPU simulation loop, or use the same matrix
formula from ``jax_backend.py`` in a batched MJX policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from recovered_rmp import frame_error, load_movax_rmp_config, solve_task_rmp
from recovered_rmp.mujoco_adapter import (
    normalized_acceleration_action,
    single_dof_joint_indices,
    site_kinematics,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = REPOSITORY_ROOT / "movax/controller/configuration/model/user/spirit01_base_e_no_neck_hh.yaml"
RIGHT_ARM_JOINTS = tuple(f"RightArm-{index}" for index in range(7))


def right_arm_rmp_step(
    model: object,
    data: object,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    dt: float,
    right_arm_joint_names: Sequence[str] = RIGHT_ARM_JOINTS,
    racket_site: str = "right_ee_site",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(qdd_rad_s2, normalized_mjx_action)`` for one control tick."""

    params, limits = load_movax_rmp_config(MODEL_CONFIG, "RightArm")
    qpos_indices, dof_indices = single_dof_joint_indices(model, right_arm_joint_names)
    kin = site_kinematics(model, data, racket_site, dof_indices)
    q = np.asarray(data.qpos[qpos_indices], dtype=float)
    qd = np.asarray(data.qvel[dof_indices], dtype=float)
    q_mid = 0.5 * (limits.q_min + limits.q_max)
    error = frame_error(kin.position, kin.rotation, target_position, target_rotation)

    result = solve_task_rmp(
        q=q,
        qd=qd,
        q_mid=q_mid,
        limits=limits,
        jacobian=kin.jacobian,
        jacobian_dot_qd=kin.jacobian_dot_qd,
        error=error,
        current_twist=kin.twist,
        target_twist=np.zeros(6),
        target_acceleration=np.zeros(6),
        params=params,
        dt=dt,
    )
    action = normalized_acceleration_action(result.qdd, limits.qdd_max, action_acc_scale=1.5)
    return result.qdd, action
