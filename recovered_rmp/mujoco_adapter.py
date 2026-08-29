"""Thin MuJoCo adapter for the recovered RMP core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SiteKinematics:
    position: np.ndarray
    rotation: np.ndarray
    twist: np.ndarray
    jacobian: np.ndarray
    jacobian_dot_qd: np.ndarray


def single_dof_joint_indices(model: object, joint_names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Resolve hinge/slide joint names to qpos and qvel/dof indices."""

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("MuJoCo is required for this adapter") from exc
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names]
    if any(joint_id < 0 for joint_id in joint_ids):
        missing = [name for name, joint_id in zip(joint_names, joint_ids) if joint_id < 0]
        raise KeyError(f"MuJoCo joints not found: {missing}")
    hinge_or_slide = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
    wrong_type = [name for name, joint_id in zip(joint_names, joint_ids) if int(model.jnt_type[joint_id]) not in hinge_or_slide]
    if wrong_type:
        raise ValueError(f"only one-DoF hinge/slide joints are supported: {wrong_type}")
    qpos = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=int)
    dofs = np.asarray([model.jnt_dofadr[joint_id] for joint_id in joint_ids], dtype=int)
    return qpos, dofs


def site_kinematics(model: object, data: object, site: str | int, controlled_dofs: Sequence[int]) -> SiteKinematics:
    """Read the spatial quantities expected by ``solve_task_rmp``.

    Returned task vectors use ``[linear, angular]`` order in the world frame.
    ``Jdot*qd`` includes every moving DoF in the MuJoCo model, while the returned
    Jacobian contains only the columns controlled by this RMP instance.
    """

    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("MuJoCo is required for this adapter") from exc
    site_id = (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if isinstance(site, str)
        else int(site)
    )
    if site_id < 0 or site_id >= model.nsite:
        raise KeyError(f"MuJoCo site not found: {site}")
    dofs = np.asarray(controlled_dofs, dtype=int)
    if np.any(dofs < 0) or np.any(dofs >= model.nv):
        raise IndexError("controlled_dofs contains an invalid qvel index")

    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    jacp_dot = np.zeros((3, model.nv), dtype=float)
    jacr_dot = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    body_id = int(model.site_bodyid[site_id])
    point = np.asarray(data.site_xpos[site_id], dtype=float)
    mujoco.mj_jacDot(model, data, jacp_dot, jacr_dot, point, body_id)

    full_jacobian = np.vstack((jacp, jacr))
    full_jacobian_dot = np.vstack((jacp_dot, jacr_dot))
    qvel = np.asarray(data.qvel, dtype=float)
    return SiteKinematics(
        position=np.asarray(data.site_xpos[site_id], dtype=float).copy(),
        rotation=np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3).copy(),
        twist=full_jacobian @ qvel,
        jacobian=full_jacobian[:, dofs],
        jacobian_dot_qd=full_jacobian_dot @ qvel,
    )


def normalized_acceleration_action(qdd: np.ndarray, qdd_limit: np.ndarray, action_acc_scale: float = 1.5) -> np.ndarray:
    """Map rad/s^2 to the action convention in ``MjxJuggleEnv``."""

    qdd = np.asarray(qdd, dtype=float)
    limit = np.asarray(qdd_limit, dtype=float)
    if np.any(limit <= 0.0) or action_acc_scale <= 0.0:
        raise ValueError("acceleration limits and action_acc_scale must be positive")
    return np.clip(qdd / (limit * action_acc_scale), -1.0, 1.0)
