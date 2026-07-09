#!/usr/bin/env python3
"""Dependency-free forward kinematics for the moz1 right arm.

The replay node can publish a Cartesian command (``MechUnitCmd.end_pose`` with
``use_jnt=False``) instead of a joint command. To build that pose it needs the
end-effector pose for a given 7-DOF right-arm joint vector. The trained policy
used MuJoCo's ``right_ee_site`` for the same purpose (see
``pingpong_controller/rl_policy.py``), but ``mujoco`` is not guaranteed to be
installed on the replay host, so this module reconstructs the same kinematics in
pure NumPy by parsing the model's MJCF (``moz1_pd.xml``).

Frame note
----------
FK is rooted at the *arm-mount* body (the parent of the first right-arm joint,
``waist03``). During replay every non-arm joint is held fixed, so the transform
from this root to whatever base frame the real controller reports its
``end_pose`` in is a *constant* rigid offset. The replay node measures that
offset once at startup against ``/cart_states`` (see ``replay_node.py``), so the
absolute frame/TCP definition used here does not need to match the controller --
only the relative motion does.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Right-arm joint names in MechUnitCmd / CSV column order. Mirrors
# pingpong_controller.rl_policy.RIGHT_ARM_JOINTS.
RIGHT_ARM_JOINTS: List[str] = [
    "RightArm-0", "RightArm-1", "RightArm-2",
    "RightArm-3", "RightArm-4", "RightArm-5", "RightArm-6",
]


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """Convert an MJCF quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation about ``axis`` (need not be unit) by ``angle`` rad."""
    a = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(a)
    if norm < 1e-12:
        return np.eye(3)
    a = a / norm
    K = np.array([[0.0, -a[2], a[1]],
                  [a[2], 0.0, -a[0]],
                  [-a[1], a[0], 0.0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def _make_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def _parse_floats(text: Optional[str], default: List[float]) -> np.ndarray:
    if text is None or not text.strip():
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in text.split()], dtype=np.float64)


class _Body:
    """One MJCF body: local transform relative to its parent + its joints."""

    __slots__ = ("name", "parent", "local_T", "joints", "sites")

    def __init__(self, name: str, parent: Optional[str],
                 local_T: np.ndarray) -> None:
        self.name = name
        self.parent = parent
        self.local_T = local_T
        # Each joint: dict(name, axis(3,), pos(3,), type). Composed in order.
        self.joints: List[dict] = []
        # site name -> local position (3,) in this body's (post-joint) frame.
        self.sites: Dict[str, np.ndarray] = {}


class RightArmFK:
    """Pure-NumPy forward kinematics for the moz1 right arm.

    Parses ``moz1_pd.xml`` once and exposes the end-effector pose and the
    shoulder/elbow/wrist points for a 7-DOF right-arm joint vector, expressed
    relative to the arm-mount body (parent of the first right-arm joint). Only
    the right-arm joints move; every other joint is treated as fixed at zero,
    which is exact for the relative arm transform regardless of the held
    whole-body posture (the mount transform cancels).
    """

    EE_SITE = "right_ee_site"

    def __init__(self, xml_path: str,
                 joint_names: Optional[List[str]] = None) -> None:
        self.xml_path = str(Path(xml_path).resolve())
        self.joint_names = list(joint_names or RIGHT_ARM_JOINTS)
        self._bodies: Dict[str, _Body] = {}
        self._joint_to_body: Dict[str, str] = {}
        self._site_to_body: Dict[str, str] = {}
        self._parse()

        # Resolve the kinematic landmarks used for FK and the arm angle.
        self._ee_body = self._site_to_body.get(self.EE_SITE)
        if self._ee_body is None:
            raise ValueError(
                f"site '{self.EE_SITE}' not found in {self.xml_path}")
        missing = [j for j in self.joint_names if j not in self._joint_to_body]
        if missing:
            raise ValueError(
                f"right-arm joints missing from model: {missing}")
        # Arm-mount root = parent of the body holding the first arm joint.
        first_body = self._joint_to_body[self.joint_names[0]]
        self._root = self._bodies[first_body].parent or first_body
        # Landmark bodies for the swivel (arm) angle: shoulder, elbow, wrist.
        self._shoulder_body = first_body
        self._elbow_body = self._joint_to_body[self.joint_names[3]]
        self._wrist_body = self._joint_to_body[self.joint_names[6]]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------
    def _parse(self) -> None:
        root = ET.parse(self.xml_path).getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError(f"no <worldbody> in {self.xml_path}")
        for body in worldbody.findall("body"):
            self._parse_body(body, parent=None)

    def _parse_body(self, elem: ET.Element, parent: Optional[str]) -> None:
        name = elem.get("name") or f"_anon_{len(self._bodies)}"
        pos = _parse_floats(elem.get("pos"), [0.0, 0.0, 0.0])
        quat = _parse_floats(elem.get("quat"), [1.0, 0.0, 0.0, 0.0])
        local_T = _make_T(_quat_wxyz_to_matrix(quat), pos)
        body = _Body(name, parent, local_T)

        for joint in elem.findall("joint"):
            jname = joint.get("name") or ""
            jtype = joint.get("type", "hinge")
            jpos = _parse_floats(joint.get("pos"), [0.0, 0.0, 0.0])
            jaxis = _parse_floats(joint.get("axis"), [0.0, 0.0, 1.0])
            body.joints.append(
                {"name": jname, "type": jtype, "axis": jaxis, "pos": jpos})
            if jname:
                self._joint_to_body[jname] = name

        for site in elem.findall("site"):
            sname = site.get("name")
            if sname:
                spos = _parse_floats(site.get("pos"), [0.0, 0.0, 0.0])
                body.sites[sname] = spos
                self._site_to_body[sname] = name

        self._bodies[name] = body
        for child in elem.findall("body"):
            self._parse_body(child, parent=name)

    # ------------------------------------------------------------------
    # Forward kinematics
    # ------------------------------------------------------------------
    def _path_root_to(self, target: str) -> List[str]:
        """List of body names from self._root (exclusive) down to target."""
        chain: List[str] = []
        cur: Optional[str] = target
        while cur is not None and cur != self._root:
            chain.append(cur)
            cur = self._bodies[cur].parent
        chain.reverse()
        return chain

    def _body_transform(self, body_name: str,
                        q_map: Dict[str, float]) -> np.ndarray:
        """Transform of ``body_name`` relative to the arm-mount root.

        Walks root -> body, composing each body's fixed local transform with the
        rotations of any joints declared on that body (evaluated from ``q_map``;
        absent joints are zero). MJCF applies a body's joints *after* its local
        placement, rotating about the joint ``pos`` in the body frame.
        """
        T = np.eye(4)
        for name in self._path_root_to(body_name):
            body = self._bodies[name]
            T = T @ body.local_T
            for j in body.joints:
                angle = float(q_map.get(j["name"], 0.0))
                if j["type"] == "slide":
                    R = np.eye(3)
                    trans = j["axis"] / max(np.linalg.norm(j["axis"]), 1e-12)
                    Tj = _make_T(R, trans * angle)
                else:  # hinge / default: rotate about axis through joint pos
                    R = _axis_angle_to_matrix(j["axis"], angle)
                    p = j["pos"]
                    Tj = _make_T(R, p - R @ p)
                T = T @ Tj
        return T

    def _q_map(self, q: np.ndarray) -> Dict[str, float]:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape[0] != len(self.joint_names):
            raise ValueError(
                f"expected {len(self.joint_names)} joint values, "
                f"got {q.shape[0]}")
        return {name: float(q[i]) for i, name in enumerate(self.joint_names)}

    def ee_pose(self, q: np.ndarray) -> np.ndarray:
        """Return the 4x4 pose of ``right_ee_site`` for joint vector ``q``.

        Expressed relative to the arm-mount root frame.
        """
        q_map = self._q_map(q)
        T_body = self._body_transform(self._ee_body, q_map)
        site_p = self._bodies[self._ee_body].sites[self.EE_SITE]
        return T_body @ _make_T(np.eye(3), site_p)

    def landmarks(self, q: np.ndarray) -> Dict[str, np.ndarray]:
        """Return shoulder/elbow/wrist origin positions (root frame).

        Used to compute the redundant arm (swivel) angle ``psi``.
        """
        q_map = self._q_map(q)
        return {
            "shoulder": self._body_transform(self._shoulder_body, q_map)[:3, 3],
            "elbow": self._body_transform(self._elbow_body, q_map)[:3, 3],
            "wrist": self._body_transform(self._wrist_body, q_map)[:3, 3],
        }

    def arm_angle(self, q: np.ndarray,
                  ref_dir: Optional[np.ndarray] = None) -> float:
        """Estimate the redundant arm (swivel) angle ``psi`` in radians.

        The swivel angle is the signed rotation of the elbow about the
        shoulder->wrist axis, measured from a reference direction (default:
        world -Z gravity projected into the plane normal to that axis). The
        replay node calibrates the constant offset between this estimate and the
        controller-reported ``psi`` at startup, so only the relative variation
        needs to be consistent.
        """
        lm = self.landmarks(q)
        sw = lm["wrist"] - lm["shoulder"]
        n = np.linalg.norm(sw)
        if n < 1e-9:
            return 0.0
        axis = sw / n
        # Elbow displacement perpendicular to the shoulder-wrist axis.
        se = lm["elbow"] - lm["shoulder"]
        perp = se - np.dot(se, axis) * axis
        pn = np.linalg.norm(perp)
        if pn < 1e-9:
            return 0.0
        perp = perp / pn
        ref = np.asarray(ref_dir if ref_dir is not None else [0.0, 0.0, -1.0],
                         dtype=np.float64)
        ref = ref - np.dot(ref, axis) * axis
        rn = np.linalg.norm(ref)
        if rn < 1e-9:
            # Reference parallel to the arm axis; fall back to a stable basis.
            ref = np.array([1.0, 0.0, 0.0]) - axis[0] * axis
            rn = np.linalg.norm(ref)
            if rn < 1e-9:
                return 0.0
        ref = ref / rn
        cos_a = float(np.clip(np.dot(perp, ref), -1.0, 1.0))
        sin_a = float(np.dot(np.cross(ref, perp), axis))
        return float(np.arctan2(sin_a, cos_a))
