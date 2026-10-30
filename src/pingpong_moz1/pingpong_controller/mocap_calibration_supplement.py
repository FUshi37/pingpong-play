"""Bounded intermediate joint postures for automatic mocap calibration.

Only new supplemental points use this position envelope. FK remains in
waist03; the workspace limits shared with the V172 probe are in sim_frame.
This module does not open ROS, a UDP socket, or a robot command publisher.
"""
from __future__ import annotations

import numpy as np

from pingpong_controller.fplc_deployment import FPLC_REAL_LOWER_M, FPLC_REAL_UPPER_M
from pingpong_controller.mocap_input import transform_waist03_to_sim


def rotation_distance(first, second):
    cosine = (np.trace(first.T @ second) - 1.) / 2.
    return float(np.arccos(np.clip(cosine, -1., 1.)))


class SupplementalPlanner:
    def __init__(self, fk, joint_limits, args):
        self.fk = fk
        self.low, self.high = joint_limits
        self.lower = np.asarray(FPLC_REAL_LOWER_M, dtype=float)
        self.upper = np.asarray(FPLC_REAL_UPPER_M, dtype=float)
        self.offset = transform_waist03_to_sim(np.zeros(3)).astype(float)
        self.margin = args.supplemental_margin_mm / 1000.
        if np.any(self.lower + self.margin >= self.upper - self.margin):
            raise ValueError('Supplemental workspace margin leaves no usable V172 workspace')
        self.jitter = np.deg2rad(args.supplemental_joint_jitter_deg)
        self.rotation_limit = np.deg2rad(args.supplemental_rotation_deg)
        self.reference_rotation = None
        self.orientation_range = np.deg2rad(4.)
        self.candidates = []

    def set_reference_rotation(self, rotation):
        rotation = np.asarray(rotation, dtype=float)
        if (rotation.shape != (3, 3) or not np.all(np.isfinite(rotation))
                or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0)
                or not np.isclose(np.linalg.det(rotation), 1., atol=1e-6, rtol=0)):
            raise ValueError('Invalid supplemental reference orientation')
        self.reference_rotation = rotation.copy()

    def pose(self, q):
        pose = np.asarray(self.fk.ee_pose(q), dtype=float).copy()
        pose[:3, 3] += self.offset
        return pose

    def contains(self, q, *, margin=0.):
        q = np.asarray(q, dtype=float)
        if (q.shape != (7,) or not np.all(np.isfinite(q))
                or np.any(q < self.low) or np.any(q > self.high)):
            return False
        pose = self.pose(q)
        attitude_limit = self.orientation_range - (np.deg2rad(.5) if margin > 0. else 0.)
        return bool(np.all(np.isfinite(pose))
                    and np.all(pose[:3, 3] >= self.lower + margin)
                    and np.all(pose[:3, 3] <= self.upper - margin)
                    and (self.reference_rotation is None
                         or rotation_distance(self.reference_rotation, pose[:3, :3]) <= attitude_limit))

    def path_is_safe(self, start, target):
        # SlowResetTrajectory follows this joint-space line with quintic time
        # scaling. Check FK along it, rather than assuming Cartesian convexity
        # implies safety for a joint interpolation. The live timer also checks
        # every command reference and current measured posture.
        start, target = np.asarray(start, dtype=float), np.asarray(target, dtype=float)
        steps = max(2, int(np.ceil(np.max(np.abs(target - start)) / np.deg2rad(.25))))
        return all(self.contains(start + alpha * (target - start))
                   for alpha in np.linspace(0., 1., steps + 1))

    def prepare(self, waypoints, budget):
        """Build a deterministic, finite candidate bank before starting ROS."""
        self.candidates.clear()
        if budget == 0:
            return
        seen = set()
        limit = max(128, budget * 8)
        for gap in range(1, len(waypoints)):
            for first in range(len(waypoints) - gap):
                second = first + gap
                for alpha in (.5, .25, .75):
                    midpoint = (1. - alpha) * waypoints[first] + alpha * waypoints[second]
                    if not self.contains(midpoint, margin=self.margin):
                        continue
                    base_pose = self.pose(midpoint)
                    perturbations = [np.zeros(7)]
                    for joint in range(7):
                        for sign in (-1., 1.):
                            delta = np.zeros(7)
                            delta[joint] = sign * self.jitter
                            perturbations.append(delta)
                    for delta in perturbations:
                        q = midpoint + delta
                        signature = tuple(np.round(q, 8))
                        if signature in seen or not self.contains(q, margin=self.margin):
                            continue
                        pose = self.pose(q)
                        if (rotation_distance(base_pose[:3, :3], pose[:3, :3]) > self.rotation_limit
                                or np.linalg.norm(pose[:3, 3] - base_pose[:3, 3]) > .01):
                            continue
                        seen.add(signature)
                        self.candidates.append(dict(
                            generation='recorded_pose_interpolation',
                            joint_positions_rad=q.tolist(),
                            source_point_indices=[first + 1, second + 1],
                            interpolation_fraction=alpha,
                            joint_perturbation_deg=np.rad2deg(delta).tolist(),
                            perturbation_rotation_deg=float(np.rad2deg(
                                rotation_distance(base_pose[:3, :3], pose[:3, :3]))),
                            position_sim_m=pose[:3, 3].tolist(),
                        ))
                        if len(self.candidates) >= limit:
                            return

    def next_candidate(self, start, previous_targets, measured_samples):
        """Choose a nearby, distinct point with an entirely in-box joint path."""
        used = [np.asarray(row['joint_positions_rad'], dtype=float)
                for row in previous_targets]
        poses = [self.pose(sample.joint_positions_rad) for sample in measured_samples]
        ordered = sorted(self.candidates, key=lambda row: float(np.linalg.norm(
            np.asarray(row['joint_positions_rad']) - start)))

        def usable(row):
            q = np.asarray(row['joint_positions_rad'], dtype=float)
            if not self.contains(q, margin=self.margin):
                return False
            if any(np.max(np.abs(q - old)) < 1e-7 for old in used):
                return False
            pose = self.pose(q)
            # Add a new pose pair, not an effectively repeated measurement.
            if any(np.linalg.norm(pose[:3, 3] - old[:3, 3]) < .002
                   and rotation_distance(pose[:3, :3], old[:3, :3]) < np.deg2rad(.5)
                   for old in poses):
                return False
            return self.path_is_safe(start, q)

        for row in ordered:
            if usable(row):
                return dict(row)
        # If no saved-pose midpoint is reachable with a small attitude change,
        # refine the current in-box posture locally. Single/two-joint stencils
        # remain bounded in translation and rotation and give a failed point
        # several distinct alternatives without walking outside the box.
        base = self.pose(start)
        perturbations = []
        for first in range(7):
            for sign in (-1., 1.):
                delta = np.zeros(7)
                delta[first] = sign * self.jitter
                perturbations.append(delta)
                for second in range(first + 1, 7):
                    for other_sign in (-1., 1.):
                        combined = delta.copy()
                        combined[second] = other_sign * self.jitter
                        perturbations.append(combined)
        for delta in perturbations:
            q = start + delta
            if not self.contains(q, margin=self.margin):
                continue
            pose = self.pose(q)
            angle = rotation_distance(base[:3, :3], pose[:3, :3])
            if angle > self.rotation_limit or np.linalg.norm(pose[:3, 3] - base[:3, 3]) > .01:
                continue
            row = dict(generation='local_refinement', source_point_indices=[],
                       seed_joint_positions_rad=start.tolist(),
                       joint_positions_rad=q.tolist(),
                       joint_perturbation_deg=np.rad2deg(delta).tolist(),
                       perturbation_rotation_deg=float(np.rad2deg(angle)),
                       position_sim_m=pose[:3, 3].tolist())
            if usable(row):
                return row
        return None

    def metadata(self):
        return dict(frame='sim_frame', site='right_ee_site',
                    lower_m=self.lower.tolist(), upper_m=self.upper.tolist(),
                    waist03_to_sim_offset_m=self.offset.tolist(),
                    target_boundary_margin_m=self.margin,
                    joint_jitter_deg=float(np.rad2deg(self.jitter)),
                    maximum_perturbation_rotation_deg=float(np.rad2deg(self.rotation_limit)),
                    reference_rotation_sim=(None if self.reference_rotation is None
                                            else self.reference_rotation.tolist()),
                    maximum_orientation_delta_from_reference_deg=float(np.rad2deg(self.orientation_range)),
                    target_orientation_margin_deg=.5,
                    candidate_count=len(self.candidates))
