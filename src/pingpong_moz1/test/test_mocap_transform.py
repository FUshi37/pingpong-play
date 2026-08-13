#!/usr/bin/env python3

import json
import sys
import unittest
from pathlib import Path

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.mocap_ball_detector import parse_mocap_objects, parse_mocap_packet  # noqa: E402
from pingpong_controller.mocap_input import transform_waist03_to_sim  # noqa: E402
from pingpong_controller.mocap_transform import (  # noqa: E402
    apply_tool_offset,
    average_quaternions_wxyz,
    fit_eye_to_hand_transform,
    fit_rigid_transform,
    pose_msg_to_matrix,
    pose_wxyz_to_matrix,
    quaternion_spread_deg,
    transform_point,
)


class MocapTransformTest(unittest.TestCase):

    def test_fit_rigid_transform_recovers_known_transform(self):
        R = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        t = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        src = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        dst = (R @ src.T).T + t

        T, rmse = fit_rigid_transform(src, dst)
        np.testing.assert_allclose(T[:3, :3], R, atol=1e-10)
        np.testing.assert_allclose(T[:3, 3], t, atol=1e-10)
        self.assertLess(rmse, 1e-9)

    def test_apply_tool_offset(self):
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = [1.0, 2.0, 3.0]
        pos = apply_tool_offset(T, [0.0, 0.0, 0.007])
        np.testing.assert_allclose(pos, [1.0, 2.0, 3.007], atol=1e-12)

    def test_pose_msg_to_matrix_and_transform_point(self):
        T = pose_msg_to_matrix([1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
        out = transform_point(T, [0.5, 0.0, 0.0])
        np.testing.assert_allclose(out, [1.5, 0.0, 0.0], atol=1e-12)

    def test_pose_wxyz_to_matrix(self):
        # 180 degrees around z: (w, x, y, z) = (0, 0, 0, 1).
        T = pose_wxyz_to_matrix([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(
            T,
            [[-1.0, 0.0, 0.0, 1.0],
             [0.0, -1.0, 0.0, 2.0],
             [0.0, 0.0, 1.0, 3.0],
             [0.0, 0.0, 0.0, 1.0]],
            atol=1e-12)

    def test_quaternion_mean_is_invariant_to_sign(self):
        angle = np.deg2rad(10.0) * 0.5
        q = np.array([np.cos(angle), 0.0, np.sin(angle), 0.0])
        mean_q = average_quaternions_wxyz([q, -q, q])
        self.assertAlmostEqual(abs(float(np.dot(mean_q, q))), 1.0, places=12)
        rms_deg, max_deg = quaternion_spread_deg([q, -q, q], mean_q)
        self.assertLess(rms_deg, 1e-6)
        self.assertLess(max_deg, 1e-6)

    def test_full_pose_handeye_recovers_offset_rigid_body_mount(self):
        try:
            import cv2
        except ImportError:
            self.skipTest('OpenCV is required for calibrateHandEye')

        def make_pose(rvec, translation):
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = cv2.Rodrigues(
                np.asarray(rvec, dtype=np.float64))[0]
            T[:3, 3] = np.asarray(translation, dtype=np.float64)
            return T

        # Fixed unknown transforms to recover.  K is deliberately far from
        # the EE origin, so a position-only Kabsch fit would be invalid.
        T_waist03_mocap_truth = make_pose(
            [0.31, -0.17, 0.42], [0.52, -0.34, 0.91])
        T_ee_rigid_truth = make_pose(
            [-0.24, 0.37, -0.19], [0.12, -0.08, 0.19])
        rng = np.random.default_rng(4)
        robot_poses = []
        mocap_poses = []
        for _ in range(15):
            T_waist03_ee = make_pose(
                rng.normal(size=3), rng.uniform(-0.6, 0.6, size=3))
            T_mocap_rigid = (
                np.linalg.inv(T_waist03_mocap_truth)
                @ T_waist03_ee @ T_ee_rigid_truth)
            robot_poses.append(T_waist03_ee)
            mocap_poses.append(T_mocap_rigid)

        T_waist03_mocap, T_ee_rigid, metrics = fit_eye_to_hand_transform(
            robot_poses, mocap_poses, method='PARK')
        np.testing.assert_allclose(
            T_waist03_mocap, T_waist03_mocap_truth, atol=1e-8)
        np.testing.assert_allclose(T_ee_rigid, T_ee_rigid_truth, atol=1e-8)
        self.assertLess(metrics['rmse_m'], 1e-9)
        self.assertLess(metrics['rotation_max_deg'], 1e-4)

    def test_handeye_translations_minimize_absolute_pose_closure(self):
        try:
            import cv2
        except ImportError:
            self.skipTest('OpenCV is required for calibrateHandEye')

        def make_pose(rvec, translation):
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = cv2.Rodrigues(
                np.asarray(rvec, dtype=np.float64))[0]
            T[:3, 3] = np.asarray(translation, dtype=np.float64)
            return T

        T_w_m_truth = make_pose([0.2, -0.3, 0.1], [0.4, -0.2, 0.8])
        T_e_k_truth = make_pose([-0.1, 0.2, 0.15], [0.02, -0.01, 0.04])
        rng = np.random.default_rng(12)
        robot_poses = []
        mocap_poses = []
        for _ in range(20):
            T_w_e = make_pose(
                rng.normal(scale=0.7, size=3),
                rng.uniform(-0.5, 0.5, size=3))
            T_m_k = np.linalg.inv(T_w_m_truth) @ T_w_e @ T_e_k_truth
            # Translation noise makes a relative-motion translation followed
            # by a one-sided tool average differ from the absolute-pose LS
            # optimum that the reported residual uses.
            T_m_k[:3, 3] += rng.normal(scale=0.002, size=3)
            robot_poses.append(T_w_e)
            mocap_poses.append(T_m_k)

        T_w_m, T_e_k, metrics = fit_eye_to_hand_transform(
            robot_poses, mocap_poses, method='PARK')
        design = np.vstack([
            np.hstack((np.eye(3), -T[:3, :3])) for T in robot_poses])
        rhs = np.hstack([
            T_w_e[:3, 3] - T_w_m[:3, :3] @ T_m_k[:3, 3]
            for T_w_e, T_m_k in zip(robot_poses, mocap_poses)])
        solution = np.hstack((T_w_m[:3, 3], T_e_k[:3, 3]))
        normal_gradient = design.T @ (design @ solution - rhs)
        np.testing.assert_allclose(normal_gradient, 0.0, atol=1e-10)
        self.assertEqual(metrics['translation_design_rank'], 6)
        self.assertEqual(
            metrics['translation_refinement'],
            'joint absolute-pose least squares')


class MocapRigidBodyParsingTest(unittest.TestCase):

    def test_rigid_body_list_by_id(self):
        payload = {
            'rigid_bodies': [
                {'id': 2, 'position': [9.0, 9.0, 9.0]},
                {'id': 1, 'position': [1000.0, 2000.0, 1500.0]},
            ],
        }
        packet = parse_mocap_packet(
            json.dumps(payload).encode('utf-8'),
            marker_id=1,
            position_unit='mm',
        )
        self.assertIsNotNone(packet)
        np.testing.assert_allclose(
            packet.position_m, [1.0, 2.0, 1.5], rtol=0.0, atol=1e-7)

    def test_rigid_body_list_by_name(self):
        payload = {
            'rigid_bodies': [
                {'name': 'ball', 'position': [9.0, 9.0, 9.0]},
                {'name': 'K_body', 'position': [1000.0, 2000.0, 1500.0]},
            ],
        }
        packet = parse_mocap_packet(
            json.dumps(payload).encode('utf-8'),
            marker_id=999,
            object_name='K_body',
            position_unit='mm',
        )
        self.assertIsNotNone(packet)
        np.testing.assert_allclose(
            packet.position_m, [1.0, 2.0, 1.5], rtol=0.0, atol=1e-7)

    def test_rigid_body_prefixed_fields(self):
        payload = {
            'rigid_body_1_px': 500.0,
            'rigid_body_1_py': -250.0,
            'rigid_body_1_pz': 1200.0,
        }
        packet = parse_mocap_packet(
            json.dumps(payload).encode('utf-8'),
            marker_id=1,
            position_unit='mm',
        )
        self.assertIsNotNone(packet)
        np.testing.assert_allclose(
            packet.position_m, [0.5, -0.25, 1.2], rtol=0.0, atol=1e-7)


    def test_transform_waist03_to_sim_matches_probe_defaults(self):
        pos = np.array([0.1, -0.2, 0.3], dtype=np.float64)
        out = transform_waist03_to_sim(pos)
        np.testing.assert_allclose(
            out, [0.1, -0.176417, 1.602396], rtol=0.0, atol=1e-6)

    def test_parse_mocap_objects_ball_and_rigid_body(self):
        payload = {
            'rigid_bodies': [
                {'id': 1, 'position': [1000.0, 2000.0, 1500.0]},
                {'id': 2, 'position': [500.0, -250.0, 1200.0]},
            ],
        }
        objects = parse_mocap_objects(
            json.dumps(payload).encode('utf-8'),
            ball_marker_id=2,
            rigid_body_id=1,
            position_unit='mm',
        )
        self.assertIsNotNone(objects['ball'])
        self.assertIsNotNone(objects['rigid_body'])
        np.testing.assert_allclose(
            objects['ball'].position_m, [0.5, -0.25, 1.2], rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(
            objects['rigid_body'].position_m, [1.0, 2.0, 1.5], rtol=0.0, atol=1e-7)

    def test_optitrack_rigid_body_slot_and_id(self):
        payload = {
            'opti_rb_2_id': 0,
            'opti_rb_2_px': 3000.0,
            'opti_rb_2_py': 0.0,
            'opti_rb_2_pz': 0.0,
            'opti_rb_2_qw': 1.0,
            'opti_rb_2_qx': 0.0,
            'opti_rb_2_qy': 0.0,
            'opti_rb_2_qz': 0.0,
            'opti_rb_2_btracked': 1,
        }
        packet = parse_mocap_packet(
            json.dumps(payload).encode('utf-8'),
            object_kind='opti_rb',
            rigid_body_slot=2,
            rigid_body_id=2,
            position_unit='mm',
        )
        self.assertIsNotNone(packet)
        np.testing.assert_allclose(
            packet.position_m, [3.0, 0.0, 0.0], rtol=0.0, atol=1e-7)
        np.testing.assert_allclose(
            packet.orientation_wxyz, [1.0, 0.0, 0.0, 0.0],
            rtol=0.0, atol=1e-12)

    def test_optitrack_validates_id_value_when_slot_differs(self):
        payload = {
            'opti_rb_2_id': 17,
            'opti_rb_2_px': 3000.0,
            'opti_rb_2_py': 0.0,
            'opti_rb_2_pz': 0.0,
            'opti_rb_2_qw': 1.0,
            'opti_rb_2_qx': 0.0,
            'opti_rb_2_qy': 0.0,
            'opti_rb_2_qz': 0.0,
            'opti_rb_2_btracked': 1,
        }
        matched = parse_mocap_packet(
            json.dumps(payload).encode('utf-8'),
            object_kind='opti_rb',
            rigid_body_slot=2,
            rigid_body_id=17,
            position_unit='mm')
        mismatched = parse_mocap_packet(
            json.dumps(payload).encode('utf-8'),
            object_kind='opti_rb',
            rigid_body_slot=2,
            rigid_body_id=18,
            position_unit='mm')
        self.assertIsNotNone(matched)
        self.assertIsNone(mismatched)


if __name__ == '__main__':
    unittest.main()
