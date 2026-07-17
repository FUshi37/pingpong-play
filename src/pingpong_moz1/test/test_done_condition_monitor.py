#!/usr/bin/env python3

import math
import sys
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.done_condition_monitor import (  # noqa: E402
    DoneConditionMonitor,
    _angle_in_limits_rad,
    _quat_angle_error_deg,
)


class DoneConditionMonitorOrientationTest(unittest.TestCase):

    @staticmethod
    def _safe_state():
        return {
            "ball_pos_m": np.array([0.0, 0.0, 1.2], dtype=np.float32),
            "ball_vel_m_s": np.array([0.0, 0.0, -1.0], dtype=np.float32),
            "racket_pos_m": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "racket_anchor_m": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        }

    def test_default_orientation_checks_are_disabled(self):
        monitor = DoneConditionMonitor()

        is_done, reason, conds = monitor.check_done(**self._safe_state())

        self.assertFalse(is_done)
        self.assertEqual(reason, "")
        self.assertNotIn("ee_psi_out_of_bounds", conds)
        self.assertNotIn("ee_orientation_error_too_large", conds)

    def test_psi_limits_trigger_done_when_outside(self):
        monitor = DoneConditionMonitor(ee_psi_limits_rad=(-0.5, 0.5))

        is_done, reason, conds = monitor.check_done(
            **self._safe_state(), ee_psi_rad=0.8)

        self.assertTrue(is_done)
        self.assertEqual(reason, "ee_psi_out_of_bounds")
        self.assertTrue(conds["ee_psi_out_of_bounds"])

    def test_workspace_still_checked_when_ball_conditions_are_skipped(self):
        monitor = DoneConditionMonitor(ee_x_limits=(-0.1, 0.1))
        state = self._safe_state()
        state["ball_pos_m"] = np.array([0.0, 0.0, 0.1], dtype=np.float32)
        state["racket_pos_m"] = np.array([0.2, 0.0, 1.0], dtype=np.float32)

        is_done, reason, conds = monitor.check_done(
            **state, check_ball_conditions=False)

        self.assertTrue(is_done)
        self.assertEqual(reason, "ee_x_out_of_workspace")
        self.assertFalse(conds["ball_too_low"])
        self.assertTrue(conds["ee_x_out_of_workspace"])

    def test_wrapped_psi_limits(self):
        limits = (2.7, -2.7)

        self.assertTrue(_angle_in_limits_rad(math.pi, limits))
        self.assertTrue(_angle_in_limits_rad(-math.pi, limits))
        self.assertFalse(_angle_in_limits_rad(0.0, limits))

    def test_quaternion_orientation_error_triggers_done(self):
        monitor = DoneConditionMonitor(
            ee_orientation_reference_xyzw=(0.0, 0.0, 0.0, 1.0),
            ee_max_orientation_error_deg=30.0,
        )
        yaw_90 = (
            0.0,
            0.0,
            math.sin(math.pi / 4.0),
            math.cos(math.pi / 4.0),
        )

        is_done, reason, conds = monitor.check_done(
            **self._safe_state(), ee_quat_xyzw=np.array(yaw_90))

        self.assertTrue(is_done)
        self.assertEqual(reason, "ee_orientation_error_too_large")
        self.assertTrue(conds["ee_orientation_error_too_large"])

    def test_missing_quaternion_does_not_raise(self):
        monitor = DoneConditionMonitor(
            ee_orientation_reference_xyzw=(0.0, 0.0, 0.0, 1.0),
            ee_max_orientation_error_deg=30.0,
        )

        is_done, reason, conds = monitor.check_done(
            **self._safe_state(), ee_quat_xyzw=None)

        self.assertFalse(is_done)
        self.assertEqual(reason, "")
        self.assertFalse(conds["ee_orientation_error_too_large"])

    def test_quaternion_sign_does_not_change_orientation_error(self):
        err = _quat_angle_error_deg(
            np.array([0.0, 0.0, 0.0, -1.0]),
            np.array([0.0, 0.0, 0.0, 1.0]),
        )

        self.assertAlmostEqual(err, 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
