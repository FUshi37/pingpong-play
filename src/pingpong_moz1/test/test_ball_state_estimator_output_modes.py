#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.ball_state_estimator import (  # noqa: E402
    BallStateEstimator,
    BallStateEstimatorConfig,
    CONTROL_OUTPUT_CONTINUOUS,
    CONTROL_OUTPUT_PREDICT_ON_FRAME_HOLD,
)


class BallStateEstimatorOutputModeTest(unittest.TestCase):

    @staticmethod
    def _estimator(
            mode, *, enable_prediction=True, enable_ballistic_dynamics=True):
        cfg = BallStateEstimatorConfig(
            gravity_mm_s2=(0.0, 0.0, -9810.0),
            enable_prediction=enable_prediction,
            enable_aero_drag=False,
            max_extrapolation_s=1.0,
            invalid_after_s=1.0,
            control_output_mode=mode,
            frame_hold_output_rate_hz=50.0,
            enable_ballistic_dynamics=enable_ballistic_dynamics,
        )
        est = BallStateEstimator(cfg)
        est.update([0.0, 0.0, 1000.0], 0.0)
        est.update([0.0, 0.0, 1000.0], 0.02)
        return est

    def test_continuous_mode_changes_between_control_ticks(self):
        est = self._estimator(CONTROL_OUTPUT_CONTINUOUS)

        first = est.predict_to(0.02)
        second = est.predict_to(0.025)

        self.assertEqual(first['output_mode'], CONTROL_OUTPUT_CONTINUOUS)
        self.assertFalse(first['snapshot_new'])
        self.assertLess(second['position'][2], first['position'][2])
        self.assertLess(second['velocity'][2], first['velocity'][2])
        self.assertGreater(second['extrapolation_s'], first['extrapolation_s'])

    def test_frame_hold_mode_keeps_state_fixed_and_ages_only_when_overdue(self):
        est = self._estimator(CONTROL_OUTPUT_PREDICT_ON_FRAME_HOLD)

        first = est.predict_to(0.02)
        held = est.predict_to(0.025)
        overdue = est.predict_to(0.045)

        self.assertTrue(first['snapshot_new'])
        self.assertFalse(held['snapshot_new'])
        np.testing.assert_array_equal(first['position'], held['position'])
        np.testing.assert_array_equal(first['velocity'], held['velocity'])
        np.testing.assert_array_equal(first['position'], overdue['position'])
        self.assertAlmostEqual(held['age_s'], 0.0, places=9)
        self.assertAlmostEqual(overdue['age_s'], 0.005, places=9)
        self.assertAlmostEqual(overdue['snapshot_hold_s'], 0.025, places=9)
        self.assertAlmostEqual(
            overdue['control_age_s'],
            first['extrapolation_s'] + 0.025,
            places=9,
        )

    def test_frame_hold_refreshes_once_due_after_new_measurement(self):
        est = self._estimator(CONTROL_OUTPUT_PREDICT_ON_FRAME_HOLD)
        first = est.predict_to(0.02)

        est.update([10.0, 0.0, 1000.0], 0.03)
        early = est.predict_to(0.035)
        refreshed = est.predict_to(0.04)
        held = est.predict_to(0.045)

        self.assertFalse(early['snapshot_new'])
        self.assertEqual(early['snapshot_sequence'], first['snapshot_sequence'])
        self.assertTrue(refreshed['snapshot_new'])
        self.assertGreater(
            refreshed['snapshot_sequence'], first['snapshot_sequence'])
        self.assertFalse(held['snapshot_new'])
        np.testing.assert_array_equal(refreshed['position'], held['position'])
        np.testing.assert_array_equal(refreshed['velocity'], held['velocity'])

    def test_prediction_disabled_preserves_no_prediction_path(self):
        est = self._estimator(
            CONTROL_OUTPUT_PREDICT_ON_FRAME_HOLD,
            enable_prediction=False,
        )

        first = est.predict_to(0.02)
        second = est.predict_to(0.04)
        stale = est.predict_to(0.14)

        self.assertEqual(first['output_mode'], 'no_prediction')
        self.assertEqual(first['extrapolation_s'], 0.0)
        self.assertFalse(stale['predicted'])
        self.assertEqual(stale['state'], 'tracking')
        np.testing.assert_array_equal(first['position'], second['position'])
        np.testing.assert_array_equal(first['velocity'], second['velocity'])
        np.testing.assert_array_equal(first['position'], stale['position'])
        np.testing.assert_array_equal(first['velocity'], stale['velocity'])

    def test_held_ball_dynamics_do_not_create_gravity_drift(self):
        est = self._estimator(
            CONTROL_OUTPUT_PREDICT_ON_FRAME_HOLD,
            enable_ballistic_dynamics=False,
        )
        snapshots = []
        next_measurement_s = 0.04
        for step in range(5, 121):
            now_s = step / 200.0
            while now_s + 1e-9 >= next_measurement_s:
                est.update([0.0, 0.0, 1000.0], next_measurement_s)
                next_measurement_s += 1.0 / 60.0
            out = est.predict_to(now_s)
            if out['snapshot_new']:
                snapshots.append(out)

        z_values = [out['position'][2] for out in snapshots]
        vz_values = [out['velocity'][2] for out in snapshots]
        self.assertLess(max(z_values) - min(z_values), 1e-6)
        self.assertLess(max(abs(vz) for vz in vz_values), 1e-6)

    def test_release_switch_reseeds_ballistic_velocity_from_history(self):
        est = self._estimator(
            CONTROL_OUTPUT_PREDICT_ON_FRAME_HOLD,
            enable_ballistic_dynamics=False,
        )
        est.predict_to(0.02)
        est.update([0.0, 0.0, 990.0], 0.04)
        est.update([0.0, 0.0, 970.0], 0.06)

        est.set_ballistic_dynamics(
            True, reinitialize_from_history=True)
        out = est.predict_to(0.065)
        diag = est.diagnostics()

        self.assertTrue(out['snapshot_new'])
        self.assertTrue(diag['ballistic_dynamics_enabled'])
        self.assertEqual(diag['dynamics_reseeds'], 1)
        self.assertLess(out['velocity'][2], 0.0)


if __name__ == '__main__':
    unittest.main()
