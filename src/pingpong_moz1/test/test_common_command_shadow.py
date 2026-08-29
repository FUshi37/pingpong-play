#!/usr/bin/env python3

import csv
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = (
    PACKAGE_ROOT / "pingpong_controller" / "tools"
    / "common_command_shadow.py"
)
SPEC = importlib.util.spec_from_file_location(
    "common_command_shadow", TOOL_PATH)
SHADOW = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = SHADOW
SPEC.loader.exec_module(SHADOW)


def observation(*, q=0.0, command_before=0.0):
    obs = np.zeros(SHADOW.OBS_DIM, dtype=float)
    obs[0:7] = q
    obs[42:49] = command_before - q
    return obs.tolist()


def compensation(*, nominal_weight=0.3, active=True):
    return {
        "enabled": True,
        "available": True,
        "active": active,
        "mode": "inverse_mpc",
        "delayMs": 72.0,
        "tauS": 0.074,
        "gain": 1.0,
        "error": None,
        "mpcBeta": 0.9,
        "mpcDelayScale": 1.0,
        "mpcTauScale": 0.75,
        "mpcHorizonSteps": 6,
        "mpcTrackingWeight": 1.0,
        "mpcNominalWeight": nominal_weight,
        "mpcDeltaWeight": 0.1,
        "mpcMaxDeltaRad": 0.35,
        "bufferLen": 19,
    }


def frame(index, real_before, real_after, sim_before, sim_after, *, comp=None):
    real = {
        "policyObs": observation(command_before=real_before),
        "policyCommand": {"position": [real_after] * 7},
        "command": {"compensation": comp or compensation()},
    }
    independent = {
        "policyObs": observation(command_before=sim_before),
        "policyCommand": [sim_after] * 7,
    }
    return SHADOW.Frame(
        source_seq=100 + index,
        real=real,
        independent=independent,
        episode=1,
        stream_seq=index + 1,
    )


class TrainingScaleTest(unittest.TestCase):

    def test_uses_std_then_relative_rms_then_absolute_floor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "obs_trace.csv"
            columns = [f"obs/{index:03d}" for index in range(SHADOW.OBS_DIM)]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                first = {name: 0.0 for name in columns}
                second = {name: 0.0 for name in columns}
                first["obs/000"] = 0.0
                second["obs/000"] = 2.0
                first["obs/001"] = 4.0
                second["obs/001"] = 4.0
                writer.writerows([first, second])

            scale = SHADOW._load_training_scale(path)

        self.assertEqual(scale.rows, 2)
        self.assertAlmostEqual(scale.scale[0], 1.0)
        self.assertAlmostEqual(scale.scale[1], 0.2)
        self.assertAlmostEqual(scale.scale[2], 1e-4)


class MetricTest(unittest.TestCase):

    def test_actual_recorded_observations_are_primary_acceptance_variant(self):
        self.assertEqual(
            SHADOW.PRIMARY_METRIC_VARIANT,
            "actualRecordedWindows",
        )

    def test_full_selected_group_and_per_dimension_scores(self):
        left = np.ones((3, SHADOW.OBS_DIM), dtype=float)
        right = np.zeros_like(left)
        scale = np.full(SHADOW.OBS_DIM, 2.0)

        result = SHADOW._gap_metrics(left, right, scale)

        self.assertAlmostEqual(result["full67"]["rawRms"], 1.0)
        self.assertAlmostEqual(result["full67"]["normalizedRms"], 0.5)
        self.assertAlmostEqual(
            result["selectedActuator39"]["normalizedRms"], 0.5)
        self.assertAlmostEqual(
            result["groups"]["racket_velocity"]["rawRms"], 1.0)
        self.assertEqual(len(result["perDimensionRawRms"]), 67)

    def test_canonicalization_audit_flags_large_racket_velocity_change(self):
        actual = np.zeros((20, SHADOW.OBS_DIM), dtype=float)
        canonical = actual.copy()
        actual[:, 29:32] = 0.1
        windows = SHADOW._window_report(
            np.arange(20),
            {"actualRealToCanonicalReal": (actual, canonical)},
            np.ones(SHADOW.OBS_DIM),
            0.005,
        )

        summary = SHADOW._canonicalization_summary(windows)

        self.assertTrue(summary["materialMismatch"])
        self.assertTrue(
            summary["streams"]["actualRealToCanonicalReal"]
            ["materialMismatch"])


class ResetMaskTest(unittest.TestCase):

    def test_asymmetric_nominal_reset_masks_remaining_tail(self):
        frames = (
            frame(0, 0.0, 0.1, 0.0, 0.1),
            frame(1, 0.1, 0.2, 0.1, 0.2),
            frame(2, 0.0, 0.1, 0.2, 0.3),
            frame(3, 0.1, 0.2, 0.3, 0.4),
        )

        plan = SHADOW._build_replay_plan(frames)

        self.assertEqual(plan.reset_boundaries, frozenset({0}))
        self.assertEqual(plan.asymmetric_tail_start, 2)
        self.assertEqual(
            plan.mask_reasons[-1]["reason"],
            "asymmetric_real_nominal_reset_tail",
        )

    def test_compensation_inactive_tick_starts_masked_tail(self):
        frames = (
            frame(0, 0.0, 0.1, 0.0, 0.1),
            frame(
                1, 0.1, 0.2, 0.1, 0.2,
                comp=compensation(active=False)),
            frame(
                2, 0.2, 0.3, 0.2, 0.3,
                comp=compensation(nominal_weight=0.7, active=False)),
        )

        plan = SHADOW._build_replay_plan(frames)

        self.assertEqual(plan.asymmetric_tail_start, 1)
        self.assertEqual(
            plan.mask_reasons[-1]["reason"],
            "real_compensation_inactive_tail",
        )

    def test_symmetric_nominal_reset_starts_new_replay_segment(self):
        frames = (
            frame(0, 0.0, 0.1, 0.0, 0.1),
            frame(1, 0.1, 0.2, 0.1, 0.2),
            frame(2, 0.0, 0.1, 0.0, 0.1),
            frame(3, 0.1, 0.2, 0.1, 0.2),
        )

        plan = SHADOW._build_replay_plan(frames)

        self.assertEqual(plan.reset_boundaries, frozenset({0, 2}))
        self.assertIsNone(plan.asymmetric_tail_start)
        self.assertEqual(
            plan.mask_reasons[-1]["reason"],
            "symmetric_nominal_history_reset",
        )


class CompensationAuditTest(unittest.TestCase):

    def test_reads_tick_payload_and_separates_tuning_from_active_status(self):
        frames = (
            frame(
                0, 0.0, 0.1, 0.0, 0.1,
                comp=compensation(active=True)),
            frame(
                1, 0.1, 0.2, 0.1, 0.2,
                comp=compensation(active=True)),
            frame(
                2, 0.2, 0.3, 0.2, 0.3,
                comp=compensation(nominal_weight=0.7, active=False)),
        )

        plan = SHADOW._build_replay_plan(frames)
        report = SHADOW._compensation_audit(frames, plan)

        self.assertFalse(report["metadataDefaultsUsed"])
        self.assertEqual(report["alignedTuning"]["cohortCount"], 2)
        self.assertEqual(
            report["metricUnmaskedTuning"]["cohortCount"], 1)
        self.assertFalse(report["metricUnmaskedTuning"]["mixedCohort"])
        self.assertEqual(report["alignedTuning"]["changeCount"], 1)
        self.assertEqual(report["runtimeStatusChangeCount"], 1)


class RacketVelocityContractTest(unittest.TestCase):

    class Policy:
        max_contact_time_s = 0.5
        contact_height_offset_m = 0.0
        contact_lost_ball_timeout_s = 0.35
        delay_conditioning_tau_s = 0.072

        @staticmethod
        def _estimate_contact_time(*_args, **_kwargs):
            return 0.2

    def test_uses_strict_five_millisecond_position_secant(self):
        row = {"policyObs": np.zeros(SHADOW.OBS_DIM).tolist()}
        row["policyObs"][26:29] = [0.001, -0.002, 0.003]
        previous = np.zeros(3)

        obs, _rpos = SHADOW._canonical_recorded_obs(
            row,
            label="test",
            previous_rpos=previous,
            dt_s=0.005,
            age_s=0.0,
            policy=self.Policy(),
        )

        np.testing.assert_allclose(obs[29:32], [0.2, -0.4, 0.6])
        self.assertAlmostEqual(obs[65], 0.2)
        self.assertAlmostEqual(obs[66], 0.128)


if __name__ == "__main__":
    unittest.main()
