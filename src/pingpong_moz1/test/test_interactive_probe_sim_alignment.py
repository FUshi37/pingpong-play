#!/usr/bin/env python3

import importlib.util
import json
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

PROBE_PATH = (
    PACKAGE_ROOT / "pingpong_controller" / "tools"
    / "interactive_policy_safety_probe.py"
)
SPEC = importlib.util.spec_from_file_location("interactive_policy_safety_probe", PROBE_PATH)
PROBE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = PROBE
SPEC.loader.exec_module(PROBE)


def real_sample(
    seq,
    *,
    mode="policy",
    ball=(0.1, -0.2, 1.3),
    policy_executed=None,
):
    sample = {
        "sourceSeq": seq,
        "tick": seq,
        "mode": mode,
        "policyInput": {
            "ball": list(ball),
            "velocity": [0.01, 0.02, -0.3],
            "valid": True,
            "age": 0.005,
            "ballObservationSeq": 7,
        },
        "policyObs": [float(seq)],
        "policyAction": [float(seq)],
    }
    if policy_executed is not None:
        sample["policyExecuted"] = bool(policy_executed)
    return sample


def sim_sample(
    seq,
    *,
    ball=(0.1, -0.2, 1.3),
    continuous=True,
    submitted=None,
    completed=None,
):
    sample = {
        "policyObs": [float(seq)],
        "policyAction": [float(seq)],
        "sync": {
            "sourceSeq": seq,
            "streamSeq": seq + 1,
            "inputBall": list(ball),
            "inputVelocity": [0.01, 0.02, -0.3],
            "inputValid": True,
            "inputAgeS": 0.005,
            "ballObservationSeq": 7,
            "streamContinuous": continuous,
            "resultDrops": 0,
        },
    }
    if submitted is not None:
        sample["sync"]["submittedWallTimeS"] = float(submitted)
    if completed is not None:
        sample["sync"]["completedWallTimeS"] = float(completed)
    return sample


class SourceSeqAlignmentTest(unittest.TestCase):

    def test_joins_out_of_order_sim_results_by_source_seq(self):
        real = [real_sample(10), real_sample(11), real_sample(12, mode="release_gate")]
        sim = [sim_sample(11), sim_sample(10)]

        merged, report = PROBE._align_sim_real_samples(real, sim)

        self.assertTrue(report["alignmentValid"])
        self.assertEqual(report["matchedCount"], 2)
        self.assertEqual(report["missingSimCount"], 0)
        self.assertEqual(merged[0]["sim"]["sync"]["sourceSeq"], 10)
        self.assertEqual(merged[1]["sim"]["sync"]["sourceSeq"], 11)
        self.assertNotIn("sim", merged[2])

    def test_missing_or_changed_policy_input_invalidates_alignment(self):
        real = [real_sample(20), real_sample(21)]
        sim = [sim_sample(20, ball=(0.1001, -0.2, 1.3))]

        _merged, report = PROBE._align_sim_real_samples(real, sim)

        self.assertFalse(report["alignmentValid"])
        self.assertEqual(report["missingSimSeq"], [21])
        self.assertEqual(report["inputMismatchSeq"], [20])

    def test_stream_gap_invalidates_even_when_all_source_seqs_exist(self):
        real = [real_sample(30)]
        sim = [sim_sample(30, continuous=False)]

        _merged, report = PROBE._align_sim_real_samples(real, sim)

        self.assertFalse(report["alignmentValid"])
        self.assertEqual(report["invalidStreamSeq"], [30])

    def test_recording_window_hole_and_orphan_sim_invalidate(self):
        real = [real_sample(40), real_sample(42)]
        sim = [sim_sample(40), sim_sample(41), sim_sample(42)]

        _merged, report = PROBE._align_sim_real_samples(real, sim)

        self.assertFalse(report["alignmentValid"])
        self.assertEqual(report["realSourceGapSeq"], [41])
        self.assertEqual(report["orphanSimInWindowSeq"], [41])

    def test_orphan_before_recording_window_is_allowed(self):
        real = [real_sample(50), real_sample(51)]
        sim = [sim_sample(49), sim_sample(50), sim_sample(51)]

        _merged, report = PROBE._align_sim_real_samples(real, sim)

        self.assertTrue(report["alignmentValid"])
        self.assertEqual(report["orphanSimCount"], 1)
        self.assertEqual(report["orphanSimInWindowCount"], 0)

    def test_done_tick_that_executed_policy_is_required_for_alignment(self):
        real = [real_sample(60, mode="done_failsafe", policy_executed=True)]

        _merged, missing_report = PROBE._align_sim_real_samples(real, [])
        merged, matched_report = PROBE._align_sim_real_samples(
            real, [sim_sample(60)])

        self.assertFalse(missing_report["alignmentValid"])
        self.assertEqual(missing_report["missingSimSeq"], [60])
        self.assertTrue(matched_report["alignmentValid"])
        self.assertTrue(merged[0]["simAligned"])

    def test_actual_policy_ball_observation_must_match(self):
        real = [real_sample(70)]
        real[0]["policyBallObservation"] = [0.1, -0.2, 1.3, 0.01, 0.02, -0.3]
        sim = [sim_sample(70)]
        sim[0]["policyBallObservation"] = [0.1, -0.2, 1.2, 0.01, 0.02, -0.3]

        _merged, report = PROBE._align_sim_real_samples(real, sim)

        self.assertFalse(report["alignmentValid"])
        self.assertEqual(report["inputMismatchSeq"], [70])

    def test_realtime_report_distinguishes_same_rate_from_catchup(self):
        real = [real_sample(seq) for seq in range(25)]
        fast_sim = [
            sim_sample(
                seq,
                submitted=100.0 + seq * 0.005,
                completed=100.001 + seq * 0.005,
            )
            for seq in range(25)
        ]
        slow_sim = [
            sim_sample(
                seq,
                submitted=100.0 + seq * 0.005,
                completed=100.001 + seq / 120.0,
            )
            for seq in range(25)
        ]

        fast = PROBE._realtime_alignment_report(real, fast_sim, 200.0)
        slow = PROBE._realtime_alignment_report(real, slow_sim, 200.0)

        self.assertTrue(fast["realtimeAligned"])
        self.assertFalse(slow["realtimeAligned"])
        self.assertAlmostEqual(fast["completionHz"], 200.0, places=3)


class RecorderCadenceTest(unittest.TestCase):

    def test_control_rate_recording_keeps_every_logical_tick(self):
        recorder = PROBE.AsyncProbeRecorder.__new__(PROBE.AsyncProbeRecorder)
        recorder.cfg = SimpleNamespace(
            recording_control_rate_hz=200.0,
            control_rate_hz=200.0,
        )
        recorder.lock = threading.Lock()
        recorder.active = True
        recorder.frozen_source_seq = None
        recorder.next_control_record_wall = 10**12
        written = []
        recorder._put = lambda item: written.append(item) or True

        for seq in range(5):
            recorder.record_control_sample({"sourceSeq": seq})

        self.assertEqual(
            [item["sample"]["sourceSeq"] for item in written],
            [0, 1, 2, 3, 4],
        )

    def test_freeze_keeps_source_seq_zero_and_rejects_later_ticks(self):
        recorder = PROBE.AsyncProbeRecorder.__new__(PROBE.AsyncProbeRecorder)
        recorder.cfg = SimpleNamespace(
            recording_control_rate_hz=200.0,
            control_rate_hz=200.0,
        )
        recorder.lock = threading.Lock()
        recorder.active = True
        recorder.frozen_source_seq = 0
        recorder.next_control_record_wall = 0.0
        written = []
        recorder._put = lambda item: written.append(item) or True

        recorder.record_control_sample({"sourceSeq": 0})
        recorder.record_control_sample({"sourceSeq": 1})

        self.assertEqual(
            [item["sample"]["sourceSeq"] for item in written], [0])


class CpuSimMirrorActuatorModelTest(unittest.TestCase):

    def test_actuator_delay_filter_runs_after_compensation_reference(self):
        mirror = PROBE.RealBallSimMirror.__new__(PROBE.RealBallSimMirror)
        mirror.dt = 0.01
        mirror.cfg = SimpleNamespace(
            comp_delay_ms=20.0,
            comp_actuator_tau_s=0.0,
            comp_actuator_gain=1.0,
            policy_obs_dim=67,
        )
        mirror.policy = SimpleNamespace(
            arm_q_low=PROBE.np.full(7, -10.0, dtype=PROBE.np.float32),
            arm_q_high=PROBE.np.full(7, 10.0, dtype=PROBE.np.float32),
        )
        mirror.warm_arm_q = PROBE.np.zeros(7, dtype=PROBE.np.float32)
        mirror.actuator_model_key = None
        mirror.command_buffer = None
        mirror.actuator_command_buffer = None
        mirror.arm_q_ref_active = None
        mirror.arm_actuator_q_ref_active = None
        mirror.arm_applied_q = None

        seed = PROBE.np.zeros(7, dtype=PROBE.np.float32)
        nominal = PROBE.np.ones(7, dtype=PROBE.np.float32)
        compensated = PROBE.np.full(7, 2.0, dtype=PROBE.np.float32)
        options = mirror.cfg

        applied_history = []
        active_compensated = None
        for _ in range(3):
            _active, active_compensated, applied = mirror._run_actuator_model(
                nominal,
                compensated,
                options,
                seed_q=seed,
            )
            applied_history.append(applied.copy())

        PROBE.np.testing.assert_allclose(applied_history[0], seed)
        PROBE.np.testing.assert_allclose(applied_history[1], seed)
        PROBE.np.testing.assert_allclose(applied_history[2], compensated)
        PROBE.np.testing.assert_allclose(active_compensated, compensated)


class AnalysisJsonTest(unittest.TestCase):

    def test_browser_json_replaces_nonfinite_values_with_null(self):
        encoded = PROBE._strict_json_dumps({
            "nan": float("nan"),
            "nested": [float("inf"), -float("inf"), 1.0],
        })

        self.assertEqual(
            json.loads(encoded),
            {"nan": None, "nested": [None, None, 1.0]},
        )

    def test_constant_series_correlation_is_strict_json_null(self):
        real = PROBE.np.zeros((4, 1), dtype=PROBE.np.float64)
        sim = PROBE.np.zeros((4, 1), dtype=PROBE.np.float64)

        rows = PROBE._per_dim_comparison(
            real, sim, group="policy_obs", names=["constant"])

        self.assertIsNone(rows[0]["corr"])
        encoded = json.dumps({"rows": rows}, allow_nan=False)
        self.assertIn('"corr": null', encoded)


class DoneLatchTest(unittest.TestCase):

    def test_latched_done_does_not_execute_policy_again(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.controller = SimpleNamespace(use_valid_ball_obs_age=True)
        probe.cfg = SimpleNamespace(control_dt=0.005, ball_lost_timeout_s=0.6)
        probe.now_s = 1.0
        probe.last_valid_time_s = None
        probe.has_captured = True
        probe.failsafe_active = False
        probe.done_failsafe_active = True
        probe.done_failsafe_reason = "workspace"
        probe.done_failsafe_conditions = {"x": True}
        calls = []
        probe._sync_policy_to_safe_pose = lambda: calls.append("safe")
        probe._mark_policy_paused = lambda: calls.append("paused")
        options = SimpleNamespace(rl_use_valid_ball_obs_age=True)

        result = probe._run_one_control_step(
            PROBE.np.zeros(3), PROBE.np.zeros(3), True, 0.0, options)

        self.assertEqual(result[-1], False)
        self.assertEqual(result[0], "done_failsafe")
        self.assertEqual(calls, ["safe", "paused"])


class WorkspaceVelocityProjectionTest(unittest.TestCase):

    def test_halfspace_projection_preserves_unconstrained_tangent(self):
        nominal = PROBE.np.array([1.0, 0.8, -0.4])
        rows = PROBE.np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        bounds = PROBE.np.zeros(2)

        selected, info = PROBE._project_velocity_halfspaces(
            nominal, rows, bounds, scale=PROBE.np.ones(3))

        PROBE.np.testing.assert_allclose(selected[:2], PROBE.np.zeros(2), atol=1e-8)
        self.assertAlmostEqual(selected[2], nominal[2])
        self.assertTrue(info["feasible"])

    def test_infeasible_barrier_returns_maximum_reachable_braking(self):
        nominal = PROBE.np.array([1.0])
        selected, info = PROBE._project_velocity_halfspaces(
            nominal,
            PROBE.np.array([[1.0]]),
            PROBE.np.array([0.0]),
            lower=PROBE.np.array([0.8]),
            upper=PROBE.np.array([1.2]),
            scale=PROBE.np.ones(1),
        )

        self.assertAlmostEqual(selected[0], 0.8, places=7)
        self.assertFalse(info["feasible"])

    def test_task_metric_does_not_create_coupled_cartesian_tangent(self):
        nominal = PROBE.np.array([1.0, 0.0])
        jacobian = PROBE.np.array([
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
        ])
        metric = jacobian.T @ jacobian + 1e-4 * PROBE.np.eye(2)

        selected, info = PROBE._project_velocity_halfspaces(
            nominal,
            PROBE.np.array([[1.0, 1.0]]),
            PROBE.np.array([0.0]),
            metric=metric,
        )

        task_selected = jacobian @ selected
        self.assertLessEqual(task_selected[0], 1e-7)
        self.assertAlmostEqual(task_selected[1], 0.0, places=3)
        self.assertTrue(info["feasible"])

    def test_uncontrollable_negative_bound_is_not_dropped(self):
        selected, info = PROBE._project_velocity_halfspaces(
            PROBE.np.array([0.25]),
            PROBE.np.array([[0.0]]),
            PROBE.np.array([-0.1]),
        )

        self.assertAlmostEqual(selected[0], 0.25)
        self.assertFalse(info["feasible"])
        self.assertTrue(info["uncontrollable"])


class WorkspaceSoftSafetyTest(unittest.TestCase):

    def test_workspace_mode_accepts_new_soft_names(self):
        self.assertEqual(
            PROBE._workspace_safety_mode("position soft"), "position_soft")
        self.assertEqual(
            PROBE._workspace_safety_mode("predictive-soft"),
            "predictive_soft",
        )
        self.assertEqual(
            PROBE._workspace_safety_mode("impedance soft"),
            "impedance_soft",
        )

    def test_soft_modes_only_remove_arm_workspace_done_conditions(self):
        conditions = {
            "ee_x_out_of_workspace": True,
            "racket_predicted_too_far_from_anchor": True,
            "ee_orientation_error_too_large": True,
            "ball_too_low": True,
            "inactive": False,
        }

        hard = PROBE._hard_done_conditions(conditions, "predictive")
        soft = PROBE._hard_done_conditions(conditions, "predictive_soft")

        self.assertEqual(
            set(hard),
            {
                "ee_x_out_of_workspace",
                "racket_predicted_too_far_from_anchor",
                "ee_orientation_error_too_large",
                "ball_too_low",
            },
        )
        self.assertEqual(soft, {
            "ee_orientation_error_too_large": True,
            "ball_too_low": True,
        })

    def test_switching_to_soft_releases_arm_only_done_latch(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(workspace_safety_mode="predictive")
        probe.done_failsafe_active = True
        probe.done_failsafe_reason = "ee_x_out_of_workspace"
        probe.done_failsafe_conditions = {"ee_x_out_of_workspace": True}

        released = probe._release_incompatible_done_latch("position_soft")

        self.assertTrue(released)
        self.assertFalse(probe.done_failsafe_active)
        self.assertEqual(probe.done_failsafe_reason, "")
        self.assertEqual(probe.done_failsafe_conditions, {})

    def test_switching_to_soft_keeps_ball_done_latched(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(workspace_safety_mode="predictive")
        probe.done_failsafe_active = True
        probe.done_failsafe_reason = "ee_x_out_of_workspace"
        probe.done_failsafe_conditions = {
            "ee_x_out_of_workspace": True,
            "ball_too_low": True,
        }

        released = probe._release_incompatible_done_latch("predictive_soft")

        self.assertFalse(released)
        self.assertTrue(probe.done_failsafe_active)
        self.assertEqual(probe.done_failsafe_reason, "ball_too_low")
        self.assertEqual(
            probe.done_failsafe_conditions, {"ball_too_low": True})

    @staticmethod
    def _probe_with_linear_fk(initial_q0=0.0):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        initial = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        initial[0] = float(initial_q0)
        probe.safety_limiter = PROBE.RightArmCommandSafetyLimiter(
            initial, dt=0.005)
        probe.controller = SimpleNamespace(
            _compute_racket_pose_quat_base=lambda q: (
                PROBE.np.array([q[0], 0.0, 0.0]),
                PROBE.np.array([0.0, 0.0, 0.0, 1.0]),
                True,
            ),
        )
        probe.anchor = lambda: PROBE.np.zeros(3, dtype=PROBE.np.float64)
        probe.cfg = SimpleNamespace(
            done_racket_anchor_distance_m=100.0,
            done_racket_z_low_offset_m=-100.0,
            done_racket_z_high_offset_m=100.0,
            ee_x_limits_m=(-1.0, 1.0),
            ee_y_limits_m=None,
            ee_z_limits_m=None,
            ee_b_limits_rad=None,
            ee_b_min_rad=None,
            ee_b_max_rad=None,
            ee_c_limits_rad=None,
            ee_c_min_rad=None,
            ee_c_max_rad=None,
            ee_orientation_reference_xyzw=None,
            ee_max_orientation_error_deg=0.0,
            workspace_hold_trigger_margin_m=0.003,
        )
        probe.workspace_soft_filter = {}
        probe.workspace_hold_active = False
        probe.workspace_hold_q = initial.copy()
        probe.workspace_hold_last_safe_q = initial.copy()
        probe.workspace_hold_trigger_reason = ""
        probe.workspace_hold_constraint = ""
        probe.ros_bridge = None
        return probe

    def test_position_soft_rate_governs_distant_target_before_boundary(self):
        probe = self._probe_with_linear_fk()
        target = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        target[0] = 2.0

        safe = probe._filter_command_with_workspace_guard(
            target, profile="normal", apply_joint_limits=False)

        # A distant position target is converted into a bounded local velocity,
        # not teleported to the edge in one control tick.
        self.assertGreater(safe[0], 0.0)
        self.assertLessEqual(safe[0], 0.04)
        self.assertLessEqual(
            probe._parking_workspace_violation_score(safe), 1e-12)
        self.assertTrue(probe.workspace_soft_filter["active"])
        self.assertLess(probe.workspace_soft_filter["alpha"], 1.0)

    def test_predictive_soft_without_prediction_does_not_freeze(self):
        probe = self._probe_with_linear_fk(initial_q0=0.5)
        target = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        target[0] = 0.8

        safe = probe._filter_command_with_workspace_guard(
            target,
            profile="normal",
            apply_joint_limits=False,
            predictive_hold=True,
        )

        # No prediction must not freeze all joints.  The shared sampled-data
        # speed envelope may still turn a distant position jump into a local
        # step before the soft band.
        self.assertGreater(safe[0], 0.5)
        self.assertLessEqual(safe[0], 0.8)
        self.assertTrue(probe.workspace_soft_filter["predictive"])
        self.assertGreater(
            probe.workspace_soft_filter["selectedTaskVelocityMps"][0], 0.0)

    def test_predictive_soft_blocks_outward_but_keeps_tangential_motion(self):
        probe = self._probe_with_linear_fk(initial_q0=0.5)
        probe.done_monitor = SimpleNamespace(last_prediction={
            "condition": "ee_predicted_x_out_of_workspace",
            "value": [1.2, 0.0, 0.0],
        })
        target = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        target[0] = 0.8
        target[1] = 0.1

        safe = probe._filter_command_with_workspace_guard(
            target,
            profile="normal",
            apply_joint_limits=False,
            predictive_hold=True,
        )

        self.assertLess(safe[0], 0.5)
        self.assertAlmostEqual(safe[1], 0.1)
        self.assertEqual(
            probe.workspace_soft_filter["model"],
            "predictive_cartesian_cbf")
        self.assertTrue(probe.workspace_soft_filter["predictionSignal"])

    def test_predictive_soft_brakes_measured_outward_motion_earlier(self):
        def run(predictive):
            probe = self._probe_with_linear_fk(initial_q0=0.95)
            measured = probe.safety_limiter.last_cmd.copy()
            measured_dq = PROBE.np.zeros(7)
            measured_dq[0] = 0.5
            probe.ros_bridge = SimpleNamespace(
                get_joint_feedback=lambda: (
                    measured.copy(), measured_dq.copy()))
            probe.done_monitor = SimpleNamespace(
                prediction_horizon_s=0.06,
                prediction_margin_m=0.005,
                last_prediction={},
            )
            target = measured.copy()
            target[0] = 2.0
            safe = probe._filter_command_with_workspace_guard(
                target,
                profile="normal",
                apply_joint_limits=True,
                predictive_hold=predictive,
            )
            return safe, probe

        position_safe, position_probe = run(False)
        predictive_safe, predictive_probe = run(True)

        self.assertLess(predictive_safe[0], position_safe[0])
        self.assertTrue(
            predictive_probe.workspace_soft_filter["predictionSignal"])
        self.assertGreater(
            predictive_probe.workspace_soft_filter["terms"][0]["reserveM"],
            0.0,
        )
        self.assertFalse(
            position_probe.workspace_soft_filter.get("predictive", False))

    def test_position_soft_preserves_tangential_joint_motion(self):
        probe = self._probe_with_linear_fk(initial_q0=1.0)
        target = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        target[0] = 2.0
        target[1] = 0.1

        safe = probe._filter_command_with_workspace_guard(
            target, profile="normal", apply_joint_limits=False)

        self.assertLessEqual(safe[0], 1.00001)
        self.assertAlmostEqual(safe[1], 0.1)
        self.assertEqual(
            probe.workspace_soft_filter["projection"], "joint_velocity_qp")

    def test_position_soft_runtime_uses_component_workspace_guard(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        published = []
        captured = {}
        probe.ros_bridge = SimpleNamespace(
            publish_right_arm_command=lambda q: published.append(
                PROBE.np.asarray(q).copy()),
        )
        probe.cfg = SimpleNamespace(
            workspace_safety_mode="position_soft",
            enable_safety_limiter=False,
        )
        probe.compensation_was_enabled = False
        # This test targets the already-armed downstream workspace path; the
        # explicit first-publish feedback hold is covered separately below.
        probe.command_stream_active = True
        probe.last_command_disable_hold = {}
        probe.workspace_soft_prediction_active = False
        probe.workspace_soft_filter = {}

        def fake_filter(raw, **kwargs):
            captured.update(kwargs)
            probe.workspace_soft_filter = {"active": False}
            return PROBE.np.asarray(raw).copy()

        probe._filter_command_with_workspace_guard = fake_filter
        options = SimpleNamespace(
            publish_commands=True,
            workspace_safety_mode="position_soft",
        )

        probe._publish_control_command(PROBE.np.zeros(7), options)

        self.assertTrue(captured["apply_joint_limits"])
        self.assertFalse(captured["predictive_hold"])
        self.assertEqual(len(published), 1)

    def test_publish_feedback_keeps_policy_compensation_and_safe_layers_separate(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        nominal = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        published = []
        observed = []
        probe.ros_bridge = SimpleNamespace(
            publish_right_arm_command=lambda q: published.append(
                PROBE.np.asarray(q).copy()))
        probe.cfg = SimpleNamespace(
            workspace_safety_mode="position_soft",
            enable_safety_limiter=False,
        )
        probe.controller = SimpleNamespace(
            arm_cmd_q=nominal.astype(PROBE.np.float32).copy(),
            arm_cmd_qvel=nominal.astype(PROBE.np.float32).copy(),
            last_cmd_q=nominal.astype(PROBE.np.float32).copy(),
            last_cmd_qvel=nominal.astype(PROBE.np.float32).copy(),
            _cmd_history=[nominal.astype(PROBE.np.float32).copy()],
        )
        probe.safety_limiter = PROBE.RightArmCommandSafetyLimiter(
            nominal, dt=0.005)
        probe.compensator = SimpleNamespace(
            observe_published=lambda q: observed.append(
                PROBE.np.asarray(q).copy()))
        probe.compensation_was_enabled = False
        probe.command_stream_active = True
        probe.last_command_disable_hold = {}
        probe.workspace_soft_filter = {}

        def fake_compensation(q, _options):
            probe.compensation_was_enabled = True
            return PROBE.np.asarray(q, dtype=PROBE.np.float64) + 0.20

        def fake_filter(compensated, **_kwargs):
            candidate = PROBE.np.asarray(compensated).copy()
            safe = candidate.copy()
            safe[0] -= 0.01
            probe._workspace_soft_candidate_q = candidate.copy()
            probe._workspace_soft_candidate_vel = PROBE.np.zeros(7)
            probe.safety_limiter.last_cmd = safe.copy()
            probe.safety_limiter.last_vel = PROBE.np.zeros(7)
            probe.workspace_soft_filter = {
                "active": True,
                "intervened": True,
            }
            return safe

        probe._apply_actuator_compensation = fake_compensation
        probe._filter_command_with_workspace_guard = fake_filter
        options = SimpleNamespace(
            publish_commands=True,
            workspace_safety_mode="position_soft",
        )

        probe._publish_control_command(
            nominal, options, apply_compensation=True)

        self.assertEqual(len(published), 1)
        self.assertEqual(len(observed), 1)
        PROBE.np.testing.assert_allclose(observed[0], published[0])
        # A rejection after compensation belongs to the compensator buffer,
        # not to a stationary nominal policy integrator.  Neither the +0.20
        # bias nor the -0.01 safety correction enters nominal policy state.
        expected_policy = nominal.copy()
        PROBE.np.testing.assert_allclose(
            probe.controller.arm_cmd_q, expected_policy)
        PROBE.np.testing.assert_allclose(
            probe.last_policy_nominal_reference, nominal)
        PROBE.np.testing.assert_allclose(
            probe.last_compensated_reference, nominal + 0.20)
        PROBE.np.testing.assert_allclose(
            probe.last_published_safe_reference, published[0])

    def test_position_soft_y_boundary_preserves_x_command(self):
        probe = self._probe_with_linear_fk(initial_q0=1.0)
        probe.controller._compute_racket_pose_quat_base = lambda q: (
            PROBE.np.array([q[1], q[0], 0.0]),
            PROBE.np.array([0.0, 0.0, 0.0, 1.0]),
            True,
        )
        probe.cfg.ee_x_limits_m = (-10.0, 10.0)
        probe.cfg.ee_y_limits_m = (-1.0, 1.0)
        target = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        target[0] = 2.0  # outward in Cartesian y
        target[1] = 0.1  # tangential motion in Cartesian x

        safe = probe._filter_command_with_workspace_guard(
            target, profile="normal", apply_joint_limits=False)
        safe_pos, _quat, _ok = (
            probe.controller._compute_racket_pose_quat_base(safe))

        self.assertLessEqual(safe_pos[1], 1.00001)
        self.assertGreater(safe_pos[0], 0.0)
        self.assertLessEqual(safe_pos[0], 0.1)
        self.assertEqual(
            probe.workspace_soft_filter["projection"], "joint_velocity_qp")

    def test_boundary_hold_repeats_exact_last_command_for_outward_target(self):
        probe = self._probe_with_linear_fk(initial_q0=1.0)
        hold = PROBE.np.zeros(7)
        hold[0] = 1.0
        measured = hold.copy()
        probe.ros_bridge = SimpleNamespace(
            get_joint_feedback=lambda: (
                measured.copy(), PROBE.np.zeros(7)),
        )
        target = PROBE.np.zeros(7)
        target[0] = 2.0

        first = probe._filter_command_with_workspace_hold(
            target, profile="normal")
        second = probe._filter_command_with_workspace_hold(
            target, profile="normal")

        PROBE.np.testing.assert_allclose(first, hold)
        PROBE.np.testing.assert_allclose(second, hold)
        self.assertTrue(probe.workspace_hold_active)
        self.assertEqual(
            probe.workspace_soft_filter["model"], "boundary_hold")
        self.assertTrue(probe.workspace_soft_filter["intervened"])

    def test_boundary_hold_engagement_has_no_position_command_jump(self):
        probe = self._probe_with_linear_fk(initial_q0=1.0)
        previous = probe.safety_limiter.last_cmd.copy()
        measured = previous.copy()
        probe.ros_bridge = SimpleNamespace(
            get_joint_feedback=lambda: (
                measured.copy(), PROBE.np.zeros(7)),
        )
        target = PROBE.np.zeros(7)
        target[0] = 2.0

        safe = probe._filter_command_with_workspace_hold(
            target, profile="normal")

        PROBE.np.testing.assert_allclose(safe, previous)
        PROBE.np.testing.assert_allclose(
            probe.safety_limiter.last_cmd, previous)
        PROBE.np.testing.assert_allclose(
            probe.safety_limiter.last_vel, PROBE.np.zeros(7))

    def test_boundary_hold_releases_inward_candidate_on_first_tick(self):
        probe = self._probe_with_linear_fk(initial_q0=0.98)
        hold = PROBE.np.zeros(7)
        hold[0] = 0.98
        measured = hold.copy()
        probe.workspace_hold_active = True
        probe.workspace_hold_q = hold.copy()
        probe.workspace_hold_trigger_reason = "command:ee_x_high"
        probe.workspace_hold_constraint = "ee_x_high"
        probe.ros_bridge = SimpleNamespace(
            get_joint_feedback=lambda: (
                measured.copy(), PROBE.np.zeros(7)),
        )
        target = PROBE.np.zeros(7)
        target[0] = 0.5

        resumed = probe._filter_command_with_workspace_hold(
            target, profile="normal")

        self.assertFalse(probe.workspace_hold_active)
        self.assertTrue(probe.workspace_soft_filter["released"])
        self.assertEqual(
            probe.workspace_soft_filter["releaseReason"],
            "candidate_fk_moves_inward",
        )
        max_first_step = (
            probe.safety_limiter.acc_limit_rad_s2[0]
            * probe.safety_limiter.dt ** 2)
        self.assertLessEqual(hold[0] - resumed[0], max_first_step + 1e-12)
        self.assertLess(resumed[0], hold[0])

    def test_measured_outside_passes_safe_fk_candidate_without_latching(self):
        probe = self._probe_with_linear_fk(initial_q0=0.98)
        measured = PROBE.np.zeros(7)
        measured[0] = 1.01
        probe.ros_bridge = SimpleNamespace(
            get_joint_feedback=lambda: (
                measured.copy(), PROBE.np.zeros(7)),
        )
        # Although this target is outward relative to the previous command,
        # its acceleration-limited FK target remains inside and therefore
        # commands the currently-outside real arm back toward the workspace.
        target = PROBE.np.zeros(7)
        target[0] = 0.99

        safe = probe._filter_command_with_workspace_hold(
            target, profile="normal")

        self.assertGreater(safe[0], 0.98)
        self.assertLessEqual(safe[0], 1.0)
        self.assertFalse(probe.workspace_hold_active)
        self.assertFalse(probe.workspace_soft_filter["intervened"])
        self.assertTrue(probe.workspace_soft_filter["released"])

    def test_held_outward_candidate_keeps_exact_fixed_target(self):
        probe = self._probe_with_linear_fk(initial_q0=1.0)
        hold = probe.safety_limiter.last_cmd.copy()
        probe.workspace_hold_active = True
        probe.workspace_hold_q = hold.copy()
        probe.workspace_hold_trigger_reason = "measured:ee_x_high"
        probe.workspace_hold_constraint = "ee_x_high"
        probe.ros_bridge = SimpleNamespace(
            get_joint_feedback=lambda: (
                hold.copy(), PROBE.np.zeros(7)),
        )
        target = hold.copy()
        target[0] = 2.0

        safe = probe._filter_command_with_workspace_hold(
            target, profile="normal")

        PROBE.np.testing.assert_allclose(safe, hold)
        self.assertTrue(probe.workspace_hold_active)
        self.assertTrue(probe.workspace_soft_filter["intervened"])

    def test_predictive_hold_uses_fk_direction_not_safe_endpoint_alone(self):
        probe = self._probe_with_linear_fk(initial_q0=0.5)
        measured = probe.safety_limiter.last_cmd.copy()
        probe.workspace_soft_prediction_active = True
        probe.done_monitor = SimpleNamespace(last_prediction={
            "condition": "ee_predicted_x_out_of_workspace",
            "value": [1.2, 0.0, 0.0],
        })
        probe.ros_bridge = SimpleNamespace(
            get_joint_feedback=lambda: (
                measured.copy(), PROBE.np.zeros(7)),
        )
        outward = measured.copy()
        outward[0] = 0.8

        first = probe._filter_command_with_workspace_hold(
            outward, profile="normal", predictive=True)
        second = probe._filter_command_with_workspace_hold(
            outward, profile="normal", predictive=True)

        PROBE.np.testing.assert_allclose(first, measured)
        PROBE.np.testing.assert_allclose(second, measured)
        self.assertTrue(probe.workspace_hold_active)
        self.assertEqual(probe.workspace_hold_constraint, "ee_x_high")

        inward = measured.copy()
        inward[0] = 0.2
        released = probe._filter_command_with_workspace_hold(
            inward, profile="normal", predictive=True)

        self.assertLess(released[0], measured[0])
        self.assertFalse(probe.workspace_hold_active)

    def test_virtual_impedance_does_not_move_outward_at_boundary(self):
        probe = self._probe_with_linear_fk(initial_q0=1.0)
        target = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        target[0] = 2.0

        safe = probe._filter_command_with_workspace_soft_field(
            target, profile="normal")

        # With a zero previous velocity the acceleration-feasible response can
        # be a local hold; the soft spring must not manufacture a discontinuous
        # inward jump merely to satisfy its comfort preference.
        self.assertLessEqual(safe[0], 1.0)
        self.assertTrue(probe.workspace_soft_filter["active"])
        self.assertEqual(
            probe.workspace_soft_filter["model"], "virtual_impedance_cbf")
        self.assertLessEqual(
            probe.workspace_soft_filter["selectedTaskVelocityMps"][0], 0.0)

    def test_virtual_impedance_keeps_tangential_joint_motion(self):
        probe = self._probe_with_linear_fk(initial_q0=1.0)
        target = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        target[0] = 2.0
        target[1] = 0.1

        safe = probe._filter_command_with_workspace_soft_field(
            target, profile="normal")

        self.assertLessEqual(safe[0], 1.0)
        self.assertGreater(safe[1], 0.0)

    def test_virtual_impedance_is_inactive_outside_soft_margin(self):
        probe = self._probe_with_linear_fk(initial_q0=0.95)
        probe.cfg.workspace_soft_margin_m = 0.03
        target = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        target[0] = 0.96

        probe._filter_command_with_workspace_soft_field(
            target, profile="normal")

        self.assertFalse(probe.workspace_soft_filter["active"])

    def test_virtual_impedance_uses_real_joint_feedback_at_boundary(self):
        probe = self._probe_with_linear_fk(initial_q0=0.95)
        probe.cfg.workspace_soft_margin_m = 0.03
        feedback_q = PROBE.np.zeros(7)
        feedback_q[0] = 1.0
        feedback_dq = PROBE.np.zeros(7)
        feedback_dq[0] = 0.2
        probe.ros_bridge = SimpleNamespace(
            get_joint_feedback=lambda: (
                feedback_q.copy(), feedback_dq.copy()),
        )
        target = probe.safety_limiter.last_cmd.copy()

        safe = probe._filter_command_with_workspace_soft_field(
            target, profile="normal")

        self.assertTrue(probe.workspace_soft_filter["active"])
        self.assertTrue(probe.workspace_soft_filter["feedbackUsed"])
        self.assertLess(safe[0], 0.95)

    def test_virtual_impedance_preserves_redundant_joint_motion(self):
        def run(nullspace_damping):
            probe = self._probe_with_linear_fk(initial_q0=1.0)
            probe.cfg.workspace_soft_margin_m = 0.03
            probe.cfg.workspace_soft_nullspace_damping = nullspace_damping
            probe.cfg.workspace_soft_posture_gain_s = 2.0
            target = PROBE.np.zeros(7, dtype=PROBE.np.float64)
            target[0] = 2.0
            target[1] = 0.15
            for _ in range(200):
                safe = probe._filter_command_with_workspace_soft_field(
                    target, profile="normal")
                if probe.workspace_soft_filter.get("active", False):
                    probe._apply_workspace_soft_anti_windup(safe)
            return probe.safety_limiter.last_cmd.copy()

        undamped = run(0.0)
        legacy_damped_setting = run(0.85)

        # Nullspace damping was the source of unrelated large joint motion.
        # The new normal-only filter deliberately ignores that legacy knob.
        self.assertAlmostEqual(undamped[1], 0.15, places=6)
        self.assertAlmostEqual(legacy_damped_setting[1], 0.15, places=6)

    def test_virtual_impedance_settles_near_boundary_without_release_cycle(self):
        probe = self._probe_with_linear_fk(initial_q0=0.90)
        probe.cfg.workspace_soft_margin_m = 0.05
        target = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        target[0] = 2.0
        commands = [probe.safety_limiter.last_cmd.copy()]

        for _ in range(800):
            safe = probe._filter_command_with_workspace_soft_field(
                target, profile="normal")
            commands.append(safe.copy())
            if probe.workspace_soft_filter.get("active", False):
                probe._apply_workspace_soft_anti_windup(safe)

        commands = PROBE.np.asarray(commands)
        velocity = PROBE.np.diff(commands, axis=0) / 0.005
        acceleration = PROBE.np.diff(velocity, axis=0) / 0.005
        self.assertLessEqual(PROBE.np.max(commands[:, 0]), 1.01)
        self.assertLess(commands[-1, 0], 1.0)
        # Small boundary oscillation is acceptable; large release cycles are
        # not.  Keep the steady-state band below 2 mm in this 1 m/rad model.
        self.assertLess(PROBE.np.ptp(commands[-200:, 0]), 2e-3)
        self.assertLessEqual(
            PROBE.np.max(PROBE.np.abs(velocity[:, 0])),
            probe.safety_limiter.vel_limit_rad_s[0] + 1e-9,
        )
        self.assertLessEqual(
            PROBE.np.max(PROBE.np.abs(acceleration[:, 0])),
            probe.safety_limiter.acc_limit_rad_s2[0] + 1e-9,
        )

    def test_impedance_spring_does_not_accumulate_into_zero_action_policy(self):
        probe = self._probe_with_linear_fk(initial_q0=0.98)
        probe.cfg.workspace_soft_margin_m = 0.05
        nominal = PROBE.np.zeros(7, dtype=PROBE.np.float32)
        nominal[0] = 0.98
        probe.controller.arm_cmd_q = nominal.copy()
        probe.controller.arm_cmd_qvel = PROBE.np.zeros(7, dtype=PROBE.np.float32)
        probe.controller.last_cmd_q = nominal.copy()
        probe.controller.last_cmd_qvel = PROBE.np.zeros(7, dtype=PROBE.np.float32)
        probe.controller._cmd_history = []
        published = []

        for _ in range(2000):
            probe.controller._cmd_history.append(
                probe.controller.arm_cmd_q.copy())
            safe = probe._filter_command_with_workspace_soft_field(
                probe.controller.arm_cmd_q.copy(), profile="normal")
            if probe.workspace_soft_filter.get("intervened", False):
                probe._apply_workspace_soft_anti_windup(
                    safe, nominal_q=probe.controller.arm_cmd_q.copy())
            published.append(float(safe[0]))

        self.assertAlmostEqual(float(probe.controller.arm_cmd_q[0]), 0.98)
        self.assertAlmostEqual(float(probe.controller.arm_cmd_qvel[0]), 0.0)
        # The comfort spring may create a small local actuator-reference
        # offset, but may not drive the policy toward the opposite wall.
        self.assertLess(PROBE.np.ptp(published), 0.002)

    def test_stale_joint_feedback_smoothly_brakes_without_policy_command(self):
        probe = self._probe_with_linear_fk(initial_q0=0.90)
        probe.safety_limiter.last_vel[0] = 0.5
        feedback_q = probe.safety_limiter.last_cmd.copy()
        probe.ros_bridge = SimpleNamespace(
            get_joint_feedback_with_age=lambda: (
                feedback_q.copy(), PROBE.np.zeros(7), 0.5, "joint_state"),
        )
        target = feedback_q.copy()
        target[0] = 2.0

        safe = probe._filter_command_with_workspace_guard(
            target, profile="normal", apply_joint_limits=True)

        max_delta_velocity = (
            probe.safety_limiter.acc_limit_rad_s2[0]
            * probe.safety_limiter.dt)
        self.assertEqual(
            probe.workspace_soft_filter["error"], "joint_feedback_stale")
        self.assertTrue(probe.workspace_soft_filter["emergency"])
        self.assertLess(probe.safety_limiter.last_vel[0], 0.5)
        self.assertGreaterEqual(
            probe.safety_limiter.last_vel[0],
            0.5 - max_delta_velocity - 1e-10,
        )
        self.assertGreaterEqual(safe[0], feedback_q[0])

    def test_runtime_metric_preserves_coupled_tangent_and_racket_angle(self):
        probe = self._probe_with_linear_fk(initial_q0=1.0)

        def coupled_fk(q):
            angle = float(q[1] - q[2])
            return (
                PROBE.np.array([q[0] + q[1], q[1] + q[2], 0.0]),
                PROBE.np.array([
                    0.0, 0.0,
                    PROBE.math.sin(0.5 * angle),
                    PROBE.math.cos(0.5 * angle),
                ]),
                True,
            )

        probe.controller._compute_racket_pose_quat_base = coupled_fk
        target = PROBE.np.zeros(7)
        target[0] = 1.01

        probe._filter_command_with_workspace_guard(
            target, profile="normal", apply_joint_limits=False)

        selected_task = probe.workspace_soft_filter[
            "selectedTaskVelocityMps"]
        selected_angular = probe.workspace_soft_filter[
            "selectedAngularVelocityRadS"]
        self.assertLess(abs(selected_task[1]), 1e-4)
        self.assertLess(abs(selected_angular[2]), 1e-3)

    def test_position_soft_rate_limits_recovery_from_outside(self):
        probe = self._probe_with_linear_fk(initial_q0=1.2)
        target = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        target[0] = 0.5

        safe = probe._filter_command_with_workspace_guard(
            target, profile="normal", apply_joint_limits=True)

        max_first_step = (
            probe.safety_limiter.acc_limit_rad_s2[0]
            * probe.safety_limiter.dt ** 2)
        self.assertLess(1.2 - safe[0], max_first_step + 1e-12)
        self.assertGreater(safe[0], 1.0)
        previous = PROBE.np.zeros(7)
        previous[0] = 1.2
        self.assertLess(
            probe._parking_workspace_violation_score(safe),
            probe._parking_workspace_violation_score(previous),
        )

    def test_position_soft_intervention_respects_joint_acceleration(self):
        probe = self._probe_with_linear_fk(initial_q0=0.995)
        previous_velocity = 0.5
        probe.safety_limiter.last_vel[0] = previous_velocity
        target = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        target[0] = 2.0

        probe._filter_command_with_workspace_guard(
            target, profile="normal", apply_joint_limits=True)

        selected_velocity = probe.safety_limiter.last_vel[0]
        max_delta_velocity = (
            probe.safety_limiter.acc_limit_rad_s2[0]
            * probe.safety_limiter.dt)
        self.assertLessEqual(
            abs(selected_velocity - previous_velocity),
            max_delta_velocity + 1e-10,
        )
        # Relax an unreachable instantaneous barrier exactly to maximum
        # reachable braking; never bypass acceleration via an FK hard clip.
        self.assertTrue(probe.workspace_soft_filter["qp"]["feasible"])
        self.assertTrue(
            probe.workspace_soft_filter["boundRelaxedForReachability"])
        self.assertLess(selected_velocity, previous_velocity)

    def test_infeasible_corner_publishes_local_brake_not_solver_iterate(self):
        probe = self._probe_with_linear_fk(initial_q0=0.5)
        previous = probe.safety_limiter.last_cmd.copy()
        probe._workspace_soft_boundary_terms = lambda _position: [
            ("contradictory_high", PROBE.np.array([1.0, 0.0, 0.0]), -0.02),
            ("contradictory_low", PROBE.np.array([-1.0, 0.0, 0.0]), -0.02),
        ]

        safe = probe._filter_command_with_workspace_guard(
            previous.copy(), profile="normal", apply_joint_limits=True)

        PROBE.np.testing.assert_allclose(safe, previous, atol=1e-12)
        PROBE.np.testing.assert_allclose(
            probe.safety_limiter.last_vel, PROBE.np.zeros(7), atol=1e-12)
        qp = probe.workspace_soft_filter["qp"]
        self.assertFalse(qp["feasible"])
        self.assertEqual(
            qp["publishedFallback"], "reachable_zero_velocity_brake")
        self.assertTrue(probe.workspace_soft_filter["emergency"])

    def test_position_soft_boundary_release_has_no_command_jump(self):
        probe = self._probe_with_linear_fk(initial_q0=0.95)
        outward = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        outward[0] = 2.0
        outward[1] = 0.1
        commands = [probe.safety_limiter.last_cmd.copy()]

        for _ in range(300):
            commands.append(probe._filter_command_with_workspace_guard(
                outward, profile="normal", apply_joint_limits=True))

        inward = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        inward[0] = 0.5
        for _ in range(80):
            commands.append(probe._filter_command_with_workspace_guard(
                inward, profile="normal", apply_joint_limits=True))

        commands = PROBE.np.asarray(commands)
        per_tick_delta = PROBE.np.abs(PROBE.np.diff(commands, axis=0))
        max_allowed_delta = (
            probe.safety_limiter.vel_limit_rad_s
            * probe.safety_limiter.dt)
        self.assertTrue(PROBE.np.all(
            per_tick_delta <= max_allowed_delta + 1e-12))
        self.assertLessEqual(PROBE.np.max(commands[:, 0]), 1.00001)
        self.assertLess(commands[-1, 0], commands[300, 0])

    def test_workspace_anti_windup_excludes_compensation_bias(self):
        probe = self._probe_with_linear_fk()
        nominal = PROBE.np.zeros(7, dtype=PROBE.np.float32)
        policy_vel = PROBE.np.full(7, 0.1, dtype=PROBE.np.float32)
        history_before = [nominal.copy(), nominal.copy() + 0.01]
        probe.controller.arm_cmd_q = nominal.copy()
        probe.controller.arm_cmd_qvel = policy_vel.copy()
        probe.controller.last_cmd_q = nominal.copy()
        probe.controller.last_cmd_qvel = policy_vel.copy()
        probe.controller.last_cmd_qacc = PROBE.np.full(7, 0.3)
        probe.controller._cmd_history = [item.copy() for item in history_before]
        compensated_candidate = nominal.astype(PROBE.np.float64) + 0.05
        candidate_vel = PROBE.np.full(7, 0.2)
        position_correction = PROBE.np.linspace(-0.01, 0.01, 7)
        velocity_correction = PROBE.np.linspace(-0.05, 0.05, 7)
        safe = compensated_candidate + position_correction
        probe._workspace_soft_candidate_q = compensated_candidate.copy()
        probe._workspace_soft_candidate_vel = candidate_vel.copy()
        probe._workspace_antiwindup_position_correction = (
            position_correction.copy())
        probe._workspace_antiwindup_policy_velocity = (
            policy_vel + velocity_correction)
        probe.safety_limiter.last_vel = candidate_vel + velocity_correction
        probe.workspace_soft_filter = {"active": True}

        probe._apply_workspace_soft_anti_windup(
            safe, nominal_q=nominal)

        PROBE.np.testing.assert_allclose(
            probe.controller.arm_cmd_q, nominal + position_correction)
        self.assertFalse(PROBE.np.allclose(probe.controller.arm_cmd_q, safe))
        PROBE.np.testing.assert_allclose(
            probe.controller.arm_cmd_qvel,
            policy_vel + velocity_correction,
        )
        PROBE.np.testing.assert_allclose(
            probe.controller._cmd_history[0], history_before[0])
        PROBE.np.testing.assert_allclose(
            probe.controller._cmd_history[-1],
            history_before[-1] + position_correction,
        )
        self.assertTrue(probe.workspace_soft_filter["antiWindup"])
        self.assertEqual(
            probe.workspace_soft_filter["antiWindupModel"],
            "outward_rejection_only")


class BallDoneDebounceTest(unittest.TestCase):

    def test_ball_violation_requires_consecutive_ticks(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(done_ball_required_consecutive_ticks=5)
        probe.ball_done_streaks = {}
        probe.ball_done_raw_conditions = {}
        violation = {
            "ball_x_out_of_bounds": True,
            "ee_x_out_of_workspace": False,
        }

        for _ in range(4):
            filtered = probe._debounce_ball_done_conditions(violation)
            self.assertFalse(filtered["ball_x_out_of_bounds"])
        filtered = probe._debounce_ball_done_conditions(violation)

        self.assertTrue(filtered["ball_x_out_of_bounds"])
        self.assertEqual(probe.ball_done_streaks["ball_x_out_of_bounds"], 5)

    def test_in_range_measurement_resets_ball_violation_streak(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(done_ball_required_consecutive_ticks=5)
        probe.ball_done_streaks = {"ball_x_out_of_bounds": 4}
        probe.ball_done_raw_conditions = {}

        filtered = probe._debounce_ball_done_conditions({
            "ball_x_out_of_bounds": False,
        })

        self.assertFalse(filtered["ball_x_out_of_bounds"])
        self.assertEqual(probe.ball_done_streaks["ball_x_out_of_bounds"], 0)


class CommandDisableHoldTest(unittest.TestCase):

    def test_enabling_stream_first_publishes_fresh_feedback_hold(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        feedback = PROBE.np.linspace(-0.2, 0.2, 7)
        published = []
        probe.ros_bridge = SimpleNamespace(
            get_joint_feedback_with_age=lambda: (
                feedback.copy(), PROBE.np.zeros(7), 0.01, "joint_state"),
            publish_right_arm_command=lambda q: published.append(
                PROBE.np.asarray(q).copy()),
        )
        probe.safety_limiter = PROBE.RightArmCommandSafetyLimiter(
            PROBE.np.zeros(7), dt=0.005)
        probe.command_stream_active = False
        probe.compensation_was_enabled = True
        probe._reset_real_policy_for_sim_reentry = lambda: feedback.copy()
        probe._reset_compensator_to_current_arm = lambda q=None: None
        options = SimpleNamespace(publish_commands=True)

        did_publish = probe._publish_control_command(
            PROBE.np.full(7, 2.0), options)

        self.assertTrue(did_publish)
        self.assertEqual(len(published), 1)
        PROBE.np.testing.assert_allclose(published[0], feedback)
        PROBE.np.testing.assert_allclose(
            probe.safety_limiter.last_cmd, feedback)
        self.assertTrue(probe.command_stream_active)
        self.assertTrue(probe.last_command_arm["armed"])

    def test_enabling_stream_rejects_stale_feedback(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        feedback = PROBE.np.zeros(7)
        published = []
        probe.ros_bridge = SimpleNamespace(
            get_joint_feedback_with_age=lambda: (
                feedback.copy(), PROBE.np.zeros(7), 0.20, "joint_state"),
            publish_right_arm_command=lambda q: published.append(
                PROBE.np.asarray(q).copy()),
        )
        probe.command_stream_active = False
        probe.compensation_was_enabled = True
        options = SimpleNamespace(publish_commands=True)

        did_publish = probe._publish_control_command(
            PROBE.np.ones(7), options)

        self.assertFalse(did_publish)
        self.assertEqual(published, [])
        self.assertFalse(probe.command_stream_active)
        self.assertFalse(probe.last_command_arm["armed"])

    def test_disabling_active_stream_publishes_feedback_hold_once(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        feedback = PROBE.np.linspace(0.0, 0.3, 7)
        published = []
        probe.ros_bridge = SimpleNamespace(
            get_joint_feedback=lambda: (feedback.copy(), PROBE.np.zeros(7)),
            publish_right_arm_command=lambda q: published.append(
                PROBE.np.asarray(q).copy()),
        )
        probe.safety_limiter = PROBE.RightArmCommandSafetyLimiter(
            PROBE.np.zeros(7), dt=0.005)
        probe.command_stream_active = True
        probe.last_command_disable_hold = {}
        probe._reset_compensator_to_current_arm = lambda q=None: None

        first = probe._publish_hold_on_command_disable()
        second = probe._publish_hold_on_command_disable()

        self.assertTrue(first)
        self.assertFalse(second)
        self.assertEqual(len(published), 1)
        PROBE.np.testing.assert_allclose(published[0], feedback)
        self.assertFalse(probe.command_stream_active)
        self.assertTrue(
            probe.last_command_disable_hold["feedbackAvailable"])


class ControlSubprocessWorkspaceAckTest(unittest.TestCase):

    def test_snapshot_workspace_mode_reads_applied_runtime_value(self):
        snapshot = {
            "runtime": {"workspaceSafetyMode": "position soft"},
        }

        mode = PROBE.ControlSubprocessProxy._snapshot_workspace_mode(snapshot)

        self.assertEqual(mode, "position_soft")

    def test_snapshot_workspace_mode_rejects_missing_runtime(self):
        self.assertIsNone(
            PROBE.ControlSubprocessProxy._snapshot_workspace_mode({}))

    def test_snapshot_publish_commands_reads_applied_value(self):
        snapshot = {"runtime": {"publishCommands": False}}

        applied = PROBE.ControlSubprocessProxy._snapshot_publish_commands(
            snapshot)

        self.assertFalse(applied)


if __name__ == "__main__":
    unittest.main()
