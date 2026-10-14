#!/usr/bin/env python3

import importlib.util
import csv
import json
import queue
import sys
import tempfile
import threading
import unittest
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


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


class PolicyObservationAgeContractTest(unittest.TestCase):

    def test_simcode4_never_ages_held_valid_ball_observation(self):
        self.assertFalse(PROBE._policy_uses_valid_ball_obs_age(
            PROBE.OBSERVATION_PROFILE_SIMCODE4_SERVO67, True))

    def test_simcode3_keeps_valid_ball_observation_age(self):
        self.assertTrue(PROBE._policy_uses_valid_ball_obs_age(
            PROBE.OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67, False))

    def test_simcode6_keeps_held_valid_ball_observation_age_at_zero(self):
        self.assertFalse(PROBE._policy_uses_valid_ball_obs_age(
            PROBE.OBSERVATION_PROFILE_SIMCODE6_QVEL67, True))


class CameraImpactRacketFrameContractTest(unittest.TestCase):

    class _Core:
        def __init__(self):
            self.racket_updates = []
            self.clear_count = 0

        def set_racket_state(self, position, timestamp):
            self.racket_updates.append((
                PROBE.np.asarray(position, dtype=PROBE.np.float64),
                float(timestamp),
            ))

        def clear_racket_state(self):
            self.clear_count += 1

    @staticmethod
    def _node(output_frame):
        node = PROBE.RosBridgeNode.__new__(PROBE.RosBridgeNode)
        node.cfg = SimpleNamespace(
            sim_cartesian_x_offset_m=0.0,
            sim_cartesian_y_offset_m=0.023583,
            sim_base_height_m=1.302396,
        )
        node.camera_lock = threading.Lock()
        node.camera_output_frame = output_frame
        node.camera_detector = SimpleNamespace(
            detector=CameraImpactRacketFrameContractTest._Core())
        node.latest_policy_viz_state = {}
        node.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(
            nanoseconds=int(PROBE.time.time()*1e9)))
        return node

    def test_base_link_detector_receives_inverse_policy_frame_translation(self):
        node = self._node("base_link")
        racket_sim = PROBE.np.array(
            [-0.114089, -0.466326, 1.048674], dtype=PROBE.np.float32)

        with mock.patch.object(PROBE.time, "time", return_value=123.0):
            node.set_policy_viz_state(
                PROBE.np.zeros(3), PROBE.np.zeros(3), True,
                PROBE.np.zeros(3), racket_sim, True)

        core = node.camera_detector.detector
        self.assertEqual(len(core.racket_updates), 1)
        position, timestamp = core.racket_updates[0]
        PROBE.np.testing.assert_allclose(
            position,
            [-0.114089, -0.489909, -0.253722],
            rtol=0.0,
            atol=2.0e-7,
        )
        self.assertAlmostEqual(timestamp, 123.0, places=9)
        # Visualization remains in the policy/sim frame.
        PROBE.np.testing.assert_allclose(
            node.latest_policy_viz_state["racket_pos_m"], racket_sim)
        # Regression fixture from fixed-ball recording 20260812_165718: the
        # old mixed-frame path reported about 1269 mm instead of 32 mm.
        ball_sim = PROBE.np.array(
            [-0.108706, -0.464246, 1.080593], dtype=PROBE.np.float64)
        frame_offset = PROBE.np.array(
            [0.0, 0.023583, 1.302396], dtype=PROBE.np.float64)
        distance_mm = 1000.0 * PROBE.np.linalg.norm(
            (ball_sim - frame_offset) - position)
        self.assertAlmostEqual(distance_mm, 32.436, delta=0.01)

    def test_unrelated_camera_frame_clears_incomparable_racket_context(self):
        node = self._node("camera")

        node.set_policy_viz_state(
            PROBE.np.zeros(3), PROBE.np.zeros(3), True,
            PROBE.np.zeros(3), PROBE.np.ones(3), True)

        core = node.camera_detector.detector
        self.assertEqual(core.racket_updates, [])
        self.assertEqual(core.clear_count, 1)


class SensorDepthAnomalyPredictionContractTest(unittest.TestCase):

    @staticmethod
    def _options(*, takeover=True):
        return SimpleNamespace(
            high_ball_occlusion_prediction=False,
            sensor_depth_anomaly_prediction=bool(takeover),
            ball_source="camera",
            use_ball_estimator=False,
            high_ball_prediction_history_frames=10,
            high_ball_prediction_activation_z_m=1.42,
            high_ball_prediction_min_upward_velocity_m_s=0.20,
            high_ball_prediction_max_prediction_s=0.45,
            high_ball_prediction_stop_z_m=1.24,
            high_ball_prediction_drag_coefficient_per_m=0.13545,
            high_ball_prediction_integration_step_s=0.005,
            high_ball_prediction_reacquisition_gate_m=0.18,
            high_ball_reacquisition_velocity_handover=False,
            high_ball_reacquisition_velocity_frames=3,
            high_ball_reacquisition_velocity_blend_s=0.04,
        )

    @staticmethod
    def _probe():
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.high_ball_occlusion_predictor = None
        probe.high_ball_occlusion_predictor_key = None
        probe.now_s = 0.0
        probe.cfg = SimpleNamespace(control_dt=0.005)
        probe.ros_bridge = None
        return probe

    def test_independent_switch_routes_adaptive_depth_fault_to_3d_model(self):
        probe = self._probe()
        options = self._options(takeover=True)
        good_measurement = {
            "depth_mode": "adaptive",
            "depth_sensor_rejected": False,
        }
        for index in range(10):
            stamp = 0.01 * index
            probe.now_s = stamp
            probe._resolve_high_ball_occlusion_state(
                PROBE.np.array([0.1 + 0.2 * stamp, -0.4, 1.0]),
                PROBE.np.array([0.2, 0.0, 0.0]),
                True,
                stamp,
                options,
                good_measurement,
            )

        probe.now_s = 0.10
        position, velocity, valid, _stamp, info = (
            probe._resolve_high_ball_occlusion_state(
                PROBE.np.array([0.0, 0.0, 2.0]),
                PROBE.np.array([0.0, 0.0, 0.0]),
                True,
                0.10,
                options,
                {
                    "depth_mode": "adaptive",
                    "depth_sensor_rejected": True,
                },
            ))

        self.assertTrue(valid)
        self.assertTrue(info["predicted"])
        self.assertTrue(info["controlApplied"])
        self.assertTrue(info["sensorDepthAnomalyDetected"])
        self.assertEqual(info["activeTrigger"], "sensor_depth_anomaly")
        self.assertLess(float(position[2]), 1.24)
        self.assertLess(float(PROBE.np.linalg.norm(velocity)), 5.0)

    def test_disabled_depth_fault_switch_never_promotes_rejected_fallback(self):
        probe = self._probe()
        fallback = PROBE.np.array([0.2, -0.3, 0.9])

        position, _velocity, valid, _stamp, info = (
            probe._resolve_high_ball_occlusion_state(
                fallback,
                PROBE.np.zeros(3),
                True,
                0.10,
                self._options(takeover=False),
                {
                    "depth_mode": "adaptive",
                    "depth_sensor_rejected": True,
                },
            ))

        self.assertFalse(valid)
        self.assertFalse(info["predicted"])
        self.assertEqual(info["reason"], "disabled")
        self.assertEqual(info["estimatorMode"], "depth_fault_invalid")
        PROBE.np.testing.assert_array_equal(position, fallback)

    def test_high_fov_reacquisition_waits_for_camera_velocity_warmup(self):
        probe = self._probe()
        options = self._options(takeover=False)
        options.high_ball_occlusion_prediction = True
        good_measurement = {
            "depth_mode": "adaptive",
            "depth_sensor_rejected": False,
            "velocity_ready": True,
        }
        for index in range(10):
            stamp = 0.01 * index
            position = PROBE.np.array([
                0.1 + 0.1 * stamp,
                -0.4 + 0.2 * stamp,
                1.45 + 2.0 * stamp - 4.9 * stamp * stamp,
            ])
            velocity = PROBE.np.array([
                0.1, 0.2, 2.0 - 9.8 * stamp,
            ])
            probe.now_s = stamp
            probe._resolve_high_ball_occlusion_state(
                position, velocity, True, stamp, options, good_measurement)

        probe.now_s = 0.10
        _position, _velocity, valid, _stamp, lost = (
            probe._resolve_high_ball_occlusion_state(
                PROBE.np.zeros(3), PROBE.np.zeros(3), False,
                0.10, options, None))
        self.assertTrue(valid)
        self.assertTrue(lost["predicted"])

        probe.now_s = 0.11
        position, velocity, valid, _stamp, warming = (
            probe._resolve_high_ball_occlusion_state(
                PROBE.np.array([0.111, -0.378, 1.61]),
                PROBE.np.zeros(3), True, 0.11, options,
                {
                    "depth_mode": "adaptive",
                    "depth_sensor_rejected": False,
                    "velocity_ready": False,
                }))

        self.assertTrue(valid)
        self.assertTrue(warming["predicted"])
        self.assertTrue(warming["reacquisitionVelocityWarmup"])
        self.assertFalse(warming["cameraVelocityReady"])
        self.assertNotEqual(float(PROBE.np.linalg.norm(velocity)), 0.0)

    def test_ir_top_edge_motion_outlier_starts_prediction_below_world_z_gate(self):
        probe = self._probe()
        options = self._options(takeover=False)
        options.high_ball_occlusion_prediction = True
        options.optimized_ball_prediction_initial_state = True
        for index in range(10):
            stamp = 0.01 * index
            position = PROBE.np.array([
                -0.08 + 0.1 * stamp,
                -0.45 + 0.1 * stamp,
                1.0 + 1.2 * stamp - 4.9 * stamp * stamp,
            ])
            velocity = PROBE.np.array([
                0.1, 0.1, 1.2 - 9.8 * stamp,
            ])
            probe.now_s = stamp
            probe._resolve_high_ball_occlusion_state(
                position, velocity, True, stamp, options, {
                    "cx": 548.0,
                    "cy": 260.0 - 10.0 * index,
                    "measurement_trusted_3d": True,
                    "observation_state": "tracking",
                    "velocity_ready": True,
                    "synchronized_depth_enabled": False,
                })

        probe.now_s = 0.10
        _position, _velocity, valid, _stamp, info = (
            probe._resolve_high_ball_occlusion_state(
                PROBE.np.zeros(3), PROBE.np.zeros(3), False,
                0.10, options, {
                    "cx": 557.0,
                    "cy": 84.0,
                    "measurement_trusted_3d": False,
                    "observation_state": "motion_outlier",
                    "velocity_ready": False,
                    "synchronized_depth_enabled": False,
                }))

        self.assertTrue(valid)
        self.assertTrue(info["predicted"])
        self.assertTrue(info["topFovExitDetected"])
        self.assertEqual(info["topFovExitReason"], "rejected_2d_centre_at_top")
        self.assertEqual(info["activeTrigger"], "high_fov_edge_exit")
        self.assertEqual(info["estimatorMode"], "high_fov_prediction")

        # The 20260812_152353 leave-and-return interval lasts about 0.40 s.
        # Once explicitly activated, loss frames do not fall back to stale
        # policy sampling merely because the predicted world Z crosses 1.24 m.
        probe.now_s = 0.49
        _p, _v, still_valid, _stamp, later = (
            probe._resolve_high_ball_occlusion_state(
                PROBE.np.zeros(3), PROBE.np.zeros(3), False,
                0.49, options, {
                    "measurement_trusted_3d": False,
                    "observation_state": "detection_loss",
                    "velocity_ready": False,
                    "synchronized_depth_enabled": False,
                }))
        self.assertTrue(still_valid)
        self.assertTrue(later["predicted"])

    def test_confirmed_detector_impact_clears_predictor_preimpact_history(self):
        probe = self._probe()
        options = self._options(takeover=False)
        options.high_ball_occlusion_prediction = True
        measurement = {
            "cx": 540.0,
            "cy": 220.0,
            "measurement_trusted_3d": True,
            "observation_state": "tracking",
            "velocity_ready": True,
            "velocity_impact_detected": False,
            "synchronized_depth_enabled": False,
        }
        for index in range(6):
            stamp = 0.01 * index
            probe.now_s = stamp
            probe._resolve_high_ball_occlusion_state(
                PROBE.np.array([0.0, -0.4, 1.2 - stamp]),
                PROBE.np.array([0.0, 0.0, -1.0]),
                True, stamp, options, measurement)

        impact = dict(measurement)
        impact["velocity_impact_detected"] = True
        probe.now_s = 0.06
        _p, _v, valid, _stamp, info = (
            probe._resolve_high_ball_occlusion_state(
                PROBE.np.array([0.0, -0.4, 1.16]),
                PROBE.np.array([0.0, 0.0, 1.5]),
                True, 0.06, options, impact))

        self.assertTrue(valid)
        self.assertTrue(info["historyResetOnImpact"])
        self.assertEqual(info["historySamples"], 1)

    def test_recorded_152353_path_predicts_from_top_exit_until_return(self):
        probe = self._probe()
        options = self._options(takeover=True)
        options.high_ball_occlusion_prediction = True
        options.optimized_ball_prediction_initial_state = True
        options.camera_reacquisition_smoothing = True
        origin = 39.4612312
        rows = [
            (39.4612312, [-0.081009, -0.444696, 1.182889], 261.8, False),
            (39.4723442, [-0.079117, -0.444166, 1.165908], 273.1, False),
            (39.4833230, [-0.078948, -0.445483, 1.147108], 283.2, False),
            (39.4943726, [-0.078799, -0.445874, 1.127360], 294.5, False),
            (39.5056580, [-0.079064, -0.451617, 1.117477], 290.4, False),
            (39.5165827, [-0.082463, -0.466667, 1.133185], 270.8, False),
            # New detector confirms the two-frame reversal here.  The
            # predictor must start a fresh launch segment at this sample.
            (39.5278847, [-0.081214, -0.468479, 1.165810], 252.9, True),
            (39.5388803, [-0.079586, -0.468886, 1.196982], 237.1, False),
            (39.5500230, [-0.077274, -0.463681, 1.234086], 216.4, False),
            (39.5611744, [-0.072880, -0.456946, 1.265660], 196.3, False),
        ]
        for absolute, position, cy, impact in rows:
            stamp = absolute - origin
            probe.now_s = stamp
            probe._resolve_high_ball_occlusion_state(
                PROBE.np.asarray(position),
                PROBE.np.array([0.0, 0.0, 1.0]),
                True, stamp, options, {
                    "cx": 540.0,
                    "cy": cy,
                    "measurement_trusted_3d": True,
                    "observation_state": "tracking",
                    "velocity_ready": True,
                    "velocity_impact_detected": impact,
                    "depth_mode": "adaptive",
                    "depth_sensor_rejected": False,
                    "synchronized_depth_enabled": False,
                })

        # The late point at camera stamp 39.610 is rejected by the tightened
        # vertical motion gate, so it is diagnostic-only and cannot flatten
        # the post-impact launch fit.
        late_stamp = 39.6101046 - origin
        probe.now_s = late_stamp
        _p, _v, late_valid, _stamp, late = (
            probe._resolve_high_ball_occlusion_state(
                PROBE.np.zeros(3), PROBE.np.zeros(3), False,
                late_stamp, options, {
                    "cx": 547.9,
                    "cy": 174.3,
                    "measurement_trusted_3d": False,
                    "observation_state": "motion_outlier",
                    "velocity_ready": False,
                    "depth_mode": "adaptive",
                    "depth_sensor_rejected": False,
                    "synchronized_depth_enabled": False,
                }))
        self.assertFalse(late_valid)
        self.assertFalse(late["predicted"])

        edge_stamp = 39.6167994 - origin
        probe.now_s = edge_stamp
        _p, _v, edge_valid, _stamp, edge = (
            probe._resolve_high_ball_occlusion_state(
                PROBE.np.zeros(3), PROBE.np.zeros(3), False,
                edge_stamp, options, {
                    "cx": 557.3,
                    "cy": 83.9,
                    "measurement_trusted_3d": False,
                    "observation_state": "motion_outlier",
                    "velocity_ready": False,
                    "depth_mode": "adaptive",
                    "depth_sensor_rejected": False,
                    "synchronized_depth_enabled": False,
                }))
        self.assertTrue(edge_valid)
        self.assertTrue(edge["predicted"])
        self.assertEqual(edge["activeTrigger"], "high_fov_edge_exit")
        self.assertGreater(edge["initialVelocityMps"][2], 2.2)

        return_stamp = 40.0172696 - origin
        probe.now_s = return_stamp
        _p, _v, return_valid, _stamp, returning = (
            probe._resolve_high_ball_occlusion_state(
                PROBE.np.array([0.022311, -0.375139, 1.363672]),
                PROBE.np.zeros(3), True, return_stamp, options, {
                    "cx": 464.2,
                    "cy": 150.5,
                    "measurement_trusted_3d": True,
                    "motion_reacquisition_confirmed": True,
                    "observation_state": "reacquiring",
                    "velocity_ready": False,
                    "depth_mode": "adaptive",
                    "depth_sensor_rejected": False,
                    "synchronized_depth_enabled": False,
                }))
        self.assertTrue(return_valid)
        self.assertTrue(returning["predicted"])
        self.assertEqual(returning["estimatorMode"], "reacquiring")
        self.assertEqual(returning["state"], "reacquisition_smoothing")


class PolicyObservationScaleContractTest(unittest.TestCase):

    def test_each_observation_dimension_keeps_an_independent_scale(self):
        requested = [1.0] * 67
        requested[23] = 0.5
        requested[24] = 0.25
        scales = PROBE._policy_observation_scales(requested, 67)
        self.assertEqual(len(scales), 67)
        self.assertEqual(scales[23], 0.5)
        self.assertEqual(scales[24], 0.25)
        self.assertEqual(scales[25], 1.0)

    def test_invalid_or_missing_scales_fall_back_to_one(self):
        scales = PROBE._policy_observation_scales(
            [0.5, float("nan"), 1000.0], 4)
        self.assertEqual(scales, (0.5, 1.0, 100.0, 1.0))

    def test_launch_dq_scale_changes_exactly_obs_7_through_13(self):
        scales = PROBE._default_policy_observation_scales(67, 0.5)
        self.assertEqual(scales[:7], (1.0,) * 7)
        self.assertEqual(scales[7:14], (0.5,) * 7)
        self.assertEqual(scales[14:], (1.0,) * 53)


class PolicyDqObservationAdapterTest(unittest.TestCase):

    def _adapter(self, *, tau=0.0, delay=0, gain=1.0, enabled=True):
        config = PROBE.PolicyDqAdapterConfig(
            enabled=enabled,
            gain=(gain,) * 7,
            tau_s=(tau,) * 7,
            delay_steps=(delay,) * 7,
            dt_s=0.005,
        )
        return PROBE.PolicyDqObservationAdapter(config)

    def test_disabled_adapter_is_exact_identity(self):
        adapter = self._adapter(tau=0.02, delay=3, gain=0.5, enabled=False)
        raw = PROBE.np.linspace(-1.0, 1.0, 7, dtype=PROBE.np.float32)
        policy, info = adapter.update(raw)
        PROBE.np.testing.assert_array_equal(policy, raw)
        self.assertFalse(info["active"])

    def test_runtime_switch_can_enable_launch_disabled_adapter(self):
        adapter = self._adapter(
            tau=0.02, delay=3, gain=0.5, enabled=False)
        raw = PROBE.np.linspace(-1.0, 1.0, 7, dtype=PROBE.np.float32)

        policy, info = adapter.update(raw, enabled=True)

        PROBE.np.testing.assert_allclose(policy, 0.5 * raw, atol=1e-7)
        self.assertTrue(info["runtimeEnabled"])
        self.assertTrue(info["active"])

    def test_web_switch_edge_resets_ema_and_delay_history(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.policy_dq_adapter = self._adapter(
            tau=0.02, delay=3, gain=0.5, enabled=False)
        probe.policy_dq_adapter_runtime_enabled = False
        probe.active_policy_profile = (
            PROBE.OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67)
        probe.cfg = SimpleNamespace(
            policy_profile=PROBE.OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67)
        one = PROBE.np.ones(7, dtype=PROBE.np.float32)

        first, first_info = probe._joint_velocity_observation_for_policy(
            one, 1.0, True)
        bypass, bypass_info = probe._joint_velocity_observation_for_policy(
            3.0 * one, 1.005, False)
        restarted, restarted_info = (
            probe._joint_velocity_observation_for_policy(
                4.0 * one, 1.010, True))

        PROBE.np.testing.assert_allclose(first, 0.5, atol=1e-7)
        PROBE.np.testing.assert_array_equal(bypass, 3.0 * one)
        PROBE.np.testing.assert_allclose(restarted, 2.0, atol=1e-7)
        self.assertEqual(first_info["sampleIndex"], 0)
        self.assertFalse(bypass_info["active"])
        self.assertEqual(restarted_info["sampleIndex"], 0)

    def test_ema_gain_and_causal_delay_follow_fixed_tick_formula(self):
        adapter = self._adapter(tau=0.01, delay=2, gain=0.5)
        one = PROBE.np.ones(7, dtype=PROBE.np.float32)
        three = 3.0 * one
        out0, _ = adapter.update(one)
        out1, _ = adapter.update(three)
        out2, info2 = adapter.update(three)
        out3, _ = adapter.update(three)
        alpha = 1.0 - PROBE.math.exp(-0.005 / 0.01)
        filtered1 = 1.0 + alpha * 2.0
        PROBE.np.testing.assert_allclose(out0, 0.5, atol=1e-7)
        PROBE.np.testing.assert_allclose(out1, 0.5, atol=1e-7)
        PROBE.np.testing.assert_allclose(out2, 0.5, atol=1e-7)
        PROBE.np.testing.assert_allclose(
            out3, 0.5 * filtered1, atol=1e-6)
        self.assertEqual(info2["delaySteps"], [2] * 7)

    def test_simcode6_never_applies_the_simcode5_fitted_dq_adapter(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.policy_dq_adapter = self._adapter(
            tau=0.02, delay=3, gain=0.5, enabled=True)
        probe.policy_dq_adapter_runtime_enabled = True
        probe.active_policy_profile = PROBE.OBSERVATION_PROFILE_SIMCODE6_QVEL67
        probe.cfg = SimpleNamespace(
            policy_profile=PROBE.OBSERVATION_PROFILE_SIMCODE6_QVEL67)
        raw = PROBE.np.linspace(-1.0, 1.0, 7, dtype=PROBE.np.float32)

        policy_dq, info = probe._joint_velocity_observation_for_policy(
            raw, 1.0, True)

        PROBE.np.testing.assert_array_equal(policy_dq, raw)
        self.assertFalse(info["active"])
        self.assertEqual(
            info["reason"], "only_enabled_for_simcode5_pure_actuator67")

    def test_simcode5_profile_resolution_keeps_aligned_cpu_backend(self):
        marker = Path("/tmp/simcode5_contract_marker")
        cfg = PROBE.SimMirrorConfig(
            model_path=marker,
            policy_xml_path=marker,
            sim_xml_path=marker,
            policy_obs_dim=67,
            control_dt=0.005,
            backend="mujoco",
            simcode5_model_path=marker,
            simcode5_policy_xml_path=marker,
            simcode5_sim_xml_path=marker,
        )
        resolved = PROBE.sim_mirror_config_for_policy_profile(
            cfg, PROBE.OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67)
        self.assertEqual(resolved.backend, "mujoco")

    def test_simcode6_profile_resolution_keeps_aligned_cpu_backend(self):
        marker = Path("/tmp/simcode6_contract_marker")
        cfg = PROBE.SimMirrorConfig(
            model_path=marker,
            policy_xml_path=marker,
            sim_xml_path=marker,
            policy_obs_dim=67,
            control_dt=0.005,
            backend="mujoco",
            simcode6_model_path=marker,
            simcode6_policy_xml_path=marker,
            simcode6_sim_xml_path=marker,
        )
        resolved = PROBE.sim_mirror_config_for_policy_profile(
            cfg, PROBE.OBSERVATION_PROFILE_SIMCODE6_QVEL67)
        self.assertEqual(resolved.backend, "mujoco")

    def test_simcode5_cpu_xml_restores_checkpoint_sport_pd_profile(self):
        from pingpong_controller.simcode2_xml import (
            SPORT_TASKSPACE_FIT_RIGHT_ARM_PD,
            build_simcode2_training_xml,
        )
        repo_root = PACKAGE_ROOT.parents[1]
        xml_path = (
            repo_root / "simcode5" / "pingpong_moz1-main"
            / "pingpong_controller" / "tools" / "rl_sim" / "moz1_pd.xml")
        if not xml_path.is_file():
            self.skipTest(
                f"simcode5 XML fixture is not installed: {xml_path}")
        patched = build_simcode2_training_xml(
            xml_path, right_arm_pd_profile="sport_taskspace_fit_v1")
        try:
            root = ET.parse(patched).getroot()
            values = {
                elem.get("name"): (
                    float(elem.get("kp", "nan")),
                    float(elem.get("kv", "nan")),
                )
                for elem in root.findall(".//actuator/position")
                if elem.get("name") in SPORT_TASKSPACE_FIT_RIGHT_ARM_PD
            }
            self.assertEqual(values, SPORT_TASKSPACE_FIT_RIGHT_ARM_PD)
        finally:
            patched.unlink(missing_ok=True)


class RecordedBallStateAnalysisTest(unittest.TestCase):

    def test_explicit_ball_state_takes_priority_and_legacy_fields_fallback(self):
        explicit = PROBE._sample_ball_state_6d({
            "ballState6D": [1, 2, 3, 4, 5, 6],
            "policyBallObservation": [7, 8, 9, 10, 11, 12],
        })
        legacy = PROBE._sample_ball_state_6d({
            "observedBall": [0.1, 0.2, 0.3],
            "observedVelocity": [1.1, 1.2, 1.3],
        })
        self.assertEqual(explicit.tolist(), [1, 2, 3, 4, 5, 6])
        self.assertEqual(legacy.tolist(), [0.1, 0.2, 0.3, 1.1, 1.2, 1.3])
        self.assertIsNone(PROBE._sample_ball_state_6d({
            "policyInput": {"ball": None, "velocity": None},
        }))

    def test_writes_one_figure_for_each_ball_state_dimension(self):
        samples = [
            {
                "time": 0.0,
                "ballState6D": [0.1, 0.2, 1.0, 1.1, 1.2, -2.0],
            },
            {
                "time": 0.005,
                "observedBall": [0.11, 0.21, 0.99],
                "observedVelocity": [1.0, 1.1, -2.1],
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            analysis_dir = Path(tmp)
            (analysis_dir / "ball_position_curves.png").write_bytes(b"old")
            (analysis_dir / "ball_velocity_curves.png").write_bytes(b"old")
            outputs = PROBE._plot_ball_state_curves(samples, analysis_dir)
            self.assertEqual(
                {path.name for path in outputs},
                {
                    "ball_position_x_curve.png",
                    "ball_position_y_curve.png",
                    "ball_position_z_curve.png",
                    "ball_velocity_x_curve.png",
                    "ball_velocity_y_curve.png",
                    "ball_velocity_z_curve.png",
                },
            )
            self.assertEqual(len(outputs), 6)
            self.assertTrue(all(path.is_file() for path in outputs))
            self.assertFalse(
                (analysis_dir / "ball_position_curves.png").exists())
            self.assertFalse(
                (analysis_dir / "ball_velocity_curves.png").exists())
            discovered = PROBE._analysis_outputs_from_dir(analysis_dir)
            self.assertEqual(
                discovered["ballPositionYCurve"],
                "ball_position_y_curve.png",
            )
            self.assertEqual(
                discovered["ballVelocityYCurve"],
                "ball_velocity_y_curve.png",
            )

    def test_saved_ball_curves_use_blue_for_measurement_and_red_for_prediction(self):
        samples = [
            {
                "time": 0.005 * index,
                "ballState6D": [
                    0.1 + index * 0.01, 0.2, 1.0,
                    1.0, 0.0, -1.0,
                ],
                "predictedBallIsPredicted": index >= 2,
            }
            for index in range(4)
        ]
        import matplotlib.axes
        original_plot = matplotlib.axes.Axes.plot
        colors = []

        def recording_plot(axis, *args, **kwargs):
            colors.append(kwargs.get("color"))
            return original_plot(axis, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
                matplotlib.axes.Axes, "plot", recording_plot):
            PROBE._plot_ball_state_curves(samples, Path(tmp))

        self.assertEqual(colors.count("#1f77b4"), 6)
        self.assertEqual(colors.count("#d62728"), 6)

    def test_writes_combined_ball_racket_contact_diagnostics(self):
        samples = [
            {
                "time": 0.000,
                "sourceSeq": 10,
                "racket": [0.0, 0.0, 1.00],
                "rawBall": [0.0, 0.0, 1.08],
                "observedBall": [0.0, 0.0, 1.07],
                "observedVelocity": [0.0, 0.0, -0.20],
                "predictedBall": [0.0, 0.0, 1.06],
                "cameraMeasurement": {
                    "sensor_depth_mm": 812.0,
                    "geom_depth_mm": 798.0,
                    "depth_mm": 812.0,
                    "depth_method": "sensor",
                },
                "age": 0.005,
            },
            {
                "time": 0.005,
                "sourceSeq": 11,
                "racket": [0.0, 0.0, 1.00],
                "rawBall": [0.0, 0.0, 1.05],
                "observedBall": [0.0, 0.0, 1.04],
                "observedVelocity": [0.0, 0.0, 0.30],
                "predictedBall": [0.0, 0.0, 1.03],
                "cameraMeasurement": {
                    "sensor_depth_mm": 810.0,
                    "geom_depth_mm": 799.0,
                    "depth_mm": 810.0,
                    "depth_method": "sensor",
                },
                "estimator": {"lossS": 0.0, "extrapolationS": 0.002},
                "age": 0.006,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            analysis_dir = Path(tmp)
            plots = PROBE._plot_ball_racket_position_curves(
                samples, analysis_dir)
            report = PROBE._analyze_ball_racket_contact_gaps(
                samples, analysis_dir)
            self.assertEqual(
                [path.name for path in plots],
                ["ball_racket_position_curves.png"],
            )
            self.assertTrue(
                (analysis_dir / "ball_racket_position_curves.png").is_file())
            self.assertTrue(
                (analysis_dir / "ball_racket_contact_gaps.csv").is_file())
            self.assertTrue(
                (analysis_dir / "ball_racket_contact_gaps.json").is_file())
            self.assertTrue(
                (analysis_dir / "ball_racket_contact_gaps.txt").is_file())
            self.assertEqual(report["contactCount"], 1)
            discovered = PROBE._analysis_outputs_from_dir(analysis_dir)
            self.assertEqual(
                discovered["ballRacketPositionCurves"],
                "ball_racket_position_curves.png",
            )
            self.assertEqual(
                discovered["ballRacketContactGapsCsv"],
                "ball_racket_contact_gaps.csv",
            )

    def test_records_and_plots_ir_center_and_depth_diagnostics(self):
        measurement = PROBE._camera_measurement_payload({
            "cx": PROBE.np.float32(321.25),
            "cy": 207.5,
            "sensor_depth_mm": 812.0,
            "geom_depth_mm": 798.0,
            "depth_mm": 812.0,
            "depth_method": "sensor",
            "candidate_motion_distance_px": 2.5,
            "raw_position_ir": [1, 2, 3],
        })
        self.assertEqual(measurement["cx"], 321.25)
        self.assertEqual(measurement["sensor_depth_mm"], 812.0)
        self.assertNotIn("raw_position_ir", measurement)
        samples = [
            {"time": 0.0, "cameraMeasurement": measurement},
            {
                "time": 0.005,
                "cameraMeasurement": {
                    "cx": 322.0,
                    "cy": 208.0,
                    "sensor_depth_mm": 810.0,
                    "geom_depth_mm": 799.0,
                    "depth_mm": 810.0,
                    "depth_method": "sensor",
                },
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            analysis_dir = Path(tmp)
            outputs = PROBE._plot_camera_measurement_curves(
                samples, analysis_dir)
            self.assertEqual(
                {path.name for path in outputs},
                {
                    "camera_ir_cx_curve.png",
                    "camera_ir_cy_curve.png",
                    "camera_sensor_depth_curve.png",
                    "camera_geometry_depth_curve.png",
                },
            )
            trace_path = PROBE._write_policy_trace_csv(samples, analysis_dir)
            with trace_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]["camera_ir_cx_px"], "321.25")
            self.assertEqual(rows[0]["camera_ir_cy_px"], "207.5")
            self.assertEqual(rows[0]["camera_sensor_depth_mm"], "812.0")
            self.assertEqual(rows[0]["camera_geometry_depth_mm"], "798.0")
            self.assertEqual(rows[0]["camera_depth_method"], "sensor")

    def test_camera_state_and_diagnostics_are_selected_atomically(self):
        class Bridge:
            def camera_ball_measurement_with_diagnostics(self):
                return (
                    PROBE.np.array([0.1, -0.2, 1.0]),
                    PROBE.np.array([0.0, 0.1, -0.5]),
                    True,
                    12.5,
                    {
                        "cx": 320.0,
                        "cy": 200.0,
                        "sensor_depth_mm": 800.0,
                        "geom_depth_mm": 790.0,
                    },
                )

        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.ros_bridge = Bridge()
        selected = probe._select_ball_source(
            PROBE.np.zeros(3),
            PROBE.np.zeros(3),
            True,
            SimpleNamespace(ball_source="camera"),
            include_feedback=False,
        )
        self.assertEqual(selected[5], 12.5)
        self.assertEqual(selected[6]["sensor_depth_mm"], 800.0)
        self.assertEqual(selected[0].tolist(), [0.1, -0.2, 1.0])


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


class HybridBallTrackerProbeAdapterTest(unittest.TestCase):

    def _adapter(self, *, prediction_enabled=True):
        config_path = (
            PACKAGE_ROOT / "pingpong_controller" / "ball_tracker" /
            "config.yaml")
        adapter = PROBE.HybridBallTrackerProbeAdapter(
            config_path,
            prediction_enabled=prediction_enabled,
            frame_rotation=None,
            ballistic_dynamics_enabled=True,
        )
        adapter.set_racket_pose(
            PROBE.np.array([0.0, 0.0, 0.5]),
            PROBE.np.array([0.0, 0.0, 1.0]),
            PROBE.np.zeros(3),
        )
        return adapter

    def test_converts_probe_mm_contract_and_reports_impact(self):
        adapter = self._adapter(prediction_enabled=True)
        last_timestamp = 0.0
        for frame in range(8):
            last_timestamp = frame / 60.0
            z = (
                1.0 - 0.5 * last_timestamp
                - 0.5 * 9.81 * last_timestamp * last_timestamp)
            accepted = adapter.update(
                PROBE.np.array([0.0, 0.0, z * 1000.0]),
                last_timestamp,
            )
            self.assertTrue(accepted)

        output = adapter.predict_to(last_timestamp + 0.005)

        self.assertTrue(output["valid"])
        self.assertTrue(output["predicted"])
        self.assertEqual(len(output["acceleration"]), 3)
        self.assertEqual(len(output["state_covariance_si"]), 9)
        self.assertTrue(output["impact_prediction"]["valid"])
        self.assertGreater(
            output["impact_prediction"]["time_to_contact"], 0.0)
        self.assertEqual(adapter.diagnostics()["backend"], "hybrid_ekf9")

    def test_prediction_switch_holds_latest_measurement_state(self):
        adapter = self._adapter(prediction_enabled=False)
        adapter.update([0.0, 0.0, 1000.0], 1.0)
        adapter.update([0.0, 0.0, 990.0], 1.016)
        at_measurement = adapter.predict_to(1.016)
        held = adapter.predict_to(1.050)

        self.assertFalse(held["predicted"])
        self.assertGreater(held["age_s"], 0.0)
        self.assertEqual(held["extrapolation_s"], 0.0)
        self.assertIsNone(held["impact_prediction"])
        PROBE.np.testing.assert_allclose(
            held["position"], at_measurement["position"])


class RosJointSampleClockTest(unittest.TestCase):

    @staticmethod
    def _node():
        node = PROBE.RosBridgeNode.__new__(PROBE.RosBridgeNode)
        node.lock = threading.Lock()
        node.latest_joint_q = None
        node.latest_joint_dq = None
        node.latest_joint_stamp_wall = None
        node.latest_joint_stamp_monotonic_s = None
        node.latest_joint_sample_stamp_s = None
        node.latest_joint_sample_dt_s = None
        node.latest_joint_header_stamp_s = None
        node.latest_joint_sample_stamp_source = "unavailable"
        node.latest_joint_dq_source = "unavailable"
        node.latest_cart_pos = None
        node.latest_cart_quat_xyzw = None
        node.latest_cart_psi = None
        node.latest_cart_stamp_wall = None
        node.publish_count = 0
        return node

    @staticmethod
    def _message(stamp_s, *, q_offset=0.0, dq_offset=0.0):
        if stamp_s is None:
            sec = 0
            nanosec = 0
        else:
            sec = int(stamp_s)
            nanosec = int(round((float(stamp_s) - sec) * 1e9))
        q = PROBE.np.linspace(-0.3, 0.3, 7) + float(q_offset)
        return SimpleNamespace(
            header=SimpleNamespace(
                stamp=SimpleNamespace(sec=sec, nanosec=nanosec)),
            name=list(PROBE.RIGHT_ARM_JOINTS),
            position=q.tolist(),
            velocity=PROBE.np.full(7, float(dq_offset)).tolist(),
        )

    def test_header_stamp_is_atomic_and_repeat_holds_same_sample(self):
        node = self._node()
        with (mock.patch.object(PROBE.time, "time", return_value=200.000),
              mock.patch.object(
                  PROBE.time, "monotonic", return_value=300.000)):
            node._joint_state_callback(self._message(100.000))
        with (mock.patch.object(PROBE.time, "time", return_value=200.005),
              mock.patch.object(
                  PROBE.time, "monotonic", return_value=300.005)):
            node._joint_state_callback(self._message(100.000))

        with (mock.patch.object(PROBE.time, "time", return_value=-500.0),
              mock.patch.object(
                  PROBE.time, "monotonic", return_value=300.006)):
            q, dq, age_s, source, sample_stamp_s, timing = (
                node.get_joint_feedback_coherent_with_timing())

        self.assertEqual(sample_stamp_s, 100.0)
        self.assertEqual(node.latest_joint_sample_dt_s, 0.0)
        self.assertEqual(node.latest_joint_sample_stamp_source, "header_repeat")
        self.assertAlmostEqual(age_s, 0.001, places=9)
        self.assertEqual(source, "joint_state")
        self.assertEqual(q.shape, (7,))
        self.assertEqual(dq.shape, (7,))
        self.assertEqual(timing["jointSampleStampS"], sample_stamp_s)
        self.assertEqual(timing["jointSampleDtS"], 0.0)
        self.assertEqual(timing["jointSampleStampSource"], "header_repeat")

    def test_acceleration_uses_joint_source_velocity_interval(self):
        node = self._node()
        with (mock.patch.object(PROBE.time, "time", return_value=200.000),
              mock.patch.object(
                  PROBE.time, "monotonic", return_value=300.000)):
            node._joint_state_callback(self._message(100.000, dq_offset=0.0))
        with (mock.patch.object(PROBE.time, "time", return_value=250.000),
              mock.patch.object(
                  PROBE.time, "monotonic", return_value=350.000)):
            node._joint_state_callback(self._message(100.005, dq_offset=0.1))

        _q, _dq, _age, _source, _stamp, timing = (
            node.get_joint_feedback_coherent_with_timing())

        self.assertEqual(
            timing["jointAccelerationSource"],
            "joint_velocity_difference",
        )
        PROBE.np.testing.assert_allclose(
            timing["jointAcceleration"], PROBE.np.full(7, 20.0),
            rtol=0.0, atol=1e-5)

    def test_nonmonotonic_header_uses_receipt_delta_without_mixing_epochs(self):
        node = self._node()
        with (mock.patch.object(PROBE.time, "time", return_value=200.000),
              mock.patch.object(
                  PROBE.time, "monotonic", return_value=300.000)):
            node._joint_state_callback(self._message(100.000))
        with (mock.patch.object(PROBE.time, "time", return_value=200.005),
              mock.patch.object(
                  PROBE.time, "monotonic", return_value=300.005)):
            node._joint_state_callback(
                self._message(99.000, q_offset=0.01))

        self.assertAlmostEqual(
            node.latest_joint_sample_stamp_s, 100.005, places=9)
        self.assertEqual(
            node.latest_joint_sample_stamp_source,
            "receipt_nonmonotonic_header",
        )

    def test_missing_header_recovers_after_one_receipt_timed_boundary(self):
        node = self._node()
        samples = (
            (100.000, 200.000, 300.000, 0.00),
            (None, 200.012, 300.005, 0.01),
            (100.010, 200.017, 300.010, 0.02),
            (100.015, 200.022, 300.015, 0.03),
        )
        sources = []
        effective_stamps = []
        for header, wall, monotonic_s, q_offset in samples:
            with (mock.patch.object(PROBE.time, "time", return_value=wall),
                  mock.patch.object(
                      PROBE.time, "monotonic", return_value=monotonic_s)):
                node._joint_state_callback(
                    self._message(header, q_offset=q_offset))
            sources.append(node.latest_joint_sample_stamp_source)
            effective_stamps.append(node.latest_joint_sample_stamp_s)

        self.assertEqual(
            sources,
            ["header", "receipt", "receipt_header_recovery", "header"],
        )
        PROBE.np.testing.assert_allclose(
            effective_stamps,
            [100.000, 100.005, 100.010, 100.015],
            rtol=0.0,
            atol=1e-9,
        )

    def test_reused_header_with_changed_q_is_not_a_repeated_sample(self):
        node = self._node()
        with mock.patch.object(
                PROBE.time, "monotonic", return_value=300.000):
            node._joint_state_callback(self._message(100.000))
        with mock.patch.object(
                PROBE.time, "monotonic", return_value=300.006):
            node._joint_state_callback(
                self._message(100.000, q_offset=0.01))

        self.assertAlmostEqual(
            node.latest_joint_sample_stamp_s, 100.006, places=9)
        self.assertEqual(
            node.latest_joint_sample_stamp_source,
            "receipt_nonmonotonic_header",
        )

    def test_header_clock_reset_resumes_header_delta_on_following_frame(self):
        node = self._node()
        for header, monotonic_s, q_offset in (
            (100.000, 300.000, 0.00),
            (1.000, 300.005, 0.01),
            (1.005, 300.010, 0.02),
        ):
            with mock.patch.object(
                    PROBE.time, "monotonic", return_value=monotonic_s):
                node._joint_state_callback(
                    self._message(header, q_offset=q_offset))

        self.assertAlmostEqual(
            node.latest_joint_sample_stamp_s, 100.010, places=9)
        self.assertEqual(node.latest_joint_sample_stamp_source, "header")

        snapshot = node.feedback_snapshot()
        self.assertAlmostEqual(snapshot["jointHeaderStampS"], 1.005)
        self.assertAlmostEqual(snapshot["jointSampleDtS"], 0.005)

    def test_probe_forwards_coherent_sample_stamp_to_controller(self):
        calls = []
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(control_dt=0.005)
        probe.controller = SimpleNamespace(
            predict=lambda *args, **kwargs: calls.append((args, kwargs)))
        stamp_s = 123.456

        probe._predict_policy_with_feedback(
            PROBE.np.asarray([0.1, -0.2, 1.0]),
            PROBE.np.asarray([0.0, 0.0, -1.0]),
            valid=True,
            age_s=0.01,
            arm_q=PROBE.np.zeros(7),
            arm_dq=PROBE.np.ones(7),
            arm_sample_stamp_s=stamp_s,
        )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["arm_sample_stamp_s"], stamp_s)

    def test_recording_keeps_inference_sample_after_competing_callback(self):
        node = self._node()
        with mock.patch.object(
                PROBE.time, "monotonic", return_value=300.000):
            node._joint_state_callback(
                self._message(100.000, q_offset=0.01, dq_offset=0.10))
        q_used, dq_used, _age, _source, _stamp, timing_used = (
            node.get_joint_feedback_coherent_with_timing())

        policy_obs = PROBE.np.zeros(67, dtype=PROBE.np.float32)
        policy_obs[0:7] = q_used
        policy_obs[7:14] = dq_used
        trace = {
            "obs": policy_obs.copy(),
            "jointFeedbackTiming": dict(timing_used),
        }

        # Simulate a newer callback winning the race before the recorder reads
        # the bridge a second time.
        with mock.patch.object(
                PROBE.time, "monotonic", return_value=300.005):
            node._joint_state_callback(
                self._message(100.005, q_offset=0.50, dq_offset=0.90))
        ros_payload = node.feedback_snapshot()
        self.assertNotEqual(
            ros_payload["jointPosition"], PROBE._rounded_vector(q_used))

        matched = PROBE._overlay_policy_trace_joint_feedback(
            ros_payload, trace, policy_obs)

        self.assertTrue(matched)
        self.assertTrue(ros_payload["jointTimingMatchesPolicyObs"])
        self.assertEqual(
            ros_payload["jointPosition"], PROBE._rounded_vector(q_used))
        self.assertEqual(
            ros_payload["jointVelocity"], PROBE._rounded_vector(dq_used))
        self.assertEqual(ros_payload["jointSampleStampS"], 100.0)
        self.assertEqual(ros_payload["jointVelocitySource"], "joint_state")

    def test_recording_keeps_raw_ros_dq_when_policy_dq_is_adapted_and_scaled(self):
        raw_q = PROBE.np.linspace(-0.3, 0.3, 7, dtype=PROBE.np.float32)
        raw_dq = PROBE.np.linspace(-1.0, 1.0, 7, dtype=PROBE.np.float32)
        adapted_dq = raw_dq * PROBE.np.float32(0.8)
        obs_unscaled = PROBE.np.zeros(67, dtype=PROBE.np.float32)
        obs_unscaled[0:7] = raw_q
        obs_unscaled[7:14] = adapted_dq
        obs_scaled = obs_unscaled.copy()
        obs_scaled[7:14] *= PROBE.np.float32(0.5)
        trace = {
            "obs": obs_scaled,
            "obsUnscaled": obs_unscaled,
            "rawJointPosition": raw_q,
            "rawJointVelocity": raw_dq,
            "jointFeedbackTiming": {"jointSampleStampS": 12.5},
        }
        ros_payload = {}

        matched = PROBE._overlay_policy_trace_joint_feedback(
            ros_payload, trace, obs_scaled)

        self.assertTrue(matched)
        self.assertEqual(
            ros_payload["jointVelocity"], PROBE._rounded_vector(raw_dq))
        self.assertEqual(
            ros_payload["policyJointVelocityUnscaled"],
            PROBE._rounded_vector(adapted_dq),
        )
        self.assertEqual(
            ros_payload["policyJointVelocityScaled"],
            PROBE._rounded_vector(adapted_dq * 0.5),
        )
        self.assertFalse(
            ros_payload["jointVelocityMatchesPolicyObsUnscaled"])


class CompensationFeedbackCoherenceTest(unittest.TestCase):

    def test_simcode3_inverse_mpc_uses_modeled_applied_state(self):
        modeled_q = PROBE.np.linspace(-0.1, 0.1, 7).astype(PROBE.np.float32)
        measured_q = modeled_q + 0.5
        captured = []
        compensator = SimpleNamespace(
            enabled=True,
            command_buffer_len=19,
            current_applied_q=lambda: modeled_q.copy(),
            step=lambda **kwargs: (
                captured.append(PROBE.np.asarray(kwargs["applied_q"]).copy())
                or PROBE.np.asarray(kwargs["q_cmd"]).copy()),
        )
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace()
        probe.compensator = compensator
        probe.compensation_was_enabled = True
        probe.compensation_error = None
        probe.ros_bridge = None
        probe.controller = SimpleNamespace(latest_command_state=lambda: {
            "velocity": PROBE.np.zeros(7, dtype=PROBE.np.float32),
            "acceleration": PROBE.np.zeros(7, dtype=PROBE.np.float32),
        })
        probe._ensure_compensator = lambda _options: None
        options = SimpleNamespace(
            enable_compensation=True,
            policy_profile=PROBE.OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67,
        )

        probe._apply_actuator_compensation(
            PROBE.np.zeros(7), options, feedback_q_override=measured_q)

        PROBE.np.testing.assert_array_equal(captured[0], modeled_q)
        self.assertEqual(
            probe.last_compensation_feedback_source,
            "modeled_actuator_applied",
        )

    def test_policy_tick_uses_actor_sample_after_competing_callback(self):
        actor_q = PROBE.np.linspace(-0.30, 0.30, 7).astype(
            PROBE.np.float32)
        callback_q = actor_q + PROBE.np.float32(0.40)
        nominal_q = actor_q + PROBE.np.float32(0.01)

        class RaceBridge:
            def __init__(self):
                self.latest_q = actor_q.copy()
                self.read_count = 0
                self.published = []

            def get_joint_feedback(self):
                self.read_count += 1
                return self.latest_q.copy(), PROBE.np.zeros(7)

            def publish_right_arm_command(self, q):
                self.published.append(PROBE.np.asarray(q).copy())

        class CapturingCompensator:
            enabled = True
            command_buffer_len = 1

            def __init__(self):
                self.reset_q = []
                self.step_applied_q = []

            def reset(self, q):
                self.reset_q.append(PROBE.np.asarray(q).copy())

            def step(self, *, q_cmd, qdot_cmd, qdd_cmd, applied_q):
                del qdot_cmd, qdd_cmd
                self.step_applied_q.append(PROBE.np.asarray(applied_q).copy())
                return PROBE.np.asarray(q_cmd).copy()

            def observe_published(self, _q):
                return None

        bridge = RaceBridge()
        compensator = CapturingCompensator()
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(
            enable_camera_ball=False,
            workspace_safety_mode="off",
            enable_safety_limiter=False,
        )
        probe.ros_bridge = bridge
        probe.controller = SimpleNamespace(
            arm_cmd_q=nominal_q.copy(),
            latest_command_state=lambda: {
                "position": nominal_q.copy(),
                "velocity": PROBE.np.zeros(7, dtype=PROBE.np.float32),
                "acceleration": PROBE.np.zeros(7, dtype=PROBE.np.float32),
            },
        )
        probe.compensator = compensator
        probe.compensation_was_enabled = False
        probe.compensation_error = None
        probe._ensure_compensator = lambda _options: None
        probe.safety_limiter = PROBE.RightArmCommandSafetyLimiter(
            actor_q, dt=0.005)
        probe.command_stream_active = True
        probe.last_command_disable_hold = {}
        probe.workspace_soft_filter = {}
        probe.done_failsafe_active = False
        probe.control_loop_tick = 0
        probe.now_s = 0.005
        probe.last_ball_pos = PROBE.np.zeros(3, dtype=PROBE.np.float32)
        probe.last_ball_vel = PROBE.np.zeros(3, dtype=PROBE.np.float32)
        probe.last_observed_ball_pos = PROBE.np.zeros(
            3, dtype=PROBE.np.float32)
        probe.last_observed_ball_vel = PROBE.np.zeros(
            3, dtype=PROBE.np.float32)
        probe.last_measured_ball_pos = PROBE.np.zeros(
            3, dtype=PROBE.np.float32)
        probe.last_measured_ball_vel = PROBE.np.zeros(
            3, dtype=PROBE.np.float32)
        probe.last_estimator_info = {}
        probe._select_ball_source = lambda *_args, **_kwargs: (
            PROBE.np.zeros(3, dtype=PROBE.np.float32),
            PROBE.np.zeros(3, dtype=PROBE.np.float32),
            True,
            "drag",
            None,
            None,
            None,
        )
        probe._apply_ball_noise = lambda p, v, *_args: (p.copy(), v.copy())
        probe._resolve_control_ball_state = lambda p, v, *_args, **_kwargs: (
            p.copy(), v.copy(), True, 0.0, {})

        def actor_step(*_args, **_kwargs):
            obs = PROBE.np.zeros(67, dtype=PROBE.np.float32)
            obs[:7] = actor_q
            probe.last_policy_execution_trace = {"obs": obs}
            # A JointState callback completes after actor inference and before
            # command publication/compensation.
            bridge.latest_q = callback_q.copy()
            return "policy", "", {}, 0.0, True

        probe._run_one_control_step = actor_step
        probe._step_sim_mirror = lambda *_args, **_kwargs: {}
        probe._record_control_sample = lambda **_kwargs: None
        options = SimpleNamespace(
            publish_commands=True,
            enable_compensation=True,
            workspace_safety_mode="off",
            ball_release_use_estimator_state=True,
            comp_delay_ms=72.0,
            comp_actuator_tau_s=0.074,
        )

        probe._run_control_tick_locked(
            PROBE.np.zeros(3), PROBE.np.zeros(3), True, 0.0, options,
            make_snapshot=False)

        # Both first-enable reset and step must use the actor's atomic sample;
        # the newer callback must not be read on this policy transition.
        self.assertEqual(bridge.read_count, 0)
        self.assertEqual(len(compensator.reset_q), 1)
        self.assertEqual(len(compensator.step_applied_q), 1)
        PROBE.np.testing.assert_array_equal(compensator.reset_q[0], actor_q)
        PROBE.np.testing.assert_array_equal(
            compensator.step_applied_q[0], actor_q)
        self.assertEqual(
            probe.last_compensation_feedback_source,
            "policy_atomic_snapshot",
        )

        # Non-policy controllers intentionally retain the freshest-feedback
        # behavior instead of reusing the previous actor snapshot.
        probe._publish_control_command(
            nominal_q,
            options,
            apply_compensation=True,
            compensation_feedback_q=probe._policy_compensation_feedback_q(
                "racket_tracking", False),
        )
        self.assertEqual(bridge.read_count, 1)
        PROBE.np.testing.assert_array_equal(
            compensator.step_applied_q[-1], callback_q)
        self.assertEqual(
            probe.last_compensation_feedback_source, "bridge_latest")


class Simcode3BallSamplingContractTest(unittest.TestCase):

    def test_fractional_90hz_refresh_count_on_200hz_clock(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(control_dt=0.005)
        probe.policy_ball_sample_profile = None
        options = SimpleNamespace(
            policy_profile=PROBE.OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67,
            control_mode="policy",
            use_ball_estimator=False,
            ball_observation_rate_hz=90.0,
        )
        sampled_positions = []
        for step in range(200):
            pos, _vel, valid, _age = (
                probe._apply_policy_ball_sampling_contract(
                    PROBE.np.full(3, step, dtype=PROBE.np.float32),
                    PROBE.np.zeros(3, dtype=PROBE.np.float32),
                    True,
                    0.0,
                    options,
                ))
            self.assertTrue(valid)
            sampled_positions.append(float(pos[0]))

        refreshes = 1 + sum(
            current != previous for previous, current in zip(
                sampled_positions, sampled_positions[1:]))
        self.assertEqual(refreshes, 90)
        self.assertEqual(probe.policy_ball_sample_sequence, 90)

    def test_fractional_60hz_refresh_and_age_match_training(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(control_dt=0.005)
        probe.policy_ball_sample_profile = None
        options = SimpleNamespace(
            policy_profile=PROBE.OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67,
            control_mode="policy",
            use_ball_estimator=False,
        )
        valid_history = []
        age_history = []
        pos_history = []
        for step in range(11):
            pos, _vel, valid, age = probe._apply_policy_ball_sampling_contract(
                PROBE.np.full(3, step, dtype=PROBE.np.float32),
                PROBE.np.zeros(3, dtype=PROBE.np.float32),
                True,
                0.0,
                options,
            )
            valid_history.append(valid)
            age_history.append(age)
            pos_history.append(float(pos[0]))

        self.assertTrue(all(valid_history))
        self.assertEqual(pos_history[:4], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(
            [i for i, age in enumerate(age_history) if age == 0.0],
            [0, 4, 7, 10],
        )
        PROBE.np.testing.assert_allclose(
            age_history[:5], [0.0, 0.005, 0.010, 0.015, 0.0])

    def test_reentry_reset_makes_next_sample_fresh(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(control_dt=0.005)
        probe.policy_ball_sample_profile = (
            PROBE.OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67)
        probe.policy_ball_sample_step = 3
        probe.policy_ball_sample_pos = PROBE.np.ones(3, dtype=PROBE.np.float32)
        probe.policy_ball_sample_vel = PROBE.np.zeros(3, dtype=PROBE.np.float32)
        probe.policy_ball_sample_age_s = 0.010
        probe.policy_ball_sample_has_valid = True
        options = SimpleNamespace(
            policy_profile=PROBE.OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67,
            control_mode="policy",
            use_ball_estimator=False,
        )

        probe._reset_policy_ball_sampling_contract()
        pos, _vel, valid, age = probe._apply_policy_ball_sampling_contract(
            PROBE.np.full(3, 7.0, dtype=PROBE.np.float32),
            PROBE.np.zeros(3, dtype=PROBE.np.float32),
            True,
            0.0,
            options,
        )

        self.assertTrue(valid)
        self.assertEqual(age, 0.0)
        PROBE.np.testing.assert_array_equal(pos, PROBE.np.full(3, 7.0))

    def test_gpu1_transient_dropout_holds_last_sample_like_simcode5(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(
            control_dt=0.005,
            ball_lost_timeout_s=0.6,
        )
        probe.policy_ball_sample_profile = None
        options = SimpleNamespace(
            policy_profile=PROBE.OBSERVATION_PROFILE_GPU1_QACC_FT_V153,
            control_mode="policy",
            use_ball_estimator=False,
            ball_observation_rate_hz=60.0,
        )
        expected_pos = PROBE.np.array(
            [0.1, -0.2, 1.4], dtype=PROBE.np.float32)
        expected_vel = PROBE.np.array(
            [0.2, 0.1, -0.3], dtype=PROBE.np.float32)

        pos, vel, valid, age = probe._apply_policy_ball_sampling_contract(
            expected_pos,
            expected_vel,
            True,
            0.0,
            options,
            source_timestamp_s=10.0,
        )
        self.assertTrue(valid)
        self.assertEqual(age, 0.0)
        PROBE.np.testing.assert_array_equal(pos, expected_pos)
        PROBE.np.testing.assert_array_equal(vel, expected_vel)

        # Recording 20260901_211102 contained one missed 90 Hz IR frame,
        # observed by three 200 Hz control ticks.  The frame immediately
        # before and after it was valid.  Those three ticks must keep running
        # from the last accepted sample instead of entering
        # policy_ball_state_hold and disarming Send Command.
        for expected_age in (0.005, 0.010, 0.015):
            pos, vel, valid, age = (
                probe._apply_policy_ball_sampling_contract(
                    PROBE.np.zeros(3, dtype=PROBE.np.float32),
                    PROBE.np.zeros(3, dtype=PROBE.np.float32),
                    False,
                    0.6,
                    options,
                    source_timestamp_s=10.011,
                ))
            self.assertTrue(valid)
            self.assertAlmostEqual(age, expected_age)
            PROBE.np.testing.assert_array_equal(pos, expected_pos)
            PROBE.np.testing.assert_array_equal(vel, expected_vel)

    def test_gpu1_continuous_dropout_still_invalidates_after_timeout(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(
            control_dt=0.005,
            ball_lost_timeout_s=0.6,
        )
        probe.policy_ball_sample_profile = None
        options = SimpleNamespace(
            policy_profile=PROBE.OBSERVATION_PROFILE_GPU1_QACC_FT_V153,
            control_mode="policy",
            use_ball_estimator=False,
            ball_observation_rate_hz=60.0,
        )
        probe._apply_policy_ball_sampling_contract(
            PROBE.np.array([0.1, -0.2, 1.4], dtype=PROBE.np.float32),
            PROBE.np.array([0.2, 0.1, -0.3], dtype=PROBE.np.float32),
            True,
            0.0,
            options,
            source_timestamp_s=10.0,
        )

        for _ in range(71):
            pos, vel, valid, age = (
                probe._apply_policy_ball_sampling_contract(
                    PROBE.np.zeros(3, dtype=PROBE.np.float32),
                    PROBE.np.zeros(3, dtype=PROBE.np.float32),
                    False,
                    0.6,
                    options,
                    source_timestamp_s=10.011,
                ))
        self.assertFalse(valid)
        self.assertGreater(age, 0.350)
        self.assertFalse(probe.policy_ball_sample_has_valid)
        PROBE.np.testing.assert_array_equal(
            pos, PROBE.np.array([0.1, -0.2, 1.4], dtype=PROBE.np.float32))
        PROBE.np.testing.assert_array_equal(
            vel, PROBE.np.array([0.2, 0.1, -0.3], dtype=PROBE.np.float32))

    def test_gpu1_high_occlusion_miss_reaches_control_as_held_valid_sample(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(
            control_dt=0.005,
            ball_lost_timeout_s=0.6,
            enable_camera_ball=False,
        )
        probe.ros_bridge = None
        probe.controller = SimpleNamespace(
            arm_cmd_q=PROBE.np.zeros(7, dtype=PROBE.np.float32))
        probe.control_loop_tick = 100
        probe.now_s = 1.0
        probe.last_ball_pos = PROBE.np.zeros(3, dtype=PROBE.np.float32)
        probe.last_ball_vel = PROBE.np.zeros(3, dtype=PROBE.np.float32)
        probe.last_observed_ball_pos = PROBE.np.zeros(
            3, dtype=PROBE.np.float32)
        probe.last_observed_ball_vel = PROBE.np.zeros(
            3, dtype=PROBE.np.float32)
        probe.last_measured_ball_pos = PROBE.np.zeros(
            3, dtype=PROBE.np.float32)
        probe.last_measured_ball_vel = PROBE.np.zeros(
            3, dtype=PROBE.np.float32)
        probe.last_estimator_info = {}
        probe.policy_needs_reentry_reset = False
        probe.policy_ball_sample_profile = None
        probe.recorder = SimpleNamespace(active=False)
        options = SimpleNamespace(
            policy_profile=PROBE.OBSERVATION_PROFILE_GPU1_QACC_FT_V153,
            control_mode="policy",
            use_ball_estimator=False,
            ball_observation_rate_hz=60.0,
            ball_release_use_estimator_state=True,
            publish_commands=True,
        )
        held_pos = PROBE.np.array(
            [0.1, -0.2, 1.4], dtype=PROBE.np.float32)
        held_vel = PROBE.np.array(
            [0.2, 0.1, -0.3], dtype=PROBE.np.float32)
        probe._apply_policy_ball_sampling_contract(
            held_pos,
            held_vel,
            True,
            0.0,
            options,
            source_timestamp_s=10.0,
        )
        probe._select_ball_source = lambda *_args, **_kwargs: (
            PROBE.np.zeros(3, dtype=PROBE.np.float32),
            PROBE.np.zeros(3, dtype=PROBE.np.float32),
            False,
            "camera",
            None,
            10.011,
            {"observation_reason": "ir_detection_lost"},
        )
        probe._apply_ball_noise = lambda p, v, *_args: (p.copy(), v.copy())
        probe._resolve_high_ball_occlusion_state = (
            lambda p, v, *_args, **_kwargs: (
                p.copy(),
                v.copy(),
                False,
                10.011,
                {
                    "enabled": True,
                    "active": False,
                    "predicted": False,
                    "state": "idle",
                    "reason": "last_measurement_below_activation_z",
                },
            ))
        probe._resolve_control_ball_state = (
            lambda p, v, *_args, **_kwargs: (
                p.copy(), v.copy(), False, 0.6, {}))
        control_inputs = []

        def control_step(pos, vel, valid, age, *_args, **_kwargs):
            control_inputs.append((pos.copy(), vel.copy(), valid, age))
            return "policy", "", {}, float(age), True

        probe._run_one_control_step = control_step
        probe._policy_compensation_feedback_q = lambda *_args: None
        probe._publish_control_command = lambda *_args, **_kwargs: True
        probe._step_sim_mirror = lambda *_args, **_kwargs: None
        probe._record_control_sample = lambda **_kwargs: None

        probe._run_control_tick_locked(
            PROBE.np.zeros(3),
            PROBE.np.zeros(3),
            True,
            0.0,
            options,
            make_snapshot=False,
        )

        self.assertEqual(len(control_inputs), 1)
        pos, vel, valid, age = control_inputs[0]
        self.assertTrue(valid)
        self.assertAlmostEqual(age, 0.005)
        PROBE.np.testing.assert_array_equal(pos, held_pos)
        PROBE.np.testing.assert_array_equal(vel, held_vel)

    def test_simcode4_fractional_hold_keeps_observation_age_zero(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(control_dt=0.005)
        probe.policy_ball_sample_profile = None
        options = SimpleNamespace(
            policy_profile=PROBE.OBSERVATION_PROFILE_SIMCODE4_SERVO67,
            control_mode="policy",
            use_ball_estimator=False,
        )
        ages = []
        positions = []
        for step in range(11):
            pos, _vel, valid, age = probe._apply_policy_ball_sampling_contract(
                PROBE.np.full(3, step, dtype=PROBE.np.float32),
                PROBE.np.zeros(3, dtype=PROBE.np.float32),
                True,
                0.0,
                options,
            )
            self.assertTrue(valid)
            ages.append(age)
            positions.append(float(pos[0]))
        self.assertEqual(positions[:4], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(positions[4:7], [4.0, 4.0, 4.0])
        self.assertEqual(ages, [0.0] * 11)

    def test_simcode6_fractional_60hz_hold_keeps_age_zero(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(control_dt=0.005)
        probe.policy_ball_sample_profile = None
        options = SimpleNamespace(
            policy_profile=PROBE.OBSERVATION_PROFILE_SIMCODE6_QVEL67,
            control_mode="policy",
            use_ball_estimator=False,
            ball_observation_rate_hz=60.0,
        )
        positions = []
        ages = []
        for step in range(200):
            pos, _vel, valid, age = (
                probe._apply_policy_ball_sampling_contract(
                    PROBE.np.full(3, step, dtype=PROBE.np.float32),
                    PROBE.np.zeros(3, dtype=PROBE.np.float32),
                    True,
                    0.0,
                    options,
                ))
            self.assertTrue(valid)
            positions.append(float(pos[0]))
            ages.append(float(age))

        refreshes = 1 + sum(
            current != previous for previous, current in zip(
                positions, positions[1:]))
        self.assertEqual(refreshes, 60)
        self.assertEqual(probe.policy_ball_sample_sequence, 60)
        self.assertEqual(ages, [0.0] * 200)

    def test_reacquisition_handover_publishes_each_fresh_camera_frame(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(control_dt=0.005)
        probe.policy_ball_sample_profile = None
        options = SimpleNamespace(
            policy_profile=PROBE.OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67,
            control_mode="policy",
            use_ball_estimator=False,
            ball_observation_rate_hz=90.0,
            camera_control_time_compensation=False,
            ball_source="camera",
        )

        first, _vel, valid, _age = (
            probe._apply_policy_ball_sampling_contract(
                PROBE.np.zeros(3, dtype=PROBE.np.float32),
                PROBE.np.zeros(3, dtype=PROBE.np.float32),
                True, 0.0, options,
                source_timestamp_s=1.000,
                force_fresh_source=True))
        second, _vel, valid2, _age = (
            probe._apply_policy_ball_sampling_contract(
                PROBE.np.ones(3, dtype=PROBE.np.float32),
                PROBE.np.zeros(3, dtype=PROBE.np.float32),
                True, 0.0, options,
                source_timestamp_s=1.011,
                force_fresh_source=True))
        repeated, _vel, valid3, _age = (
            probe._apply_policy_ball_sampling_contract(
                PROBE.np.full(3, 2.0, dtype=PROBE.np.float32),
                PROBE.np.zeros(3, dtype=PROBE.np.float32),
                True, 0.0, options,
                source_timestamp_s=1.011,
                force_fresh_source=True))

        self.assertTrue(valid and valid2 and valid3)
        PROBE.np.testing.assert_array_equal(first, PROBE.np.zeros(3))
        PROBE.np.testing.assert_array_equal(second, PROBE.np.ones(3))
        # A repeated 200 Hz tick still holds the same camera state; this path
        # is not the disabled 90->200 trajectory propagation feature.
        PROBE.np.testing.assert_array_equal(repeated, PROBE.np.ones(3))
        self.assertEqual(probe.policy_ball_sample_sequence, 2)

    def test_gpu1_reacquisition_cannot_bypass_60hz_schedule(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(control_dt=0.005)
        probe.policy_ball_sample_profile = None
        options = SimpleNamespace(
            policy_profile=PROBE.OBSERVATION_PROFILE_GPU1_QACC_FT_V153,
            control_mode="policy",
            use_ball_estimator=False,
            ball_observation_rate_hz=60.0,
            camera_control_time_compensation=False,
            ball_source="camera",
        )

        first, _vel, valid, _age = (
            probe._apply_policy_ball_sampling_contract(
                PROBE.np.zeros(3, dtype=PROBE.np.float32),
                PROBE.np.zeros(3, dtype=PROBE.np.float32),
                True, 0.0, options,
                source_timestamp_s=1.000,
                force_fresh_source=True))
        held, _vel, held_valid, held_age = (
            probe._apply_policy_ball_sampling_contract(
                PROBE.np.ones(3, dtype=PROBE.np.float32),
                PROBE.np.zeros(3, dtype=PROBE.np.float32),
                True, 0.0, options,
                source_timestamp_s=1.017,
                force_fresh_source=True))

        self.assertTrue(valid and held_valid)
        PROBE.np.testing.assert_array_equal(first, PROBE.np.zeros(3))
        PROBE.np.testing.assert_array_equal(held, PROBE.np.zeros(3))
        self.assertAlmostEqual(float(held_age), 0.005)
        self.assertEqual(probe.policy_ball_sample_sequence, 1)

    def test_gpu1_sensor_state_requires_finite_nonfuture_timestamp(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(
            control_dt=0.005,
            ball_lost_timeout_s=0.6,
            racket_tracking_max_ball_age_s=0.12,
        )
        probe.ros_bridge = SimpleNamespace(now_sec=lambda: 10.0)
        options = SimpleNamespace(
            policy_profile=PROBE.OBSERVATION_PROFILE_GPU1_QACC_FT_V153,
            control_mode="policy",
            ball_source="camera",
        )

        self.assertEqual(
            probe._raw_measurement_age_s(True, None, options), 0.6)
        self.assertEqual(
            probe._raw_measurement_age_s(True, 10.001, options), 0.6)
        self.assertAlmostEqual(
            probe._raw_measurement_age_s(True, 9.98, options), 0.02)


class SourceSeqAlignmentTest(unittest.TestCase):

    def test_joint_acceleration_uses_only_contiguous_source_seq_interval(self):
        rows = []
        for seq, velocity in ((10, 0.0), (11, 0.1), (13, 10.0)):
            rows.append({
                "sourceSeq": seq,
                "sourceTimeS": seq * 0.005,
                "policyObs": [0.0] * 7 + [velocity] * 7,
            })

        acceleration = PROBE._aligned_joint_acceleration(rows, "policyObs")

        PROBE.np.testing.assert_allclose(
            acceleration[0], PROBE.np.full(7, 20.0),
            rtol=0.0, atol=1e-9)
        self.assertTrue(PROBE.np.all(PROBE.np.isnan(acceleration[1:])))

    def test_real_acceleration_uses_joint_sample_clock_not_policy_clock(self):
        rows = []
        for seq, stamp, velocity in (
                (10, 1.000, 0.0),
                (11, 1.000, 0.0),
                (12, 1.020, 0.2)):
            rows.append({
                "sourceSeq": seq,
                "sourceTimeS": seq * 0.005,
                "policyObs": [0.0] * 7 + [velocity] * 7,
                "ros": {"jointSampleStampS": stamp},
            })

        acceleration = PROBE._aligned_joint_acceleration(
            rows, "policyObs", time_key="ros.jointSampleStampS")

        self.assertTrue(PROBE.np.all(PROBE.np.isnan(acceleration[0])))
        PROBE.np.testing.assert_allclose(
            acceleration[1], PROBE.np.full(7, 10.0),
            rtol=0.0, atol=1e-9)

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


class IdealPd67ResetAndGapAnalysisTest(unittest.TestCase):

    def test_idealpd67_reset_ball_matches_racket_launch_geometry(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.active_policy_profile = (
            PROBE.OBSERVATION_PROFILE_SIMCODE2_IDEALPD67)
        probe.reset_ball_rng = PROBE.np.random.default_rng(123)
        probe.controller = SimpleNamespace(
            _checkpoint_env_cfg={
                "racket_launch_surface_gap_range_m": (0.005, 0.010),
                "racket_launch_xy_jitter": 0.004,
                "racket_launch_vxy_max": 0.003,
                "racket_launch_vnormal_max": 0.003,
                "racket_launch_edge_margin": 0.005,
                "ball_launch_height": 0.30,
            },
            _compute_racket_launch_frame_base=lambda _q: (
                PROBE.np.asarray(
                    [0.12, -0.34, 1.01], dtype=PROBE.np.float32),
                PROBE.np.asarray([1.0, 0.0, 0.0], dtype=PROBE.np.float32),
                PROBE.np.asarray([0.0, 1.0, 0.0], dtype=PROBE.np.float32),
                PROBE.np.asarray([0.0, 0.0, 1.0], dtype=PROBE.np.float32),
                0.08,
                0.002,
                True,
            ))

        position, velocity, contract = probe._reset_ball_state(
            PROBE.np.zeros(7, dtype=PROBE.np.float32))

        self.assertLessEqual(abs(float(position[0]) - 0.12), 0.004001)
        self.assertLessEqual(abs(float(position[1]) + 0.34), 0.004001)
        self.assertGreaterEqual(float(position[2]), 1.037)
        self.assertLessEqual(float(position[2]), 1.042001)
        self.assertLessEqual(float(PROBE.np.max(PROBE.np.abs(velocity))), 0.003001)
        self.assertEqual(contract["mode"], "racket_launch")
        self.assertGreaterEqual(contract["surfaceGapM"], 0.005)
        self.assertLessEqual(contract["surfaceGapM"], 0.010001)
        self.assertNotAlmostEqual(float(position[2]), 1.31, places=2)

    @staticmethod
    def _gap_row(seq, *, profile, generation, episode, q_gap):
        real_obs = [0.0] * 67
        sim_obs = [0.0] * 67
        for idx in range(7):
            real_obs[idx] = float(q_gap)
            real_obs[42 + idx] = 0.05
            sim_obs[42 + idx] = 0.001
        real_obs[20:26] = [0.1, -0.2, 1.3, 0.0, 0.0, -0.4]
        sim_obs[20:26] = list(real_obs[20:26])
        nominal = [0.1] * 7
        compensated = [0.624] * 7
        return {
            "sourceSeq": seq,
            "time": seq * 0.005,
            "mode": "policy",
            "policyExecuted": True,
            "simAligned": True,
            "simInputMatch": True,
            "policyObs": real_obs,
            "policyAction": [0.3] * 7,
            "policyCommand": {"position": nominal},
            "runtime": {
                "activePolicyProfile": profile,
                "policyGeneration": generation,
            },
            "command": {
                "compensatedReferencePosition": compensated,
                "publishedSafePosition": compensated,
                "compensation": {"mpcMaxDeltaRad": 0.524},
            },
            "sim": {
                "policyObs": sim_obs,
                "policyAction": [0.0] * 7,
                "policyCommand": [0.0] * 7,
                "sync": {
                    "episode": episode,
                    "policyProfile": profile,
                    "policyGeneration": generation,
                },
            },
        }

    def test_gap_analysis_selects_ideal_cohort_and_attributes_tracking(self):
        legacy = [
            self._gap_row(
                seq,
                profile=PROBE.OBSERVATION_PROFILE_LEGACY_BUG_COMPAT,
                generation=1,
                episode=2,
                q_gap=0.0,
            )
            for seq in range(20)
        ]
        ideal = [
            self._gap_row(
                100 + seq,
                profile=PROBE.OBSERVATION_PROFILE_SIMCODE2_IDEALPD67,
                generation=3,
                episode=4,
                q_gap=0.0 if seq == 0 else 0.03,
            )
            for seq in range(12)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            report = PROBE._analyze_sim_real_gap(
                legacy + ideal,
                output,
                alignment={"alignmentValid": True},
            )

            self.assertEqual(
                report["selectedCohort"]["policyProfile"],
                PROBE.OBSERVATION_PROFILE_SIMCODE2_IDEALPD67)
            self.assertEqual(report["simSampleCount"], 12)
            self.assertEqual(
                report["groups"]["ball_observation"]["maxAbs"], 0.0)
            self.assertEqual(
                report["diagnostics"]["primaryGapHint"],
                "real_actuation_tracking")
            self.assertAlmostEqual(
                report["diagnostics"]["resetJointPositionRmsRad"],
                0.0)
            self.assertEqual(
                report["diagnostics"]["compensationSaturation"][
                    "anyJointTickFraction"],
                1.0)
            self.assertTrue((output / "sim2real_gap_timeseries.csv").is_file())
            self.assertTrue((output / "sim2real_gap_timeseries.png").is_file())

    def test_analysis_schema_marks_old_artifacts_for_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            analysis_dir = Path(tmp)
            summary = analysis_dir / "analysis_summary.json"
            summary.write_text(
                json.dumps({"analysisSchemaVersion": 1}), encoding="utf-8")
            self.assertLess(
                PROBE._analysis_schema_version_from_dir(analysis_dir),
                PROBE.ANALYSIS_SCHEMA_VERSION)
            summary.write_text(
                json.dumps({
                    "analysisSchemaVersion": PROBE.ANALYSIS_SCHEMA_VERSION,
                }),
                encoding="utf-8",
            )
            self.assertEqual(
                PROBE._analysis_schema_version_from_dir(analysis_dir),
                PROBE.ANALYSIS_SCHEMA_VERSION)

    def test_control_proxy_reset_waits_for_matching_ack_not_stale_snapshot(self):
        proxy = PROBE.ControlSubprocessProxy.__new__(
            PROBE.ControlSubprocessProxy)
        proxy.lock = threading.Lock()
        proxy.command_q = queue.Queue(maxsize=1)
        proxy.latest_snapshot = {
            "mode": "old",
            "controlResetRequestId": 0,
        }
        proxy.latest_recording_status = {"active": False}
        proxy._reset_request_seq = 0
        proxy._reset_in_progress = False

        def acknowledge(**_kwargs):
            try:
                command = proxy.command_q.get_nowait()
            except queue.Empty:
                return
            request_id = int(
                command["payload"]["_controlResetRequestId"])
            with proxy.lock:
                proxy.latest_snapshot = {
                    "mode": "reset",
                    "controlResetRequestId": request_id,
                }

        proxy._drain_status = acknowledge
        snapshot = proxy.reset({"policyProfile": "simcode2_idealpd67"})

        self.assertEqual(snapshot["mode"], "reset")
        self.assertEqual(snapshot["controlResetRequestId"], 1)
        self.assertFalse(proxy._reset_in_progress)


class Gpu1HotSwitchRegressionTest(unittest.TestCase):

    @staticmethod
    def _controller(profile, model_id=""):
        return SimpleNamespace(
            observation_profile=profile,
            policy_model_id=model_id,
            reset=mock.Mock(),
        )

    def test_gpu1_snapshot_and_payload_model_are_profile_aware(self):
        snapshot = {
            "runtime": {
                "activePolicyProfile": (
                    PROBE.OBSERVATION_PROFILE_GPU1_QACC_FT_V153),
                "activePolicyModel": "v153",
            },
        }
        self.assertEqual(
            PROBE.ControlSubprocessProxy._snapshot_policy_model(snapshot),
            PROBE.GPU1_QACC_FT_MODEL_V153,
        )

    def test_original_profile_choices_remain_enabled(self):
        self.assertEqual(set(PROBE.RUNTIME_SWITCHABLE_POLICY_PROFILES), {
            "fplc_gpu0_v152", "fplc_gpu1_v151", "fplc_gpu0_v161", "fplc_gpu1_v172",
            "fplc_v220_s17_u839_20261013_r1",
            PROBE.OBSERVATION_PROFILE_LEGACY_BUG_COMPAT,
            PROBE.OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67,
            PROBE.OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67,
            PROBE.OBSERVATION_PROFILE_SIMCODE6_QVEL67,
            PROBE.OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85,
            PROBE.OBSERVATION_PROFILE_GPU1_QACC_FT_V153,
        })
        self.assertNotIn(
            PROBE.OBSERVATION_PROFILE_SIMCODE2_IDEALPD67,
            PROBE.RUNTIME_SWITCHABLE_POLICY_PROFILES,
        )
        self.assertNotIn(
            PROBE.OBSERVATION_PROFILE_SIMCODE4_SERVO67,
            PROBE.RUNTIME_SWITCHABLE_POLICY_PROFILES,
        )
        self.assertEqual(
            PROBE._normalize_policy_model_for_profile(
                PROBE.OBSERVATION_PROFILE_GPU1_QACC_FT_V153,
                "v91",  # disabled GPU0 selector can retain this stale value
            ),
            PROBE.GPU1_QACC_FT_MODEL_V153,
        )

    def test_gpu1_uses_67_scales_even_when_probe_started_with_gpu0_57d(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(
            policy_obs_dim=57,
            policy_dq_observation_scale=1.0,
        )
        probe.policy_controllers = {
            PROBE.OBSERVATION_PROFILE_GPU1_QACC_FT_V153: SimpleNamespace(
                _expected_obs_dim=67),
        }
        self.assertEqual(
            probe._policy_observation_dim_for_profile(
                PROBE.OBSERVATION_PROFILE_GPU1_QACC_FT_V153),
            67,
        )
        self.assertEqual(
            probe._policy_observation_dim_for_profile(
                PROBE.OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67),
            67,
        )

    def test_preloaded_gpu0_gpu1_switch_does_not_rebuild_or_kill_runtime(self):
        gpu0 = self._controller(
            PROBE.OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85, "v91")
        gpu1 = self._controller(
            PROBE.OBSERVATION_PROFILE_GPU1_QACC_FT_V153)
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.active_policy_profile = (
            PROBE.OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85)
        probe.active_policy_model = "v91"
        probe.controller = gpu0
        probe.policy_controllers = {
            probe.active_policy_profile: gpu0,
            PROBE.OBSERVATION_PROFILE_GPU1_QACC_FT_V153: gpu1,
        }
        probe.policy_generation = 7
        probe.command_stream_active = True
        probe.policy_switch_publish_rearm_required = False
        probe._publish_hold_on_command_disable = mock.Mock(
            side_effect=lambda: setattr(
                probe, "command_stream_active", False) or True)
        probe._safety_limiter_reset_reference = mock.Mock(
            return_value=PROBE.np.zeros(7, dtype=PROBE.np.float32))
        probe._sync_policy_to_safe_pose = mock.Mock()
        probe._reset_compensator_to_current_arm = mock.Mock()
        probe._reset_servo_target_planner = mock.Mock()
        probe._reset_policy_ball_sampling_contract = mock.Mock()
        probe._reset_policy_dq_adapter = mock.Mock()

        probe._set_policy_workspace_limits = mock.Mock()
        changed = probe._select_policy_profile(
            PROBE.OBSERVATION_PROFILE_GPU1_QACC_FT_V153)

        self.assertTrue(changed)
        self.assertIs(probe.controller, gpu1)
        self.assertEqual(probe.active_policy_model, "v153")
        self.assertEqual(probe.policy_generation, 8)
        self.assertTrue(probe.policy_switch_publish_rearm_required)
        probe._publish_hold_on_command_disable.assert_called_once_with()
        gpu1.reset.assert_called_once_with()

    def test_restored_profile_is_loaded_on_demand_without_release_lock_exit(self):
        gpu0 = self._controller(
            PROBE.OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85, "v91")
        simcode5 = self._controller(
            PROBE.OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67)
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.active_policy_profile = (
            PROBE.OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85)
        probe.active_policy_model = "v91"
        probe.controller = gpu0
        probe.policy_controllers = {probe.active_policy_profile: gpu0}
        probe.policy_generation = 2
        probe.command_stream_active = False
        probe.policy_switch_publish_rearm_required = False
        probe._build_release_locked_policy_controller = mock.Mock(
            return_value=simcode5)
        probe._safety_limiter_reset_reference = mock.Mock(
            return_value=PROBE.np.zeros(7, dtype=PROBE.np.float32))
        probe._sync_policy_to_safe_pose = mock.Mock()
        probe._reset_compensator_to_current_arm = mock.Mock()
        probe._reset_servo_target_planner = mock.Mock()
        probe._reset_policy_ball_sampling_contract = mock.Mock()
        probe._reset_policy_dq_adapter = mock.Mock()

        probe._set_policy_workspace_limits = mock.Mock()
        changed = probe._select_policy_profile(
            PROBE.OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67)

        self.assertTrue(changed)
        self.assertIs(probe.controller, simcode5)
        self.assertIs(
            probe.policy_controllers[
                PROBE.OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67],
            simcode5,
        )
        probe._build_release_locked_policy_controller.assert_called_once_with(
            PROBE.OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67)

    def test_dead_control_process_reset_fails_immediately(self):
        proxy = PROBE.ControlSubprocessProxy.__new__(
            PROBE.ControlSubprocessProxy)
        proxy.lock = threading.Lock()
        proxy._reset_request_lock = threading.Lock()
        proxy.command_q = queue.Queue(maxsize=1)
        proxy.latest_config = {
            "policy": {
                "profiles": [
                    PROBE.OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85,
                    PROBE.OBSERVATION_PROFILE_GPU1_QACC_FT_V153,
                ],
            },
        }
        proxy.latest_snapshot = {"controlResetRequestId": 0}
        proxy.latest_recording_status = {"active": False}
        proxy._reset_request_seq = 0
        proxy._reset_in_progress = False
        proxy.last_error = "profile switch failed"
        proxy.process = SimpleNamespace(
            is_alive=lambda: False,
            exitcode=1,
        )
        proxy._drain_status = lambda **_kwargs: None

        with self.assertRaisesRegex(
                RuntimeError, "control subprocess failed during reset"):
            proxy.reset({
                "policyProfile": (
                    PROBE.OBSERVATION_PROFILE_GPU1_QACC_FT_V153),
            })
        self.assertFalse(proxy._reset_in_progress)

    def test_disconnected_json_client_does_not_raise_twice(self):
        handler = PROBE.ProbeRequestHandler.__new__(PROBE.ProbeRequestHandler)
        handler.send_response = mock.Mock()
        handler.send_header = mock.Mock()
        handler.end_headers = mock.Mock()
        handler.wfile = SimpleNamespace(
            write=mock.Mock(side_effect=BrokenPipeError(32, "broken pipe")))

        self.assertFalse(handler._send_json({"ok": True}))


class SimMirrorLifecycleContractTest(unittest.TestCase):

    def test_transient_real_hold_preserves_mirror_episode(self):
        class FakeSimMirrorSubprocess:
            def __init__(self):
                self.cfg = SimpleNamespace()
                self.episode = 0
                self.stream_seq = 0
                self.reset_count = 0
                self.step_source_seqs = []

            def reset(self, arm_q=None, **_kwargs):
                self.reset_count += 1
                self.episode += 1
                self.stream_seq = 0
                return self.snapshot()

            def step(self, _ball_pos, _ball_vel, **kwargs):
                self.stream_seq += 1
                self.step_source_seqs.append(int(kwargs["source_seq"]))
                return self.snapshot()

            def snapshot(self):
                return {
                    "available": True,
                    "sync": {
                        "ready": True,
                        "episode": self.episode,
                        "streamSeq": self.stream_seq,
                    },
                }

            @staticmethod
            def drain_completed_samples():
                return []

        mirror = FakeSimMirrorSubprocess()
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(enable_sim_mirror=True)
        probe.sim_mirror = mirror
        probe.sim_mirror_error = None
        probe.latest_sim_snapshot = None
        probe.sim_needs_policy_reset = True
        probe.policy_needs_reentry_reset = False
        probe.policy_reentry_reset_q = PROBE.np.zeros(
            7, dtype=PROBE.np.float32)
        probe.active_policy_profile = (
            PROBE.OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85)
        probe.active_policy_model = PROBE.GPU0_RMP_MODEL_V91
        probe.policy_generation = 7
        probe.recorder = SimpleNamespace(
            record_sim_control_samples=lambda _samples: None)
        probe._sim_compensation_options = lambda options: options
        probe._reset_policy_dq_adapter = lambda _reason: None
        options = SimpleNamespace(enable_sim_mirror=True)

        def step(*, source_seq, control_mode, policy_active):
            return probe._step_sim_mirror(
                PROBE.np.asarray([0.1, -0.2, 1.2]),
                PROBE.np.asarray([0.0, 0.0, -0.5]),
                ball_valid=True,
                age_s=0.0,
                options=options,
                source_seq=source_seq,
                source_time_s=source_seq * 0.005,
                ball_observation_seq=source_seq,
                camera_stamp_s=None,
                control_mode=control_mode,
                policy_active=policy_active,
            )

        with mock.patch.object(
                PROBE, "SimMirrorSubprocess", FakeSimMirrorSubprocess):
            first = step(
                source_seq=10, control_mode="policy", policy_active=True)
            feedback_held = step(
                source_seq=11,
                control_mode="policy_feedback_hold",
                policy_active=False,
            )
            ball_held = step(
                source_seq=12,
                control_mode="policy_ball_state_hold",
                policy_active=False,
            )
            resumed = step(
                source_seq=13, control_mode="policy", policy_active=True)

            self.assertEqual(mirror.reset_count, 1)
            self.assertEqual(mirror.step_source_seqs, [10, 13])
            self.assertEqual(first["sync"]["episode"], 1)
            self.assertEqual(feedback_held["sync"]["episode"], 1)
            self.assertEqual(ball_held["sync"]["episode"], 1)
            self.assertEqual(resumed["sync"]["episode"], 1)
            self.assertTrue(probe.policy_needs_reentry_reset)
            self.assertFalse(probe.sim_needs_policy_reset)

            # An actual strategy boundary still starts a new mirror episode.
            probe.sim_needs_policy_reset = True
            probe.policy_reentry_reset_q = PROBE.np.ones(
                7, dtype=PROBE.np.float32)
            restarted = step(
                source_seq=14, control_mode="policy", policy_active=True)

        self.assertEqual(mirror.reset_count, 2)
        self.assertEqual(restarted["sync"]["episode"], 2)


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


@dataclass(frozen=True)
class _CompensationOptions:
    compensation_parameter_target: str
    compensation_mode: str
    comp_delay_ms: float
    comp_actuator_tau_s: float
    comp_actuator_gain: float
    comp_mpc_beta: float
    comp_mpc_delay_scale: float
    comp_mpc_tau_scale: float
    comp_mpc_horizon_steps: int
    comp_mpc_tracking_weight: float
    comp_mpc_nominal_weight: float
    comp_mpc_delta_weight: float
    comp_mpc_max_delta_rad: float


class CompensationParameterRoutingTest(unittest.TestCase):

    @staticmethod
    def _probe():
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(
            compensation_mode="inverse_mpc",
            comp_delay_ms=72.0,
            comp_actuator_tau_s=0.074,
            comp_actuator_gain=1.0,
            comp_mpc_beta=1.2,
            comp_mpc_delay_scale=1.05,
            comp_mpc_tau_scale=0.75,
            comp_mpc_horizon_steps=6,
            comp_mpc_tracking_weight=1.0,
            comp_mpc_nominal_weight=0.25,
            comp_mpc_delta_weight=0.05,
            comp_mpc_max_delta_rad=0.0,
        )
        return probe

    @staticmethod
    def _live(target):
        return _CompensationOptions(
            compensation_parameter_target=target,
            compensation_mode="lead",
            comp_delay_ms=11.0,
            comp_actuator_tau_s=0.012,
            comp_actuator_gain=1.3,
            comp_mpc_beta=2.0,
            comp_mpc_delay_scale=1.2,
            comp_mpc_tau_scale=0.9,
            comp_mpc_horizon_steps=9,
            comp_mpc_tracking_weight=2.0,
            comp_mpc_nominal_weight=0.4,
            comp_mpc_delta_weight=0.1,
            comp_mpc_max_delta_rad=0.2,
        )

    def test_real_target_keeps_live_real_and_startup_sim_parameters(self):
        probe = self._probe()
        live = self._live("real")

        real = probe._real_compensation_options(live)
        sim = probe._sim_compensation_options(live)

        self.assertIs(real, live)
        self.assertEqual(sim.compensation_mode, "inverse_mpc")
        self.assertEqual(sim.comp_delay_ms, 72.0)
        self.assertEqual(sim.comp_actuator_tau_s, 0.074)

    def test_sim_target_keeps_live_sim_and_startup_real_parameters(self):
        probe = self._probe()
        probe.cfg.real_comp_delay_ms = 55.0
        probe.cfg.real_comp_actuator_tau_s = 0.158
        probe.cfg.real_comp_actuator_gain = 1.043
        probe.cfg.real_comp_mpc_beta = 1.175
        probe.cfg.real_comp_mpc_delay_scale = 1.176
        probe.cfg.real_comp_mpc_tau_scale = 0.854
        probe.cfg.real_comp_mpc_horizon_steps = 5
        probe.cfg.real_comp_mpc_tracking_weight = 1.0
        probe.cfg.real_comp_mpc_nominal_weight = 0.167
        probe.cfg.real_comp_mpc_delta_weight = 0.08
        probe.cfg.real_comp_mpc_max_delta_rad = 0.309
        live = self._live("sim")

        real = probe._real_compensation_options(live)
        sim = probe._sim_compensation_options(live)

        self.assertIs(sim, live)
        self.assertEqual(real.compensation_mode, "inverse_mpc")
        self.assertEqual(real.comp_delay_ms, 55.0)
        self.assertEqual(real.comp_actuator_tau_s, 0.158)
        self.assertEqual(real.comp_actuator_gain, 1.043)
        self.assertEqual(real.comp_mpc_beta, 1.175)
        self.assertEqual(real.comp_mpc_delay_scale, 1.176)
        self.assertEqual(real.comp_mpc_tau_scale, 0.854)
        self.assertEqual(real.comp_mpc_horizon_steps, 5)
        self.assertEqual(real.comp_mpc_nominal_weight, 0.167)
        self.assertEqual(real.comp_mpc_delta_weight, 0.08)
        self.assertEqual(real.comp_mpc_max_delta_rad, 0.309)

    def test_real_target_does_not_route_real_defaults_into_sim(self):
        probe = self._probe()
        probe.cfg.real_comp_delay_ms = 55.0
        probe.cfg.real_comp_actuator_tau_s = 0.158
        probe.cfg.real_comp_actuator_gain = 1.043
        probe.cfg.real_comp_mpc_beta = 1.175
        probe.cfg.real_comp_mpc_delay_scale = 1.176
        probe.cfg.real_comp_mpc_tau_scale = 0.854
        probe.cfg.real_comp_mpc_horizon_steps = 5
        probe.cfg.real_comp_mpc_nominal_weight = 0.167
        probe.cfg.real_comp_mpc_delta_weight = 0.08
        probe.cfg.real_comp_mpc_max_delta_rad = 0.309
        live = self._live("real")

        sim = probe._sim_compensation_options(live)

        self.assertEqual(sim.comp_delay_ms, 72.0)
        self.assertEqual(sim.comp_actuator_tau_s, 0.074)
        self.assertEqual(sim.comp_actuator_gain, 1.0)
        self.assertEqual(sim.comp_mpc_beta, 1.2)
        self.assertEqual(sim.comp_mpc_delay_scale, 1.05)
        self.assertEqual(sim.comp_mpc_tau_scale, 0.75)
        self.assertEqual(sim.comp_mpc_horizon_steps, 6)
        self.assertEqual(sim.comp_mpc_nominal_weight, 0.25)
        self.assertEqual(sim.comp_mpc_delta_weight, 0.05)
        self.assertEqual(sim.comp_mpc_max_delta_rad, 0.0)


class CpuSimMirrorActuatorModelTest(unittest.TestCase):

    def test_simcode4_compensation_uses_actual_joint_feedback(self):
        class _Compensator:
            enabled = True

            def __init__(self):
                self.feedback = None

            def step(self, *, q_cmd, qdot_cmd, qdd_cmd, applied_q):
                self.feedback = PROBE.np.asarray(applied_q).copy()
                return PROBE.np.asarray(q_cmd).copy()

        mirror = PROBE.RealBallSimMirror.__new__(PROBE.RealBallSimMirror)
        mirror._servo_planner_profile = True
        mirror.compensator = _Compensator()
        mirror.compensation_error = None
        mirror.arm_applied_q = PROBE.np.full(
            7, 1.0, dtype=PROBE.np.float32)
        mirror.policy = SimpleNamespace(
            latest_command_state=lambda: {
                "velocity": PROBE.np.zeros(7, dtype=PROBE.np.float32),
                "acceleration": PROBE.np.zeros(7, dtype=PROBE.np.float32),
            })
        mirror._ensure_compensator = lambda _options: None
        options = SimpleNamespace(enable_compensation=True)
        actual_q = PROBE.np.linspace(
            -0.3, 0.3, 7, dtype=PROBE.np.float32)

        mirror._apply_compensation(
            PROBE.np.zeros(7, dtype=PROBE.np.float32),
            options,
            actual_q=actual_q,
        )

        PROBE.np.testing.assert_array_equal(
            mirror.compensator.feedback, actual_q)

    def test_simcode3_compensation_keeps_modeled_applied_feedback(self):
        class _Compensator:
            enabled = True

            def __init__(self):
                self.feedback = None

            def step(self, *, q_cmd, qdot_cmd, qdd_cmd, applied_q):
                self.feedback = PROBE.np.asarray(applied_q).copy()
                return PROBE.np.asarray(q_cmd).copy()

        mirror = PROBE.RealBallSimMirror.__new__(PROBE.RealBallSimMirror)
        mirror._servo_planner_profile = False
        mirror.compensator = _Compensator()
        mirror.compensation_error = None
        modeled_q = PROBE.np.linspace(
            0.1, 0.7, 7, dtype=PROBE.np.float32)
        mirror.arm_applied_q = modeled_q.copy()
        mirror.policy = SimpleNamespace(
            latest_command_state=lambda: {
                "velocity": PROBE.np.zeros(7, dtype=PROBE.np.float32),
                "acceleration": PROBE.np.zeros(7, dtype=PROBE.np.float32),
            })
        mirror._ensure_compensator = lambda _options: None
        options = SimpleNamespace(enable_compensation=True)

        mirror._apply_compensation(
            PROBE.np.zeros(7, dtype=PROBE.np.float32),
            options,
            actual_q=PROBE.np.full(7, -2.0, dtype=PROBE.np.float32),
        )

        PROBE.np.testing.assert_array_equal(
            mirror.compensator.feedback, modeled_q)

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

    def test_simcode5_cpu_mirror_uses_per_joint_second_order_actuator(self):
        mirror = PROBE.RealBallSimMirror.__new__(PROBE.RealBallSimMirror)
        mirror.dt = 0.01
        mirror.cfg = SimpleNamespace(
            comp_delay_ms=0.0,
            comp_actuator_tau_s=0.0,
            comp_actuator_gain=1.0,
            policy_obs_dim=67,
        )
        mirror.policy = SimpleNamespace(
            arm_q_low=PROBE.np.full(7, -10.0, dtype=PROBE.np.float32),
            arm_q_high=PROBE.np.full(7, 10.0, dtype=PROBE.np.float32),
        )
        mirror._pure_actuator_profile = True
        mirror._servo_planner_profile = False
        mirror.pure_actuator_wn = PROBE.np.full(
            7, 20.0, dtype=PROBE.np.float32)
        mirror.pure_actuator_zeta = PROBE.np.full(
            7, 0.4, dtype=PROBE.np.float32)
        mirror.pure_actuator_gain = PROBE.np.ones(7, dtype=PROBE.np.float32)
        mirror.pure_actuator_delay_steps = PROBE.np.asarray(
            [0, 1, 0, 1, 0, 1, 0], dtype=PROBE.np.int32)
        mirror.warm_arm_q = PROBE.np.zeros(7, dtype=PROBE.np.float32)
        mirror.actuator_model_key = None
        mirror.command_buffer = None
        mirror.actuator_command_buffer = None
        mirror.arm_q_ref_active = None
        mirror.arm_actuator_q_ref_active = None
        mirror.arm_applied_q = None
        mirror.arm_applied_qvel = None

        _active, active_actuator, applied = mirror._run_actuator_model(
            PROBE.np.ones(7, dtype=PROBE.np.float32),
            PROBE.np.ones(7, dtype=PROBE.np.float32),
            mirror.cfg,
            seed_q=PROBE.np.zeros(7, dtype=PROBE.np.float32),
        )

        self.assertGreater(float(applied[0]), 0.0)
        self.assertEqual(float(applied[1]), 0.0)
        self.assertEqual(float(active_actuator[0]), 1.0)
        self.assertEqual(float(active_actuator[1]), 0.0)
        self.assertGreater(float(mirror.arm_applied_qvel[0]), 0.0)

    def test_simcode6_cpu_history_delay_ignores_legacy_ui_delay(self):
        mirror = PROBE.RealBallSimMirror.__new__(PROBE.RealBallSimMirror)
        mirror.dt = 0.005
        mirror.cfg = SimpleNamespace(
            comp_delay_ms=72.0,
            comp_actuator_tau_s=0.074,
            comp_actuator_gain=1.0,
            policy_obs_dim=67,
        )
        mirror.checkpoint_env_cfg = {"delay_min_ms": 45.0}
        mirror.policy = SimpleNamespace(
            arm_q_low=PROBE.np.full(7, -10.0, dtype=PROBE.np.float32),
            arm_q_high=PROBE.np.full(7, 10.0, dtype=PROBE.np.float32),
        )
        mirror._pure_actuator_profile = True
        mirror._servo_planner_profile = False
        mirror.pure_actuator_wn = PROBE.np.full(
            7, 20.0, dtype=PROBE.np.float32)
        mirror.pure_actuator_zeta = PROBE.np.full(
            7, 0.4, dtype=PROBE.np.float32)
        mirror.pure_actuator_gain = PROBE.np.ones(
            7, dtype=PROBE.np.float32)
        mirror.pure_actuator_delay_steps = PROBE.np.asarray(
            [9, 10, 9, 8, 7, 9, 11], dtype=PROBE.np.int32)
        mirror.warm_arm_q = PROBE.np.zeros(7, dtype=PROBE.np.float32)
        mirror.actuator_model_key = None
        mirror.command_buffer = None
        mirror.actuator_command_buffer = None
        mirror.arm_q_ref_active = None
        mirror.arm_actuator_q_ref_active = None
        mirror.arm_applied_q = None
        mirror.arm_applied_qvel = None

        mirror._reset_actuator_model(
            PROBE.np.zeros(7, dtype=PROBE.np.float32))

        self.assertEqual(mirror.actuator_delay_steps, 9)
        self.assertEqual(mirror.actuator_command_buffer_len, 16)


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


class PolicyWorkspaceSwitchRegressionTest(unittest.TestCase):

    @staticmethod
    def _probe(limits, initial_profile):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(
            ee_x_limits_m=limits[0], ee_y_limits_m=limits[1],
            ee_z_limits_m=limits[2], workspace_safety_mode="position")
        probe.base_ee_limits_m = limits
        probe.active_policy_profile = initial_profile
        probe.active_policy_model = "v85"
        probe.policy_generation = 0
        probe.command_stream_active = False
        probe.done_monitor = SimpleNamespace()
        probe.policy_controllers = {
            profile: SimpleNamespace(policy_model_id="v85", reset=mock.Mock())
            for profile in PROBE.RUNTIME_SWITCHABLE_POLICY_PROFILES}
        probe.controller = probe.policy_controllers[initial_profile]
        probe._set_policy_workspace_limits(initial_profile)
        probe._safety_limiter_reset_reference = mock.Mock(
            return_value=PROBE.np.zeros(7, dtype=PROBE.np.float32))
        probe._resolve_reset_initial_pose = mock.Mock(return_value=(
            PROBE.RESET_INITIAL_POSE_DEFAULT, PROBE.np.zeros(7)))
        for method in (
                "_sync_policy_to_safe_pose", "_reset_compensator_to_current_arm",
                "_reset_servo_target_planner", "_reset_policy_ball_sampling_contract",
                "_reset_policy_dq_adapter"):
            setattr(probe, method, mock.Mock())
        return probe

    def assert_workspace(self, probe, expected):
        self.assertEqual((probe.cfg.ee_x_limits_m, probe.cfg.ee_y_limits_m,
                          probe.cfg.ee_z_limits_m), expected)
        self.assertEqual((probe.done_monitor.ee_x_limits, probe.done_monitor.ee_y_limits,
                          probe.done_monitor.ee_z_limits), expected)

    def test_round_trip_restores_general_limits_for_every_non_fplc_profile(self):
        general = ((-.348, .263), (-.702, -.230), (.842, 1.462))
        small = ((-.1, .1), (-.6, -.4), (.95, 1.2))
        for initial in (PROBE.FPLC_PROFILES[0], PROBE.OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85):
            probe = self._probe(general, initial)
            # Two rounds catch cumulative intersection with the previous box.
            for _ in range(2):
                for profile in PROBE.RUNTIME_SWITCHABLE_POLICY_PROFILES:
                    with self.subTest(initial=initial, profile=profile):
                        probe._select_policy_profile(profile)
                        fplc = profile in PROBE.FPLC_PROFILES
                        self.assert_workspace(probe, small if fplc else general)
                        self.assertEqual(probe.base_ee_limits_m, general)
                        conditions = probe._workspace_translation_done_conditions(
                            PROBE.np.array([.2, -.49, 1.1]), True, "position")
                        self.assertEqual(conditions, {"ee_x_out_of_workspace": True} if fplc else {})

    def test_custom_general_limits_survive_a_fplc_round_trip(self):
        general = ((-.2, .22), (-.57, -.3), (1.02, 1.4))
        profile = PROBE.OBSERVATION_PROFILE_GPU1_QACC_FT_V153
        probe = self._probe(general, profile)
        probe._select_policy_profile(PROBE.FPLC_PROFILES[0])
        self.assert_workspace(probe, ((-.1, .1), (-.57, -.4), (1.02, 1.2)))
        probe._select_policy_profile(profile)
        self.assert_workspace(probe, general)

    def test_disabled_axis_stays_disabled_for_non_fplc_and_rejects_fplc(self):
        general = (None, (-.7, -.2), (.8, 1.5))
        probe = self._probe(general, PROBE.OBSERVATION_PROFILE_GPU1_QACC_FT_V153)
        probe._select_policy_profile(PROBE.OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85)
        self.assert_workspace(probe, general)
        with self.assertRaises(ValueError):
            probe._select_policy_profile(PROBE.FPLC_PROFILES[0])
        self.assertEqual(probe.active_policy_profile, PROBE.OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85)
        self.assert_workspace(probe, general)


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

    def test_soft_modes_share_hard_workspace_done_conditions(self):
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
        self.assertEqual(soft, hard)

    def test_soft_mode_routes_every_done_condition_to_reset(self):
        self.assertEqual(
            PROBE._workspace_done_action(
                "position_soft", {"ball_x_out_of_bounds": True}),
            "reset",
        )
        self.assertEqual(
            PROBE._workspace_done_action(
                "position_soft", {"ee_y_out_of_workspace": True}),
            "reset",
        )
        self.assertEqual(
            PROBE._workspace_done_action(
                "position", {"ball_x_out_of_bounds": True}),
            "parking",
        )
        self.assertEqual(
            PROBE._workspace_done_action(
                "position", {"ee_y_out_of_workspace": True}),
            "stop",
        )

    def test_publishing_workspace_safety_uses_joint_feedback_fk(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        command_q = PROBE.np.zeros(7)
        feedback_q = PROBE.np.zeros(7)
        feedback_q[:3] = [0.20, -0.60, 1.60]
        probe.controller = SimpleNamespace(
            arm_cmd_q=command_q.copy(),
            _compute_racket_pose_quat_base=lambda q: (
                PROBE.np.asarray(q[:3], dtype=PROBE.np.float64),
                PROBE.np.asarray([0.0, 0.0, 0.0, 1.0]),
                True,
            ),
        )
        probe.ros_bridge = SimpleNamespace(
            get_joint_feedback_with_age=lambda: (
                feedback_q.copy(), PROBE.np.zeros(7), 0.01, "joint_state"),
        )

        measured_pos, _quat, measured_ok = (
            probe._workspace_safety_racket_pose(
                SimpleNamespace(publish_commands=True)))
        dry_run_pos, _quat, dry_run_ok = (
            probe._workspace_safety_racket_pose(
                SimpleNamespace(publish_commands=False)))

        self.assertTrue(measured_ok)
        self.assertTrue(dry_run_ok)
        PROBE.np.testing.assert_allclose(measured_pos, feedback_q[:3])
        PROBE.np.testing.assert_allclose(dry_run_pos, command_q[:3])
        # The last call is dry-run; verify each path's selected source at call
        # time by invoking the hardware path once more.
        probe._workspace_safety_racket_pose(
            SimpleNamespace(publish_commands=True))
        self.assertEqual(
            probe.last_workspace_safety_pose_source, "joint_state_fk")

    def test_workspace_preflight_checks_all_translation_directions(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(
            workspace_safety_mode="position_soft",
            ee_x_limits_m=(-0.3, 0.1),
            ee_y_limits_m=(-0.5, -0.3),
            ee_z_limits_m=(0.7, 1.4),
        )

        self.assertEqual(
            probe._workspace_translation_done_conditions(
                PROBE.np.asarray([0.2, -0.6, 1.5]),
                True,
                "position_soft",
            ),
            {
                "ee_x_out_of_workspace": True,
                "ee_y_out_of_workspace": True,
                "ee_z_out_of_workspace": True,
            },
        )
        self.assertEqual(
            probe._workspace_translation_done_conditions(
                PROBE.np.asarray([0.0, -0.4, 1.0]),
                True,
                "position_soft",
            ),
            {},
        )
        for point, expected in (
            ([-0.31, -0.4, 1.0], "ee_x_out_of_workspace"),
            ([0.11, -0.4, 1.0], "ee_x_out_of_workspace"),
            ([0.0, -0.51, 1.0], "ee_y_out_of_workspace"),
            ([0.0, -0.29, 1.0], "ee_y_out_of_workspace"),
            ([0.0, -0.4, 0.69], "ee_z_out_of_workspace"),
            ([0.0, -0.4, 1.41], "ee_z_out_of_workspace"),
        ):
            with self.subTest(point=point):
                self.assertEqual(
                    probe._workspace_translation_done_conditions(
                        PROBE.np.asarray(point), True, "position_soft"),
                    {expected: True},
                )

    def test_workspace_preflight_upgrades_latched_ball_done_before_gates(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.controller = SimpleNamespace(use_valid_ball_obs_age=True)
        probe.cfg = SimpleNamespace(
            control_dt=0.005,
            ball_lost_timeout_s=0.6,
            workspace_safety_mode="position_soft",
            ee_x_limits_m=(-0.3, 0.1),
            ee_y_limits_m=(-0.5, -0.3),
            ee_z_limits_m=(0.7, 1.4),
        )
        probe.now_s = 1.0
        probe.last_valid_time_s = None
        probe.has_captured = True
        probe.failsafe_active = False
        probe.done_failsafe_active = True
        probe.done_failsafe_reason = "ball_x_out_of_bounds"
        probe.done_failsafe_conditions = {"ball_x_out_of_bounds": True}
        probe._release_incompatible_done_latch = lambda _mode: False
        probe._reset_recovery_active = lambda: False
        probe._workspace_safety_racket_pose = lambda _options: (
            PROBE.np.asarray([0.0, -0.51, 1.0]),
            PROBE.np.asarray([0.0, 0.0, 0.0, 1.0]),
            True,
        )
        probe._soft_workspace_reset_active = lambda: False
        calls = []
        probe._latch_soft_workspace_reset = (
            lambda reason, conditions: calls.append(
                ("latch", reason, dict(conditions))))
        reset_q = PROBE.np.zeros(7)
        probe._current_reset_pose_q = lambda: reset_q.copy()
        probe._sync_policy_to_safe_pose = (
            lambda q: calls.append(("sync", q.copy())))
        probe._hold_racket_tracker = (
            lambda reason, q: calls.append(("track", reason, q.copy())))
        probe._hold_juggle_controller = (
            lambda reason, q, **kwargs: calls.append(
                ("juggle", reason, q.copy(), kwargs)))
        probe._mark_policy_paused = lambda: calls.append(("paused",))
        options = SimpleNamespace(
            rl_use_valid_ball_obs_age=True,
            workspace_safety_mode="position_soft",
            control_mode="policy",
            enable_ball_release_gate=True,
        )

        result = probe._run_one_control_step(
            PROBE.np.zeros(3), PROBE.np.zeros(3), True, 0.0, options)

        self.assertEqual(result[0], "done_failsafe")
        self.assertEqual(result[1], "ee_y_out_of_workspace")
        self.assertEqual(
            result[2],
            {
                "ball_x_out_of_bounds": True,
                "ee_y_out_of_workspace": True,
            },
        )
        self.assertEqual(calls[0][0], "latch")
        self.assertIn(("paused",), calls)

    def test_switching_to_soft_keeps_arm_done_latched(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(workspace_safety_mode="predictive")
        probe.done_failsafe_active = True
        probe.done_failsafe_reason = "ee_x_out_of_workspace"
        probe.done_failsafe_conditions = {"ee_x_out_of_workspace": True}

        released = probe._release_incompatible_done_latch("position_soft")

        self.assertFalse(released)
        self.assertTrue(probe.done_failsafe_active)
        self.assertEqual(
            probe.done_failsafe_reason, "ee_x_out_of_workspace")
        self.assertEqual(
            probe.done_failsafe_conditions,
            {"ee_x_out_of_workspace": True},
        )

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
        self.assertEqual(
            probe.done_failsafe_reason, "ee_x_out_of_workspace")
        self.assertEqual(
            probe.done_failsafe_conditions,
            {
                "ee_x_out_of_workspace": True,
                "ball_too_low": True,
            },
        )

    def test_soft_workspace_done_publishes_slow_reset_without_limiter(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        reset_q = PROBE.np.linspace(-0.3, 0.3, 7)
        reset_seed_q = PROBE.np.linspace(0.1, 0.2, 7)
        published = []
        limiter_calls = []
        limiter_resets = []

        class RecordingLimiter:
            def reset(self, q):
                limiter_resets.append(PROBE.np.asarray(q).copy())

            def filter(self, q, *, profile="normal"):
                limiter_calls.append((PROBE.np.asarray(q).copy(), profile))
                return PROBE.np.asarray(q).copy()

        probe.reset_initial_pose_q = reset_q.copy()
        probe.safe_q = PROBE.np.zeros(7)
        probe.safety_limiter = RecordingLimiter()
        probe.ros_bridge = SimpleNamespace(
            get_joint_feedback=lambda: (
                reset_seed_q.copy(), PROBE.np.zeros(7)),
            publish_right_arm_command=lambda q: published.append(
                PROBE.np.asarray(q).copy()),
        )
        probe._reset_compensator_to_current_arm = lambda q=None: None
        probe.command_stream_active = True
        probe.done_failsafe_active = True
        probe.done_failsafe_reason = "ee_x_out_of_workspace"
        probe._latch_soft_workspace_reset(
            "ee_x_out_of_workspace",
            {"ee_x_out_of_workspace": True},
        )

        did_publish = probe._publish_control_command(
            PROBE.np.ones(7),
            SimpleNamespace(publish_commands=True),
        )

        self.assertTrue(did_publish)
        self.assertEqual(limiter_resets, [])
        self.assertEqual(limiter_calls, [])
        self.assertEqual(len(published), 1)
        first_reset_command = published[0]
        # The boundary-trigger tick holds the fresh seed.  The next control
        # tick advances exactly one trajectory sample; publication itself must
        # not consume a second sample.
        PROBE.np.testing.assert_allclose(
            first_reset_command, reset_seed_q, atol=1e-12)
        self.assertEqual(probe.slow_reset_trajectory.elapsed_s, 0.0)
        next_reset_command = probe._advance_slow_reset_command(advance=True)
        self.assertEqual(probe.slow_reset_trajectory.elapsed_s, 0.005)
        self.assertTrue(probe._publish_soft_workspace_reset_command(
            next_reset_command))
        self.assertEqual(probe.slow_reset_trajectory.elapsed_s, 0.005)
        PROBE.np.testing.assert_allclose(
            published[-1], next_reset_command, atol=1e-12)
        self.assertEqual(
            probe.last_command_arm["reason"],
            "soft_workspace_reset_latched",
        )

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

    def test_position_soft_runtime_matches_hard_command_path(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        published = []
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
        probe.safety_limiter = PROBE.RightArmCommandSafetyLimiter(
            PROBE.np.zeros(7), dt=0.005)

        def fail_if_called(*_args, **_kwargs):
            self.fail("soft runtime must not apply the old workspace filter")

        probe._filter_command_with_workspace_guard = fail_if_called
        probe._filter_command_with_workspace_soft_field = fail_if_called
        options = SimpleNamespace(
            publish_commands=True,
            workspace_safety_mode="position_soft",
        )

        target = PROBE.np.linspace(-0.2, 0.2, 7)
        probe._publish_control_command(target, options)

        self.assertEqual(len(published), 1)
        PROBE.np.testing.assert_allclose(published[0], target)

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

        def fake_compensation(q, _options, *, feedback_q_override=None):
            self.assertIsNone(feedback_q_override)
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

    def test_simcode4_plans_compensation_before_physical_plant(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        nominal = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        raw_compensated = PROBE.np.full(7, 0.40)
        planned = PROBE.np.full(7, 0.02)
        published = []
        planner_inputs = []
        observed_published = []
        advances = []
        applied_states = []
        probe.ros_bridge = SimpleNamespace(
            publish_right_arm_command=lambda q: published.append(
                PROBE.np.asarray(q).copy()))
        probe.cfg = SimpleNamespace(
            workspace_safety_mode="off",
            enable_safety_limiter=False,
            comp_delay_ms=72.0,
            comp_actuator_tau_s=0.074,
            simcode4_real_servo_planner=True,
        )
        probe.controller = SimpleNamespace(_checkpoint_env_cfg={
            "arm_servo_target_planner_before_actuator": True,
        })
        probe.safety_limiter = SimpleNamespace(
            observe=lambda _q: None,
        )
        probe.servo_target_planner = SimpleNamespace(
            step=lambda q: planner_inputs.append(
                PROBE.np.asarray(q).copy()) or planned.copy(),
        )

        def advance_applied_model():
            advances.append(True)
            return PROBE.np.full(7, 0.05)

        probe.compensator = SimpleNamespace(
            advance_applied_model=advance_applied_model,
            observe_applied_state=lambda q: applied_states.append(
                PROBE.np.asarray(q).copy()),
            observe_published=lambda q: observed_published.append(
                PROBE.np.asarray(q).copy()),
        )
        probe.compensation_was_enabled = False
        probe.command_stream_active = True
        probe.last_command_disable_hold = {}
        probe.workspace_soft_filter = {}

        def fake_compensation(q, _options, *, feedback_q_override=None):
            self.assertIsNone(feedback_q_override)
            probe.compensation_was_enabled = True
            return raw_compensated.copy()

        probe._apply_actuator_compensation = fake_compensation
        options = SimpleNamespace(
            publish_commands=True,
            workspace_safety_mode="off",
            policy_profile=PROBE.OBSERVATION_PROFILE_SIMCODE4_SERVO67,
            comp_delay_ms=72.0,
            comp_actuator_tau_s=0.074,
        )

        did_publish = probe._publish_control_command(
            nominal, options, apply_compensation=True)

        self.assertTrue(did_publish)
        self.assertEqual(advances, [])
        PROBE.np.testing.assert_allclose(planner_inputs, [raw_compensated])
        self.assertEqual(applied_states, [])
        PROBE.np.testing.assert_allclose(published[0], planned)
        PROBE.np.testing.assert_allclose(observed_published, [planned])
        self.assertFalse(probe.last_servo_drive_model_advanced)
        PROBE.np.testing.assert_allclose(
            probe.last_servo_planner_input_q, raw_compensated)
        PROBE.np.testing.assert_allclose(
            probe.last_servo_planner_output_q, planned)

    def test_simcode4_opt_in_planner_blocks_incompatible_frozen_checkpoint(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        published = []
        probe.ros_bridge = SimpleNamespace(
            publish_right_arm_command=lambda q: published.append(
                PROBE.np.asarray(q).copy()))
        probe.cfg = SimpleNamespace(
            workspace_safety_mode="off",
            enable_safety_limiter=False,
            comp_delay_ms=72.0,
            comp_actuator_tau_s=0.074,
            simcode4_real_servo_planner=True,
        )
        probe.controller = SimpleNamespace(_checkpoint_env_cfg={
            "arm_servo_target_tracking_planner": True,
            "arm_servo_target_planner_before_actuator": False,
        })
        probe.command_stream_active = True
        probe.compensation_was_enabled = True
        probe.last_simcode4_plant_order_matches_checkpoint = True
        probe.last_simcode4_publish_block_reason = ""
        options = SimpleNamespace(
            publish_commands=True,
            workspace_safety_mode="off",
            policy_profile=PROBE.OBSERVATION_PROFILE_SIMCODE4_SERVO67,
            comp_delay_ms=72.0,
            comp_actuator_tau_s=0.074,
        )

        did_publish = probe._publish_control_command(
            PROBE.np.zeros(7), options, apply_compensation=True)

        self.assertFalse(did_publish)
        self.assertEqual(published, [])
        self.assertFalse(probe.compensation_was_enabled)
        self.assertFalse(
            probe.last_simcode4_plant_order_matches_checkpoint)
        self.assertIn(
            "requires a deployable V6 checkpoint",
            probe.last_simcode4_publish_block_reason,
        )

    def test_simcode4_default_planner_off_publishes_frozen_checkpoint(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        nominal = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        compensated = PROBE.np.full(7, 0.12)
        published = []
        observed = []
        advanced = []
        probe.ros_bridge = SimpleNamespace(
            publish_right_arm_command=lambda q: published.append(
                PROBE.np.asarray(q).copy()))
        probe.cfg = SimpleNamespace(
            workspace_safety_mode="off",
            enable_safety_limiter=False,
            comp_delay_ms=72.0,
            comp_actuator_tau_s=0.074,
            simcode4_real_servo_planner=False,
        )
        probe.controller = SimpleNamespace(_checkpoint_env_cfg={
            "arm_servo_target_tracking_planner": True,
            "arm_servo_target_planner_before_actuator": False,
        })
        probe.safety_limiter = SimpleNamespace(observe=lambda _q: None)
        probe.servo_target_planner = SimpleNamespace(
            step=lambda _q: self.fail(
                "default simcode4 real path must not call servo planner"))
        probe.compensator = SimpleNamespace(
            observe_published=lambda q: observed.append(
                PROBE.np.asarray(q).copy()),
            advance_applied_model=lambda: advanced.append(True),
        )
        probe.compensation_was_enabled = False
        probe.command_stream_active = True
        probe.last_command_disable_hold = {}
        probe.workspace_soft_filter = {}

        def fake_compensation(q, _options, *, feedback_q_override=None):
            self.assertIsNone(feedback_q_override)
            probe.compensation_was_enabled = True
            return compensated.copy()

        probe._apply_actuator_compensation = fake_compensation
        options = SimpleNamespace(
            publish_commands=True,
            workspace_safety_mode="off",
            policy_profile=PROBE.OBSERVATION_PROFILE_SIMCODE4_SERVO67,
            comp_delay_ms=72.0,
            comp_actuator_tau_s=0.074,
        )

        did_publish = probe._publish_control_command(
            nominal, options, apply_compensation=True)

        self.assertTrue(did_publish)
        PROBE.np.testing.assert_allclose(published, [compensated])
        PROBE.np.testing.assert_allclose(observed, [compensated])
        self.assertEqual(advanced, [])
        self.assertIsNone(probe.last_servo_planner_input_q)
        self.assertIsNone(probe.last_servo_planner_output_q)
        self.assertEqual(probe.last_simcode4_publish_block_reason, "")
        self.assertFalse(
            probe.last_simcode4_plant_order_matches_checkpoint)

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

    def test_disabling_stream_never_publishes_stale_feedback_as_new_hold(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        feedback = PROBE.np.linspace(0.0, 0.3, 7)
        published = []
        probe.ros_bridge = SimpleNamespace(
            get_joint_feedback_with_age=lambda: (
                feedback.copy(), PROBE.np.zeros(7), 0.20, "test"),
            publish_right_arm_command=lambda q: published.append(
                PROBE.np.asarray(q).copy()),
        )
        probe.safety_limiter = PROBE.RightArmCommandSafetyLimiter(
            PROBE.np.zeros(7), dt=0.005)
        probe.command_stream_active = True
        probe.last_command_disable_hold = {}
        probe._reset_compensator_to_current_arm = lambda q=None: None

        did_publish = probe._publish_hold_on_command_disable()

        self.assertFalse(did_publish)
        self.assertEqual(published, [])
        self.assertFalse(probe.command_stream_active)
        self.assertFalse(probe.last_command_disable_hold["published"])
        self.assertFalse(probe.last_command_disable_hold["feedbackFresh"])
        self.assertAlmostEqual(
            probe.last_command_disable_hold["feedbackAgeS"], 0.20)


class ResetTerminalHoldRegressionTest(unittest.TestCase):

    @staticmethod
    def _probe(feedback_q, feedback_dq):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.cfg = SimpleNamespace(
            control_dt=0.005,
            ros_publish_commands=True,
            reset_velocity_limit_deg_s=10.0,
            reset_acceleration_limit_deg_s2=25.0,
            safety_parking_velocity_scale=0.08,
            safety_parking_acceleration_scale=0.06,
        )
        probe.ros_bridge = SimpleNamespace(
            get_joint_feedback_with_age=lambda: (
                PROBE.np.asarray(feedback_q, dtype=PROBE.np.float64).copy(),
                PROBE.np.asarray(feedback_dq, dtype=PROBE.np.float64).copy(),
                0.001,
                "joint_state",
            ),
        )
        probe.safety_limiter = SimpleNamespace(
            last_cmd=PROBE.np.zeros(7, dtype=PROBE.np.float64),
            last_vel=PROBE.np.zeros(7, dtype=PROBE.np.float64),
            dt=0.005,
        )
        probe.slow_reset_trajectory = PROBE.SlowResetTrajectory(
            dt_s=0.005,
            velocity_limit_deg_s=10.0,
            acceleration_limit_deg_s2=25.0,
        )
        probe.runtime_options = None
        probe.command_stream_active = True
        probe.policy_switch_publish_rearm_required = False
        probe.publish_rearm_required = False
        probe.publish_rearm_off_seen = False
        probe.publish_rearm_reason = ""
        probe.last_publish_arm_token = ""
        probe.reset_start_publish_arm_token = ""
        probe.reset_policy_arm_required = False
        probe.reset_policy_arm_reason = ""
        probe.has_captured = False
        probe.has_released = False
        probe.release_candidate_ticks = 0
        probe.release_samples = []
        probe.web_drag_ball_pos = PROBE.np.asarray(
            [-0.15, -0.40, 1.20], dtype=PROBE.np.float32)
        probe.reset_workspace_bypass_until_wall = 0.0
        probe.reset_workspace_bypass_target_q = None
        probe.soft_workspace_reset_latched = True
        probe.soft_workspace_reset_reason = "ee_y_out_of_workspace"
        probe.soft_workspace_reset_conditions = {
            "ee_y_out_of_workspace": True,
        }
        probe.done_failsafe_active = True
        probe.done_failsafe_reason = "ee_y_out_of_workspace"
        probe.done_failsafe_conditions = {
            "ee_y_out_of_workspace": True,
        }
        probe.last_control_mode = "reset"
        probe._sync_policy_to_safe_pose = mock.Mock()
        probe._mark_policy_paused = mock.Mock()
        return probe

    def test_completed_reset_requires_fresh_send_command_token(self):
        @dataclass(frozen=True)
        class RuntimeStub:
            publish_commands: bool

        target = PROBE.np.asarray(
            [0.10, -0.77, 0.54, 1.70, 0.86, -0.21, 0.25],
            dtype=PROBE.np.float64,
        )
        probe = self._probe(target, PROBE.np.zeros(7))
        probe.last_publish_arm_token = "arm-before-reset"
        probe.runtime_options = RuntimeStub(publish_commands=True)
        probe._start_supervised_reset(
            target, target, reason="ee_y_out_of_workspace")

        required_ticks = int(PROBE.RESET_SETTLE_DURATION_S / 0.005)
        for _ in range(required_ticks):
            command = probe._advance_slow_reset_command(advance=True)
            PROBE.np.testing.assert_allclose(command, target, atol=1e-12)

        self.assertTrue(probe.reset_command_trajectory_complete)
        self.assertTrue(probe.reset_settled)
        self.assertTrue(probe.reset_hold_latched)
        self.assertTrue(probe._reset_recovery_active())
        self.assertTrue(probe.reset_policy_arm_required)
        self.assertFalse(probe.publish_rearm_required)
        self.assertFalse(probe.runtime_options.publish_commands)

        # Replaying the old checked state cannot leave reset hold.
        self.assertTrue(probe._consume_reset_policy_arm_request(
            {
                "publishArmToken": "arm-before-reset",
            },
            publish_requested=True,
            switch_requested=False,
        ))
        self.assertTrue(probe.reset_hold_latched)
        self.assertTrue(probe.command_stream_active)

        # One real Send-command enable gesture creates a new token, releases
        # reset, and forces a measured-position arm hold before policy.
        self.assertFalse(probe._consume_reset_policy_arm_request(
            {
                "publishArmToken": "arm-after-reset",
            },
            publish_requested=True,
            switch_requested=False,
        ))
        self.assertFalse(probe.reset_hold_latched)
        self.assertFalse(probe.soft_workspace_reset_latched)
        self.assertFalse(probe.command_stream_active)
        self.assertFalse(probe.has_released)
        probe._mark_policy_paused.assert_called_once_with()

    def test_send_enabled_after_reset_start_runs_policy_after_settle(self):
        @dataclass(frozen=True)
        class RuntimeStub:
            publish_commands: bool

        target = PROBE.np.asarray(
            [0.10, -0.77, 0.54, 1.70, 0.86, -0.21, 0.25],
            dtype=PROBE.np.float64,
        )
        probe = self._probe(target, PROBE.np.zeros(7))
        probe.last_publish_arm_token = "arm-before-reset"
        probe.runtime_options = RuntimeStub(publish_commands=True)
        probe._start_supervised_reset(
            target, target, reason="operator_reset")
        self.assertFalse(probe._consume_reset_policy_arm_request(
            {"publishArmToken": "arm-after-reset"},
            publish_requested=True,
            switch_requested=False,
        ))
        required_ticks = int(PROBE.RESET_SETTLE_DURATION_S / 0.005)
        for _ in range(required_ticks):
            probe._advance_slow_reset_command(advance=True)

        self.assertFalse(probe.reset_hold_latched)
        self.assertFalse(probe.reset_policy_arm_required)
        self.assertTrue(probe.runtime_options.publish_commands)
        self.assertFalse(probe.command_stream_active)
        self.assertFalse(probe.has_released)

    def test_feedback_reanchor_preserves_post_reset_arm_token_edge(self):
        start = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        target = PROBE.np.full(7, 0.2, dtype=PROBE.np.float64)
        probe = self._probe(start, PROBE.np.zeros(7))
        probe.last_publish_arm_token = "arm-before-reset"
        probe._start_supervised_reset(
            start, target, reason="operator_reset")

        # Enabling Send after reset starts updates the latest token.  Arming
        # the command stream re-anchors the in-flight trajectory at measured
        # feedback, but it is not a new reset and must not consume that edge.
        probe.last_publish_arm_token = "arm-after-reset"
        probe._start_supervised_reset(
            PROBE.np.full(7, 0.05, dtype=PROBE.np.float64),
            target,
            reason="operator_reset",
            new_reset_cycle=False,
        )

        self.assertEqual(
            probe.reset_start_publish_arm_token, "arm-before-reset")
        self.assertEqual(probe.last_publish_arm_token, "arm-after-reset")

    def test_tracked_command_rearm_preserves_reset_trajectory_elapsed(self):
        start = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        target = PROBE.np.full(7, 1e-6, dtype=PROBE.np.float64)
        probe = self._probe(start, PROBE.np.zeros(7))
        probe._start_supervised_reset(
            start, target, reason="operator_reset")
        for _ in range(200):
            probe.slow_reset_trajectory.step()
        status_before = probe.slow_reset_trajectory.status()

        reanchored = probe._resume_or_reanchor_active_reset(
            status_before.position_rad)

        self.assertFalse(reanchored)
        self.assertAlmostEqual(
            probe.slow_reset_trajectory.elapsed_s, 1.0, places=9)
        for _ in range(200):
            _q, _dq, done = probe.slow_reset_trajectory.step()
        self.assertTrue(done)
        self.assertFalse(probe.slow_reset_trajectory.active)

    def test_diverged_command_rearm_replans_without_new_two_second_minimum(self):
        start = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        target = PROBE.np.full(7, 0.10, dtype=PROBE.np.float64)
        probe = self._probe(start, PROBE.np.zeros(7))
        probe.last_publish_arm_token = "arm-before-reset"
        probe._start_supervised_reset(
            start, target, reason="operator_reset")
        for _ in range(20):
            probe.slow_reset_trajectory.step()
        feedback = probe.slow_reset_trajectory.status().position_rad.copy()
        feedback += 0.02

        reanchored = probe._resume_or_reanchor_active_reset(feedback)

        self.assertTrue(reanchored)
        self.assertEqual(
            probe.reset_start_publish_arm_token, "arm-before-reset")
        self.assertLess(
            float(PROBE.np.max(probe.slow_reset_trajectory._duration_s)),
            2.0,
        )

    def test_reset_does_not_advance_while_arm_is_fast_or_publish_is_off(self):
        start = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        target = PROBE.np.full(7, 0.4, dtype=PROBE.np.float64)
        feedback_dq = PROBE.np.zeros(7, dtype=PROBE.np.float64)
        feedback_dq[0] = PROBE.np.deg2rad(25.0)
        probe = self._probe(start, feedback_dq)
        probe._start_supervised_reset(start, target, reason="safety_reset")

        held = probe._advance_slow_reset_command(advance=True)
        PROBE.np.testing.assert_allclose(held, start, atol=1e-12)
        self.assertTrue(probe.reset_tracking_paused)
        self.assertEqual(probe.slow_reset_trajectory.elapsed_s, 0.0)

        held = probe._advance_slow_reset_command(advance=False)
        PROBE.np.testing.assert_allclose(held, start, atol=1e-12)
        self.assertEqual(probe.slow_reset_trajectory.elapsed_s, 0.0)

    def test_parking_transition_reanchors_at_feedback_not_stale_history(self):
        feedback_q = PROBE.np.linspace(-0.2, 0.2, 7)
        probe = self._probe(feedback_q, PROBE.np.zeros(7))
        probe.safety_limiter.last_cmd = PROBE.np.full(7, -1.0)
        probe.safety_limiter.last_vel = PROBE.np.full(7, 5.0)
        probe.last_published_command_profile = "normal"

        command = probe._limiter_free_parking_candidate(
            PROBE.np.full(7, 1.0))

        acceleration_limit = PROBE.np.deg2rad(PROBE.np.asarray(
            PROBE.RightArmCommandSafetyLimiter.ACC_LIMIT_DEG_S2,
            dtype=PROBE.np.float64,
        )) * 0.06
        self.assertTrue(PROBE.np.all(
            PROBE.np.abs(command - feedback_q)
            <= acceleration_limit * 0.005 ** 2 + 1e-12))
        for method_name in ("filter", "reset", "observe"):
            self.assertFalse(hasattr(probe.safety_limiter, method_name))


class ControlSubprocessWorkspaceAckTest(unittest.TestCase):

    def test_background_client_cannot_overwrite_active_control_owner(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.command_owner_session_id = "active-tab"
        probe.last_command_ownership = {}

        accepted = probe._accept_control_client_payload({
            "clientSessionId": "background-tab",
            "publishCommands": False,
        })

        self.assertFalse(accepted)
        self.assertEqual(probe.command_owner_session_id, "active-tab")
        self.assertEqual(
            probe.last_command_ownership["reason"],
            "non_owner_control_request_ignored",
        )

    def test_explicit_reset_transfers_control_owner(self):
        probe = PROBE.PolicySafetyProbe.__new__(PROBE.PolicySafetyProbe)
        probe.command_owner_session_id = "old-tab"
        probe.last_command_ownership = {}

        accepted = probe._accept_control_client_payload(
            {
                "clientSessionId": "reset-tab",
                "publishCommands": False,
            },
            allow_reset_takeover=True,
        )

        self.assertTrue(accepted)
        self.assertEqual(probe.command_owner_session_id, "reset-tab")

    def test_external_mpc_juggle_request_is_forced_to_nonpublishing_policy(self):
        payload, blocked = PROBE._sanitize_external_control_payload({
            "controlMode": "mpc_juggle",
            "publishCommands": True,
        })

        self.assertTrue(blocked)
        self.assertEqual(payload["controlMode"], "policy")
        self.assertFalse(payload["publishCommands"])
        self.assertNotIn("mpc_juggle", PROBE.SELECTABLE_CONTROL_MODES)

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

    def test_snapshot_publish_arm_token_reads_control_ack(self):
        snapshot = {"runtime": {"publishArmToken": "arm-after-reset"}}

        applied = PROBE.ControlSubprocessProxy._snapshot_publish_arm_token(
            snapshot)

        self.assertEqual(applied, "arm-after-reset")

    def test_step_marks_new_arm_pending_when_snapshot_has_old_token(self):
        proxy = PROBE.ControlSubprocessProxy.__new__(
            PROBE.ControlSubprocessProxy)
        proxy.cfg = SimpleNamespace(
            workspace_safety_mode="position_soft",
            ros_publish_commands=True,
            ros_bridge=True,
            ball_estimator_enable_prediction=False,
            default_control_mode="policy",
            policy_profile=PROBE.OBSERVATION_PROFILE_LEGACY_BUG_COMPAT,
            racket_tracking_height_offset_m=0.25,
            racket_tracking_use_mpc=False,
        )
        proxy.lock = threading.Lock()
        proxy.latest_config = {}
        proxy.latest_recording_status = {"active": False}
        proxy.last_error = None
        proxy.latest_snapshot = {
            "runtime": {
                "workspaceSafetyMode": "position_soft",
                "publishCommands": True,
                "publishArmToken": "arm-before-reset",
                "estimatorPrediction": False,
                "controlMode": "policy",
                "racketTrackingHeightOffsetM": 0.25,
                "racketTrackingUseMpc": False,
                "activePolicyProfile": (
                    PROBE.OBSERVATION_PROFILE_LEGACY_BUG_COMPAT),
                "activePolicyModel": "v85",
            },
            "resetRecovery": {"policyArmRequired": True},
        }
        proxy._maybe_send_step_payload = mock.Mock(return_value=False)
        proxy._drain_status = mock.Mock()

        snapshot = proxy.step({
            "publishCommands": True,
            "publishArmToken": "arm-after-reset",
            "workspaceSafetyMode": "position_soft",
            "controlMode": "policy",
            "policyProfile": (
                PROBE.OBSERVATION_PROFILE_LEGACY_BUG_COMPAT),
            "racketTrackingHeightOffsetM": 0.25,
            "racketTrackingUseMpc": False,
            "estimatorPrediction": False,
        })

        self.assertTrue(snapshot["commandPublishPending"])
        self.assertEqual(
            snapshot["commandPublishArmTokenRequested"], "arm-after-reset")
        self.assertEqual(
            snapshot["commandPublishArmTokenApplied"], "arm-before-reset")

    def test_proxy_does_not_queue_background_client_control_payload(self):
        proxy = PROBE.ControlSubprocessProxy.__new__(
            PROBE.ControlSubprocessProxy)
        proxy.cfg = SimpleNamespace(
            workspace_safety_mode="position_soft",
            ros_publish_commands=True,
            ros_bridge=True,
            ball_estimator_enable_prediction=False,
            default_control_mode="policy",
            policy_profile=PROBE.OBSERVATION_PROFILE_LEGACY_BUG_COMPAT,
            racket_tracking_height_offset_m=0.25,
            racket_tracking_use_mpc=False,
        )
        proxy.lock = threading.Lock()
        proxy.latest_config = {}
        proxy.latest_recording_status = {"active": False}
        proxy.last_error = None
        proxy.latest_snapshot = {
            "runtime": {
                "workspaceSafetyMode": "position_soft",
                "publishCommands": True,
                "publishArmToken": "owner-arm",
                "estimatorPrediction": False,
                "controlMode": "policy",
                "racketTrackingHeightOffsetM": 0.25,
                "racketTrackingUseMpc": False,
                "activePolicyProfile": (
                    PROBE.OBSERVATION_PROFILE_LEGACY_BUG_COMPAT),
                "activePolicyModel": "v85",
            },
            "commandOwnership": {"ownerSessionId": "active-tab"},
        }
        proxy._maybe_send_step_payload = mock.Mock(return_value=True)
        proxy._drain_status = mock.Mock()

        snapshot = proxy.step({
            "clientSessionId": "background-tab",
            "publishCommands": False,
            "workspaceSafetyMode": "position_soft",
            "controlMode": "policy",
            "policyProfile": (
                PROBE.OBSERVATION_PROFILE_LEGACY_BUG_COMPAT),
            "racketTrackingHeightOffsetM": 0.25,
            "racketTrackingUseMpc": False,
            "estimatorPrediction": False,
        })

        proxy._maybe_send_step_payload.assert_not_called()
        self.assertTrue(snapshot["commandRequestIgnored"])
        self.assertEqual(
            snapshot["commandOwnership"]["reason"],
            "non_owner_control_request_ignored",
        )


if __name__ == "__main__":
    unittest.main()
