#!/usr/bin/env python3
"""Regression tests for legacy and simcode2/3/4/5 deployment contracts."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import sys
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.mjx_rl_policy import (  # noqa: E402
    MjxRLPolicyController,
    Simcode2IdealPd67PolicyController,
    Simcode3Actuator67PolicyController,
    Simcode4Servo67PolicyController,
    Simcode5PureActuator67PolicyController,
    inspect_mjx_checkpoint_metadata,
)
from pingpong_controller.rl_policy import (  # noqa: E402
    OBSERVATION_PROFILE_LEGACY_BUG_COMPAT,
    OBSERVATION_PROFILE_SIMCODE2_IDEALPD67,
    OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67,
    OBSERVATION_PROFILE_SIMCODE4_SERVO67,
    OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67,
    RLPolicyController,
    uses_corrected_67d_observation,
)
from pingpong_controller.sim_mirror import (  # noqa: E402
    RealBallMjxJaxSimMirror,
    SimMirrorConfig,
    sim_mirror_config_for_policy_profile,
)
from pingpong_controller.safety_limiter import (  # noqa: E402
    RightArmCommandSafetyLimiter,
    ServoTargetTrackingPlanner,
)


LEGACY_MODEL = (
    PACKAGE_ROOT / "pingpong_controller" / "outputs" / "rl_sim"
    / "inverse_compensation" / "18_stage4g_strong_contact_dr_inverse_mpc.pkl"
)
LEGACY_POLICY_XML = (
    PACKAGE_ROOT / "pingpong_controller" / "models" / "moz1_pd.xml"
)
LEGACY_SIM_XML = (
    PACKAGE_ROOT / "pingpong_controller" / "tools" / "rl_sim"
    / "moz1_pd.xml"
)
SIMCODE2_ARTIFACT_DIR = (
    REPO_ROOT / "simcode2" / "pingpong_moz1-main"
    / "pingpong_controller" / "outputs" / "rl_sim"
    / "goal_d455_autolaunch_idealpd67_viewdense_v1_seed976_gpu1_resume_launch14_best55_20260722"
    / "launch19_best2040_video_gpu1_seed20260722"
)
SIMCODE2_MODEL = (
    SIMCODE2_ARTIFACT_DIR / "policy_launch19_best_u2040_frozen.pkl"
)
SIMCODE2_XML = (
    REPO_ROOT / "simcode2" / "pingpong_moz1-main"
    / "pingpong_controller" / "tools" / "rl_sim" / "moz1_pd.xml"
)
SIMCODE2_SHA256 = (
    "357fe60d51594304ebfdcf936c7f9f0b09bb0850dc09352d24492e82c37e86a0"
)
SIMCODE3_MODEL = (
    REPO_ROOT / "simcode3" / "pingpong_moz1-main"
    / "pingpong_controller" / "outputs" / "rl_sim"
    / "goal_d455_gpu1_resume8_launch19_survival_lr1e4_20260727"
    / "mjx_curriculum_best.pkl"
)
SIMCODE3_XML = (
    REPO_ROOT / "simcode3" / "pingpong_moz1-main"
    / "pingpong_controller" / "tools" / "rl_sim" / "moz1_pd.xml"
)
SIMCODE4_MODEL = (
    REPO_ROOT / "simcode4" / "pingpong_moz1-main"
    / "pingpong_controller" / "outputs" / "rl_sim"
    / "goal_d455_gpu0_obsres2mm_servo_submission_20260729"
    / "mjx_curriculum_best.pkl"
)
SIMCODE4_XML = (
    REPO_ROOT / "simcode4" / "pingpong_moz1-main"
    / "pingpong_controller" / "tools" / "rl_sim" / "moz1_pd.xml"
)
SIMCODE4_TRACE_DIR = (
    SIMCODE4_MODEL.parent / "video"
)
SIMCODE4_SHA256 = (
    "ed495f4445c21b4af7fd1acba420ba3e76e68a86dfdf2b35a7201acd641559c4"
)
SIMCODE5_MODEL = (
    REPO_ROOT / "simcode5" / "pingpong_moz1-main"
    / "pingpong_controller" / "outputs" / "rl_sim"
    / "goal_d455_gpu1_pure_stable_v5_recovery_20260802"
    / "mjx_curriculum_best.pkl"
)
SIMCODE5_XML = (
    REPO_ROOT / "simcode5" / "pingpong_moz1-main"
    / "pingpong_controller" / "tools" / "rl_sim" / "moz1_pd.xml"
)


class _Space:
    def __init__(self, shape):
        self.shape = shape


class _Model:
    def __init__(self, obs_dim: int, act_dim: int):
        self.observation_space = _Space((obs_dim,))
        self.action_space = _Space((act_dim,))


class _FixedActionController(RLPolicyController):
    """Model-free controller used to exercise the observation state machine."""

    def __init__(self, *args, fixed_action=None, **kwargs):
        self.fixed_action = np.asarray(
            np.zeros(7) if fixed_action is None else fixed_action,
            dtype=np.float32,
        ).reshape(7)
        super().__init__(*args, **kwargs)

    def _load_model(self, model_path: str, device: str) -> None:
        del model_path, device
        self.model = _Model(self._expected_obs_dim, self.n_arm)

    def _policy_action(self, obs: np.ndarray) -> np.ndarray:
        del obs
        return self.fixed_action.copy()


def _controller(profile: str, *, action=None) -> _FixedActionController:
    return _FixedActionController(
        "unused",
        dt=0.005,
        fixed_action=action,
        enable_delay_conditioning=True,
        observation_profile=profile,
        ball_obs_age_clip_s=0.5,
        arm_action_limiter=True,
    )


class ObservationProfileContractTest(unittest.TestCase):

    def _observation(self, controller: RLPolicyController) -> np.ndarray:
        racket_pos = np.asarray([0.0, 0.0, 0.50], dtype=np.float32)
        controller._prev_racket_pos_base = np.asarray(
            [0.0, 0.0, 0.49], dtype=np.float32)
        controller._compute_racket_pose_base = lambda _q: (
            racket_pos.copy(), True)
        q = np.zeros(7, dtype=np.float32)
        dq = np.zeros(7, dtype=np.float32)
        dq[0] = 2.0
        return controller._build_obs(
            np.asarray([0.20, 0.0, 1.10], dtype=np.float32),
            np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
            0.10,
            0.005,
            arm_q=q,
            arm_dq=dq,
        )

    def test_legacy_profile_preserves_zero_racket_velocity_and_rel_x_bug(self):
        controller = _controller(OBSERVATION_PROFILE_LEGACY_BUG_COMPAT)
        controller._compute_racket_pose_jacobians_base = mock.Mock(
            side_effect=AssertionError(
                "legacy profile must retain its FK-difference path"))

        obs = self._observation(controller)

        np.testing.assert_array_equal(obs[29:32], np.zeros(3, dtype=np.float32))
        legacy_contact = controller._estimate_contact_time(
            rel_z=float(obs[32]),
            ball_vz=float(obs[25]),
            racket_vz=0.0,
            ball_obs_age_s=0.0,
            max_time=0.5,
            lost_ball_timeout_ms=150.0,
        )
        self.assertAlmostEqual(float(obs[65]), legacy_contact, places=6)

        age_s = 0.10
        rel_z_now = float(obs[34]) + float(obs[25]) * age_s - 0.5 * 9.81 * age_s**2
        corrected_contact = controller._estimate_contact_time(
            rel_z_now,
            float(obs[25]) - 9.81 * age_s,
            2.0,
            ball_obs_age_s=age_s,
            max_time=0.5,
            lost_ball_timeout_ms=350.0,
        )
        self.assertGreater(abs(float(obs[65]) - corrected_contact), 1e-3)

    def test_ideal_profile_uses_causal_racket_velocity_rel_z_and_age(self):
        controller = _controller(OBSERVATION_PROFILE_SIMCODE2_IDEALPD67)

        obs = self._observation(controller)

        np.testing.assert_allclose(
            obs[29:32], np.asarray([0.0, 0.0, 2.0]), atol=3e-6)
        self.assertAlmostEqual(float(obs[34]), 0.60, places=6)
        self.assertAlmostEqual(float(obs[49]), 0.20, places=6)
        self.assertAlmostEqual(float(obs[50]), 1.0, places=6)

        age_s = 0.10
        rel_z_now = float(obs[34]) + float(obs[25]) * age_s - 0.5 * 9.81 * age_s**2
        ball_vz_now = float(obs[25]) - 9.81 * age_s
        expected_contact = controller._estimate_contact_time(
            rel_z_now,
            ball_vz_now,
            float(obs[31]),
            ball_obs_age_s=age_s,
            max_time=controller.max_contact_time_s,
            contact_height_offset=controller.contact_height_offset_m,
            lost_ball_timeout_ms=controller.contact_lost_ball_timeout_s * 1000.0,
        )
        self.assertAlmostEqual(float(obs[65]), expected_contact, places=6)
        self.assertAlmostEqual(
            float(obs[66]), expected_contact - 0.072, places=6)

    def test_ideal_profile_without_sample_stamp_keeps_training_tick_difference(self):
        controller = _controller(OBSERVATION_PROFILE_SIMCODE2_IDEALPD67)
        q = np.linspace(-0.3, 0.3, 7, dtype=np.float32)
        racket_pos = np.asarray([0.12, -0.34, 0.87], dtype=np.float32)
        controller._prev_racket_pos_base = np.asarray(
            [0.11, -0.32, 0.84], dtype=np.float32)
        controller._compute_racket_pose_base = mock.Mock(
            return_value=(racket_pos.copy(), True))

        obs = controller._build_obs(
            np.asarray([0.20, 0.0, 1.10], dtype=np.float32),
            np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
            0.0,
            0.005,
            arm_q=q,
            arm_dq=None,
        )

        controller._compute_racket_pose_base.assert_called_once()
        np.testing.assert_array_equal(
            controller._compute_racket_pose_base.call_args.args[0], q)
        np.testing.assert_array_equal(obs[26:29], racket_pos)
        np.testing.assert_allclose(
            obs[29:32],
            np.asarray([2.0, -4.0, 6.0], dtype=np.float32),
            rtol=0.0,
            atol=1e-5,
        )

    def test_ideal_profile_source_sample_difference_and_repeat_hold(self):
        controller = _controller(OBSERVATION_PROFILE_SIMCODE2_IDEALPD67)
        q = np.zeros(7, dtype=np.float32)
        racket_pos = np.asarray([0.0, 0.0, 0.50], dtype=np.float32)
        controller._compute_racket_pose_base = mock.Mock(
            side_effect=lambda _q: (racket_pos.copy(), True))

        def build(stamp):
            return controller._build_obs(
                np.asarray([0.20, 0.0, 1.10], dtype=np.float32),
                np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
                0.0,
                0.005,
                arm_q=q,
                arm_dq=np.zeros(7, dtype=np.float32),
                arm_sample_stamp_s=stamp,
            )

        first = build(10.000)
        racket_pos[2] = np.float32(0.510)
        second = build(10.005)
        repeated = build(10.005)

        np.testing.assert_array_equal(
            first[29:32], np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(
            second[29:32], [0.0, 0.0, 2.0], atol=5e-6)
        np.testing.assert_array_equal(repeated[29:32], second[29:32])

    def test_ideal_profile_source_sample_uses_actual_ten_ms_dt(self):
        controller = _controller(OBSERVATION_PROFILE_SIMCODE2_IDEALPD67)
        q = np.zeros(7, dtype=np.float32)
        racket_pos = np.asarray([0.10, -0.20, 0.50], dtype=np.float32)
        controller._compute_racket_pose_base = mock.Mock(
            side_effect=lambda _q: (racket_pos.copy(), True))

        kwargs = dict(
            ball_pos_m=np.asarray([0.20, 0.0, 1.10], dtype=np.float32),
            ball_vel_m_s=np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
            ball_obs_age_s=0.0,
            step_dt=0.005,
            arm_q=q,
            arm_dq=np.zeros(7, dtype=np.float32),
        )
        controller._build_obs(**kwargs, arm_sample_stamp_s=20.000)
        racket_pos[:] += np.asarray([0.02, -0.01, 0.03], dtype=np.float32)
        obs = controller._build_obs(
            **kwargs, arm_sample_stamp_s=20.010)

        np.testing.assert_allclose(
            obs[29:32], [2.0, -1.0, 3.0], atol=5e-6)

    def test_simcode4_uses_training_tick_racket_velocity_with_joint_stamp(self):
        controller = _controller(OBSERVATION_PROFILE_SIMCODE4_SERVO67)
        q = np.zeros(7, dtype=np.float32)
        racket_pos = np.asarray([0.0, 0.0, 0.50], dtype=np.float32)
        controller._prev_racket_pos_base = np.asarray(
            [0.0, 0.0, 0.49], dtype=np.float32)
        controller._compute_racket_pose_base = mock.Mock(
            return_value=(racket_pos.copy(), True))

        first = controller._build_obs(
            np.asarray([0.20, 0.0, 1.10], dtype=np.float32),
            np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
            0.0,
            0.005,
            arm_q=q,
            arm_dq=np.zeros(7, dtype=np.float32),
            arm_sample_stamp_s=10.000,
        )
        repeated = controller._build_obs(
            np.asarray([0.20, 0.0, 1.10], dtype=np.float32),
            np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
            0.0,
            0.005,
            arm_q=q,
            arm_dq=np.zeros(7, dtype=np.float32),
            arm_sample_stamp_s=10.000,
        )

        np.testing.assert_allclose(
            first[29:32], [0.0, 0.0, 2.0], atol=5e-6)
        np.testing.assert_array_equal(
            repeated[29:32], np.zeros(3, dtype=np.float32))

    def test_ideal_profile_reused_stamp_with_changed_q_reanchors(self):
        controller = _controller(OBSERVATION_PROFILE_SIMCODE2_IDEALPD67)
        q = np.zeros(7, dtype=np.float32)
        racket_pos = np.asarray([0.0, 0.0, 0.50], dtype=np.float32)
        controller._compute_racket_pose_base = mock.Mock(
            side_effect=lambda _q: (racket_pos.copy(), True))

        def build(stamp):
            return controller._build_obs(
                np.asarray([0.20, 0.0, 1.10], dtype=np.float32),
                np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
                0.0,
                0.005,
                arm_q=q,
                arm_dq=np.zeros(7, dtype=np.float32),
                arm_sample_stamp_s=stamp,
            )

        build(10.000)
        racket_pos[2] = np.float32(0.510)
        incoherent_repeat = build(10.000)
        racket_pos[2] = np.float32(0.520)
        recovered = build(10.005)

        np.testing.assert_array_equal(
            incoherent_repeat[29:32], np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(
            recovered[29:32], [0.0, 0.0, 2.0], atol=5e-6)

    def test_ideal_profile_bad_sample_dt_reanchors_and_reset_clears_cache(self):
        controller = _controller(OBSERVATION_PROFILE_SIMCODE2_IDEALPD67)
        q = np.zeros(7, dtype=np.float32)
        racket_pos = np.asarray([0.0, 0.0, 0.50], dtype=np.float32)
        controller._compute_racket_pose_base = mock.Mock(
            side_effect=lambda _q: (racket_pos.copy(), True))

        def build(stamp):
            return controller._build_obs(
                np.asarray([0.20, 0.0, 1.10], dtype=np.float32),
                np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
                0.0,
                0.005,
                arm_q=q,
                arm_dq=np.zeros(7, dtype=np.float32),
                arm_sample_stamp_s=stamp,
            )

        first = build(30.000)
        racket_pos[2] = np.float32(0.60)
        backward = build(29.000)
        racket_pos[2] = np.float32(0.62)
        recovered = build(29.010)

        np.testing.assert_array_equal(
            first[29:32], np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(
            backward[29:32], np.zeros(3, dtype=np.float32))
        np.testing.assert_allclose(
            recovered[29:32], [0.0, 0.0, 2.0], atol=5e-6)
        controller.reset()
        self.assertIsNone(controller._prev_racket_sample_pos_base)
        self.assertIsNone(controller._prev_racket_sample_stamp_s)
        np.testing.assert_array_equal(
            controller._last_racket_sample_vel_base,
            np.zeros(3, dtype=np.float32),
        )

    def test_ideal_history_is_19_nominal_commands_with_14_tick_lookup(self):
        controller = _controller(
            OBSERVATION_PROFILE_SIMCODE2_IDEALPD67,
            action=np.full(7, 0.20, dtype=np.float32),
        )
        q0 = controller.arm_cmd_q.copy()
        self.assertEqual(controller._delay_steps, 14)
        self.assertEqual(controller._command_buffer_len, 19)
        self.assertEqual(len(controller._cmd_history), 19)
        np.testing.assert_array_equal(
            np.stack(controller._cmd_history),
            np.broadcast_to(q0, (19, 7)),
        )

        returned = controller.predict(
            np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
            np.asarray([0.0, 0.0, -1.0], dtype=np.float32),
            ball_valid=True,
            arm_q=q0,
            arm_dq=np.zeros(7, dtype=np.float32),
        )
        np.testing.assert_array_equal(returned, controller.arm_cmd_q)
        self.assertGreater(float(np.max(np.abs(returned - q0))), 0.0)
        np.testing.assert_array_equal(controller._cmd_history[-1], returned)
        np.testing.assert_array_equal(
            controller._get_active_delayed_command(), q0)

        controller.reset()
        q0 = controller.arm_cmd_q.copy()
        for tick in range(1, 15):
            q_tick = q0 + np.float32(tick * 1e-4)
            controller._cmd_history.append(q_tick.copy())
            controller._cmd_history.pop(0)
            controller.arm_cmd_q = q_tick.copy()
        np.testing.assert_array_equal(
            controller._get_active_delayed_command(), q0)

        q15 = q0 + np.float32(15e-4)
        controller._cmd_history.append(q15.copy())
        controller._cmd_history.pop(0)
        controller.arm_cmd_q = q15.copy()
        np.testing.assert_allclose(
            controller._get_active_delayed_command(),
            q0 + np.float32(1e-4),
            rtol=0.0,
            atol=1e-7,
        )


class FrozenCheckpointContractTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.controller = Simcode2IdealPd67PolicyController(
            str(SIMCODE2_MODEL),
            robot_xml_path=None,
            use_valid_ball_obs_age=False,
        )

    def test_checkpoint_hash_and_metadata(self):
        digest = hashlib.sha256()
        with SIMCODE2_MODEL.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.assertEqual(digest.hexdigest(), SIMCODE2_SHA256)

        metadata = inspect_mjx_checkpoint_metadata(str(SIMCODE2_MODEL))
        self.assertEqual(metadata["obs_dim"], 67)
        self.assertEqual(metadata["act_dim"], 7)
        cfg = metadata["env_cfg"]
        self.assertTrue(cfg["enable_delay_conditioning"])
        self.assertTrue(cfg["actuator_delay_observation_only"])
        self.assertFalse(cfg["actuator_cmd_filter"])
        self.assertEqual(cfg["actuator_compensation_mode"], "none")
        self.assertEqual(float(cfg["delay_min_ms"]), 72.0)
        self.assertEqual(float(cfg["delay_max_ms"]), 72.0)
        self.assertEqual(int(cfg["command_buffer_extra_steps"]), 4)
        self.assertEqual(float(cfg["lost_ball_timeout_ms"]), 350.0)

        weights = self.controller._pi_weights
        self.assertEqual(weights["l1_w"].shape, (67, 256))
        self.assertEqual(weights["l2_w"].shape, (256, 256))
        self.assertEqual(weights["out_w"].shape, (256, 7))
        np.testing.assert_allclose(
            np.rad2deg(self.controller.initial_arm_q_rad),
            np.asarray([
                5.736, -44.399, 30.683, 97.142,
                49.323, -12.269, 14.214,
            ]),
            rtol=0.0,
            atol=2e-5,
        )
        self.assertEqual(self.controller._delay_steps, 14)
        self.assertEqual(self.controller._command_buffer_len, 19)
        self.assertAlmostEqual(self.controller.ball_obs_age_clip_s, 0.5)
        self.assertAlmostEqual(
            self.controller.contact_lost_ball_timeout_s, 0.35)
        self.assertTrue(self.controller.use_valid_ball_obs_age)

    def test_actor_mean_matches_frozen_trace_without_normalization(self):
        with (SIMCODE2_ARTIFACT_DIR / "obs_trace.csv").open(newline="") as handle:
            obs_rows = list(csv.DictReader(handle))
        with (SIMCODE2_ARTIFACT_DIR / "action_trace.csv").open(newline="") as handle:
            action_rows = list(csv.DictReader(handle))
        self.assertEqual(len(obs_rows), len(action_rows))

        for index in (0, len(obs_rows) // 2, len(obs_rows) - 1):
            obs = np.asarray([
                float(obs_rows[index][f"obs/{dim:03d}"])
                for dim in range(67)
            ], dtype=np.float32)
            expected = np.asarray([
                float(action_rows[index][f"policy_mean/RightArm-{joint}"])
                for joint in range(7)
            ], dtype=np.float32)
            actual = self.controller._policy_action(obs)
            np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-5)

    def test_ideal_fk_uses_launch19_patched_xml_and_warm_state(self):
        controller = Simcode2IdealPd67PolicyController(
            str(SIMCODE2_MODEL),
            robot_xml_path=str(SIMCODE2_XML),
        )
        (
            center, _tx, _ty, _normal, radius, half_thickness, ok,
        ) = controller._compute_racket_launch_frame_base(
            controller.initial_arm_q_rad)

        self.assertTrue(ok)
        self.assertAlmostEqual(radius, 0.075, places=7)
        self.assertAlmostEqual(half_thickness, 0.001, places=7)
        np.testing.assert_allclose(
            center,
            np.asarray([-0.114089, -0.466326, 1.048675]),
            rtol=0.0,
            atol=3e-5,
        )

    def test_legacy_checkpoint_keeps_original_reset_and_history_contract(self):
        controller = MjxRLPolicyController(
            str(LEGACY_MODEL),
            robot_xml_path=None,
            enable_delay_conditioning=True,
            observation_profile=OBSERVATION_PROFILE_LEGACY_BUG_COMPAT,
        )
        np.testing.assert_allclose(
            np.rad2deg(controller.initial_arm_q_rad),
            np.asarray([9.0, -50.0, 20.0, 90.0, 45.0, -8.0, 45.0]),
            rtol=0.0,
            atol=2e-5,
        )
        self.assertEqual(controller._delay_steps, 15)
        self.assertEqual(controller._command_buffer_len, 25)
        self.assertEqual(controller._cmd_history, [])
        self.assertAlmostEqual(controller.contact_lost_ball_timeout_s, 0.15)



class Simcode4FrozenCheckpointContractTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.controller = Simcode4Servo67PolicyController(
            str(SIMCODE4_MODEL), robot_xml_path=None)
        cls.fk_controller = Simcode4Servo67PolicyController(
            str(SIMCODE4_MODEL), robot_xml_path=str(SIMCODE4_XML))

    @staticmethod
    def _trace_rows():
        with (SIMCODE4_TRACE_DIR / "obs_trace.csv").open(
                newline="") as handle:
            return list(csv.DictReader(handle))

    def test_checkpoint_identity_and_execution_contract(self):
        digest = hashlib.sha256()
        with SIMCODE4_MODEL.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        self.assertEqual(digest.hexdigest(), SIMCODE4_SHA256)
        metadata = inspect_mjx_checkpoint_metadata(str(SIMCODE4_MODEL))
        self.assertEqual((metadata["obs_dim"], metadata["act_dim"]), (67, 7))
        cfg = metadata["env_cfg"]
        self.assertEqual(cfg["actuator_mpc_feedback_source"], "actual")
        self.assertTrue(cfg["arm_servo_target_tracking_planner"])
        self.assertEqual(cfg["arm_servo_target_velocity_scale"], 1.0)
        self.assertEqual(cfg["arm_servo_target_acceleration_scale"], 0.8)
        self.assertFalse(cfg["ball_obs_age_tracks_stale"])
        self.assertEqual(cfg["ball_obs_rate_hz"], 60.0)

    def test_actor_mean_matches_frozen_simcode4_trace(self):
        obs_rows = self._trace_rows()
        with (SIMCODE4_TRACE_DIR / "action_trace.csv").open(newline="") as handle:
            action_rows = list(csv.DictReader(handle))
        self.assertEqual(len(obs_rows), len(action_rows))
        for index in (0, len(obs_rows) // 2, len(obs_rows) - 1):
            obs = np.asarray([
                float(obs_rows[index][f"obs/{dim:03d}"])
                for dim in range(67)
            ], dtype=np.float32)
            expected = np.asarray([
                float(action_rows[index][f"policy_mean/RightArm-{joint}"])
                for joint in range(7)
            ], dtype=np.float32)
            np.testing.assert_allclose(
                self.controller._policy_action(obs), expected,
                rtol=0.0, atol=1e-5)

    def test_warm_base_observation_matches_frozen_trace(self):
        row = self._trace_rows()[0]
        expected = np.asarray([
            float(row[f"obs/{dim:03d}"])
            for dim in range(14, 20)
        ], dtype=np.float32)
        actual = np.concatenate([
            self.fk_controller._base_obs_q,
            self.fk_controller._base_obs_dq,
        ])
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-9)

    def test_deployment_assembles_frozen_67d_fields_with_same_units(self):
        row = self._trace_rows()[0]
        expected = np.asarray([
            float(row[f"obs/{dim:03d}"])
            for dim in range(67)
        ], dtype=np.float32)
        controller = self.fk_controller
        controller.reset()
        q = expected[0:7].copy()
        controller.arm_cmd_q = (q + expected[42:49]).astype(np.float32)
        controller.arm_cmd_qvel = expected[51:58].copy()
        controller.prev_action = expected[35:42].copy()
        controller._cmd_history = [
            controller.arm_cmd_q.copy()
            for _ in range(controller._command_buffer_len)
        ]
        active_index = (
            len(controller._cmd_history) - 1 - controller._delay_steps)
        controller._cmd_history[active_index] = (
            q + expected[58:65]).astype(np.float32)
        controller._prev_racket_pos_base = (
            expected[26:29] - expected[29:32] * np.float32(0.005)
        ).astype(np.float32)

        with mock.patch.object(
                controller, "_compute_racket_pose_base",
                return_value=(expected[26:29].copy(), True)):
            actual = controller._build_obs(
                expected[20:23],
                expected[23:26],
                float(expected[49]) * controller.ball_obs_age_clip_s,
                0.005,
                arm_q=q,
                arm_dq=expected[7:14],
                arm_sample_stamp_s=123.0,
            )

        # Phase fields 65:67 use the episode-randomized simulated gravity.
        # Hardware correctly uses physical gravity, so compare every directly
        # observable/state-history field here and test phase separately.
        np.testing.assert_allclose(
            actual[:65], expected[:65], rtol=0.0, atol=1e-6)
        self.assertEqual(actual.dtype, np.float32)


class Simcode4ServoPlannerContractTest(unittest.TestCase):

    def test_numpy_planner_replays_frozen_mjx_servo_trace(self):
        with (SIMCODE4_TRACE_DIR / "action_trace.csv").open(
                newline="") as handle:
            rows = list(csv.DictReader(handle))

        def vector(row, prefix):
            return np.asarray([
                float(row[f"{prefix}/RightArm-{joint}"])
                for joint in range(7)
            ], dtype=np.float64)

        planner = ServoTargetTrackingPlanner(
            vector(rows[0], "arm_applied_q"),
            dt=0.005,
            velocity_scale=1.0,
            acceleration_scale=0.8,
        )
        errors = []
        for row in rows[1:]:
            actual = planner.step(
                vector(row, "arm_servo_target_unlimited"))
            expected = vector(row, "arm_applied_q")
            errors.append(actual - expected)

        # NumPy float64 versus frozen JAX float32 accumulates only sub-micron
        # angular differences over all 1199 causal planner steps.
        self.assertLessEqual(
            float(np.max(np.abs(np.asarray(errors)))), 3e-7)

    def test_target_tracking_planner_respects_checkpoint_acceleration_scale(self):
        initial = np.zeros(7, dtype=np.float64)
        planner = ServoTargetTrackingPlanner(
            initial, dt=0.005, velocity_scale=1.0, acceleration_scale=0.8)
        target = np.full(7, 0.5, dtype=np.float64)
        velocities = [planner.last_vel.copy()]
        positions = [planner.last_cmd.copy()]
        for _ in range(80):
            positions.append(planner.step(target))
            velocities.append(planner.last_vel.copy())
            self.assertTrue(planner.last_feasible)
        velocities = np.asarray(velocities)
        accelerations = np.diff(velocities, axis=0) / 0.005
        limits = np.deg2rad(
            RightArmCommandSafetyLimiter.ACC_LIMIT_DEG_S2) * 0.8
        self.assertTrue(np.all(np.abs(accelerations) <= limits + 1e-9))
        self.assertTrue(np.all(np.diff(np.asarray(positions)[:, 0]) >= -1e-12))
        self.assertLessEqual(float(positions[-1][0]), 0.5 + 1e-9)


@dataclass
class _MjxConfigFields:
    domain_randomization: bool = False
    enable_delay_conditioning: bool = True
    ball_obs_rate_hz: float = 60.0
    ball_obs_fractional_rate: bool = False
    ball_obs_dropout_prob: float = 0.0
    ball_obs_dropout_burst_prob: float = 0.0
    ball_obs_dropout_max_steps: int = 1
    ball_obs_dropout_burst_max_steps: int = 1
    ball_obs_age_tracks_stale: bool = True
    ball_obs_dropout_on_refresh_only: bool = False
    ball_obs_require_camera_visible: bool = False
    dr_randomize_latency: bool = False
    dr_randomize_actuator_cmd_filter: bool = False
    dr_randomize_actuator: bool = False
    dr_randomize_second_order_actuator: bool = False
    delay_min_ms: float = 0.0
    delay_max_ms: float = 0.0
    delay_jitter_ms: float = 0.0
    delay_sampling_mode: str = "uniform"
    actuator_delay_observation_only: bool = False
    actuator_cmd_filter: bool = False
    actuator_cmd_model: str = "first_order"
    actuator_cmd_natural_frequency_rad_s: tuple = ()
    actuator_cmd_damping_ratio: tuple = ()
    actuator_cmd_gain_per_joint: tuple = ()
    actuator_cmd_delay_ms_per_joint: tuple = ()
    actuator_cmd_tau: float = 0.0
    actuator_cmd_gain: float = 1.0
    actuator_compensation_mode: str = "none"
    actuator_mpc_beta: float = 0.0
    actuator_mpc_delay_scale: float = 0.0
    actuator_mpc_tau_scale: float = 0.0
    actuator_mpc_horizon_steps: int = 1
    actuator_mpc_tracking_weight: float = 0.0
    actuator_mpc_nominal_weight: float = 0.0
    actuator_mpc_delta_weight: float = 0.0
    actuator_mpc_max_delta_rad: float = 0.0
    actuator_mpc_feedback_source: str = "applied"
    arm_servo_target_tracking_planner: bool = False
    arm_servo_target_planner_before_actuator: bool = False
    arm_servo_target_planner_before_actuator_model: bool = False
    arm_servo_target_limiter: bool = False
    arm_post_compensation_limiter: bool = False
    arm_actual_state_limiter: bool = False
    arm_servo_target_velocity_scale: float = 1.0
    arm_servo_target_acceleration_scale: float = 1.0
    right_arm_pd_profile: str = "default"


class SimMirrorProfileContractTest(unittest.TestCase):

    def _base_config(self) -> SimMirrorConfig:
        return SimMirrorConfig(
            model_path=LEGACY_MODEL,
            policy_xml_path=LEGACY_POLICY_XML,
            sim_xml_path=LEGACY_SIM_XML,
            policy_obs_dim=67,
            control_dt=0.005,
            enable_compensation=True,
            compensation_mode="inverse_mpc",
            comp_delay_ms=91.0,
            comp_actuator_tau_s=0.123,
            rl_use_valid_ball_obs_age=False,
            simcode2_model_path=SIMCODE2_MODEL,
            simcode2_policy_xml_path=SIMCODE2_XML,
            simcode2_sim_xml_path=SIMCODE2_XML,
            simcode3_model_path=SIMCODE3_MODEL,
            simcode3_policy_xml_path=SIMCODE3_XML,
            simcode3_sim_xml_path=SIMCODE3_XML,
            simcode4_model_path=SIMCODE4_MODEL,
            simcode4_policy_xml_path=SIMCODE4_XML,
            simcode4_sim_xml_path=SIMCODE4_XML,
            simcode5_model_path=SIMCODE5_MODEL,
            simcode5_policy_xml_path=SIMCODE5_XML,
            simcode5_sim_xml_path=SIMCODE5_XML,
        )

    def _kwargs(self, cfg: SimMirrorConfig) -> dict:
        mirror = RealBallMjxJaxSimMirror.__new__(RealBallMjxJaxSimMirror)
        mirror.cfg = cfg
        mirror.policy_profile = cfg.policy_profile
        mirror._ideal_pd_profile = (
            cfg.policy_profile == OBSERVATION_PROFILE_SIMCODE2_IDEALPD67)
        mirror._servo_planner_profile = (
            cfg.policy_profile == OBSERVATION_PROFILE_SIMCODE4_SERVO67)
        mirror._pure_actuator_profile = (
            cfg.policy_profile
            == OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67)
        return mirror._mjx_env_kwargs(_MjxConfigFields)

    def test_profile_selection_swaps_model_and_both_xmls_as_one_unit(self):
        base = self._base_config()
        legacy = sim_mirror_config_for_policy_profile(
            base, OBSERVATION_PROFILE_LEGACY_BUG_COMPAT)
        ideal = sim_mirror_config_for_policy_profile(
            base, OBSERVATION_PROFILE_SIMCODE2_IDEALPD67)

        self.assertEqual(legacy.model_path, LEGACY_MODEL)
        self.assertEqual(legacy.policy_xml_path, LEGACY_POLICY_XML)
        self.assertEqual(legacy.sim_xml_path, LEGACY_SIM_XML)
        self.assertEqual(ideal.model_path, SIMCODE2_MODEL)
        self.assertEqual(ideal.policy_xml_path, SIMCODE2_XML)
        self.assertEqual(ideal.sim_xml_path, SIMCODE2_XML)
        self.assertFalse(legacy.rl_use_valid_ball_obs_age)
        self.assertTrue(ideal.rl_use_valid_ball_obs_age)

    def test_simcode3_is_independent_and_keeps_the_actuator_plant(self):
        cfg = sim_mirror_config_for_policy_profile(
            self._base_config(), OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67)

        self.assertEqual(cfg.model_path, SIMCODE3_MODEL)
        self.assertEqual(cfg.policy_xml_path, SIMCODE3_XML)
        self.assertEqual(cfg.sim_xml_path, SIMCODE3_XML)
        self.assertTrue(cfg.rl_use_valid_ball_obs_age)
        self.assertTrue(uses_corrected_67d_observation(cfg.policy_profile))

        kwargs = self._kwargs(cfg)
        self.assertTrue(kwargs["actuator_cmd_filter"])
        self.assertEqual(kwargs["actuator_compensation_mode"], "inverse_mpc")
        self.assertEqual(float(kwargs["delay_min_ms"]), 91.0)

    def test_simcode3_controller_forces_corrected_67d_observations(self):
        with mock.patch.object(
                MjxRLPolicyController, "_load_model",
                lambda controller, _path, _device: setattr(
                    controller, "model", _Model(67, 7))):
            controller = Simcode3Actuator67PolicyController("unused")

        self.assertEqual(
            controller.observation_profile,
            OBSERVATION_PROFILE_SIMCODE3_ACTUATOR67,
        )
        self.assertTrue(controller.enable_delay_conditioning)
        self.assertTrue(controller.use_valid_ball_obs_age)

    def test_simcode4_keeps_checkpoint_servo_plant_and_zero_age_contract(self):
        cfg = sim_mirror_config_for_policy_profile(
            self._base_config(), OBSERVATION_PROFILE_SIMCODE4_SERVO67)
        self.assertEqual(cfg.model_path, SIMCODE4_MODEL)
        self.assertEqual(cfg.policy_xml_path, SIMCODE4_XML)
        self.assertEqual(cfg.sim_xml_path, SIMCODE4_XML)
        self.assertEqual(cfg.backend, "mjx")
        kwargs = self._kwargs(cfg)
        self.assertEqual(kwargs["actuator_mpc_feedback_source"], "actual")
        self.assertTrue(kwargs["arm_servo_target_tracking_planner"])
        self.assertFalse(kwargs["arm_servo_target_planner_before_actuator"])
        self.assertFalse(kwargs["arm_servo_target_limiter"])
        self.assertEqual(kwargs["arm_servo_target_velocity_scale"], 1.0)
        self.assertEqual(kwargs["arm_servo_target_acceleration_scale"], 0.8)

        with mock.patch.object(
                MjxRLPolicyController, "_load_model",
                lambda controller, _path, _device: setattr(
                    controller, "model", _Model(67, 7))):
            controller = Simcode4Servo67PolicyController("unused")
        self.assertEqual(
            controller.observation_profile,
            OBSERVATION_PROFILE_SIMCODE4_SERVO67)
        self.assertTrue(controller.enable_delay_conditioning)
        self.assertFalse(controller.use_valid_ball_obs_age)

    def test_legacy_mjx_plant_keeps_compensation_filter_and_runtime_delay(self):
        cfg = sim_mirror_config_for_policy_profile(
            self._base_config(), OBSERVATION_PROFILE_LEGACY_BUG_COMPAT)

        kwargs = self._kwargs(cfg)

        self.assertTrue(kwargs["actuator_cmd_filter"])
        self.assertEqual(kwargs["actuator_compensation_mode"], "inverse_mpc")
        self.assertEqual(float(kwargs["delay_min_ms"]), 91.0)
        self.assertEqual(float(kwargs["delay_max_ms"]), 91.0)

    def test_simcode5_restores_pure_second_order_plant_and_zero_age(self):
        cfg = sim_mirror_config_for_policy_profile(
            self._base_config(),
            OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67)
        self.assertEqual(cfg.model_path, SIMCODE5_MODEL)
        self.assertEqual(cfg.policy_xml_path, SIMCODE5_XML)
        self.assertEqual(cfg.sim_xml_path, SIMCODE5_XML)
        self.assertEqual(cfg.backend, "mujoco")
        self.assertFalse(cfg.rl_use_valid_ball_obs_age)

        mjx_cfg = sim_mirror_config_for_policy_profile(
            replace(self._base_config(), backend="mjx"),
            OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67)
        self.assertEqual(mjx_cfg.backend, "mjx")

        kwargs = self._kwargs(cfg)
        metadata = inspect_mjx_checkpoint_metadata(str(SIMCODE5_MODEL))
        env_cfg = metadata["env_cfg"]
        self.assertEqual(kwargs["actuator_cmd_model"], "second_order")
        self.assertEqual(kwargs["actuator_compensation_mode"], "none")
        self.assertFalse(kwargs["arm_servo_target_tracking_planner"])
        self.assertFalse(kwargs["ball_obs_age_tracks_stale"])
        self.assertFalse(kwargs["dr_randomize_second_order_actuator"])
        self.assertEqual(
            tuple(kwargs["actuator_cmd_natural_frequency_rad_s"]),
            tuple(env_cfg["actuator_cmd_natural_frequency_rad_s"]))
        self.assertEqual(
            tuple(kwargs["actuator_cmd_delay_ms_per_joint"]),
            tuple(env_cfg["actuator_cmd_delay_ms_per_joint"]))
        self.assertEqual(kwargs["right_arm_pd_profile"], "sport_taskspace_fit_v1")

        with mock.patch.object(
                MjxRLPolicyController, "_load_model",
                lambda controller, _path, _device: setattr(
                    controller, "model", _Model(67, 7))):
            controller = Simcode5PureActuator67PolicyController("unused")
        self.assertEqual(
            controller.observation_profile,
            OBSERVATION_PROFILE_SIMCODE5_PURE_ACTUATOR67)
        self.assertTrue(controller.enable_delay_conditioning)
        self.assertFalse(controller.use_valid_ball_obs_age)

    def test_ideal_mjx_plant_is_checkpoint_ideal_pd_observation_only_delay(self):
        cfg = sim_mirror_config_for_policy_profile(
            self._base_config(), OBSERVATION_PROFILE_SIMCODE2_IDEALPD67)

        kwargs = self._kwargs(cfg)

        self.assertFalse(kwargs["actuator_cmd_filter"])
        self.assertEqual(kwargs["actuator_compensation_mode"], "none")
        self.assertTrue(kwargs["actuator_delay_observation_only"])
        self.assertEqual(float(kwargs["delay_min_ms"]), 72.0)
        self.assertEqual(float(kwargs["delay_max_ms"]), 72.0)


def _load_probe_module():
    module_name = "_policy_profile_contract_probe"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    path = (
        PACKAGE_ROOT / "pingpong_controller" / "tools"
        / "interactive_policy_safety_probe.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ProfileGenerationAlignmentTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.probe = _load_probe_module()

    def _real(self, profile: str, generation: int) -> dict:
        return {
            "sourceSeq": 10,
            "tick": 10,
            "mode": "policy",
            "policyExecuted": True,
            "policyProfile": profile,
            "policyGeneration": generation,
            "runtime": {
                "policyProfile": profile,
                "activePolicyProfile": profile,
                "policyGeneration": generation,
            },
            "policyInput": {
                "ball": [0.1, -0.2, 1.3],
                "velocity": [0.01, 0.02, -0.3],
                "valid": True,
                "age": 0.005,
                "ballObservationSeq": 7,
            },
        }

    def _sim(self, profile: str, generation: int) -> dict:
        return {
            "sync": {
                "sourceSeq": 10,
                "streamSeq": 1,
                "policyProfile": profile,
                "policyGeneration": generation,
                "inputBall": [0.1, -0.2, 1.3],
                "inputVelocity": [0.01, 0.02, -0.3],
                "inputValid": True,
                "inputAgeS": 0.005,
                "ballObservationSeq": 7,
                "streamContinuous": True,
                "resultDrops": 0,
                "runtimeConfigMatch": True,
            },
        }

    def test_same_profile_and_generation_align(self):
        real = self._real(OBSERVATION_PROFILE_SIMCODE2_IDEALPD67, 3)
        sim = self._sim(OBSERVATION_PROFILE_SIMCODE2_IDEALPD67, 3)
        self.assertTrue(self.probe._policy_inputs_match(real, sim))
        _merged, report = self.probe._align_sim_real_samples([real], [sim])
        self.assertTrue(report["alignmentValid"])

    def test_profile_or_generation_mismatch_never_aligns(self):
        real = self._real(OBSERVATION_PROFILE_SIMCODE2_IDEALPD67, 3)
        cases = (
            self._sim(OBSERVATION_PROFILE_LEGACY_BUG_COMPAT, 3),
            self._sim(OBSERVATION_PROFILE_SIMCODE2_IDEALPD67, 2),
        )
        for sim in cases:
            with self.subTest(sync=sim["sync"]):
                self.assertFalse(self.probe._policy_inputs_match(real, sim))
                _merged, report = self.probe._align_sim_real_samples(
                    [real], [sim])
                self.assertFalse(report["alignmentValid"])


class ProbeSimInitializationContractTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.probe_module = _load_probe_module()

    def test_new_profile_start_keeps_legacy_paths_for_subprocess_switch_back(self):
        captured: list[SimMirrorConfig] = []

        class _FakeSimMirrorSubprocess:
            def __init__(self, cfg):
                captured.append(cfg)

            def wait_until_ready(self, _timeout_s):
                return {"available": True, "sync": {"ready": True}}

        cfg = SimpleNamespace(
            enable_sim_mirror=True,
            sim_backend="mujoco",
            model_path=LEGACY_MODEL,
            robot_xml_path=LEGACY_POLICY_XML,
            sim_xml_path=LEGACY_SIM_XML,
            policy_obs_dim=67,
            control_dt=0.005,
            policy_profile=OBSERVATION_PROFILE_SIMCODE2_IDEALPD67,
            simcode2_model_path=SIMCODE2_MODEL,
            simcode2_policy_xml_path=SIMCODE2_XML,
            simcode2_sim_xml_path=SIMCODE2_XML,
            sim_frame_rate_hz=30.0,
            sim_render_width=640,
            sim_render_height=480,
            sim_enable_video_frames=False,
            sim_use_mujoco_renderer=False,
            enable_compensation=True,
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
            comp_mpc_max_delta_rad=np.deg2rad(30.0),
            rl_use_valid_ball_obs_age=True,
            sim_subprocess=True,
        )
        policy_probe = self.probe_module.PolicySafetyProbe.__new__(
            self.probe_module.PolicySafetyProbe)
        policy_probe.cfg = cfg
        policy_probe.active_policy_profile = (
            OBSERVATION_PROFILE_SIMCODE2_IDEALPD67)
        policy_probe.policy_generation = 0

        with mock.patch.object(
                self.probe_module,
                "SimMirrorSubprocess",
                _FakeSimMirrorSubprocess):
            policy_probe._init_sim_mirror()

        self.assertEqual(len(captured), 1)
        unresolved = captured[0]
        self.assertEqual(unresolved.model_path, LEGACY_MODEL)
        self.assertEqual(unresolved.policy_xml_path, LEGACY_POLICY_XML)
        self.assertEqual(unresolved.sim_xml_path, LEGACY_SIM_XML)
        ideal = sim_mirror_config_for_policy_profile(
            unresolved, OBSERVATION_PROFILE_SIMCODE2_IDEALPD67)
        switched_back = sim_mirror_config_for_policy_profile(
            unresolved, OBSERVATION_PROFILE_LEGACY_BUG_COMPAT)
        self.assertEqual(ideal.model_path, SIMCODE2_MODEL)
        self.assertEqual(switched_back.model_path, LEGACY_MODEL)
        self.assertEqual(switched_back.policy_xml_path, LEGACY_POLICY_XML)
        self.assertEqual(switched_back.sim_xml_path, LEGACY_SIM_XML)

    def test_raw_camera_frame_age_is_used_only_by_corrected_policy_profile(self):
        policy_probe = self.probe_module.PolicySafetyProbe.__new__(
            self.probe_module.PolicySafetyProbe)
        policy_probe.cfg = SimpleNamespace(
            ball_lost_timeout_s=0.35,
            racket_tracking_max_ball_age_s=0.12,
            control_dt=0.005,
        )
        policy_probe.ros_bridge = SimpleNamespace(now_sec=lambda: 100.0)
        policy_probe.ball_estimator = None
        policy_probe.ball_estimator_key = None
        policy_probe.last_ball_observation_stamp_s = None
        ball_pos = np.asarray([0.1, -0.3, 1.2], dtype=np.float32)
        ball_vel = np.asarray([0.0, 0.0, -0.5], dtype=np.float32)

        ideal_options = SimpleNamespace(
            control_mode="policy",
            policy_profile=OBSERVATION_PROFILE_SIMCODE2_IDEALPD67,
            ball_source="camera",
            use_ball_estimator=False,
        )
        out_pos, out_vel, valid, ideal_age, _info = (
            policy_probe._resolve_control_ball_state(
                ball_pos,
                ball_vel,
                True,
                ideal_options,
                measurement_stamp_s=99.975,
            ))
        np.testing.assert_array_equal(out_pos, ball_pos)
        np.testing.assert_array_equal(out_vel, ball_vel)
        self.assertTrue(valid)
        self.assertAlmostEqual(ideal_age, 0.025, places=8)

        legacy_options = SimpleNamespace(
            control_mode="policy",
            policy_profile=OBSERVATION_PROFILE_LEGACY_BUG_COMPAT,
            ball_source="camera",
            use_ball_estimator=False,
        )
        _pos, _vel, _valid, legacy_age, _info = (
            policy_probe._resolve_control_ball_state(
                ball_pos,
                ball_vel,
                True,
                legacy_options,
                measurement_stamp_s=99.975,
            ))
        self.assertIsNone(legacy_age)


if __name__ == "__main__":
    unittest.main()
