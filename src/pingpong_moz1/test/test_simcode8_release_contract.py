#!/usr/bin/env python3
"""Frozen deployment contracts for the selected simcode8 releases."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.mjx_rl_policy import (  # noqa: E402
    Gpu0QvelRealRmpV85PolicyController,
    Gpu0QvelRealRmpV97SourceClockPolicyController,
    Gpu1QaccFtV153PolicyController,
)
from pingpong_controller.reset_trajectory import (  # noqa: E402
    SlowResetTrajectory,
)
from pingpong_controller.rl_policy import (  # noqa: E402
    OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85,
    OBSERVATION_PROFILE_GPU1_QACC_FT_V153,
)
from pingpong_controller.sim_mirror import (  # noqa: E402
    RealBallSimMirror,
    SimMirrorConfig,
    sim_mirror_config_for_policy_profile,
)


SIMCODE8_ROOT = REPO_ROOT / "simcode8" / "pingpong_moz1-main"
SIMCODE8_XML = (
    SIMCODE8_ROOT / "pingpong_controller" / "tools" / "rl_sim"
    / "moz1_pd.xml"
)
GPU0_V97_MODEL = (
    SIMCODE8_ROOT / "pingpong_controller" / "outputs" / "rl_sim"
    / "selected_best_models_and_normal_reset_videos_20260901"
    / "gpu0_v97" / "gpu0_v97_best_step5836242944.pkl"
)
GPU1_V153_MODEL = (
    SIMCODE8_ROOT / "pingpong_controller" / "outputs" / "rl_sim"
    / "selected_best_models_and_normal_reset_videos_20260901"
    / "gpu1_v153" / "gpu1_v153_best_step14841511936.pkl"
)
GPU1_REFERENCE_PATH = (
    SIMCODE8_ROOT / "pingpong_controller" / "tools" / "rl_2real"
    / "gpu1_v153_qacc_reference.py"
)


def requires_fixtures(*paths: Path):
    missing = [str(path) for path in paths if not path.is_file()]
    return unittest.skipUnless(
        not missing,
        "fixture missing (supply these paths): " + ", ".join(missing),
    )


def load_gpu1_reference_module():
    spec = importlib.util.spec_from_file_location(
        "simcode8_gpu1_v153_reference", GPU1_REFERENCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the simcode8 GPU1 reference adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@requires_fixtures(GPU0_V97_MODEL)
class Gpu0V97ReleaseContractTest(unittest.TestCase):

    def test_checkpoint_identity_and_actor_golden_vector(self):
        controller = Gpu0QvelRealRmpV85PolicyController(
            str(GPU0_V97_MODEL), robot_xml_path=None)

        self.assertEqual(controller.policy_model_id, "v97")
        self.assertEqual(
            controller._checkpoint_sha256,
            "f166a4b8adb6b5b6a846b14407baa5da32ba3535492d23cb0ad5978cece8568d",
        )
        np.testing.assert_allclose(
            controller._policy_action(np.zeros(57, dtype=np.float32)),
            [-0.013275788, 0.009566929, 0.019368205, 0.067597359,
             0.120253950, -0.124855369, 0.022645233],
            rtol=0.0,
            atol=2e-7,
        )

    def test_source_clock_adapter_has_distinct_runtime_identity(self):
        controller = Gpu0QvelRealRmpV97SourceClockPolicyController(
            str(GPU0_V97_MODEL), robot_xml_path=None)

        self.assertEqual(controller.policy_model_id, "v97_source_clock")
        self.assertEqual(controller.checkpoint_policy_model_id, "v97")
        self.assertEqual(
            controller.racket_velocity_observation_contract,
            "joint_state_source_clock_v1",
        )
        self.assertEqual(
            controller._checkpoint_sha256,
            Gpu0QvelRealRmpV85PolicyController.V97_CHECKPOINT_SHA256,
        )

    @requires_fixtures(SIMCODE8_XML)
    def test_cpu_mirror_restores_v97_rmp_only_on_sim_side(self):
        cfg = SimMirrorConfig(
            model_path=GPU0_V97_MODEL,
            policy_xml_path=SIMCODE8_XML,
            sim_xml_path=SIMCODE8_XML,
            policy_obs_dim=57,
            control_dt=0.005,
            policy_profile=OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85,
            policy_model="v97",
            simcode7_model_path=GPU0_V97_MODEL,
            simcode7_v97_model_path=GPU0_V97_MODEL,
            simcode8_policy_xml_path=SIMCODE8_XML,
            simcode8_sim_xml_path=SIMCODE8_XML,
            backend="mujoco",
            enable_video_frames=False,
            use_mujoco_renderer=False,
        )
        mirror = RealBallSimMirror(cfg)
        try:
            self.assertTrue(mirror._real_rmp_profile)
            self.assertEqual(
                mirror.right_arm_pd_profile, "recovered_rmp_rmpmd_v2")
            self.assertIsNotNone(mirror.rmp_planner)
            self.assertEqual(mirror.rmp_target_update_delay_steps, 3)
            np.testing.assert_array_equal(
                mirror.rmp_output_delay_steps,
                [18, 19, 19, 18, 18, 18, 18],
            )
        finally:
            mirror.close()

    @requires_fixtures(SIMCODE8_XML)
    def test_source_clock_release_uses_v97_assets_in_training_clock_mirror(self):
        base = SimMirrorConfig(
            model_path=GPU0_V97_MODEL,
            policy_xml_path=SIMCODE8_XML,
            sim_xml_path=SIMCODE8_XML,
            policy_obs_dim=57,
            control_dt=0.005,
            policy_profile=OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85,
            policy_model="v97_source_clock",
            simcode7_model_path=GPU0_V97_MODEL,
            simcode7_v97_model_path=GPU0_V97_MODEL,
            simcode7_policy_xml_path=SIMCODE8_XML,
            simcode7_sim_xml_path=SIMCODE8_XML,
            simcode8_policy_xml_path=SIMCODE8_XML,
            simcode8_sim_xml_path=SIMCODE8_XML,
            backend="mujoco",
            enable_video_frames=False,
            use_mujoco_renderer=False,
        )
        resolved = sim_mirror_config_for_policy_profile(
            base, OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85)
        self.assertEqual(resolved.model_path, GPU0_V97_MODEL)
        self.assertEqual(resolved.policy_xml_path, SIMCODE8_XML)
        mirror = RealBallSimMirror(resolved)
        try:
            self.assertEqual(mirror.cfg.policy_model, "v97_source_clock")
            # The mirror represents a new simulated state every 5 ms and keeps
            # the checkpoint's original training-clock observation adapter.
            self.assertIsInstance(
                mirror.policy, Gpu0QvelRealRmpV85PolicyController)
            self.assertNotIsInstance(
                mirror.policy, Gpu0QvelRealRmpV97SourceClockPolicyController)
        finally:
            mirror.close()


@requires_fixtures(GPU1_V153_MODEL, SIMCODE8_XML, GPU1_REFERENCE_PATH)
class Gpu1V153ReleaseContractTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.reference_module = load_gpu1_reference_module()

    def test_robot_adapter_matches_authenticated_simcode8_reference(self):
        controller = Gpu1QaccFtV153PolicyController(
            str(GPU1_V153_MODEL), robot_xml_path=str(SIMCODE8_XML))
        self.assertEqual(controller.policy_model_id, "v153")
        self.assertEqual(
            controller._checkpoint_sha256,
            "a02b0421029e808b84ea0ee5cf615b3e65c6df3bc28c0f4766ad9953c0e85176",
        )
        np.testing.assert_array_equal(
            controller._base_obs_q, np.zeros(3, dtype=np.float32))
        np.testing.assert_array_equal(
            controller._base_obs_dq, np.zeros(3, dtype=np.float32))

        q = controller.current_arm_cmd_q().copy()
        dq = np.linspace(-0.03, 0.03, 7, dtype=np.float32)
        racket_position, fk_ok = controller._compute_racket_pose_base(q)
        self.assertTrue(fk_ok)
        ball_position = np.asarray([0.12, -0.42, 1.16], dtype=np.float32)
        ball_velocity = np.asarray([0.10, -0.20, -0.70], dtype=np.float32)
        zero3 = np.zeros(3, dtype=np.float32)

        reference = self.reference_module.GPU1QaccFtReference(
            controller._policy_action,
            joint_position_low_rad=controller.arm_q_low,
            joint_position_high_rad=controller.arm_q_high,
        )
        reference_state = reference.reset(q, dq)

        for tick, confirmed_hits in enumerate((0, 1)):
            control_time = 2.0 + tick * 0.005
            represented_time = control_time - 0.017
            expected = reference.step(
                reference_state,
                measured_q_rad=q,
                measured_dq_rad_s=dq,
                joint_feedback_age_s=0.001,
                base_q=zero3,
                base_dq=zero3,
                ball_position_base_m=ball_position,
                ball_velocity_base_m_s=ball_velocity,
                racket_position_base_m=racket_position,
                racket_velocity_base_m_s=zero3,
                control_time_s=control_time,
                ball_state_time_s=represented_time,
                confirmed_hit_count=confirmed_hits,
            )
            actual_q = controller.predict(
                ball_position,
                ball_velocity,
                ball_valid=True,
                ball_obs_age_s=0.017,
                dt=0.005,
                arm_q=q,
                arm_dq=dq,
                arm_sample_stamp_s=control_time,
                confirmed_hit_count=confirmed_hits,
            )
            np.testing.assert_allclose(
                controller.last_obs,
                expected.observation,
                rtol=0.0,
                atol=1e-7,
            )
            np.testing.assert_allclose(
                actual_q, expected.drive_input_q_rad, rtol=0.0, atol=1e-7)
            expected_age = 0.0 if confirmed_hits == 0 else 0.034
            self.assertAlmostEqual(
                float(controller.last_obs[49]), expected_age, places=6)
            reference_state = expected.state

    def test_ball_age_over_350_ms_fails_closed(self):
        controller = Gpu1QaccFtV153PolicyController(
            str(GPU1_V153_MODEL), robot_xml_path=str(SIMCODE8_XML))
        q = controller.current_arm_cmd_q().copy()
        with self.assertRaisesRegex(RuntimeError, "exceeds 350 ms"):
            controller.predict(
                np.asarray([0.1, -0.4, 1.1], dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                ball_valid=True,
                ball_obs_age_s=0.351,
                dt=0.005,
                arm_q=q,
                arm_dq=np.zeros(7, dtype=np.float32),
                arm_sample_stamp_s=1.0,
                confirmed_hit_count=1,
            )

    def test_cpu_mirror_restores_v153_actuator_surrogate(self):
        cfg = SimMirrorConfig(
            model_path=GPU1_V153_MODEL,
            policy_xml_path=SIMCODE8_XML,
            sim_xml_path=SIMCODE8_XML,
            policy_obs_dim=67,
            control_dt=0.005,
            policy_profile=OBSERVATION_PROFILE_GPU1_QACC_FT_V153,
            gpu1_qacc_ft_model_path=GPU1_V153_MODEL,
            gpu1_qacc_ft_policy_xml_path=SIMCODE8_XML,
            gpu1_qacc_ft_sim_xml_path=SIMCODE8_XML,
            backend="mujoco",
            enable_video_frames=False,
            use_mujoco_renderer=False,
        )
        mirror = RealBallSimMirror(cfg)
        try:
            self.assertTrue(mirror._pure_actuator_profile)
            np.testing.assert_array_equal(
                mirror.pure_actuator_delay_steps,
                [9, 10, 9, 8, 7, 9, 10],
            )
            np.testing.assert_allclose(
                mirror.pure_actuator_wn,
                [21.8911379437, 22.0895753812, 22.6504705, 21.730533,
                 20.1549483562, 22.2451380938, 23.0546718187],
                rtol=0.0,
                atol=2e-6,
            )
            np.testing.assert_allclose(
                mirror.pure_actuator_zeta,
                [0.3330028, 0.3295521, 0.3111642, 0.3114,
                 0.3131064, 0.3112326, 0.28553475],
                rtol=0.0,
                atol=2e-7,
            )
            np.testing.assert_allclose(
                mirror.pure_actuator_gain,
                [0.99788351, 0.9966119, 0.99231151, 0.99094239,
                 0.982326685, 0.992694585, 0.983194325],
                rtol=0.0,
                atol=2e-7,
            )
        finally:
            mirror.close()


class SlowResetTrajectoryContractTest(unittest.TestCase):

    def test_reset_is_monotonic_and_respects_sampled_limits(self):
        trajectory = SlowResetTrajectory(
            dt_s=0.005,
            velocity_limit_deg_s=10.0,
            acceleration_limit_deg_s2=25.0,
        )
        target = np.asarray(
            [1.0, -1.0, 0.5, -0.5, 0.2, -0.2, 0.1],
            dtype=np.float64,
        )
        trajectory.start(np.zeros(7, dtype=np.float64), target)
        positions = [trajectory.position_rad.copy()]
        velocities = [trajectory.velocity_rad_s.copy()]
        for _ in range(10_000):
            position, velocity, done = trajectory.step()
            positions.append(position)
            velocities.append(velocity)
            if done:
                break
        self.assertTrue(done)
        position_samples = np.asarray(positions)
        velocity_samples = np.asarray(velocities)
        acceleration_samples = np.diff(velocity_samples, axis=0) / 0.005

        np.testing.assert_allclose(position_samples[-1], target, atol=1e-12)
        np.testing.assert_allclose(velocity_samples[-1], 0.0, atol=1e-12)
        self.assertLessEqual(
            float(np.max(np.abs(velocity_samples))),
            float(np.deg2rad(10.0)) + 1e-10,
        )
        self.assertLessEqual(
            float(np.max(np.abs(acceleration_samples))),
            float(np.deg2rad(25.0)) + 1e-8,
        )
        self.assertLessEqual(
            float(np.max(np.abs(np.diff(position_samples, axis=0)))),
            float(np.deg2rad(10.0) * 0.005) + 1e-12,
        )
        self.assertTrue(np.all(np.diff(position_samples[:, 0]) >= -1e-12))
        self.assertTrue(np.all(np.diff(position_samples[:, 1]) <= 1e-12))

    def test_reset_is_synchronized_and_small_motion_takes_two_seconds(self):
        trajectory = SlowResetTrajectory(dt_s=0.005)
        np.testing.assert_allclose(
            np.rad2deg(trajectory.velocity_limit_rad_s), 15.0,
            rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(
            np.rad2deg(trajectory.acceleration_limit_rad_s2), 40.0,
            rtol=0.0, atol=1e-12)
        target = np.deg2rad(np.asarray(
            [1.0, -0.5, 0.25, 0.0, 0.1, -0.75, 0.4],
            dtype=np.float64,
        ))
        trajectory.start(np.zeros(7, dtype=np.float64), target)

        moving = np.abs(target) > 1e-12
        elapsed_before_done = 0.0
        for _ in range(399):
            position, _velocity, done = trajectory.step()
            self.assertFalse(done)
            elapsed_before_done += 0.005
            normalized_progress = position[moving] / target[moving]
            np.testing.assert_allclose(
                normalized_progress,
                np.full_like(normalized_progress, normalized_progress[0]),
                atol=1e-12,
            )
        self.assertAlmostEqual(elapsed_before_done, 1.995, places=9)
        position, velocity, done = trajectory.step()
        self.assertTrue(done)
        np.testing.assert_allclose(position, target, atol=1e-12)
        np.testing.assert_allclose(velocity, 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
