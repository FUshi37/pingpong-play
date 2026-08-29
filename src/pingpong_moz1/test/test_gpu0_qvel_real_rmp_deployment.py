"""Production-boundary checks for the simcode7 V85 deployment adapter."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pingpong_controller.mjx_rl_policy import (
    Gpu0QvelRealRmpV85PolicyController,
)
from pingpong_controller.rl_policy import (
    OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85,
    normalize_observation_profile,
)
from pingpong_controller.sim_mirror import (
    SimMirrorConfig,
    sim_mirror_config_for_policy_profile,
)
from pingpong_controller.tools.interactive_policy_safety_probe import (
    PolicySafetyProbe,
    RightArmCommandSafetyLimiter,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = (
    REPO_ROOT / "simcode7" / "pingpong_moz1-main" / "pingpong_controller"
    / "outputs" / "rl_sim"
    / "measured_qvel_rmp_vertical_v85_gpu0_seed20261004_20260828_stage23_failure_time_survival_online1"
    / "mjx_curriculum_best.pkl"
)
POLICY_XML = (
    REPO_ROOT / "simcode7" / "pingpong_moz1-main" / "pingpong_controller"
    / "tools" / "rl_sim" / "moz1_pd.xml"
)


def _controller() -> Gpu0QvelRealRmpV85PolicyController:
    controller = Gpu0QvelRealRmpV85PolicyController(
        str(CHECKPOINT), robot_xml_path=None)
    controller._compute_racket_pose_base = lambda _q: (
        np.asarray([0.1, -0.2, 1.0], dtype=np.float32), True)
    return controller


def test_v85_observation_and_persistent_bounded_qref_match_contract() -> None:
    controller = _controller()
    controller._policy_action = lambda _obs: np.ones(7, dtype=np.float32)
    q = np.zeros(7, dtype=np.float32)
    dq = np.linspace(0.0, 0.6, 7, dtype=np.float32)
    kwargs = dict(
        ball_pos_m=np.asarray([0.2, -0.2, 1.2], dtype=np.float32),
        ball_vel_m_s=np.asarray([0.0, 0.0, -0.3], dtype=np.float32),
        ball_valid=True,
        ball_obs_age_s=0.01,
        arm_q=q,
    )

    first = controller.predict(arm_dq=dq, **kwargs)
    first_obs = controller.last_obs.copy()
    expected_step = controller.arm_vel_limit_rad_s * np.float32(0.005)
    np.testing.assert_allclose(first, expected_step, rtol=0.0, atol=1e-7)
    np.testing.assert_array_equal(first_obs[42:49], np.zeros(7))
    np.testing.assert_allclose(first_obs[49], 0.02, rtol=0.0, atol=1e-7)
    np.testing.assert_array_equal(first_obs[50:57], np.zeros(7))
    np.testing.assert_array_equal(first_obs[29:32], np.zeros(3))

    next_dq = dq + np.float32(0.05)
    second = controller.predict(arm_dq=next_dq, **kwargs)
    second_obs = controller.last_obs.copy()
    np.testing.assert_allclose(second_obs[42:49], next_dq - dq)
    np.testing.assert_allclose(second_obs[50:57], first)
    np.testing.assert_array_equal(second_obs[35:42], np.ones(7))
    np.testing.assert_allclose(second, 2.0 * expected_step, atol=1e-7)


def test_v85_refuses_noncontract_inputs() -> None:
    controller = _controller()
    q = np.zeros(7, dtype=np.float32)
    dq = np.zeros(7, dtype=np.float32)
    common = dict(
        ball_pos_m=np.zeros(3, dtype=np.float32),
        ball_vel_m_s=np.zeros(3, dtype=np.float32),
        ball_valid=True,
        arm_q=q,
        arm_dq=dq,
    )
    with pytest.raises(RuntimeError, match="350 ms"):
        controller.predict(ball_obs_age_s=0.351, **common)
    with pytest.raises(RuntimeError, match="observation scaling"):
        controller.predict(
            ball_obs_age_s=0.0,
            observation_scales=np.full(57, 0.5, dtype=np.float32),
            **common,
        )


def test_simcode7_profile_defaults_to_cpu_mujoco_training_plant() -> None:
    base = SimMirrorConfig(
        model_path=CHECKPOINT,
        policy_xml_path=POLICY_XML,
        sim_xml_path=POLICY_XML,
        policy_obs_dim=67,
        control_dt=0.005,
        backend="mujoco",
        simcode7_model_path=CHECKPOINT,
        simcode7_policy_xml_path=POLICY_XML,
        simcode7_sim_xml_path=POLICY_XML,
    )
    resolved = sim_mirror_config_for_policy_profile(
        base, "GPU0-QVEL/REAL=RMP")
    assert normalize_observation_profile("simcode7") == (
        OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85)
    assert resolved.policy_profile == OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85
    assert resolved.policy_obs_dim == 57
    assert resolved.backend == "mujoco"
    assert resolved.rl_use_valid_ball_obs_age is True


def test_probe_uses_strict_90hz_hold_and_growing_age_for_v85() -> None:
    probe = PolicySafetyProbe.__new__(PolicySafetyProbe)
    probe.cfg = SimpleNamespace(
        control_dt=0.005,
        ball_lost_timeout_s=0.6,
        policy_ball_handover_mode="legacy_force_fresh",
    )
    probe.policy_ball_sample_step = 0
    probe.policy_ball_sample_profile = None
    probe.policy_ball_sample_pos = np.zeros(3, dtype=np.float32)
    probe.policy_ball_sample_vel = np.zeros(3, dtype=np.float32)
    probe.policy_ball_sample_has_valid = False
    probe.policy_ball_sample_age_s = 0.0
    probe.policy_ball_sample_source_stamp_s = None
    probe.policy_ball_sample_sequence = 0
    options = SimpleNamespace(
        policy_profile=OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85,
        control_mode="policy",
        camera_control_time_compensation=True,
        ball_source="camera",
        use_ball_estimator=False,
        ball_observation_rate_hz=90.0,
        policy_ball_handover_mode="legacy_force_fresh",
    )

    samples = []
    for step in range(4):
        samples.append(probe._apply_policy_ball_sampling_contract(
            np.full(3, step, dtype=np.float32),
            np.full(3, -step, dtype=np.float32),
            True,
            0.0,
            options,
            source_timestamp_s=float(step),
            force_fresh_source=True,
        ))
    np.testing.assert_array_equal(samples[0][0], np.zeros(3))
    np.testing.assert_array_equal(samples[1][0], samples[0][0])
    np.testing.assert_array_equal(samples[2][0], samples[0][0])
    np.testing.assert_array_equal(samples[3][0], np.full(3, 3.0))
    assert samples[1][3] == pytest.approx(0.005)
    assert samples[2][3] == pytest.approx(0.010)
    assert probe.last_policy_ball_handover_info["forcedRefresh"] is False

    probe.policy_ball_sample_step = 1  # a held, non-refresh policy tick
    probe.policy_ball_sample_age_s = 0.35
    stale = probe._apply_policy_ball_sampling_contract(
        np.ones(3, dtype=np.float32),
        np.zeros(3, dtype=np.float32),
        True,
        0.0,
        options,
    )
    assert stale[2] is False
    assert stale[3] == pytest.approx(0.355)


def test_limiter_cannot_be_reenabled_for_policy_or_parking_commands() -> None:
    """Even a stale true config cannot restore policy/hold filtering."""
    probe = PolicySafetyProbe.__new__(PolicySafetyProbe)
    published: list[np.ndarray] = []
    probe.cfg = SimpleNamespace(
        enable_safety_limiter=True,
        workspace_safety_mode="off",
        policy_dq_mode="raw_coherent",
        policy_plant_matcher_mode="off",
    )
    probe.ros_bridge = SimpleNamespace(
        publish_right_arm_command=lambda q: published.append(
            np.asarray(q, dtype=np.float64).copy()),
    )
    probe.safety_limiter = RightArmCommandSafetyLimiter(
        np.zeros(7, dtype=np.float64), dt=0.005)
    probe.command_stream_active = True
    probe.done_failsafe_active = False
    probe.hard_workspace_stop_latched = False
    probe.soft_workspace_reset_latched = False
    probe.compensation_was_enabled = False
    probe.compensator = None
    probe.policy_plant_matcher = None
    probe.workspace_soft_filter = {}
    probe.gpu0_rmp_safety_intervention_streak = 2
    probe.last_command_disable_hold = {}
    options = SimpleNamespace(
        publish_commands=True,
        policy_profile=OBSERVATION_PROFILE_GPU0_QVEL_REAL_RMP_V85,
        control_mode="policy",
        workspace_safety_mode="off",
    )

    did_publish = probe._publish_control_command(
        np.full(7, 0.5, dtype=np.float64),
        options,
        profile="parking",
    )

    assert did_publish is True
    assert len(published) == 1
    np.testing.assert_allclose(published[0], np.full(7, 0.5))
    np.testing.assert_array_equal(probe.safety_limiter.last_cmd, np.zeros(7))
    np.testing.assert_array_equal(probe.safety_limiter.last_vel, np.zeros(7))
    assert probe.command_stream_active is True
    assert probe.gpu0_rmp_safety_intervention_streak == 0
