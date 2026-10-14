from __future__ import annotations

import sys
from dataclasses import asdict, fields, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RL_SIM_DIR = ROOT / "pingpong_controller" / "tools" / "rl_sim"
if str(RL_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(RL_SIM_DIR))


PROFILES = ("goal_d455_autolaunch_v1", "goal_d455_release_v1")


def _stages(profile: str):
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    from train_juggle_mjx_curriculum import build_curriculum

    return build_curriculum(curriculum_profile=profile)


def test_w019_constrained_compensation_contract() -> None:
    stages = _stages("goal_d455_autolaunch_viewdense_constrained_mpc_v1")
    assert len(stages) == 20
    expected_gaps = (
        (0.080, 0.120),
        (0.060, 0.100),
        (0.040, 0.080),
        (0.020, 0.050),
    )
    for index, stage in enumerate(stages):
        cfg = stage.cfg
        assert cfg.arm_action_limiter
        assert cfg.actuator_compensation_mode == "inverse_mpc"
        assert cfg.actuator_mpc_feedback_source == "actual"
        assert cfg.actuator_mpc_beta == pytest.approx(1.2)
        assert cfg.actuator_mpc_delay_scale == pytest.approx(1.05)
        assert cfg.actuator_mpc_tau_scale == pytest.approx(0.75)
        assert not cfg.actuator_mpc_command_dynamics_constraint
        assert not cfg.arm_post_compensation_limiter
        assert not cfg.arm_servo_target_limiter
        assert cfg.arm_servo_target_tracking_planner
        assert cfg.arm_servo_target_velocity_scale == pytest.approx(1.0)
        assert cfg.arm_servo_target_acceleration_scale == pytest.approx(0.8)

        assert not cfg.arm_actual_state_limiter
        assert not cfg.arm_actual_target_tracking_governor
        assert cfg.right_arm_pd_profile == "xml"
        assert cfg.terminate_on_base_stability
        assert cfg.base_z_deviation_limit_m == pytest.approx(0.03)
        assert np.rad2deg(cfg.base_roll_pitch_limit_rad) == pytest.approx(5.0)
        assert cfg.racket_launch_surface_gap_range_m == (
            expected_gaps[index] if index < len(expected_gaps) else (0.005, 0.010)
        )


def test_w019_drbridge_preserves_plant_and_repairs_tail() -> None:
    original = _stages("goal_d455_autolaunch_viewdense_constrained_mpc_v1")
    repaired = _stages("goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v1")

    assert len(repaired) == 22
    assert [stage.name for stage in repaired[15:]] == [
        "launch15_observation_calibration_micro_bridge",
        "launch16_observation_calibration_three_eighths_bridge",
        "launch17_observation_calibration_bridge",
        "launch18_observation_calibration_three_quarters_bridge",
        "launch19_observation_calibration_wide",
        "launch20_camera_missing_wide",
        "launch21_final_consolidation",
    ]
    for source, candidate in zip(original[:15], repaired[:15], strict=True):
        assert asdict(source) == asdict(candidate)

    expected_pos_noise = (0.00475, 0.005125, 0.0055, 0.00625, 0.007)
    expected_vel_noise = (0.0475, 0.05125, 0.055, 0.0625, 0.07)
    for stage, pos_noise, vel_noise in zip(
        repaired[15:20], expected_pos_noise, expected_vel_noise, strict=True
    ):
        cfg = stage.cfg
        assert cfg.ball_obs_pos_noise_std == pytest.approx(pos_noise)
        assert cfg.ball_obs_vel_noise_std == pytest.approx(vel_noise)
        assert stage.gate_mode == "strict"
        assert stage.advance_gate_mode == "strict"
        assert cfg.arm_servo_target_tracking_planner
        assert cfg.arm_servo_target_velocity_scale == pytest.approx(1.0)
        assert cfg.arm_servo_target_acceleration_scale == pytest.approx(0.8)

        assert cfg.actuator_compensation_mode == "inverse_mpc"
        assert cfg.actuator_mpc_feedback_source == "actual"
        assert not cfg.arm_actual_state_limiter
        assert not cfg.arm_actual_target_tracking_governor

    assert asdict(repaired[-1].cfg) == asdict(original[-1].cfg)
    for field_name in (
        "target_mean_hits",
        "target_mean_len_frac",
        "target_hit1_rate",
        "target_hit3_rate",
        "target_hit12_rate",
        "target_mean_hits_ge3",
        "target_episode_truncation_rate",
        "max_recent_hit_next_contact_anchor_err",
    ):
        assert getattr(repaired[-1], field_name) == getattr(original[-1], field_name)


def test_w019_drbridge_v2_separates_learning_and_transfer_gates() -> None:
    v1 = _stages("goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v1")
    v2 = _stages("goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2")

    assert len(v2) == 22
    for source, candidate in zip(v1, v2, strict=True):
        assert asdict(source.cfg) == asdict(candidate.cfg)
    assert [stage.min_updates for stage in v2[15:21]] == [80, 80, 100, 120, 140, 160]
    assert all(stage.gate_mode == "strict" for stage in v2[15:])
    assert all(stage.advance_gate_mode == "collapse" for stage in v2[15:21])
    assert v2[-1].advance_gate_mode == "strict"
    assert v2[16].target_mean_hits == pytest.approx(9.30)
    assert v2[16].target_mean_hits_ge3 == v1[16].target_mean_hits_ge3

    # The final objective and all stage distributions/rewards/control settings
    # are unchanged; v2 changes only gate scheduling metadata.
    for field_name in (
        "target_mean_hits",
        "target_mean_len_frac",
        "target_hit1_rate",
        "target_hit3_rate",
        "target_hit12_rate",
        "target_mean_hits_ge3",
        "target_episode_truncation_rate",
        "max_recent_hit_next_contact_anchor_err",
    ):
        assert getattr(v2[-1], field_name) == getattr(v1[-1], field_name)


def test_launch17_orthogonal_bridge_v4_separates_learning_and_transfer_gates() -> None:
    stages = _stages(
        "goal_d455_autolaunch_viewdense_constrained_mpc_launch17_orthogonal_bridge_v4"
    )
    bridge = stages[17:22]

    assert [stage.name for stage in bridge] == [
        "launch17a_refresh_noise_only_long_juggle_1200",
        "launch17b_frame_dr_only_long_juggle_1200",
        "launch17c_combined_launch15_long_juggle_1200",
        "launch17d_combined_launch16_long_juggle_1200",
        "launch17_full_observation_long_juggle_1200",
    ]
    assert all(stage.gate_mode == "strict" for stage in bridge)
    assert all(stage.advance_gate_mode == "collapse" for stage in bridge)
    assert all(stage.target_mean_hits == pytest.approx(12.0) for stage in bridge)
    assert all(stage.target_mean_len_frac == pytest.approx(0.90) for stage in bridge)
    assert all(
        stage.target_episode_truncation_rate == pytest.approx(0.75)
        for stage in bridge
    )

    for stage in bridge:
        cfg = stage.cfg
        assert cfg.actuator_compensation_mode == "inverse_mpc"
        assert cfg.actuator_mpc_feedback_source == "actual"
        assert cfg.arm_servo_target_tracking_planner
        assert cfg.arm_servo_target_velocity_scale == pytest.approx(1.0)
        assert cfg.arm_servo_target_acceleration_scale == pytest.approx(0.8)

    refresh_only, frame_only = bridge[:2]
    assert refresh_only.cfg.ball_obs_pos_noise_std == pytest.approx(0.0055)
    assert refresh_only.cfg.dr_ball_obs_pos_bias_base_m == pytest.approx((0.0, 0.0, 0.0))
    assert frame_only.cfg.ball_obs_pos_noise_std == pytest.approx(0.0)
    assert frame_only.cfg.dr_ball_obs_pos_bias_base_m == pytest.approx(
        (0.004, 0.004, 0.004)
    )


def test_launch17_obsres2mm_servo_v5_changes_only_measured_observation_fields() -> None:
    v4 = _stages(
        "goal_d455_autolaunch_viewdense_constrained_mpc_launch17_orthogonal_bridge_v4"
    )
    v5 = _stages(
        "goal_d455_autolaunch_viewdense_constrained_mpc_launch17_obsres2mm_servo_v5"
    )

    assert len(v5) == 21
    assert v5[:19] == v4[:19]
    assert v5[19].name == "launch17c_measured_obsres2mm_servo_long_juggle_1200"
    assert v5[20].name == "launch19_final_measured_obsres2mm_servo_consolidation"

    allowed_changes = {
        "ball_obs_pos_noise_std",
        "ball_obs_vel_noise_std",
        "dr_ball_obs_pos_bias_base_m",
        "dr_ball_obs_rot_bias_deg",
        "dr_ball_obs_vel_bias_base_m_s",
        "dr_ball_obs_scale_range",
    }
    for source, repaired in ((v4[19], v5[19]), (v4[-1], v5[-1])):
        changed = {
            field.name
            for field in fields(type(source.cfg))
            if getattr(source.cfg, field.name) != getattr(repaired.cfg, field.name)
        }
        assert changed <= allowed_changes
        assert changed >= allowed_changes - {"ball_obs_vel_noise_std"}
        cfg = repaired.cfg
        assert cfg.ball_obs_pos_noise_std == pytest.approx(0.002)
        assert cfg.ball_obs_vel_noise_std == pytest.approx(0.07)
        assert cfg.dr_ball_obs_pos_bias_base_m == pytest.approx((0.002, 0.002, 0.002))
        assert cfg.dr_ball_obs_rot_bias_deg == pytest.approx((0.0, 0.0, 0.0))
        assert cfg.dr_ball_obs_vel_bias_base_m_s == pytest.approx((0.0, 0.0, 0.0))
        assert cfg.dr_ball_obs_scale_range == pytest.approx((1.0, 1.0))
        assert cfg.ball_obs_camera_missing_prob == pytest.approx(0.0)
        assert cfg.ball_obs_view_bounds_missing_prob == pytest.approx(0.0)
        assert cfg.ball_obs_dropout_prob == pytest.approx(0.0)
        assert cfg.ball_obs_burst_dropout_prob == pytest.approx(0.0)
        assert cfg.actuator_compensation_mode == "inverse_mpc"
        assert cfg.actuator_mpc_feedback_source == "actual"
        assert cfg.arm_servo_target_tracking_planner
        assert cfg.arm_servo_target_velocity_scale == pytest.approx(1.0)
        assert cfg.arm_servo_target_acceleration_scale == pytest.approx(0.8)


    launch17, final = v5[-2:]
    assert launch17.gate_mode == "strict"
    assert launch17.advance_gate_mode == "collapse"
    assert launch17.target_mean_hits == pytest.approx(12.0)
    assert launch17.target_mean_len_frac == pytest.approx(0.90)
    assert launch17.target_episode_truncation_rate == pytest.approx(0.75)
    assert final.gate_mode == "strict"
    assert final.advance_gate_mode == "strict"
    assert final.target_mean_hits == pytest.approx(13.0)
    assert final.target_mean_len_frac == pytest.approx(0.95)
    assert final.target_episode_truncation_rate == pytest.approx(0.86)


def test_sport_second_order_actuator_dr_follows_actuator_stage_gate() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    from train_juggle_mjx_curriculum import build_curriculum

    stages = build_curriculum(
        gate_preset="v7_strict",
        curriculum_profile="goal_d455_autolaunch_sport_actuator_obsres2mm_nocomp_v1",
        delay_ablation_preset="sport_actuator_replay_dr",
        actuator_compensation_mode="none",
        arm_servo_target_tracking_planner=False,
        asymmetric_critic=True,
        critic_command_history_steps=12,
    )

    assert [stage.name for stage in stages[:6]] == [
        "launch00_acquisition",
        "launch01_local_workspace",
        "launch02_workspace",
        "launch03_ball_dynamics_mild",
        "launch04_contact_dynamics_mild",
        "launch05_actuator_pd_mild",
    ]
    assert all(
        not stage.cfg.dr_randomize_second_order_actuator for stage in stages[:5]
    )
    assert stages[5].cfg.dr_randomize_actuator
    assert stages[5].cfg.dr_randomize_second_order_actuator
    assert all(
        stage.cfg.dr_randomize_second_order_actuator
        == stage.cfg.dr_randomize_actuator
        for stage in stages
    )


def test_sport_successref_profile_builds_hit_ladder_before_reference_dr_tail() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    from train_juggle_mjx_curriculum import GOAL_D455_PROFILES, build_curriculum

    assert "goal_d455_sport_taskspace_successref_obsres2mm_nocomp_v1" in GOAL_D455_PROFILES

    sport = build_curriculum(
        gate_preset="v7_strict",
        curriculum_profile="goal_d455_sport_taskspace_successref_obsres2mm_nocomp_v1",
        delay_ablation_preset="sport_actuator_replay_dr",
        actuator_compensation_mode="none",
        arm_servo_target_tracking_planner=False,
        asymmetric_critic=True,
        critic_command_history_steps=12,
    )

    assert len(sport) == 25
    assert [stage.name for stage in sport[:7]] == [
        "sport00_first_hit_acquisition",
        "sport01_two_hit_acquisition",
        "sport02_three_hit_acquisition",
        "sport03_four_hit_recovery",
        "sport04_centered_six_hit",
        "sport05_nominal_seven_hit",
        "sport06_nominal_nine_hit",
    ]
    assert [stage.target_mean_hits for stage in sport[:7]] == pytest.approx(
        [0.95, 1.80, 2.65, 3.40, 5.0, 7.0, 8.0]
    )
    assert [stage.target_mean_hits_ge3 for stage in sport[2:7]] == pytest.approx(
        [3.0, 3.8, 5.6, 7.6, 8.3]
    )
    assert [stage.target_mean_hits_ge3 for stage in sport[2:]] == sorted(
        stage.target_mean_hits_ge3 for stage in sport[2:]
    )
    assert [stage.target_mean_hits for stage in sport] == sorted(
        stage.target_mean_hits for stage in sport
    )
    assert [stage.target_episode_truncation_rate for stage in sport[3:]] == sorted(
        stage.target_episode_truncation_rate for stage in sport[3:]
    )

    # Acquisition is unconstrained by final-task centre-return penalties.
    for stage in sport[:3]:
        assert stage.cfg.post_hit_ball_vxy_penalty_weight == pytest.approx(0.0)
        assert stage.cfg.hit_vxy_penalty_weight == pytest.approx(0.0)
        assert stage.cfg.hit_next_contact_anchor_penalty_weight == pytest.approx(0.0)
    assert [stage.cfg.hit_reward_combo for stage in sport[:4]] == pytest.approx(
        [0.0, 0.03, 0.06, 0.08]
    )
    assert sport[3].cfg.hit_vxy_penalty_weight == pytest.approx(0.20)
    assert sport[6].cfg.hit_vxy_penalty_weight == pytest.approx(0.90)

    # Ball and contact DR remain separated from actuator DR.  The two
    # inherited bridge stages follow the seven-stage nominal hit ladder.
    assert sport[7].name == "launch03_ball_dynamics_mild"
    assert sport[8].name == "launch04_contact_dynamics_mild"
    assert sport[9].name == "launch05_actuator_pd_mild"
    assert all(not stage.cfg.dr_randomize_second_order_actuator for stage in sport[:9])
    assert sport[9].cfg.dr_randomize_actuator
    assert sport[9].cfg.dr_randomize_second_order_actuator

    for stage in sport:
        assert stage.cfg.right_arm_pd_profile == "sport_taskspace_fit_v1"
        assert stage.cfg.actuator_compensation_mode == "none"
        assert not stage.cfg.arm_servo_target_tracking_planner

    final = sport[-1]
    assert final.name == "launch19_final_measured_obsres2mm_sport_nocomp_consolidation"
    assert final.target_mean_hits == pytest.approx(13.0)
    assert final.target_mean_hits_ge3 == pytest.approx(13.5)
    assert final.target_mean_len_frac == pytest.approx(0.95)
    assert final.target_episode_truncation_rate == pytest.approx(0.86)
    assert final.cfg.ball_obs_pos_noise_std == pytest.approx(0.002)
    assert final.cfg.ball_obs_vel_noise_std == pytest.approx(0.07)
    assert final.cfg.dr_ball_obs_pos_bias_base_m == pytest.approx((0.002, 0.002, 0.002))
    assert final.cfg.dr_ball_obs_rot_bias_deg == pytest.approx((0.0, 0.0, 0.0))
    assert final.cfg.dr_ball_obs_vel_bias_base_m_s == pytest.approx((0.0, 0.0, 0.0))
    assert final.cfg.dr_ball_obs_scale_range == pytest.approx((1.0, 1.0))
    assert final.cfg.ball_obs_dropout_prob == pytest.approx(0.0)


def test_sport_direct_profile_is_exact_reference_course_with_actuator_adaptation() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    from train_juggle_mjx_curriculum import GOAL_D455_PROFILES, build_curriculum

    profile = "goal_d455_sport_taskspace_obsres2mm_nocomp_direct_v1"
    removed_profiles = {
        "goal_d455_sport_taskspace_successref_obsres2mm_nocomp_v2",
        "goal_d455_sport_taskspace_successref_count_ablation_v1",
        "goal_d455_sport_taskspace_successref_recover_ablation_v1",
        "goal_d455_sport_taskspace_successref_combined_ablation_v1",
        "goal_d455_sport_taskspace_successref_indexed_recovery_ablation_v1",
        "goal_d455_sport_taskspace_successref_preparatory_ablation_v1",
        "goal_d455_sport_taskspace_successref_contact_velocity_ablation_v1",
        "goal_d455_sport_taskspace_successref_contact_quality_ablation_v1",
        "goal_d455_sport_taskspace_successref_posture003_ablation_v1",
        "goal_d455_sport_taskspace_successref_posture010_ablation_v1",
        "goal_d455_sport_taskspace_successref_posture100_ablation_v1",
    }
    assert profile in GOAL_D455_PROFILES
    assert removed_profiles.isdisjoint(GOAL_D455_PROFILES)

    stages = build_curriculum(
        gate_preset="v7_strict",
        curriculum_profile=profile,
        delay_ablation_preset="sport_actuator_replay_dr",
        actuator_compensation_mode="none",
        arm_servo_target_tracking_planner=False,
        asymmetric_critic=True,
        critic_command_history_steps=12,
    )

    assert len(stages) == 21
    assert [stage.name for stage in stages[:6]] == [
        "launch00_acquisition",
        "launch01_local_workspace",
        "launch02_workspace",
        "launch03_ball_dynamics_mild",
        "launch04_contact_dynamics_mild",
        "launch05_actuator_pd_mild",
    ]
    assert all(not stage.name.startswith("sport") for stage in stages)
    assert [stage.target_mean_hits for stage in stages[:5]] == pytest.approx(
        [1.0, 2.0, 3.2, 4.5, 5.8]
    )
    assert [stage.target_episode_truncation_rate for stage in stages] == pytest.approx(
        [
            0.02, 0.05, 0.10, 0.16, 0.23, 0.31, 0.40, 0.48, 0.54,
            0.59, 0.64, 0.68, 0.72, 0.75, 0.78, 0.50, 0.42, 0.75,
            0.75, 0.75, 0.86,
        ]
    )

    assert all(not stage.cfg.dr_randomize_actuator for stage in stages[:5])
    assert all(not stage.cfg.dr_randomize_second_order_actuator for stage in stages[:5])
    assert stages[5].cfg.dr_randomize_actuator
    assert stages[5].cfg.dr_randomize_second_order_actuator
    assert all(
        stage.cfg.dr_randomize_second_order_actuator
        == stage.cfg.dr_randomize_actuator
        for stage in stages
    )
    assert stages[0].cfg.hit_height_penalty_weight == pytest.approx(0.0)
    assert all(stage.cfg.hit_height_penalty_weight >= 3.0 for stage in stages[1:])

    assert stages[-1].target_mean_hits_ge3 == pytest.approx(13.5)
    for stage in stages:
        assert stage.cfg.posture_weight >= 0.02
        assert stage.cfg.arm_posture_penalty_weight >= 0.10
        assert stage.cfg.arm_command_posture_penalty_weight >= 0.06
        assert stage.cfg.arm_posture_soft_limit_penalty_weight >= 0.80
        assert stage.cfg.arm_velocity_usage_penalty_weight >= 0.04
        assert stage.cfg.arm_acceleration_usage_penalty_weight >= 0.015
        assert stage.cfg.arm_vel_limit_penalty_weight >= 0.06
        assert stage.cfg.arm_acc_limit_penalty_weight >= 0.08
        assert stage.cfg.arm_limiter_penalty_weight >= 0.05
        assert stage.cfg.enable_anti_windup
        assert stage.cfg.anti_windup_error_threshold == pytest.approx(0.25)
        assert stage.cfg.anti_windup_min_scale == pytest.approx(0.05)
        assert stage.cfg.arm_vel_limit_deg_s == pytest.approx(
            (210.0, 210.0, 240.0, 240.0, 300.0, 300.0, 300.0)
        )
        assert stage.cfg.arm_acc_limit_deg_s2 == pytest.approx(
            (1300.0, 1300.0, 1800.0, 3000.0, 3000.0, 3000.0, 3000.0)
        )
        assert stage.cfg.action_delta_penalty_weight >= 0.0012
        assert stage.cfg.delay_action_jerk_penalty_weight >= 3.0e-7
        assert stage.cfg.actuator_cmd_model == "second_order"
        assert stage.cfg.actuator_cmd_delay_ms_per_joint == pytest.approx(
            (45.0, 50.0, 45.0, 40.0, 35.0, 45.0, 55.0)
        )
        assert stage.cfg.actuator_compensation_mode == "none"
        assert not stage.cfg.arm_servo_target_tracking_planner

    from train_juggle_mjx_curriculum import stage_best_score

    common_score_row = {
        "convergence/recent_mean_hits": 4.0,
        "convergence/recent_mean_len_frac": 0.4,
        "convergence/recent_mean_return": 8.0,
        "arm_posture_soft_exceed_fraction": 0.0,
        "arm_command_posture_soft_exceed_fraction": 0.0,
        "arm_qvel_limit_exceed_fraction": 0.0,
        "arm_qacc_limit_exceed_fraction": 0.0,
        "arm_posture_error_max_rad": 0.25,
        "arm_command_posture_error_max_rad": 0.20,
    }
    assert stage_best_score(common_score_row, stages[0]) is not None
    assert stage_best_score(
        {**common_score_row, "arm_posture_soft_exceed_fraction": 0.03},
        stages[0],
    ) is None

    final = stages[-1]
    assert final.name == "launch19_final_measured_obsres2mm_sport_nocomp_consolidation"
    assert final.cfg.ball_obs_pos_noise_std == pytest.approx(0.002)
    assert final.cfg.dr_ball_obs_pos_bias_base_m == pytest.approx((0.002, 0.002, 0.002))

def test_w020_countcredit_changes_only_terminal_credit_on_w019_v2() -> None:
    baseline = _stages(
        "goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2"
    )
    repaired = _stages(
        "goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2_countcredit_v1"
    )

    expected_changes = {
        "hit_reward_cap_mode",
        "hit_reward_count_cap",
        "termination_miss_penalty_base",
        "termination_miss_penalty_per_hit",
        "racket_z_limit_termination_penalty_base",
        "racket_z_limit_termination_penalty_per_hit",
        "racket_anchor_termination_penalty_base",
    }
    assert len(baseline) == len(repaired) == 22
    for before, after in zip(baseline, repaired, strict=True):
        changed = {
            field.name
            for field in fields(type(before.cfg))
            if getattr(before.cfg, field.name) != getattr(after.cfg, field.name)
        }
        assert changed == expected_changes
        assert replace(before, cfg=after.cfg, notes=after.notes) == after
        reference_hits = max(1.0, float(before.target_mean_hits))
        miss_barrier = (
            before.cfg.termination_miss_penalty_base
            + before.cfg.termination_miss_penalty_per_hit * reference_hits
        )
        racket_barrier = (
            before.cfg.racket_z_limit_termination_penalty_base
            + before.cfg.racket_z_limit_termination_penalty_per_hit * reference_hits
        )
        assert after.cfg.hit_reward_cap_mode == "off"
        assert after.cfg.hit_reward_count_cap == 0
        assert after.cfg.termination_miss_penalty_base == pytest.approx(miss_barrier)
        assert after.cfg.termination_miss_penalty_per_hit == pytest.approx(0.0)
        assert after.cfg.racket_z_limit_termination_penalty_base == pytest.approx(
            racket_barrier
        )
        assert after.cfg.racket_z_limit_termination_penalty_per_hit == pytest.approx(0.0)
        assert after.cfg.racket_anchor_termination_penalty_base == pytest.approx(
            max(before.cfg.racket_anchor_termination_penalty_base, miss_barrier, racket_barrier)
        )


def test_w021_recoverability_changes_tail_rewards_not_gates() -> None:
    baseline = _stages(
        "goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2_countcredit_v1"
    )
    repaired = _stages(
        "goal_d455_autolaunch_viewdense_constrained_mpc_recoverability_v1"
    )

    expected_cfg_changes = {
        "post_hit_ball_vxy_penalty_weight",
        "hit_vxy_penalty_weight",
        "hit_next_contact_anchor_penalty_weight",
    }
    assert len(baseline) == len(repaired) == 22
    for index, (before, after) in enumerate(
        zip(baseline, repaired, strict=True)
    ):
        if index < 17:
            assert asdict(before) == asdict(after)
            continue
        changed = {
            field.name
            for field in fields(type(before.cfg))
            if getattr(before.cfg, field.name) != getattr(after.cfg, field.name)
        }
        assert changed == expected_cfg_changes
        assert replace(before, cfg=after.cfg, notes=after.notes) == after
        assert after.cfg.post_hit_ball_vxy_penalty_weight == pytest.approx(0.18)
        assert after.cfg.hit_vxy_penalty_weight == pytest.approx(0.90)
        assert after.cfg.hit_next_contact_anchor_penalty_weight >= 0.06


def test_w022_intercept_adds_only_execution_aware_tail_reward() -> None:
    baseline = _stages(
        "goal_d455_autolaunch_viewdense_constrained_mpc_recoverability_v1"
    )
    repaired = _stages(
        "goal_d455_autolaunch_viewdense_constrained_mpc_intercept_v1"
    )

    assert len(baseline) == len(repaired) == 22
    for index, (before, after) in enumerate(
        zip(baseline, repaired, strict=True)
    ):
        if index < 17:
            assert asdict(before) == asdict(after)
            continue
        changed = {
            field.name
            for field in fields(type(before.cfg))
            if getattr(before.cfg, field.name) != getattr(after.cfg, field.name)
        }
        assert changed == {"descending_intercept_reward_weight"}
        assert after.cfg.descending_intercept_reward_weight == pytest.approx(1.20)
        assert replace(before, cfg=after.cfg, notes=after.notes) == after


def test_w024_count_progress_changes_only_tail_combo_reward() -> None:
    baseline = _stages(
        "goal_d455_autolaunch_viewdense_constrained_mpc_intercept_v1"
    )
    repaired = _stages(
        "goal_d455_autolaunch_viewdense_constrained_mpc_count_progress_v1"
    )

    assert len(baseline) == len(repaired) == 22
    for index, (before, after) in enumerate(
        zip(baseline, repaired, strict=True)
    ):
        if index < 17:
            assert asdict(before) == asdict(after)
            continue
        changed = {
            field.name
            for field in fields(type(before.cfg))
            if getattr(before.cfg, field.name) != getattr(after.cfg, field.name)
        }
        assert changed == {"hit_reward_combo"}
        assert before.cfg.hit_reward_combo == pytest.approx(0.0)
        assert after.cfg.hit_reward_combo == pytest.approx(0.08)
        assert after.cfg.hit_combo_count_cap == before.cfg.hit_combo_count_cap == 14
        assert replace(before, cfg=after.cfg, notes=after.notes) == after


def test_actuator_final_recovery_changes_only_launch19_reward_contract() -> None:
    baseline = _stages(
        "goal_d455_autolaunch_actuator_inversempc_successref_nogov_v1"
    )
    recovered = _stages(
        "goal_d455_autolaunch_actuator_inversempc_final_recovery_nogov_v1"
    )

    assert len(baseline) == len(recovered) == 20
    for before, after in zip(baseline[:-1], recovered[:-1], strict=True):
        assert asdict(before) == asdict(after)

    before = baseline[-1]
    after = recovered[-1]
    expected_changes = {
        "hit_reward_cap_mode",
        "hit_reward_count_cap",
        "termination_miss_penalty_base",
        "termination_miss_penalty_per_hit",
        "racket_z_limit_termination_penalty_base",
        "racket_z_limit_termination_penalty_per_hit",
        "racket_anchor_termination_penalty_base",
        "post_hit_survival_reward_weight",
        "hit_next_contact_anchor_penalty_weight",
    }
    changed = {
        field.name
        for field in fields(type(before.cfg))
        if getattr(before.cfg, field.name) != getattr(after.cfg, field.name)
    }
    assert changed == expected_changes
    assert replace(before, cfg=after.cfg, notes=after.notes) == after

    reference_hits = float(before.target_mean_hits)
    miss_barrier = (
        before.cfg.termination_miss_penalty_base
        + before.cfg.termination_miss_penalty_per_hit * reference_hits
    )
    racket_barrier = (
        before.cfg.racket_z_limit_termination_penalty_base
        + before.cfg.racket_z_limit_termination_penalty_per_hit * reference_hits
    )
    assert after.cfg.hit_reward_cap_mode == "off"
    assert after.cfg.hit_reward_count_cap == 0
    assert after.cfg.termination_miss_penalty_base == pytest.approx(miss_barrier)
    assert after.cfg.termination_miss_penalty_per_hit == pytest.approx(0.0)
    assert after.cfg.racket_z_limit_termination_penalty_base == pytest.approx(
        racket_barrier
    )
    assert after.cfg.racket_z_limit_termination_penalty_per_hit == pytest.approx(0.0)
    assert after.cfg.racket_anchor_termination_penalty_base == pytest.approx(
        max(before.cfg.racket_anchor_termination_penalty_base, miss_barrier, racket_barrier)
    )
    assert after.cfg.post_hit_survival_reward_weight == pytest.approx(2.4)
    assert after.cfg.hit_next_contact_anchor_penalty_weight == pytest.approx(0.12)


def test_actuator_final_cadence_changes_only_launch19_cadence_contract() -> None:
    baseline = _stages(
        "goal_d455_autolaunch_actuator_inversempc_successref_nogov_v1"
    )
    cadence = _stages(
        "goal_d455_autolaunch_actuator_inversempc_final_cadence_nogov_v1"
    )

    assert len(baseline) == len(cadence) == 20
    for before, after in zip(baseline[:-1], cadence[:-1], strict=True):
        assert asdict(before) == asdict(after)

    before = baseline[-1]
    after = cadence[-1]
    changed = {
        field.name
        for field in fields(type(before.cfg))
        if getattr(before.cfg, field.name) != getattr(after.cfg, field.name)
    }
    assert changed == {
        "hit_cadence_reward_weight",
        "hit_cadence_target_interval",
        "hit_cadence_sigma",
        "hit_min_interval",
        "hit_min_interval_penalty_weight",
        "fast_hit_penalty_weight",
    }
    assert after.cfg.hit_cadence_reward_weight == pytest.approx(0.30)
    assert after.cfg.hit_cadence_target_interval == pytest.approx(0.45)
    assert after.cfg.hit_cadence_sigma == pytest.approx(0.05)
    assert after.cfg.hit_min_interval == pytest.approx(0.38)
    assert after.cfg.hit_min_interval_penalty_weight == pytest.approx(0.50)
    assert after.cfg.fast_hit_penalty_weight == pytest.approx(0.50)
    assert after.target_max_hit_interval_s == pytest.approx(0.47)


def test_stage_best_score_penalizes_cadence_outside_gate_band() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    from train_juggle_mjx_curriculum import stage_best_score

    stage = _stages(
        "goal_d455_autolaunch_actuator_inversempc_final_cadence_nogov_v1"
    )[-1]
    common = {
        "convergence/recent_mean_hits": 11.0,
        "convergence/recent_mean_len_frac": 0.8,
        "convergence/recent_mean_return": 16.0,
    }
    in_band = stage_best_score(
        {**common, "convergence/recent_mean_hit_interval_ge3_s": 0.45},
        stage,
    )
    too_slow = stage_best_score(
        {**common, "convergence/recent_mean_hit_interval_ge3_s": 0.49},
        stage,
    )
    too_fast = stage_best_score(
        {**common, "convergence/recent_mean_hit_interval_ge3_s": 0.30},
        stage,
    )
    assert in_band is not None and too_slow is not None and too_fast is not None
    assert in_band > too_slow
    assert in_band > too_fast


def test_sport_actuator_learning_ablation_presets_isolate_delay_and_overshoot() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    from train_juggle_mjx_curriculum import _delay_conditioned_control_kwargs

    ideal = _delay_conditioned_control_kwargs("sport_actuator_ablation_ideal")
    delay = _delay_conditioned_control_kwargs(
        "sport_actuator_ablation_delay_only"
    )
    overshoot = _delay_conditioned_control_kwargs(
        "sport_actuator_ablation_overshoot_only"
    )
    full = _delay_conditioned_control_kwargs("sport_actuator_replay_fit")

    assert all(cfg["right_arm_pd_profile"] == "sport_taskspace_fit_v1" for cfg in (ideal, delay, overshoot, full))
    assert all(cfg["include_command_state"] for cfg in (ideal, delay, overshoot, full))
    assert all(cfg["include_active_command_error"] for cfg in (ideal, delay, overshoot, full))
    assert all(cfg["include_phase_features"] for cfg in (ideal, delay, overshoot, full))

    assert not ideal["actuator_cmd_filter"]
    assert ideal["delay_min_ms"] == pytest.approx(0.0)
    assert not delay["actuator_cmd_filter"]
    assert delay["delay_min_ms"] == pytest.approx(45.0)
    assert overshoot["actuator_cmd_filter"]
    assert overshoot["actuator_cmd_model"] == "second_order"
    assert overshoot["actuator_cmd_delay_ms_per_joint"] == pytest.approx((0.0,) * 7)
    assert full["actuator_cmd_filter"]
    assert full["actuator_cmd_delay_ms_per_joint"] == pytest.approx(
        (45.0, 50.0, 45.0, 40.0, 35.0, 45.0, 55.0)
    )


def test_actuator_final_survival_changes_only_launch19_recoverability() -> None:
    cadence = _stages(
        "goal_d455_autolaunch_actuator_inversempc_final_cadence_nogov_v1"
    )
    survival = _stages(
        "goal_d455_autolaunch_actuator_inversempc_final_survival_nogov_v1"
    )

    assert len(cadence) == len(survival) == 20
    for before, after in zip(cadence[:-1], survival[:-1], strict=True):
        assert asdict(before) == asdict(after)

    before = cadence[-1]
    after = survival[-1]
    changed = {
        field.name
        for field in fields(type(before.cfg))
        if getattr(before.cfg, field.name) != getattr(after.cfg, field.name)
    }
    assert changed == {
        "post_hit_survival_reward_weight",
        "post_hit_ball_vxy_penalty_weight",
        "hit_vxy_penalty_weight",
        "hit_next_contact_anchor_penalty_weight",
    }
    assert after.cfg.post_hit_survival_reward_weight == pytest.approx(1.70)
    assert after.cfg.post_hit_ball_vxy_penalty_weight == pytest.approx(0.18)
    assert after.cfg.hit_vxy_penalty_weight == pytest.approx(0.90)
    assert after.cfg.hit_next_contact_anchor_penalty_weight == pytest.approx(0.06)
    assert replace(after, cfg=before.cfg, notes=before.notes) == before


def test_actuator_final_obsres2mm_changes_only_launch19_observation_residual() -> None:
    survival = _stages(
        "goal_d455_autolaunch_actuator_inversempc_final_survival_nogov_v1"
    )
    obsres = _stages(
        "goal_d455_autolaunch_actuator_inversempc_final_obsres2mm_nogov_v1"
    )

    assert len(survival) == len(obsres) == 20
    for before, after in zip(survival[:-1], obsres[:-1], strict=True):
        assert asdict(before) == asdict(after)

    before = survival[-1]
    after = obsres[-1]
    changed = {
        field.name
        for field in fields(type(before.cfg))
        if getattr(before.cfg, field.name) != getattr(after.cfg, field.name)
    }
    assert changed == {
        "ball_obs_pos_noise_std",
        "dr_ball_obs_pos_bias_base_m",
        "dr_ball_obs_rot_bias_deg",
        "dr_ball_obs_vel_bias_base_m_s",
        "dr_ball_obs_scale_range",
    }
    assert after.cfg.ball_obs_pos_noise_std == pytest.approx(0.002)
    assert after.cfg.dr_randomize_ball_obs_frame is True
    assert after.cfg.dr_ball_obs_pos_bias_base_m == pytest.approx((0.002, 0.002, 0.002))
    assert after.cfg.dr_ball_obs_rot_bias_deg == pytest.approx((0.0, 0.0, 0.0))
    assert after.cfg.dr_ball_obs_vel_bias_base_m_s == pytest.approx((0.0, 0.0, 0.0))
    assert after.cfg.dr_ball_obs_scale_range == pytest.approx((1.0, 1.0))
    assert after.target_mean_hits == before.target_mean_hits
    assert after.target_mean_len_frac == before.target_mean_len_frac
    assert after.target_episode_truncation_rate == before.target_episode_truncation_rate
    assert replace(after, cfg=before.cfg, notes=before.notes) == before


def test_actuator_final_survival_countcredit_preserves_gate_and_fixes_credit() -> None:
    survival = _stages(
        "goal_d455_autolaunch_actuator_inversempc_final_survival_nogov_v1"
    )
    repaired = _stages(
        "goal_d455_autolaunch_actuator_inversempc_final_survival_countcredit_nogov_v1"
    )

    assert len(survival) == len(repaired) == 20
    for before, after in zip(survival[:-1], repaired[:-1], strict=True):
        assert asdict(before) == asdict(after)

    before = survival[-1]
    after = repaired[-1]
    assert replace(after, cfg=before.cfg, notes=before.notes) == before
    assert after.target_mean_hits == before.target_mean_hits == pytest.approx(13.0)
    assert after.target_mean_len_frac == before.target_mean_len_frac
    assert after.target_hit3_rate == before.target_hit3_rate
    assert after.target_hit12_rate == before.target_hit12_rate

    expected_miss = before.cfg.termination_miss_penalty_base + (
        before.cfg.termination_miss_penalty_per_hit * before.target_mean_hits
    )
    expected_racket = before.cfg.racket_z_limit_termination_penalty_base + (
        before.cfg.racket_z_limit_termination_penalty_per_hit
        * before.target_mean_hits
    )
    assert expected_miss == pytest.approx(12.9)
    assert expected_racket == pytest.approx(15.5)
    assert after.cfg.termination_miss_penalty_base == pytest.approx(expected_miss)
    assert after.cfg.termination_miss_penalty_per_hit == pytest.approx(0.0)
    assert after.cfg.racket_z_limit_termination_penalty_base == pytest.approx(
        expected_racket
    )
    assert after.cfg.racket_z_limit_termination_penalty_per_hit == pytest.approx(0.0)
    assert after.cfg.racket_anchor_termination_penalty_base == pytest.approx(
        expected_racket
    )


def test_actuator_final_missing_age_matches_sensor_contract_without_gate_change() -> None:
    countcredit = _stages(
        "goal_d455_autolaunch_actuator_inversempc_final_survival_countcredit_nogov_v1"
    )
    missing_age = _stages(
        "goal_d455_autolaunch_actuator_inversempc_final_missing_age_nogov_v1"
    )

    assert len(countcredit) == 20
    assert len(missing_age) == 23
    for before, after in zip(countcredit[:-1], missing_age[:19], strict=True):
        assert asdict(before) == asdict(after)

    before = countcredit[-1]
    assert [s.cfg.ball_obs_view_z_high_missing_range_m for s in missing_age[19:]] == [
        (1.80, 1.80),
        (1.65, 1.65),
        (1.55, 1.55),
        before.cfg.ball_obs_view_z_high_missing_range_m,
    ]
    for after in missing_age[19:]:
        assert after.target_mean_hits == before.target_mean_hits == pytest.approx(13.0)
        assert after.target_mean_len_frac == before.target_mean_len_frac
        assert after.target_hit12_rate == before.target_hit12_rate
        assert after.gate_mode == before.gate_mode
        assert after.advance_gate_mode == before.advance_gate_mode
        assert not after.cfg.terminate_on_ball_view_bounds
        assert not after.cfg.ball_obs_require_camera_visible
        assert not after.cfg.ball_obs_require_view_bounds
        assert after.cfg.ball_obs_require_view_z_high
        assert after.cfg.ball_obs_missing_episode_coherent_prob == pytest.approx(0.0)
        assert after.cfg.ball_obs_age_tracks_stale
        assert not after.cfg.ball_obs_reset_respects_camera_visibility


def test_actuator_final_intercept_changes_only_launch19_intercept_reward() -> None:
    survival = _stages(
        "goal_d455_autolaunch_actuator_inversempc_final_survival_nogov_v1"
    )
    intercept = _stages(
        "goal_d455_autolaunch_actuator_inversempc_final_intercept_nogov_v1"
    )

    assert len(survival) == len(intercept) == 20
    for before, after in zip(survival[:-1], intercept[:-1], strict=True):
        assert asdict(before) == asdict(after)

    before = survival[-1]
    after = intercept[-1]
    changed = {
        field.name
        for field in fields(type(before.cfg))
        if getattr(before.cfg, field.name) != getattr(after.cfg, field.name)
    }
    assert changed == {"descending_intercept_reward_weight"}
    assert after.cfg.descending_intercept_reward_weight == pytest.approx(1.20)
    assert replace(after, cfg=before.cfg, notes=before.notes) == before


def test_goal_profiles_fix_camera_stack_reward_reset_and_budgets() -> None:
    from camera_calibration import (
        D455_848_UNDISTORTED_HEIGHT,
        D455_848_UNDISTORTED_HFOV_DEG,
        D455_848_UNDISTORTED_SIM_BASE_BODY,
        D455_848_UNDISTORTED_VFOV_DEG,
        D455_848_UNDISTORTED_WIDTH,
    )
    from mjx_juggle_env import MjxJuggleConfig
    from train_juggle_mjx_curriculum import (
        GOAL_D455_AUTOLAUNCH_TAIL_NEXT_CONTACT_PENALTY_WEIGHT,
        GOAL_D455_RELEASE_NEXT_CONTACT_PENALTY_WEIGHT,
    )

    branches = {profile: _stages(profile) for profile in PROFILES}
    for profile, stages in branches.items():
        expected_stage_count = 20 if profile.endswith("autolaunch_v1") else 18
        expected_final_index = 19 if profile.endswith("autolaunch_v1") else 17
        assert len(stages) == expected_stage_count
        assert stages[0].name.endswith("00_acquisition")
        assert stages[-1].name.endswith(f"{expected_final_index:02d}_final_consolidation")
        assert stages[-1].target_mean_hits == pytest.approx(13.0)
        assert stages[-1].target_mean_len_frac == pytest.approx(0.95)
        expected_hit12 = 0.76 if profile.endswith("autolaunch_v1") else 0.78
        assert stages[-1].target_hit12_rate == pytest.approx(expected_hit12)
        assert stages[-1].target_episode_truncation_rate == pytest.approx(0.86)
        assert [stage.min_updates for stage in stages] == sorted(
            stage.min_updates for stage in stages
        )
        assert all(stage.max_updates is None for stage in stages)
        expected_gate_modes = (
            ["strict"] * 15 + ["balanced_probe"] * 4 + ["strict"]
            if profile == "goal_d455_autolaunch_v1"
            else ["strict"] * 18
        )
        assert [stage.gate_mode for stage in stages] == expected_gate_modes
        assert all(stage.advance_gate_mode == "collapse" for stage in stages)

        first_cfg = stages[0].cfg
        reward_fields = tuple(
            field.name
            for field in fields(MjxJuggleConfig)
            if "reward" in field.name or "penalty" in field.name
        ) + (
            "target_height",
            "hit_height_center",
            "hit_reward_cap_mode",
            "hit_reward_count_cap",
            "hit_combo_count_cap",
            "hit_min_count_interval",
            "termination_miss_penalty_requires_hit",
            "terminate_on_ball_view_bounds",
        )
        reward_signature = tuple(getattr(first_cfg, name) for name in reward_fields)
        reward_stable_stages = (
            stages[:15]
            if profile == "goal_d455_autolaunch_v1"
            else stages
        )
        assert all(
            tuple(getattr(stage.cfg, name) for name in reward_fields) == reward_signature
            for stage in reward_stable_stages
        )
        if profile == "goal_d455_autolaunch_v1":
            tail_reward_signature = tuple(
                getattr(stages[15].cfg, name) for name in reward_fields
            )
            assert {
                name
                for name, before, after in zip(
                    reward_fields,
                    reward_signature,
                    tail_reward_signature,
                )
                if before != after
            } == {"hit_next_contact_anchor_penalty_weight"}
            assert all(
                tuple(getattr(stage.cfg, name) for name in reward_fields)
                == tail_reward_signature
                for stage in stages[15:]
            )
        active_weight_fields = {
            field.name
            for field in fields(MjxJuggleConfig)
            if ("reward" in field.name or "penalty" in field.name)
            and field.name.endswith("_weight")
            and float(getattr(first_cfg, field.name)) != 0.0
        }
        expected_active_weight_fields = {
            "non_racket_ball_contact_penalty_weight",
            "stick_contact_penalty_weight",
        }
        if profile == "goal_d455_release_v1":
            expected_active_weight_fields.add(
                "hit_next_contact_anchor_penalty_weight"
            )
        assert active_weight_fields == expected_active_weight_fields
        assert first_cfg.hit_reward_base == pytest.approx(1.0)
        assert first_cfg.hit_reward_combo == pytest.approx(0.0)
        assert first_cfg.hit_reward_count_cap == 15
        assert first_cfg.termination_miss_penalty_per_hit == pytest.approx(0.0)
        assert first_cfg.racket_z_limit_termination_penalty_per_hit == pytest.approx(0.0)
        assert not first_cfg.terminate_on_ball_view_bounds

        for stage in stages:
            cfg = stage.cfg
            assert cfg.camera_image_width == D455_848_UNDISTORTED_WIDTH == 848
            assert cfg.camera_image_height == D455_848_UNDISTORTED_HEIGHT == 480
            assert cfg.virtual_camera_base_body_name == D455_848_UNDISTORTED_SIM_BASE_BODY
            assert cfg.virtual_camera_require_base_body
            assert cfg.camera_hfov_deg == pytest.approx(D455_848_UNDISTORTED_HFOV_DEG)
            assert cfg.camera_vfov_deg == pytest.approx(D455_848_UNDISTORTED_VFOV_DEG)
            assert cfg.ball_obs_noise_warmup_ratio == pytest.approx(0.0)
            assert cfg.ball_obs_noise_ramp_ratio == pytest.approx(0.0)
            assert cfg.enable_delay_conditioning
            assert cfg.include_tau_act_norm
            assert cfg.include_command_state
            assert cfg.include_active_command_error
            assert cfg.include_phase_features
            assert cfg.actuator_cmd_filter
            assert cfg.actuator_compensation_mode == "inverse_mpc"
            assert cfg.actuator_mpc_horizon_steps == 6
            assert cfg.asymmetric_critic
            assert cfg.critic_command_history_steps == 12
            assert cfg.hit_next_contact_anchor_sigma_m == pytest.approx(0.10)

    launch = branches[PROFILES[0]]
    assert all(stage.cfg.ball_reset_mode == "racket_launch" for stage in launch)
    assert all(
        stage.cfg.hit_next_contact_anchor_penalty_weight == pytest.approx(0.0)
        for stage in launch[:15]
    )
    assert all(
        stage.cfg.hit_next_contact_anchor_penalty_weight
        == pytest.approx(
            GOAL_D455_AUTOLAUNCH_TAIL_NEXT_CONTACT_PENALTY_WEIGHT
        )
        for stage in launch[15:]
    )
    assert all(
        stage.cfg.racket_launch_surface_gap_range_m == (0.005, 0.010)
        for stage in launch
    )

    release = branches[PROFILES[1]]
    assert all(
        stage.cfg.hit_next_contact_anchor_penalty_weight
        == pytest.approx(GOAL_D455_RELEASE_NEXT_CONTACT_PENALTY_WEIGHT)
        for stage in release
    )
    release_contracts = {
        (
            stage.cfg.ball_reset_mode,
            stage.cfg.ball_launch_height,
            stage.cfg.ball_spawn_xy_jitter,
            stage.cfg.ball_spawn_z_jitter,
            stage.cfg.ball_init_vxy_max,
            stage.cfg.ball_init_vz,
            stage.cfg.ball_init_vz_jitter,
        )
        for stage in release
    }
    assert release_contracts == {("anchor_drop", 0.32, 0.025, 0.035, 0.012, -0.28, 0.0)}


def test_goal_profile_stage_diffs_are_one_declared_axis() -> None:
    expected_release = [
        {"episode_target_x_range_m", "episode_target_y_range_m", "episode_racket_anchor_z_range_m"},
        {"episode_target_x_range_m", "episode_target_y_range_m", "episode_racket_anchor_z_range_m"},
        {"domain_randomization", "dr_randomize_ball", "dr_ball_mass_range", "dr_gravity_z_range"},
        {"dr_randomize_contact", "dr_ball_friction_range", "dr_racket_friction_range", "dr_ball_solref_time_range", "dr_ball_solref_damping_range"},
        {"dr_randomize_actuator", "dr_action_scale_mult_range", "dr_armature_mult_range", "dr_damping_mult_range", "dr_randomize_pd", "dr_pd_kp_mult_range", "dr_pd_kv_mult_range", "dr_randomize_actuator_cmd_filter", "dr_actuator_cmd_tau_range", "dr_actuator_cmd_gain_range"},
        {"dr_randomize_racket_mount", "dr_racket_pos_offset_m", "dr_racket_rot_offset_rad", "dr_racket_radius_offset_m"},
        {"ball_obs_pos_noise_std", "ball_obs_vel_noise_std", "dr_randomize_ball_obs_frame", "dr_ball_obs_pos_bias_base_m", "dr_ball_obs_rot_bias_deg", "dr_ball_obs_vel_bias_base_m_s", "dr_ball_obs_scale_range"},
        {"ball_obs_dropout_prob"},
        {"ball_obs_dropout_prob", "ball_obs_require_camera_visible", "ball_obs_camera_missing_prob", "ball_obs_require_view_bounds", "ball_obs_view_bounds_missing_prob"},
        {"episode_target_x_range_m", "episode_target_y_range_m", "episode_racket_anchor_z_range_m"},
        {"dr_ball_mass_range", "dr_gravity_z_range"},
        {"dr_ball_friction_range", "dr_racket_friction_range", "dr_ball_solref_time_range", "dr_ball_solref_damping_range"},
        {"dr_action_scale_mult_range", "dr_armature_mult_range", "dr_damping_mult_range", "dr_pd_kp_mult_range", "dr_pd_kv_mult_range", "dr_actuator_cmd_tau_range", "dr_actuator_cmd_gain_range"},
        {"dr_racket_pos_offset_m", "dr_racket_rot_offset_rad", "dr_racket_radius_offset_m"},
        {"ball_obs_pos_noise_std", "ball_obs_vel_noise_std", "dr_ball_obs_pos_bias_base_m", "dr_ball_obs_rot_bias_deg", "dr_ball_obs_vel_bias_base_m_s", "dr_ball_obs_scale_range"},
        {"ball_obs_dropout_prob", "ball_obs_dropout_max_steps", "ball_obs_camera_missing_prob", "ball_obs_view_bounds_missing_prob"},
        set(),
    ]
    for profile in PROFILES:
        stages = _stages(profile)
        actual = []
        for previous, current in zip(stages, stages[1:]):
            before = asdict(previous.cfg)
            after = asdict(current.cfg)
            actual.append({name for name, value in after.items() if value != before[name]})
        expected = list(expected_release)
        if profile.endswith("autolaunch_v1"):
            expected.insert(15, set(expected_release[14]))
            expected.insert(16, set(expected_release[14]))
            expected[14] = {
                *expected[14],
                "hit_next_contact_anchor_penalty_weight",
            }
        assert actual == expected


def test_autolaunch_observation_bridges_are_exact_quarters_and_release_is_unchanged() -> None:
    launch = _stages("goal_d455_autolaunch_v1")
    release = _stages("goal_d455_release_v1")

    assert [stage.name for stage in launch[14:]] == [
        "launch14_racket_geometry_wide",
        "launch15_observation_calibration_micro_bridge",
        "launch16_observation_calibration_bridge",
        "launch17_observation_calibration_wide",
        "launch18_camera_missing_wide",
        "launch19_final_consolidation",
    ]
    assert [stage.name for stage in release[14:]] == [
        "release14_racket_geometry_wide",
        "release15_observation_calibration_wide",
        "release16_camera_missing_wide",
        "release17_final_consolidation",
    ]

    mild = launch[14]
    micro_bridge = launch[15]
    bridge = launch[16]
    wide = launch[17]
    midpoint_fields = (
        "ball_obs_pos_noise_std",
        "ball_obs_vel_noise_std",
        "dr_ball_obs_pos_bias_base_m",
        "dr_ball_obs_rot_bias_deg",
        "dr_ball_obs_vel_bias_base_m_s",
        "dr_ball_obs_scale_range",
    )
    for name in midpoint_fields:
        low = np.asarray(getattr(mild.cfg, name), dtype=np.float64)
        micro = np.asarray(getattr(micro_bridge.cfg, name), dtype=np.float64)
        mid = np.asarray(getattr(bridge.cfg, name), dtype=np.float64)
        high = np.asarray(getattr(wide.cfg, name), dtype=np.float64)
        np.testing.assert_allclose(
            micro, 0.75 * low + 0.25 * high, rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(mid, 0.5 * (low + high), rtol=0.0, atol=1e-12)

    midpoint_gate_fields = (
        "target_mean_hits",
        "target_mean_len_frac",
        "target_camera_visible",
        "target_ball_view_in_bounds",
        "target_ball_view_z_ideal",
        "target_hit1_rate",
        "target_hit3_rate",
        "target_hit12_rate",
        "target_mean_hits_ge3",
        "target_min_hit_interval_s",
        "target_max_hit_interval_s",
        "target_hit_camera_visible_rate",
        "target_hit_camera_lower_band_rate",
        "max_recent_mean_hit_vxy",
        "max_recent_hit_next_contact_anchor_err",
        "max_recent_mean_hit_camera_v_frac",
        "target_episode_truncation_rate",
        "target_racket_up_cos",
        "min_ball_obs_missing_refresh_rate",
        "max_ball_obs_lost_rate",
    )
    for name in midpoint_gate_fields:
        assert getattr(micro_bridge, name) == pytest.approx(
            0.5 * (getattr(mild, name) + getattr(bridge, name))
        )
    assert bridge.target_mean_hits == pytest.approx(
        0.5 * (mild.target_mean_hits + wide.target_mean_hits)
    )
    assert bridge.target_mean_len_frac == pytest.approx(
        0.5 * (mild.target_mean_len_frac + wide.target_mean_len_frac)
    )
    assert micro_bridge.min_updates == 225
    assert micro_bridge.max_updates is None
    assert bridge.min_updates == 230
    assert bridge.max_updates is None


def test_autolaunch_viewdense_profile_only_adds_declared_view_shaping() -> None:
    from train_juggle_mjx_curriculum import (
        GOAL_D455_AUTOLAUNCH_PROFILE,
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_BOUNDS_WEIGHT,
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_OOB_WEIGHT,
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_PROFILE,
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_XY_WEIGHT,
    )

    base = _stages(GOAL_D455_AUTOLAUNCH_PROFILE)
    shaped = _stages(GOAL_D455_AUTOLAUNCH_VIEWDENSE_PROFILE)

    assert [stage.name for stage in shaped] == [stage.name for stage in base]
    assert [stage.gate_mode for stage in shaped] == [stage.gate_mode for stage in base]
    assert [stage.advance_gate_mode for stage in shaped] == [
        stage.advance_gate_mode for stage in base
    ]
    assert len(shaped) == len(base) == 20

    allowed_cfg_diffs = {
        "ball_view_xy_center_penalty_weight",
        "ball_view_bounds_penalty_weight",
        "ball_view_out_of_bounds_penalty_weight",
    }
    for index, (before_stage, after_stage) in enumerate(zip(base, shaped)):
        before = asdict(before_stage.cfg)
        after = asdict(after_stage.cfg)
        assert {name for name, value in after.items() if value != before[name]} == allowed_cfg_diffs
        assert after_stage.cfg.ball_view_xy_center_penalty_weight == pytest.approx(
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_XY_WEIGHT
        )
        assert after_stage.cfg.ball_view_bounds_penalty_weight == pytest.approx(
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_BOUNDS_WEIGHT
        )
        assert after_stage.cfg.ball_view_out_of_bounds_penalty_weight == pytest.approx(
            GOAL_D455_AUTOLAUNCH_VIEWDENSE_OOB_WEIGHT
        )
        assert after_stage.target_mean_hits == before_stage.target_mean_hits
        assert after_stage.target_mean_len_frac == before_stage.target_mean_len_frac
        assert after_stage.target_ball_view_in_bounds == before_stage.target_ball_view_in_bounds
        assert after_stage.target_ball_view_z_ideal == before_stage.target_ball_view_z_ideal
        if index == 0:
            assert before_stage.target_episode_truncation_rate == pytest.approx(0.02)
            assert after_stage.target_episode_truncation_rate == pytest.approx(0.0)
        else:
            assert (
                after_stage.target_episode_truncation_rate
                == before_stage.target_episode_truncation_rate
            )


def test_autolaunch_viewdense_relaxtrunc_only_relaxes_early_truncation() -> None:
    from train_juggle_mjx_curriculum import (
        GOAL_D455_AUTOLAUNCH_RELAXED_EARLY_TRUNCATION,
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_PROFILE,
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_RELAXTRUNC_PROFILE,
    )

    base = _stages(GOAL_D455_AUTOLAUNCH_VIEWDENSE_PROFILE)
    relaxed = _stages(GOAL_D455_AUTOLAUNCH_VIEWDENSE_RELAXTRUNC_PROFILE)

    assert [stage.name for stage in relaxed] == [stage.name for stage in base]
    assert [stage.gate_mode for stage in relaxed] == [stage.gate_mode for stage in base]
    assert [stage.advance_gate_mode for stage in relaxed] == [
        stage.advance_gate_mode for stage in base
    ]
    for index, (before_stage, after_stage) in enumerate(zip(base, relaxed)):
        assert asdict(after_stage.cfg) == asdict(before_stage.cfg)
        assert after_stage.target_mean_hits == before_stage.target_mean_hits
        assert after_stage.target_mean_len_frac == before_stage.target_mean_len_frac
        assert after_stage.target_ball_view_in_bounds == before_stage.target_ball_view_in_bounds
        assert after_stage.target_ball_view_z_ideal == before_stage.target_ball_view_z_ideal
        if index in GOAL_D455_AUTOLAUNCH_RELAXED_EARLY_TRUNCATION:
            assert after_stage.target_episode_truncation_rate == pytest.approx(
                GOAL_D455_AUTOLAUNCH_RELAXED_EARLY_TRUNCATION[index]
            )
            assert after_stage.target_episode_truncation_rate <= before_stage.target_episode_truncation_rate
        else:
            assert (
                after_stage.target_episode_truncation_rate
                == before_stage.target_episode_truncation_rate
            )

    assert relaxed[0].target_episode_truncation_rate == pytest.approx(0.0)
    assert relaxed[1].target_episode_truncation_rate == pytest.approx(0.0)
    assert relaxed[7].target_episode_truncation_rate == pytest.approx(0.30)
    assert relaxed[8].target_episode_truncation_rate == base[8].target_episode_truncation_rate
    assert relaxed[-1].target_episode_truncation_rate == base[-1].target_episode_truncation_rate


def test_autolaunch_viewdense_fullsafe_restores_full_and_only_adds_safety_costs() -> None:
    from train_juggle_mjx_curriculum import (
        GOAL_D455_AUTOLAUNCH_FULLSAFE_ACTION_CLIP_WEIGHT,
        GOAL_D455_AUTOLAUNCH_FULLSAFE_ACTION_JERK_WEIGHT,
        GOAL_D455_AUTOLAUNCH_FULLSAFE_LIMITER_WEIGHT,
        GOAL_D455_AUTOLAUNCH_PROFILE,
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_FULLSAFE_PROFILE,
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_PROFILE,
    )

    original = _stages(GOAL_D455_AUTOLAUNCH_PROFILE)
    viewdense = _stages(GOAL_D455_AUTOLAUNCH_VIEWDENSE_PROFILE)
    fullsafe = _stages(GOAL_D455_AUTOLAUNCH_VIEWDENSE_FULLSAFE_PROFILE)

    assert [stage.name for stage in fullsafe] == [stage.name for stage in original]
    allowed_cfg_differences = {
        "action_clip_excess_penalty_weight",
        "arm_limiter_penalty_weight",
        "ball_view_bounds_penalty_weight",
        "ball_view_out_of_bounds_penalty_weight",
        "ball_view_xy_center_penalty_weight",
        "delay_action_jerk_penalty_weight",
    }
    for original_stage, viewdense_stage, fullsafe_stage in zip(
        original,
        viewdense,
        fullsafe,
    ):
        original_cfg = asdict(original_stage.cfg)
        fullsafe_cfg = asdict(fullsafe_stage.cfg)
        assert {
            name
            for name, value in fullsafe_cfg.items()
            if value != original_cfg[name]
        } == allowed_cfg_differences
        assert fullsafe_stage.target_episode_truncation_rate == pytest.approx(
            original_stage.target_episode_truncation_rate
        )
        assert fullsafe_stage.target_episode_truncation_rate >= (
            viewdense_stage.target_episode_truncation_rate
        )
        assert fullsafe_stage.cfg.action_clip_excess_penalty_weight == pytest.approx(
            GOAL_D455_AUTOLAUNCH_FULLSAFE_ACTION_CLIP_WEIGHT
        )
        assert fullsafe_stage.cfg.delay_action_jerk_penalty_weight == pytest.approx(
            GOAL_D455_AUTOLAUNCH_FULLSAFE_ACTION_JERK_WEIGHT
        )
        assert fullsafe_stage.cfg.arm_limiter_penalty_weight == pytest.approx(
            GOAL_D455_AUTOLAUNCH_FULLSAFE_LIMITER_WEIGHT
        )
        assert fullsafe_stage.cfg.right_arm_pd_profile == original_stage.cfg.right_arm_pd_profile
        assert fullsafe_stage.cfg.actuator_cmd_tau == original_stage.cfg.actuator_cmd_tau
        assert fullsafe_stage.cfg.actuator_cmd_gain == original_stage.cfg.actuator_cmd_gain
        assert (
            fullsafe_stage.cfg.actuator_compensation_mode
            == original_stage.cfg.actuator_compensation_mode
        )

    assert fullsafe[0].target_episode_truncation_rate == pytest.approx(0.02)
    assert fullsafe[1].target_episode_truncation_rate == pytest.approx(0.05)
    assert fullsafe[4].target_episode_truncation_rate == pytest.approx(0.23)
    assert fullsafe[-1].target_episode_truncation_rate == pytest.approx(0.86)


def test_autolaunch_drive_governor_preserves_course_and_original_actuator_stack() -> None:
    from train_juggle_mjx_curriculum import (
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_PROFILE,
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_FULLSAFE_PROFILE,
    )

    fullsafe = _stages(GOAL_D455_AUTOLAUNCH_VIEWDENSE_FULLSAFE_PROFILE)
    governed = _stages(GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_PROFILE)
    assert [stage.name for stage in governed] == [stage.name for stage in fullsafe]
    assert [stage.target_episode_truncation_rate for stage in governed] == [
        stage.target_episode_truncation_rate for stage in fullsafe
    ]

    for old_stage, stage in zip(fullsafe, governed):
        old_cfg = asdict(old_stage.cfg)
        cfg = asdict(stage.cfg)
        assert {
            name for name, value in cfg.items() if value != old_cfg[name]
        } == {
            "actuator_mpc_feedback_source",
            "arm_actual_state_limiter",
            "arm_actual_target_tracking_governor",
            "arm_limiter_penalty_weight",
        }
        assert stage.cfg.actuator_compensation_mode == "inverse_mpc"
        assert stage.cfg.actuator_mpc_beta == pytest.approx(1.2)
        assert stage.cfg.actuator_mpc_delay_scale == pytest.approx(1.05)
        assert stage.cfg.actuator_mpc_tau_scale == pytest.approx(0.75)
        assert stage.cfg.actuator_mpc_horizon_steps == 6
        assert stage.cfg.actuator_mpc_tracking_weight == pytest.approx(1.0)
        assert stage.cfg.actuator_mpc_nominal_weight == pytest.approx(0.25)
        assert stage.cfg.actuator_mpc_delta_weight == pytest.approx(0.05)
        assert stage.cfg.actuator_mpc_max_delta_rad == pytest.approx(np.deg2rad(30.0))
        assert stage.cfg.actuator_mpc_feedback_source == "actual"
        assert stage.cfg.right_arm_pd_profile == "xml"
        assert stage.cfg.arm_actual_state_limiter
        assert stage.cfg.arm_actual_target_tracking_governor
        assert stage.cfg.arm_actual_governor_natural_frequency_hz == pytest.approx(8.0)
        assert stage.cfg.arm_actual_governor_damping_ratio == pytest.approx(1.0)
        assert stage.cfg.arm_actual_jerk_limit_deg_s3 == (175000.0,) * 7
        assert not stage.cfg.arm_post_compensation_limiter
        assert not stage.cfg.arm_servo_target_limiter
        assert not stage.cfg.arm_servo_target_tracking_planner
        assert stage.cfg.arm_limiter_penalty_weight == pytest.approx(0.0)


def test_autolaunch_drive_governor_terminalsafe_changes_only_escape_penalty() -> None:
    from train_juggle_mjx_curriculum import (
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_PROFILE,
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_TERMINALSAFE_PROFILE,
    )

    governed = _stages(GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_PROFILE)
    terminalsafe = _stages(
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_TERMINALSAFE_PROFILE
    )
    assert [stage.name for stage in terminalsafe] == [stage.name for stage in governed]
    for old_stage, stage in zip(governed, terminalsafe):
        old_cfg = asdict(old_stage.cfg)
        cfg = asdict(stage.cfg)
        assert {
            name for name, value in cfg.items() if value != old_cfg[name]
        } == {"racket_anchor_termination_penalty_base"}
        assert stage.cfg.racket_anchor_termination_penalty_base == pytest.approx(2.5)
        assert stage.cfg.racket_anchor_termination_penalty_per_hit == pytest.approx(0.0)


def test_autolaunch_drive_governor_successref_changes_only_learning_signals() -> None:
    from train_juggle_mjx_curriculum import (
        GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_DELTA_WEIGHT,
        GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_JERK_WEIGHT,
        GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_WEIGHT,
        GOAL_D455_AUTOLAUNCH_SUCCESSREF_COMMAND_TRACKING_WEIGHT,
        GOAL_D455_AUTOLAUNCH_SUCCESSREF_MISS_PENALTY_PER_HIT,
        GOAL_D455_AUTOLAUNCH_SUCCESSREF_POST_HIT_SURVIVAL_WEIGHT,
        GOAL_D455_AUTOLAUNCH_SUCCESSREF_RACKET_Z_PENALTY_PER_HIT,
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_SUCCESSREF_PROFILE,
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_TERMINALSAFE_PROFILE,
        D455_USER_REQUESTED_RACKET_RESET_DEGREES,
    )

    terminalsafe = _stages(
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_TERMINALSAFE_PROFILE
    )
    successref = _stages(
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_SUCCESSREF_PROFILE
    )
    assert [stage.name for stage in successref] == [
        stage.name for stage in terminalsafe
    ]
    allowed = {
        "action_penalty_weight",
        "action_delta_penalty_weight",
        "command_tracking_error_penalty_weight",
        "delay_action_jerk_penalty_weight",
        "post_hit_survival_reward_weight",
        "termination_miss_penalty_per_hit",
        "racket_z_limit_termination_penalty_per_hit",
    }
    for old_stage, stage in zip(terminalsafe, successref):
        old_cfg = asdict(old_stage.cfg)
        cfg = asdict(stage.cfg)
        assert {
            name for name, value in cfg.items() if value != old_cfg[name]
        } == allowed
        assert stage.cfg.action_penalty_weight == pytest.approx(
            GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_WEIGHT
        )
        assert stage.cfg.action_delta_penalty_weight == pytest.approx(
            GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_DELTA_WEIGHT
        )
        assert stage.cfg.command_tracking_error_penalty_weight == pytest.approx(
            GOAL_D455_AUTOLAUNCH_SUCCESSREF_COMMAND_TRACKING_WEIGHT
        )
        assert stage.cfg.delay_action_jerk_penalty_weight == pytest.approx(
            GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_JERK_WEIGHT
        )
        assert stage.cfg.post_hit_survival_reward_weight == pytest.approx(
            GOAL_D455_AUTOLAUNCH_SUCCESSREF_POST_HIT_SURVIVAL_WEIGHT
        )
        assert stage.cfg.termination_miss_penalty_per_hit == pytest.approx(
            GOAL_D455_AUTOLAUNCH_SUCCESSREF_MISS_PENALTY_PER_HIT
        )
        assert stage.cfg.racket_z_limit_termination_penalty_per_hit == pytest.approx(
            GOAL_D455_AUTOLAUNCH_SUCCESSREF_RACKET_Z_PENALTY_PER_HIT
        )
        assert stage.cfg.racket_anchor_termination_penalty_base == pytest.approx(2.5)
        assert stage.cfg.actuator_compensation_mode == "inverse_mpc"
        assert stage.cfg.arm_actual_target_tracking_governor
        assert stage.cfg.arm_actual_governor_natural_frequency_hz == pytest.approx(8.0)
        assert stage.cfg.ball_reset_mode == "racket_launch"
        assert stage.cfg.right_arm_reset_degrees == tuple(
            D455_USER_REQUESTED_RACKET_RESET_DEGREES
        )
        assert stage.cfg.racket_launch_surface_gap_range_m == (0.005, 0.010)
        assert stage.cfg.camera_image_width == 848
        assert stage.cfg.camera_image_height == 480
        assert stage.cfg.virtual_camera_base_body_name == "waist03"
        assert stage.cfg.ball_obs_rate_hz == pytest.approx(60.0)
        assert stage.cfg.ball_obs_fractional_rate
        assert stage.cfg.ball_obs_frame_pivot_mode == "camera_center"

    assert successref[0].cfg.episode_target_x_range_m == (0.0, 0.0)
    assert successref[-1].cfg.episode_target_x_range_m == (-0.09, 0.09)
    assert successref[-1].cfg.episode_target_y_range_m == (-0.07, 0.07)
    assert successref[-1].cfg.episode_racket_anchor_z_range_m == (-0.035, 0.035)


def test_autolaunch_highapex_profile_aligns_physical_goal_without_plant_changes() -> None:
    from train_juggle_mjx_curriculum import (
        GOAL_D455_AUTOLAUNCH_HIGHAPEX_CADENCE_TARGET_S,
        GOAL_D455_AUTOLAUNCH_HIGHAPEX_HIT_HEIGHT,
        GOAL_D455_AUTOLAUNCH_HIGHAPEX_TARGET_ABS_Z,
        GOAL_D455_AUTOLAUNCH_HIGHAPEX_TARGET_HEIGHT,
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_HIGHAPEX_PROFILE,
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_SUCCESSREF_PROFILE,
    )

    old = _stages(GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_SUCCESSREF_PROFILE)
    high = _stages(GOAL_D455_AUTOLAUNCH_VIEWDENSE_DRIVEGOV_HIGHAPEX_PROFILE)
    assert len(high) == 20
    assert [stage.name for stage in high] == [stage.name for stage in old]

    plant_fields = (
        "actuator_compensation_mode",
        "actuator_mpc_feedback_source",
        "actuator_mpc_beta",
        "actuator_mpc_delay_scale",
        "actuator_mpc_tau_scale",
        "actuator_mpc_horizon_steps",
        "actuator_mpc_tracking_weight",
        "actuator_mpc_nominal_weight",
        "actuator_mpc_delta_weight",
        "actuator_mpc_max_delta_rad",
        "right_arm_pd_profile",
        "arm_post_compensation_limiter",
        "arm_servo_target_limiter",
        "arm_servo_target_tracking_planner",
        "arm_actual_state_limiter",
        "arm_actual_target_tracking_governor",
        "arm_actual_governor_natural_frequency_hz",
        "arm_actual_governor_damping_ratio",
        "arm_actual_jerk_limit_deg_s3",
    )
    for old_stage, stage in zip(old, high):
        for name in plant_fields:
            assert getattr(stage.cfg, name) == getattr(old_stage.cfg, name)
        assert stage.cfg.target_height == pytest.approx(
            GOAL_D455_AUTOLAUNCH_HIGHAPEX_TARGET_HEIGHT
        )
        assert stage.cfg.hit_height_center == pytest.approx(
            GOAL_D455_AUTOLAUNCH_HIGHAPEX_HIT_HEIGHT
        )
        assert stage.cfg.hit_apex_target_abs_z == pytest.approx(
            GOAL_D455_AUTOLAUNCH_HIGHAPEX_TARGET_ABS_Z
        )
        assert stage.cfg.hit_cadence_target_interval == pytest.approx(
            GOAL_D455_AUTOLAUNCH_HIGHAPEX_CADENCE_TARGET_S
        )
        assert stage.cfg.post_hit_ball_vxy_penalty_weight == pytest.approx(0.18)
        assert stage.cfg.descending_intercept_reward_weight == pytest.approx(0.8)
        assert stage.cfg.hit_next_contact_anchor_penalty_weight == pytest.approx(0.05)

    launch02 = high[2]
    assert launch02.target_mean_hits == pytest.approx(3.2)
    assert launch02.target_min_hit_interval_s == pytest.approx(0.46)
    assert launch02.target_max_hit_interval_s == pytest.approx(0.60)
    assert launch02.target_camera_visible == pytest.approx(0.95)
    assert launch02.target_ball_view_in_bounds == pytest.approx(0.75)
    assert launch02.target_ball_view_z_ideal == pytest.approx(0.90)

    final = high[-1]
    assert final.target_mean_hits == pytest.approx(10.3)
    assert final.target_hit12_rate == pytest.approx(0.36)
    assert final.target_min_hit_interval_s == pytest.approx(0.49)
    assert final.target_max_hit_interval_s == pytest.approx(0.57)
    assert final.target_episode_truncation_rate == pytest.approx(0.86)


def test_autolaunch_idealpd_profile_reuses_original_autolaunch_course_without_actuator_stack() -> None:
    from train_juggle_mjx_curriculum import (
        GOAL_D455_AUTOLAUNCH_IDEALPD_PROFILE,
        GOAL_D455_AUTOLAUNCH_PROFILE,
        GOAL_D455_AUTOLAUNCH_VIEWDENSE_RELAXTRUNC_PROFILE,
    )

    original = _stages(GOAL_D455_AUTOLAUNCH_PROFILE)
    ideal = _stages(GOAL_D455_AUTOLAUNCH_IDEALPD_PROFILE)
    w013 = _stages(GOAL_D455_AUTOLAUNCH_VIEWDENSE_RELAXTRUNC_PROFILE)

    assert [stage.name for stage in ideal] == [stage.name for stage in original]
    assert [stage.gate_mode for stage in ideal] == [stage.gate_mode for stage in original]
    assert [stage.advance_gate_mode for stage in ideal] == [
        stage.advance_gate_mode for stage in original
    ]
    stage_contract_fields = [
        field.name
        for field in fields(type(original[0]))
        if field.name not in {"cfg", "notes"}
    ]
    for before_stage, after_stage in zip(original, ideal):
        for name in stage_contract_fields:
            assert getattr(after_stage, name) == getattr(before_stage, name)
        assert after_stage.cfg.ball_reset_mode == before_stage.cfg.ball_reset_mode
        assert after_stage.cfg.episode_target_x_range_m == before_stage.cfg.episode_target_x_range_m
        assert after_stage.cfg.episode_target_y_range_m == before_stage.cfg.episode_target_y_range_m
        assert after_stage.cfg.episode_racket_anchor_z_range_m == before_stage.cfg.episode_racket_anchor_z_range_m
        assert after_stage.cfg.ball_obs_pos_noise_std == before_stage.cfg.ball_obs_pos_noise_std
        assert after_stage.cfg.ball_obs_vel_noise_std == before_stage.cfg.ball_obs_vel_noise_std
        assert after_stage.cfg.ball_obs_camera_missing_prob == before_stage.cfg.ball_obs_camera_missing_prob
        assert after_stage.cfg.ball_obs_view_bounds_missing_prob == before_stage.cfg.ball_obs_view_bounds_missing_prob
        assert after_stage.cfg.ball_view_xy_center_penalty_weight == pytest.approx(
            before_stage.cfg.ball_view_xy_center_penalty_weight
        )
        assert after_stage.cfg.ball_view_bounds_penalty_weight == pytest.approx(
            before_stage.cfg.ball_view_bounds_penalty_weight
        )
        assert after_stage.cfg.ball_view_out_of_bounds_penalty_weight == pytest.approx(
            before_stage.cfg.ball_view_out_of_bounds_penalty_weight
        )
        assert not after_stage.cfg.enable_delay_conditioning
        assert not after_stage.cfg.actuator_cmd_filter
        assert after_stage.cfg.actuator_compensation_mode == "none"

    assert original[0].target_episode_truncation_rate == pytest.approx(0.02)
    assert ideal[0].target_episode_truncation_rate == pytest.approx(0.02)
    assert w013[0].target_episode_truncation_rate == pytest.approx(0.0)
    assert ideal[0].cfg.ball_view_xy_center_penalty_weight == pytest.approx(0.0)
    assert w013[0].cfg.ball_view_xy_center_penalty_weight > 0.0
    assert original[0].cfg.enable_delay_conditioning
    assert original[0].cfg.actuator_cmd_filter
    assert original[0].cfg.actuator_compensation_mode == "inverse_mpc"


def test_autolaunch_idealpd67_changes_only_actuator_execution_semantics() -> None:
    from train_juggle_mjx_curriculum import (
        GOAL_D455_AUTOLAUNCH_IDEALPD67_PROFILE,
        GOAL_D455_AUTOLAUNCH_PROFILE,
    )

    original = _stages(GOAL_D455_AUTOLAUNCH_PROFILE)
    ideal67 = _stages(GOAL_D455_AUTOLAUNCH_IDEALPD67_PROFILE)

    assert [stage.name for stage in ideal67] == [stage.name for stage in original]
    stage_contract_fields = [
        field.name
        for field in fields(type(original[0]))
        if field.name not in {"cfg", "notes"}
    ]
    allowed_cfg_differences = {
        "actuator_cmd_filter",
        "actuator_compensation_mode",
        "actuator_delay_observation_only",
        "actuator_mpc_beta",
        "actuator_mpc_delay_scale",
        "actuator_mpc_delta_weight",
        "actuator_mpc_horizon_steps",
        "actuator_mpc_max_delta_rad",
        "actuator_mpc_tau_scale",
    }
    for before_stage, after_stage in zip(original, ideal67):
        for name in stage_contract_fields:
            assert getattr(after_stage, name) == getattr(before_stage, name)
        actual_cfg_differences = {
            field.name
            for field in fields(type(before_stage.cfg))
            if getattr(before_stage.cfg, field.name)
            != getattr(after_stage.cfg, field.name)
        }
        assert actual_cfg_differences == allowed_cfg_differences

        cfg = after_stage.cfg
        assert cfg.enable_delay_conditioning
        assert cfg.include_tau_act_norm
        assert cfg.include_command_state
        assert cfg.include_active_command_error
        assert cfg.include_phase_features
        assert cfg.delay_min_ms == pytest.approx(72.0)
        assert cfg.delay_max_ms == pytest.approx(72.0)
        assert cfg.delay_jitter_ms == pytest.approx(0.0)
        assert cfg.delay_sampling_mode == "uniform"
        assert cfg.actuator_delay_observation_only
        assert not cfg.actuator_cmd_filter
        assert cfg.actuator_compensation_mode == "none"
        assert cfg.right_arm_pd_profile == "xml"

    assert ideal67[0].target_episode_truncation_rate == pytest.approx(0.02)
    assert ideal67[-1].target_episode_truncation_rate == pytest.approx(0.86)


def test_autolaunch_idealpd67_viewdense_keeps_plant_and_full_gates() -> None:
    from train_juggle_mjx_curriculum import (
        GOAL_D455_AUTOLAUNCH_IDEALPD67_PROFILE,
        GOAL_D455_AUTOLAUNCH_IDEALPD67_VIEWDENSE_PROFILE,
        GOAL_D455_AUTOLAUNCH_IDEALPD67_VIEWDENSE_MIN_UPDATES,
        GOAL_D455_AUTOLAUNCH_TAIL_NEXT_CONTACT_PENALTY_WEIGHT,
    )

    original = _stages(GOAL_D455_AUTOLAUNCH_IDEALPD67_PROFILE)
    shaped = _stages(GOAL_D455_AUTOLAUNCH_IDEALPD67_VIEWDENSE_PROFILE)

    assert [stage.name for stage in shaped] == [stage.name for stage in original]
    for index, (before_stage, after_stage) in enumerate(zip(original, shaped)):
        assert (
            after_stage.target_episode_truncation_rate
            == before_stage.target_episode_truncation_rate
        )
        assert after_stage.cfg.ball_view_xy_center_penalty_weight > 0.0
        assert after_stage.cfg.ball_view_bounds_penalty_weight > 0.0
        assert after_stage.cfg.ball_view_out_of_bounds_penalty_weight > 0.0
        assert after_stage.cfg.actuator_delay_observation_only
        assert not after_stage.cfg.actuator_cmd_filter
        assert after_stage.cfg.actuator_compensation_mode == "none"
        assert after_stage.cfg.right_arm_pd_profile == "xml"
        assert not after_stage.cfg.arm_post_compensation_limiter
        assert not after_stage.cfg.arm_servo_target_limiter
        assert not after_stage.cfg.arm_servo_target_tracking_planner
        assert not after_stage.cfg.arm_actual_state_limiter
        assert after_stage.min_updates == (
            GOAL_D455_AUTOLAUNCH_IDEALPD67_VIEWDENSE_MIN_UPDATES.get(
                index,
                before_stage.min_updates,
            )
        )

    assert shaped[14].cfg.hit_next_contact_anchor_penalty_weight == pytest.approx(
        GOAL_D455_AUTOLAUNCH_TAIL_NEXT_CONTACT_PENALTY_WEIGHT
    )
    assert shaped[13].cfg.hit_next_contact_anchor_penalty_weight == pytest.approx(0.0)
    assert shaped[15].cfg.hit_next_contact_anchor_penalty_weight == pytest.approx(
        GOAL_D455_AUTOLAUNCH_TAIL_NEXT_CONTACT_PENALTY_WEIGHT
    )


def test_autolaunch_idealpd67_final_recovery_changes_only_final_survival() -> None:
    from train_juggle_mjx_curriculum import (
        GOAL_D455_AUTOLAUNCH_IDEALPD67_FINAL_MIN_UPDATES,
        GOAL_D455_AUTOLAUNCH_IDEALPD67_FINAL_RECOVERY_PROFILE,
        GOAL_D455_AUTOLAUNCH_IDEALPD67_FINAL_SURVIVAL_WEIGHT,
        GOAL_D455_AUTOLAUNCH_IDEALPD67_VIEWDENSE_PROFILE,
    )

    baseline = _stages(GOAL_D455_AUTOLAUNCH_IDEALPD67_VIEWDENSE_PROFILE)
    recovery = _stages(GOAL_D455_AUTOLAUNCH_IDEALPD67_FINAL_RECOVERY_PROFILE)

    assert [stage.name for stage in recovery] == [stage.name for stage in baseline]
    assert recovery[:-1] == baseline[:-1]
    before, after = baseline[-1], recovery[-1]
    stage_differences = {
        field.name
        for field in fields(type(before))
        if field.name not in {"cfg", "notes"}
        and getattr(before, field.name) != getattr(after, field.name)
    }
    cfg_differences = {
        field.name
        for field in fields(type(before.cfg))
        if getattr(before.cfg, field.name) != getattr(after.cfg, field.name)
    }
    assert stage_differences == {"min_updates"}
    assert cfg_differences == {"post_hit_survival_reward_weight"}
    assert after.min_updates == GOAL_D455_AUTOLAUNCH_IDEALPD67_FINAL_MIN_UPDATES
    assert after.cfg.post_hit_survival_reward_weight == pytest.approx(
        GOAL_D455_AUTOLAUNCH_IDEALPD67_FINAL_SURVIVAL_WEIGHT
    )
    assert after.target_mean_hits == before.target_mean_hits
    assert after.target_mean_len_frac == before.target_mean_len_frac
    assert after.target_episode_truncation_rate == before.target_episode_truncation_rate
    assert after.cfg.actuator_delay_observation_only
    assert not after.cfg.actuator_cmd_filter
    assert after.cfg.actuator_compensation_mode == "none"
    assert after.cfg.right_arm_pd_profile == "xml"
    assert not after.cfg.arm_post_compensation_limiter
    assert not after.cfg.arm_servo_target_limiter
    assert not after.cfg.arm_servo_target_tracking_planner
    assert not after.cfg.arm_actual_state_limiter


def test_idealpd67_actuator_inversempc_finetune_restores_original_67d_stack_only() -> None:
    from mjx_juggle_env import MjxJuggleConfig
    from train_juggle_mjx_curriculum import (
        GOAL_D455_AUTOLAUNCH_IDEALPD67_ACTUATOR_INVERSEMPC_FINETUNE_PROFILE,
        GOAL_D455_AUTOLAUNCH_IDEALPD67_FINAL_RECOVERY_PROFILE,
        build_curriculum,
    )

    ideal = _stages(GOAL_D455_AUTOLAUNCH_IDEALPD67_FINAL_RECOVERY_PROFILE)
    adapted = build_curriculum(
        curriculum_profile=(
            GOAL_D455_AUTOLAUNCH_IDEALPD67_ACTUATOR_INVERSEMPC_FINETUNE_PROFILE
        ),
        delay_ablation_preset="real_actuator_replay_fit",
        actuator_compensation_mode="inverse_mpc",
        actuator_cmd_filter=True,
        actuator_mpc_feedback_source="actual",
        asymmetric_critic=True,
        critic_command_history_steps=12,
        arm_post_compensation_limiter=False,
        arm_servo_target_limiter=False,
        arm_servo_target_tracking_planner=False,
        arm_actual_state_limiter=False,
        right_arm_pd_profile="xml",
    )

    assert [stage.name for stage in adapted] == [stage.name for stage in ideal]
    stage_contract_fields = [
        field.name
        for field in fields(type(ideal[0]))
        if field.name not in {"cfg", "notes"}
    ]
    reward_fields = tuple(
        field.name
        for field in fields(MjxJuggleConfig)
        if "reward" in field.name or "penalty" in field.name
    )
    for before_stage, after_stage in zip(ideal, adapted):
        for name in stage_contract_fields:
            assert getattr(after_stage, name) == getattr(before_stage, name)
        for name in reward_fields:
            assert getattr(after_stage.cfg, name) == getattr(before_stage.cfg, name)

        cfg = after_stage.cfg
        assert cfg.enable_delay_conditioning
        assert cfg.include_tau_act_norm
        assert cfg.include_command_state
        assert cfg.include_active_command_error
        assert cfg.include_phase_features
        assert not cfg.actuator_delay_observation_only
        assert cfg.actuator_cmd_filter
        assert cfg.actuator_cmd_tau == pytest.approx(0.074)
        assert cfg.actuator_cmd_gain == pytest.approx(1.0)
        assert cfg.actuator_compensation_mode == "inverse_mpc"
        assert cfg.actuator_mpc_beta == pytest.approx(1.2)
        assert cfg.actuator_mpc_delay_scale == pytest.approx(1.05)
        assert cfg.actuator_mpc_tau_scale == pytest.approx(0.75)
        assert cfg.actuator_mpc_horizon_steps == 6
        assert cfg.actuator_mpc_tracking_weight == pytest.approx(1.0)
        assert cfg.actuator_mpc_nominal_weight == pytest.approx(0.25)
        assert cfg.actuator_mpc_delta_weight == pytest.approx(0.05)
        assert np.rad2deg(cfg.actuator_mpc_max_delta_rad) == pytest.approx(30.0)
        assert cfg.actuator_mpc_feedback_source == "actual"
        assert cfg.asymmetric_critic
        assert cfg.critic_command_history_steps == 12
        assert cfg.right_arm_pd_profile == "xml"
        assert not cfg.arm_post_compensation_limiter
        assert not cfg.arm_servo_target_limiter
        assert not cfg.arm_servo_target_tracking_planner
        assert not cfg.arm_actual_state_limiter

    assert adapted[-1].cfg.post_hit_survival_reward_weight == pytest.approx(
        ideal[-1].cfg.post_hit_survival_reward_weight
    )

    with pytest.raises(ValueError, match="bottom actual-state limiters"):
        build_curriculum(
            curriculum_profile=(
                GOAL_D455_AUTOLAUNCH_IDEALPD67_ACTUATOR_INVERSEMPC_FINETUNE_PROFILE
            ),
            actuator_compensation_mode="inverse_mpc",
            actuator_cmd_filter=True,
            arm_actual_state_limiter=True,
        )


def test_actuator_inversempc_successref_nogov_uses_full_d455_course_and_reward_safety() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    from train_juggle_mjx_curriculum import (
        GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_SUCCESSREF_NOGOV_PROFILE,
        GOAL_D455_AUTOLAUNCH_IDEALPD67_VIEWDENSE_PROFILE,
        GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_DELTA_WEIGHT,
        GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_JERK_WEIGHT,
        GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_WEIGHT,
        GOAL_D455_AUTOLAUNCH_SUCCESSREF_ARM_ACC_LIMIT_WEIGHT,
        GOAL_D455_AUTOLAUNCH_SUCCESSREF_ARM_LIMITER_WEIGHT,
        GOAL_D455_AUTOLAUNCH_SUCCESSREF_ARM_VEL_LIMIT_WEIGHT,
        GOAL_D455_AUTOLAUNCH_SUCCESSREF_COMMAND_TRACKING_WEIGHT,
        GOAL_D455_AUTOLAUNCH_SUCCESSREF_POST_HIT_SURVIVAL_WEIGHT,
        build_curriculum,
    )

    ideal = _stages(GOAL_D455_AUTOLAUNCH_IDEALPD67_VIEWDENSE_PROFILE)
    stages = build_curriculum(
        curriculum_profile=(
            GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_SUCCESSREF_NOGOV_PROFILE
        ),
        delay_ablation_preset="real_actuator_replay_fit",
        actuator_cmd_filter=True,
        actuator_compensation_mode="inverse_mpc",
        asymmetric_critic=True,
        critic_command_history_steps=12,
    )

    assert len(stages) == len(ideal) == 20
    assert [stage.name for stage in stages] == [stage.name for stage in ideal]
    stage_contract_fields = [
        field.name
        for field in fields(type(ideal[0]))
        if field.name not in {"cfg", "notes"}
    ]
    environment_fields = (
        "right_arm_reset_degrees",
        "ball_reset_mode",
        "racket_launch_surface_gap_range_m",
        "racket_launch_xy_jitter",
        "episode_target_x_range_m",
        "episode_target_y_range_m",
        "episode_racket_anchor_z_range_m",
        "camera_image_width",
        "camera_image_height",
        "camera_pixel_margin",
        "ball_view_x_bounds_m",
        "ball_view_y_bounds_m",
        "ball_view_z_bounds_m",
        "domain_randomization",
        "dr_randomize_ball",
        "dr_randomize_contact",
        "dr_randomize_actuator",
        "dr_randomize_pd",
        "dr_randomize_racket_mount",
        "dr_randomize_ball_obs_frame",
        "ball_obs_dropout_prob",
        "ball_obs_camera_missing_prob",
        "ball_obs_view_bounds_missing_prob",
    )
    for ideal_stage, stage in zip(ideal, stages):
        for name in stage_contract_fields:
            assert getattr(stage, name) == getattr(ideal_stage, name)
        for name in environment_fields:
            assert getattr(stage.cfg, name) == getattr(ideal_stage.cfg, name)

        cfg = stage.cfg
        assert cfg.enable_delay_conditioning
        assert not cfg.actuator_delay_observation_only
        assert cfg.actuator_cmd_filter
        assert cfg.actuator_cmd_tau == pytest.approx(0.074)
        assert cfg.actuator_compensation_mode == "inverse_mpc"
        assert cfg.actuator_mpc_feedback_source == "applied"
        assert cfg.actuator_mpc_beta == pytest.approx(1.2)
        assert cfg.actuator_mpc_delay_scale == pytest.approx(1.05)
        assert cfg.actuator_mpc_tau_scale == pytest.approx(0.75)
        assert cfg.actuator_mpc_horizon_steps == 6
        assert cfg.action_penalty_weight == pytest.approx(
            GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_WEIGHT
        )
        assert cfg.action_delta_penalty_weight == pytest.approx(
            GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_DELTA_WEIGHT
        )
        assert cfg.command_tracking_error_penalty_weight == pytest.approx(
            GOAL_D455_AUTOLAUNCH_SUCCESSREF_COMMAND_TRACKING_WEIGHT
        )
        assert cfg.delay_action_jerk_penalty_weight == pytest.approx(
            GOAL_D455_AUTOLAUNCH_SUCCESSREF_ACTION_JERK_WEIGHT
        )
        assert cfg.post_hit_survival_reward_weight == pytest.approx(
            GOAL_D455_AUTOLAUNCH_SUCCESSREF_POST_HIT_SURVIVAL_WEIGHT
        )
        assert cfg.arm_vel_limit_penalty_weight == pytest.approx(
            GOAL_D455_AUTOLAUNCH_SUCCESSREF_ARM_VEL_LIMIT_WEIGHT
        )
        assert cfg.arm_acc_limit_penalty_weight == pytest.approx(
            GOAL_D455_AUTOLAUNCH_SUCCESSREF_ARM_ACC_LIMIT_WEIGHT
        )
        assert cfg.arm_limiter_penalty_weight == pytest.approx(
            GOAL_D455_AUTOLAUNCH_SUCCESSREF_ARM_LIMITER_WEIGHT
        )
        assert not cfg.arm_post_compensation_limiter
        assert not cfg.arm_servo_target_limiter
        assert not cfg.arm_servo_target_tracking_planner
        assert not cfg.arm_actual_state_limiter
        assert not cfg.arm_actual_target_tracking_governor

    assert stages[10].cfg.episode_target_x_range_m == (-0.090, 0.090)
    assert stages[18].cfg.ball_obs_camera_missing_prob == pytest.approx(0.50)
    assert stages[18].cfg.ball_obs_view_bounds_missing_prob == pytest.approx(0.50)

    with pytest.raises(ValueError, match="governor"):
        build_curriculum(
            curriculum_profile=(
                GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_SUCCESSREF_NOGOV_PROFILE
            ),
            arm_actual_target_tracking_governor=True,
        )


def test_goal_profiles_have_exact_actor_critic_action_dimensions() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    from mjx_juggle_env import MjxJuggleEnv

    for profile in PROFILES:
        cfg = _stages(profile)[-1].cfg
        env = MjxJuggleEnv(RL_SIM_DIR / "moz1_pd.xml", n_envs=1, cfg=cfg)
        assert (env.obs_dim, env.critic_obs_dim, env.act_dim) == (67, 231, 7)
        assert env.virtual_camera_base_body_id >= 0


def test_release_workspace_keeps_anchor_local_reset_distribution() -> None:
    jax = pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    from mjx_juggle_env import MjxJuggleEnv

    cfg = _stages("goal_d455_release_v1")[10].cfg
    env = MjxJuggleEnv(RL_SIM_DIR / "moz1_pd.xml", n_envs=16, cfg=cfg)
    state, _obs = env.reset(jax.random.split(jax.random.PRNGKey(1601), 16))
    local = np.asarray(state.reset_ball_pos - state.racket_anchor)
    vel = np.asarray(state.reset_ball_vel)
    assert float(np.abs(local[:, 0]).max()) <= 0.025 + 2e-6
    assert float(np.abs(local[:, 1]).max()) <= 0.025 + 2e-6
    assert float(local[:, 2].min()) >= 0.32 - 0.035 - 2e-6
    assert float(local[:, 2].max()) <= 0.32 + 0.035 + 2e-6
    assert float(np.abs(vel[:, 0]).max()) <= 0.012 + 2e-6
    assert float(np.abs(vel[:, 1]).max()) <= 0.012 + 2e-6
    np.testing.assert_allclose(vel[:, 2], -0.28, atol=2e-6)
    assert float(np.abs(np.asarray(state.reset_target_offset)).max()) > 0.0


def test_passive_racket_support_does_not_count_as_autonomous_launch() -> None:
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("mujoco")
    from mjx_juggle_env import MjxJuggleEnv

    cfg = _stages("goal_d455_autolaunch_v1")[0].cfg
    env = MjxJuggleEnv(RL_SIM_DIR / "moz1_pd.xml", n_envs=1, cfg=cfg)
    state, obs = env.reset(jax.random.split(jax.random.PRNGKey(1600), 1))
    zero_action = jnp.zeros((1, env.act_dim), dtype=jnp.float32)

    @jax.jit
    def rollout(initial_state, initial_obs):
        def step(carry, _unused):
            current_state, current_obs = carry
            next_state, next_obs, _reward, done, metrics = env.step(
                current_state,
                zero_action,
            )
            output = (metrics["hit_count"], metrics["reward/hit_bonus"], done)
            return (next_state, next_obs), output

        return jax.lax.scan(step, (initial_state, initial_obs), None, length=120)[1]

    hit_count, hit_bonus, _done = rollout(state, obs)
    assert int(np.asarray(hit_count).max()) == 0
    np.testing.assert_allclose(np.asarray(hit_bonus), 0.0, atol=1e-7)


def test_reset_bucket_cvar_skips_constants_and_requires_each_varying_field() -> None:
    from train_juggle_mjx_curriculum import summarize_reset_bucket_outputs

    values = np.linspace(0.005, 0.010, 18, dtype=np.float64)[None, :]
    shape = values.shape
    metrics = {
        "reset_ball_surface_gap": values,
        "reset_ball_vz": np.zeros(shape, dtype=np.float64),
        "ball_view_in_bounds": np.ones(shape, dtype=np.float64),
        "ball_view_z_ideal": np.ones(shape, dtype=np.float64),
    }
    done = np.ones(shape, dtype=bool)
    hit_count = np.tile(np.arange(18, dtype=np.float64)[None, :] % 6 + 8, (1, 1))
    result = summarize_reset_bucket_outputs(
        metrics,
        hit_count,
        done,
        mode="cvar",
        min_episodes=2,
        cvar_frac=0.5,
        fields=("reset_ball_surface_gap", "reset_ball_vz"),
    )
    assert result["advance_eval/reset_bucket_field_count"] == pytest.approx(1.0)
    assert result["advance_eval/reset_bucket_eligible_field_count"] == pytest.approx(1.0)
    assert result["advance_eval/reset_bucket_bin_count"] == pytest.approx(3.0)


def _threshold_args(profile: str) -> SimpleNamespace:
    return SimpleNamespace(
        curriculum_profile=profile,
        advance_validation_mode="block",
        advance_eval_hit_ratio=0.90,
        advance_eval_len_ratio=0.90,
        advance_eval_min_hits=0.5,
        advance_eval_min_len_frac=0.05,
        advance_eval_camera_margin=0.05,
        advance_eval_ball_view_margin=0.05,
        advance_eval_z_ideal_margin=0.05,
        advance_eval_hit_rate_margin=0.05,
        advance_eval_cond_hit_ratio=0.90,
        advance_eval_camera_reward_margin=0.02,
        advance_eval_hit_interval_margin=0.03,
        advance_eval_min_return=-100.0,
    )


def _convergence_args() -> SimpleNamespace:
    return SimpleNamespace(
        advance_mode="converged",
        convergence_min_episodes=64,
        convergence_window=1,
        min_stage_updates=0,
    )


def _passing_convergence_row(stage, *, next_contact_err: float) -> dict[str, float]:
    return {
        "episodes": 64,
        "mean_hits": float(stage.target_mean_hits),
        "mean_len": 1200.0 * float(stage.target_mean_len_frac),
        "mean_return": 0.0,
        "camera_visible": float(stage.target_camera_visible),
        "reward/camera_reward_dense": 0.0,
        "ball_view_in_bounds": float(stage.target_ball_view_in_bounds),
        "ball_view_z_ideal": float(stage.target_ball_view_z_ideal),
        "hit1_rate": float(stage.target_hit1_rate),
        "hit3_rate": float(stage.target_hit3_rate),
        "hit12_rate": float(stage.target_hit12_rate),
        "mean_hits_ge3": float(stage.target_mean_hits_ge3),
        "mean_hit_interval_s": 0.40,
        "mean_hit_interval_ge3_s": 0.40,
        "episode_truncation_rate": float(stage.target_episode_truncation_rate),
        "ball_obs_missing_refresh_rate": float(
            stage.min_ball_obs_missing_refresh_rate or 0.01
        ),
        "ball_obs_lost_active": 0.0,
        "hit_camera_visible_rate": float(stage.target_hit_camera_visible_rate),
        "hit_camera_lower_band_rate": float(stage.target_hit_camera_lower_band_rate),
        "mean_hit_camera_v_frac": float(stage.max_recent_mean_hit_camera_v_frac) - 0.01,
        "mean_hit_vxy": float(stage.max_recent_mean_hit_vxy) - 0.01,
        "mean_hit_next_contact_anchor_err": float(next_contact_err),
        "racket_up_cos": float(stage.target_racket_up_cos),
    }


def test_balanced_probe_uses_finite_next_contact_as_diagnostic_only() -> None:
    from train_juggle_mjx_curriculum import convergence_status

    stages = _stages("goal_d455_autolaunch_v1")
    bridge = stages[16]
    over_limit = float(bridge.max_recent_hit_next_contact_anchor_err) + 0.02
    row = _passing_convergence_row(bridge, next_contact_err=over_limit)
    status = convergence_status(
        [row], bridge, SimpleNamespace(max_steps=1200), _convergence_args(), 230
    )
    assert status["convergence/gate_mode_balanced_probe"] == pytest.approx(1.0)
    assert status["convergence/hit_next_contact_anchor_ok"] == pytest.approx(0.0)
    assert status["convergence/hit_recoverability_ok"] == pytest.approx(0.0)
    assert status["convergence/hit_probe_readiness_ok"] == pytest.approx(1.0)
    assert status["convergence/performance_gate_ok"] == pytest.approx(1.0)
    assert status["convergence/stage_converged"] == pytest.approx(1.0)

    missing = dict(row, mean_hit_next_contact_anchor_err=float("nan"))
    missing_status = convergence_status(
        [missing], bridge, SimpleNamespace(max_steps=1200), _convergence_args(), 230
    )
    assert missing_status["convergence/hit_probe_readiness_ok"] == pytest.approx(0.0)
    assert missing_status["convergence/performance_gate_ok"] == pytest.approx(0.0)

    excessive_vxy = dict(row, mean_hit_vxy=float(bridge.max_recent_mean_hit_vxy) + 0.01)
    vxy_status = convergence_status(
        [excessive_vxy], bridge, SimpleNamespace(max_steps=1200), _convergence_args(), 230
    )
    assert vxy_status["convergence/hit_probe_readiness_ok"] == pytest.approx(0.0)
    assert vxy_status["convergence/performance_gate_ok"] == pytest.approx(0.0)

    final_stage = stages[-1]
    final_row = _passing_convergence_row(
        final_stage,
        next_contact_err=float(final_stage.max_recent_hit_next_contact_anchor_err) + 0.02,
    )
    final_status = convergence_status(
        [final_row],
        final_stage,
        SimpleNamespace(max_steps=1200),
        _convergence_args(),
        final_stage.min_updates,
    )
    assert final_status["convergence/gate_mode_balanced_probe"] == pytest.approx(0.0)
    assert final_status["convergence/hit_next_contact_anchor_ok"] == pytest.approx(0.0)
    assert final_status["convergence/performance_gate_ok"] == pytest.approx(0.0)
    assert final_status["convergence/stage_converged"] == pytest.approx(0.0)


def test_intermediate_probe_thresholds_remain_collapse_only() -> None:
    from train_juggle_mjx_curriculum import advance_validation_thresholds

    stages = _stages("goal_d455_autolaunch_v1")
    args = _threshold_args("goal_d455_autolaunch_v1")
    for stage in stages[15:19]:
        thresholds = advance_validation_thresholds(args, stage)
        assert thresholds["target_mean_hits"] == pytest.approx(
            0.35 * stage.target_mean_hits
        )
        assert thresholds["target_mean_len_frac"] == pytest.approx(
            0.30 * stage.target_mean_len_frac
        )
        assert thresholds["target_hit1_rate"] == pytest.approx(0.50)
        assert np.isnan(thresholds["max_mean_hit_next_contact_anchor_err"])


def test_final_self_probe_selects_strict_result_for_collapse_profile() -> None:
    from train_juggle_mjx_curriculum import _advance_validation_gate_passed

    assert _advance_validation_gate_passed(
        "collapse",
        final_self_probe=False,
        collapse_core_ok=True,
        strict_ok=False,
    )
    assert not _advance_validation_gate_passed(
        "collapse",
        final_self_probe=True,
        collapse_core_ok=True,
        strict_ok=False,
    )
    assert _advance_validation_gate_passed(
        "collapse",
        final_self_probe=True,
        collapse_core_ok=True,
        strict_ok=True,
    )
    assert _advance_validation_gate_passed(
        "strict",
        final_self_probe=False,
        collapse_core_ok=False,
        strict_ok=True,
    )
    with pytest.raises(ValueError, match="advance_gate_mode"):
        _advance_validation_gate_passed(
            "unknown",
            final_self_probe=False,
            collapse_core_ok=True,
            strict_ok=True,
        )


def test_final_stage_requires_strict_self_validation_with_full_noise() -> None:
    from train_juggle_mjx_curriculum import (
        _advance_validation_probe_spec,
        advance_validation_defaults,
        advance_validation_env_cfg,
        advance_validation_thresholds,
    )

    stages = _stages("goal_d455_autolaunch_v1")
    args = _threshold_args("goal_d455_autolaunch_v1")

    entry_stage, entry_index, entry_is_final = _advance_validation_probe_spec(
        args,
        len(stages) - 1,
        stages,
    )
    assert entry_stage is stages[-1]
    assert entry_index == len(stages) - 1
    assert not entry_is_final
    entry_thresholds = advance_validation_thresholds(args, entry_stage)
    assert np.isnan(entry_thresholds["target_hit12_rate"])

    final_stage, final_index, final_is_final = _advance_validation_probe_spec(
        args,
        len(stages),
        stages,
    )
    assert final_stage is stages[-1]
    assert final_index == len(stages) - 1
    assert final_is_final
    defaults = advance_validation_defaults(args, len(stages), stages)
    assert defaults["advance_eval/required"] == pytest.approx(1.0)
    assert defaults["advance_eval/final_self_probe"] == pytest.approx(1.0)
    assert defaults["advance_eval/gate_mode_collapse"] == pytest.approx(0.0)
    final_thresholds = advance_validation_thresholds(
        args,
        final_stage,
        force_strict=True,
    )
    assert np.isfinite(final_thresholds["target_hit12_rate"])
    assert np.isfinite(final_thresholds["max_mean_hit_vxy"])
    assert np.isfinite(final_thresholds["max_mean_hit_camera_v_frac"])
    assert np.isfinite(final_thresholds["max_mean_hit_next_contact_anchor_err"])
    assert np.isfinite(final_thresholds["target_racket_up_cos"])

    eval_cfg = advance_validation_env_cfg(final_stage)
    assert eval_cfg.ball_obs_noise_warmup_ratio == pytest.approx(0.0)
    assert eval_cfg.ball_obs_noise_ramp_ratio == pytest.approx(0.0)
    assert eval_cfg.total_training_steps == 1


def test_goal_profiles_have_no_stage_specific_update_caps_but_cli_cap_still_works() -> None:
    from train_juggle_mjx_curriculum import stage_update_cap

    stages = _stages("goal_d455_autolaunch_v1")
    args = SimpleNamespace(advance_mode="converged", max_stage_updates=0)
    assert [stage_update_cap(stage, args, 1024 * 256) for stage in stages] == [
        None for _stage in stages
    ]
    args.max_stage_updates = 77
    assert stage_update_cap(stages[-1], args, 1024 * 256) == 77


def test_stage_steps_override_does_not_change_profile_contract() -> None:
    from train_juggle_mjx_curriculum import build_curriculum

    stages = build_curriculum(
        stage_steps_override=4096,
        curriculum_profile="goal_d455_autolaunch_v1",
    )
    assert len(stages) == 20
    assert all(stage.total_steps == 4096 for stage in stages)
    assert all(stage.cfg.ball_reset_mode == "racket_launch" for stage in stages)


def test_countcredit_nogov_changes_only_hit_credit_objective() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    from train_juggle_mjx_curriculum import (
        GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_COUNTCREDIT_NOGOV_PROFILE,
        GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_SUCCESSREF_NOGOV_PROFILE,
        build_curriculum,
    )

    common = dict(
        delay_ablation_preset="real_actuator_replay_fit",
        actuator_cmd_filter=True,
        actuator_compensation_mode="inverse_mpc",
        asymmetric_critic=True,
        critic_command_history_steps=12,
    )
    baseline = build_curriculum(
        curriculum_profile=(
            GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_SUCCESSREF_NOGOV_PROFILE
        ),
        **common,
    )
    countcredit = build_curriculum(
        curriculum_profile=(
            GOAL_D455_AUTOLAUNCH_ACTUATOR_INVERSEMPC_COUNTCREDIT_NOGOV_PROFILE
        ),
        **common,
    )

    expected_changes = {
        "hit_reward_cap_mode",
        "hit_reward_count_cap",
        "termination_miss_penalty_base",
        "termination_miss_penalty_per_hit",
        "racket_z_limit_termination_penalty_base",
        "racket_z_limit_termination_penalty_per_hit",
        "racket_anchor_termination_penalty_base",
    }
    assert len(baseline) == len(countcredit) == 20
    for before, after in zip(baseline, countcredit):
        changed = {
            field.name
            for field in fields(type(before.cfg))
            if getattr(before.cfg, field.name) != getattr(after.cfg, field.name)
        }
        assert changed == expected_changes
        assert after.cfg.hit_reward_cap_mode == "off"
        assert after.cfg.hit_reward_count_cap == 0
        assert after.cfg.hit_reward_combo == pytest.approx(0.0)
        reference_hits = max(1.0, float(before.target_mean_hits))
        assert after.cfg.termination_miss_penalty_base == pytest.approx(
            before.cfg.termination_miss_penalty_base
            + before.cfg.termination_miss_penalty_per_hit * reference_hits
        )
        assert after.cfg.termination_miss_penalty_per_hit == pytest.approx(0.0)
        assert after.cfg.racket_z_limit_termination_penalty_base == pytest.approx(
            before.cfg.racket_z_limit_termination_penalty_base
            + before.cfg.racket_z_limit_termination_penalty_per_hit * reference_hits
        )
        assert after.cfg.racket_z_limit_termination_penalty_per_hit == pytest.approx(0.0)
        assert after.cfg.racket_anchor_termination_penalty_base == pytest.approx(
            max(
                before.cfg.racket_anchor_termination_penalty_base,
                after.cfg.termination_miss_penalty_base,
                after.cfg.racket_z_limit_termination_penalty_base,
            )
        )


def test_nomissing_hardtail_changes_only_training_density_from_launch17() -> None:
    baseline = _stages(
        "goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2_countcredit_nomissing_v1"
    )
    hardtail = _stages(
        "goal_d455_autolaunch_viewdense_constrained_mpc_drbridge_v2_countcredit_nomissing_hardtail_v1"
    )

    assert len(baseline) == len(hardtail) == 22
    for index, (before, after) in enumerate(zip(baseline, hardtail, strict=True)):
        assert before.min_ball_obs_missing_refresh_rate is None
        assert before.max_ball_obs_lost_rate is None
        assert after.min_ball_obs_missing_refresh_rate is None
        assert after.max_ball_obs_lost_rate is None
        changed = {
            field.name
            for field in fields(type(before.cfg))
            if getattr(before.cfg, field.name) != getattr(after.cfg, field.name)
        }
        if index < 17:
            assert asdict(before) == asdict(after)
        else:
            assert changed == {"dr_hard_tail_fraction"}
            assert after.cfg.dr_hard_tail_fraction == pytest.approx(0.50)
            assert after.cfg.dr_hard_tail_lower_quantile == pytest.approx(2.0 / 3.0)
            assert before.cfg.dr_ball_solref_time_range == after.cfg.dr_ball_solref_time_range
            assert before.cfg.dr_actuator_cmd_tau_range == after.cfg.dr_actuator_cmd_tau_range
            assert before.cfg.arm_servo_target_tracking_planner
            assert after.cfg.arm_servo_target_tracking_planner
            assert before.cfg.actuator_compensation_mode == after.cfg.actuator_compensation_mode == "inverse_mpc"


def test_intercept_nomissing_survival_contract() -> None:
    intercept = _stages(
        "goal_d455_autolaunch_viewdense_constrained_mpc_intercept_v1"
    )
    survival = _stages(
        "goal_d455_autolaunch_viewdense_constrained_mpc_intercept_nomissing_survival_v1"
    )

    assert len(intercept) == len(survival) == 22
    launch17 = survival[17]
    source = intercept[17]
    cfg = launch17.cfg

    assert launch17.name == "launch17_observation_calibration_bridge"
    assert launch17.min_ball_obs_missing_refresh_rate is None
    assert launch17.max_ball_obs_lost_rate is None
    assert not cfg.ball_obs_require_camera_visible
    assert not cfg.ball_obs_require_view_bounds
    assert not cfg.ball_obs_require_view_z_high
    assert cfg.ball_obs_camera_missing_prob == pytest.approx(0.0)
    assert cfg.ball_obs_view_bounds_missing_prob == pytest.approx(0.0)
    assert cfg.ball_obs_missing_episode_coherent_prob == pytest.approx(0.0)
    assert cfg.dr_hard_tail_fraction == pytest.approx(0.0)
    assert cfg.post_hit_ball_vxy_penalty_weight == pytest.approx(0.18)
    assert cfg.hit_vxy_penalty_weight == pytest.approx(0.90)
    assert cfg.hit_next_contact_anchor_penalty_weight == pytest.approx(0.06)
    assert cfg.descending_intercept_reward_weight == pytest.approx(1.20)
    assert cfg.actuator_compensation_mode == "inverse_mpc"
    assert cfg.actuator_mpc_feedback_source == "actual"
    assert cfg.arm_servo_target_tracking_planner

    for field_name in (
        "target_mean_hits",
        "target_mean_len_frac",
        "target_ball_view_in_bounds",
        "target_hit1_rate",
        "target_hit3_rate",
        "target_hit12_rate",
        "target_mean_hits_ge3",
        "target_episode_truncation_rate",
    ):
        assert getattr(launch17, field_name) == getattr(source, field_name)


def test_sport_direct_consolidates_before_nominal_task_progression() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    from train_juggle_mjx_curriculum import build_curriculum

    common = dict(
        curriculum_profile="goal_d455_sport_taskspace_obsres2mm_nocomp_direct_v1",
        gate_preset="v7_strict",
        actuator_compensation_mode="none",
        arm_servo_target_tracking_planner=False,
        asymmetric_critic=True,
        critic_command_history_steps=12,
    )
    nominal = build_curriculum(
        delay_ablation_preset="sport_actuator_replay_dr", **common
    )
    homotopy = build_curriculum(
        delay_ablation_preset="sport_actuator_replay_homotopy_dr", **common
    )

    assert len(nominal) == len(homotopy) == 21
    for stages in (nominal, homotopy):
        launch00 = stages[0]
        assert launch00.min_updates == 30
        assert launch00.target_mean_hits == pytest.approx(1.0)
        assert launch00.target_mean_len_frac == pytest.approx(0.10)
        assert launch00.target_hit3_rate is None
        assert launch00.target_mean_hits_ge3 is None
        assert launch00.target_episode_truncation_rate == pytest.approx(0.02)
        assert launch00.cfg.actuator_compensation_mode == "none"
        assert not launch00.cfg.arm_servo_target_tracking_planner
        assert not launch00.cfg.enable_anti_windup
        assert not launch00.cfg.dr_randomize_second_order_actuator

    assert nominal[0].cfg.actuator_cmd_damping_ratio[0] == pytest.approx(0.391768)
    assert nominal[0].cfg.actuator_cmd_delay_ms_per_joint == (
        45.0, 50.0, 45.0, 40.0, 35.0, 45.0, 55.0
    )
    assert homotopy[0].cfg.actuator_cmd_damping_ratio == pytest.approx((0.8,) * 7)
    assert homotopy[1].cfg.actuator_cmd_damping_ratio == pytest.approx((0.55,) * 7)
    assert homotopy[2].cfg.actuator_cmd_damping_ratio == pytest.approx(
        nominal[2].cfg.actuator_cmd_damping_ratio
    )
    assert homotopy[2].cfg.actuator_cmd_delay_ms_per_joint == pytest.approx(
        nominal[2].cfg.actuator_cmd_delay_ms_per_joint
    )


def test_taskspace_phase_teacher_aligns_impact_and_dense_local_xz_objectives() -> None:
    pytest.importorskip("jax")
    pytest.importorskip("mujoco")
    from train_juggle_mjx_curriculum import (
        apply_phase_teacher_reference,
        build_curriculum,
    )

    stages = build_curriculum(
        gate_preset="v7_strict",
        curriculum_profile="goal_d455_sport_taskspace_obsres2mm_nocomp_direct_v1",
        delay_ablation_preset="sport_actuator_replay_homotopy_dr",
        actuator_compensation_mode="none",
        arm_servo_target_tracking_planner=False,
        asymmetric_critic=True,
        critic_command_history_steps=12,
    )
    guided = apply_phase_teacher_reference(
        stages,
        RL_SIM_DIR / "references" / "gpu0_obsres2mm_servo_phase_teacher_v1.npz",
        1.0,
        "taskspace_only",
    )

    assert guided[0].cfg.racket_stability_angular_speed_mode == "local_xz"
    assert guided[0].cfg.racket_stability_angular_speed_penalty_weight >= 0.12
    for stage in guided[1:]:
        assert stage.cfg.racket_stability_angular_speed_mode == "local_xz"
        assert stage.cfg.racket_stability_angular_speed_penalty_weight >= 0.50
        assert stage.cfg.racket_stability_angular_speed_soft_limit_rad_s == pytest.approx(0.50)
        assert stage.cfg.racket_stability_angular_speed_scale_rad_s <= 0.70
        assert stage.cfg.hit_racket_angular_speed_penalty_weight >= 1.00
        assert stage.cfg.hit_racket_angular_speed_soft_limit_rad_s <= 0.70
        assert stage.cfg.hit_racket_angular_speed_scale_rad_s <= 0.70
