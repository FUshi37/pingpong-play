from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from omegaconf import OmegaConf

from pingpong_hil_rl.train_flashsac import (
    CompetenceCurriculum,
    _agent_config,
    _assert_contract,
    _episode_meets_stage_goal,
    _is_known_simulator_idle_pose,
    _resolve_run_dir,
    _sample_truncated_zeta,
    _truncated_zeta_cdf,
)


ROOT = Path(__file__).resolve().parents[3]


def test_default_contract_is_hil_only():
    cfg = OmegaConf.load(ROOT / "src/pingpong_hil_rl/config/hil_flashsac.yaml")
    _assert_contract(cfg)


def test_short_history_is_rejected():
    cfg = OmegaConf.load(ROOT / "src/pingpong_hil_rl/config/hil_flashsac.yaml")
    cfg.plant.command_history_steps = 10
    with pytest.raises(ValueError, match="100-120ms"):
        _assert_contract(cfg)


def test_compensation_is_rejected():
    cfg = OmegaConf.load(ROOT / "src/pingpong_hil_rl/config/hil_flashsac.yaml")
    cfg.plant.compensation = True
    with pytest.raises(ValueError, match="forbids"):
        _assert_contract(cfg)


def test_default_curriculum_requires_hits_and_bounds_shaping():
    cfg = OmegaConf.load(ROOT / "src/pingpong_hil_rl/config/hil_flashsac.yaml")
    assert all(stage.success_mode != "launch_progress" for stage in cfg.curriculum)
    for stage in cfg.curriculum:
        shaping_cap = (
            float(stage.pre_hit_flatness_weight)
            + float(stage.pre_hit_lift_weight)
            + float(stage.pre_hit_launch_weight)
            + float(stage.pre_hit_impulse_weight)
        )
        assert shaping_cap <= 6.0
        assert float(stage.valid_hit_bonus) > shaping_cap


def test_launch_progress_promotion_is_rejected_by_hil_contract():
    cfg = OmegaConf.load(ROOT / "src/pingpong_hil_rl/config/hil_flashsac.yaml")
    cfg.curriculum[0].success_mode = "launch_progress"
    with pytest.raises(ValueError, match="confirmed hits"):
        _assert_contract(cfg)


def test_launch_gate_requires_contact_upward_velocity_and_apex():
    stage = {"success_mode": "launch_progress"}
    assert not _episode_meets_stage_goal(
        stage,
        {"contact_detected": True, "max_ball_vz": 0.19, "max_predicted_apex_rel_z": 0.2},
    )
    assert _episode_meets_stage_goal(
        stage,
        {"contact_detected": True, "max_ball_vz": 0.21, "max_predicted_apex_rel_z": 0.051},
    )
    assert _episode_meets_stage_goal(stage, {"max_hits": 1})


def test_curriculum_promotes_on_competence_not_elapsed_steps():
    stages = [
        {
            "name": "launch", "success_mode": "launch_progress",
            "window_episodes": 4, "min_episodes": 3,
            "promote_success_rate": 2 / 3, "demote_success_rate": 0.0,
        },
        {
            "name": "hit", "success_mode": "first_hit",
            "window_episodes": 4, "min_episodes": 3,
            "promote_success_rate": 1.0, "demote_success_rate": 0.0,
        },
    ]
    curriculum = CompetenceCurriculum(stages)
    failure = {"contact_detected": False, "max_hits": 0}
    for _ in range(20):
        _, event = curriculum.observe_episode(failure)
        assert event is None
    success = {
        "contact_detected": True,
        "max_ball_vz": 0.3,
        "max_predicted_apex_rel_z": 0.1,
        "max_hits": 0,
    }
    for _ in range(3):
        _, event = curriculum.observe_episode(success)
    assert event is not None
    assert event["type"] == "promotion"
    assert curriculum.stage["name"] == "hit"


def test_evaluation_only_curriculum_ignores_assisted_hits():
    stages = [
        {
            "name": "g1", "success_mode": "first_hit",
            "window_episodes": 4, "min_episodes": 3,
            "promote_success_rate": 2 / 3, "demote_success_rate": 0.0,
            "promotion_requires_evaluation": True,
        },
        {
            "name": "g2", "success_mode": "first_hit",
            "window_episodes": 4, "min_episodes": 3,
            "promote_success_rate": 1.0, "demote_success_rate": 0.0,
            "promotion_requires_evaluation": True,
        },
    ]
    curriculum = CompetenceCurriculum(stages)
    for _ in range(20):
        success, event = curriculum.observe_episode(
            {"max_hits": 1, "evaluation_episode": False}
        )
        assert success
        assert event is None
    assert curriculum.eligible_episodes_in_stage == 0
    for hits in (1, 1, 0):
        _, event = curriculum.observe_episode(
            {"max_hits": hits, "evaluation_episode": True}
        )
    assert event is not None
    assert event["type"] == "promotion"


def test_curriculum_state_round_trip_preserves_stage_window():
    cfg = OmegaConf.load(ROOT / "src/pingpong_hil_rl/config/hil_flashsac.yaml")
    stages = [OmegaConf.to_container(stage, resolve=True) for stage in cfg.curriculum]
    source = CompetenceCurriculum(stages)
    source.index = 2
    source.episodes_in_stage = 17
    source.eligible_episodes_in_stage = 4
    source._outcomes.extend([True, False, True, True])
    restored = CompetenceCurriculum(stages)
    restored.load_state_dict(source.state_dict())
    assert restored.stage["name"] == "first_hit_g25"
    assert restored.episodes_in_stage == 17
    assert restored.eligible_episodes_in_stage == 4
    assert list(restored._outcomes) == [True, False, True, True]


def test_revised_curriculum_can_restart_at_known_stage_without_stale_failures():
    cfg = OmegaConf.load(ROOT / "src/pingpong_hil_rl/config/hil_flashsac.yaml")
    stages = [OmegaConf.to_container(stage, resolve=True) for stage in cfg.curriculum]
    curriculum = CompetenceCurriculum(stages)
    curriculum.load_state_dict(
        {
            "stage_name": "first_hit_g5",
            "episodes_in_stage": 60,
            "eligible_episodes_in_stage": 15,
            "outcomes": [False] * 15,
        }
    )
    curriculum.reset_to_stage("first_hit_g35")
    assert curriculum.stage["name"] == "first_hit_g35"
    assert curriculum.episodes_in_stage == 0
    assert curriculum.eligible_episodes_in_stage == 0
    assert list(curriculum._outcomes) == []


def test_first_hit_gravity_curriculum_has_no_reward_or_distribution_cliff():
    cfg = OmegaConf.load(ROOT / "src/pingpong_hil_rl/config/hil_flashsac.yaml")
    first_hit = [stage for stage in cfg.curriculum if stage.success_mode == "first_hit"]
    gravity_ladder = first_hit[:13]
    assert [float(stage.gravity_z) for stage in gravity_ladder] == [
        -1.0, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5,
        -5.0, -6.0, -7.0, -8.0, -9.0, -9.81,
    ]
    assert all(float(stage.target_height) == pytest.approx(0.16) for stage in gravity_ladder[4:])
    assert all(float(stage.action_acc_scale) == pytest.approx(0.46) for stage in gravity_ladder[4:])
    for stage in first_hit:
        assert float(stage.valid_hit_bonus) == 25.0
        assert (
            float(stage.pre_hit_flatness_weight)
            + float(stage.pre_hit_lift_weight)
            + float(stage.pre_hit_launch_weight)
            + float(stage.pre_hit_impulse_weight)
        ) == pytest.approx(6.0)
        assert float(stage.racket_pos_dr_m) == 0.0
        assert float(stage.racket_rot_dr_deg) == 0.0
        assert float(stage.racket_radius_dr_m) == 0.0
        assert bool(stage.ball_dr)
        assert bool(stage.contact_dr)
        assert float(stage.demote_success_rate) == 0.0
    expected_from = [-1.0, -1.0, -2.0, -2.5, -3.0, -3.5, -4.0, -4.5, -5.0, -6.0, -7.0, -8.0, -9.0]
    assert [float(stage.gravity_train_from_z) for stage in gravity_ladder] == expected_from


def test_flashsac_uses_one_update_per_measured_transition():
    cfg = OmegaConf.load(ROOT / "src/pingpong_hil_rl/config/hil_flashsac.yaml")
    assert float(cfg.updates_per_step) == 1.0


def test_structured_exploration_is_a_minor_recovery_mixture():
    cfg = OmegaConf.load(ROOT / "src/pingpong_hil_rl/config/hil_flashsac.yaml")
    assert not bool(cfg.structured_exploration.enabled)
    assert float(cfg.structured_exploration.initial_probability) == 0.0
    assert float(cfg.structured_exploration.final_probability) == 0.0
    assert int(cfg.structured_exploration.warmup_steps) == 0


def test_flashsac_batch_matches_measured_single_environment_tradeoff():
    cfg = OmegaConf.load(ROOT / "src/pingpong_hil_rl/config/hil_flashsac.yaml")
    assert int(cfg.flashsac.sample_batch_size) == 1024


def test_resume_appends_to_original_run_directory(tmp_path):
    run_dir = tmp_path / "20260803-103244"
    checkpoint = run_dir / "step000435326"
    checkpoint.mkdir(parents=True)
    assert _resolve_run_dir("runs/hil_flashsac", checkpoint) == run_dir.resolve()


def test_numpy_zeta_sampler_matches_official_truncated_distribution_boundaries():
    cdf = _truncated_zeta_cdf(mu=2.0, max_n=16)
    expected = np.cumsum(np.arange(1, 17, dtype=np.float64) ** -2.0)
    expected /= expected[-1]
    np.testing.assert_allclose(cdf, expected)
    assert _sample_truncated_zeta(cdf, 0.0) == 1
    assert _sample_truncated_zeta(cdf, float(cdf[0])) == 2
    assert _sample_truncated_zeta(cdf, np.nextafter(1.0, 0.0)) == 16


def test_flashsac_lr_schedule_is_counted_in_learner_updates():
    cfg = OmegaConf.load(ROOT / "src/pingpong_hil_rl/config/hil_flashsac.yaml")
    agent_cfg = _agent_config(cfg.flashsac, total_steps=100_000, seed=7, updates_per_step=0.25)
    assert agent_cfg.learning_rate_decay_step == 25_000
    assert agent_cfg.learning_rate_warmup_step == 250


def test_only_known_near_zero_simulator_idle_pose_gets_special_recovery():
    cfg = SimpleNamespace(
        allow_simulator_idle_recovery=True,
        simulator_idle_degrees=[0.0] * 7,
        simulator_idle_tolerance_deg=10.0,
    )
    assert _is_known_simulator_idle_pose(np.deg2rad([1, -2, 3, 4, -5, 6, -7]), cfg)
    assert not _is_known_simulator_idle_pose(np.deg2rad([1, -2, 3, 40, -5, 6, -7]), cfg)
