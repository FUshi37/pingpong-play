from pathlib import Path
from types import SimpleNamespace
import csv
import time

import numpy as np
import pytest

from pingpong_hil_rl.hil_env import (
    HilSafetyConfig,
    bounded_contact_impulse_reward,
    build_simcode4_hil_config,
    load_simcode4_env,
    make_hil_env_class,
)
from pingpong_hil_rl.ros_plant import JointSample
from pingpong_hil_rl.replay_equivalence import mirror_capture


ROOT = Path(__file__).resolve().parents[3]
RL_DIR = ROOT / "simcode4/pingpong_moz1-main/pingpong_controller/tools/rl_sim"


class DummyPlant:
    def __init__(self):
        self.cfg = SimpleNamespace(control_hz=200.0, state_timeout_s=0.05)
        self.sequence = 1
        self.source_ns = 1_000_000_000
        self.q = np.deg2rad([5.736, -44.399, 30.683, 97.142, 49.323, -12.269, 14.214])
        self.dq = np.zeros(7)
        self.targets = []
        self.move_calls = 0

    def _sample(self):
        return JointSample(
            self.sequence,
            self.source_ns,
            time.monotonic_ns(),
            self.q.copy(),
            self.dq.copy(),
            "joint_state",
        )

    def wait_for_state(self, timeout_s=None):
        return self._sample()

    def publish_and_wait(self, target):
        before = self._sample()
        old_q = self.q.copy()
        self.targets.append(np.asarray(target, dtype=np.float64).copy())
        self.q = old_q + 0.05 * (np.asarray(target) - old_q)
        self.dq = (self.q - old_q) / 0.005
        self.sequence += 1
        self.source_ns += 5_000_000
        return before, self._sample()

    def publish(self, target):
        pass

    def move_to_pose(self, target, *, speed_deg_s, tolerance_deg, timeout_s):
        self.move_calls += 1
        self.q = np.asarray(target, dtype=np.float64).copy()
        self.dq[:] = 0.0
        self.sequence += 1
        self.source_ns += 5_000_000
        return self._sample()


def test_hermite_endpoints_and_velocity():
    module = load_simcode4_env(RL_DIR)
    cls = make_hil_env_class(module)
    q0 = np.arange(7, dtype=float) * 0.1
    q1 = q0 + 0.02
    dq0 = np.full(7, 0.3)
    dq1 = np.full(7, -0.1)
    qa, dqa = cls._hermite(q0, dq0, q1, dq1, 0.005, 0.0)
    qb, dqb = cls._hermite(q0, dq0, q1, dq1, 0.005, 1.0)
    np.testing.assert_allclose(qa, q0)
    np.testing.assert_allclose(qb, q1)
    np.testing.assert_allclose(dqa, dq0)
    np.testing.assert_allclose(dqb, dq1)


def test_contact_impulse_reward_is_bounded_and_edge_gated():
    reward, delta = bounded_contact_impulse_reward(
        -0.4,
        -0.1,
        active_contact_candidate=True,
        target_delta_vz=1.0,
        weight=2.0,
    )
    assert delta == pytest.approx(0.3)
    assert reward == pytest.approx(0.6)
    capped, _ = bounded_contact_impulse_reward(
        -0.5,
        1.0,
        active_contact_candidate=True,
        target_delta_vz=1.0,
        weight=2.0,
    )
    assert capped == pytest.approx(2.0)
    inactive, _ = bounded_contact_impulse_reward(
        -0.4,
        0.2,
        active_contact_candidate=False,
        target_delta_vz=1.0,
        weight=2.0,
    )
    assert inactive == 0.0


def test_hil_observation_and_step_use_measured_arm():
    module = load_simcode4_env(RL_DIR)
    cfg = build_simcode4_hil_config(module)
    cfg.domain_randomization = False
    cls = make_hil_env_class(module)
    plant = DummyPlant()
    env = cls(
        xml_path=str(RL_DIR / "moz1_pd.xml"),
        plant=plant,
        cfg=cfg,
        hil_cfg=HilSafetyConfig(command_history_steps=40),
    )
    obs, info = env.reset(seed=3)
    assert info["actuator_model"] is False
    assert obs.shape == env.observation_space.shape
    assert obs.shape == (50 + 40 * 7 + 7 + 2 + 5,)
    next_obs, reward, terminated, truncated, step_info = env.step(np.zeros(7, dtype=np.float32))
    assert next_obs.shape == obs.shape
    assert np.isfinite(reward)
    assert not terminated
    assert not truncated
    assert step_info["hil/source_dt_s"] == 0.005
    np.testing.assert_allclose(next_obs[:7], plant.q.astype(np.float32), atol=1e-5)


def test_synthetic_plant_randomization_is_disabled():
    module = load_simcode4_env(RL_DIR)
    cfg = build_simcode4_hil_config(module)
    assert cfg.dr_randomize_actuator is False
    assert cfg.dr_randomize_latency is False
    assert cfg.dr_randomize_ball is True
    assert cfg.dr_randomize_contact is True


def test_curriculum_defers_ball_and_contact_dr_until_requested():
    module = load_simcode4_env(RL_DIR)
    cfg = build_simcode4_hil_config(module)
    cls = make_hil_env_class(module)
    env = cls(
        xml_path=str(RL_DIR / "moz1_pd.xml"),
        plant=DummyPlant(),
        cfg=cfg,
        hil_cfg=HilSafetyConfig(),
    )
    env.reset(seed=31)
    common = {
        "action_acc_scale": 0.46,
        "spawn_xy_jitter": 0.0,
        "init_vz": 0.0,
        "horizon_sec": 3.0,
    }
    env.set_curriculum_stage({**common, "ball_dr": False, "contact_dr": False})
    assert not env.cfg.domain_randomization
    assert not env.cfg.dr_randomize_ball
    assert not env.cfg.dr_randomize_contact
    env.set_curriculum_stage(
        {**common, "gravity_z": -3.5, "ball_dr": False, "contact_dr": False}
    )
    env.reset(seed=32)
    assert env.model.opt.gravity[2] == pytest.approx(-3.5)
    env.set_curriculum_stage({**common, "ball_dr": True, "contact_dr": True})
    assert env.cfg.domain_randomization
    assert env.cfg.dr_randomize_ball
    assert env.cfg.dr_randomize_contact
    env.close()


def test_training_samples_gravity_continuation_but_evaluation_uses_target():
    module = load_simcode4_env(RL_DIR)
    cfg = build_simcode4_hil_config(module)
    cls = make_hil_env_class(module)
    env = cls(
        xml_path=str(RL_DIR / "moz1_pd.xml"),
        plant=DummyPlant(),
        cfg=cfg,
        hil_cfg=HilSafetyConfig(),
    )
    stage = {
        "action_acc_scale": 0.46,
        "spawn_xy_jitter": 0.0,
        "init_vz": 0.0,
        "horizon_sec": 3.0,
        "gravity_z": -6.0,
        "gravity_train_from_z": -5.0,
        "ball_dr": True,
        "contact_dr": True,
    }
    env.set_curriculum_stage(stage)
    env.set_evaluation_mode(False)
    sampled = []
    for seed in range(10):
        obs, _ = env.reset(seed=seed)
        sampled.append(float(env.model.opt.gravity[2]))
        assert obs[-5] == pytest.approx(sampled[-1] / 9.81, abs=1e-6)
    assert all(-6.0 <= value <= -5.0 for value in sampled)
    assert np.ptp(sampled) > 0.25
    env.set_evaluation_mode(True)
    obs, _ = env.reset(seed=99)
    assert env.model.opt.gravity[2] == pytest.approx(-6.0)
    assert obs[-5] == pytest.approx(-6.0 / 9.81, abs=1e-6)
    env.close()


def test_hil_ball_starts_five_to_ten_mm_above_racket_surface():
    module = load_simcode4_env(RL_DIR)
    cfg = build_simcode4_hil_config(module)
    cfg.domain_randomization = False
    cls = make_hil_env_class(module)
    env = cls(
        xml_path=str(RL_DIR / "moz1_pd.xml"),
        plant=DummyPlant(),
        cfg=cfg,
        hil_cfg=HilSafetyConfig(),
    )
    env.reset(seed=19)
    radius = float(env.model.geom_size[env.ball_geom_id, 0])
    surface_gap = float(
        np.dot(env._ball_pos() - env._racket_pos(), env._racket_xmat()[:, 2])
        - radius
    )
    assert 0.005 - 1e-6 <= surface_gap <= 0.010 + 1e-6
    assert env.cfg.ball_termination_max_z > float(env._ball_pos()[2])
    assert env.cfg.ball_termination_min_z < float(env._ball_pos()[2])
    env.close()


def test_recorded_feedback_mirror_uses_exact_hil_injection_path(tmp_path):
    capture = tmp_path / "capture.csv"
    q0 = np.deg2rad(np.asarray(HilSafetyConfig().reset_degrees))
    fieldnames = ["step", "timestamp"]
    for prefix in ("cmd_q", "actual_q", "actual_qvel"):
        fieldnames.extend(f"{prefix}/RightArm-{i}" for i in range(7))
    with capture.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(3):
            q = q0 + step * 1e-4
            row = {"step": step, "timestamp": 10.0 + step * 0.005}
            for i in range(7):
                row[f"cmd_q/RightArm-{i}"] = q[i]
                row[f"actual_q/RightArm-{i}"] = q[i]
                row[f"actual_qvel/RightArm-{i}"] = 0.02
            writer.writerow(row)
    summary = mirror_capture(
        "synthetic", capture, tmp_path / "mirror", RL_DIR, RL_DIR / "moz1_pd.xml"
    )
    assert summary["state_replacement"]["q_max_abs_rad"] <= 1e-12
    assert summary["state_replacement"]["dq_max_abs_rad_s"] <= 1e-12
    assert 5.0 <= summary["ball_contact_replay"]["initial_surface_gap_mm"] <= 10.0


def test_episode_reset_uses_feedback_gated_low_speed_recovery():
    module = load_simcode4_env(RL_DIR)
    cfg = build_simcode4_hil_config(module)
    cfg.domain_randomization = False
    cls = make_hil_env_class(module)
    plant = DummyPlant()
    hil_cfg = HilSafetyConfig(episode_reset_threshold_deg=5.0)
    env = cls(xml_path=str(RL_DIR / "moz1_pd.xml"), plant=plant, cfg=cfg, hil_cfg=hil_cfg)
    env.reset(seed=5)
    plant.q[3] += np.deg2rad(6.0)
    env._last_plant_sample = plant._sample()
    env.reset()
    assert plant.move_calls == 1
    np.testing.assert_allclose(plant.q, np.deg2rad(hil_cfg.reset_degrees), atol=1e-12)


def test_command_is_clipped_to_recovery_workspace():
    module = load_simcode4_env(RL_DIR)
    cfg = build_simcode4_hil_config(module)
    cfg.domain_randomization = False
    cls = make_hil_env_class(module)
    plant = DummyPlant()
    hil_cfg = HilSafetyConfig(max_command_delta_deg=1.25)
    env = cls(xml_path=str(RL_DIR / "moz1_pd.xml"), plant=plant, cfg=cfg, hil_cfg=hil_cfg)
    env.reset(seed=7)
    previous = plant.q.copy()
    env.arm_q_target = np.full(7, 1e3, dtype=np.float32)
    env._advance_physics()
    target = plant.targets[-1]
    reset_q = np.deg2rad(np.asarray(hil_cfg.reset_degrees))
    window = np.deg2rad(np.asarray(hil_cfg.max_pose_deviation_deg))
    assert np.all(target <= reset_q + window + 1e-12)
    assert np.all(target >= reset_q - window - 1e-12)
    assert np.max(np.abs(target - previous)) <= np.deg2rad(1.25) + 1e-12


def test_early_curriculum_adds_pre_hit_scaffold():
    module = load_simcode4_env(RL_DIR)
    cfg = build_simcode4_hil_config(module)
    cfg.domain_randomization = False
    cls = make_hil_env_class(module)
    env = cls(
        xml_path=str(RL_DIR / "moz1_pd.xml"),
        plant=DummyPlant(),
        cfg=cfg,
        hil_cfg=HilSafetyConfig(),
    )
    env.reset(seed=11)
    env.set_curriculum_stage(
        {
            "action_acc_scale": 0.18,
            "spawn_xy_jitter": 0.0,
            "spawn_z_jitter": 0.0,
            "init_vz": -0.18,
            "horizon_sec": 2.0,
            "pre_hit_flatness_weight": 4.0,
            "pre_hit_lift_weight": 2.0,
        }
    )
    _, reward, _, _, info = env.step(np.zeros(7, dtype=np.float32))
    assert np.isfinite(reward)
    assert info["hil/pre_hit_scaffold"] > 0.0


def test_coordinated_exploration_action_is_finite_and_bounded():
    module = load_simcode4_env(RL_DIR)
    cfg = build_simcode4_hil_config(module)
    cfg.domain_randomization = False
    cls = make_hil_env_class(module)
    env = cls(
        xml_path=str(RL_DIR / "moz1_pd.xml"),
        plant=DummyPlant(),
        cfg=cfg,
        hil_cfg=HilSafetyConfig(),
    )
    env.reset(seed=23)
    env.set_curriculum_stage(
        {
            "action_acc_scale": 0.30,
            "spawn_xy_jitter": 0.0,
            "init_vz": 0.0,
            "horizon_sec": 2.0,
        }
    )
    action = env.coordinated_exploration_action(
        position_offset_m=np.asarray([0.01, -0.01, 0.03]),
        linear_velocity_m_s=np.asarray([0.0, 0.0, 0.25]),
        linear_acceleration_m_s2=np.asarray([0.0, 0.0, 2.0]),
        normal_tilt_xy=np.asarray([0.02, -0.02]),
        angular_velocity_rad_s=np.zeros(3),
        angular_acceleration_rad_s2=np.zeros(3),
    )
    assert action.shape == (7,)
    assert np.all(np.isfinite(action))
    assert np.max(np.abs(action)) <= 1.0
    assert np.linalg.norm(action) > 0.05


def test_coordinated_exploration_can_bias_inclined_face_toward_world_up():
    module = load_simcode4_env(RL_DIR)
    cfg = build_simcode4_hil_config(module)
    cfg.domain_randomization = False
    cls = make_hil_env_class(module)
    env = cls(
        xml_path=str(RL_DIR / "moz1_pd.xml"),
        plant=DummyPlant(),
        cfg=cfg,
        hil_cfg=HilSafetyConfig(),
    )
    env.reset(seed=29)
    common = dict(
        position_offset_m=np.zeros(3),
        linear_velocity_m_s=np.zeros(3),
        linear_acceleration_m_s2=np.zeros(3),
        normal_tilt_xy=np.zeros(2),
        angular_velocity_rad_s=np.zeros(3),
        angular_acceleration_rad_s2=np.zeros(3),
    )
    preserve = env.coordinated_exploration_action(**common, normal_up_blend=0.0)
    flatten = env.coordinated_exploration_action(**common, normal_up_blend=1.0)
    assert np.all(np.isfinite(flatten))
    assert np.linalg.norm(flatten - preserve) > 0.05
