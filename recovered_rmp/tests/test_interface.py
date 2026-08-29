"""Tests for the JSON-driven RMP interface layer."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from recovered_rmp.interface import (
    RMPConfig,
    commands_to_arrays,
    create_planner,
    create_planner_from_dict,
    drive_joint_rmp,
    load_rmp_config,
)
from recovered_rmp.params import RMPParams
from recovered_rmp.trajectory import JointRMPPlanner

CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "rmp_equiv_refine_baseline.json"
)


class InterfaceConfigTests(unittest.TestCase):
    def test_loads_bundled_config(self) -> None:
        config = load_rmp_config(CONFIG_PATH)
        self.assertIsInstance(config, RMPConfig)
        self.assertIsInstance(config.rmp, RMPParams)
        self.assertAlmostEqual(config.dt, 0.001)
        self.assertEqual(config.rmp.filter_length, 35)
        self.assertEqual(config.rmp.target_filter_length, 10)
        self.assertEqual(len(config.limits.q_min), 7)
        self.assertEqual(len(config.output_delay_steps_per_joint), 7)
        self.assertEqual(len(config.xml_pd.get("kp", [])), 7)
        # The JSON defaults must keep the recovered YAML gains unless scaled.
        self.assertAlmostEqual(config.rmp.jnt_kp, 400.0)
        self.assertAlmostEqual(config.rmp.jnt_kd, 40.0)

    def test_missing_fields_fall_back_to_yaml_defaults(self) -> None:
        minimal = {
            "planning": {"dt": 0.001},
            "rmp": {},
            "joint_limits": {},
        }
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as stream:
            json.dump(minimal, stream)
            stream.flush()
            config = load_rmp_config(stream.name)
        self.assertAlmostEqual(config.rmp.jnt_kp, 400.0)
        self.assertAlmostEqual(config.rmp.kp, 324.0)
        self.assertEqual(len(config.limits.q_min), 7)

    def test_per_joint_scale_application(self) -> None:
        config = load_rmp_config(CONFIG_PATH, apply_per_joint_scales=True)
        kp = np.asarray(config.rmp.jnt_kp)
        kd = np.asarray(config.rmp.jnt_kd)
        self.assertEqual(kp.shape, (7,))
        self.assertEqual(kd.shape, (7,))
        # All kp per-joint scales are 1.0 in the bundled config; kd joint 2
        # carries the fitted 0.92 scale.
        self.assertAlmostEqual(kp[2], 400.0)
        self.assertAlmostEqual(kd[2], 40.0 * 0.9200000166893005)
        self.assertAlmostEqual(kd[3], 40.0)

    def test_invalid_vectors_raise(self) -> None:
        bad = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        bad["joint_limits"]["q_min"] = [0.0]  # wrong length
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as stream:
            json.dump(bad, stream)
            stream.flush()
            with self.assertRaises(ValueError):
                load_rmp_config(stream.name)


class InterfacePlannerTests(unittest.TestCase):
    def test_create_planner_and_step(self) -> None:
        planner = create_planner(CONFIG_PATH)
        self.assertIsInstance(planner, JointRMPPlanner)
        planner.reset(np.zeros(7))
        command = planner.step(np.zeros(7), target_updated=True)
        self.assertEqual(command.position.shape, (7,))
        self.assertEqual(command.acceleration.shape, (7,))

    def test_create_planner_from_dict(self) -> None:
        planner = create_planner_from_dict(
            {"planning": {"dt": 0.001}, "rmp": {}, "joint_limits": {}}
        )
        planner.reset(np.zeros(7))
        command = planner.step(np.zeros(7), target_updated=True)
        self.assertTrue(np.all(np.isfinite(command.position)))

    def test_drive_joint_rmp_tick_structure(self) -> None:
        targets = np.zeros((20, 7))
        commands = drive_joint_rmp(
            targets, CONFIG_PATH, control_period=0.005
        )
        # 20 targets x 5 held ticks each = 100 servo ticks.
        self.assertEqual(len(commands), 100)
        arrays = commands_to_arrays(commands)
        for key in (
            "position",
            "velocity",
            "acceleration",
            "target_position",
        ):
            self.assertEqual(arrays[key].shape, (100, 7))
        self.assertTrue(np.all(np.isfinite(arrays["acceleration"])))

    def test_drive_joint_rmp_tracks_step_target(self) -> None:
        config = load_rmp_config(CONFIG_PATH)
        targets = np.zeros((400, 7))
        targets[:, 0] = 0.2
        commands = drive_joint_rmp(
            targets, config, initial_position=np.zeros(7)
        )
        arrays = commands_to_arrays(commands)
        # With a stationary 0.2 rad target the commanded position must
        # converge close to it after two seconds of 1 kHz ticks.
        self.assertLess(np.abs(arrays["position"][-1, 0] - 0.2), 0.02)
        self.assertLess(np.abs(arrays["velocity"][-1, 0]), 0.05)

    def test_drive_joint_rmp_rejects_wrong_width(self) -> None:
        with self.assertRaises(ValueError):
            drive_joint_rmp(np.zeros((10, 6)), CONFIG_PATH)


if __name__ == "__main__":
    unittest.main()
