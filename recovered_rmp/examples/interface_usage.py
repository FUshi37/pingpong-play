"""Minimal usage example for the RMP interface layer.

Runs a short target sweep through the JSON-configured planner with no
simulator attached and prints the resulting command arrays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from recovered_rmp.interface import (
    commands_to_arrays,
    create_planner,
    drive_joint_rmp,
    load_rmp_config,
)

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "rmp_equiv_refine_baseline.json"


def main() -> None:
    # 1. Load and inspect the JSON config.
    config = load_rmp_config(CONFIG)
    print(f"loaded {config.meta.get('name')}: dt={config.dt}s, "
          f"jnt_kp={config.rmp.jnt_kp}, filter={config.rmp.filter_length}")
    print("xml_pd kp_scale_per_joint:", config.xml_pd.get("kp_scale_per_joint"))
    print("xml_pd output delay steps:", config.output_delay_steps_per_joint)

    # 2. Build a planner directly (useful for step-by-step integration into a
    #    custom simulation loop).
    planner = create_planner(config)
    planner.reset(np.zeros(7))
    command = planner.step(np.zeros(7), target_updated=True)
    print("first tick command qdd:", np.round(command.acceleration, 4))

    # 3. Or run a whole target trajectory in one call.  Targets arrive at
    #    200 Hz (5 ms), the planner internally advances at 1 kHz.
    t = np.linspace(0.0, 1.0, 200)
    sweep = 0.1 * np.sin(2.0 * np.pi * 0.5 * t)
    targets = np.zeros((t.size, 7))
    targets[:, 0] = sweep
    targets[:, 1] = -0.5 * sweep

    commands = drive_joint_rmp(
        targets,
        config,
        initial_position=np.zeros(7),
        control_period=0.005,
    )
    arrays = commands_to_arrays(commands)
    print(f"ran {len(commands)} servo ticks ({len(commands) * config.dt:.2f}s)")
    print("final command q:", np.round(arrays["position"][-1], 4))
    print("final command qd:", np.round(arrays["velocity"][-1], 4))
    print("final command qdd:", np.round(arrays["acceleration"][-1], 4))


if __name__ == "__main__":
    main()
