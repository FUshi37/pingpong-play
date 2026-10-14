"""Visualize the juggle_random MuJoCo environment to sanity-check the racket model."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco as mj
import mujoco.viewer as mj_viewer
import numpy as np

# Make sibling modules importable when the script is launched via absolute
# path (e.g. `python3 /path/to/tools/rl_sim/show_env.py`).
_RL_SIM_DIR = Path(__file__).resolve().parent
if str(_RL_SIM_DIR) not in sys.path:
    sys.path.insert(0, str(_RL_SIM_DIR))

from rl_juggle_env_random import JuggleConfig, JuggleEnv


# Non-ROS standalone script; resolve paths inside the pingpong_controller
# package so the XML and any generated outputs stay with the source tree.
RL_SIM_DIR = _RL_SIM_DIR
DEFAULT_XML = str(RL_SIM_DIR / "moz1_pd.xml")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=DEFAULT_XML)
    parser.add_argument("--steps", type=int, default=2000)
    args = parser.parse_args()

    print(f"[show_env] xml path: {args.xml}")
    print("[show_env] racket center offset = 0.215 m")
    print("[show_env] racket radius = 0.075 m")
    print("[show_env] racket total thickness = 0.011 m")

    cfg = JuggleConfig(domain_randomization=False)
    env = JuggleEnv(xml_path=args.xml, cfg=cfg)
    obs, _ = env.reset(seed=0)

    zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
    with mj_viewer.launch_passive(env.model, env.data) as viewer:
        for _ in range(args.steps):
            if not viewer.is_running():
                break
            obs, _, terminated, truncated, _ = env.step(zero_action)
            viewer.sync()
            time.sleep(env.dt)
            if terminated or truncated:
                obs, _ = env.reset()


if __name__ == "__main__":
    main()
