#!/usr/bin/env python3
"""Reproduce the MOZ1 base settling and verify the fixed-base MJX variant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
from jax import lax
import mujoco
from mujoco import mjx
import numpy as np

from mjx_smoke import _write_mjx_contact_only_xml
from rl_juggle_env_random import TARGET_DEGREES, _build_temp_xml_with_ball


WORKSPACE = Path(__file__).resolve().parents[5]
DEFAULT_XML = (
    WORKSPACE
    / "simcode7/pingpong_moz1-main/pingpong_controller/tools/rl_sim/moz1_pd.xml"
)
DEFAULT_OUTPUT = (
    WORKSPACE
    / "analysis_results/ball_racket_contact_identification_v91_static_"
    "20260901_152749_153517/base_stability_audit.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--audit-step", type=int, default=700)
    return parser.parse_args()


def _seed_posture(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for joint_name, degrees in TARGET_DEGREES.items():
        value = float(np.deg2rad(degrees))
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
        )
        if joint_id >= 0:
            data.qpos[int(model.jnt_qposadr[joint_id])] = value
        if actuator_id >= 0:
            data.ctrl[actuator_id] = value
    ball_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"
    )
    if ball_joint_id >= 0:
        ball_qadr = int(model.jnt_qposadr[ball_joint_id])
        data.qpos[ball_qadr : ball_qadr + 3] = (0.0, 0.0, 10.0)


def _ids(model: mujoco.MjModel) -> dict[str, int]:
    output = {}
    for name in (
        "base_x",
        "base_y",
        "base_z",
        "base_roll",
        "base_pitch",
        "base_yaw",
    ):
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        output[f"{name}_qadr"] = int(model.jnt_qposadr[joint_id])
    output["racket_site_id"] = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "right_ee_site"
    )
    return output


def _cpu_rollout(
    model: mujoco.MjModel,
    steps: int,
    audit_step: int,
) -> dict[str, Any]:
    data = mujoco.MjData(model)
    _seed_posture(model, data)
    mujoco.mj_forward(model, data)
    ids = _ids(model)
    initial = {
        "base_z_m": float(data.qpos[ids["base_z_qadr"]]),
        "base_roll_rad": float(data.qpos[ids["base_roll_qadr"]]),
        "base_pitch_rad": float(data.qpos[ids["base_pitch_qadr"]]),
        "racket_site_z_m": float(data.site_xpos[ids["racket_site_id"], 2]),
    }
    selected: dict[int, dict[str, float]] = {}
    for step in range(1, steps + 1):
        mujoco.mj_step(model, data)
        if step in {audit_step, steps}:
            selected[step] = {
                "base_z_m": float(data.qpos[ids["base_z_qadr"]]),
                "base_roll_rad": float(data.qpos[ids["base_roll_qadr"]]),
                "base_pitch_rad": float(data.qpos[ids["base_pitch_qadr"]]),
                "racket_site_z_m": float(
                    data.site_xpos[ids["racket_site_id"], 2]
                ),
            }
    return {
        "initial": initial,
        f"step_{audit_step}": selected[audit_step],
        f"step_{steps}": selected[steps],
        f"base_z_delta_at_{audit_step}_mm": 1000.0
        * (selected[audit_step]["base_z_m"] - initial["base_z_m"]),
        f"racket_z_delta_at_{audit_step}_mm": 1000.0
        * (
            selected[audit_step]["racket_site_z_m"]
            - initial["racket_site_z_m"]
        ),
    }


def _mjx_rollout(
    model: mujoco.MjModel,
    steps: int,
    audit_step: int,
) -> dict[str, Any]:
    cpu_data = mujoco.MjData(model)
    _seed_posture(model, cpu_data)
    mujoco.mj_forward(model, cpu_data)
    ids = _ids(model)
    initial_base_z = float(cpu_data.qpos[ids["base_z_qadr"]])
    initial_racket_z = float(cpu_data.site_xpos[ids["racket_site_id"], 2])
    mjx_model = mjx.put_model(model)
    data = mjx.put_data(model, cpu_data)

    @jax.jit
    def rollout(initial_data):
        def step(carry, _):
            carry = mjx.step(mjx_model, carry)
            return carry, (
                carry.qpos[ids["base_z_qadr"]],
                carry.qpos[ids["base_roll_qadr"]],
                carry.qpos[ids["base_pitch_qadr"]],
                carry.site_xpos[ids["racket_site_id"], 2],
            )

        return lax.scan(step, initial_data, xs=None, length=steps)

    _, trajectory = rollout(data)
    base_z, base_roll, base_pitch, racket_z = (
        np.asarray(value, dtype=np.float64) for value in trajectory
    )

    def row(index: int) -> dict[str, float]:
        return {
            "base_z_m": float(base_z[index]),
            "base_roll_rad": float(base_roll[index]),
            "base_pitch_rad": float(base_pitch[index]),
            "racket_site_z_m": float(racket_z[index]),
        }

    audit = row(audit_step - 1)
    final = row(steps - 1)
    return {
        "initial": {
            "base_z_m": initial_base_z,
            "base_roll_rad": 0.0,
            "base_pitch_rad": 0.0,
            "racket_site_z_m": initial_racket_z,
        },
        f"step_{audit_step}": audit,
        f"step_{steps}": final,
        f"base_z_delta_at_{audit_step}_mm": 1000.0
        * (audit["base_z_m"] - initial_base_z),
        f"racket_z_delta_at_{audit_step}_mm": 1000.0
        * (audit["racket_site_z_m"] - initial_racket_z),
    }


def main() -> None:
    args = parse_args()
    if args.audit_step < 1 or args.steps < args.audit_step:
        raise SystemExit("require 1 <= audit-step <= steps")
    source = args.xml.expanduser().resolve()
    ball_xml = _build_temp_xml_with_ball(source)
    variants = {
        "current_support_contact": _write_mjx_contact_only_xml(ball_xml),
        "fixed_base_equality": _write_mjx_contact_only_xml(
            ball_xml, fixed_base=True
        ),
    }
    results = {}
    for name, path in variants.items():
        model = mujoco.MjModel.from_xml_path(str(path))
        results[name] = {
            "equality_constraint_count": int(model.neq),
            "cpu_mujoco": _cpu_rollout(
                model, int(args.steps), int(args.audit_step)
            ),
            "mjx": _mjx_rollout(model, int(args.steps), int(args.audit_step)),
        }
        print(
            f"[{name}] MJX base_z@{args.audit_step}="
            f"{results[name]['mjx'][f'base_z_delta_at_{args.audit_step}_mm']:.6f} mm, "
            f"racket_z="
            f"{results[name]['mjx'][f'racket_z_delta_at_{args.audit_step}_mm']:.6f} mm",
            flush=True,
        )
    fixed_mjx = results["fixed_base_equality"]["mjx"]
    base_delta = abs(
        fixed_mjx[f"base_z_delta_at_{args.audit_step}_mm"]
    )
    output = {
        "schema_version": 1,
        "source_xml": str(source),
        "timestep_s": float(
            mujoco.MjModel.from_xml_path(
                str(variants["fixed_base_equality"])
            ).opt.timestep
        ),
        "steps": int(args.steps),
        "audit_step": int(args.audit_step),
        "cause": (
            "transparent support bottom starts about 10 mm above the floor while "
            "base_z/roll/pitch are unactuated"
        ),
        "fix": (
            "six base joint equality constraints; qpos/dof/actuator layout retained"
        ),
        "default_training_contract_changed": False,
        "fixed_base_pass": bool(base_delta <= 0.01),
        "fixed_base_abs_base_z_delta_limit_mm": 0.01,
        "variants": results,
    }
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
