#!/usr/bin/env python3
"""Replay a record_new3/4 trajectory through the recovered real RMP path."""

from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import sys


def _workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "recovered_rmp").is_dir():
            return parent
    raise RuntimeError("cannot locate workspace containing recovered_rmp")


def _joint_scales(value: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value.split(","))
    if len(result) != 7:
        raise argparse.ArgumentTypeError("expected seven comma-separated values")
    return result


WORKSPACE = _workspace_root()
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from recovered_rmp.replay import (  # noqa: E402
    IdealRMPControllerConfig,
    load_control_recording,
    replay_in_mujoco,
    save_replay_result,
)
from recovered_rmp.params import load_movax_rmp_config  # noqa: E402


DEFAULT_MODEL = Path(__file__).resolve().with_name("moz1_pd.xml")
DEFAULT_MOVAX_CONFIG = (
    WORKSPACE
    / "movax/controller/configuration/model/user/spirit01_base_e_no_neck_hh.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a recorded position-only command through recovered Movax "
            "RMP q/qd/qdd generation and an ideal MuJoCo torque-PD controller."
        )
    )
    parser.add_argument("recording", type=Path, help="probe_recording directory or control_samples.jsonl")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--movax-config", type=Path, default=DEFAULT_MOVAX_CONFIG)
    parser.add_argument("--output", type=Path, default=Path("rmp_replay_output"))
    parser.add_argument("--segment", choices=("longest", "first", "last"), default="longest")
    parser.add_argument(
        "--calibration-profile",
        choices=(
            "recovered_yaml",
            "recovered_yaml_pd_fit",
            "record_new3_4",
            "record_new3_4_q_qd_qdd",
            "record_new3_4_q_qd_qdd_cpu",
        ),
        default="recovered_yaml_pd_fit",
        help=(
            "recovered_yaml keeps recovered YAML RMP and unscaled XML PD; "
            "recovered_yaml_pd_fit changes only the unknown low-level PD; "
            "record_new3_4 also applies the explicitly empirical RMP fit"
        ),
    )
    parser.add_argument("--kp-scale", type=float)
    parser.add_argument("--kd-scale", type=float)
    parser.add_argument("--kp-scale-per-joint", type=_joint_scales, default=())
    parser.add_argument("--kd-scale-per-joint", type=_joint_scales, default=())
    parser.add_argument("--qdd-feedforward", type=float)
    parser.add_argument("--inverse-dynamics-feedforward", type=float)
    parser.add_argument("--rmp-jnt-kp-scale", type=float)
    parser.add_argument("--rmp-jnt-kd-scale", type=float)
    parser.add_argument("--torque-limit-nm", type=float, default=0.0)
    parser.add_argument("--rmp-period-ms", type=float, default=1.0)
    parser.add_argument("--disable-rmp-safety-bounds", action="store_true")
    parser.add_argument(
        "--enable-rmp-jerk-limit",
        action="store_true",
        help="enable the optional jerk bound (the recovered hardware path leaves it off)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace = load_control_recording(args.recording, segment=args.segment)
    profile_defaults = {
        # kp scale, kd scale, qdd FF, inverse-dynamics FF, RMP Kp, RMP Kd.
        "recovered_yaml": (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        "recovered_yaml_pd_fit": (0.02, 0.10, 0.0, 1.0, 1.0, 1.0),
        "record_new3_4": (0.03, 0.10, 0.0, 1.0, 0.30, math.sqrt(0.30)),
        "record_new3_4_q_qd_qdd": (
            1.0,
            1.0,
            0.0,
            1.0,
            0.37799499004617304,
            0.33600674482495757,
        ),
        "record_new3_4_q_qd_qdd_cpu": (
            1.0,
            1.0,
            0.0,
            1.0,
            0.37799499004617304,
            0.33600674482495757,
        ),
    }
    profile = profile_defaults[args.calibration_profile]

    def selected(value: float | None, default: float) -> float:
        return float(default) if value is None else float(value)

    kp_scale = selected(args.kp_scale, profile[0])
    kd_scale = selected(args.kd_scale, profile[1])
    qdd_feedforward = selected(args.qdd_feedforward, profile[2])
    inverse_dynamics_feedforward = selected(args.inverse_dynamics_feedforward, profile[3])
    rmp_jnt_kp_scale = selected(args.rmp_jnt_kp_scale, profile[4])
    rmp_jnt_kd_scale = selected(args.rmp_jnt_kd_scale, profile[5])
    cpu_fitted_kp_per_joint = (
        0.017514998814844487,
        0.017514998814844487,
        0.014124999044229425,
        0.01129999923538354,
        0.007457999495353137,
        0.011017499254498951,
        0.007227479510951313,
    )
    cpu_fitted_kd_per_joint = (
        0.14869937517868198,
        0.05756104845626399,
        0.14869937517868198,
        0.08826027429960479,
        0.07482936299314319,
        0.131910736045605,
        0.07237342492567592,
    )
    mjx_fitted_kp_per_joint = (
        0.017514998093247414,
        0.010158699005842209,
        0.014124998822808266,
        0.011299999430775642,
        0.009098759852349758,
        0.013441349379718304,
        0.010841218754649162,
    )
    mjx_fitted_kd_per_joint = (
        0.08624563366174698,
        0.08634157478809357,
        0.08624563366174698,
        0.05119095742702484,
        0.04340102896094322,
        0.07650823146104813,
        0.04197658970952034,
    )
    kp_scale_per_joint = args.kp_scale_per_joint
    kd_scale_per_joint = args.kd_scale_per_joint
    if args.calibration_profile == "record_new3_4_q_qd_qdd":
        kp_scale_per_joint = kp_scale_per_joint or mjx_fitted_kp_per_joint
        kd_scale_per_joint = kd_scale_per_joint or mjx_fitted_kd_per_joint
    elif args.calibration_profile == "record_new3_4_q_qd_qdd_cpu":
        kp_scale_per_joint = kp_scale_per_joint or cpu_fitted_kp_per_joint
        kd_scale_per_joint = kd_scale_per_joint or cpu_fitted_kd_per_joint
    base_rmp_params, _ = load_movax_rmp_config(args.movax_config, "RightArm")
    rmp_params = replace(
        base_rmp_params,
        jnt_kp=base_rmp_params.jnt_kp * rmp_jnt_kp_scale,
        jnt_kd=base_rmp_params.jnt_kd * rmp_jnt_kd_scale,
    )
    controller = IdealRMPControllerConfig(
        kp_scale=kp_scale,
        kd_scale=kd_scale,
        kp_scale_per_joint=kp_scale_per_joint,
        kd_scale_per_joint=kd_scale_per_joint,
        qdd_feedforward=qdd_feedforward,
        inverse_dynamics_feedforward=inverse_dynamics_feedforward,
        torque_limit_nm=args.torque_limit_nm,
        rmp_period_s=args.rmp_period_ms * 1.0e-3,
        use_rmp_safety_bounds=not args.disable_rmp_safety_bounds,
        use_rmp_jerk_limit=args.enable_rmp_jerk_limit,
    )
    result = replay_in_mujoco(
        trace,
        args.model,
        args.movax_config,
        controller,
        rmp_params=rmp_params,
    )
    trace_path, report_path = save_replay_result(result, args.output)
    metrics = result.metrics
    print(f"recording: {trace.source}")
    print(f"calibration profile: {args.calibration_profile}")
    print(
        "RMP joint gains: "
        f"kp={rmp_params.jnt_kp:.6g}, kd={rmp_params.jnt_kd:.6g}"
    )
    print(f"samples: {metrics.samples} (policy={metrics.policy_samples})")
    print(
        "position RMS: "
        f"{metrics.position_rms_rad:.6f} rad / {metrics.position_rms_deg:.3f} deg"
    )
    print(f"velocity RMS: {metrics.velocity_rms_rad_s:.6f} rad/s")
    print(f"acceleration RMS: {metrics.acceleration_rms_rad_s2:.6f} rad/s^2")
    print(f"RMP-to-real position RMS: {metrics.rmp_position_rms_rad:.6f} rad")
    print(f"trace: {trace_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
