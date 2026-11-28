#!/usr/bin/env python3
"""Offline unit & integration tests for MOZ1 base dynamics identification toolchain.

Strictly offline (zero real ROS publishing, zero robot motion). Verifies:
  1. Synthetic FOPDT + asymmetric accel/brake recovery with mandatory +6.094 s
     header clock bias (<5 ms delay error, <5% accel/brake error, zero bias leakage).
  2. All low/med/high waveforms & stages: zero initial/final velocity, tier bounds,
     geofence compliance, and SE(2) return-to-origin symmetry.
  3. Safety fail-closed gates: missing --live, missing CONFIRM_MOZ1_BASE_ID_LIVE,
     wrong confirmation code, existing output dir, geofence trip (odom & mocap),
     stale feedback (>100 ms), backward timestamps, NaN/Inf, conflicting publishers,
     and guaranteed >=50 Hz / >=0.5 s zero-velocity brake bursts in ``finally``.
  4. Contract fail-closed validation on ``UNRESOLVED_REQUIRE_OPERATOR_FILL``, CLI
     ``--set`` overrides, and both base/arm command protocols.
  5. Reuse of repository ``pingpong_replay.cartesian_fk.RightArmFK``.
  6. Read-only interface inspection graceful degradation and CLI ``--dry-run``.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_MOZ1 = REPO_ROOT / "src" / "pingpong_moz1"
SRC_REPLAY = REPO_ROOT / "src" / "pingpong_replay"
for _p in (str(SRC_MOZ1), str(SRC_REPLAY)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pingpong_controller.tools.base_id.contract import (
    CONTRACT_SCHEMA_VERSION,
    UNRESOLVED_SENTINEL,
    apply_dot_overrides,
    build_default_contract_template,
    validate_contract,
    ContractValidationError,
)
from pingpong_controller.tools.base_id.inspect_base_interface import (
    TopicEndpointSummary,
    format_inspection_report,
    inspect_base_interface,
    write_contract_yaml,
)
from pingpong_controller.tools.base_id.waveforms import (
    DEFAULT_TIERS,
    SAFE_MAX_WHEEL_VEL_RAD_S,
    VALID_STAGES,
    build_stage_plan,
    combined_diag_and_arc,
    max_omni_wheel_speed_rad_s,
    prbs_or_multisine_small,
    ramp_accel_sweep,
    step_vx_pos_neg,
    step_vy_pos_neg,
    step_wz_pos_neg,
    verify_plan_within_geofence,
)
from pingpong_controller.tools.base_id.safety import (
    DeadReckoningOdom,
    FeedbackSnapshot,
    LIVE_CONFIRM_ENV_VAR,
    LiveGateError,
    OutputExistsError,
    RealTimeSafetySupervisor,
    RobotStatusSnapshot,
    SafetyTripError,
    create_unique_run_dir,
    enforce_live_gates,
    execute_zero_brake_burst,
)
from pingpong_controller.tools.base_id.run_base_id_experiment import (
    TelemetryStore,
    compile_aligned_timeseries_npz,
    compute_racket_pose_and_normal,
    decode_mocap_base_datagram,
    execute_live_stage,
    load_existing_right_arm_fk,
    load_mocap_calibration_matrix,
    main as run_experiment_main,
    populate_ros_arm_hold_msg,
    populate_ros_base_command_msg,
)
from pingpong_controller.tools.base_id.fit_moz1_base_dynamics import (
    LoadedTrial,
    OutputDirExistsError,
    PARAMS_SCHEMA_VERSION,
    analyze_odom_vs_mocap,
    analyze_racket_jitter,
    estimate_rigid_body_lever_arm,
    run_offline_fit,
    simulate_fopdt_asymmetric,
)


def _make_resolved_contract(protocol: str = "vector3_vel") -> Dict[str, Any]:
    c = build_default_contract_template()
    if protocol == "vector3_vel":
        c["base_command"].update({
            "protocol": "vector3_vel",
            "topic": "/mx_base_vel_command",
            "msg_type": "geometry_msgs/msg/Vector3",
            "field_mapping": {"vx_m_s": "x", "vy_m_s": "y", "wz_rad_s": "z"},
        })
    else:
        c["base_command"].update({
            "protocol": "mech_unit_value_array",
            "topic": "/mx_base_cmd",
            "msg_type": "mc_core_interface/msg/MechUnitValueArray",
            "mech_unit_name": "base",
            "mode": 1,
            "frame_type": 0,
            "field_mapping": {
                "vx_m_s": "cartesian.twist.linear.x",
                "vy_m_s": "cartesian.twist.linear.y",
                "wz_rad_s": "cartesian.twist.angular.z",
            },
        })
    c["base_feedback"].update({
        "primary_topic": "/mx_robot_state",
        "primary_msg_type": "mc_core_interface/msg/MechUnitValueArray",
    })
    c["arm_feedback"].update({
        "primary_topic": "/mx_robot_state",
        "primary_msg_type": "mc_core_interface/msg/MechUnitValueArray",
    })
    c["arm_concurrency_command"].update({
        "protocol": "mech_unit_value_array",
        "topic": "/mx_robot_cmd",
        "msg_type": "mc_core_interface/msg/MechUnitValueArray",
    })
    c["operator_site_verification"].update({
        "base_plus_x_physical_direction": "verified_test_x",
        "base_plus_y_physical_direction": "verified_test_y",
        "base_plus_yaw_ccw_from_top_verified": True,
        "controller_base_vel_override_verified": 1.0,
        "teleop_publisher_conflict_cleared": True,
    })
    return c


# ====================================================================
# 1. Synthetic System Identification & 6.094 s Clock Bias Tests
# ====================================================================

def _build_synthetic_run_dir(
    run_dir: Path,
    *,
    stage: str,
    trials: Sequence[Any],
    true_params: Mapping[str, Mapping[str, float]],
    header_bias_s: float = 6.094,
    noise_std: float = 0.006,
    seed: int = 42,
    include_bad_trial: bool = False,
    include_mocap: bool = True,
) -> Path:
    rng = np.random.default_rng(seed)
    run_dir.mkdir(parents=True, exist_ok=False)
    arrays: Dict[str, np.ndarray] = {}
    trial_metas: List[Dict[str, Any]] = []
    epoch_base = 1795800000.0

    for idx, tr in enumerate(trials):
        t = tr.time_s.copy()
        tid = tr.trial_id
        vx_clean = simulate_fopdt_asymmetric(
            t, tr.cmd_vx_m_s, **true_params["vx"]
        )
        vy_clean = simulate_fopdt_asymmetric(
            t, tr.cmd_vy_m_s, **true_params["vy"]
        )
        wz_clean = simulate_fopdt_asymmetric(
            t, tr.cmd_wz_rad_s, **true_params["wz"]
        )
        vx_meas = vx_clean + rng.normal(0.0, noise_std, size=len(t))
        vy_meas = vy_clean + rng.normal(0.0, noise_std, size=len(t))
        wz_meas = wz_clean + rng.normal(0.0, noise_std, size=len(t))

        # Integrate odom from measured twist.
        odom = DeadReckoningOdom()
        ox = np.zeros_like(t)
        oy = np.zeros_like(t)
        oyaw = np.zeros_like(t)
        for k in range(len(t)):
            px, py, pyaw = odom.update(float(t[k]), float(vx_meas[k]), float(vy_meas[k]), float(wz_meas[k]))
            ox[k], oy[k], oyaw[k] = px, py, pyaw

        # Synthetic racket normal tilt proportional to |a_lin| + idle baseline.
        ax_g = np.gradient(vx_clean, t)
        ay_g = np.gradient(vy_clean, t)
        a_lin = np.hypot(ax_g, ay_g)
        tilt_rad = np.deg2rad(0.08 + 0.45 * a_lin + np.abs(rng.normal(0.0, 0.03, size=len(t))))
        racket_norm = np.column_stack([
            np.sin(tilt_rad),
            np.zeros_like(tilt_rad),
            np.cos(tilt_rad),
        ])
        racket_pos = np.column_stack([
            0.001 * a_lin + rng.normal(0.0, 0.0002, size=len(t)),
            rng.normal(0.0, 0.0002, size=len(t)),
            0.90 + rng.normal(0.0, 0.0002, size=len(t)),
        ])

        wall_offset = idx * 30.0
        wall_t = wall_offset + t
        epoch_t = epoch_base + wall_t
        # Mandatory +6.094 s constant clock bias on ROS header timestamps!
        header_t = epoch_t + header_bias_s

        arrays[f"{tid}__wall_time_s"] = wall_t
        arrays[f"{tid}__t_rel_s"] = t
        arrays[f"{tid}__cmd_vx_m_s"] = tr.cmd_vx_m_s
        arrays[f"{tid}__cmd_vy_m_s"] = tr.cmd_vy_m_s
        arrays[f"{tid}__cmd_wz_rad_s"] = tr.cmd_wz_rad_s
        arrays[f"{tid}__meas_vx_m_s"] = vx_meas
        arrays[f"{tid}__meas_vy_m_s"] = vy_meas
        arrays[f"{tid}__meas_wz_rad_s"] = wz_meas
        arrays[f"{tid}__odom_x_m"] = ox
        arrays[f"{tid}__odom_y_m"] = oy
        arrays[f"{tid}__odom_yaw_rad"] = oyaw
        arrays[f"{tid}__racket_pos_m"] = racket_pos
        arrays[f"{tid}__racket_normal"] = racket_norm
        arrays[f"{tid}__phase_labels"] = np.array(tr.phase_labels)
        arrays[f"{tid}__base_raw_wall_s"] = wall_t
        arrays[f"{tid}__base_raw_epoch_s"] = epoch_t
        arrays[f"{tid}__base_raw_header_s"] = header_t
        arrays[f"{tid}__arm_raw_wall_s"] = wall_t
        arrays[f"{tid}__arm_raw_epoch_s"] = epoch_t
        arrays[f"{tid}__arm_raw_header_s"] = header_t

        if tr.arm_concurrency_enabled:
            # 200 Hz arm command timestamps with tiny realistic jitter (~0.15 ms).
            n_arm = max(20, int(round(tr.duration_s * 200.0)))
            arm_t = wall_offset + np.arange(n_arm, dtype=np.float64) * 0.005
            arm_t += rng.normal(0.0, 0.00015, size=n_arm)
            arm_t = np.maximum.accumulate(arm_t + 1e-6)
            arrays[f"{tid}__arm_cmd_wall_s"] = arm_t

        if include_mocap and tr.stage != "s0_idle_noise":
            # External mocap matching odom with 1.5% scale factor.
            arrays[f"{tid}__mocap_raw_wall_s"] = wall_t
            arrays[f"{tid}__mocap_raw_epoch_s"] = epoch_t
            arrays[f"{tid}__mocap_raw_header_s"] = epoch_t - 0.012
            arrays[f"{tid}__mocap_pos_xyz_m"] = np.column_stack([ox * 1.015, oy * 1.015, np.zeros_like(ox)])
            arrays[f"{tid}__mocap_yaw_rad"] = oyaw

        trial_metas.append({**tr.summary_dict(), "status": "completed", "error": None})

    if include_bad_trial:
        # Inject one aborted trial so DATA_AUDIT.json must explicitly reject and record it.
        bad_id = "bad_aborted_trial"
        t_bad = np.linspace(0.0, 1.0, 50)
        arrays[f"{bad_id}__t_rel_s"] = t_bad
        arrays[f"{bad_id}__cmd_vx_m_s"] = np.zeros_like(t_bad)
        arrays[f"{bad_id}__cmd_vy_m_s"] = np.zeros_like(t_bad)
        arrays[f"{bad_id}__cmd_wz_rad_s"] = np.zeros_like(t_bad)
        arrays[f"{bad_id}__meas_vx_m_s"] = np.zeros_like(t_bad)
        arrays[f"{bad_id}__meas_vy_m_s"] = np.zeros_like(t_bad)
        arrays[f"{bad_id}__meas_wz_rad_s"] = np.zeros_like(t_bad)
        trial_metas.append({
            "trial_id": bad_id,
            "waveform_name": "step_vx_pos_neg",
            "stage": stage,
            "tier": "med",
            "split_role": "train",
            "primary_axis": "vx",
            "status": "aborted_safety",
            "error": "GEOFENCE_RADIUS_ODOM",
        })

    np.savez_compressed(str(run_dir / "aligned_timeseries.npz"), **arrays)
    manifest = {
        "schema_version": "moz1_base_id_run_v1",
        "stage": stage,
        "mode": "live",
        "status": "completed",
        "trials": trial_metas,
        "clock_offset_summary_s": {
            "base_state": {
                "n_samples": 1000,
                "median_header_minus_epoch_s": header_bias_s,
                "mad_s": 0.0001,
            }
        },
    }
    (run_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return run_dir


def test_synthetic_fopdt_recovery_with_6094ms_clock_bias(tmp_path: Path) -> None:
    """Verify <5 ms delay accuracy, <5% accel/brake accuracy, and 6.094 s bias separation."""
    true_params = {
        "vx": {
            "delay_s": 0.110,
            "time_constant_s": 0.140,
            "gain": 1.00,
            "a_acc_max": 1.20,
            "a_brake_max": 1.60,
        },
        "vy": {
            "delay_s": 0.125,
            "time_constant_s": 0.155,
            "gain": 0.98,
            "a_acc_max": 1.10,
            "a_brake_max": 1.50,
        },
        "wz": {
            "delay_s": 0.095,
            "time_constant_s": 0.120,
            "gain": 1.00,
            "a_acc_max": 2.20,
            "a_brake_max": 2.80,
        },
    }

    med = DEFAULT_TIERS["med"]
    high = DEFAULT_TIERS["high"]
    # Build rich step trials (high steps cleanly excite both accel & brake saturation limits).
    trials_s2 = [
        step_vx_pos_neg(0.45, tier_cfg=med, dt_s=0.02, stage="s2_step_ramp_med", trial_id="vx_med_train", split_role="train"),
        step_vx_pos_neg(0.85, tier_cfg=high, dt_s=0.02, stage="s3_step_brake_high", trial_id="vx_high_train", split_role="train"),
        step_vx_pos_neg(0.75, tier_cfg=high, dt_s=0.02, stage="s3_step_brake_high", trial_id="vx_high_heldout", split_role="heldout"),
        step_vy_pos_neg(0.80, tier_cfg=high, dt_s=0.02, stage="s3_step_brake_high", trial_id="vy_high_train", split_role="train"),
        step_vy_pos_neg(0.50, tier_cfg=med, dt_s=0.02, stage="s2_step_ramp_med", trial_id="vy_med_heldout", split_role="heldout"),
        step_wz_pos_neg(1.30, tier_cfg=high, dt_s=0.02, stage="s3_step_brake_high", trial_id="wz_high_train", split_role="train"),
        step_wz_pos_neg(0.80, tier_cfg=med, dt_s=0.02, stage="s2_step_ramp_med", trial_id="wz_med_heldout", split_role="heldout"),
    ]
    trials_s0 = build_stage_plan("s0_idle_noise", dt_s=0.02)
    trials_s5 = build_stage_plan("s5_arm_concurrency", dt_s=0.02)[:1]

    run_s0 = _build_synthetic_run_dir(
        tmp_path / "run_s0", stage="s0_idle_noise", trials=trials_s0,
        true_params=true_params, header_bias_s=6.094, noise_std=0.004, seed=1,
    )
    run_dyn = _build_synthetic_run_dir(
        tmp_path / "run_dyn", stage="s2_step_ramp_med", trials=trials_s2,
        true_params=true_params, header_bias_s=6.094, noise_std=0.005, seed=2,
        include_bad_trial=True,
    )
    run_s5 = _build_synthetic_run_dir(
        tmp_path / "run_s5", stage="s5_arm_concurrency", trials=trials_s5,
        true_params=true_params, header_bias_s=6.094, noise_std=0.005, seed=3,
    )

    out_dir = tmp_path / "fit_summary"
    params = run_offline_fit([run_s0, run_dyn, run_s5], out_dir)

    # 1. Verify required artifacts exist.
    assert (out_dir / "DATA_AUDIT.json").is_file()
    assert (out_dir / "moz1_base_identified_params_v1.json").is_file()
    assert (out_dir / "BASE_ID_REPORT_CN.md").is_file()
    assert (out_dir / "fig_moz1_base_id_summary.pdf").is_file()
    assert (out_dir / "fig_moz1_base_id_summary.png").is_file()

    # 2. Verify precision on vx (explicitly specified benchmark in docs/basetest.md).
    vx = params["axes"]["vx"]
    assert abs(vx["delay_s"] - 0.110) < 0.005, f"vx delay error too large: {vx['delay_s']}"
    assert abs(vx["a_acc_max"] - 1.20) / 1.20 < 0.05, f"vx a_acc error >5%: {vx['a_acc_max']}"
    assert abs(vx["a_brake_max"] - 1.60) / 1.60 < 0.05, f"vx a_brake error >5%: {vx['a_brake_max']}"
    assert vx["heldout_rmse"] < 0.025
    assert vx["heldout_r2"] > 0.99

    # Also verify vy and wz recovery (and zero 6.094 s leakage!).
    for ax in ("vx", "vy", "wz"):
        assert params["axes"][ax]["delay_s"] < 0.5, f"6.094s clock bias leaked into {ax} delay!"
        assert abs(params["axes"][ax]["delay_s"] - true_params[ax]["delay_s"]) < 0.008

    # 3. Verify 6.094 s clock bias was explicitly detected and recorded in DATA_AUDIT.json.
    audit = json.loads((out_dir / "DATA_AUDIT.json").read_text(encoding="utf-8"))
    assert audit["n_rejected_trials"] == 1
    assert audit["rejected_trials"][0]["trial_id"] == "bad_aborted_trial"
    base_clk = audit["clock_offset_audit_s"]["base_state"]
    assert abs(base_clk["median_s"] - 6.094) < 0.01
    assert base_clk["large_bias_separated"] is True

    # 4. Verify schema completeness & zero NaN/Inf.
    assert params["schema_version"] == PARAMS_SCHEMA_VERSION
    assert params["kinematic_type"] == "omnidirectional_4wheel"
    assert len(params["cross_coupling_matrix_3x3"]) == 3
    rj = params["racket_jitter_model"]
    assert rj["normal_deg_per_mps2"] > 0.25
    assert set(rj["normal_deg_p95_by_tier"].keys()) == {"low", "med", "high"}
    assert set(rj["pos_mm_p95_by_tier"].keys()) == {"low", "med", "high"}
    cc = params["arm_base_concurrency"]
    assert cc["has_s5_concurrency_trials"] is True
    assert cc["blocking_detected"] is False
    assert abs(cc["arm_dt_p50_ms"] - 5.0) < 0.5
    assert params["odom_vs_mocap"]["available"] is True
    assert len(params["source_run_hashes"]) == 3

    # 5. Verify --out refuses overwrite.
    with pytest.raises(OutputDirExistsError):
        run_offline_fit([run_dyn], out_dir)


# ====================================================================
# 2. Waveform Safety, Boundary & Geofence Tests
# ====================================================================

@pytest.mark.parametrize("stage", VALID_STAGES)
def test_all_stages_zero_boundaries_and_geofence_compliance(stage: str) -> None:
    geofence_r = 1.0
    geofence_yaw = 3.5
    trials = build_stage_plan(
        stage,
        tier_limits=DEFAULT_TIERS,
        geofence_radius_m=geofence_r,
        geofence_yaw_rad=geofence_yaw,
        dt_s=0.02,
    )
    assert len(trials) >= 1
    verify_plan_within_geofence(
        trials,
        geofence_radius_m=geofence_r,
        geofence_yaw_rad=geofence_yaw,
        tier_limits=DEFAULT_TIERS,
    )
    for tr in trials:
        tcfg = DEFAULT_TIERS[tr.tier]
        # Zero initial & final velocity.
        assert tr.cmd_vx_m_s[0] == 0.0 and tr.cmd_vy_m_s[0] == 0.0 and tr.cmd_wz_rad_s[0] == 0.0
        assert tr.cmd_vx_m_s[-1] == 0.0 and tr.cmd_vy_m_s[-1] == 0.0 and tr.cmd_wz_rad_s[-1] == 0.0
        # Strict tier & geofence bounds.
        assert tr.max_theoretical_radius_m <= min(tcfg.max_disp_m, geofence_r) + 1e-6
        assert tr.max_theoretical_abs_yaw_rad <= geofence_yaw + 1e-6
        # Closed-loop return-to-origin symmetry.
        assert tr.end_pose_radius_m < 1e-3, f"{tr.trial_id} drifted by {tr.end_pose_radius_m} m"
        assert tr.end_abs_yaw_rad < 1e-3, f"{tr.trial_id} yaw drifted by {tr.end_abs_yaw_rad} rad"
        # Combined multi-axis waveforms must stay inside mc_core wheel speed polytope (<5.445 rad/s).
        if tr.waveform_name in ("combined_diag_and_arc", "prbs_or_multisine_small"):
            w_peak = max_omni_wheel_speed_rad_s(tr.cmd_vx_m_s, tr.cmd_vy_m_s, tr.cmd_wz_rad_s)
            assert w_peak <= SAFE_MAX_WHEEL_VEL_RAD_S + 1e-6, (
                f"{tr.trial_id} wheel speed {w_peak:.3f} rad/s exceeds safe polytope bound"
            )


def test_waveform_all_primitives_across_tiers() -> None:
    for tier_name, tcfg in DEFAULT_TIERS.items():
        wfs = [
            step_vx_pos_neg(tcfg.v_max_m_s, tier_cfg=tcfg),
            step_vy_pos_neg(tcfg.v_max_m_s, tier_cfg=tcfg),
            step_wz_pos_neg(tcfg.w_max_rad_s, tier_cfg=tcfg),
            ramp_accel_sweep("vx", [0.3, 0.6, 1.0], v_peak=tcfg.v_max_m_s, tier_cfg=tcfg),
            combined_diag_and_arc(
                v_diag_m_s=tcfg.v_max_m_s,
                v_arc_m_s=0.9 * tcfg.v_max_m_s,
                w_arc_rad_s=tcfg.w_max_rad_s,
                tier_cfg=tcfg,
            ),
            prbs_or_multisine_small(
                v_amp_m_s=0.7 * tcfg.v_max_m_s,
                w_amp_rad_s=0.7 * tcfg.w_max_rad_s,
                tier_cfg=tcfg,
            ),
        ]
        verify_plan_within_geofence(
            wfs, geofence_radius_m=1.0, geofence_yaw_rad=3.5, tier_limits=DEFAULT_TIERS
        )
        for wf in wfs:
            assert wf.end_pose_radius_m < 1e-3, f"{tier_name}:{wf.waveform_name} failed return"


# ====================================================================
# 3. Safety Gate, Real-Time Watchdog & Brake-Burst Tests
# ====================================================================

def test_live_gates_fail_closed() -> None:
    # Default non-live returns dry_run without prompting.
    assert enforce_live_gates(live=False, env={}, stage="s1_kinematic_low") == "dry_run"

    # Missing CONFIRM_MOZ1_BASE_ID_LIVE env var -> error.
    with pytest.raises(LiveGateError, match=LIVE_CONFIRM_ENV_VAR):
        enforce_live_gates(live=True, env={}, stage="s1_kinematic_low", prompt_fn=lambda _: "s1_kinematic_low")

    # Wrong env var value -> error.
    with pytest.raises(LiveGateError, match=LIVE_CONFIRM_ENV_VAR):
        enforce_live_gates(
            live=True,
            env={LIVE_CONFIRM_ENV_VAR: "yes"},
            stage="s1_kinematic_low",
            prompt_fn=lambda _: "s1_kinematic_low",
        )

    # Wrong interactive stage confirmation -> error.
    with pytest.raises(LiveGateError, match="expected stage confirmation"):
        enforce_live_gates(
            live=True,
            env={LIVE_CONFIRM_ENV_VAR: "YES"},
            stage="s1_kinematic_low",
            prompt_fn=lambda _: "wrong_stage",
        )

    # All three valid -> live armed.
    assert (
        enforce_live_gates(
            live=True,
            env={LIVE_CONFIRM_ENV_VAR: "YES"},
            stage="s1_kinematic_low",
            prompt_fn=lambda _: "s1_kinematic_low",
        )
        == "live"
    )


def test_create_unique_run_dir_refuses_overwrite(tmp_path: Path) -> None:
    d1 = create_unique_run_dir(tmp_path, "s1_kinematic_low", timestamp_str="20261128_000001")
    assert d1.is_dir()
    with pytest.raises(OutputExistsError):
        create_unique_run_dir(tmp_path, "s1_kinematic_low", timestamp_str="20261128_000001")


def test_realtime_supervisor_geofence_stale_backward_nan() -> None:
    sup = RealTimeSafetySupervisor(
        geofence_radius_m=0.50,
        geofence_yaw_rad=2.0,
        feedback_stale_timeout_s=0.100,
        tier_limits=DEFAULT_TIERS,
    )
    ok_base = FeedbackSnapshot(10.0, 100.0, np.array([0.0, 0.0, 0.0]), pose_xyyaw=(0.0, 0.0, 0.0))
    ok_arm = FeedbackSnapshot(10.0, 100.0, np.zeros(14))
    ok_mocap = FeedbackSnapshot(10.0, 100.0, np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), pose_xyyaw=(0.0, 0.0, 0.0))
    ok_status = RobotStatusSnapshot(10.0, 0, (1, 1, 1, 1), (1, 0, 0, 0), 1, 1)

    # Conflicting publisher fails closed when not explicitly allowed.
    with pytest.raises(SafetyTripError) as ei:
        sup.preflight_check(
            now_wall_s=10.01, base_snap=ok_base, arm_snap=ok_arm, robot_status=ok_status,
            existing_base_cmd_publishers=1, allow_existing_publishers=False,
        )
    assert ei.value.reason_code == "CONFLICTING_BASE_PUBLISHER"

    # Normal preflight latches origin.
    sup.preflight_check(
        now_wall_s=10.01, base_snap=ok_base, arm_snap=ok_arm,
        robot_status=ok_status, mocap_snap=ok_mocap, mocap_required=True,
    )

    # 1. Odom geofence breach.
    with pytest.raises(SafetyTripError) as ei:
        sup.check_step_safety(
            now_wall_s=10.02, trial_id="t1", base_snap=ok_base, arm_snap=ok_arm,
            odom_pose_xyyaw=(0.55, 0.0, 0.0), mocap_snap=ok_mocap, robot_status=ok_status,
        )
    assert ei.value.reason_code == "GEOFENCE_RADIUS_ODOM"

    # 2. Mocap geofence breach even when odom is inside.
    far_mocap = FeedbackSnapshot(10.02, 100.02, np.array([0.6, 0.0, 0.0, 0, 0, 0, 1.0]), pose_xyyaw=(0.6, 0.0, 0.0))
    with pytest.raises(SafetyTripError) as ei:
        sup.check_step_safety(
            now_wall_s=10.02, trial_id="t1", base_snap=ok_base, arm_snap=ok_arm,
            odom_pose_xyyaw=(0.1, 0.0, 0.0), mocap_snap=far_mocap, robot_status=ok_status,
        )
    assert ei.value.reason_code == "GEOFENCE_RADIUS_MOCAP"

    # 3. Stale base feedback (>100 ms).
    with pytest.raises(SafetyTripError) as ei:
        sup.check_step_safety(
            now_wall_s=10.15, trial_id="t1", base_snap=ok_base, arm_snap=FeedbackSnapshot(10.14, 100.14, np.zeros(14)),
            odom_pose_xyyaw=(0.0, 0.0, 0.0), robot_status=ok_status,
        )
    assert ei.value.reason_code == "BASE_STALE"

    # 4. Backward header timestamp.
    back_base = FeedbackSnapshot(10.05, 99.5, np.zeros(3), pose_xyyaw=(0.0, 0.0, 0.0))
    with pytest.raises(SafetyTripError) as ei:
        sup.check_step_safety(
            now_wall_s=10.06, trial_id="t1", base_snap=back_base,
            arm_snap=FeedbackSnapshot(10.05, 100.05, np.zeros(14)),
            odom_pose_xyyaw=(0.0, 0.0, 0.0), robot_status=ok_status,
        )
    assert ei.value.reason_code == "BASE_HEADER_TIME_BACKWARD"

    # 5. NaN in feedback values.
    nan_base = FeedbackSnapshot(10.07, 100.07, np.array([math.nan, 0.0, 0.0]), pose_xyyaw=(0.0, 0.0, 0.0))
    with pytest.raises(SafetyTripError) as ei:
        sup.check_step_safety(
            now_wall_s=10.08, trial_id="t1", base_snap=nan_base,
            arm_snap=FeedbackSnapshot(10.07, 100.07, np.zeros(14)),
            odom_pose_xyyaw=(0.0, 0.0, 0.0), robot_status=ok_status,
        )
    assert ei.value.reason_code == "BASE_NON_FINITE_VALUE"

    # 6. NaN in command.
    with pytest.raises(SafetyTripError) as ei:
        sup.clamp_command(math.nan, 0.0, 0.0, tier="low", dt_s=0.02)
    assert ei.value.reason_code == "CMD_NON_FINITE"


def test_zero_brake_burst_meets_50hz_and_stop_threshold() -> None:
    fake_t = [0.0]
    published: List[float] = []

    def _clock() -> float:
        return fake_t[0]

    def _sleep(dt: float) -> None:
        fake_t[0] += dt

    def _pub_zero() -> None:
        published.append(fake_t[0])

    # Simulated coasting down over 0.7 s (> 0.5 s minimum).
    def _meas() -> Tuple[float, float, float]:
        v = max(0.0, 0.5 * (1.0 - fake_t[0] / 0.7))
        return (v, 0.0, 0.0)

    rec = execute_zero_brake_burst(
        _pub_zero, _meas, rate_hz=50.0, min_duration_s=0.5, clock_fn=_clock, sleep_fn=_sleep
    )
    assert rec["stopped_confirmed"] is True
    assert rec["duration_s"] >= 0.68
    assert rec["zero_packets_sent"] >= 34


def test_execute_live_stage_trips_geofence_and_always_sends_zero_brake(tmp_path: Path) -> None:
    """Verify full executor aborts on geofence breach and ALWAYS emits zero-brake burst in finally."""
    contract = _make_resolved_contract("vector3_vel")
    contract["safety"]["geofence_radius_m"] = 0.25
    contract["safety"]["tier_limits"]["low"]["max_disp_m"] = 0.25
    fk = load_existing_right_arm_fk(contract, repo_root=REPO_ROOT)

    trial = step_vx_pos_neg(0.20, tier_cfg=DEFAULT_TIERS["low"], dt_s=0.02, hold_s=0.8, trial_id="gf_trip_test")
    fake_t = [100.0]
    sent_cmds: List[Tuple[float, float, float]] = []

    class _MockBackend:
        existing_base_cmd_publishers = 0

        def __init__(self, _cfg: Any, store: TelemetryStore) -> None:
            self.store = store
            self.sim_x = 0.0
            self.sim_vx = 0.0
            # Seed initial fresh telemetry.
            self._push_state(0.0)

        def _push_state(self, vx: float) -> None:
            self.sim_vx = vx
            self.store.on_base_state(
                wall_time_s=fake_t[0],
                epoch_time_s=1795800000.0 + fake_t[0],
                header_time_s=1795800000.0 + fake_t[0] + 6.094,
                vx_m_s=vx,
                vy_m_s=0.0,
                wz_rad_s=0.0,
                reported_pose_xyyaw=(self.sim_x, 0.0, 0.0),
                pose_source="cartesian_pose_world",
            )
            self.store.on_arm_state(
                wall_time_s=fake_t[0],
                epoch_time_s=1795800000.0 + fake_t[0],
                header_time_s=1795800000.0 + fake_t[0] + 6.094,
                q_rad=[0.1, -0.5, 0.2, 1.5, 0.8, -0.1, 0.1],
                dq_rad_s=[0.0] * 7,
            )
            self.store.on_robot_status(
                RobotStatusSnapshot(fake_t[0], 0, (1, 1, 1, 1), (1, 0, 0, 0), 1, 1),
                epoch_time_s=1795800000.0 + fake_t[0],
            )

        def publish_base_cmd(self, vx: float, vy: float, wz: float) -> None:
            sent_cmds.append((vx, vy, wz))
            # Deliberately exaggerate travel speed (3x) to force real-time geofence breach!
            self.sim_x += 3.0 * vx * 0.02
            self._push_state(vx)

        def publish_arm_hold_cmd(self, _q: Sequence[float]) -> None:
            pass

        def shutdown(self) -> None:
            pass

    def _clock() -> float:
        return fake_t[0]

    def _sleep(dt: float) -> None:
        fake_t[0] += dt

    run_dir = create_unique_run_dir(tmp_path, "s1_kinematic_low", timestamp_str="20261128_120000")
    with pytest.raises(SafetyTripError) as ei:
        execute_live_stage(
            stage="s1_kinematic_low",
            contract=contract,
            trials=[trial],
            run_dir=run_dir,
            fk=fk,
            backend_factory=lambda cfg, st: _MockBackend(cfg, st),
            repo_root=REPO_ROOT,
            clock_fn=_clock,
            epoch_fn=lambda: 1795800000.0 + fake_t[0],
            sleep_fn=_sleep,
            preflight_wait_s=0.0,
        )
    assert ei.value.reason_code == "GEOFENCE_RADIUS_ODOM"

    # Trailing commands MUST be zero-velocity brake packets (>= 25 packets at 50 Hz * 0.5 s).
    assert len(sent_cmds) >= 30
    assert all(cmd == (0.0, 0.0, 0.0) for cmd in sent_cmds[-25:])
    manifest = json.loads((run_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "aborted"
    assert len(manifest["safety_triggers"]) >= 1
    assert len(manifest["brake_bursts"]) >= 1


@pytest.mark.parametrize("fault_kind,expected_code", [
    ("stale", "BASE_STALE"),
    ("nan", "BASE_NON_FINITE_VALUE"),
])
def test_execute_live_stage_stale_and_nan_trigger_zero_brake(
    tmp_path: Path, fault_kind: str, expected_code: str
) -> None:
    """Verify stale feedback (>100 ms) and NaN both abort fail-closed and send zero-brake bursts."""
    contract = _make_resolved_contract("mech_unit_value_array")
    fk = load_existing_right_arm_fk(contract, repo_root=REPO_ROOT)
    trial = step_vx_pos_neg(0.20, tier_cfg=DEFAULT_TIERS["low"], dt_s=0.02, hold_s=0.6, trial_id=f"fault_{fault_kind}")
    fake_t = [50.0]
    sent_cmds: List[Tuple[float, float, float]] = []

    class _FaultBackend:
        existing_base_cmd_publishers = 0

        def __init__(self, _cfg: Any, store: TelemetryStore) -> None:
            self.store = store
            self.calls = 0
            self._push_ok(0.0)

        def _push_ok(self, vx: float) -> None:
            self.store.on_base_state(
                wall_time_s=fake_t[0], epoch_time_s=1795800000.0 + fake_t[0],
                header_time_s=1795800000.0 + fake_t[0], vx_m_s=vx, vy_m_s=0.0, wz_rad_s=0.0,
            )
            self.store.on_arm_state(
                wall_time_s=fake_t[0], epoch_time_s=1795800000.0 + fake_t[0],
                header_time_s=1795800000.0 + fake_t[0],
                q_rad=[0.1, -0.5, 0.2, 1.5, 0.8, -0.1, 0.1], dq_rad_s=[0.0] * 7,
            )
            self.store.on_robot_status(
                RobotStatusSnapshot(fake_t[0], 0, (1, 1, 1, 1), (1, 0, 0, 0), 1, 1),
                epoch_time_s=1795800000.0 + fake_t[0],
            )

        def publish_base_cmd(self, vx: float, vy: float, wz: float) -> None:
            sent_cmds.append((vx, vy, wz))
            self.calls += 1
            if self.calls < 10:
                self._push_ok(vx)
            elif fault_kind == "stale":
                # Freeze base updates while arm stays fresh -> trips >100 ms BASE_STALE.
                self.store.on_arm_state(
                    wall_time_s=fake_t[0], epoch_time_s=1795800000.0 + fake_t[0],
                    header_time_s=1795800000.0 + fake_t[0],
                    q_rad=[0.1, -0.5, 0.2, 1.5, 0.8, -0.1, 0.1], dq_rad_s=[0.0] * 7,
                )
            elif fault_kind == "nan":
                self.store.on_base_state(
                    wall_time_s=fake_t[0], epoch_time_s=1795800000.0 + fake_t[0],
                    header_time_s=1795800000.0 + fake_t[0],
                    vx_m_s=math.nan, vy_m_s=0.0, wz_rad_s=0.0,
                )

        def publish_arm_hold_cmd(self, _q: Sequence[float]) -> None:
            pass

        def shutdown(self) -> None:
            pass

    def _sleep(dt: float) -> None:
        fake_t[0] += dt

    run_dir = create_unique_run_dir(tmp_path, "s1_kinematic_low", timestamp_str=f"20261128_{fault_kind}")
    with pytest.raises(SafetyTripError) as ei:
        execute_live_stage(
            stage="s1_kinematic_low", contract=contract, trials=[trial], run_dir=run_dir,
            fk=fk, backend_factory=lambda cfg, st: _FaultBackend(cfg, st),
            repo_root=REPO_ROOT, clock_fn=lambda: fake_t[0],
            epoch_fn=lambda: 1795800000.0 + fake_t[0], sleep_fn=_sleep, preflight_wait_s=0.0,
        )
    assert ei.value.reason_code == expected_code
    assert len(sent_cmds) >= 25
    assert all(cmd == (0.0, 0.0, 0.0) for cmd in sent_cmds[-25:])


# ====================================================================
# 4. Contract Validation, Protocols & Existing RightArmFK Tests
# ====================================================================

def test_contract_unresolved_sentinel_fails_closed(tmp_path: Path) -> None:
    raw = build_default_contract_template()
    with pytest.raises(ContractValidationError, match=UNRESOLVED_SENTINEL):
        validate_contract(raw, repo_root=REPO_ROOT)

    # Also verify CLI returns non-zero when YAML has UNRESOLVED_REQUIRE_OPERATOR_FILL.
    bad_yaml = tmp_path / "unresolved.yaml"
    write_contract_yaml(raw, bad_yaml)
    rc = run_experiment_main([
        "--stage", "s1_kinematic_low",
        "--contract", str(bad_yaml),
        "--dry-run",
    ])
    assert rc == 2


def test_contract_both_base_protocols_and_message_builders(tmp_path: Path) -> None:
    # 1. vector3_vel (/mx_base_vel_command).
    c_vec = _make_resolved_contract("vector3_vel")
    validate_contract(c_vec, repo_root=REPO_ROOT)

    class _FakeVec3:
        def __init__(self) -> None:
            self.x = self.y = self.z = 0.0

    msg_v = populate_ros_base_command_msg(
        "vector3_vel", 0.3, -0.2, 0.4,
        base_cmd_cfg=c_vec["base_command"],
        msg_classes={"Vector3": _FakeVec3},
    )
    assert (msg_v.x, msg_v.y, msg_v.z) == (0.3, -0.2, 0.4)

    # 2. mech_unit_value_array (/mx_base_cmd, mode=1, cartesian.twist).
    c_arr = _make_resolved_contract("mech_unit_value_array")
    validate_contract(c_arr, repo_root=REPO_ROOT)

    class _FakeVal:
        def __init__(self) -> None:
            self.name = ""
            self.mode = 0
            self.joint = SimpleNamespace(position=[])
            self.cartesian = SimpleNamespace(
                frame_type=0,
                pose=SimpleNamespace(orientation=SimpleNamespace(w=0.0)),
                twist=SimpleNamespace(
                    linear=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                    angular=SimpleNamespace(x=0.0, y=0.0, z=0.0),
                ),
            )

    class _FakeArr:
        def __init__(self) -> None:
            self.header = SimpleNamespace(stamp=SimpleNamespace(sec=0, nanosec=0))
            self.values: List[Any] = []

    msg_a = populate_ros_base_command_msg(
        "mech_unit_value_array", 0.25, 0.15, -0.35,
        base_cmd_cfg=c_arr["base_command"],
        msg_classes={"MechUnitValueArray": _FakeArr, "MechUnitValue": _FakeVal},
        stamp_sec=12, stamp_nanosec=34,
    )
    assert len(msg_a.values) == 1
    v0 = msg_a.values[0]
    assert v0.name == "base"
    assert v0.mode == 1
    assert v0.cartesian.frame_type == 0
    assert (v0.cartesian.twist.linear.x, v0.cartesian.twist.linear.y, v0.cartesian.twist.angular.z) == (0.25, 0.15, -0.35)

    # Also test arm hold command builder (mode=0 joint hold on /mx_robot_cmd).
    msg_arm = populate_ros_arm_hold_msg(
        "mech_unit_value_array", [0.1, -0.5, 0.2, 1.5, 0.8, -0.1, 0.1],
        arm_cmd_cfg=c_arr["arm_concurrency_command"],
        msg_classes={"MechUnitValueArray": _FakeArr, "MechUnitValue": _FakeVal},
    )
    assert msg_arm.values[0].name == "right_arm"
    assert msg_arm.values[0].mode == 0
    assert len(msg_arm.values[0].joint.position) == 7

    # Reject unsafe frame_type=2 (World position mode) on mech_unit_value_array base velocity.
    bad_ft = apply_dot_overrides(c_arr, ["base_command.frame_type=2"])
    with pytest.raises(ContractValidationError, match="frame_type"):
        validate_contract(bad_ft, repo_root=REPO_ROOT)


def test_existing_right_arm_fk_reuse() -> None:
    c = _make_resolved_contract("vector3_vel")
    fk = load_existing_right_arm_fk(c, repo_root=REPO_ROOT)
    q = np.deg2rad([9.0, -50.0, 20.0, 90.0, 45.0, -8.0, 45.0])
    pos, normal = compute_racket_pose_and_normal(fk, q, normal_axis_index=2)
    assert pos.shape == (3,) and np.all(np.isfinite(pos))
    assert normal.shape == (3,) and abs(float(np.linalg.norm(normal)) - 1.0) < 1e-9


# ====================================================================
# 5. Read-Only Inspector & CLI --dry-run End-to-End Tests
# ====================================================================

def test_inspect_base_interface_offline_and_mocked_ros(tmp_path: Path) -> None:
    # Offline repo-only scan must degrade gracefully and flag unresolved items.
    res_off = inspect_base_interface(repo_root=REPO_ROOT, enable_ros_snapshot=False)
    assert res_off.ros_snapshot_available is False
    assert res_off.contract["base_command"]["protocol"] == UNRESOLVED_SENTINEL
    assert any(path == "base_command.protocol" for path, _ in res_off.unresolved_items)
    report_off = format_inspection_report(res_off)
    assert "UNRESOLVED_REQUIRE_OPERATOR_FILL" in report_off

    # Mocked live ROS snapshot with MCControlNode on /mx_base_vel_command & /mx_robot_cmd.
    topics = {
        "/mx_base_vel_command": "geometry_msgs/msg/Vector3",
        "/mx_robot_cmd": "mc_core_interface/msg/MechUnitValueArray",
        "/mx_robot_state": "mc_core_interface/msg/MechUnitValueArray",
        "/imu_states": "geometry_msgs/msg/Vector3Stamped",
    }
    details = {
        "/mx_base_vel_command": TopicEndpointSummary(
            "/mx_base_vel_command", "geometry_msgs/msg/Vector3",
            publisher_count=1, subscription_count=1,
            publisher_nodes=["MozTeleopRobotNode"], subscriber_nodes=["MCControlNode"],
        ),
        "/mx_robot_cmd": TopicEndpointSummary(
            "/mx_robot_cmd", "mc_core_interface/msg/MechUnitValueArray",
            publisher_count=0, subscription_count=1,
            subscriber_nodes=["MCControlNode"],
        ),
    }
    res_live = inspect_base_interface(
        repo_root=REPO_ROOT,
        ros_snapshot_override=(True, topics, details, []),
    )
    assert res_live.contract["base_command"]["protocol"] == "vector3_vel"
    assert res_live.contract["base_command"]["topic"] == "/mx_base_vel_command"
    assert res_live.contract["arm_concurrency_command"]["topic"] == "/mx_robot_cmd"
    assert any("CONFLICT CHECK" in adv for adv in res_live.advisories)

    # Dry-run CLI succeeds when site checks are supplied via --set overrides.
    yaml_path = tmp_path / "contract.yaml"
    write_contract_yaml(res_live.contract, yaml_path)
    rc = run_experiment_main([
        "--stage", "s1_kinematic_low",
        "--contract", str(yaml_path),
        "--dry-run",
        "--set", "operator_site_verification.base_plus_x_physical_direction=forward",
        "--set", "operator_site_verification.base_plus_y_physical_direction=left",
        "--set", "operator_site_verification.base_plus_yaw_ccw_from_top_verified=true",
        "--set", "operator_site_verification.controller_base_vel_override_verified=1.0",
        "--set", "operator_site_verification.teleop_publisher_conflict_cleared=true",
    ])
    assert rc == 0


def test_mocap_single_marker_and_rigid_body_decoding_with_calibration() -> None:
    """Verify single-marker & rigid-body UDP decoding + T_robot_mocap transform + geofence."""
    T_cal, loaded = load_mocap_calibration_matrix("mocap_calibration_auto.txt", repo_root=REPO_ROOT)
    assert loaded is True and T_cal.shape == (4, 4)

    # 1. Single marker JSON datagram (mm -> m auto-scaling + T_robot_mocap transform).
    marker_pkt = json.dumps({
        "marker_frmIdx": 42,
        "marker_fTimestamp": 123.456,
        "marker_1_px": 500.0,
        "marker_1_py": 1200.0,
        "marker_1_pz": 300.0,
    }).encode("utf-8")
    mocap_marker_cfg = {
        "enabled": True,
        "base_track_mode": "marker",
        "base_marker_id": 1,
        "position_unit": "auto",
        "timestamp_unit": "s",
    }
    dec_m = decode_mocap_base_datagram(marker_pkt, mocap_marker_cfg, T_robot_mocap=T_cal)
    assert dec_m is not None
    assert dec_m["track_mode"] == "marker"
    assert dec_m["has_yaw"] is False
    assert dec_m["quat_xyzw"] is None
    expected_pos = (T_cal[:3, :3] @ np.array([0.5, 1.2, 0.3])) + T_cal[:3, 3]
    assert np.allclose(dec_m["pos_xyz_m"], expected_pos, atol=1e-5)

    # 2. Rigid body JSON datagram (orientation_wxyz -> xyzw, respects tracked=False).
    rb_pkt = json.dumps({
        "marker_fTimestamp": 200.5,
        "opti_rb_1_id": 7,
        "opti_rb_1_btracked": True,
        "opti_rb_1_px": 0.2,
        "opti_rb_1_py": 0.4,
        "opti_rb_1_pz": 0.1,
        "opti_rb_1_qw": math.cos(0.25),
        "opti_rb_1_qx": 0.0,
        "opti_rb_1_qy": 0.0,
        "opti_rb_1_qz": math.sin(0.25),
    }).encode("utf-8")
    rb_cfg = {
        "enabled": True,
        "base_track_mode": "rigid_body",
        "base_rigid_body_slot": 1,
        "base_rigid_body_id": 7,
        "position_unit": "m",
        "timestamp_unit": "s",
    }
    dec_rb = decode_mocap_base_datagram(rb_pkt, rb_cfg, T_robot_mocap=np.eye(4))
    assert dec_rb is not None and dec_rb["has_yaw"] is True
    assert dec_rb["quat_xyzw"] is not None
    assert abs(dec_rb["quat_xyzw"][3] - math.cos(0.25)) < 1e-6

    # Untracked rigid body must be rejected (return None).
    rb_untracked = rb_pkt.replace(b"true", b"false")
    assert decode_mocap_base_datagram(rb_untracked, rb_cfg, T_robot_mocap=np.eye(4)) is None

    # 3. Single-marker contract validation & radial-only mocap geofence check.
    c_m = _make_resolved_contract("mech_unit_value_array")
    c_m["mocap"].update({"enabled": True, "base_track_mode": "marker", "base_marker_id": 1})
    validate_contract(c_m, repo_root=REPO_ROOT)

    sup = RealTimeSafetySupervisor(
        geofence_radius_m=0.50, geofence_yaw_rad=0.5,
        feedback_stale_timeout_s=0.100, tier_limits=DEFAULT_TIERS,
    )
    ok_base = FeedbackSnapshot(1.0, 1.0, np.zeros(3), pose_xyyaw=(0.0, 0.0, 0.0))
    ok_arm = FeedbackSnapshot(1.0, 1.0, np.zeros(14))
    ok_status = RobotStatusSnapshot(1.0, 0, (1, 1, 1, 1), (1, 0, 0, 0), 1, 1)
    m_origin = FeedbackSnapshot(1.0, 1.0, np.zeros(3), pose_xyyaw=(0.0, 0.0, 0.0), extra={"has_yaw": False})
    sup.preflight_check(
        now_wall_s=1.01, base_snap=ok_base, arm_snap=ok_arm,
        robot_status=ok_status, mocap_snap=m_origin, mocap_required=True,
    )
    # Single marker breaching radial geofence triggers GEOFENCE_RADIUS_MOCAP.
    m_far = FeedbackSnapshot(1.02, 1.02, np.array([0.6, 0.0, 0.0]), pose_xyyaw=(0.6, 0.0, 0.0), extra={"has_yaw": False})
    with pytest.raises(SafetyTripError) as ei:
        sup.check_step_safety(
            now_wall_s=1.03, trial_id="m1", base_snap=ok_base, arm_snap=ok_arm,
            odom_pose_xyyaw=(0.0, 0.0, 0.0), mocap_snap=m_far, robot_status=ok_status,
        )
    assert ei.value.reason_code == "GEOFENCE_RADIUS_MOCAP"


def test_mocap_rigid_body_yup_to_zup_rotation_and_lever_arm_geofence() -> None:
    """Verify Y-up->Z-up SO(3) similarity transform and lever-arm real-time center recovery."""
    # Exact Y-up (OptiTrack) -> Z-up (robot) orthogonal basis mapping (+Y_mocap -> +Z_robot).
    T_exact = np.eye(4, dtype=np.float64)
    T_exact[:3, :3] = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    T_real, loaded = load_mocap_calibration_matrix("mocap_calibration_auto.txt", repo_root=REPO_ROOT)
    assert loaded is True

    psi = 0.65  # rad CCW around vertical axis (+Y in OptiTrack, +Z in robot)
    lever_xy = [0.28, -0.22]
    # Rigid body sits at chassis center (1.0, -0.5, 0.2) + R_2(psi) @ lever_xy in robot frame.
    center_robot = np.array([1.0, -0.5, 0.2], dtype=np.float64)
    rb_robot = center_robot.copy()
    rb_robot[0] += math.cos(psi) * lever_xy[0] - math.sin(psi) * lever_xy[1]
    rb_robot[1] += math.sin(psi) * lever_xy[0] + math.cos(psi) * lever_xy[1]
    rb_mocap = T_exact[:3, :3].T @ rb_robot

    pkt_yaw = json.dumps({
        "marker_fTimestamp": 12.34,
        "opti_rb_1_id": 1,
        "opti_rb_1_btracked": True,
        "opti_rb_1_px": float(rb_mocap[0]),
        "opti_rb_1_py": float(rb_mocap[1]),
        "opti_rb_1_pz": float(rb_mocap[2]),
        "opti_rb_1_qw": math.cos(0.5 * psi),
        "opti_rb_1_qx": 0.0,
        "opti_rb_1_qy": math.sin(0.5 * psi),
        "opti_rb_1_qz": 0.0,
    }).encode("utf-8")

    rb_cfg = {
        "enabled": True,
        "base_track_mode": "rigid_body",
        "base_rigid_body_slot": 1,
        "base_rigid_body_id": 1,
        "base_lever_arm_xy_m": lever_xy,
        "position_unit": "m",
        "timestamp_unit": "s",
    }
    dec = decode_mocap_base_datagram(pkt_yaw, rb_cfg, T_robot_mocap=T_exact)
    assert dec is not None and dec["has_yaw"] is True
    assert dec["rpy_rad"] is not None
    # Exact Y-up +Y rotation maps to exact Z-up +Z yaw with zero roll/pitch distortion.
    assert abs(dec["rpy_rad"][0]) < 1e-9
    assert abs(dec["rpy_rad"][1]) < 1e-9
    assert abs(dec["rpy_rad"][2] - psi) < 1e-9
    # pos_xyz_m remains unshifted rigid-body pivot; pos_center_xyz_m recovers chassis center.
    assert np.allclose(dec["pos_xyz_m"], rb_robot, atol=1e-6)
    assert np.allclose(dec["pos_center_xyz_m"], center_robot, atol=1e-6)

    # Real mocap_calibration_auto.txt: identity maps to (0,0,0), +Y rotation maps cleanly to +Z yaw.
    pkt_id = json.dumps({
        "marker_fTimestamp": 12.00,
        "opti_rb_1_id": 1,
        "opti_rb_1_btracked": True,
        "opti_rb_1_px": 0.0,
        "opti_rb_1_py": 0.0,
        "opti_rb_1_pz": 0.0,
        "opti_rb_1_qw": 1.0,
        "opti_rb_1_qx": 0.0,
        "opti_rb_1_qy": 0.0,
        "opti_rb_1_qz": 0.0,
    }).encode("utf-8")
    dec_id = decode_mocap_base_datagram(pkt_id, rb_cfg, T_robot_mocap=T_real)
    assert dec_id is not None and dec_id["rpy_rad"] is not None
    assert np.allclose(dec_id["rpy_rad"], [0.0, 0.0, 0.0], atol=1e-9)

    dec_real_yaw = decode_mocap_base_datagram(pkt_yaw, rb_cfg, T_robot_mocap=T_real)
    assert dec_real_yaw is not None and dec_real_yaw["rpy_rad"] is not None
    assert abs(dec_real_yaw["rpy_rad"][2] - psi) < 0.005


def test_off_center_rigid_body_lever_arm_calibration_and_tilt_jitter() -> None:
    """Verify closed-form lever-arm calibration, arc-motion recovery, and mocap tilt folding."""
    rng = np.random.default_rng(123)
    r_b_true = np.array([0.28, -0.22], dtype=np.float64)  # ~35.6 cm off-center corner mount
    yaw_mount_offset = 0.42  # arbitrary Motive rigid-body creation yaw offset
    static_roll_mount = np.deg2rad(4.0)  # static 4-deg bracket tilt must NOT look like jitter!

    def _make_rb_trial(wf: Any, *, include_chassis_tilt: bool = False) -> LoadedTrial:
        t = wf.time_s.copy()
        vx = wf.cmd_vx_m_s.copy()
        vy = wf.cmd_vy_m_s.copy()
        wz = wf.cmd_wz_rad_s.copy()
        odom = DeadReckoningOdom()
        ox = np.zeros_like(t)
        oy = np.zeros_like(t)
        oyaw = np.zeros_like(t)
        for k in range(len(t)):
            ox[k], oy[k], oyaw[k] = odom.update(float(t[k]), float(vx[k]), float(vy[k]), float(wz[k]))

        # True ground position has a known 2.0% odom scale error on translation, 1.5% on yaw.
        true_x = ox * 1.020
        true_y = oy * 1.020
        true_yaw = oyaw * 1.015 + yaw_mount_offset

        # Off-center 4-ball rigid-body trajectory: p_rb(t) = c(t) + R_2(yaw_rb(t)) @ r_b.
        # Note r_b is expressed in the rigid-body zero-yaw frame.
        c_y = np.cos(true_yaw)
        s_y = np.sin(true_yaw)
        rb_x = true_x + (c_y * r_b_true[0] - s_y * r_b_true[1]) + rng.normal(0.0, 0.0005, size=len(t))
        rb_y = true_y + (s_y * r_b_true[0] + c_y * r_b_true[1]) + rng.normal(0.0, 0.0005, size=len(t))
        rb_z = np.full_like(t, 0.25)
        rb_yaw = true_yaw + rng.normal(0.0, 0.0015, size=len(t))

        # Chassis dynamic roll/pitch tilt proportional to linear acceleration (0.6 deg / (m/s^2)).
        ax = np.gradient(vx, t)
        ay = np.gradient(vy, t)
        a_lin = np.hypot(ax, ay)
        dyn_pitch = np.deg2rad(0.60 * a_lin) if include_chassis_tilt else np.zeros_like(t)
        rb_rpy = np.column_stack([
            np.full_like(t, static_roll_mount),
            dyn_pitch,
            rb_yaw,
        ])
        # Rigid arm (zero internal joint deflection) so any detected racket tilt comes solely from mocap RPY!
        racket_pos = np.tile(np.array([0.4, -0.2, 0.9]), (len(t), 1))
        racket_norm = np.tile(np.array([0.0, 0.0, 1.0]), (len(t), 1))

        return LoadedTrial(
            run_dir=Path("/tmp/synthetic_rb"),
            trial_id=wf.trial_id,
            waveform_name=wf.waveform_name,
            stage=wf.stage,
            tier=wf.tier,
            split_role=wf.split_role,
            primary_axis=wf.primary_axis,
            status="completed",
            t_s=t,
            cmd={"vx": vx, "vy": vy, "wz": wz},
            meas={"vx": vx, "vy": vy, "wz": wz},
            odom_x_m=ox,
            odom_y_m=oy,
            odom_yaw_rad=oyaw,
            racket_pos_m=racket_pos,
            racket_normal=racket_norm,
            imu_rpy_rad=None,
            phase_labels=np.array(wf.phase_labels),
            base_raw_wall_s=t.copy(),
            base_raw_epoch_s=t.copy(),
            base_raw_header_s=t.copy(),
            arm_raw_wall_s=None,
            arm_raw_epoch_s=None,
            arm_raw_header_s=None,
            arm_cmd_wall_s=None,
            mocap_raw_wall_s=t.copy(),
            mocap_raw_epoch_s=t.copy(),
            mocap_raw_header_s=t.copy(),
            mocap_pos_xyz_m=np.column_stack([rb_x, rb_y, rb_z]),
            mocap_yaw_rad=rb_yaw,
            mocap_rpy_rad=rb_rpy,
        )

    med = DEFAULT_TIERS["med"]
    tr_wz = _make_rb_trial(
        step_wz_pos_neg(0.8, tier_cfg=med, dt_s=0.02, stage="s2_step_ramp_med", trial_id="wz_cal"),
        include_chassis_tilt=False,
    )
    tr_arc = _make_rb_trial(
        combined_diag_and_arc(v_diag_m_s=0.40, v_arc_m_s=0.35, w_arc_rad_s=0.55, tier_cfg=med, dt_s=0.02, stage="s4_combined_and_prbs", trial_id="arc_test"),
        include_chassis_tilt=False,
    )
    tr_vx = _make_rb_trial(
        step_vx_pos_neg(0.50, tier_cfg=med, dt_s=0.02, stage="s2_step_ramp_med", trial_id="vx_tilt"),
        include_chassis_tilt=True,
    )

    # 1. Exact closed-form lever-arm recovery from pure-yaw trial (< 1.5 mm error).
    lever = estimate_rigid_body_lever_arm([tr_wz, tr_arc, tr_vx])
    assert lever["calibrated"] is True
    assert lever["n_wz_trials_used"] == 1
    assert np.allclose(lever["lever_arm_xy_m"], r_b_true, atol=0.0015)
    assert lever["fit_residual_mm"] < 2.0

    # 2. Compensated odom-vs-mocap on simultaneous arc + translation accurately recovers 2% lin & 1.5% yaw scale.
    om = analyze_odom_vs_mocap([tr_wz, tr_arc])
    assert om["available"] is True
    assert om["mocap_mode"] == "rigid_body_lever_arm_compensated"
    assert abs(om["velocity_scale_error_frac"] - (1.0 - 1.0 / 1.020)) < 0.01
    assert abs(om["yaw_rate_scale_error_frac"] - (1.0 - 1.0 / 1.015)) < 0.01
    assert om["position_drift_m_per_m"] < 0.03

    # 3. Mocap rigid-body roll/pitch folds into racket jitter when /imu_states is None,
    #    while static 4-deg mounting roll does NOT leak into idle baseline or wz tilt.
    rj, _ = analyze_racket_jitter([tr_wz, tr_vx])
    assert rj["imu_included"] is True
    assert rj["normal_deg_rms_idle"] < 0.05
    assert rj["normal_deg_per_mps2"] > 0.20
    assert rj["normal_deg_max_by_tier"]["med"] > 1.0


def test_k_rigid_body_vertical_mount_and_slot_permutation_bounds() -> None:
    """Verify 90-deg vertical K-rigid-body mount (gimbal-lock immune), multi-K slot shuffling, and brief occlusion."""
    from pingpong_controller.tools.base_id.run_base_id_experiment import _rotmat_to_quat_xyzw

    # Exact Y-up (OptiTrack) -> Z-up (robot) rotation matrix.
    T_exact = np.eye(4, dtype=np.float64)
    T_exact[:3, :3] = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)

    # K-shaped rigid body glued strictly VERTICALLY (pitch = +90 deg -> exact ZYX gimbal lock!).
    R_mount_vertical = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ], dtype=np.float64)

    def _rz(ang: float) -> np.ndarray:
        c, s = math.cos(ang), math.sin(ang)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def _build_4k_packet(chassis_slot: int, chassis_pos_mocap: Sequence[float], R_robot: np.ndarray) -> bytes:
        # Transform robot-world rotation back into Y-up mocap world rotation.
        R_mocap = T_exact[:3, :3].T @ R_robot @ T_exact[:3, :3]
        qx, qy, qz, qw = _rotmat_to_quat_xyzw(R_mocap)
        # 3 fixed distractor K-shaped rigid bodies at heights Y=0.89, 0.82, 1.74 m (matching lab measurements).
        distractors = [
            (-1.78, 0.89, -0.07),
            (-1.32, 0.82, -0.02),
            (-0.05, 1.74, 0.61),
        ]
        payload: Dict[str, Any] = {"marker_fTimestamp": 100.0}
        other_slots = [s for s in (1, 2, 3, 4) if s != chassis_slot]
        for s_idx, pos_d in zip(other_slots, distractors):
            payload.update({
                f"opti_rb_{s_idx}_id": s_idx,
                f"opti_rb_{s_idx}_btracked": True,
                f"opti_rb_{s_idx}_px": pos_d[0],
                f"opti_rb_{s_idx}_py": pos_d[1],
                f"opti_rb_{s_idx}_pz": pos_d[2],
                f"opti_rb_{s_idx}_qw": 1.0,
                f"opti_rb_{s_idx}_qx": 0.0,
                f"opti_rb_{s_idx}_qy": 0.0,
                f"opti_rb_{s_idx}_qz": 0.0,
            })
        payload.update({
            f"opti_rb_{chassis_slot}_id": 99,  # arbitrary shuffled ID
            f"opti_rb_{chassis_slot}_btracked": True,
            f"opti_rb_{chassis_slot}_px": float(chassis_pos_mocap[0]),
            f"opti_rb_{chassis_slot}_py": float(chassis_pos_mocap[1]),
            f"opti_rb_{chassis_slot}_pz": float(chassis_pos_mocap[2]),
            f"opti_rb_{chassis_slot}_qw": qw,
            f"opti_rb_{chassis_slot}_qx": qx,
            f"opti_rb_{chassis_slot}_qy": qy,
            f"opti_rb_{chassis_slot}_qz": qz,
        })
        return json.dumps(payload).encode("utf-8")

    k_cfg = {
        "enabled": True,
        "base_track_mode": "rigid_body",
        "base_rigid_body_slot": None,
        "base_rigid_body_id": None,
        # Isolate chassis K-body solely by raw mocap height Y in [0.05, 0.65] m!
        "base_rigid_body_observation_bounds_m": [None, [0.05, 0.65], None],
        "position_unit": "m",
        "timestamp_unit": "s",
    }

    # Frame 0 (stationary, slot=1): latch R_ref_robot despite pitch=+90 deg vertical mounting!
    pkt0 = _build_4k_packet(1, [0.80, 0.30, 0.40], R_mount_vertical)
    dec0_raw = decode_mocap_base_datagram(pkt0, k_cfg, T_robot_mocap=T_exact)
    assert dec0_raw is not None and dec0_raw["rotmat_robot"] is not None
    R_ref = np.asarray(dec0_raw["rotmat_robot"], dtype=np.float64)
    dec0 = decode_mocap_base_datagram(pkt0, k_cfg, T_robot_mocap=T_exact, R_ref_robot=R_ref)
    assert dec0 is not None and np.allclose(dec0["rpy_rad"], [0.0, 0.0, 0.0], atol=1e-9)

    # Frame 1 (chassis turned by +0.85 rad, UDP slot randomly jumped 1 -> 3!):
    psi = 0.85
    pkt1 = _build_4k_packet(3, [0.85, 0.30, 0.45], _rz(psi) @ R_mount_vertical)
    dec1 = decode_mocap_base_datagram(pkt1, k_cfg, T_robot_mocap=T_exact, R_ref_robot=R_ref)
    assert dec1 is not None and dec1["rpy_rad"] is not None
    assert abs(dec1["rpy_rad"][0]) < 1e-6
    assert abs(dec1["rpy_rad"][1]) < 1e-6
    assert abs(dec1["rpy_rad"][2] - psi) < 1e-6

    # Offline recovery directly from 90-deg vertical mocap_quat_xyzw (with yaw/rpy=None and a 200 ms dropout!).
    r_b_true = np.array([-0.25, 0.31], dtype=np.float64)
    med = DEFAULT_TIERS["med"]

    def _make_vert_trial(wf: Any, *, drop_slice: Optional[slice] = None) -> LoadedTrial:
        t = wf.time_s.copy()
        vx, vy, wz = wf.cmd_vx_m_s.copy(), wf.cmd_vy_m_s.copy(), wf.cmd_wz_rad_s.copy()
        odom = DeadReckoningOdom()
        ox, oy, oyaw = np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)
        for k in range(len(t)):
            ox[k], oy[k], oyaw[k] = odom.update(float(t[k]), float(vx[k]), float(vy[k]), float(wz[k]))
        c_y, s_y = np.cos(oyaw), np.sin(oyaw)
        pos_xyz = np.column_stack([
            ox + (c_y * r_b_true[0] - s_y * r_b_true[1]),
            oy + (s_y * r_b_true[0] + c_y * r_b_true[1]),
            np.full_like(t, 0.30),
        ])
        quats = np.array([
            _rotmat_to_quat_xyzw(_rz(float(y)) @ R_mount_vertical) for y in oyaw
        ], dtype=np.float64)
        keep = np.ones(len(t), dtype=bool)
        if drop_slice is not None:
            keep[drop_slice] = False
        return LoadedTrial(
            run_dir=Path("/tmp/vert_k"),
            trial_id=wf.trial_id,
            waveform_name=wf.waveform_name,
            stage=wf.stage,
            tier=wf.tier,
            split_role=wf.split_role,
            primary_axis=wf.primary_axis,
            status="completed",
            t_s=t,
            cmd={"vx": vx, "vy": vy, "wz": wz},
            meas={"vx": vx, "vy": vy, "wz": wz},
            odom_x_m=ox,
            odom_y_m=oy,
            odom_yaw_rad=oyaw,
            racket_pos_m=None,
            racket_normal=None,
            imu_rpy_rad=None,
            phase_labels=np.array(wf.phase_labels),
            base_raw_wall_s=t.copy(),
            base_raw_epoch_s=t.copy(),
            base_raw_header_s=t.copy(),
            arm_raw_wall_s=None,
            arm_raw_epoch_s=None,
            arm_raw_header_s=None,
            arm_cmd_wall_s=None,
            mocap_raw_wall_s=t[keep],
            mocap_raw_epoch_s=t[keep],
            mocap_raw_header_s=t[keep],
            mocap_pos_xyz_m=pos_xyz[keep],
            mocap_yaw_rad=None,
            mocap_rpy_rad=None,
            mocap_quat_xyzw=quats[keep],
        )

    tr_wz = _make_vert_trial(
        step_wz_pos_neg(0.8, tier_cfg=med, dt_s=0.02, stage="s2_step_ramp_med", trial_id="vert_wz"),
        drop_slice=slice(60, 72),  # 240 ms brief occlusion mid-turn!
    )
    tr_arc = _make_vert_trial(
        combined_diag_and_arc(v_diag_m_s=0.40, v_arc_m_s=0.35, w_arc_rad_s=0.55, tier_cfg=med, dt_s=0.02, stage="s4_combined_turn", trial_id="vert_arc"),
        drop_slice=slice(80, 92),  # 240 ms brief occlusion mid-arc!
    )
    lever = estimate_rigid_body_lever_arm([tr_wz, tr_arc])
    assert lever["calibrated"] is True
    assert np.allclose(lever["lever_arm_xy_m"], r_b_true, atol=0.001)
    assert lever["fit_residual_mm"] < 0.5

    om = analyze_odom_vs_mocap([tr_wz, tr_arc])
    assert om["available"] is True
    assert om["mocap_mode"] == "rigid_body_lever_arm_compensated"
    assert om["velocity_scale_error_frac"] < 0.02
    assert om["yaw_rate_scale_error_frac"] < 0.02
    assert om["position_drift_m_per_m"] < 0.01
