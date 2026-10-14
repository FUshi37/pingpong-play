#!/usr/bin/env python3
"""Validate evidence required by the measured-QVEL recovered-RMP course."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np


PROFILE = "goal_d455_measured_qvel_rmp_vertical_v1"
PROFILE_V2 = "goal_d455_measured_qvel_rmp_vertical_v2"
REQUIRED_RECORDING_GROUPS = {"record_new3", "record_new4", "record_new5"}
REQUIRED_COMPONENTS = {
    "joint_position_rad",
    "joint_velocity_rad_s",
    "joint_acceleration_rad_s2",
    "racket_position_m",
    "racket_orientation_rad",
    "racket_linear_velocity_m_s",
    "racket_angular_velocity_rad_s",
    "racket_linear_acceleration_m_s2",
    "racket_angular_acceleration_rad_s2",
}
REQUIRED_BALL_OUTCOME_COMPONENTS = {
    "free_flight_acceleration_m_s2",
    "normal_restitution_ratio",
    "tangential_velocity_delta_m_s",
    "post_contact_spin_rad_s",
    "apex_height_gain_m",
}
EXPECTED_RMP_PD_DR_RANGES = {
    "rmp_jnt_kp_mult": (0.75, 1.25),
    "rmp_jnt_kd_mult": (0.75, 1.25),
    "rmp_estimator_process_mult": (0.60, 1.60),
    "rmp_estimator_measure_mult": (0.50, 2.00),
    "rmp_velocity_feedforward": (0.35, 0.65),
    "rmp_acceleration_weight_mult": (0.50, 2.00),
    "rmp_target_filter_length": (8.0, 12.0),
    "rmp_output_delay_offset_steps": (-3.0, 3.0),
    "pd_kp_mult": (0.70, 1.30),
    "pd_kv_mult": (0.70, 1.30),
    "dof_damping_mult": (0.65, 1.45),
    "dof_armature_mult": (0.65, 1.50),
}
EXPECTED_RMP_PD_DR_RANGES_V2 = {
    "rmp_jnt_kp_mult": (0.50, 1.50),
    "rmp_jnt_kd_mult": (0.50, 1.50),
    "rmp_estimator_process_mult": (0.25, 2.50),
    "rmp_estimator_measure_mult": (0.01, 3.00),
    "rmp_velocity_feedforward": (0.20, 0.80),
    "rmp_acceleration_weight_mult": (0.25, 3.00),
    "rmp_target_filter_length": (5.0, 15.0),
    "rmp_output_delay_offset_steps": (-8.0, 5.0),
    "pd_kp_mult": (0.50, 1.50),
    "pd_kv_mult": (0.50, 1.50),
    "dof_damping_mult": (0.40, 1.80),
    "dof_armature_mult": (0.40, 2.00),
}
EXPECTED_BALL_DR_RANGES = {
    "normalized_inertia": (0.40, 2.0 / 3.0),
    "spin_x_rad_s": (-55.0, 55.0),
    "spin_y_rad_s": (-55.0, 55.0),
    "spin_z_rad_s": (-40.0, 40.0),
    "ball_friction": (0.10, 0.38),
    "racket_friction": (0.22, 0.62),
    "solref_time_s": (0.002, 0.008),
    "solref_damping": (0.62, 1.02),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_rmp_output_replay_report(path: str | Path) -> dict[str, Any]:
    """Recompute the RMP.md threshold and input-integrity gates."""

    report_path = Path(path).expanduser().resolve()
    data = json.loads(report_path.read_text(encoding="utf-8"))
    if data.get("implementation") != "training-local recovered_rmp_jax.py":
        raise ValueError("RMP replay does not identify the training-local implementation")
    expected_thresholds = {
        "position_rms_deg": 0.1,
        "velocity_rms_rad_s": 0.05,
        "acceleration_rms_rad_s2": 1.0,
    }
    thresholds = data.get("rmp_md_thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("RMP replay is missing rmp_md_thresholds")
    for name, expected in expected_thresholds.items():
        threshold = float(thresholds.get(name, np.nan))
        result = float(data.get(name, np.nan))
        if not np.isclose(threshold, expected, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"RMP replay threshold changed for {name}")
        if not np.isfinite(result) or result > threshold:
            raise ValueError(
                f"RMP output replay failed {name}: {result} > {threshold}"
            )
    if int(data.get("samples", 0)) < 3000:
        raise ValueError("RMP output replay must evaluate at least 3000 samples")
    inputs = data.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("RMP replay must include hashed input provenance")
    for label in ("datatracer", "recording", "movax_config", "implementation"):
        input_path = Path(str(inputs.get(label, ""))).expanduser().resolve()
        expected_hash = str(inputs.get(f"{label}_sha256", ""))
        if not input_path.is_file():
            raise ValueError(f"RMP replay input is missing: {input_path}")
        if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
            raise ValueError(f"RMP replay {label}_sha256 is invalid")
        if _sha256(input_path) != expected_hash:
            raise ValueError(f"RMP replay input hash changed: {input_path}")
    data["passes_rmp_md_thresholds"] = True
    data["input_hashes_verified"] = True
    return data


def _finite_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return array


def _finite_range(value: Any, name: str) -> tuple[float, float]:
    array = _finite_vector(value, name)
    if array.shape != (2,) or array[0] > array[1]:
        raise ValueError(f"{name} must be an ordered two-value range")
    return float(array[0]), float(array[1])


def _require_exact_ranges(
    actual: Any,
    expected: dict[str, tuple[float, float]],
    name: str,
) -> None:
    if not isinstance(actual, dict):
        raise ValueError(f"{name} must be an object")
    for key, expected_range in expected.items():
        actual_range = _finite_range(actual.get(key), f"{name}.{key}")
        if not np.allclose(actual_range, expected_range, rtol=0.0, atol=1.0e-9):
            raise ValueError(
                f"{name}.{key} does not match the immutable v1 course: "
                f"expected {expected_range}, got {actual_range}"
            )


def _validate_component_envelopes(
    components: Any,
    required: set[str],
    name: str,
) -> dict[str, dict[str, Any]]:
    if not isinstance(components, dict):
        raise ValueError(f"{name} must be an object")
    missing = sorted(required - set(components))
    if missing:
        raise ValueError(f"{name} is missing components: " + ", ".join(missing))
    recomputed: dict[str, dict[str, Any]] = {}
    for component_name in sorted(required):
        component = components[component_name]
        if not isinstance(component, dict):
            raise ValueError(f"{name}.{component_name} must be an object")
        real_low = _finite_vector(
            component.get("real_lower"),
            f"{name}.{component_name}.real_lower",
        )
        real_high = _finite_vector(
            component.get("real_upper"),
            f"{name}.{component_name}.real_upper",
        )
        sim_low = _finite_vector(
            component.get("dr_lower"),
            f"{name}.{component_name}.dr_lower",
        )
        sim_high = _finite_vector(
            component.get("dr_upper"),
            f"{name}.{component_name}.dr_upper",
        )
        if not (real_low.shape == real_high.shape == sim_low.shape == sim_high.shape):
            raise ValueError(f"{name}.{component_name} vectors must have identical shapes")
        if np.any(real_low > real_high) or np.any(sim_low > sim_high):
            raise ValueError(f"{name}.{component_name} contains reversed bounds")
        lower_margin = real_low - sim_low
        upper_margin = sim_high - real_high
        covered = (lower_margin >= 0.0) & (upper_margin >= 0.0)
        recomputed[component_name] = {
            "covered": bool(np.all(covered)),
            "coverage_fraction": float(np.mean(covered)),
            "worst_margin": float(min(np.min(lower_margin), np.min(upper_margin))),
        }
        if not bool(np.all(covered)):
            worst_index = int(np.argmin(np.minimum(lower_margin, upper_margin)))
            raise ValueError(
                f"{name}.{component_name} DR fails real-tail coverage at "
                f"component {worst_index}: "
                f"margin={recomputed[component_name]['worst_margin']:.6g}"
            )
    return recomputed


def _validate_pointwise_artifacts(
    manifest_path: Path,
    artifacts: Any,
    components: Any,
    required: set[str],
    name: str,
) -> dict[str, dict[str, Any]]:
    """Verify hashes and recompute every per-sample support decision."""

    if not isinstance(artifacts, dict):
        raise ValueError(f"{name} must be an object")
    if not isinstance(components, dict):
        raise ValueError(f"{name} summary object is missing")
    missing = sorted(required - set(artifacts))
    if missing:
        raise ValueError(f"{name} is missing artifacts: " + ", ".join(missing))
    recomputed: dict[str, dict[str, Any]] = {}
    for component_name in sorted(required):
        entry = artifacts[component_name]
        if not isinstance(entry, dict):
            raise ValueError(f"{name}.{component_name} must be an object")
        artifact_path = Path(str(entry.get("path", "")))
        if not artifact_path.is_absolute():
            artifact_path = manifest_path.parent / artifact_path
        expected_hash = str(entry.get("sha256", ""))
        if not artifact_path.is_file():
            raise ValueError(
                f"{name}.{component_name} artifact is missing: {artifact_path}"
            )
        if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
            raise ValueError(f"{name}.{component_name} sha256 is invalid")
        if _sha256(artifact_path) != expected_hash:
            raise ValueError(f"{name}.{component_name} artifact hash changed")
        try:
            with np.load(artifact_path, allow_pickle=False) as artifact:
                real = np.asarray(artifact["real"], dtype=float)
                sim_low = np.asarray(artifact["dr_lower"], dtype=float)
                sim_high = np.asarray(artifact["dr_upper"], dtype=float)
                axes = np.asarray(artifact["axes"])
        except (KeyError, OSError, ValueError) as exc:
            raise ValueError(
                f"{name}.{component_name} artifact is invalid: {exc}"
            ) from exc
        if (
            real.ndim != 2
            or real.shape[0] == 0
            or real.shape != sim_low.shape
            or real.shape != sim_high.shape
            or axes.ndim != 1
            or axes.size != real.shape[1]
        ):
            raise ValueError(
                f"{name}.{component_name} arrays must share nonempty [sample, axis] shape"
            )
        if not (
            np.all(np.isfinite(real))
            and np.all(np.isfinite(sim_low))
            and np.all(np.isfinite(sim_high))
        ):
            raise ValueError(f"{name}.{component_name} artifact contains nonfinite values")
        if np.any(sim_low > sim_high):
            raise ValueError(f"{name}.{component_name} artifact has reversed support")
        covered = (real >= sim_low) & (real <= sim_high)
        margin = np.minimum(real - sim_low, sim_high - real)
        fractions = np.mean(covered, axis=0)
        worst_margin = np.min(margin, axis=0)
        summary = components[component_name]
        expected_summary = {
            "real_lower": np.min(real, axis=0),
            "real_upper": np.max(real, axis=0),
            "dr_lower": np.min(sim_low, axis=0),
            "dr_upper": np.max(sim_high, axis=0),
            "pointwise_coverage_fraction": fractions,
            "pointwise_worst_margin": worst_margin,
        }
        for field, expected in expected_summary.items():
            actual = _finite_vector(
                summary.get(field), f"components.{component_name}.{field}"
            )
            if actual.shape != expected.shape or not np.allclose(
                actual, expected, rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(
                    f"components.{component_name}.{field} does not match "
                    "the hashed pointwise artifact"
                )
        if int(summary.get("pointwise_samples", -1)) != int(real.shape[0]):
            raise ValueError(
                f"components.{component_name}.pointwise_samples does not "
                "match the hashed pointwise artifact"
            )
        recomputed[component_name] = {
            "covered": bool(np.all(covered)),
            "coverage_fraction": fractions.tolist(),
            "worst_margin": worst_margin.tolist(),
            "samples": int(real.shape[0]),
            "artifact": str(artifact_path),
        }
        if not bool(np.all(covered)):
            flat_index = int(np.argmin(margin))
            sample_index, axis_index = np.unravel_index(flat_index, margin.shape)
            raise ValueError(
                f"{name}.{component_name} pointwise DR fails at sample "
                f"{sample_index}, component {axis_index}: "
                f"margin={margin[sample_index, axis_index]:.6g}"
            )
    return recomputed


def _validate_candidate_stability_artifact(
    manifest_path: Path,
    entry: Any,
    summary: Any,
) -> dict[str, Any]:
    """Verify that tail coverage was not obtained from unstable candidates."""

    if not isinstance(entry, dict) or not isinstance(summary, dict):
        raise ValueError("v2 coverage requires candidate stability evidence")
    artifact_path = Path(str(entry.get("path", "")))
    if not artifact_path.is_absolute():
        artifact_path = manifest_path.parent / artifact_path
    expected_hash = str(entry.get("sha256", ""))
    if not artifact_path.is_file():
        raise ValueError(f"candidate stability artifact is missing: {artifact_path}")
    if re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None:
        raise ValueError("candidate stability artifact sha256 is invalid")
    if _sha256(artifact_path) != expected_hash:
        raise ValueError("candidate stability artifact hash changed")
    try:
        with np.load(artifact_path, allow_pickle=False) as artifact:
            candidate_id = np.asarray(artifact["candidate_id"])
            finite = np.asarray(artifact["finite"], dtype=bool)
            position = np.asarray(
                artifact["max_position_limit_violation_rad"], dtype=float
            )
            velocity = np.asarray(
                artifact["max_velocity_limit_utilization"], dtype=float
            )
            acceleration = np.asarray(
                artifact["max_acceleration_limit_utilization"], dtype=float
            )
            stored_stable = np.asarray(artifact["stable"], dtype=bool)
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"candidate stability artifact is invalid: {exc}") from exc
    shape = candidate_id.shape
    if (
        candidate_id.ndim != 1
        or candidate_id.size == 0
        or any(
            value.shape != shape
            for value in (
                finite,
                position,
                velocity,
                acceleration,
                stored_stable,
            )
        )
        or not all(
            np.all(np.isfinite(value))
            for value in (position, velocity, acceleration)
        )
    ):
        raise ValueError("candidate stability arrays must share finite [candidate] shape")
    thresholds = summary.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("candidate stability thresholds are missing")
    expected_thresholds = {
        "max_position_limit_violation_rad": 0.02,
        "max_velocity_limit_utilization": 1.10,
        "max_acceleration_limit_utilization": 2.00,
    }
    for name, expected in expected_thresholds.items():
        if not np.isclose(
            float(thresholds.get(name, np.nan)),
            expected,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(f"candidate stability threshold changed: {name}")
    recomputed = (
        finite
        & (position <= expected_thresholds["max_position_limit_violation_rad"])
        & (velocity <= expected_thresholds["max_velocity_limit_utilization"])
        & (
            acceleration
            <= expected_thresholds["max_acceleration_limit_utilization"]
        )
    )
    if not np.array_equal(stored_stable, recomputed):
        raise ValueError("candidate stability mask does not match artifact values")
    if not bool(np.all(recomputed)):
        failed = ", ".join(str(value) for value in candidate_id[~recomputed][:8])
        raise ValueError(f"RMP/PD coverage contains unstable candidates: {failed}")
    if int(summary.get("stable_candidate_count", -1)) != int(candidate_id.size):
        raise ValueError("candidate stability summary count does not match artifact")
    if not bool(summary.get("all_candidates_stable", False)):
        raise ValueError("candidate stability summary does not pass")
    return {
        "artifact": str(artifact_path),
        "candidate_count": int(candidate_id.size),
        "all_candidates_stable": True,
    }


def validate_rmp_dr_coverage_manifest(
    path: str | Path,
    requested_mass_range_kg: tuple[float, float],
    *,
    expected_profile: str | None = None,
) -> dict[str, Any]:
    """Return the decoded manifest after recomputing every coverage gate."""

    manifest_path = Path(path).expanduser().resolve()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(data.get("schema_version", 0)) != 1:
        raise ValueError("RMP DR coverage manifest schema_version must be 1")
    profile = data.get("profile")
    supported_profiles = {PROFILE, PROFILE_V2}
    if profile not in supported_profiles:
        raise ValueError(
            "RMP DR coverage manifest profile must be one of "
            + ", ".join(sorted(repr(value) for value in supported_profiles))
        )
    if expected_profile is not None and profile != expected_profile:
        raise ValueError(
            f"RMP DR coverage manifest profile must be {expected_profile!r}"
        )
    groups = set(data.get("real_recording_groups", []))
    if not REQUIRED_RECORDING_GROUPS.issubset(groups):
        raise ValueError(
            "RMP DR coverage must include record_new3, record_new4, and record_new5"
        )
    if int(data.get("real_recording_count", 0)) < 64:
        raise ValueError("RMP DR coverage must include at least 64 usable recordings")
    independent_candidates = int(
        data.get("dr_independent_candidate_count", data.get("dr_candidate_count", 0))
    )
    if independent_candidates < 64:
        raise ValueError("RMP DR coverage must use at least 64 independent candidates")
    if data.get("dr_sampling_method") != "deterministic_lhs_plus_stress":
        raise ValueError("RMP DR coverage must use deterministic_lhs_plus_stress")
    if data.get("tail_contract") != "observed_support_min_max_after_declared_warmup":
        raise ValueError("RMP DR coverage has the wrong tail_contract")
    if data.get("orientation_representation") != "continuous_rotation_vector_rad":
        raise ValueError("racket orientation must use continuous rotation vectors")
    if float(data.get("warmup_s", -1.0)) < 0.0:
        raise ValueError("RMP DR coverage must declare a nonnegative warmup_s")
    for field in ("real_dataset_sha256", "dr_sweep_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(data.get(field, ""))) is None:
            raise ValueError(f"{field} must be a lowercase SHA-256 digest")

    expected_rmp_ranges = (
        EXPECTED_RMP_PD_DR_RANGES_V2
        if profile == PROFILE_V2
        else EXPECTED_RMP_PD_DR_RANGES
    )
    _require_exact_ranges(
        data.get("rmp_pd_dr_ranges"),
        expected_rmp_ranges,
        "rmp_pd_dr_ranges",
    )
    _require_exact_ranges(
        data.get("ball_dr_ranges"),
        EXPECTED_BALL_DR_RANGES,
        "ball_dr_ranges",
    )
    candidate_stability: dict[str, Any] | None = None
    if profile == PROFILE_V2:
        if data.get("right_arm_pd_profile") != "recovered_rmp_rmpmd_v2":
            raise ValueError("v2 coverage must use recovered_rmp_rmpmd_v2 PD")
        if data.get("time_alignment") != "wall":
            raise ValueError("v2 coverage must use wall-clock alignment")
        if data.get("time_base_contract") != (
            "wallTimeS_uniform_target_zoh_header_timed_state"
        ):
            raise ValueError("v2 coverage has the wrong wall-clock contract")
        if not np.isclose(
            float(data.get("control_dt_s", np.nan)),
            0.005,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("v2 coverage must use a 5 ms control grid")
        candidate_stability = _validate_candidate_stability_artifact(
            manifest_path,
            data.get("candidate_stability_artifact"),
            data.get("candidate_stability"),
        )
        if candidate_stability["candidate_count"] != int(
            data.get("dr_candidate_count", -1)
        ):
            raise ValueError(
                "candidate stability artifact count differs from DR sweep"
            )

    mass_samples = _finite_vector(
        data.get("new_ball_mass_measurements_kg"),
        "new_ball_mass_measurements_kg",
    )
    if mass_samples.size < 3 or np.any(mass_samples <= 0.0):
        raise ValueError("new ball mass needs at least three positive measurements")
    mass_low, mass_high = sorted(float(value) for value in requested_mass_range_kg)
    if mass_low <= 0.0 or not np.isfinite([mass_low, mass_high]).all():
        raise ValueError("requested ball mass range must be positive and finite")
    if mass_low > float(np.min(mass_samples)) or mass_high < float(np.max(mass_samples)):
        raise ValueError("training ball-mass support does not contain all measurements")
    manifest_mass_range = _finite_range(
        data.get("training_ball_mass_range_kg"),
        "training_ball_mass_range_kg",
    )
    if not np.allclose(
        manifest_mass_range,
        (mass_low, mass_high),
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError("launcher and coverage-manifest ball-mass ranges differ")
    diameter_samples = _finite_vector(
        data.get("new_ball_diameter_measurements_m"),
        "new_ball_diameter_measurements_m",
    )
    if diameter_samples.size < 3 or np.any(
        (diameter_samples < 0.0395) | (diameter_samples > 0.0405)
    ):
        raise ValueError(
            "new ball needs three diameter measurements within the fixed "
            "40 mm simulation geometry tolerance [0.0395, 0.0405] m"
        )

    recomputed = _validate_component_envelopes(
        data.get("components"), REQUIRED_COMPONENTS, "components"
    )
    recomputed_pointwise = _validate_pointwise_artifacts(
        manifest_path,
        data.get("component_artifacts"),
        data.get("components"),
        REQUIRED_COMPONENTS,
        "component_artifacts",
    )
    recomputed_ball = _validate_component_envelopes(
        data.get("ball_outcome_components"),
        REQUIRED_BALL_OUTCOME_COMPONENTS,
        "ball_outcome_components",
    )
    recomputed_ball_pointwise = _validate_pointwise_artifacts(
        manifest_path,
        data.get("ball_outcome_artifacts"),
        data.get("ball_outcome_components"),
        REQUIRED_BALL_OUTCOME_COMPONENTS,
        "ball_outcome_artifacts",
    )
    if int(data.get("ball_outcome_trial_count", 0)) < 30:
        raise ValueError("ball outcome coverage requires at least 30 physical trials")

    curve_paths = data.get("worst_component_curves")
    required_curve_count = len(REQUIRED_COMPONENTS) + len(
        REQUIRED_BALL_OUTCOME_COMPONENTS
    )
    if not isinstance(curve_paths, list) or len(curve_paths) < required_curve_count:
        raise ValueError(
            "coverage manifest needs one worst-component curve per RMP/PD and "
            "ball-outcome component family"
        )
    for value in curve_paths:
        curve = Path(str(value))
        if not curve.is_absolute():
            curve = manifest_path.parent / curve
        if not curve.is_file() or curve.stat().st_size <= 0:
            raise ValueError(f"missing worst-component coverage curve: {curve}")
    data["recomputed_components"] = recomputed
    data["recomputed_pointwise_components"] = recomputed_pointwise
    data["recomputed_ball_outcome_components"] = recomputed_ball
    data["recomputed_pointwise_ball_outcomes"] = recomputed_ball_pointwise
    if candidate_stability is not None:
        data["recomputed_candidate_stability"] = candidate_stability
    data["passes_all_components"] = True
    data["new_ball_mass_measured"] = True
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--mass-range-kg",
        nargs=2,
        type=float,
        required=True,
        metavar=("LOW", "HIGH"),
    )
    parser.add_argument(
        "--rmp-output-replay-report",
        type=Path,
        default=None,
        help="Also validate the hashed training-local RMP output replay.",
    )
    parser.add_argument(
        "--expected-profile",
        choices=(PROFILE, PROFILE_V2),
        default=None,
    )
    args = parser.parse_args()
    try:
        result = validate_rmp_dr_coverage_manifest(
            args.manifest,
            tuple(args.mass_range_kg),
            expected_profile=args.expected_profile,
        )
        if args.rmp_output_replay_report is not None:
            result["rmp_output_replay"] = validate_rmp_output_replay_report(
                args.rmp_output_replay_report
            )
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
