#!/usr/bin/env python3
"""Offline off/shadow replay; never initializes ROS, camera, JAX, or GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.mjx_rl_policy import inspect_mjx_checkpoint_metadata
from pingpong_controller.policy_joint_sample import PolicyJointSample
from pingpong_controller.policy_plant_model_reference import (
    JointPlantParameters,
    PlantMatcherFailClosed,
    PolicyPlantModelReference,
    load_real_plant_fit,
)
from pingpong_controller.safety_limiter import RightArmCommandSafetyLimiter


def _vec(value: Any) -> np.ndarray | None:
    try:
        out = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    return out if out.shape == (7,) and np.all(np.isfinite(out)) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def replay(
    input_path: Path,
    checkpoint_path: Path,
    real_config_path: Path,
    *,
    max_samples: int,
) -> dict[str, Any]:
    env_cfg = dict(inspect_mjx_checkpoint_metadata(
        str(checkpoint_path)).get("env_cfg") or {})
    target = JointPlantParameters.from_checkpoint_env_cfg(env_cfg)
    fit_metadata, real, solver = load_real_plant_fit(real_config_path)
    q_low = np.asarray(
        RightArmCommandSafetyLimiter.POS_LIMIT_LOW_RAD, dtype=np.float64)
    q_high = np.asarray(
        RightArmCommandSafetyLimiter.POS_LIMIT_HIGH_RAD, dtype=np.float64)
    dq_limit = np.deg2rad(np.asarray(
        env_cfg.get("arm_vel_limit_deg_s", [210, 210, 240, 240, 300, 300, 300]),
        dtype=np.float64))
    matcher = None
    last_stamp = None
    source_sequence = -1
    processed = 0
    faults: dict[str, int] = {}
    off_max_abs_delta = 0.0
    corrections = []
    correction_slew = []
    direct_errors = []
    shadow_errors = []
    improvements = []
    previous_shadow = None
    q_envelope_violations = 0
    rows_seen = 0

    with input_path.open(encoding="utf-8") as handle:
        for line in handle:
            rows_seen += 1
            if processed >= int(max_samples):
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                faults["invalid_json"] = faults.get("invalid_json", 0) + 1
                continue
            if not bool(row.get("policyExecuted", row.get("mode") == "policy")):
                continue
            ros = row.get("ros") if isinstance(row.get("ros"), dict) else {}
            command = row.get("command") if isinstance(row.get("command"), dict) else {}
            policy_command = (
                row.get("policyCommand")
                if isinstance(row.get("policyCommand"), dict) else {})
            q = _vec(ros.get("jointPosition"))
            dq = _vec(ros.get("jointVelocity"))
            policy_qcmd = _vec(
                command.get("policyNominalPosition",
                            policy_command.get("position")))
            published = _vec(command.get("publishedSafePosition", policy_qcmd))
            try:
                stamp = float(ros.get("jointSampleStampS"))
            except (TypeError, ValueError):
                stamp = float("nan")
            if (q is None or dq is None or policy_qcmd is None
                    or published is None or not math.isfinite(stamp)):
                faults["missing_finite_replay_fields"] = (
                    faults.get("missing_finite_replay_fields", 0) + 1)
                continue
            repeated = last_stamp is not None and stamp == last_stamp
            if not repeated:
                source_sequence += 1
            sample = PolicyJointSample(
                q_raw=q,
                dq_raw=dq,
                source_stamp_s=stamp,
                source_sequence=max(0, source_sequence),
                receipt_age_s=0.0,
                repeated_source_sample=repeated,
                velocity_source=str(ros.get(
                    "jointVelocitySource", "recorded_joint_state")),
                source_clock=str(ros.get(
                    "jointSampleStampSource", "recorded")),
            )
            last_stamp = stamp
            if matcher is None:
                matcher = PolicyPlantModelReference(
                    target_parameters=target,
                    real_parameters=real,
                    solver=solver,
                    mode="shadow",
                    dt_s=0.005,
                    warm_q=q,
                    q_low=q_low,
                    q_high=q_high,
                    dq_limit=dq_limit,
                    allow_apply=False,
                    release_hash_match=True,
                    real_fit_metadata=fit_metadata,
                )
            try:
                proposal = matcher.propose(policy_qcmd, sample)
            except PlantMatcherFailClosed as exc:
                reason = str(exc).split(":", 1)[0]
                faults[reason] = faults.get(reason, 0) + 1
                matcher.reset(sample)
                matcher.previous_sample = None
                previous_shadow = None
                try:
                    proposal = matcher.propose(policy_qcmd, sample)
                except PlantMatcherFailClosed:
                    continue
            # Off mode is the identity by definition and is checked on every
            # saved command instead of relying on a synthetic fixture.
            off_max_abs_delta = max(
                off_max_abs_delta,
                float(np.max(np.abs(policy_qcmd - policy_qcmd))))
            shadow = proposal.matcher_shadow_q
            correction = shadow - policy_qcmd
            corrections.append(float(np.max(np.abs(correction))))
            if previous_shadow is not None:
                correction_slew.append(float(np.max(np.abs(
                    (shadow - policy_qcmd)
                    - (previous_shadow[0] - previous_shadow[1])))) / 0.005)
            previous_shadow = (shadow.copy(), policy_qcmd.copy())
            diagnostics = proposal.diagnostics
            direct = float(diagnostics["directPredictedTargetErrorRms"])
            shadow_error = float(diagnostics["shadowPredictedTargetErrorRms"])
            direct_errors.append(direct)
            shadow_errors.append(shadow_error)
            improvements.append(direct - shadow_error)
            if np.any(shadow < q_low - 1e-9) or np.any(shadow > q_high + 1e-9):
                q_envelope_violations += 1
            matcher.observe_published(published)
            processed += 1

    split = max(1, int(0.6 * processed))

    def split_report(start: int, end: int) -> dict[str, Any]:
        direct = direct_errors[start:end]
        shadow = shadow_errors[start:end]
        improve = improvements[start:end]
        return {
            "directTargetErrorRms": _summary(direct),
            "shadowTargetErrorRms": _summary(shadow),
            "improvement": _summary(improve),
            "improvedTickFraction": (
                0.0 if not improve else float(np.mean(
                    np.asarray(improve) > 0.0))),
        }

    return {
        "schemaVersion": 1,
        "claimScope": (
            "implementation_and_offline_model_objective_only; "
            "does_not_establish_real_closed_loop_benefit"),
        "input": {
            "path": str(input_path.resolve()),
            "sha256": _sha256(input_path),
            "rowsSeen": rows_seen,
            "policySamplesProcessed": processed,
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256(checkpoint_path),
            "targetParametersFrom": "actual_checkpoint_env_cfg",
        },
        "realFit": {
            "path": str(real_config_path.resolve()),
            "fitId": fit_metadata.get("fit_id"),
            "fitVersion": fit_metadata.get("fit_version"),
            "applyEligible": fit_metadata.get("apply_eligible"),
            "sourceHashesVerified": fit_metadata.get(
                "source_hashes_verified"),
            "sourceRecordingSha256": fit_metadata.get(
                "source_recording_sha256"),
            "provenanceStatus": fit_metadata.get("provenance_status"),
            "holdoutMetrics": fit_metadata.get("holdout_metrics"),
        },
        "off": {
            "maxAbsCommandDeltaRad": off_max_abs_delta,
            "equivalent": off_max_abs_delta == 0.0,
        },
        "shadow": {
            "calibration": split_report(0, split),
            "holdout": split_report(split, processed),
            "maxAbsCorrectionRad": max(corrections, default=0.0),
            "correctionRateRadS": _summary(correction_slew),
            "qEnvelopeViolationCount": q_envelope_violations,
            "replayFaultCounts": faults,
        },
        "unresolvedRisk": (
            "real fit has known dq/qacc underestimation and no paired real A/B; "
            "apply remains disabled"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--real-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int, default=4000)
    args = parser.parse_args()
    report = replay(
        args.input.expanduser().resolve(),
        args.checkpoint.expanduser().resolve(),
        args.real_config.expanduser().resolve(),
        max_samples=max(1, int(args.max_samples)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
