#!/usr/bin/env python3
"""Compare source-sequence-aligned real/sim racket orientation for 2026-08-10..13."""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RECORD_ROOT = ROOT / "src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings"
RUN_ROOTS = {
    "2026-08-10": RECORD_ROOT / "record_new3",
    "2026-08-11": RECORD_ROOT / "record_new4",
    "2026-08-12": RECORD_ROOT / "record_new5",
    "2026-08-13": RECORD_ROOT / "record_new5",
}
CHECKPOINT = ROOT / (
    "simcode5/pingpong_moz1-main/pingpong_controller/outputs/rl_sim/"
    "goal_d455_gpu1_pure_stable_v5_recovery_20260802/mjx_curriculum_best.pkl"
)
XML = ROOT / "simcode5/pingpong_moz1-main/pingpong_controller/tools/rl_sim/moz1_pd.xml"
OUT = ROOT / "analysis_results/racket_pose_sim2real_20260810_13"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def vec(value: Any, n: int) -> np.ndarray | None:
    try:
        out = np.asarray(value, dtype=np.float64).reshape(-1)[:n]
    except (TypeError, ValueError):
        return None
    return out if out.size == n and np.all(np.isfinite(out)) else None


def sim_seq(row: dict[str, Any]) -> int | None:
    try:
        return int((row.get("sync") or {}).get("sourceSeq"))
    except (TypeError, ValueError):
        return None


def real_seq(row: dict[str, Any]) -> int | None:
    try:
        return int(row.get("sourceSeq", row.get("tick")))
    except (TypeError, ValueError):
        return None


def inputs_match(real: dict[str, Any], sim: dict[str, Any]) -> bool:
    """Require the recorded real/sim policy input envelope to be identical."""
    policy_input = real.get("policyInput")
    sync = sim.get("sync")
    if not isinstance(policy_input, dict) or not isinstance(sync, dict):
        return False
    try:
        runtime = real.get("runtime") or {}
        real_profile = str(runtime.get(
            "activePolicyProfile", runtime.get("policyProfile", "")))
        sim_profile = str(sync.get("policyProfile", sim.get("policyProfile", "")))
        same = (
            np.array_equal(np.asarray(policy_input["ball"], np.float32),
                           np.asarray(sync["inputBall"], np.float32))
            and np.array_equal(np.asarray(policy_input["velocity"], np.float32),
                               np.asarray(sync["inputVelocity"], np.float32))
            and bool(policy_input.get("valid")) == bool(sync.get("inputValid"))
            and int(policy_input.get("ballObservationSeq", 0) or 0)
            == int(sync.get("ballObservationSeq", 0) or 0)
            and real_profile == sim_profile
            and int(runtime.get("policyGeneration", 0) or 0)
            == int(sync.get("policyGeneration", 0) or 0)
        )
        real_ball = real.get("policyBallObservation")
        sim_ball = sim.get("policyBallObservation")
        if real_ball is not None or sim_ball is not None:
            same = same and np.array_equal(
                np.asarray(real_ball, np.float32), np.asarray(sim_ball, np.float32))
        return bool(same)
    except (KeyError, TypeError, ValueError):
        return False


def mirror_contract_error(sim: dict[str, Any]) -> str | None:
    """Return a concrete mirror configuration/health error, if declared."""
    if sim.get("backend") != "mujoco_actuator_mirror":
        return "wrong_backend"
    if not bool(sim.get("available", True)):
        return "unavailable"
    if not bool(sim.get("active", True)):
        return "inactive"
    if not bool(sim.get("valid", False)):
        return "invalid_state"
    sync = sim.get("sync") or {}
    if sync.get("streamContinuous") is False:
        return "noncontinuous_stream"
    if sync.get("alignmentValid") is False:
        return "invalid_alignment"
    plant = sim.get("plantContract")
    if isinstance(plant, dict):
        declared = [plant.get("checkpointPlantSettingsAligned"),
                    plant.get("simcode5SettingsAligned")]
        if any(value is False for value in declared):
            return "plant_settings_misaligned"
        if not any(value is True for value in declared):
            return "plant_alignment_not_declared"
        if abs(float(plant.get("controlDtS", 0.005)) - 0.005) > 1e-9:
            return "wrong_control_dt"
        if str(plant.get("actuatorCmdModel", "second_order")) != "second_order":
            return "wrong_actuator_model"
    # 2026-08-10 is a legacy schema without plantContract. It is retained only
    # because the existing sourceSeq replay reports independently validated
    # that cohort; any row that explicitly declares a bad contract is excluded.
    return None


def quat_normal(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    # Third column of the quaternion rotation matrix.
    n = np.array([2 * (x * z + w * y),
                  2 * (y * z - w * x),
                  1 - 2 * (x * x + y * y)])
    return -n if n[2] < 0 else n


def quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return math.degrees(2.0 * math.acos(float(np.clip(abs(a @ b), 0.0, 1.0))))


def normal_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip(a @ b, -1.0, 1.0))))


def tilt_components_deg(n: np.ndarray) -> tuple[float, float, float]:
    # Signed normal lean toward base +x/+y, plus total departure from horizontal.
    tx = math.degrees(math.atan2(float(n[0]), float(n[2])))
    ty = math.degrees(math.atan2(float(n[1]), float(n[2])))
    total = math.degrees(math.acos(float(np.clip(n[2], -1.0, 1.0))))
    return tx, ty, total


def detect_contacts(rows: list[dict[str, Any]]) -> list[dict[str, float]]:
    """Detect real impacts by downward-to-upward reversal plus racket proximity."""
    refresh: list[dict[str, Any]] = []
    last_obs = None
    for row in rows:
        obs_seq = int(row.get("ballObservationSeq", 0) or 0)
        if obs_seq > 0 and obs_seq != last_obs:
            refresh.append(row)
            last_obs = obs_seq
    contacts: list[dict[str, float]] = []
    last_t = -1e30
    for i in range(1, len(refresh)):
        after = refresh[i]
        va = vec(after.get("measuredVelocity"), 3)
        if va is None or va[2] <= 0.10:
            continue
        recent = refresh[max(0, i - 5):i]
        candidates = [(r, vec(r.get("measuredVelocity"), 3)) for r in recent]
        candidates = [(r, v) for r, v in candidates if v is not None]
        if not candidates:
            continue
        before, vb = min(candidates, key=lambda rv: rv[1][2])
        if vb[2] >= -0.10:
            continue
        event_t = float(after.get("wallTimeS", 0.0) or 0.0)
        if event_t - last_t < 0.18:
            continue
        ball = vec(after.get("measuredBall"), 3)
        racket = vec(after.get("racket"), 3)
        if ball is None or racket is None:
            continue
        rel = ball - racket
        geometry_ok = np.linalg.norm(rel[:2]) <= 0.085 and -0.025 <= rel[2] <= 0.105
        bball = vec(before.get("measuredBall"), 3)
        bracket = vec(before.get("racket"), 3)
        held_ok = (bball is not None and bracket is not None
                   and event_t - float(before.get("wallTimeS", event_t)) <= 0.09
                   and np.linalg.norm(bball - bracket) <= 0.10)
        if not (geometry_ok or held_ok):
            continue
        near = [r for r in rows
                if event_t - 0.085 <= float(r.get("wallTimeS", 0.0)) <= event_t + 0.005]
        physical_t = event_t
        if near:
            def distance(r: dict[str, Any]) -> float:
                b, p = vec(r.get("measuredBall"), 3), vec(r.get("racket"), 3)
                return float(np.linalg.norm(b - p)) if b is not None and p is not None else 1e9
            physical_t = float(min(near, key=distance).get("wallTimeS", event_t))
        contacts.append({"detected_t": event_t, "physical_t": physical_t})
        last_t = event_t
    return contacts


def stats(values: Iterable[float]) -> dict[str, float | int]:
    a = np.asarray(list(values), dtype=np.float64)
    a = a[np.isfinite(a)]
    if not a.size:
        return {"n": 0, "mean": math.nan, "median": math.nan,
                "p95": math.nan, "rms": math.nan, "std": math.nan}
    return {"n": int(a.size), "mean": float(np.mean(a)),
            "median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
            "rms": float(np.sqrt(np.mean(a * a))), "std": float(np.std(a))}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    sys.path.insert(0, str(ROOT / "src/pingpong_moz1"))
    from pingpong_controller.mjx_rl_policy import Simcode5PureActuator67PolicyController

    fk = Simcode5PureActuator67PolicyController(
        str(CHECKPOINT), robot_xml_path=str(XML))
    OUT.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, Any]] = []
    contacts_out: list[dict[str, Any]] = []
    runs_out: list[dict[str, Any]] = []
    exclusions: Counter[str] = Counter()

    for day, base in RUN_ROOTS.items():
        stamp = day.replace("-", "")
        for run_dir in sorted(base.glob(f"probe_recording_{stamp}_*")):
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            expected_mirror_profile = str(
                (((metadata.get("config") or {}).get("simMirror") or {})
                 .get("policyProfile", "")))
            real_rows = [r for r in read_jsonl(run_dir / "control_samples.jsonl")
                         if bool(r.get("policyExecuted")) and r.get("mode") == "policy"]
            sim_by_seq: dict[int, dict[str, Any]] = {}
            for row in read_jsonl(run_dir / "sim_control_samples.jsonl"):
                seq = sim_seq(row)
                if seq is not None and seq not in sim_by_seq:
                    sim_by_seq[seq] = row
            aligned: list[dict[str, Any]] = []
            for real in real_rows:
                seq = real_seq(real)
                sim = sim_by_seq.get(seq) if seq is not None else None
                if sim is None:
                    exclusions["no_source_seq_pair"] += 1
                    continue
                contract_error = mirror_contract_error(sim)
                if contract_error is not None:
                    exclusions[contract_error] += 1
                    continue
                actual_mirror_profile = str(
                    (sim.get("sync") or {}).get(
                        "policyProfile", sim.get("policyProfile", "")))
                if (expected_mirror_profile
                        and actual_mirror_profile != expected_mirror_profile):
                    exclusions["metadata_mirror_profile_mismatch"] += 1
                    continue
                if not inputs_match(real, sim):
                    exclusions["policy_input_or_cohort_mismatch"] += 1
                    continue
                robs, sobs = vec(real.get("policyObs"), 67), vec(sim.get("policyObs"), 67)
                if robs is None or sobs is None:
                    continue
                _rp, rq, rok = fk._compute_racket_pose_quat_base(robs[:7])
                _sp, sq_fk, sok = fk._compute_racket_pose_quat_base(sobs[:7])
                sq_recorded = vec(sim.get("racketQuatXyzw"), 4)
                if not rok or not sok or sq_recorded is None:
                    continue
                rn, sn = quat_normal(np.asarray(rq)), quat_normal(np.asarray(sq_fk))
                rtx, rty, rt = tilt_components_deg(rn)
                stx, sty, st = tilt_components_deg(sn)
                item = {
                    "day": day, "run": run_dir.name, "source_seq": seq,
                    "policy_profile": actual_mirror_profile,
                    "wall_time_s": float(real.get("wallTimeS", 0.0)),
                    "full_orientation_gap_deg": quat_angle_deg(np.asarray(rq), np.asarray(sq_fk)),
                    "normal_gap_deg": normal_angle_deg(rn, sn),
                    "real_tilt_x_deg": rtx, "sim_tilt_x_deg": stx,
                    "tilt_x_real_minus_sim_deg": rtx - stx,
                    "real_tilt_y_deg": rty, "sim_tilt_y_deg": sty,
                    "tilt_y_real_minus_sim_deg": rty - sty,
                    "real_total_tilt_deg": rt, "sim_total_tilt_deg": st,
                    "total_tilt_real_minus_sim_deg": rt - st,
                    "recorded_sim_fk_quat_gap_deg": quat_angle_deg(sq_recorded, np.asarray(sq_fk)),
                    "near_contact": False,
                }
                aligned.append(item)
                samples.append(item)

            contacts = detect_contacts(real_rows)
            matched_contacts = 0
            for index, contact in enumerate(contacts, start=1):
                if not aligned:
                    continue
                nearest = min(aligned, key=lambda r: abs(r["wall_time_s"] - contact["physical_t"]))
                dt = nearest["wall_time_s"] - contact["physical_t"]
                if abs(dt) > 0.035:
                    continue
                matched_contacts += 1
                for item in aligned:
                    if abs(item["wall_time_s"] - contact["physical_t"]) <= 0.030:
                        item["near_contact"] = True
                contacts_out.append({
                    "day": day, "run": run_dir.name, "contact_index": index,
                    "aligned_dt_ms": 1000.0 * dt,
                    **{k: nearest[k] for k in nearest if k not in {
                        "day", "run", "source_seq", "wall_time_s", "near_contact"}},
                })
            runs_out.append({
                "day": day, "run": run_dir.name,
                "real_policy_samples": len(real_rows), "sim_rows": len(sim_by_seq),
                "aligned_samples": len(aligned), "detected_contacts": len(contacts),
                "aligned_contacts": matched_contacts,
                "median_normal_gap_deg": stats(r["normal_gap_deg"] for r in aligned)["median"],
                "p95_normal_gap_deg": stats(r["normal_gap_deg"] for r in aligned)["p95"],
            })

    day_summary: dict[str, Any] = {}
    for day in RUN_ROOTS:
        ds = [r for r in samples if r["day"] == day]
        cs = [r for r in contacts_out if r["day"] == day]
        day_summary[day] = {
            "runs": sum(1 for r in runs_out if r["day"] == day),
            "runs_with_aligned_contacts": sum(1 for r in runs_out
                                              if r["day"] == day and r["aligned_contacts"] > 0),
            "aligned_samples": len(ds), "aligned_contacts": len(cs),
            "all": {key: stats(r[key] for r in ds) for key in (
                "full_orientation_gap_deg", "normal_gap_deg",
                "tilt_x_real_minus_sim_deg", "tilt_y_real_minus_sim_deg",
                "total_tilt_real_minus_sim_deg", "real_total_tilt_deg",
                "sim_total_tilt_deg", "recorded_sim_fk_quat_gap_deg")},
            "contact": {key: stats(r[key] for r in cs) for key in (
                "full_orientation_gap_deg", "normal_gap_deg",
                "tilt_x_real_minus_sim_deg", "tilt_y_real_minus_sim_deg",
                "total_tilt_real_minus_sim_deg", "real_total_tilt_deg",
                "sim_total_tilt_deg")},
        }
    pooled_contact = [r for r in samples if r["near_contact"]]
    pooled = {
        "all": {key: stats(r[key] for r in samples) for key in (
            "full_orientation_gap_deg", "normal_gap_deg",
            "tilt_x_real_minus_sim_deg", "tilt_y_real_minus_sim_deg",
            "total_tilt_real_minus_sim_deg")},
        "near_contact_30ms": {key: stats(r[key] for r in pooled_contact) for key in (
            "full_orientation_gap_deg", "normal_gap_deg",
            "tilt_x_real_minus_sim_deg", "tilt_y_real_minus_sim_deg",
            "total_tilt_real_minus_sim_deg")},
        "contact_events": {key: stats(r[key] for r in contacts_out) for key in (
            "full_orientation_gap_deg", "normal_gap_deg",
            "tilt_x_real_minus_sim_deg", "tilt_y_real_minus_sim_deg",
            "total_tilt_real_minus_sim_deg")},
    }
    payload = {"method": {
        "alignment": "sourceSeq + exact policy input envelope",
        "realPose": "MuJoCo FK(policyObs[0:7] real feedback q)",
        "simPose": "same MuJoCo FK(policyObs[0:7] mirror q)",
        "contact": "measured vz downward-to-upward reversal + racket proximity",
        "mirrorExclusion": "explicit config/health errors + profile/generation/input mismatch",
        "legacyPlantContract": "2026-08-10 has no plantContract field; retained because prior sourceSeq replay validated that cohort",
    }, "excluded_samples": dict(exclusions), "days": day_summary, "pooled": pooled}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(OUT / "per_run.csv", runs_out)
    write_csv(OUT / "contacts.csv", contacts_out)
    write_csv(OUT / "aligned_samples.csv", samples)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
