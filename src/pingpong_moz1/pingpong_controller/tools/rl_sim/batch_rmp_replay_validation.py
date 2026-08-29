#!/usr/bin/env python3
"""Batch replay all real recordings through the MJX/RMP chain."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "recovered_rmp").is_dir():
            return parent
    raise RuntimeError("cannot locate workspace containing recovered_rmp")


WORKSPACE = _workspace_root()
REPLAY_SCRIPT = (
    WORKSPACE
    / "simcode5/pingpong_moz1-main/pingpong_controller/tools/rl_sim/replay_rmp_recording_mjx.py"
)
DEFAULT_RECORD_ROOT = (
    WORKSPACE / "src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings"
)


@dataclass(frozen=True)
class RecordingInfo:
    group: str
    path: Path
    recording_name: str


def _recording_info(record_root: Path, path: Path) -> RecordingInfo:
    rel = path.relative_to(record_root)
    if len(rel.parts) >= 2 and rel.parts[0].startswith("record_new"):
        group = rel.parts[0]
    else:
        group = "ungrouped"
    return RecordingInfo(group=group, path=path, recording_name=path.parent.name)


def _discover_recordings(record_root: Path, include_groups: set[str] | None) -> list[RecordingInfo]:
    infos: list[RecordingInfo] = []
    for path in sorted(record_root.rglob("control_samples.jsonl")):
        try:
            info = _recording_info(record_root, path)
        except ValueError:
            continue
        if include_groups and info.group not in include_groups:
            continue
        infos.append(info)
    return infos


def _load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trace_info(path: Path) -> dict[str, object]:
    sys.path.insert(0, str(WORKSPACE))
    from recovered_rmp.replay import load_control_recording  # noqa: WPS433

    trace = load_control_recording(path)
    mask = np.asarray(trace.policy_mask, dtype=bool)
    valid = np.flatnonzero(mask)
    if valid.size == 0:
        raise ValueError(f"no valid policy phase in {path}")
    start = int(valid[0])
    stop = int(valid[-1]) + 1
    return {
        "samples": int(trace.time.size),
        "policy_samples": int(valid.size),
        "valid_start_index": start,
        "valid_stop_index": stop,
        "valid_start_s": float(trace.time[start]),
        "valid_stop_s": float(trace.time[stop - 1]),
        "valid_duration_s": float(trace.time[stop - 1] - trace.time[start]) if valid.size > 1 else 0.0,
        "median_dt": float(trace.median_dt),
    }


def _csv_items(values: str) -> str:
    return ",".join(item for item in values.split(",") if item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--calibration-profile", default="recovered_yaml_pd_fit")
    parser.add_argument("--controller-kp-scale-per-joint", type=str, required=True)
    parser.add_argument("--controller-kd-scale-per-joint", type=str, required=True)
    parser.add_argument("--rmp-jnt-kp-scale-per-joint", type=str, required=True)
    parser.add_argument("--rmp-jnt-kd-scale-per-joint", type=str, required=True)
    parser.add_argument("--dof-damping-scale-per-joint", type=str, required=True)
    parser.add_argument("--dof-armature-scale-per-joint", type=str, required=True)
    parser.add_argument("--rmp-output-delay-steps-per-joint", type=str, required=True)
    parser.add_argument("--include-group", action="append", default=[])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-format", choices=("json", "markdown"), default="json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record_root = args.record_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    include_groups = set(args.include_group) if args.include_group else None
    recordings = _discover_recordings(record_root, include_groups)
    if not recordings:
        raise SystemExit(f"no recordings found under {record_root}")

    def run_one(info: RecordingInfo) -> dict[str, object]:
        output_dir = output_root / info.group / info.recording_name
        report_path = output_dir / "rmp_mjx_replay_report.json"
        trace_path = output_dir / "rmp_mjx_replay_trace.npz"
        row: dict[str, object] = {
            "group": info.group,
            "recording": str(info.path),
            "recording_name": info.recording_name,
            "status": "pending",
            "reason": None,
            "trace_dir": str(output_dir),
            "report_path": str(report_path),
            "trace_path": str(trace_path),
        }
        try:
            trace_info = _load_trace_info(info.path)
        except Exception as exc:  # noqa: BLE001
            row.update({"status": "skipped", "reason": str(exc)})
            return row

        row.update(trace_info)
        if report_path.exists() and trace_path.exists() and not args.rerun:
            report = _load_report(report_path)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(REPLAY_SCRIPT),
                str(info.path),
                "--model",
                str(args.model),
                "--output",
                str(output_dir),
                "--calibration-profile",
                args.calibration_profile,
                "--controller-kp-scale-per-joint",
                args.controller_kp_scale_per_joint,
                "--controller-kd-scale-per-joint",
                args.controller_kd_scale_per_joint,
                "--rmp-jnt-kp-scale-per-joint",
                args.rmp_jnt_kp_scale_per_joint,
                "--rmp-jnt-kd-scale-per-joint",
                args.rmp_jnt_kd_scale_per_joint,
                "--dof-damping-scale-per-joint",
                args.dof_damping_scale_per_joint,
                "--dof-armature-scale-per-joint",
                args.dof_armature_scale_per_joint,
                "--rmp-output-delay-steps-per-joint",
                args.rmp_output_delay_steps_per_joint,
            ]
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:  # noqa: BLE001
                row.update({"status": "failed", "reason": f"replay exited {exc.returncode}"})
                return row
            report = _load_report(report_path)

        row.update(
            {
                "status": "pass",
                "reason": None,
                "valid_phase_only": bool(report.get("valid_phase_only", False)),
                "position_rms_rad": float(report["metrics"]["position_rms_rad"]),
                "velocity_rms_rad_s": float(report["metrics"]["velocity_rms_rad_s"]),
                "acceleration_rms_rad_s2": float(report["metrics"]["acceleration_rms_rad_s2"]),
                "maximum_position_error_rad": float(report["metrics"]["maximum_position_error_rad"]),
                "racket_metrics": report.get("racket_metrics", {}),
            }
        )
        return row

    if args.workers <= 1:
        rows = [run_one(info) for info in recordings]
    else:
        with cf.ThreadPoolExecutor(max_workers=args.workers) as executor:
            rows = list(executor.map(run_one, recordings))

    status_counts = {"pass": 0, "skipped": 0, "failed": 0, "pending": 0}
    for row in rows:
        status = row.get("status", "pending")
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "count": len(rows),
        "status_counts": status_counts,
        "records": rows,
        "groups": {},
    }
    for group_name in sorted({row["group"] for row in rows}):
        group_rows = [row for row in rows if row["group"] == group_name]
        pass_rows = [row for row in group_rows if row.get("status") == "pass"]
        summary["groups"][group_name] = {
            "count": len(group_rows),
            "pass_count": len(pass_rows),
            "skip_count": sum(1 for row in group_rows if row.get("status") == "skipped"),
            "failed_count": sum(1 for row in group_rows if row.get("status") == "failed"),
            "position_rms_rad_mean": float(np.mean([row["position_rms_rad"] for row in pass_rows])) if pass_rows else None,
            "velocity_rms_rad_s_mean": float(np.mean([row["velocity_rms_rad_s"] for row in pass_rows])) if pass_rows else None,
            "acceleration_rms_rad_s2_mean": float(np.mean([row["acceleration_rms_rad_s2"] for row in pass_rows])) if pass_rows else None,
            "max_position_error_rad": float(max((row.get("maximum_position_error_rad", 0.0) for row in group_rows), default=0.0)),
        }

    (output_root / "batch_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "batch_summary.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
