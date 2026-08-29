#!/usr/bin/env python3
"""Merge the real-recording catalog with whatever replay reports already exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(row: dict, key: str) -> float | None:
    value = row.get(key)
    return float(value) if isinstance(value, (int, float)) and np.isfinite(value) else None


def main() -> None:
    args = parse_args()
    catalog = _load_json(args.catalog)
    replay_root = args.replay_root.expanduser().resolve()
    rows: list[dict[str, object]] = []
    for item in catalog["records"]:
        row = {
            "group": item["group"],
            "recording": item["recording"],
            "recording_name": item["recording_name"],
            "status": "skipped" if item["status"] == "skipped" else "pending",
            "reason": item.get("skip_reason"),
            "samples": item.get("samples"),
            "policy_samples": item.get("policy_samples"),
            "valid_start_index": item.get("valid_start_index"),
            "valid_stop_index": item.get("valid_stop_index"),
            "valid_duration_s": item.get("valid_duration_s"),
            "trace_dir": str(replay_root / item["group"] / item["recording_name"]),
            "report_path": str(replay_root / item["group"] / item["recording_name"] / "rmp_mjx_replay_report.json"),
            "trace_path": str(replay_root / item["group"] / item["recording_name"] / "rmp_mjx_replay_trace.npz"),
        }
        report_path = Path(row["report_path"])
        if item["status"] == "valid" and report_path.exists():
            report = _load_json(report_path)
            metrics = report["metrics"]
            row.update(
                {
                    "status": "pass",
                    "reason": None,
                    "policy_samples": metrics.get("policy_samples"),
                    "acceleration_samples": metrics.get("acceleration_samples"),
                    "position_rms_rad": metrics.get("position_rms_rad"),
                    "velocity_rms_rad_s": metrics.get("velocity_rms_rad_s"),
                    "acceleration_rms_rad_s2": metrics.get("acceleration_rms_rad_s2"),
                    "rmp_position_rms_rad": metrics.get("rmp_position_rms_rad"),
                    "maximum_position_error_rad": metrics.get("maximum_position_error_rad"),
                    "valid_phase_only": report.get("valid_phase_only", True),
                }
            )
        rows.append(row)

    summary = {
        "model": str(args.model),
        "record_root": catalog["record_root"],
        "replay_root": str(replay_root),
        "calibration_profile": "recovered_yaml_pd_fit",
        "records": rows,
        "status_counts": {
            status: sum(1 for row in rows if row["status"] == status)
            for status in ("pass", "pending", "skipped")
        },
        "groups": {},
    }
    for group in sorted({str(row["group"]) for row in rows}):
        all_rows = [row for row in rows if row["group"] == group]
        pass_rows = [row for row in all_rows if row["status"] == "pass"]
        summary["groups"][group] = {
            "count": len(all_rows),
            "pass_count": len(pass_rows),
            "skip_count": len([row for row in all_rows if row["status"] == "skipped"]),
            "position_rms_rad_mean": float(np.mean([_metric(row, "position_rms_rad") for row in pass_rows])) if pass_rows else None,
            "velocity_rms_rad_s_mean": float(np.mean([_metric(row, "velocity_rms_rad_s") for row in pass_rows])) if pass_rows else None,
            "acceleration_rms_rad_s2_mean": float(np.mean([_metric(row, "acceleration_rms_rad_s2") for row in pass_rows])) if pass_rows else None,
            "max_position_error_rad": float(max((_metric(row, "maximum_position_error_rad") or 0.0 for row in pass_rows), default=0.0)),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[merge] wrote: {args.output}")
    print(json.dumps(summary["status_counts"], indent=2))


if __name__ == "__main__":
    main()
