#!/usr/bin/env python3
"""Scan all real recordings and summarize valid juggling phases."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def _workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "recovered_rmp").is_dir():
            return parent
    raise RuntimeError("cannot locate workspace containing recovered_rmp")


WORKSPACE = _workspace_root()
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from recovered_rmp.replay import load_control_recording  # noqa: E402

DEFAULT_RECORD_ROOT = (
    WORKSPACE / "src/pingpong_moz1/pingpong_controller/outputs/interactive_recordings"
)


def _group_for(path: Path, record_root: Path) -> str:
    rel = path.relative_to(record_root)
    if len(rel.parts) >= 2 and rel.parts[0].startswith("record_new"):
        return rel.parts[0]
    return "ungrouped"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record_root = args.record_root.expanduser().resolve()
    records: list[dict[str, object]] = []
    skip_reasons = Counter()
    for path in sorted(record_root.rglob("control_samples.jsonl")):
        group = _group_for(path, record_root)
        try:
            trace = load_control_recording(path)
            mask = np.asarray(trace.policy_mask, dtype=bool)
            valid = np.flatnonzero(mask)
            if valid.size == 0:
                raise ValueError("no valid policy phase")
            start = int(valid[0])
            stop = int(valid[-1]) + 1
            record = {
                "group": group,
                "recording": str(path),
                "recording_name": path.parent.name,
                "status": "valid",
                "samples": int(trace.time.size),
                "policy_samples": int(valid.size),
                "valid_start_index": start,
                "valid_stop_index": stop,
                "valid_start_s": float(trace.time[start]),
                "valid_stop_s": float(trace.time[stop - 1]),
                "valid_duration_s": float(trace.time[stop - 1] - trace.time[start]) if valid.size > 1 else 0.0,
                "median_dt": float(trace.median_dt),
                "skip_reason": None,
            }
        except Exception as exc:  # noqa: BLE001
            skip_reasons[str(exc)] += 1
            record = {
                "group": group,
                "recording": str(path),
                "recording_name": path.parent.name,
                "status": "skipped",
                "samples": None,
                "policy_samples": None,
                "valid_start_index": None,
                "valid_stop_index": None,
                "valid_start_s": None,
                "valid_stop_s": None,
                "valid_duration_s": None,
                "median_dt": None,
                "skip_reason": str(exc),
            }
        records.append(record)

    groups: dict[str, dict[str, object]] = {}
    for group in sorted({str(row["group"]) for row in records}):
        group_rows = [row for row in records if row["group"] == group]
        valid_rows = [row for row in group_rows if row["status"] == "valid"]
        groups[group] = {
            "count": len(group_rows),
            "valid_count": len(valid_rows),
            "skipped_count": len(group_rows) - len(valid_rows),
            "mean_policy_samples": float(np.mean([row["policy_samples"] for row in valid_rows])) if valid_rows else None,
            "mean_valid_duration_s": float(np.mean([row["valid_duration_s"] for row in valid_rows])) if valid_rows else None,
            "median_dt_mean": float(np.mean([row["median_dt"] for row in valid_rows])) if valid_rows else None,
        }

    summary = {
        "record_root": str(record_root),
        "total_files": len(records),
        "valid_files": sum(1 for row in records if row["status"] == "valid"),
        "skipped_files": sum(1 for row in records if row["status"] == "skipped"),
        "skip_reasons": skip_reasons.most_common(),
        "groups": groups,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
