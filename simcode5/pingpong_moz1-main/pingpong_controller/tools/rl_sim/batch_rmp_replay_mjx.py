#!/usr/bin/env python3
"""Run record_new3/4/5 recovered-RMP MJX replays as vectorized env lanes."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from mjx_juggle_env import MjxJuggleEnv
from rmp_mjx_replay_batch_core import (
    add_replay_config_args,
    build_replay_batch,
    initialize_batch_state,
    make_record_artifacts,
    prepare_recording,
    record_summary_row,
    render_markdown_report,
    resolve_replay_config,
    run_batched_replay,
    summarize_records,
    write_record_artifacts,
)


def _workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "recovered_rmp").is_dir():
            return parent
    raise RuntimeError("cannot locate workspace containing recovered_rmp")


WORKSPACE = _workspace_root()
DEFAULT_RECORD_ROOT = WORKSPACE / "src" / "pingpong_moz1" / "pingpong_controller" / "outputs" / "interactive_recordings"


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Replay record_new3/4/5 through recovered RMP and MJX in vectorized batches. "
            "Each MJX environment lane replays one real recording."
        )
    )
    parser.add_argument("--model", type=Path, default=here / "moz1_pd.xml")
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument(
        "--include-group",
        action="append",
        dest="include_groups",
        default=None,
        help="Recording group under --record-root. Defaults to record_new3/4/5.",
    )
    parser.add_argument("--output-root", type=Path, default=Path("rmp_mjx_batch_replay_output"))
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-report", type=Path, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Vectorized lanes per wave. Default 0 means all loaded recordings in one wave.",
    )
    parser.add_argument(
        "--limit-recordings",
        type=int,
        default=0,
        help="Optional smoke-test limit after deterministic discovery sorting. Default 0 means no limit.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Recompute per-record artifacts even when existing reports are present.",
    )
    add_replay_config_args(parser)
    return parser.parse_args()


def discover_recordings(record_root: Path, include_groups: Sequence[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    root = record_root.expanduser().resolve()
    for group in include_groups:
        group_dir = root / group
        if not group_dir.exists():
            continue
        if (group_dir / "control_samples.jsonl").exists():
            result.append((group, group_dir))
            continue
        for path in sorted(group_dir.glob("probe_recording_*")):
            if path.is_dir() and (path / "control_samples.jsonl").exists():
                result.append((group, path))
    return sorted(result, key=lambda item: (item[0], str(item[1])))


def _existing_row(group: str, recording_path: Path, output_dir: Path) -> dict[str, object] | None:
    report_path = output_dir / "rmp_mjx_replay_report.json"
    trace_path = output_dir / "rmp_mjx_replay_trace.npz"
    if not report_path.exists() or not trace_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    report.setdefault("group", group)
    report.setdefault("recording_name", recording_path.name)
    return record_summary_row(
        report,
        trace_path=trace_path,
        report_path=report_path,
        output_dir=output_dir,
        status="pass",
    )


def _chunks(items: Sequence[object], size: int) -> list[Sequence[object]]:
    if size <= 0 or size >= len(items):
        return [items]
    return [items[i : i + size] for i in range(0, len(items), size)]


def _write_summary(path: Path, summary: dict[str, object], markdown_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    jsonl_path = path.with_suffix(".jsonl")
    with jsonl_path.open("w", encoding="utf-8") as stream:
        for row in summary.get("records", []):
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown_report(summary), encoding="utf-8")


def main() -> None:
    args = parse_args()
    start_wall = time.perf_counter()
    include_groups = args.include_groups or ["record_new3", "record_new4", "record_new5"]
    output_root = args.output_root.expanduser().resolve()
    output_json = (args.output_json or (output_root / "batch_summary.json")).expanduser().resolve()
    output_report = (args.output_report or (output_root / "batch_summary.md")).expanduser().resolve()
    record_root = args.record_root.expanduser().resolve()
    discovered = discover_recordings(record_root, include_groups)
    if int(args.limit_recordings) > 0:
        discovered = discovered[: int(args.limit_recordings)]
    if not discovered:
        raise SystemExit(f"no control_samples.jsonl recordings found under {record_root}")

    config = resolve_replay_config(args)
    all_rows: list[dict[str, object]] = []
    pending: list[tuple[str, Path]] = []
    for group, recording_path in discovered:
        out_dir = output_root / group / recording_path.name
        if not args.rerun:
            row = _existing_row(group, recording_path, out_dir)
            if row is not None:
                all_rows.append(row)
                continue
        pending.append((group, recording_path))

    prepared = []
    for group, recording_path in pending:
        try:
            prepared.append(
                prepare_recording(
                    recording_path,
                    group=group,
                    segment=args.segment,
                    valid_phase_only=bool(args.valid_phase_only),
                    max_samples=int(args.max_samples),
                )
            )
        except Exception as exc:  # noqa: BLE001 - preserve per-record failure in report.
            all_rows.append({
                "group": group,
                "recording_name": recording_path.name,
                "recording": str(recording_path),
                "status": "failed",
                "reason": f"preflight: {exc}",
                "batch_index": None,
                "environment_index": None,
                "active_samples": 0,
                "padded_batch_samples": 0,
                "trace_dir": str(output_root / group / recording_path.name),
                "trace_path": "",
                "report_path": "",
                "position_rms_rad": None,
                "velocity_rms_rad_s": None,
                "acceleration_rms_rad_s2": None,
                "rmp_position_rms_rad": None,
                "maximum_position_error_rad": None,
                "racket_metrics": {"available": False, "keys": []},
            })

    batch_size = int(args.batch_size) if int(args.batch_size) > 0 else len(prepared)
    waves = _chunks(prepared, batch_size) if prepared else []
    device_seconds = 0.0
    max_padded_samples = 0
    total_active_samples = 0
    env_dt = None
    completed_waves = 0
    for batch_index, wave in enumerate(waves):
        if not wave:
            continue
        batch = build_replay_batch(wave)
        max_padded_samples = max(max_padded_samples, int(batch.targets.shape[0]))
        total_active_samples += int(batch.lengths.sum())
        env = MjxJuggleEnv(args.model, n_envs=len(wave), cfg=config.env_cfg)
        env_dt = float(env.dt)
        try:
            state = initialize_batch_state(env, batch, seed=int(args.seed) + batch_index)
            result = run_batched_replay(env, state, batch)
            device_seconds += float(result.device_seconds)
        except Exception as exc:  # noqa: BLE001
            for lane, record in enumerate(wave):
                all_rows.append({
                    "group": record.group,
                    "recording_name": record.recording_name,
                    "recording": str(record.recording_path),
                    "status": "failed",
                    "reason": f"batch execution: {exc}",
                    "batch_index": batch_index,
                    "environment_index": lane,
                    "active_samples": int(record.sample_count),
                    "padded_batch_samples": int(batch.targets.shape[0]),
                    "trace_dir": str(output_root / record.group / record.recording_name),
                    "trace_path": "",
                    "report_path": "",
                    "position_rms_rad": None,
                    "velocity_rms_rad_s": None,
                    "acceleration_rms_rad_s2": None,
                    "rmp_position_rms_rad": None,
                    "maximum_position_error_rad": None,
                    "racket_metrics": {"available": False, "keys": []},
                })
            continue
        completed_waves += 1
        for lane, record in enumerate(wave):
            out_dir = output_root / record.group / record.recording_name
            try:
                arrays, report = make_record_artifacts(
                    record,
                    result,
                    lane,
                    env_dt,
                    config,
                    batch_index=batch_index,
                    padded_batch_samples=int(batch.targets.shape[0]),
                )
                trace_path, report_path = write_record_artifacts(out_dir, arrays, report)
                all_rows.append(record_summary_row(
                    report,
                    trace_path=trace_path,
                    report_path=report_path,
                    output_dir=out_dir,
                    status="pass",
                ))
            except Exception as exc:  # noqa: BLE001
                all_rows.append({
                    "group": record.group,
                    "recording_name": record.recording_name,
                    "recording": str(record.recording_path),
                    "status": "failed",
                    "reason": f"postprocess: {exc}",
                    "batch_index": batch_index,
                    "environment_index": lane,
                    "active_samples": int(record.sample_count),
                    "padded_batch_samples": int(batch.targets.shape[0]),
                    "trace_dir": str(out_dir),
                    "trace_path": "",
                    "report_path": "",
                    "position_rms_rad": None,
                    "velocity_rms_rad_s": None,
                    "acceleration_rms_rad_s2": None,
                    "rmp_position_rms_rad": None,
                    "maximum_position_error_rad": None,
                    "racket_metrics": {"available": False, "keys": []},
                })

    all_rows.sort(key=lambda row: (str(row.get("group", "")), str(row.get("recording_name", ""))))
    wall_seconds = time.perf_counter() - start_wall
    summary = summarize_records(
        model=args.model.expanduser().resolve(),
        record_root=record_root,
        include_groups=include_groups,
        calibration_profile=args.calibration_profile,
        output_root=output_root,
        records=all_rows,
        discovered_count=len(discovered),
        loaded_count=len(prepared),
        batch_size_requested=batch_size,
        vectorized_batch_count=completed_waves,
        max_padded_samples=max_padded_samples,
        total_active_samples=total_active_samples,
        device_seconds=device_seconds,
        wall_seconds=wall_seconds,
        env_dt=env_dt,
    )
    _write_summary(output_json, summary, output_report)
    print(json.dumps({
        "summary": str(output_json),
        "report": str(output_report),
        "status_counts": summary["status_counts"],
        "vectorized_batch_count": completed_waves,
    }, indent=2, ensure_ascii=False))
    if summary["status_counts"].get("failed", 0):
        sys.exit(1)


if __name__ == "__main__":
    main()
