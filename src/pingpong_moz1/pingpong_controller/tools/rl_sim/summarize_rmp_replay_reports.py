#!/usr/bin/env python3
"""Summarize RMP replay reports under a directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np


def _format_float(v: float | None, digits: int = 6) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _write_markdown(summary: dict, output: Path) -> None:
    lines = []
    lines.append(f"# RMP replay summary")
    lines.append("")
    lines.append(f"- Count: {summary.get('count', 0)}")
    lines.append(f"- Mean q RMS: {_format_float(summary.get('mean_q'))}")
    lines.append(f"- Mean dq RMS: {_format_float(summary.get('mean_dq'))}")
    lines.append(f"- Mean qdd RMS: {_format_float(summary.get('mean_qdd'))}")
    lines.append("")
    lines.append("## Records")
    lines.append("")
    lines.append("| group | recording | q | dq | qdd | samples | racket_x | racket_y | racket_z | racket_vx | racket_vy | racket_vz | racket_ax | racket_ay | racket_az | racket_omx | racket_omy | racket_omz | racket_alx | racket_aly | racket_alz |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary.get("records", []):
        racket = row.get("racket_metrics") or {}
        pos = racket.get("center_pos_m") or [None, None, None]
        vel = racket.get("center_vel_m_s") or [None, None, None]
        acc = racket.get("center_acc_m_s2") or [None, None, None]
        omega = racket.get("center_angular_velocity_world_rad_s") or [None, None, None]
        alpha = racket.get("center_angular_acceleration_world_rad_s2") or [None, None, None]
        lines.append(
            f"| {row.get('group','')} | {row.get('recording','')} | {_format_float(row.get('q'))} | {_format_float(row.get('dq'))} | {_format_float(row.get('qdd'))} | {row.get('samples','')} | {_format_float(pos[0])} | {_format_float(pos[1])} | {_format_float(pos[2])} | {_format_float(vel[0])} | {_format_float(vel[1])} | {_format_float(vel[2])} | {_format_float(acc[0])} | {_format_float(acc[1])} | {_format_float(acc[2])} | {_format_float(omega[0])} | {_format_float(omega[1])} | {_format_float(omega[2])} | {_format_float(alpha[0])} | {_format_float(alpha[1])} | {_format_float(alpha[2])} |"
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = []
    for path in sorted(args.root.rglob("rmp_mjx_replay_report.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics = report["metrics"]
        rel = path.relative_to(args.root)
        group = rel.parts[0] if len(rel.parts) > 2 else "unknown"
        recording = rel.parts[1] if len(rel.parts) > 2 else rel.parent.name
        reports.append(
            {
                "group": group,
                "recording": recording,
                "path": str(path),
                "q": float(metrics["position_rms_rad"]),
                "dq": float(metrics["velocity_rms_rad_s"]),
                "qdd": float(metrics["acceleration_rms_rad_s2"]),
                "max_q": float(metrics["maximum_position_error_rad"]),
                "q_per_joint": metrics["position_rms_per_joint_rad"],
                "dq_per_joint": metrics["velocity_rms_per_joint_rad_s"],
                "qdd_per_joint": metrics["acceleration_rms_per_joint_rad_s2"],
                "samples": int(metrics["policy_samples"]),
                "acc_samples": int(metrics["acceleration_samples"]),
                "valid_phase_only": bool(report.get("valid_phase_only", False)),
            }
        )
    if reports:
        summary = {
            "count": len(reports),
            "mean_q": float(np.mean([r["q"] for r in reports])),
            "mean_dq": float(np.mean([r["dq"] for r in reports])),
            "mean_qdd": float(np.mean([r["qdd"] for r in reports])),
            "worst_q": max(reports, key=lambda r: r["q"]),
            "worst_dq": max(reports, key=lambda r: r["dq"]),
            "worst_qdd": max(reports, key=lambda r: r["qdd"]),
            "mean_q_per_joint": np.mean([r["q_per_joint"] for r in reports], axis=0).tolist(),
            "mean_dq_per_joint": np.mean([r["dq_per_joint"] for r in reports], axis=0).tolist(),
            "mean_qdd_per_joint": np.mean([r["qdd_per_joint"] for r in reports], axis=0).tolist(),
            "records": reports,
        }
    else:
        summary = {"count": 0, "records": []}
    text = json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    markdown_output = args.markdown_output
    if markdown_output:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(summary, markdown_output)
        print(f"[summary] wrote markdown: {markdown_output}")
    print(text)


if __name__ == "__main__":
    main()
