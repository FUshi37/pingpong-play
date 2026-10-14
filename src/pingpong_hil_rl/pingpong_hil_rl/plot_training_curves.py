from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUNS = {
    "baseline": "20260731-112119",
    "weak_scaffold": "20260731-113751",
    "strong_scaffold": "20260731-114727",
    "low_height_curriculum": "20260731-115345",
}


def moving_mean(values: np.ndarray, window: int = 15) -> np.ndarray:
    if values.size < 2:
        return values
    window = min(window, values.size)
    kernel = np.ones(window, dtype=np.float64)
    numerator = np.convolve(values, kernel, mode="same")
    denominator = np.convolve(np.ones(values.size), kernel, mode="same")
    return numerator / denominator


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs/hil_flashsac"))
    parser.add_argument("--output", type=Path, default=Path("docs/hil_training_curves.png"))
    parser.add_argument("--summary-csv", type=Path, default=Path("docs/hil_training_curve_summary.csv"))
    args = parser.parse_args()

    runs: dict[str, list[dict]] = {}
    for label, run_name in DEFAULT_RUNS.items():
        metrics = args.runs_root / run_name / "metrics.jsonl"
        runs[label] = load_records(metrics)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=False)
    plots = (
        ("episode_return", "Episode return (moving mean)"),
        ("critic/loss", "Critic loss (moving mean)"),
        ("racket_up_cos", "Paddle up cosine (moving mean)"),
        ("ball_vz", "Logged ball vertical velocity (moving mean)"),
        ("episode_max_tracking_error_deg", "Episode max tracking error (deg)"),
        ("hits", "Confirmed hits"),
    )
    for axis, (key, title) in zip(axes.flat, plots):
        for label, records in runs.items():
            selected = [(float(r["step"]), float(r[key])) for r in records if key in r]
            if not selected:
                continue
            steps = np.asarray([item[0] for item in selected])
            values = np.asarray([item[1] for item in selected])
            smooth = values if key == "hits" else moving_mean(values)
            axis.plot(steps, smooth, linewidth=1.6, label=label)
        axis.set_title(title)
        axis.set_xlabel("Interaction step")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Single-environment ROS2 HIL FlashSAC trials", fontsize=15)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)

    fields = [
        "run",
        "bin_start",
        "bin_end",
        "samples",
        "episode_return_mean",
        "critic_loss_mean",
        "racket_up_cos_mean",
        "racket_up_cos_max",
        "ball_vz_max",
        "tracking_error_max_deg",
        "confirmed_hits_max",
    ]
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label, records in runs.items():
            final_step = int(records[-1]["step"])
            for start in range(0, final_step + 1, 1000):
                subset = [r for r in records if start < int(r["step"]) <= start + 1000]
                if not subset:
                    continue
                value = lambda key: np.asarray([float(r[key]) for r in subset if key in r])
                returns, losses = value("episode_return"), value("critic/loss")
                up_cos, ball_vz = value("racket_up_cos"), value("ball_vz")
                tracking, hits = value("episode_max_tracking_error_deg"), value("hits")
                writer.writerow(
                    {
                        "run": label,
                        "bin_start": start,
                        "bin_end": start + 1000,
                        "samples": len(subset),
                        "episode_return_mean": returns.mean() if returns.size else "",
                        "critic_loss_mean": losses.mean() if losses.size else "",
                        "racket_up_cos_mean": up_cos.mean() if up_cos.size else "",
                        "racket_up_cos_max": up_cos.max() if up_cos.size else "",
                        "ball_vz_max": ball_vz.max() if ball_vz.size else "",
                        "tracking_error_max_deg": tracking.max() if tracking.size else "",
                        "confirmed_hits_max": hits.max() if hits.size else "",
                    }
                )


if __name__ == "__main__":
    main()
