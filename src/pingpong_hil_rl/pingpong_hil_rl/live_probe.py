from __future__ import annotations

import argparse
import json
import time

import numpy as np

from .ros_plant import PlantConfig, RosPlantRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only HIL timing and state probe; publishes no commands")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--joint-topic", default="/joint_states")
    parser.add_argument("--command-topic", default="/mx_mix_command")
    args = parser.parse_args()

    runtime = RosPlantRuntime(
        PlantConfig(joint_topic=args.joint_topic, command_topic=args.command_topic, read_only=True)
    )
    try:
        runtime.plant.wait_for_state(timeout_s=3.0)
        samples = runtime.plant.collect_samples(args.samples, timeout_s=max(5.0, args.samples / 100.0))
        source_dt = np.diff([s.source_stamp_ns for s in samples]).astype(np.float64) * 1e-6
        receive_dt = np.diff([s.receive_stamp_ns for s in samples]).astype(np.float64) * 1e-6
        q = np.stack([s.q for s in samples])
        dq = np.stack([s.dq for s in samples])
        publishers = runtime.plant.get_publishers_info_by_topic(args.command_topic)
        report = {
            "read_only": True,
            "samples": len(samples),
            "source_rate_hz": 1000.0 / float(np.median(source_dt)),
            "source_dt_ms": {
                "p50": float(np.percentile(source_dt, 50)),
                "p99": float(np.percentile(source_dt, 99)),
                "max": float(np.max(source_dt)),
            },
            "receive_dt_ms": {
                "p50": float(np.percentile(receive_dt, 50)),
                "p99": float(np.percentile(receive_dt, 99)),
                "max": float(np.max(receive_dt)),
            },
            "source_non_monotonic": int(np.sum(source_dt <= 0.0)),
            "source_gaps_over_7_5ms": int(np.sum(source_dt > 7.5)),
            "q_mean_deg": np.rad2deg(np.mean(q, axis=0)).round(4).tolist(),
            "q_span_deg": np.rad2deg(np.ptp(q, axis=0)).round(4).tolist(),
            "dq_abs_max_deg_s": np.rad2deg(np.max(np.abs(dq), axis=0)).round(3).tolist(),
            "velocity_sources": sorted({s.velocity_source for s in samples}),
            "command_publishers": sorted(
                {f"{info.node_namespace}/{info.node_name}".replace("//", "/") for info in publishers}
            ),
        }
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
