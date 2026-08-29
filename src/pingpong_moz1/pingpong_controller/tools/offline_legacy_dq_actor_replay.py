#!/usr/bin/env python3
"""Replay legacy dq payloads and the frozen actor without ROS/JAX/GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
PACKAGE_ROOT = SCRIPT_PATH.parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from pingpong_controller.mjx_rl_policy import (  # noqa: E402
    _CheckpointUnpickler,
    _flatten_params,
)
from pingpong_controller.policy_dq_observation_adapter import (  # noqa: E402
    PolicyDqAdapterConfig,
    PolicyDqObservationAdapter,
    SIMCODE5_DQ_ADAPTER_DELAY_STEPS,
    SIMCODE5_DQ_ADAPTER_GAIN,
    SIMCODE5_DQ_ADAPTER_TAU_S,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def _actor_weights(checkpoint_path: Path) -> dict[str, np.ndarray]:
    with checkpoint_path.open("rb") as handle:
        payload = _CheckpointUnpickler(handle).load()
    flat = _flatten_params(payload["params"])
    return {
        "l1_w": np.asarray(flat["pi/l1/w"], dtype=np.float32),
        "l1_b": np.asarray(flat["pi/l1/b"], dtype=np.float32),
        "l2_w": np.asarray(flat["pi/l2/w"], dtype=np.float32),
        "l2_b": np.asarray(flat["pi/l2/b"], dtype=np.float32),
        "out_w": np.asarray(flat["pi/out/w"], dtype=np.float32),
        "out_b": np.asarray(flat["pi/out/b"], dtype=np.float32),
    }


def _actor_forward(obs: Any, weights: dict[str, np.ndarray]) -> np.ndarray:
    value = np.asarray(obs, dtype=np.float32).reshape(67)
    value = np.tanh(value @ weights["l1_w"] + weights["l1_b"])
    value = np.tanh(value @ weights["l2_w"] + weights["l2_b"])
    value = value @ weights["out_w"] + weights["out_b"]
    return np.clip(value, -1.0, 1.0).astype(np.float32)


def replay_recording(
    path: Path,
    weights: dict[str, np.ndarray],
) -> dict[str, Any]:
    config = PolicyDqAdapterConfig(
        enabled=True,
        gain=SIMCODE5_DQ_ADAPTER_GAIN,
        tau_s=SIMCODE5_DQ_ADAPTER_TAU_S,
        delay_steps=SIMCODE5_DQ_ADAPTER_DELAY_STEPS,
        dt_s=0.005,
    )
    adapter = PolicyDqObservationAdapter(config)
    last_index = -1
    adapter_resets = 0
    adapter_errors: list[float] = []
    actor_errors: list[float] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            obs = row.get("policyObs")
            action = row.get("policyAction")
            if (bool(row.get("policyExecuted")) and isinstance(obs, list)
                    and len(obs) == 67 and isinstance(action, list)
                    and len(action) == 7):
                replayed = _actor_forward(obs, weights)
                recorded = np.asarray(action, dtype=np.float32)
                actor_errors.append(float(np.sqrt(np.mean(
                    np.square(replayed - recorded)))))

            payload = row.get("policyDqAdapter")
            if not isinstance(payload, dict) or not payload.get("active"):
                continue
            index = int(payload["sampleIndex"])
            if index <= last_index:
                adapter.reset("recorded_segment_boundary")
                adapter_resets += 1
            replayed_dq, replay_payload = adapter.update(
                payload["rawDqRadS"],
                source_stamp_s=payload.get("sourceSampleStampS"),
                enabled=True,
            )
            if int(replay_payload["sampleIndex"]) != index:
                raise RuntimeError(
                    f"adapter sample index mismatch in {path}: "
                    f"recorded={index}, replay={replay_payload['sampleIndex']}")
            recorded_dq = np.asarray(
                payload["policyDqRadS"], dtype=np.float32)
            adapter_errors.append(float(np.sqrt(np.mean(
                np.square(replayed_dq - recorded_dq)))))
            last_index = index

    if not actor_errors or not adapter_errors:
        raise RuntimeError(f"recording lacks required replay rows: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "actorActionRms": _summary(actor_errors),
        "legacyDqRms": _summary(adapter_errors),
        "adapterSegmentResetCount": adapter_resets,
        "actorP99Below1e5": bool(np.percentile(actor_errors, 99) < 1e-5),
        "legacyDqExactFloat32": bool(max(adapter_errors) == 0.0),
    }


def replay(
    paths: list[Path],
    checkpoint_path: Path,
) -> dict[str, Any]:
    weights = _actor_weights(checkpoint_path)
    recordings = [replay_recording(path, weights) for path in paths]
    return {
        "schemaVersion": 1,
        "executionContract": "pure_numpy_no_ros_no_jax_no_gpu",
        "claimScope": (
            "historical_actor_and_legacy_dq_reproduction_only; "
            "does_not_claim_plant_alignment_or_real_robot_benefit"),
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": _sha256(checkpoint_path),
        },
        "recordings": recordings,
        "acceptance": {
            "allActorP99RmsBelow1e5": all(
                item["actorP99Below1e5"] for item in recordings),
            "allLegacyDqExactFloat32": all(
                item["legacyDqExactFloat32"] for item in recordings),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay(
        [path.expanduser().resolve() for path in args.input],
        args.checkpoint.expanduser().resolve(),
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
