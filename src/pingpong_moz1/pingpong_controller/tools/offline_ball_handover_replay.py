#!/usr/bin/env python3
"""Paired frozen-record handover replay using the actual actor, pure NumPy."""

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
from pingpong_controller.policy_ball_handover import (  # noqa: E402
    BoundedHandoverBlend,
    PolicyBallHandoverMode,
    is_fractional_refresh,
)


HANDOVER_STATES = {"reacquisition_handover", "reacquisition_smoothing"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _actor(checkpoint: Path):
    with checkpoint.open("rb") as handle:
        payload = _CheckpointUnpickler(handle).load()
    flat = _flatten_params(payload["params"])
    weights = {
        name: np.asarray(flat[key], dtype=np.float32)
        for name, key in {
            "l1_w": "pi/l1/w", "l1_b": "pi/l1/b",
            "l2_w": "pi/l2/w", "l2_b": "pi/l2/b",
            "out_w": "pi/out/w", "out_b": "pi/out/b",
        }.items()
    }

    def forward(obs: np.ndarray) -> np.ndarray:
        value = np.asarray(obs, dtype=np.float32).reshape(67)
        value = np.tanh(value @ weights["l1_w"] + weights["l1_b"])
        value = np.tanh(value @ weights["l2_w"] + weights["l2_b"])
        return np.clip(
            value @ weights["out_w"] + weights["out_b"],
            -1.0, 1.0).astype(np.float64)

    return forward


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if (not bool(row.get("policyExecuted"))
                    or len(row.get("policyObs") or []) != 67):
                continue
            rows.append(row)
    return rows


def _events(rows: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    handovers: list[int] = []
    impacts: list[int] = []
    previous_state = ""
    previous_impact = False
    for index, row in enumerate(rows):
        state = str((row.get("highBallOcclusionPrediction") or {}).get(
            "state", ""))
        if state in HANDOVER_STATES and previous_state not in HANDOVER_STATES:
            handovers.append(index)
        impact = bool((row.get("cameraMeasurement") or {}).get(
            "velocity_impact_detected", False))
        if impact and not previous_impact:
            impacts.append(index)
        previous_state = state
        previous_impact = impact
    return handovers, impacts


def replay(path: Path, checkpoint: Path) -> dict[str, Any]:
    rows = _load_rows(path)
    handovers, impacts = _events(rows)
    selection = None
    for handover in handovers:
        # A handover already active at the first saved policy row has no
        # preceding held state and therefore cannot identify the transition.
        if handover < 30:
            continue
        future_impacts = [index for index in impacts if index > handover]
        if len(future_impacts) >= 2:
            selection = (handover, future_impacts[1])
            break
    if selection is None:
        raise RuntimeError(
            "frozen input has no handover followed by two inferred impacts")
    handover_index, second_impact_index = selection
    start_index = max(0, handover_index - 30)
    end_index = min(len(rows), second_impact_index + 1)
    actor = _actor(checkpoint)

    def run(mode: PolicyBallHandoverMode) -> tuple[np.ndarray, np.ndarray]:
        held = np.asarray(rows[start_index]["policyObs"][20:26],
                          dtype=np.float64)
        blend = BoundedHandoverBlend(0.18, 8.0, 0.04)
        observations = []
        actions = []
        previous_state = str((rows[start_index].get(
            "highBallOcclusionPrediction") or {}).get("state", ""))
        for index in range(start_index, end_index):
            row = rows[index]
            obs = np.asarray(row["policyObs"], dtype=np.float32)
            candidate = np.asarray(obs[20:26], dtype=np.float64)
            state = str((row.get("highBallOcclusionPrediction") or {}).get(
                "state", ""))
            handover = (
                state in HANDOVER_STATES
                and previous_state not in HANDOVER_STATES)
            scheduled = is_fractional_refresh(index, 90.0, 0.005)
            if mode is PolicyBallHandoverMode.LEGACY_FORCE_FRESH:
                if scheduled or handover:
                    held = candidate.copy()
            else:
                if handover and not blend.active:
                    blend.begin(
                        held[:3], held[3:], candidate[:3], candidate[3:])
                if scheduled:
                    if blend.active:
                        position, velocity, _ = blend.scheduled_step(
                            held[:3], held[3:], 1.0 / 90.0)
                        held = np.concatenate((position, velocity))
                    else:
                        held = candidate.copy()
            paired_obs = obs.copy()
            paired_obs[20:26] = held.astype(np.float32)
            observations.append(held.copy())
            actions.append(actor(paired_obs))
            previous_state = state
        return np.asarray(observations), np.asarray(actions)

    legacy_obs, legacy_action = run(
        PolicyBallHandoverMode.LEGACY_FORCE_FRESH)
    strict_obs, strict_action = run(
        PolicyBallHandoverMode.STRICT_FRACTIONAL_CONTINUOUS)
    handover_local = handover_index - start_index
    second_local = second_impact_index - start_index

    def max_jump(values: np.ndarray) -> float:
        return float(np.max(np.linalg.norm(
            np.diff(values, axis=0), axis=1), initial=0.0))

    cumulative = float(np.sum(np.linalg.norm(
        strict_obs[handover_local:second_local + 1]
        - legacy_obs[handover_local:second_local + 1], axis=1)))
    return {
        "schemaVersion": 1,
        "executionContract": "same_saved_rows_pure_numpy_no_ros_no_jax_no_gpu",
        "claimScope": (
            "frozen one-step actor sensitivity only; not a closed-loop replay "
            "and not paired real A/B evidence"),
        "input": {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "policyRows": len(rows),
            "windowStartIndex": start_index,
            "handoverIndex": handover_index,
            "secondInferredImpactIndex": second_impact_index,
            "windowEndIndexInclusive": end_index - 1,
        },
        "checkpoint": {
            "path": str(checkpoint.resolve()),
            "sha256": _sha256(checkpoint),
        },
        "rates": {
            "controlHz": 200.0,
            "deploymentBallObservationHz": 90.0,
        },
        "pairedResult": {
            "legacyObservationJumpMax": max_jump(legacy_obs),
            "strictObservationJumpMax": max_jump(strict_obs),
            "legacyActionJumpMax": max_jump(legacy_action),
            "strictActionJumpMax": max_jump(strict_action),
            "handoverObservationJump": {
                "legacy": float(np.linalg.norm(
                    legacy_obs[handover_local]
                    - legacy_obs[handover_local - 1])),
                "strict": float(np.linalg.norm(
                    strict_obs[handover_local]
                    - strict_obs[handover_local - 1])),
            },
            "handoverActionJump": {
                "legacy": float(np.linalg.norm(
                    legacy_action[handover_local]
                    - legacy_action[handover_local - 1])),
                "strict": float(np.linalg.norm(
                    strict_action[handover_local]
                    - strict_action[handover_local - 1])),
            },
            "preSecondImpactCumulativeObservationDifference": cumulative,
            "finalPreSecondImpactActionDifferenceRms": float(np.sqrt(
                np.mean(np.square(
                    strict_action[second_local] - legacy_action[second_local])))),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = replay(
        args.input.expanduser().resolve(),
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
