#!/usr/bin/env python3
"""Replay one recording through the in-environment 1 kHz RMP/MJX path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from mjx_juggle_env import MjxJuggleEnv
from rmp_mjx_replay_batch_core import (
    add_replay_config_args,
    build_replay_batch,
    initialize_batch_state,
    make_record_artifacts,
    prepare_recording,
    resolve_replay_config,
    run_batched_replay,
    write_record_artifacts,
)


def _workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "recovered_rmp").is_dir():
            return parent
    raise RuntimeError("cannot locate workspace containing recovered_rmp")


WORKSPACE = _workspace_root()
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Replay a recorded position command through the exact MJX chain: "
            "target -> five 1 ms recovered-RMP updates -> five MJX substeps."
        )
    )
    parser.add_argument("recording", type=Path)
    parser.add_argument("--model", type=Path, default=here / "moz1_pd.xml")
    parser.add_argument("--output", type=Path, default=Path("rmp_mjx_replay_output"))
    parser.add_argument("--seed", type=int, default=0)
    add_replay_config_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = resolve_replay_config(args)
    record = prepare_recording(
        args.recording,
        group="recording",
        segment=args.segment,
        valid_phase_only=bool(args.valid_phase_only),
        max_samples=int(args.max_samples),
    )
    batch = build_replay_batch([record])
    env = MjxJuggleEnv(args.model, n_envs=1, cfg=config.env_cfg)
    state = initialize_batch_state(env, batch, seed=int(args.seed))
    result = run_batched_replay(env, state, batch)
    arrays, report = make_record_artifacts(
        record,
        result,
        0,
        float(env.dt),
        config,
        batch_index=0,
        padded_batch_samples=int(batch.targets.shape[0]),
    )
    output = args.output.expanduser().resolve()
    trace_path, report_path = write_record_artifacts(output, arrays, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"trace: {trace_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
