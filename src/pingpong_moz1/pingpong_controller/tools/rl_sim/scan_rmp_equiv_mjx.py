#!/usr/bin/env python3
"""Run MJX RMP/PD calibration against the dedicated equivalence XML.

This is a thin wrapper around the existing simcode5 calibration script.  The
purpose is to keep all RMP-equivalence experiments pointed at
``moz1_rmp_equiv.xml`` without touching the deployment XML used elsewhere in
repository scripts.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _workspace_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "recovered_rmp").is_dir():
            return parent
    raise RuntimeError("cannot locate workspace containing recovered_rmp")


WORKSPACE = _workspace_root()
CALIBRATE_SCRIPT = (
    WORKSPACE
    / "simcode5/pingpong_moz1-main/pingpong_controller/tools/rl_sim/calibrate_rmp_pd_mjx.py"
)
DEFAULT_MODEL = (
    WORKSPACE
    / "src/pingpong_moz1/pingpong_controller/tools/rl_sim/moz1_rmp_equiv.xml"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=Path("rmp_equiv_scan.json"))
    parser.add_argument("--train", type=Path, nargs="+", default=None)
    parser.add_argument("--validation", type=Path, nargs="+", default=None)
    parser.add_argument("--warmup-samples", type=int, default=100)
    parser.add_argument("--cooldown-samples", type=int, default=20)
    parser.add_argument("--joint-sweeps", type=int, default=2)
    parser.add_argument("--delay-joint-sweeps", type=int, default=2)
    parser.add_argument("--rmp-sweeps", type=int, default=0)
    parser.add_argument("--rmp-joint-sweeps", type=int, default=0)
    parser.add_argument("--dynamics-joint-sweeps", type=int, default=2)
    parser.add_argument("--fine-search", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    args = parser.parse_args()

    command = [
        str(args.python),
        str(CALIBRATE_SCRIPT),
        "--model",
        str(args.model),
        "--output",
        str(args.output),
        "--warmup-samples",
        str(args.warmup_samples),
        "--cooldown-samples",
        str(args.cooldown_samples),
        "--joint-sweeps",
        str(args.joint_sweeps),
        "--delay-joint-sweeps",
        str(args.delay_joint_sweeps),
        "--rmp-sweeps",
        str(args.rmp_sweeps),
        "--rmp-joint-sweeps",
        str(args.rmp_joint_sweeps),
        "--dynamics-joint-sweeps",
        str(args.dynamics_joint_sweeps),
    ]
    if args.fine_search:
        command.append("--fine-search")
    if args.train is not None:
        command.append("--train")
        command.extend(str(item) for item in args.train)
    if args.validation is not None:
        command.append("--validation")
        command.extend(str(item) for item in args.validation)
    command.extend(args.extra_arg)

    print("[scan_rmp_equiv_mjx]", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
