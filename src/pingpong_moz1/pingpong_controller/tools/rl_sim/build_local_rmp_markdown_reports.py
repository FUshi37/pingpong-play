#!/usr/bin/env python3
"""Generate local markdown summaries for selected replay batches."""

from __future__ import annotations

import argparse
from pathlib import Path

from summarize_rmp_replay_reports import main as summarize_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for root in args.roots:
        out_json = args.output_dir / f"{root.name}.json"
        out_md = args.output_dir / f"{root.name}.md"
        import sys
        old_argv = sys.argv
        try:
            sys.argv = [
                "summarize_rmp_replay_reports.py",
                str(root),
                "--markdown-output",
                str(out_md),
            ]
            summarize_main()
        finally:
            sys.argv = old_argv
        print(f"[batch-summary] wrote {out_json} and {out_md}")
