#!/usr/bin/env python3
"""Build a small HTML dashboard from replay summary JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    html = f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Replay Summary</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #f6f8fb; color: #102033; }}
.card {{ background: white; border: 1px solid #d9e0ea; border-radius: 12px; padding: 16px; margin-bottom: 14px; }}
.grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }}
.k {{ color: #5f6f82; font-size: 12px; }}
.v {{ font-size: 24px; font-weight: 700; }}
table {{ width: 100%; border-collapse: collapse; background: white; }}
th, td {{ border-bottom: 1px solid #edf1f5; text-align: left; padding: 8px 10px; font-size: 13px; }}
</style>
</head>
<body>
<h1>Replay Summary</h1>
<div class='card grid'>
  <div><div class='k'>Count</div><div class='v'>{summary.get('count', 0)}</div></div>
  <div><div class='k'>Mean q RMS</div><div class='v'>{summary.get('mean_q', 0):.6f}</div></div>
  <div><div class='k'>Mean dq RMS</div><div class='v'>{summary.get('mean_dq', 0):.6f}</div></div>
  <div><div class='k'>Mean qdd RMS</div><div class='v'>{summary.get('mean_qdd', 0):.6f}</div></div>
  <div><div class='k'>Worst q</div><div class='v'>{summary.get('worst_q', {}).get('q', 0):.6f}</div></div>
  <div><div class='k'>Worst qdd</div><div class='v'>{summary.get('worst_qdd', {}).get('qdd', 0):.6f}</div></div>
</div>
<table>
<thead><tr><th>Group</th><th>Recording</th><th>q</th><th>dq</th><th>qdd</th><th>samples</th></tr></thead>
<tbody>
{''.join(f"<tr><td>{r.get('group','')}</td><td>{r.get('recording','')}</td><td>{r.get('q',0):.6f}</td><td>{r.get('dq',0):.6f}</td><td>{r.get('qdd',0):.6f}</td><td>{r.get('samples','')}</td></tr>" for r in summary.get('records', []))}
</tbody>
</table>
</body>
</html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding='utf-8')
    print(f"[summary-dashboard] wrote: {args.output}")


if __name__ == '__main__':
    main()
