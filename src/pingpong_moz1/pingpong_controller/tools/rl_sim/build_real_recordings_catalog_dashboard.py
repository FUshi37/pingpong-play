#!/usr/bin/env python3
"""Build an HTML dashboard for the full real-recording catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    data_json = json.dumps(catalog, ensure_ascii=False)
    html_doc = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>All Real Recordings Catalog</title>
  <style>
    :root { --bg:#f6f8fb; --panel:#fff; --line:#d9e0ea; --text:#102033; --muted:#5f6f82; --good:#2ca02c; --warn:#d98e00; --bad:#d62728; }
    body { margin:0; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:var(--bg); color:var(--text); }
    header { padding:18px 22px 12px; background:var(--panel); border-bottom:1px solid var(--line); }
    h1 { margin:0; font-size:20px; }
    .sub { color:var(--muted); margin-top:6px; font-size:13px; }
    .toolbar { display:flex; gap:12px; flex-wrap:wrap; align-items:center; padding:12px 22px; background:var(--panel); border-bottom:1px solid var(--line); }
    select, input { padding:8px 10px; border:1px solid #b9c4d2; border-radius:8px; background:white; }
    main { display:grid; grid-template-columns: 1fr 1fr; gap:14px; padding:14px 16px 20px; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:14px; box-shadow:0 1px 2px rgba(15,23,42,0.04); }
    .panel h2 { margin:0; padding:14px 16px 10px; border-bottom:1px solid #eef2f6; font-size:15px; }
    .card-grid { display:grid; grid-template-columns: repeat(4,1fr); gap:10px; padding:14px 16px; }
    .card { background:#f9fbfe; border:1px solid #e2e8f0; border-radius:12px; padding:12px; }
    .card .label { color:var(--muted); font-size:12px; }
    .card .value { font-size:20px; font-weight:700; margin-top:4px; }
    table { width:100%; border-collapse:collapse; font-size:13px; }
    th, td { text-align:left; padding:8px 10px; border-bottom:1px solid #edf1f5; vertical-align:top; }
    tr:hover { background:#f7fbff; }
    .pill { display:inline-block; padding:2px 8px; border-radius:999px; background:#edf4ff; color:#2456cc; font-size:12px; }
    .pill.good { background:#ebf8ef; color:var(--good); }
    .pill.warn { background:#fff5df; color:var(--warn); }
    .pill.bad { background:#fdecec; color:var(--bad); }
    .recordings { max-height:520px; overflow:auto; }
    .small { color:var(--muted); font-size:12px; }
  </style>
</head>
<body>
  <header>
    <h1 id="title"></h1>
    <div class="sub">All 368 real recordings, including 333 valid phase traces and 35 skipped files. The valid phase is the contiguous policy-mask interval; failure tails are excluded from optimization.</div>
  </header>
  <div class="toolbar">
    <label>Group <select id="groupSel"></select></label>
    <label>Status <select id="statusSel"><option value="all">all</option><option value="valid">valid only</option><option value="skipped">skipped only</option></select></label>
    <label>Search <input id="searchBox" type="search" placeholder="recording name" /></label>
    <span id="status"></span>
  </div>
  <main>
    <section class="panel">
      <h2>Catalog summary</h2>
      <div class="card-grid" id="cards"></div>
      <div class="small" style="padding:0 16px 14px;" id="summaryText"></div>
    </section>
    <section class="panel">
      <h2>All recordings</h2>
      <div class="recordings"><table id="table"></table></div>
    </section>
  </main>
  <script>
    const DATA = __DATA__;
    document.getElementById('title').textContent = `All Real Recordings — ${DATA.record_root.split('/').pop()}`;
    const groups = ['all', ...Object.keys(DATA.groups).sort()];
    const groupSel = document.getElementById('groupSel');
    const statusSel = document.getElementById('statusSel');
    const searchBox = document.getElementById('searchBox');
    const cards = document.getElementById('cards');
    const table = document.getElementById('table');
    const status = document.getElementById('status');
    const summaryText = document.getElementById('summaryText');

    function fmt(v) { return Number.isFinite(v) ? v.toFixed(4) : 'n/a'; }
    function cnt(v) { return v == null ? 'n/a' : String(v); }
    function visibleRows() {
      const term = searchBox.value.trim().toLowerCase();
      return DATA.records.filter(row => {
        if (groupSel.value !== 'all' && row.group !== groupSel.value) return false;
        if (statusSel.value !== 'all' && row.status !== statusSel.value) return false;
        if (term && !row.recording_name.toLowerCase().includes(term)) return false;
        return true;
      });
    }

    function renderCards(rows) {
      const valid = rows.filter(r => r.status === 'valid');
      cards.innerHTML = [
        ['Total files', DATA.total_files],
        ['Valid traces', DATA.valid_files],
        ['Skipped files', DATA.skipped_files],
        ['Groups', Object.keys(DATA.groups).length],
      ].map(([label, value]) => `<div class="card"><div class="label">${label}</div><div class="value">${value}</div></div>`).join('');
      summaryText.textContent = `${rows.length} rows shown. ${valid.length} are valid phase traces.`;
      status.textContent = `showing ${rows.length} / ${DATA.total_files}`;
    }

    function renderTable(rows) {
      table.innerHTML = `
        <thead>
          <tr><th>Status</th><th>Group</th><th>Recording</th><th>Valid phase</th><th>Samples</th><th>Reason / stats</th></tr>
        </thead>
        <tbody>
          ${rows.map(row => {
            const pill = row.status === 'valid' ? 'good' : (row.skip_reason ? 'bad' : 'warn');
            const stats = row.status === 'valid'
              ? `valid ${cnt(row.policy_samples)} / ${cnt(row.samples)} samples, ${fmt(row.valid_duration_s)} s`
              : row.skip_reason;
            return `<tr><td><span class="pill ${pill}">${row.status}</span></td><td><span class="pill">${row.group}</span></td><td>${row.recording_name}</td><td>${cnt(row.policy_samples)}</td><td>${cnt(row.samples)}</td><td>${stats || ''}</td></tr>`;
          }).join('')}
        </tbody>`;
    }

    function renderGroupOptions() {
      groupSel.innerHTML = groups.map(g => `<option value="${g}">${g}</option>`).join('');
    }

    function render() {
      const rows = visibleRows();
      renderCards(rows);
      renderTable(rows);
    }

    groupSel.addEventListener('change', render);
    statusSel.addEventListener('change', render);
    searchBox.addEventListener('input', render);
    renderGroupOptions();
    render();
  </script>
</body>
</html>
"""
    html_doc = html_doc.replace("__DATA__", data_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_doc, encoding="utf-8")
    print(f"[catalog-dashboard] wrote: {args.output}")


if __name__ == "__main__":
    main()
