#!/usr/bin/env python3
"""Build an interactive HTML dashboard for batch RMP replay validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_trace(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {key: np.asarray(data[key]) for key in data.files}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = _load_json(args.batch_summary)
    records: list[dict[str, object]] = []
    for row in summary["records"]:
        if row.get("status") != "pass":
            records.append({**row, "replay_report": None, "trace": None})
            continue
        trace_dir = Path(row["trace_dir"])
        trace = _load_trace(trace_dir / "rmp_mjx_replay_trace.npz")
        replay = _load_json(trace_dir / "rmp_mjx_replay_report.json")
        records.append(
            {
                **row,
                "replay_report": replay,
                "trace": {key: np.asarray(value).tolist() for key, value in trace.items()},
            }
        )

    payload = {
        "title": f"RMP Batch Validation — {Path(summary['model']).name}",
        "summary": summary,
        "records": records,
        "all_records": summary["records"],
        "groups": summary["groups"],
    }
    data_json = json.dumps(payload, ensure_ascii=False)

    html_doc = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>RMP Batch Validation</title>
  <style>
    :root {
      --bg: #f6f8fb;
      --panel: #ffffff;
      --line: #d9e0ea;
      --line-soft: #eef2f6;
      --text: #102033;
      --muted: #5f6f82;
      --accent: #2456cc;
      --good: #2ca02c;
      --warn: #d98e00;
      --bad: #d62728;
    }
    body { margin: 0; font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: var(--bg); color: var(--text); }
    header { padding: 18px 22px 12px; background: var(--panel); border-bottom: 1px solid var(--line); }
    h1 { margin: 0; font-size: 20px; }
    .sub { color: var(--muted); margin-top: 6px; font-size: 13px; }
    .toolbar { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; padding: 12px 22px; background: var(--panel); border-bottom: 1px solid var(--line); }
    select, button, input { padding: 8px 10px; border: 1px solid #b9c4d2; border-radius: 8px; background: white; }
    button { cursor: pointer; }
    main { display: grid; grid-template-columns: 1.2fr 0.8fr; gap: 14px; padding: 14px 16px 20px; }
    .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04); min-width: 0; }
    .panel h2 { margin: 0; padding: 14px 16px 10px; border-bottom: 1px solid var(--line-soft); font-size: 15px; }
    .card-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; padding: 14px 16px; }
    .card { background: #f9fbfe; border: 1px solid #e2e8f0; border-radius: 12px; padding: 12px; }
    .card .label { color: var(--muted); font-size: 12px; }
    .card .value { font-size: 20px; font-weight: 700; margin-top: 4px; }
    .card .subvalue { color: var(--muted); font-size: 12px; margin-top: 4px; }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 0 16px 12px; }
    .subpanel { background: #fbfcfe; border: 1px solid #e6edf5; border-radius: 12px; padding: 10px 12px; }
    .subpanel h3 { margin: 0 0 8px; font-size: 13px; }
    canvas { width: 100%; height: 460px; display: block; }
    .chartbox { padding: 0 16px 14px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--line-soft); vertical-align: top; }
    tr:hover { background: #f7fbff; }
    tr.skipped { color: #8b97a6; }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #edf4ff; color: var(--accent); font-size: 12px; }
    .pill.good { background: #ebf8ef; color: var(--good); }
    .pill.warn { background: #fff5df; color: var(--warn); }
    .pill.bad { background: #fdecec; color: var(--bad); }
    .legend { display: flex; flex-wrap: wrap; gap: 10px; padding: 0 16px 14px; color: var(--muted); font-size: 12px; }
    .swatch { width: 10px; height: 10px; display: inline-block; border-radius: 999px; margin-right: 6px; }
    .recordings { max-height: 440px; overflow: auto; }
    .timeline { display: flex; gap: 8px; align-items: center; padding: 0 16px 12px; color: var(--muted); font-size: 12px; }
    .timeline .bar { flex: 1; height: 12px; border-radius: 999px; background: linear-gradient(90deg, #e5ebf2 0%, #d7f0da 35%, #ffe7bf 70%, #f6c5c5 100%); position: relative; overflow: hidden; }
    .timeline .marker { position: absolute; top: 0; bottom: 0; width: 2px; background: #102033; }
    .small { font-size: 12px; color: var(--muted); }
  </style>
</head>
<body>
  <header>
    <h1 id="title"></h1>
    <div class="sub">All recordings analyzed with valid juggling phase only. Failure tails are excluded from optimization and reported separately for context.</div>
  </header>
  <div class="toolbar">
    <label>Group <select id="groupSel"></select></label>
    <label>Recording <select id="recordSel"></select></label>
    <label>Metric <select id="metricSel"><option value="q">q</option><option value="dq">dq</option><option value="qdd">qdd</option></select></label>
    <label><input id="showSkipped" type="checkbox" /> show skipped</label>
    <button id="prevBtn">Prev</button>
    <button id="nextBtn">Next</button>
    <span id="status"></span>
  </div>
  <main>
    <section class="panel">
      <h2>Overview</h2>
      <div class="card-grid" id="cards"></div>
      <div class="grid2">
        <div class="subpanel">
          <h3>Group summary</h3>
          <div id="groupSummary"></div>
        </div>
        <div class="subpanel">
          <h3>Phase timeline</h3>
          <div class="timeline">
            <span>wait</span>
            <div class="bar"><div class="marker" id="phaseMarker"></div></div>
            <span>fail tail</span>
          </div>
          <div class="small">The optimization uses only the green valid-phase interval from each trace.</div>
        </div>
      </div>
      <div class="legend" id="legend"></div>
      <div class="chartbox"><canvas id="chart" width="1200" height="520"></canvas></div>
    </section>
    <section class="panel" id="detail">
      <h2>All recordings</h2>
      <div class="recordings">
        <table id="table"></table>
      </div>
    </section>
  </main>
  <script>
    const DATA = __DATA__;
    document.getElementById('title').textContent = DATA.title;
    const chart = document.getElementById('chart');
    const ctx = chart.getContext('2d');
    const groupSel = document.getElementById('groupSel');
    const recordSel = document.getElementById('recordSel');
    const metricSel = document.getElementById('metricSel');
    const showSkipped = document.getElementById('showSkipped');
    const cards = document.getElementById('cards');
    const table = document.getElementById('table');
    const status = document.getElementById('status');
    const legend = document.getElementById('legend');
    const groupSummary = document.getElementById('groupSummary');
    const phaseMarker = document.getElementById('phaseMarker');
    const groups = Object.keys(DATA.groups).sort();
    let currentGroup = groups[0] || 'all';
    let currentIndex = 0;

    function format(v) { return Number.isFinite(v) ? v.toFixed(4) : 'n/a'; }
    function formatCount(v) { return Number.isFinite(v) ? String(v) : 'n/a'; }
    function currentRecords() {
      let rows = DATA.records.filter(r => r.group === currentGroup);
      if (!showSkipped.checked) rows = rows.filter(r => r.status === 'pass');
      return rows;
    }
    function currentRecord() { return currentRecords().filter(r => r.status === 'pass' && r.trace)[currentIndex]; }

    function buildGroups() {
      groupSel.innerHTML = groups.map(g => `<option value="${g}">${g}</option>`).join('');
      groupSel.value = currentGroup;
      buildRecords();
      buildGroupSummary();
    }

    function buildGroupSummary() {
      groupSummary.innerHTML = groups.map(g => {
        const s = DATA.groups[g];
        return `<div class="card" style="margin-bottom:8px"><div class="label">${g}</div><div class="small">count: ${s.count} / pass: ${s.pass_count}</div><div class="small">q=${format(s.position_rms_rad_mean)} dq=${format(s.velocity_rms_rad_s_mean)} qdd=${format(s.acceleration_rms_rad_s2_mean)}</div></div>`;
      }).join('');
    }

    function buildRecords() {
      const rows = currentRecords();
      recordSel.innerHTML = rows.map((r, i) => `<option value="${i}">${r.recording_name} — q=${format(r.position_rms_rad)} dq=${format(r.velocity_rms_rad_s)} qdd=${format(r.acceleration_rms_rad_s2)}${r.status !== 'pass' ? ' (' + r.status + ')' : ''}</option>`).join('');
      if (rows.length === 0) {
        recordSel.innerHTML = '<option value="0">No records</option>';
        currentIndex = 0;
        return;
      }
      if (currentIndex >= rows.length) currentIndex = 0;
      recordSel.value = String(currentIndex);
      drawTable();
      updateDetail();
    }

    function drawCards(record) {
      cards.innerHTML = [
        ['Records analyzed', DATA.summary.status_counts.pass],
        ['Skipped', DATA.summary.status_counts.skipped],
        ['Current valid samples', record.policy_samples],
        ['Current phase duration', format(record.valid_duration_s)],
      ].map(([label, value]) => `<div class="card"><div class="label">${label}</div><div class="value">${value}</div></div>`).join('');
    }

    function drawTable() {
      const rows = DATA.records.filter(r => showSkipped.checked || r.status === 'pass');
      table.innerHTML = `
        <thead>
          <tr><th>Status</th><th>Group</th><th>Recording</th><th>Valid phase</th><th>q</th><th>dq</th><th>qdd</th><th>Samples</th></tr>
        </thead>
        <tbody>
          ${rows.map((r, i) => `<tr data-index="${i}" class="${r.status === 'skipped' ? 'skipped' : ''}"><td><span class="pill ${r.status === 'pass' ? 'good' : 'warn'}">${r.status}</span></td><td><span class="pill">${r.group}</span></td><td>${r.recording_name}</td><td>${formatCount(r.policy_samples)} samples</td><td>${format(r.position_rms_rad)}</td><td>${format(r.velocity_rms_rad_s)}</td><td>${format(r.acceleration_rms_rad_s2)}</td><td>${formatCount(r.samples)}</td></tr>`).join('')}
        </tbody>`;
      table.querySelectorAll('tr[data-index]').forEach(tr => tr.addEventListener('click', () => {
        const idx = Number(tr.dataset.index);
        const row = rows[idx];
        if (row.status !== 'pass') return;
        currentGroup = row.group;
        groupSel.value = currentGroup;
        currentIndex = currentRecords().findIndex(r => r.recording_name === row.recording_name);
        buildRecords();
      }));
    }

    function drawPlot(record) {
      const trace = record.trace;
      const keyMap = { q: ['actual_q', 'sim_q'], dq: ['actual_qd', 'sim_qd'], qdd: ['actual_qdd', 'sim_qdd'] };
      const mode = metricSel.value;
      const [realKey, simKey] = keyMap[mode];
      const time = trace.time;
      const real = trace[realKey];
      const sim = trace[simKey];
      const colors = ['#2456cc', '#e15759', '#2ca02c', '#ff9d00', '#9b59b6', '#17becf', '#8c564b'];
      const validStart = record.valid_start_index ?? 0;
      const validStop = record.valid_stop_index ?? (time.length - 1);
      const pad = 44;
      const w = chart.width, h = chart.height;
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, w, h);
      ctx.strokeStyle = '#dbe4ef';
      ctx.lineWidth = 1;
      ctx.strokeRect(pad, 24, w - pad * 1.5, h - 64);
      const flat = [].concat(...real, ...sim).flat().filter(Number.isFinite);
      const min = Math.min(...flat);
      const max = Math.max(...flat);
      const y0 = 24, y1 = h - 40;
      const x0 = pad, x1 = w - pad * 0.5;
      const tEnd = Math.max(time[time.length - 1], 1e-9);
      function X(t) { return x0 + (t / tEnd) * (x1 - x0); }
      function Y(v) { return y1 - ((v - min) / Math.max(max - min, 1e-9)) * (y1 - y0); }
      ctx.font = '12px sans-serif';
      ctx.fillStyle = '#102033';
      ctx.fillText(`${mode} comparison`, x0, 16);
      for (let j = 0; j < 7; j++) {
        ctx.beginPath();
        ctx.strokeStyle = colors[j];
        ctx.lineWidth = 1.6;
        for (let i = 0; i < time.length; i++) {
          const x = X(time[i]);
          const y = Y(real[i][j]);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        for (let i = 0; i < time.length; i++) {
          const x = X(time[i]);
          const y = Y(sim[i][j]);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
      }
      const startX = X(time[Math.max(validStart, 0)] || 0);
      const stopX = X(time[Math.min(validStop, time.length - 1)] || 0);
      ctx.fillStyle = 'rgba(36, 86, 204, 0.08)';
      ctx.fillRect(startX, 24, Math.max(stopX - startX, 2), h - 64);
      phaseMarker.style.left = `${Math.max(0, Math.min(100, (startX - x0) / (x1 - x0) * 100))}%`;
      legend.innerHTML = [
        'solid = real',
        'dashed = sim',
        'blue band = valid juggling phase',
        'table shows all recordings, including skipped ones',
      ].map((t, idx) => `<span><span class="swatch" style="background:${['#2456cc', '#e15759', '#2ca02c', '#7f8b99'][idx]}"></span>${t}</span>`).join('');
    }

    function updateDetail() {
      const rec = currentRecord();
      if (!rec) {
        status.textContent = 'No recording for current filter';
        return;
      }
      drawCards(rec);
      status.textContent = `${rec.group} / ${rec.recording_name} / valid samples=${rec.policy_samples} / status=${rec.status}`;
      drawPlot(rec);
    }

    groupSel.addEventListener('change', () => { currentGroup = groupSel.value; currentIndex = 0; buildRecords(); });
    recordSel.addEventListener('change', () => { currentIndex = Number(recordSel.value); updateDetail(); });
    metricSel.addEventListener('change', () => { updateDetail(); });
    showSkipped.addEventListener('change', () => { currentIndex = 0; buildRecords(); });
    document.getElementById('prevBtn').addEventListener('click', () => { currentIndex = Math.max(0, currentIndex - 1); recordSel.value = String(currentIndex); updateDetail(); });
    document.getElementById('nextBtn').addEventListener('click', () => { currentIndex = Math.min(currentRecords().length - 1, currentIndex + 1); recordSel.value = String(currentIndex); updateDetail(); });

    buildGroups();
    updateDetail();
  </script>
</body>
</html>
"""

    html_doc = html_doc.replace("__DATA__", data_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html_doc, encoding="utf-8")
    print(f"[dashboard] wrote: {args.output}")


if __name__ == "__main__":
    main()
