#!/usr/bin/env python3
"""Generate an HTML heard/not-heard report from a JS8 JSONL decode log.

Usage:
    python3 js8_report.py build/js8_20260613.jsonl
    python3 js8_report.py build/js8_20260613.jsonl --open
"""

import json, re, sys, os, collections
from datetime import datetime

BAND_COLOR = {
    '160m': '#9b59b6', '80m': '#e74c3c', '60m': '#e67e22',
    '40m': '#27ae60',  '30m': '#2980b9', '20m': '#1abc9c', 'other': '#95a5a6',
}

def band(f):
    f /= 1e6
    if 1.8  <= f < 2.0:   return '160m'
    if 3.5  <= f < 4.0:   return '80m'
    if 5.3  <= f < 5.5:   return '60m'
    if 7.0  <= f < 7.3:   return '40m'
    if 10.1 <= f < 10.15: return '30m'
    if 14.0 <= f < 14.35: return '20m'
    return 'other'

def band_badges(bands):
    return ' '.join(
        f'<span class="badge" style="background:{BAND_COLOR.get(b,"#95a5a6")}">{b}</span>'
        for b in sorted(bands)
    )

def mode_short(modes):
    abbr = {'JS8': 'N', 'JS8 Fast': 'F', 'JS8 Slow': 'S', 'JS8 Turbo': 'T', 'JS8 Ultra': 'U'}
    return '/'.join(abbr.get(m, m) for m in sorted(modes))

def snr_bar(avg, mn, mx):
    pct = max(0, min(100, int((avg + 25) / 25 * 100)))
    color = '#e74c3c' if avg < -18 else '#e67e22' if avg < -12 else '#27ae60'
    return (f'<div class="snr-wrap" title="avg {avg:+.1f} dB  min {mn:+.0f}  max {mx:+.0f}">'
            f'<div class="snr-bar" style="width:{pct}%;background:{color}"></div>'
            f'<span class="snr-label">{avg:+.1f}</span></div>')

def build_report(jsonl_path):
    records = [json.loads(l) for l in open(jsonl_path)]
    if not records:
        sys.exit('No records found.')

    grid_re = re.compile(r'\b([A-R]{2}\d{2}(?:[a-x]{2})?)\b')

    heard = {}
    for r in records:
        call = r.get('call', '').strip()
        if not call:
            continue
        if call not in heard:
            heard[call] = {'snrs': [], 'modes': set(), 'bands': set(), 'count': 0,
                           'first': r['unix'], 'last': r['unix'], 'grids': set()}
        h = heard[call]
        h['snrs'].append(r['snr'])
        h['modes'].add(r['mode'])
        h['bands'].add(band(r['freq']))
        h['count'] += 1
        h['first'] = min(h['first'], r['unix'])
        h['last']  = max(h['last'],  r['unix'])
        for g in grid_re.findall(r.get('text', '')):
            if len(g) >= 4:
                h['grids'].add(g)

    mentioned_by = collections.defaultdict(list)
    for r in records:
        sender = r.get('call', '').strip()
        if not sender:
            continue
        m = re.match(r'^\s*\S+:\s+(\S+)\s+(HEARTBEAT|SNR\s*[+-])', r.get('text', ''))
        if m:
            target = m.group(1).strip()
            if not target.startswith('@') and target != sender:
                mentioned_by[target].append(sender)

    not_heard = {
        t: {'mention_count': len(senders),
            'heard_by': dict(collections.Counter(senders).most_common(5))}
        for t, senders in mentioned_by.items() if t not in heard
    }

    timeline = collections.defaultdict(int)
    for r in records:
        timeline[datetime.utcfromtimestamp(r['unix']).strftime('%H:00')] += 1

    times = [r['unix'] for r in records]
    date_str = datetime.utcfromtimestamp(min(times)).strftime('%Y-%m-%d')
    t_start  = datetime.utcfromtimestamp(min(times)).strftime('%H:%M UTC')
    t_end    = datetime.utcfromtimestamp(max(times)).strftime('%H:%M UTC')

    # --- Heard rows ---
    heard_rows = []
    for call, h in sorted(heard.items(), key=lambda x: -x[1]['count']):
        avg = sum(h['snrs']) / len(h['snrs'])
        grids = ', '.join(sorted(h['grids'])[:3]) or '—'
        first = datetime.utcfromtimestamp(h['first']).strftime('%H:%M')
        last  = datetime.utcfromtimestamp(h['last']).strftime('%H:%M')
        heard_rows.append(
            f'<tr>'
            f'<td class="call">{call}</td>'
            f'<td>{band_badges(h["bands"])}</td>'
            f'<td class="num">{h["count"]}</td>'
            f'<td>{snr_bar(avg, min(h["snrs"]), max(h["snrs"]))}</td>'
            f'<td class="mode-cell">{mode_short(h["modes"])}</td>'
            f'<td class="grid">{grids}</td>'
            f'<td class="time">{first}–{last}</td>'
            f'</tr>'
        )

    # --- Not-heard rows ---
    not_heard_rows = []
    for call, info in sorted(not_heard.items(), key=lambda x: -x[1]['mention_count']):
        heard_by_str = ', '.join(
            f'{s}×{n}' if n > 1 else s for s, n in info['heard_by'].items()
        )
        not_heard_rows.append(
            f'<tr>'
            f'<td class="call dim">{call}</td>'
            f'<td class="num">{info["mention_count"]}</td>'
            f'<td class="heard-by">{heard_by_str}</td>'
            f'</tr>'
        )

    # --- Timeline bars ---
    hours = sorted(timeline.keys())
    max_cnt = max(timeline.values()) if timeline else 1
    tl_bars = []
    for h in hours:
        cnt = timeline[h]
        pct = int(cnt / max_cnt * 100)
        tl_bars.append(
            f'<div class="tl-col">'
            f'<div class="tl-bar-wrap">'
            f'<div class="tl-bar" style="height:{pct}%" title="{h}: {cnt} decodes"></div>'
            f'</div>'
            f'<div class="tl-label">{h}</div>'
            f'</div>'
        )

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>JS8 Heard Log — {date_str}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: "Segoe UI", system-ui, sans-serif; background: #0f1117; color: #c9d1d9; font-size: 14px; }}
  .page {{ max-width: 1100px; margin: 0 auto; padding: 24px 16px; }}
  h1 {{ font-size: 1.5rem; color: #e6edf3; margin-bottom: 4px; }}
  .subtitle {{ color: #8b949e; margin-bottom: 24px; font-size: 0.85rem; }}
  .stats {{ display: flex; gap: 16px; margin-bottom: 28px; flex-wrap: wrap; }}
  .stat {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px 18px; min-width: 120px; }}
  .stat-val {{ font-size: 1.6rem; font-weight: 700; color: #58a6ff; }}
  .stat-lbl {{ font-size: 0.75rem; color: #8b949e; margin-top: 2px; }}
  h2 {{ font-size: 1.1rem; color: #e6edf3; margin: 28px 0 12px; border-left: 3px solid #58a6ff; padding-left: 10px; }}
  .tl {{ display: flex; align-items: flex-end; gap: 3px; height: 80px; margin-bottom: 24px; padding: 0 4px; }}
  .tl-col {{ display: flex; flex-direction: column; align-items: center; flex: 1; min-width: 0; }}
  .tl-bar-wrap {{ flex: 1; width: 100%; display: flex; align-items: flex-end; }}
  .tl-bar {{ width: 100%; background: #58a6ff; border-radius: 2px 2px 0 0; min-height: 2px; opacity: 0.8; }}
  .tl-bar:hover {{ opacity: 1; }}
  .tl-label {{ font-size: 9px; color: #484f58; margin-top: 3px; white-space: nowrap; overflow: hidden; }}
  table {{ width: 100%; border-collapse: collapse; margin-bottom: 28px; }}
  th {{ background: #161b22; color: #8b949e; text-align: left; padding: 8px 10px; font-size: 0.75rem; text-transform: uppercase; letter-spacing: .05em; border-bottom: 1px solid #30363d; position: sticky; top: 0; }}
  td {{ padding: 6px 10px; border-bottom: 1px solid #21262d; vertical-align: middle; }}
  tr:hover td {{ background: #161b22; }}
  .call {{ font-family: monospace; font-weight: 600; color: #79c0ff; font-size: 0.9rem; white-space: nowrap; }}
  .call.dim {{ color: #8b949e; }}
  .num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .mode-cell {{ font-family: monospace; font-size: 0.8rem; color: #8b949e; }}
  .grid {{ font-family: monospace; font-size: 0.8rem; color: #8b949e; }}
  .time {{ font-family: monospace; font-size: 0.78rem; color: #6e7681; white-space: nowrap; }}
  .heard-by {{ font-size: 0.8rem; color: #6e7681; }}
  .badge {{ display: inline-block; border-radius: 3px; padding: 1px 5px; font-size: 0.7rem; font-weight: 600; color: #fff; margin-right: 2px; }}
  .snr-wrap {{ display: flex; align-items: center; gap: 6px; min-width: 130px; }}
  .snr-bar {{ height: 8px; border-radius: 2px; min-width: 2px; }}
  .snr-label {{ font-family: monospace; font-size: 0.78rem; color: #8b949e; white-space: nowrap; }}
  .section-note {{ font-size: 0.8rem; color: #6e7681; margin: -8px 0 12px; }}
  input[type=search] {{ background: #161b22; border: 1px solid #30363d; color: #c9d1d9; border-radius: 6px; padding: 6px 10px; font-size: 0.85rem; width: 240px; margin-bottom: 10px; }}
  input[type=search]:focus {{ outline: none; border-color: #58a6ff; }}
</style>
</head>
<body>
<div class="page">
  <h1>JS8 Heard Log — {date_str}</h1>
  <div class="subtitle">{t_start} – {t_end} &nbsp;·&nbsp; granolasdr / KF0RRR</div>
  <div class="stats">
    <div class="stat"><div class="stat-val">{len(records)}</div><div class="stat-lbl">total decodes</div></div>
    <div class="stat"><div class="stat-val">{len(heard)}</div><div class="stat-lbl">stations heard</div></div>
    <div class="stat"><div class="stat-val">{len(not_heard)}</div><div class="stat-lbl">heard by them, not us</div></div>
  </div>
  <h2>Decode activity by hour (UTC)</h2>
  <div class="tl">{''.join(tl_bars)}</div>
  <h2>Stations heard directly</h2>
  <p class="section-note">Stations whose transmissions we decoded. Mode: N=Normal F=Fast S=Slow T=Turbo U=Ultra.</p>
  <input type="search" placeholder="Filter callsign…" oninput="filterTable('heard-body', this.value)">
  <table>
    <thead><tr><th>Callsign</th><th>Band</th><th>Decodes</th><th style="min-width:160px">Avg SNR</th><th>Mode</th><th>Grid</th><th>Window</th></tr></thead>
    <tbody id="heard-body">{''.join(heard_rows)}</tbody>
  </table>
  <h2>Mentioned in traffic — not decoded by us</h2>
  <p class="section-note">Heard by others in our network but their signals never reached us. Count = times mentioned in heartbeats/SNR reports.</p>
  <input type="search" placeholder="Filter callsign…" oninput="filterTable('nh-body', this.value)">
  <table>
    <thead><tr><th>Callsign</th><th>Mentions</th><th>Heard by (in our network)</th></tr></thead>
    <tbody id="nh-body">{''.join(not_heard_rows)}</tbody>
  </table>
</div>
<script>
function filterTable(id, q) {{
  q = q.toLowerCase();
  for (const tr of document.getElementById(id).rows)
    tr.style.display = tr.cells[0].textContent.toLowerCase().includes(q) ? '' : 'none';
}}
</script>
</body>
</html>'''


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(f'Usage: {sys.argv[0]} <file.jsonl> [--open]')

    src = sys.argv[1]
    out = os.path.splitext(src)[0] + '.html'
    html = build_report(src)
    with open(out, 'w') as f:
        f.write(html)
    print(f'Wrote {out}')

    if '--open' in sys.argv:
        import subprocess
        subprocess.Popen(['xdg-open', out])
