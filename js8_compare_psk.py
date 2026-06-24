#!/usr/bin/env python3
"""js8_compare_psk.py — Compare granolasdr JS8 decodes against PSKReporter.

Reads js8_YYYYMMDD.jsonl files produced by js8_logger.py, then fetches
PSKReporter JS8 reception reports for nearby receivers and shows what we missed.

Usage:
    python3 js8_compare_psk.py js8_20260613.jsonl
    python3 js8_compare_psk.py js8_*.jsonl --grid DM78 --max-dist 2000
    python3 js8_compare_psk.py js8_20260613.jsonl --period 7200
"""
import json
import math
import re
import sys
import argparse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from collections import defaultdict

CALLSIGN_RE = re.compile(r'^[A-Z0-9]{1,3}[0-9][A-Z]{1,4}(/[A-Z0-9]{1,5})?$')
GRID_RE     = re.compile(r'^[A-R]{2}[0-9]{2}([A-Z]{2})?([0-9]{2})?$')
SKIP        = {'CQ', 'DX', 'RR73', 'RRR', '73', 'POTA', 'SOTA', 'B4', 'DE', 'QRZ', 'TU;'}
REPORT_RE   = re.compile(r'^R?[+-][0-9]+$')

BANDS = [
    (1_800_000,  2_000_000, '160m'),
    (3_500_000,  4_000_000,  '80m'),
    (5_330_000,  5_410_000,  '60m'),
    (7_000_000,  7_300_000,  '40m'),
    (10_100_000, 10_150_000, '30m'),
    (14_000_000, 14_350_000, '20m'),
    (18_068_000, 18_168_000, '17m'),
    (21_000_000, 21_450_000, '15m'),
    (24_890_000, 24_990_000, '12m'),
    (28_000_000, 29_700_000, '10m'),
    (50_000_000, 54_000_000,   '6m'),
]


def is_callsign(s):
    if s in SKIP or REPORT_RE.match(s) or GRID_RE.match(s):
        return False
    return bool(CALLSIGN_RE.match(s))


def band(hz):
    for lo, hi, name in BANDS:
        if lo <= hz <= hi:
            return name
    return '??m'


def grid_to_latlon(grid):
    g = grid.upper().strip()
    if len(g) < 2 or not g[0].isalpha() or not g[1].isalpha():
        return None
    try:
        lon = (ord(g[0]) - ord('A')) * 20 - 180
        lat = (ord(g[1]) - ord('A')) * 10 - 90
        if len(g) >= 4 and g[2].isdigit() and g[3].isdigit():
            lon += int(g[2]) * 2
            lat += int(g[3])
        if len(g) >= 6 and g[4].isalpha() and g[5].isalpha():
            lon += (ord(g[4]) - ord('A')) * (5 / 60)
            lat += (ord(g[5]) - ord('A')) * (2.5 / 60)
            lon += 5 / 60 / 2
            lat += 2.5 / 60 / 2
        elif len(g) >= 4:
            lon += 1.0
            lat += 0.5
        else:
            lon += 10.0
            lat += 5.0
        return lat, lon
    except Exception:
        return None


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(min(1.0, a)))


def parse_jsonl(paths):
    """Read js8_YYYYMMDD.jsonl files; return (call_set, first_unix, last_unix, call_freqs, call_snrs)."""
    calls = set()
    timestamps = []
    call_freqs = defaultdict(list)
    call_snrs  = defaultdict(list)
    for path in paths:
        try:
            f = open(path)
        except FileNotFoundError:
            print(f'  Warning: {path} not found — skipped', file=sys.stderr)
            continue
        with f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                call = msg.get('call', '').strip().upper()
                freq = msg.get('freq', 0)
                snr  = msg.get('snr', None)
                unix = msg.get('unix', None)
                if not call or not is_callsign(call):
                    continue
                calls.add(call)
                if unix:
                    timestamps.append(float(unix))
                if freq and freq >= 1_000_000:
                    call_freqs[call].append(int(freq))
                if snr is not None:
                    call_snrs[call].append(float(snr))
    first = min(timestamps) if timestamps else None
    last  = max(timestamps) if timestamps else None
    return calls, first, last, call_freqs, call_snrs


def fetch_psk(grid, period):
    """Fetch PSKReporter JS8 reception reports for receiver grid prefix."""
    url = (
        'https://retrieve.pskreporter.info/query'
        f'?rronly=true&mode=JS8&period={period}&receiverGrid={grid}'
    )
    print(f'  {url}', file=sys.stderr)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'granolasdr-js8/1.0'})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
    except urllib.error.URLError as e:
        sys.exit(f'PSKReporter fetch failed: {e}')

    root = ET.fromstring(data)
    # sender -> list of (receiver_call, rx_grid, freq_hz, snr_db, sender_grid)
    reports = defaultdict(list)
    for rr in root.iter('receptionReport'):
        sender = rr.get('senderCallsign', '').upper()
        if sender:
            try:
                freq_hz = int(rr.get('frequency', 0))
            except ValueError:
                freq_hz = 0
            reports[sender].append((
                rr.get('receiverCallsign', ''),
                rr.get('receiverLocator', ''),
                freq_hz,
                rr.get('sNR', '?'),
                rr.get('senderLocator', ''),
            ))
    return reports


def snr_val(r):
    try:
        return int(r[3])
    except (ValueError, TypeError):
        return -99


def nearby_best(reports, my_lat, my_lon, max_dist_km):
    best_r = best_snr = best_dist = None
    for r in reports:
        pos = grid_to_latlon(r[1])
        if pos is None:
            continue
        dist = haversine_km(my_lat, my_lon, pos[0], pos[1])
        if dist > max_dist_km:
            continue
        snr = snr_val(r)
        if best_r is None or snr > best_snr:
            best_r, best_snr, best_dist = r, snr, dist
    return best_r, best_dist


def sender_dist_km(reports, my_lat, my_lon):
    for r in reports:
        pos = grid_to_latlon(r[4])
        if pos:
            return haversine_km(my_lat, my_lon, pos[0], pos[1])
    return None


def main():
    ap = argparse.ArgumentParser(description='Compare granolasdr JS8 decodes vs PSKReporter')
    ap.add_argument('files', nargs='+', help='js8_YYYYMMDD.jsonl file(s) from js8_logger.py')
    ap.add_argument('--period', type=int, default=0,
                    help='PSKReporter lookback in seconds (default: auto from file timestamps)')
    ap.add_argument('--grid', default='DM78',
                    help='Your grid square, used as PSKReporter query prefix (default: DM78)')
    ap.add_argument('--max-dist', type=int, default=2000,
                    help='Count missed only if a nearby receiver within this many km heard it (default: 2000)')
    args = ap.parse_args()

    my_pos = grid_to_latlon(args.grid)
    if my_pos is None:
        sys.exit(f'Could not parse grid square: {args.grid}')
    my_lat, my_lon = my_pos
    print(f'Your position: {my_lat:.1f}°N {my_lon:.1f}°E  (grid {args.grid})', file=sys.stderr)

    print(f'Parsing {len(args.files)} file(s) ...', file=sys.stderr)
    your_calls, first_ts, last_ts, call_freqs, call_snrs = parse_jsonl(args.files)

    if not your_calls:
        sys.exit(
            'No JS8 callsigns found in input files.\n'
            'Make sure js8_logger.py is running and has collected some data:\n'
            '  python3 js8_logger.py --dir .'
        )

    period = args.period
    if not period:
        if first_ts and last_ts:
            period = min(int(last_ts - first_ts) + 300, 86400)
        else:
            period = 3600
    period = max(period, 900)

    span_min = int((last_ts - first_ts) / 60) if (first_ts and last_ts) else 0

    # PSKReporter grid query uses only the first 2 chars (field) of the grid
    psk_grid_prefix = args.grid[:2].upper()
    print(f'Fetching PSKReporter JS8 ({psk_grid_prefix}* receivers, {period // 60} min window) ...',
          file=sys.stderr)
    psk = fetch_psk(psk_grid_prefix, period)
    psk_calls = set(psk.keys())

    both     = sorted(your_calls & psk_calls)
    you_only = sorted(your_calls - psk_calls)

    psk_only_nearby = []
    for call in psk_calls - your_calls:
        r, rx_dist = nearby_best(psk[call], my_lat, my_lon, args.max_dist)
        if r is None:
            continue
        tx_dist = sender_dist_km(psk[call], my_lat, my_lon)
        psk_only_nearby.append((call, r, rx_dist, tx_dist))
    psk_only_nearby.sort(key=lambda x: x[3] if x[3] is not None else 99999)

    hr = '─' * 68
    print(f'\n{hr}')
    print(f'JS8 DECODER vs PSKReporter  —  {args.grid} receiver area  (KF0RRR)')
    print(f'File span: {span_min} min  |  PSK window: {period // 60} min  |  max dist: {args.max_dist} km')
    print(hr)
    print(f'  Our decoder (JS8):               {len(your_calls):4d} unique callsigns')
    print(f'  PSKReporter JS8 ({psk_grid_prefix}*):         {len(psk_calls):4d} unique callsigns')
    pct = 100 * len(both) // len(your_calls) if your_calls else 0
    print(f'  Confirmed in both:               {len(both):4d}  ({pct}% of ours)')
    print(f'  We decoded, PSK missed:          {len(you_only):4d}')
    print(f'  PSK heard, we missed (<{args.max_dist}km):  {len(psk_only_nearby):4d}  (sorted by sender distance)')

    print(f'\n── CONFIRMED IN BOTH ({len(both)}) ──────────────────────────────────')
    print(f'  {"Call":<12}  {"Band":<5}  {"Our freq":>10}  {"PSK freq":>10}  {"Delta":>7}  {"Our SNR":>7}  {"PSK SNR":>7}  dist    via')
    for call in both:
        r, dist = nearby_best(psk[call], my_lat, my_lon, args.max_dist)
        if r is None:
            r = max(psk[call], key=snr_val)
            dist_str = '  ???km'
        else:
            dist_str = f'{dist:5.0f}km'
        b = band(r[2])
        psk_freq = r[2]

        all_freqs = call_freqs.get(call, [])
        same_band_freqs = [f for f in all_freqs if band(f) == b]
        my_freqs = same_band_freqs if same_band_freqs else all_freqs
        if my_freqs:
            my_freq = round(sum(my_freqs) / len(my_freqs))
            delta = my_freq - psk_freq
            my_freq_str = f'{my_freq:10d}'
            delta_str   = f'{delta:+7d}' if same_band_freqs else f'{"~band":>7}'
        else:
            my_freq_str = f'{"?":>10}'
            delta_str   = f'{"?":>7}'

        my_snrs = call_snrs.get(call, [])
        my_snr_str = f'{sum(my_snrs)/len(my_snrs):+7.1f}' if my_snrs else f'{"?":>7}'

        print(f'  {call:<12}  {b:<5}  {my_freq_str}  {psk_freq:10d}  {delta_str}  {my_snr_str}  {r[3]:>4} dB  {dist_str}  via {r[0]} ({r[1]})')

    print(f'\n── WE DECODED, PSK MISSED ({len(you_only)}) ────────────────────────────')
    print(  '  (signals we received that no nearby PSK JS8 station reported)')
    for call in you_only:
        freqs = call_freqs.get(call, [])
        b = band(round(sum(freqs)/len(freqs))) if freqs else '??m'
        snrs = call_snrs.get(call, [])
        snr_str = f'{sum(snrs)/len(snrs):+.1f} dB' if snrs else '?'
        print(f'  {call:<12}  {b}  {snr_str}')

    print(f'\n── PSK HEARD, WE MISSED — within {args.max_dist} km ({len(psk_only_nearby)} stations) ──')
    print(  '  Sorted by tx (sender distance from us) — nearest = most likely decoder failure')
    print(  '  SNR is what the nearby PSKReporter station measured')
    for call, r, rx_dist, tx_dist in psk_only_nearby[:80]:
        b = band(r[2])
        tx_str = f'{tx_dist:5.0f}km' if tx_dist is not None else '  ???km'
        rx_str = f'{rx_dist:5.0f}km' if rx_dist is not None else '  ???km'
        print(f'  {call:<12}  {b}  SNR {r[3]:>4} dB  tx={tx_str}  rx={rx_str}  via {r[0]} ({r[1]})')
    if len(psk_only_nearby) > 80:
        print(f'  ... and {len(psk_only_nearby) - 80} more')

    # Band breakdown of misses
    miss_by_band = defaultdict(int)
    hit_by_band  = defaultdict(int)
    for call, r, _, _ in psk_only_nearby:
        miss_by_band[band(r[2])] += 1
    for call in both:
        reps = psk[call]
        if reps:
            hit_by_band[band(reps[0][2])] += 1

    print(f'\n── MISSES BY BAND ──────────────────────────────────────')
    all_bands = sorted(set(list(miss_by_band.keys()) + list(hit_by_band.keys())))
    for b in all_bands:
        miss  = miss_by_band.get(b, 0)
        hit   = hit_by_band.get(b, 0)
        total = miss + hit
        pct_b = 100 * hit // total if total else 0
        bar   = '#' * (hit * 30 // max(total, 1))
        print(f'  {b:>5}  hit={hit:3d}  miss={miss:3d}  {pct_b:3d}%  {bar}')


if __name__ == '__main__':
    main()
