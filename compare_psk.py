#!/usr/bin/env python3
"""compare_psk.py — Compare FT8 decoder output against PSKReporter.

Capture your decoder's stdout, then run:
  python3 compare_psk.py messages.txt
  python3 compare_psk.py messages.txt --grid DM78 --max-dist 1500
  python3 compare_psk.py messages.txt --period 7200
"""
import re
import sys
import math
import argparse
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from collections import defaultdict

CALLSIGN_RE = re.compile(r'^[A-Z0-9]{1,3}[0-9][A-Z]{1,4}(/[A-Z0-9]{1,5})?$')
GRID_RE     = re.compile(r'^[A-R]{2}[0-9]{2}([A-Z]{2})?([0-9]{2})?$')
SKIP        = {'CQ', 'DX', 'RR73', 'RRR', '73', 'POTA', 'SOTA', 'B4', 'DE'}
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
    """Return (lat, lon) center of a Maidenhead grid square (2, 4, or 6 chars)."""
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


def parse_file(path):
    """Return (callsigns_set, first_unix_ts, last_unix_ts, call_freqs, call_snrs).

    call_freqs: callsign -> list of valid freq_hz (>= 1 MHz, i.e. not fallback bins).
    call_snrs:  callsign -> list of snr_dB values.
    """
    calls = set()
    timestamps = []
    call_freqs = defaultdict(list)
    call_snrs  = defaultdict(list)
    with open(path) as f:
        for line in f:
            m = re.match(r'DECODED:\s+(.+?)\s+time_offset=', line)
            if not m:
                continue
            ts_m = re.search(r'\bunix=(\d+\.?\d*)', line)
            if ts_m:
                timestamps.append(float(ts_m.group(1)))
            freq_m = re.search(r'\bfreq=(\d+\.?\d*)Hz', line)
            freq_hz = float(freq_m.group(1)) if freq_m else None
            snr_m = re.search(r'\bsnr=([+-]?\d+\.?\d*)', line)
            snr_db = float(snr_m.group(1)) if snr_m else None
            # Frequencies below 1 MHz are raw composite bin fallbacks — ignore them
            valid_freq = freq_hz if (freq_hz is not None and freq_hz >= 1_000_000) else None
            for word in m.group(1).split():
                word = word.strip('<>').strip()
                if is_callsign(word):
                    calls.add(word)
                    if valid_freq is not None:
                        call_freqs[word].append(valid_freq)
                    if snr_db is not None:
                        call_snrs[word].append(snr_db)
    # Fall back to WSJT-X style timestamp on any line if unix= absent
    if not timestamps:
        with open(path) as f:
            for line in f:
                for ts_m in re.finditer(r'\+(\d{10}\.\d)', line):
                    timestamps.append(float(ts_m.group(1)))
    first = min(timestamps) if timestamps else None
    last  = max(timestamps) if timestamps else None
    return calls, first, last, call_freqs, call_snrs


def fetch_psk(grid, period, mode='FT8'):
    url = (
        f'https://retrieve.pskreporter.info/query'
        f'?rronly=true&mode={mode}&period={period}&receiverGrid={grid}'
    )
    print(f'  {url}', file=sys.stderr)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'ft8-compare/1.0'})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
    except urllib.error.URLError as e:
        sys.exit(f'PSKReporter fetch failed: {e}')

    root = ET.fromstring(data)
    # sender -> list of (receiver_call, rx_grid, freq_hz, snr_db, sender_grid)
    reports = defaultdict(list)
    for rr in root.iter('receptionReport'):
        sender = rr.get('senderCallsign', '')
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
    """Best-SNR report from a receiver within max_dist_km.

    Returns (report, rx_dist_km) or (None, None) if no receiver is close enough.
    """
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
    """Distance from user to sender, using senderLocator from any report."""
    for r in reports:
        pos = grid_to_latlon(r[4])
        if pos:
            return haversine_km(my_lat, my_lon, pos[0], pos[1])
    return None


def main():
    ap = argparse.ArgumentParser(description='Compare FT8 decoder vs PSKReporter')
    ap.add_argument('file', help='Decoder output file (stdout captured to file)')
    ap.add_argument('--period', type=int, default=0,
                    help='PSKReporter lookback in seconds (default: auto-detect from file)')
    ap.add_argument('--grid', default='DM',
                    help='Your grid square, also used as PSKReporter query prefix (default: DM)')
    ap.add_argument('--max-dist', type=int, default=2000,
                    help='Only count signals missed if a PSKReporter receiver within this many km heard it (default: 2000)')
    args = ap.parse_args()

    my_pos = grid_to_latlon(args.grid)
    if my_pos is None:
        sys.exit(f'Could not parse grid square: {args.grid}')
    my_lat, my_lon = my_pos
    print(f'Your position: {my_lat:.1f}°N {my_lon:.1f}°E  (grid {args.grid})', file=sys.stderr)

    print(f'Parsing {args.file} ...', file=sys.stderr)
    your_calls, first_ts, last_ts, call_freqs, call_snrs = parse_file(args.file)

    if not your_calls:
        sys.exit('No callsigns found — check the file path and format.')

    period = args.period
    if not period:
        if first_ts and last_ts:
            period = min(int(last_ts - first_ts) + 300, 86400)
        else:
            period = 3600
    period = max(period, 900)

    span_min = int((last_ts - first_ts) / 60) if (first_ts and last_ts) else 0

    print(f'Fetching PSKReporter ({args.grid}* receivers, {period // 60} min window) ...', file=sys.stderr)
    psk = fetch_psk(args.grid, period)
    psk_calls = set(psk.keys())

    both     = sorted(your_calls & psk_calls)
    you_only = sorted(your_calls - psk_calls)

    # "You missed" = heard by PSKReporter but not by you,
    # AND at least one receiving station is within max_dist_km of you.
    # Sorted by how far the SENDER is from you — nearest senders first,
    # because those are the signals you most likely should have decoded.
    psk_only_nearby = []
    for call in psk_calls - your_calls:
        r, rx_dist = nearby_best(psk[call], my_lat, my_lon, args.max_dist)
        if r is None:
            continue
        tx_dist = sender_dist_km(psk[call], my_lat, my_lon)
        psk_only_nearby.append((call, r, rx_dist, tx_dist))
    psk_only_nearby.sort(key=lambda x: x[3] if x[3] is not None else 99999)

    hr = '─' * 66
    print(f'\n{hr}')
    print(f'FT8 DECODER vs PSKReporter  —  {args.grid}* receiver area')
    print(f'File span: {span_min} min  |  PSK window: {period // 60} min  |  max dist: {args.max_dist} km')
    print(hr)
    print(f'  Your decoder:                    {len(your_calls):4d} unique callsigns')
    print(f'  PSKReporter ({args.grid}*):             {len(psk_calls):4d} unique callsigns')
    pct = 100 * len(both) // len(your_calls) if your_calls else 0
    print(f'  Confirmed in both:               {len(both):4d}  ({pct}% of yours)')
    print(f'  You decoded, PSK missed:         {len(you_only):4d}')
    print(f'  PSK heard, you missed (<{args.max_dist}km):  {len(psk_only_nearby):4d}  (sorted by sender distance from you)')

    print(f'\n── CONFIRMED IN BOTH ({len(both)}) ──────────────────────────────────')
    print(f'  {"Call":<12}  {"Band":<5}  {"Your freq":>10}  {"PSK freq":>10}  {"Delta":>7}  {"My SNR":>6}  {"PSK SNR":>7}  dist    via')
    for call in both:
        r, dist = nearby_best(psk[call], my_lat, my_lon, args.max_dist)
        if r is None:
            r = max(psk[call], key=snr_val)
            dist_str = '  ???km'
        else:
            dist_str = f'{dist:5.0f}km'
        b = band(r[2])
        psk_freq = r[2]

        my_freqs = call_freqs.get(call, [])
        if my_freqs:
            my_freq = round(sum(my_freqs) / len(my_freqs))
            delta = my_freq - psk_freq
            my_freq_str = f'{my_freq:10d}'
            delta_str   = f'{delta:+7d}'
        else:
            my_freq_str = f'{"?":>10}'
            delta_str   = f'{"?":>7}'

        my_snrs = call_snrs.get(call, [])
        my_snr_str = f'{sum(my_snrs)/len(my_snrs):+6.1f}' if my_snrs else f'{"?":>6}'

        print(f'  {call:<12}  {b:<5}  {my_freq_str}  {psk_freq:10d}  {delta_str}  {my_snr_str}  {r[3]:>4} dB  {dist_str}  via {r[0]} ({r[1]})')

    print(f'\n── YOU DECODED, PSK MISSED ({len(you_only)}) ────────────────────────────')
    print(  '  (signals you received that no PSK station in your area reported)')
    for call in you_only:
        print(f'  {call}')

    print(f'\n── PSK HEARD, YOU MISSED — within {args.max_dist} km ({len(psk_only_nearby)} stations) ──')
    print(  '  Sorted by tx (sender distance from you) — nearest = most likely decoder failure')
    print(  '  SNR is what the nearby PSKReporter station measured')
    for call, r, rx_dist, tx_dist in psk_only_nearby[:60]:
        b = band(r[2])
        tx_str = f'{tx_dist:5.0f}km' if tx_dist is not None else '  ???km'
        rx_str = f'{rx_dist:5.0f}km' if rx_dist is not None else '  ???km'
        print(f'  {call:<12}  {b}  SNR {r[3]:>4} dB  tx={tx_str}  rx={rx_str}  via {r[0]} ({r[1]})')
    if len(psk_only_nearby) > 60:
        print(f'  ... and {len(psk_only_nearby) - 60} more')


if __name__ == '__main__':
    main()
