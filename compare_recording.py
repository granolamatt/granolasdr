#!/usr/bin/env python3
"""compare_recording.py — Compare hf_rx decoder log against PSKReporter ground truth JSON.

Usage:
  python3 compare_recording.py ~/hfrx.log ~/psk_groundtruth_8h.json
  python3 compare_recording.py ~/playback_new.log ~/psk_groundtruth_8h.json
  python3 compare_recording.py ~/hfrx.log ~/psk_groundtruth_8h.json --band 20m
"""
import re
import sys
import json
import argparse
import calendar
import datetime
from collections import defaultdict

CALLSIGN_RE = re.compile(r'^[A-Z0-9]{1,3}[0-9][A-Z]{1,4}(/[A-Z0-9]{1,5})?$')
SKIP = {'CQ', 'DX', 'RR73', 'RRR', '73', 'POTA', 'SOTA', 'B4', 'DE', 'QRZ', 'TU;'}
REPORT_RE = re.compile(r'^R?[+-][0-9]+$')
GRID_RE = re.compile(r'^[A-R]{2}[0-9]{2}([A-Z]{2})?([0-9]{2})?$')

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


def freq_to_band(hz):
    for lo, hi, name in BANDS:
        if lo <= hz <= hi:
            return name
    return '??m'


def parse_log(path, start_unix=None, end_unix=None, filter_band=None):
    """Return dict: (call, band) -> list of (freq_hz, snr, unix_ts)."""
    results = defaultdict(list)
    with open(path) as f:
        for line in f:
            if 'DECODED:' not in line:
                continue
            m = re.match(r'DECODED:\s+(.+?)\s+time_offset=', line)
            if not m:
                continue
            ts_m = re.search(r'\bunix=(\d+\.?\d*)', line)
            if ts_m:
                unix_ts = float(ts_m.group(1))
            else:
                unix_ts = None
            if start_unix is not None and unix_ts is not None and unix_ts < start_unix:
                continue
            if end_unix is not None and unix_ts is not None and unix_ts > end_unix:
                continue

            freq_m = re.search(r'\bfreq=(\d+\.?\d*)Hz', line)
            freq_hz = int(float(freq_m.group(1))) if freq_m else 0
            snr_m = re.search(r'\bsnr=([+-]?\d+\.?\d*)', line)
            snr = float(snr_m.group(1)) if snr_m else None

            b = freq_to_band(freq_hz)
            if filter_band and b != filter_band:
                continue

            for word in m.group(1).split():
                word = word.strip('<>').strip()
                if is_callsign(word):
                    results[(word, b)].append((freq_hz, snr, unix_ts))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log', help='hf_rx decoder log file')
    ap.add_argument('gt', help='PSKReporter ground truth JSON (psk_groundtruth_8h.json)')
    ap.add_argument('--band', help='Filter to one band (e.g. 20m)')
    ap.add_argument('--top', type=int, default=40, help='Max missed to show per section')
    args = ap.parse_args()

    with open(args.gt) as f:
        gt = json.load(f)

    generated = gt['generated_utc']
    period_h  = gt['period_hours']
    gen_dt    = datetime.datetime.strptime(generated, '%Y-%m-%dT%H:%M:%SZ')
    start_dt  = gen_dt - datetime.timedelta(hours=period_h)
    start_unix = calendar.timegm(start_dt.timetuple())
    end_unix   = calendar.timegm(gen_dt.timetuple())

    print(f'Ground truth window: {start_dt.strftime("%Y-%m-%d %H:%M")} – '
          f'{gen_dt.strftime("%H:%M")} UTC  ({period_h}h)')
    if args.band:
        print(f'Filtering to band: {args.band}')

    print(f'Parsing {args.log} ...', flush=True)
    decoded = parse_log(args.log, start_unix, end_unix, args.band)
    our_pairs = set(decoded.keys())

    # Ground truth sets: (call, band) tuples
    gt_received_pairs = {(r['call'], r['band']) for r in gt['kf_received']
                         if not args.band or r['band'] == args.band}
    gt_area_pairs     = {(r['call'], r['band']) for r in gt['area_heard']
                         if not args.band or r['band'] == args.band}
    gt_missed_pairs   = {(r['call'], r['band']) for r in gt['missed']
                         if not args.band or r['band'] == args.band}

    gt_missed_by_pair = {(r['call'], r['band']): r for r in gt['missed']}
    gt_area_by_pair   = {(r['call'], r['band']): r for r in gt['area_heard']}

    # Confirmed: we decoded AND PSKReporter credited KF0RRR
    confirmed     = sorted(our_pairs & gt_received_pairs)
    # We decoded but PSK didn't credit us (decode but not uploaded / timing)
    we_only       = sorted(our_pairs - gt_area_pairs)
    # PSK says area heard it, we decoded it too (local hit)
    local_hit     = sorted(our_pairs & gt_area_pairs)
    # PSK says area heard it, we missed it
    local_miss    = sorted(gt_missed_pairs - our_pairs)

    hr = '─' * 70

    print(f'\n{hr}')
    print(f'DECODER vs PSKReporter — KF0RRR  ground truth {period_h}h window')
    print(hr)
    print(f'  Our decoder (window):            {len(our_pairs):4d} unique (call,band) pairs')
    print(f'  KF0RRR credited by PSK:          {len(gt_received_pairs):4d}')
    print(f'  Area (DM78*) heard:              {len(gt_area_pairs):4d}')
    print(f'  PSK missed list (area heard):    {len(gt_missed_pairs):4d}')
    print()
    print(f'  Confirmed (we got PSK credit):   {len(confirmed):4d}')
    pct_area = 100 * len(local_hit) // len(gt_area_pairs) if gt_area_pairs else 0
    print(f'  Local hit (in area, we decoded): {len(local_hit):4d}  ({pct_area}% of area)')
    pct_miss = 100 * len(local_miss) // len(gt_missed_pairs) if gt_missed_pairs else 0
    print(f'  Local miss (area heard, missed): {len(local_miss):4d}  ({pct_miss}% of missed list)')
    print(f'  We decoded, not in area list:    {len(we_only):4d}')

    # Band breakdown of misses
    miss_by_band = defaultdict(int)
    for call, b in local_miss:
        miss_by_band[b] += 1
    hit_by_band = defaultdict(int)
    for call, b in local_hit:
        hit_by_band[b] += 1

    print(f'\n── MISSES BY BAND ──────────────────────────────────')
    all_bands = sorted(set(list(miss_by_band.keys()) + list(hit_by_band.keys())))
    for b in all_bands:
        miss = miss_by_band.get(b, 0)
        hit  = hit_by_band.get(b, 0)
        total = miss + hit
        pct = 100 * hit // total if total else 0
        bar = '#' * (hit * 30 // max(total, 1))
        print(f'  {b:>5}  hit={hit:3d}  miss={miss:3d}  {pct:3d}%  {bar}')

    # Show top missed — sorted by SNR (best SNR = should have decoded)
    def miss_snr(pair):
        r = gt_missed_by_pair.get(pair)
        if r is None:
            return 99
        try:
            return -int(r['snr'])
        except (ValueError, TypeError):
            return 0

    top_miss = sorted(local_miss, key=miss_snr)[:args.top]

    print(f'\n── LOCAL MISS — top {args.top} (area heard, we missed) ──────────────────')
    print(f'  {"Call":<14}  {"Band":<5}  {"Freq":>10}  SNR   Receivers (nearby)')
    for call, b in top_miss:
        r = gt_missed_by_pair.get((call, b), {})
        rxs = ', '.join(r.get('receivers', [])[:3])
        freq = r.get('freq', 0)
        snr  = r.get('snr', '?')
        print(f'  {call:<14}  {b:<5}  {freq:10d}  {str(snr):>4}  {rxs}')

    # Show confirmed breakdown
    print(f'\n── CONFIRMED (we decoded + PSK credited) — {len(confirmed)} ──────────')
    band_conf = defaultdict(int)
    for call, b in confirmed:
        band_conf[b] += 1
    for b, n in sorted(band_conf.items()):
        print(f'  {b}: {n}')


if __name__ == '__main__':
    main()
