#!/usr/bin/env python3
"""
check_psk_freqs.py — Fetch live FT8/JS8 frequencies from PSKReporter and
verify they fall within granolasdr's hf_bands.h windows.

Usage:
    python3 check_psk_freqs.py
    python3 check_psk_freqs.py --grid DM78 --period 3600
    python3 check_psk_freqs.py --period 7200 --min-reports 5
"""

import argparse
import sys
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from collections import defaultdict

# Current hf_bands.h windows (wb_start, wb_end) in 100-Hz bins → Hz
WINDOWS = [
    ( 18390*100,  18460*100, "160m"),
    ( 35720*100,  35820*100,  "80m"),
    ( 53560*100,  53740*100,  "60m"),
    ( 70680*100,  70820*100,  "40m"),
    (101290*100, 101420*100,  "30m"),
    (140700*100, 141000*100,  "20m"),
    (180700*100, 181130*100,  "17m"),
    (210720*100, 210950*100,  "15m"),
    (249140*100, 249310*100,  "12m"),
    (280730*100, 281000*100,  "10m"),
    (503110*100, 503200*100,   "6m"),
]


def in_window(freq_hz):
    for lo, hi, name in WINDOWS:
        if lo <= freq_hz < hi:
            return name
    return None


def fetch_psk(mode, period, grid=None):
    url = f'https://retrieve.pskreporter.info/query?mode={mode}&period={period}'
    if grid:
        url += f'&receiverGrid={grid}'
    print(f'  Fetching {mode}: {url}', file=sys.stderr)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'granolasdr-check/1.0'})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
    except urllib.error.URLError as e:
        print(f'  PSKReporter fetch failed for {mode}: {e}', file=sys.stderr)
        return {}

    # freq_hz -> report count
    freqs = defaultdict(int)
    root = ET.fromstring(data)
    for rr in root.iter('receptionReport'):
        try:
            freq_hz = int(rr.get('frequency', 0))
        except ValueError:
            continue
        if freq_hz >= 1_000_000:
            freqs[freq_hz] += 1
    return freqs


def check_coverage(freqs, mode, min_reports):
    in_win = defaultdict(lambda: defaultdict(int))   # band -> freq -> count
    out_win = defaultdict(int)                        # freq -> count

    for freq_hz, count in freqs.items():
        if count < min_reports:
            continue
        band = in_window(freq_hz)
        if band:
            in_win[band][freq_hz] += count
        else:
            out_win[freq_hz] += count

    total_in  = sum(sum(v.values()) for v in in_win.values())
    total_out = sum(out_win.values())
    total     = total_in + total_out

    print(f'\n=== {mode} coverage ({total} reports, min {min_reports}/freq) ===')
    for lo, hi, name in WINDOWS:
        band_counts = in_win.get(name, {})
        n = sum(band_counts.values())
        if n:
            flo = min(band_counts)
            fhi = max(band_counts)
            print(f'  {name:4s}: {n:6d} reports  {flo:>11d}-{fhi:<11d} Hz  [window {lo}-{hi}]')
        else:
            print(f'  {name:4s}:      0 reports  [window {lo}-{hi}]')

    if out_win:
        print(f'\n  OUTSIDE windows: {total_out} reports ({100*total_out/total:.1f}%)')
        # Cluster by MHz
        clusters = defaultdict(lambda: defaultdict(int))
        for freq_hz, count in out_win.items():
            clusters[freq_hz // 1_000_000][freq_hz] += count
        for mhz in sorted(clusters):
            freqs_here = clusters[mhz]
            total_here = sum(freqs_here.values())
            flo, fhi = min(freqs_here), max(freqs_here)
            # find nearest window
            nearest = min(WINDOWS, key=lambda w: min(abs(w[0]-flo), abs(w[1]-fhi)))
            print(f'  {mhz}MHz: {total_here:5d} reports  {flo}-{fhi} Hz')
            print(f'         nearest window: {nearest[2]} [{nearest[0]}-{nearest[1]}]')
            # Show top frequencies
            top = sorted(freqs_here.items(), key=lambda x: -x[1])[:5]
            for f, c in top:
                print(f'         {f:>11d} Hz  {c:4d} reports')
    else:
        print(f'\n  All {mode} frequencies covered. ✓')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grid', default='',
                    help='Receiver grid prefix to filter by (e.g. DM78, DM, blank=global)')
    ap.add_argument('--period', type=int, default=3600,
                    help='PSKReporter lookback seconds (default 3600)')
    ap.add_argument('--min-reports', type=int, default=2,
                    help='Min reports per frequency to include (filters noise, default 2)')
    args = ap.parse_args()

    grid = args.grid or None

    ft8_freqs = fetch_psk('FT8', args.period, grid)
    js8_freqs = fetch_psk('JS8', args.period, grid)

    check_coverage(ft8_freqs, 'FT8', args.min_reports)
    check_coverage(js8_freqs, 'JS8', args.min_reports)


if __name__ == '__main__':
    main()
