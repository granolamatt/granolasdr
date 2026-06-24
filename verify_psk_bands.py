#!/usr/bin/env python3
"""
verify_psk_bands.py — Cross-reference what PSKReporter received from us against
our local ft8_timing.csv decode log to confirm no band mismatches.

For each callsign PSKReporter credits to our receiver:
  • Find the matching local decode in ft8_timing.csv
  • Compare our reported frequency against PSKReporter's frequency
  • Flag any band mismatches (same callsign, different band)

Usage:
    python3 verify_psk_bands.py --call KF0RRR
    python3 verify_psk_bands.py --call KF0RRR --period 3600
    python3 verify_psk_bands.py --call KF0RRR --csv ft8_timing.csv --period 7200
"""

import argparse
import csv
import sys
import urllib.request
import urllib.error
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timezone

PSK_URL = "https://retrieve.pskreporter.info/query"

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
    (50_000_000, 54_000_000,  '6m'),
]

def freq_to_band(hz):
    for lo, hi, name in BANDS:
        if lo <= hz <= hi:
            return name
    return "??"


def fetch_psk(callsign, period):
    """Return list of dicts for all reception reports where we are the receiver."""
    url = (f"{PSK_URL}?receiverCallsign={callsign}"
           f"&rronly=true&period={period}")
    print(f"Querying PSKReporter: {url}", file=sys.stderr)
    req = urllib.request.Request(url, headers={"User-Agent": "granolasdr/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except urllib.error.URLError as e:
        sys.exit(f"PSKReporter fetch failed: {e}")

    reports = []
    root = ET.fromstring(data)
    for rr in root.iter("receptionReport"):
        freq = int(rr.get("frequency", 0))
        t    = int(rr.get("flowStartSeconds", 0))
        reports.append({
            "sender": rr.get("senderCallsign", "").upper(),
            "freq":   freq,
            "band":   freq_to_band(freq),
            "snr":    rr.get("sNR", "?"),
            "time":   t,
        })
    return reports


def load_csv(path, start_unix=None, end_unix=None):
    """Return dict: callsign -> list of (freq_hz, snr, unix_ts) from ft8_timing.csv."""
    decoded = defaultdict(list)
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts  = float(row["wall_clock"])
                    freq = float(row["freq_hz"])
                    snr  = float(row["snr"])
                    call = row["callsign"].strip().split()[0].upper()
                except (KeyError, ValueError):
                    continue
                if freq < 1_000_000:
                    continue
                if start_unix and ts < start_unix:
                    continue
                if end_unix and ts > end_unix:
                    continue
                decoded[call].append((int(freq), snr, ts))
    except FileNotFoundError:
        print(f"Warning: {path} not found — skipping local CSV cross-check", file=sys.stderr)
    return decoded


def main():
    ap = argparse.ArgumentParser(description="Verify PSKReporter band reports match local decodes")
    ap.add_argument("--call",   required=True, help="Your receiver callsign (e.g. KF0RRR)")
    ap.add_argument("--period", type=int, default=1800,
                    help="PSKReporter lookback in seconds (default 1800)")
    ap.add_argument("--csv",    default="ft8_timing.csv",
                    help="Local FT8 decode log (default: ft8_timing.csv)")
    args = ap.parse_args()

    rx_call = args.call.upper()
    psk = fetch_psk(rx_call, args.period)
    if not psk:
        print(f"\nNo PSKReporter records found for {rx_call} in the last "
              f"{args.period//60} min.")
        return

    oldest = min(r["time"] for r in psk)
    newest = max(r["time"] for r in psk)
    decoded = load_csv(args.csv, oldest - 60, newest + 60)

    # Group PSKReporter reports by sender callsign
    psk_by_call = defaultdict(list)
    for r in psk:
        psk_by_call[r["sender"]].append(r)

    # Build summary
    band_ok   = defaultdict(int)
    band_miss = defaultdict(int)   # in PSKReporter but not in local CSV
    mismatches = []

    for call, reports in sorted(psk_by_call.items()):
        psk_freqs = [r["freq"] for r in reports]
        avg_psk   = round(sum(psk_freqs) / len(psk_freqs))
        psk_band  = freq_to_band(avg_psk)

        local = decoded.get(call, [])
        if not local:
            band_miss[psk_band] += 1
            continue

        # Find the local decode closest in frequency to what PSKReporter has.
        # This avoids false-positive decodes at garbage frequencies skewing the comparison.
        closest_freq = min((f for f, _, _ in local), key=lambda f: abs(f - avg_psk))
        local_band   = freq_to_band(closest_freq)

        if local_band == psk_band:
            band_ok[psk_band] += 1
        else:
            mismatches.append((call, psk_band, avg_psk, local_band, closest_freq))

    # ── Report ─────────────────────────────────────────────────────────────
    oldest_dt = datetime.fromtimestamp(oldest, tz=timezone.utc)
    newest_dt = datetime.fromtimestamp(newest, tz=timezone.utc)
    hr = "─" * 70

    print(f"\n{hr}")
    print(f"PSKReporter band verification  —  receiver {rx_call}")
    print(f"Window: {oldest_dt.strftime('%H:%M')}–{newest_dt.strftime('%H:%M')} UTC  "
          f"({args.period//60} min)  {len(psk)} total reports  {len(psk_by_call)} senders")
    print(hr)

    if mismatches:
        print(f"\n{'BAND MISMATCHES':}")
        print(f"  {'Callsign':<14} {'PSK band':<8} {'PSK freq':>10}  {'Local band':<10} {'Local freq':>10}  Note")
        for call, pb, pf, lb, lf in sorted(mismatches):
            delta = lf - pf
            print(f"  {call:<14} {pb:<8} {pf:>10}  {lb:<10} {lf:>10}  delta={delta:+d} Hz")
        print()
    else:
        print(f"\n  No band mismatches — all PSKReporter-credited callsigns match local decode bands ✓")

    print(f"\n{'Band':<6}  {'Confirmed':>9}  {'No local match':>14}  {'% matched':>9}")
    print("─" * 46)
    all_bands = sorted(set(list(band_ok.keys()) + list(band_miss.keys())))
    for b in all_bands:
        ok   = band_ok.get(b, 0)
        miss = band_miss.get(b, 0)
        total = ok + miss
        pct = f"{100*ok//total}%" if total else "—"
        print(f"{b:<6}  {ok:>9}  {miss:>14}  {pct:>9}")

    if not decoded:
        print(f"\n  (No local CSV data — band mismatch check skipped; "
              f"run with --csv to point at ft8_timing.csv)")


if __name__ == "__main__":
    main()
