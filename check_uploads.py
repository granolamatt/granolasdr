#!/usr/bin/env python3
"""
check_uploads.py — Verify that your PSKReporter uploads were accepted.

Queries PSKReporter for recent reception reports attributed to YOUR callsign
as the receiver, and shows a summary of what they have on record.

Usage:
    python3 check_uploads.py --call W1AW [--period 1800]
"""

import argparse
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
    (50_000_000, 54_000_000,   '6m'),
]

def band(freq_hz: int) -> str:
    for lo, hi, name in BANDS:
        if lo <= freq_hz <= hi:
            return name
    return "??"


def fetch(callsign: str, period: int) -> list[dict]:
    url = (f"{PSK_URL}?receiverCallsign={callsign}"
           f"&rronly=true&mode=FT8&period={period}")
    print(f"Querying: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "granolasdr/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except urllib.error.URLError as e:
        print(f"Error fetching PSKReporter: {e}")
        return []

    root = ET.fromstring(data)
    reports = []
    for rr in root.iter("receptionReport"):
        freq = int(rr.get("frequency", 0))
        reports.append({
            "sender":   rr.get("senderCallsign", ""),
            "receiver": rr.get("receiverCallsign", ""),
            "freq":     freq,
            "band":     band(freq),
            "snr":      rr.get("sNR", "?"),
            "time":     int(rr.get("flowStartSeconds", 0)),
            "sender_grid": rr.get("senderLocator", ""),
        })
    return reports


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--call",   required=True, help="Your callsign (receiver)")
    ap.add_argument("--period", type=int, default=1800,
                    help="Look-back window in seconds (default 1800 = 30 min)")
    args = ap.parse_args()

    reports = fetch(args.call.upper(), args.period)

    if not reports:
        print(f"\nNo reports found for {args.call.upper()} in the last "
              f"{args.period//60} minutes.")
        print("Either no uploads have been accepted yet, or the window is too short.")
        return

    # Sort by time
    reports.sort(key=lambda r: r["time"])
    oldest = datetime.fromtimestamp(reports[0]["time"],  tz=timezone.utc)
    newest = datetime.fromtimestamp(reports[-1]["time"], tz=timezone.utc)

    print(f"\nPSKReporter has {len(reports)} reception reports for "
          f"{args.call.upper()} in the last {args.period//60} min")
    print(f"  oldest: {oldest.strftime('%H:%M:%S UTC')}")
    print(f"  newest: {newest.strftime('%H:%M:%S UTC')}")

    # Summary by band
    by_band = defaultdict(list)
    for r in reports:
        by_band[r["band"]].append(r)

    print(f"\n{'Band':<6}  {'Reports':>7}  {'Unique senders':>14}")
    print("-" * 32)
    for b, rlist in sorted(by_band.items(), key=lambda x: x[1][0]["freq"]):
        senders = {r["sender"] for r in rlist}
        print(f"{b:<6}  {len(rlist):>7}  {len(senders):>14}")

    # Recent sample
    print(f"\nMost recent 10 reports:")
    print(f"  {'Time(UTC)':<10}  {'Sender':<12}  {'Band':<5}  {'SNR':>4}  Grid")
    print("  " + "-" * 48)
    for r in reports[-10:]:
        t = datetime.fromtimestamp(r["time"], tz=timezone.utc).strftime("%H:%M:%S")
        print(f"  {t:<10}  {r['sender']:<12}  {r['band']:<5}  {r['snr']:>4}  {r['sender_grid']}")


if __name__ == "__main__":
    main()
