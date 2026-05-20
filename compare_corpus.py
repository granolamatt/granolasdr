#!/usr/bin/env python3
"""
compare_corpus.py — compare granolasdr decodes vs JTDX decodes for the same WAV files.

Usage:
    python3 compare_corpus.py GRANOLA_WSJT JTDX_ALL_TXT [--show-both]

GRANOLA_WSJT:  output of: python3 corpus_log.py granola_log.jsonl --wsjt
JTDX_ALL_TXT:  JTDX 202605_ALL.TXT (or WSJT-X ALL.TXT)

Both files use YYYYMMDD_HHMMSS timestamps so no --date flag is needed.
"""

import argparse
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone


def epoch15(unix_t):
    return int(unix_t) // 15 * 15


# JTDX / corpus_log --wsjt: YYYYMMDD_HHMMSS  SNR  DT  FREQ ~ MSG
_JTDX = re.compile(
    r"^(\d{8})_(\d{6})\s+([-+]?\d+)\s+([-+]?\d+\.\d+)\s+(\d+)\s+~\s*(.+?)(?:\s+\*)?$"
)


def parse_log(path):
    decodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = _JTDX.match(line)
            if not m:
                continue
            date_s, time_s, snr_s, dt_s, freq_s, msg = m.groups()
            try:
                unix_t = datetime.strptime(date_s + time_s, "%Y%m%d%H%M%S") \
                                 .replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue

            # Extract primary callsign from message
            tokens = msg.split()
            call = ""
            for tok in tokens:
                if tok in ("CQ", "DE", "QRZ", "73", "RR73", "R"):
                    continue
                if re.match(r"^[A-R]{2}\d{2}[A-X]{0,2}$", tok, re.IGNORECASE):
                    continue  # Maidenhead grid
                if re.match(r"^[-+]\d+$", tok):
                    continue  # report
                if re.match(r"^[A-Z0-9]{2,}(/[A-Z0-9]+)?$", tok):
                    call = tok
                    break
            if not call:
                call = tokens[-1] if tokens else "?"

            decodes.append({
                "call": call.upper(),
                "snr":  int(snr_s),
                "freq": int(freq_s),
                "ep":   epoch15(unix_t),
                "msg":  msg,
            })
    return decodes


def main():
    ap = argparse.ArgumentParser(
        description="Compare granolasdr vs JTDX decode coverage for corpus WAV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("granola_wsjt", help="corpus_log.py --wsjt output")
    ap.add_argument("jtdx_all_txt", help="JTDX 202605_ALL.TXT")
    ap.add_argument("--show-both", action="store_true",
                    help="also print signals both decoders agreed on")
    args = ap.parse_args()

    granola   = parse_log(args.granola_wsjt)
    reference = parse_log(args.jtdx_all_txt)

    if not granola:
        print(f"WARNING: no granolasdr decodes loaded from {args.granola_wsjt}", file=sys.stderr)
    if not reference:
        print(f"WARNING: no JTDX decodes loaded from {args.jtdx_all_txt}", file=sys.stderr)

    g_by_ep = defaultdict(dict)
    r_by_ep = defaultdict(dict)
    for d in granola:
        g_by_ep[d["ep"]][d["call"]] = d
    for d in reference:
        r_by_ep[d["ep"]][d["call"]] = d

    all_epochs = sorted(set(g_by_ep) | set(r_by_ep))

    both_n = granola_only_n = ref_only_n = 0
    rows = []

    for ep in all_epochs:
        g_calls = g_by_ep[ep]
        r_calls = r_by_ep[ep]
        ep_dt = datetime.fromtimestamp(ep, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        for call in sorted(set(g_calls) | set(r_calls)):
            in_g = call in g_calls
            in_r = call in r_calls
            g_snr = f"{g_calls[call]['snr']:+d}" if in_g else "—"
            r_snr = f"{r_calls[call]['snr']:+d}" if in_r else "—"
            if in_g and in_r:
                status = "BOTH"
                both_n += 1
            elif in_g:
                status = "GRANOLA-ONLY"
                granola_only_n += 1
            else:
                status = "JTDX-ONLY"
                ref_only_n += 1
            rows.append((ep_dt, call, g_snr, r_snr, status))

    print(f"{'Epoch (UTC)':<22} {'Call':<12} {'Granola':>8} {'JTDX':>6}  Status")
    print("-" * 66)
    for ep_dt, call, g_snr, r_snr, status in rows:
        if status == "BOTH" and not args.show_both:
            continue
        print(f"{ep_dt:<22} {call:<12} {g_snr:>8} {r_snr:>6}  {status}")

    total = both_n + granola_only_n + ref_only_n
    print()
    print(f"Epochs compared:       {len(all_epochs)}")
    print(f"Unique (call, epoch):  {total}")
    print(f"  Both decoded:        {both_n:4d}  ({100*both_n/max(total,1):.1f}%)")
    print(f"  Granola only:        {granola_only_n:4d}  ({100*granola_only_n/max(total,1):.1f}%)")
    print(f"  JTDX only:           {ref_only_n:4d}  ({100*ref_only_n/max(total,1):.1f}%)")
    jtdx_union = both_n + ref_only_n
    if jtdx_union:
        print(f"  Granola recall:      {100*both_n/jtdx_union:.1f}%  (of signals JTDX heard)")


if __name__ == "__main__":
    main()
