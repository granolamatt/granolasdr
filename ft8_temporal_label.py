#!/usr/bin/env python3
"""
Temporal-coherence truth labeler for FT8 reach measurement.

Problem: granolasdr LLR captures have NO ground truth -- a "fail" frame
(BP+OSD both missed) is ambiguous between a real-weak-signal-we-missed (the
reach prize) and noise/not-a-signal. Lowering the capture threshold floods the
log with both.

Oracle: FT8's 15s-cycle repeat structure. A real station transmits on a stable
RF frequency across multiple cycles. So for a fail frame at (cycle c, freq f):

  - SPILLOVER (not a prize): a pass/osd exists at freq~f in the SAME cycle c.
    The slot WAS decoded this cycle; this fail is a redundant candidate bin of
    an already-decoded signal. Excluded.

  - SIGNAL-PRESENT (real, missed): no same-cycle decode at f, but a pass/osd
    exists at freq~f in an ADJACENT cycle (c +/- W). A real recurring station
    was there; BP+OSD missed it THIS cycle. This is the reach prize.

  - KNOWN-CODEWORD (real + truth): an adjacent-cycle decode at freq~f has
    IDENTICAL message text -> the fail frame almost certainly carries that same
    transmission. Re-encoding that text gives a ground-truth 91-bit codeword for
    the fail frame -- a false-accept-immune label for training AND eval.

  - UNLABELED: no anchor at freq~f in the window. Cannot judge -> excluded from
    the reach denominator (never guessed).

Cycle bucketing: int(unix // 15). Freq match: |freq_a - freq_b| <= --freq-tol Hz
(FT8 tone spacing 6.25 Hz; a signal drifts ~1-2 Hz/min). Window: +/- --window
cycles.

Outputs a report and (optionally) a JSONL of KNOWN-CODEWORD fail frames with
their truth text + LLR, for downstream decode attempts.
"""
import argparse
import json
import re
from collections import defaultdict

# A loose FT8 callsign token: optional prefix digit, 1-2 letters, a digit,
# 1-3 trailing alnum ending in a letter. Catches most standard calls; not
# meant to be exhaustive (used only for "same station" consistency).
_CALL_RE = re.compile(r"\b[A-Z0-9]{0,2}[0-9][A-Z]{1,3}\b|\b[A-Z]{1,2}[0-9][A-Z0-9]{0,3}[A-Z]\b")


def callsigns(text):
    if not text:
        return frozenset()
    return frozenset(_CALL_RE.findall(text.upper()))


def cycle_of(unix):
    return int(unix // 15)


def load(path):
    frames = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if d.get("unix") is None or d.get("freq") is None or d.get("status") is None:
                continue
            frames.append(d)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--freq-tol", type=float, default=3.0, help="Hz tolerance for same-slot match")
    ap.add_argument("--window", type=int, default=2, help="+/- cycles for adjacency")
    ap.add_argument("--out", default=None, help="write known-codeword fails (truth+llr) to this JSONL")
    args = ap.parse_args()

    frames = load(args.jsonl)
    if not frames:
        print("No usable frames.")
        return

    # Anchors = decoded frames (pass or osd) carrying text, indexed by cycle.
    # Each anchor: (freq, text, callsigns)
    anchors_by_cycle = defaultdict(list)
    fails = []
    n_pass = n_osd = n_fail = 0
    for d in frames:
        st = d["status"]
        c = cycle_of(d["unix"])
        if st in ("pass", "osd") and d.get("text"):
            anchors_by_cycle[c].append((d["freq"], d["text"], callsigns(d["text"])))
            n_pass += st == "pass"
            n_osd += st == "osd"
        elif st == "fail":
            n_fail += 1
            fails.append(d)

    tol = args.freq_tol
    W = args.window

    def anchors_at(freq, cyc_lo, cyc_hi):
        """All anchors within tol Hz of freq, in cycles [cyc_lo, cyc_hi]."""
        out = []
        for c in range(cyc_lo, cyc_hi + 1):
            for (af, atext, acalls) in anchors_by_cycle.get(c, []):
                if abs(af - freq) <= tol:
                    out.append((c, af, atext, acalls))
        return out

    n_spillover = 0          # same-cycle decode at freq -> dup of decoded signal
    n_signal_present = 0     # real, missed this cycle
    n_callsign_consistent = 0
    n_known = 0
    n_unlabeled = 0
    known_records = []

    span_lo = min(cycle_of(d["unix"]) for d in frames)
    span_hi = max(cycle_of(d["unix"]) for d in frames)

    for d in fails:
        c = cycle_of(d["unix"])
        fq = d["freq"]
        same_cycle = anchors_at(fq, c, c)
        if same_cycle:
            n_spillover += 1
            continue  # slot already decoded this cycle -> not a missed signal
        neigh = anchors_at(fq, c - W, c + W)
        # exclude same-cycle (already none here) -> pure adjacent
        if not neigh:
            n_unlabeled += 1
            continue
        n_signal_present += 1
        # callsign consistency across the neighbor anchors
        common = frozenset.intersection(*[a[3] for a in neigh]) if all(a[3] for a in neigh) else frozenset()
        if common:
            n_callsign_consistent += 1
        # known codeword: an adjacent anchor with identical text
        identical = [a for a in neigh if a[2].strip() == (d.get("text") or "").strip() and a[2].strip()]
        # fail.text is null, so match identical TEXT across the neighbor anchors themselves
        texts = [a[2].strip() for a in neigh if a[2].strip()]
        truth_text = None
        if texts:
            # most common neighbor text; require it appears in >=2 cycles OR is the unique neighbor text
            from collections import Counter
            cnt = Counter(texts)
            top_text, top_n = cnt.most_common(1)[0]
            distinct_cycles = len({a[0] for a in neigh if a[2].strip() == top_text})
            if distinct_cycles >= 2 or len(cnt) == 1:
                truth_text = top_text
        if truth_text:
            n_known += 1
            known_records.append({
                "unix": d["unix"], "freq": fq, "fo": d.get("fo"),
                "snr": d.get("snr"), "score": d.get("score"),
                "truth_text": truth_text, "llr": d.get("llr"),
            })

    print("=" * 64)
    print(f"FT8 temporal-coherence labeler: {args.jsonl}")
    print(f"  capture span: cycles {span_lo}..{span_hi} ({span_hi-span_lo+1} cycles, ~{(span_hi-span_lo+1)*15}s)")
    print(f"  freq-tol={tol} Hz   window=+/-{W} cycles")
    print("=" * 64)
    print(f"Frames: pass={n_pass}  osd={n_osd}  fail={n_fail}")
    print("-" * 64)
    print(f"FAIL-frame classification ({n_fail} total):")
    print(f"  spillover (same-cycle decode at freq, dup of decoded signal): {n_spillover}")
    print(f"  unlabeled (no anchor at freq in window -> likely noise):      {n_unlabeled}")
    print(f"  SIGNAL-PRESENT (real, missed this cycle):                    {n_signal_present}")
    print(f"    of which callsign-consistent across neighbors:            {n_callsign_consistent}")
    print(f"    of which KNOWN-CODEWORD (identical adjacent text = truth): {n_known}")
    print("-" * 64)
    denom = n_fail
    print(f"Reach prize estimate (the REAL F):")
    print(f"  signal-present / all fails:        {n_signal_present}/{denom} = {100*n_signal_present/denom:.1f}%")
    print(f"  known-codeword (gradeable truth):  {n_known}/{denom} = {100*n_known/denom:.1f}%")
    print(f"  -> {n_known} fail frames have a FALSE-ACCEPT-IMMUNE ground-truth label.")
    print(f"     These are the frames a neural decoder must recover to prove reach.")
    if span_hi - span_lo + 1 < 40:
        print()
        print("  NOTE: short capture (<40 cycles) -> signal-present/known counts are")
        print("        LOWER BOUNDS; a longer run will confirm more recurring stations.")

    if args.out and known_records:
        with open(args.out, "w") as f:
            for r in known_records:
                f.write(json.dumps(r) + "\n")
        print(f"\nWrote {len(known_records)} known-codeword labeled fails -> {args.out}")


if __name__ == "__main__":
    main()
