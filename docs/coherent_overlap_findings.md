# FT8 STFT overlap and coherent-refine — offline findings

Offline study on the `coherent-demod` branch, June/July 2026. Question: does the
FT8/JS8 decode pipeline's time/frequency STFT oversampling (`time_osr`,
`freq_osr` — "the over-processing") earn its cost, can it be replaced, and can we
go coherent by using the Costas symbols as a phase pilot?

Data: one `--record` capture (`raw.dat`, 4.51 min, 409.6 kHz complex composite),
analyzed on 20m (the busiest band in the capture, FT8 energy 7.9× median).

## Harness (all host-only, `coherent/`)

- `gnlh.py` — memmap reader for GNLH `--record` captures + composite→RF band map.
- `probe.py` — per-band energy, validated the composite→RF mapping.
- `ft8.py` — extract a band to complex baseband, STFT with overlap knobs, numpy Costas sync.
- `ft8decode.cc` — pybind11 binding: build ft8_lib's waterfall from our STFT and run
  its sync + LDPC + CRC. Exposes `decode_audio`, `find_candidates`, `decode_llr`.
- `decode_overlap.py`, `coherent_refine.py` — the two experiments below.

CRC is the oracle throughout: a decode counts only if ft8_lib's LDPC+CRC pass.

## Finding 1 — the overlap roughly doubles decodes (CRC truth)

Decoding the full recording at different STFT overlap:

| overlap | unique CRC-valid decodes |
|---------|--------------------------|
| (4,4) — granolasdr Normal | 51 |
| (2,2) — WSJT-X default    | 46 |
| (1,1) — no overlap        | 29 |

The sets are nested: `(1,1) ⊂ (2,2) ⊂ (4,4)`, `only-(1,1) = 0`. Dropping the
oversampling loses ~43% of decodes, concentrated in weak/DX signals. So the
over-processing is NOT dead weight.

Note: an earlier sync-level proxy (Costas score, no decoder) suggested overlap
barely helped. That was wrong — the benefit lives in the marginal signals a
sync-SNR threshold excludes. The decoder binding (CRC oracle) was necessary.

## Finding 2 — but coarse-detect + per-candidate refine beats the grid

Why does overlap help? It brute-forces the right time/frequency *alignment* for
marginal signals. A discrete fine grid is a crude way to do that. Instead:
detect candidates cheaply at (2,2), then per candidate run a continuous
freq+time search (maximize Costas energy) and decode.

Full recording, 20m:

| approach | decodes |
|----------|---------|
| (2,2) direct        | 46 |
| (4,4) direct (grid) | 51 |
| **(2,2) coarse + continuous refine** | **67** |

Refine recall of (4,4) is effectively 100% (49/51 exact; the other 2 are the
same messages with a hashed callsign resolved *better* by refine), and it adds
18 the grid missed → **+31% over brute-force 4×4**, at lower detection cost
(a narrow search on candidates, not a fine STFT over the whole band).

**Conclusion: the over-processing is replaceable.** Coarse scan + per-candidate
alignment refine is a strict improvement over the fixed 4×4 grid. This is the
actionable result for the runtime pipeline.

### Detect at (2,2), not (1,1) — overlap gives detection SNR too

The overlap benefit is two separable things: **detection SNR** (finding marginal
signals) and **alignment** (extracting good LLRs). The refine handles alignment;
it cannot help detection, you can't refine a candidate you never found.

- (1,1) coarse + refine: 19 decodes, only 73% recall of the 4×4 grid — it misses
  weak signals outright.
- The missed signals score only 5–10 in (1,1) Costas sync (below a clean
  threshold). Lowering the threshold to ~5 does detect them, but floods to ~8,000
  candidates (47×) because (1,1) scalloping buries real signals in the noise band.
- (2,2) concentrates each signal's energy, lifting its sync score above the noise,
  so the same signals are caught at a clean high threshold with ~300 candidates.

**Recipe: (2,2) coarse detect + per-candidate refine.** (2,2) for detection SNR,
refine for alignment. This beat the 4×4 grid (67 vs 51) at lower STFT cost.

## Finding 3 — coherent phase does NOT work for FT8

The original idea: conjugate the known Costas pattern to recover the carrier
phase, de-rotate, and detect coherently (a ~1–2 dB gain). We built it (phase
tracked from the 3 Costas blocks, coherent metric `Re(z·e^{-jφ})`). Result: 17
decodes vs 51 — it fails.

Reason, and it is fundamental: **FT8 is continuous-phase GFSK.** Each symbol's
phase carries modulation memory from all prior tones, not just a clean carrier.
The Costas pilots give the carrier phase, but after removing it each *data*
symbol still has an unknown, data-dependent phase. Per-symbol coherent detection
of the data would need a CPM trellis (Viterbi over the phase states), not a
simple de-rotation. This is exactly why WSJT-X / ft8_lib are non-coherent per
symbol. JS8 shares the frame structure and the same limitation.

## Implications

- **Runtime:** prototype coarse-detect (1×1 or 2×2) + per-candidate refine in the
  FT8/JS8 GPU path; expect ≥ current decode rate at lower STFT cost. This is the
  path worth implementing.
- **Coherent gain** for FT8/JS8 is blocked by the modulation (CPM); a trellis is a
  large, uncertain effort and out of scope unless the ~1–2 dB is worth it.

## Caveats

- One band (20m), one recording. The effect is large and consistent (26>22 at
  120 s, 67>51 at full) but a breadth check (17m/15m, a second capture) would
  fully settle it.
- The Python refine is unoptimized (~minutes); the *algorithm* is cheap (few
  candidates × narrow search).
- The binding builds its own STFT (ft8_lib's CPU monitor is stripped in this
  fork); magnitudes are noise-floor-referenced since ft8_lib uses only relative
  values.
