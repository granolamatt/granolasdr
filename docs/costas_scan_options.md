# Costas Scan Optimization Options

Trade-off analysis pending. Options to evaluate:

---

## Option A — Increase scan stride (6 → 12)

Halve the scan rate for both FT8 and JS8 Normal.

- Change `cont_stride_` in `FT8Cuda.h` and `STRIDE` in `JS8Cuda.cc`
- FT8 and JS8 still share the MagBlock<200> ring; both trigger every 12 writes (~1.9 s)
- GPU load per scanner: ~6% → 3% each
- Trade-off: scan rate drops from ~1/s to ~0.5/s; may miss short-lived signals

---

## Option B — Reduce num_bins to relevant HF band

Currently scanning the full 0–6.5 MHz output of HFChannelizer (1,048,576 bins).
Most amateur HF activity is in narrow sub-bands (e.g., 200 kHz windows).

- Reduce `rfft_length` passed to MagBlock / scan kernels
- Ring slot stride shrinks from 16 MB to ~500 KB
- Scan kernel goes from ~2M blocks to ~60k blocks
- Expected scan time: 110 ms → ~5–10 ms
- Trade-off: loses out-of-band visibility; may need per-band MagBlock instances

---

## Option C — Coherent matched filter (conj-multiply + FFT)

Replace magnitude-domain Costas correlation with a time-domain matched filter.

**Concept:**  
For each of 21 Costas symbol positions, extract the IQ segment from the
HFChannelizer buffer and multiply by `conj(exp(2πi × kCostas[k] × Δf × t))`.
Coherently sum all 21 derotated segments. One FFT of the sum gives correlation
scores at ALL frequency offsets simultaneously.

**Why the memory pattern is better:**  
Current kernel reads 21 ring slots separated by 16 MB strides (scattered DRAM).  
Matched filter reads 21 × 1M consecutive complex samples from a sequential IQ
ring — saturates DRAM bandwidth instead of fighting it.

**Advantages:**
- ~3 dB coherent SNR gain over non-coherent magnitude approach
- Sequential memory access → much better DRAM utilization
- One FFT per (time_off, sub) combo replaces the full frequency scan

**Disadvantages / risks:**
- Requires a complex float32 IQ ring alongside the existing uint8 mag ring
  (~800 MB for 106 symbols at 1M bins vs. 3.2 GB for full uint8 ring)
- Phase coherence needed across 21 symbols (~12 s for FT8) — ionospheric
  Doppler and multipath can decorrelate on HF, making coherent gain unreliable
- MagBlock currently discards complex IQ immediately; architecture change needed

**Hybrid variant:**  
Coherently combine within each Costas group (7 symbols, ~1 s each) → 3 FFTs,
then sum magnitudes non-coherently across the 3 groups.  
Gets most of the within-group coherent gain while tolerating slow Doppler drift.

**Files that would change:**  
`MagBlock.cc`, `FT8Cuda.cc` / `JS8Cuda.cc`, new `IQRing` buffer type,
new scan kernel replacing `ft8SyncScanKernel`.

---

## Option D — Chi-distribution pre-filter (two-stage detector)

Use the waterfall observation that active signals appear as visually obvious narrow
elevated-power columns.  A fast statistical test over the time-averaged magnitude
ring can reduce the ~1M candidate bins to a few thousand before the Costas scan
ever runs.

**Concept:**  
For each frequency bin `fo`, integrate the uint8 magnitude over the recent
79–106 ring slots (one FT8 frame worth).  Under noise alone, the per-bin power
samples are i.i.d. Rayleigh² (chi-squared, 2 dof each); their sum over T slots
follows chi-squared(2T).  A bin containing part of an FSK signal will have a
higher sum — detectable at any desired false-alarm rate via the chi CDF.

Bins that survive the threshold feed the Costas scan; everything else is skipped.

**Expected reduction:**  
In typical HF conditions there may be ~100–500 active FT8/JS8 signals across
6.5 MHz, each occupying ~8 bins.  That is ~800–4000 signal bins out of 1,048,576.
Even with a generous false-alarm budget (e.g., 10,000 noise bins passing the test),
the candidate set is <2% of the full frequency range.

Costas scan grid reduction:
- Current:  ceil(1,048,576 / 256) × 480 ≈ 2,000,000 blocks  →  ~110 ms
- After D:  ceil(15,000 / 256)    × 480 ≈    28,000 blocks  →  ~1.5 ms

The pre-filter itself reads the ring buffer in a single sequential pass and is
memory-bandwidth-friendly — nothing like the scattered 16 MB stride reads of
the Costas kernel.

**Threshold selection:**  
For T=79 time slots, noise sum follows chi-squared(158).  Setting the per-bin
threshold at the 99.9th percentile gives a 0.1% false alarm rate per bin, or
roughly 1,000 false alarms across 1M bins — entirely manageable as Costas
scan input noise.  The exact factor can be tuned empirically against the
waterfall noise floor.

**Variants:**
- *Broadband noise tracking:* estimate noise floor as the median power across
  all bins in the same time window; threshold as a multiple of that estimate
  (cell-averaging CFAR).
- *Sub-band CFAR:* divide spectrum into 1 kHz sub-bands; estimate noise floor
  locally within each sub-band to handle frequency-dependent RFI levels.
- *Variance-based:* 8-FSK signal bins have higher variance than noise (energy
  hops between 8 tones); a variance threshold can complement the mean-power test.

**Advantages:**
- Works entirely on the existing uint8 mag ring — no architecture changes
- Composable with any of A/B/C; especially synergistic with C (only run
  matched filter on pre-selected bins)
- Directly models what the eye sees in the waterfall — interpretable threshold

**Disadvantages / risks:**
- Adds ~1 ms per scan for the pre-filter pass (negligible vs. 110 ms scan)
- Very weak signals (near noise floor) may not survive the chi test and will be
  missed entirely — the Costas scan currently catches them via the matched-filter
  gain; the two-stage approach trades sensitivity for speed
- Requires careful threshold tuning: too tight → miss signals; too loose →
  no scan-time benefit

**Files that would change:**  
New `ChiFilter` CUDA kernel (reads mag ring, writes candidate bin list).
`FT8Cuda.cc` / `JS8Cuda.cc` scan launch uses candidate list instead of full grid.

---

## Option E — Split HFChannelizer into known data-band segments

FT8 and JS8 operate only at internationally agreed spot frequencies.  Rather
than scanning the full 6.5 MHz output, channelize directly to those allocations
and run independent narrow-band scan streams for each one.

**Known FT8/JS8 HF allocations (common):**

| Band | FT8 (MHz) | JS8 (MHz) |
|------|-----------|-----------|
| 160m | 1.840     | 1.842     |
| 80m  | 3.573     | 3.578     |
| 40m  | 7.074     | 7.078     |
| 30m  | 10.136    | 10.130    |
| 20m  | 14.074    | 14.078    |
| 17m  | 18.100    | 18.104    |
| 15m  | 21.074    | 21.078    |
| 12m  | 24.915    | 24.922    |
| 10m  | 28.074    | 28.078    |

Typical activity fits within a ~3 kHz window per allocation.  At 6.25 Hz/bin
that is ~480 bins — a 2,000:1 reduction from 1,048,576.

**Grid at 480 bins per band:**
- Current:   ceil(1,048,576 / 256) × 480 = ~2,000,000 blocks
- Per band:  ceil(480 / 256)       × 480 =       960 blocks
- 9 FT8 bands: ~9,000 blocks total — essentially free GPU time, no serialization

**Complications the user identified:**

1. **Voice/data overlap.** HF SSB voice (2.4 kHz) sits immediately adjacent to
   and sometimes overlapping data allocations.  A clean channelizer split needs
   guard bands or a polyphase filter bank that handles the transition precisely.

2. **Still has the slow scan on full-band view.** If the goal is also to catch
   out-of-band or pirate activity anywhere in the 6.5 MHz window, E doesn't
   help — you'd need the full-band scan running in parallel.

3. **Architectural complexity.** Requires multiple narrow-band MagBlock
   instances (one per allocation, or one per cluster of nearby allocations),
   coordinated scan streams, and a reconfiguration path when the operator
   wants to add/remove bands.  HFChannelizer would need to output multiple
   narrow complex streams rather than one wide stream.

4. **Propagation-dependent activity.** Some bands are dead at certain times
   (e.g., 10m overnight).  Scanning dead bands wastes nothing at 960 blocks,
   but the channelizer filter still runs.

**Relationship to other options:**
- E is a targeted, pre-planned version of B (reduce num_bins), applied per
  known allocation rather than uniformly
- E and D are complementary: E removes the dead spectrum structurally; D
  removes the noise bins statistically within whatever band survives
- E alone does not solve the FT8+JS8 GPU serialization problem (both still
  trigger at the same ring write if sharing a band), but at 960 blocks each
  the serialization cost is negligible

**Files that would change:**  
`HFRx.cc` pipeline setup; `HFChannelizer` output split or duplicated;
multiple narrow `MagBlock` instances with small `rfft_length`; band
configuration table (center frequency + width per allocation).
