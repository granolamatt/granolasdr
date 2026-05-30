# JS8 Decode Algorithm — Linear Reference

Source: `~/workarea/js8/` (JS8Call-improved repo)
Purpose: Reference for adding JS8 as a second protocol to granolasdr CUDA pipeline.

---

## 1. Signal Parameters

### Submodes

| Mode ID | Name     | Samples/Symbol | Baud Rate  | Period | Tone Spacing | Bandwidth | SNR Floor |
|---------|----------|---------------|------------|--------|-------------|-----------|-----------|
| 0 (A)   | NORMAL   | 1920          | 6.25 Hz    | 15 s   | 6.25 Hz     | ~50 Hz    | −24 dB    |
| 1 (B)   | FAST     | 1200          | 10 Hz      | 10 s   | 10 Hz       | ~80 Hz    | −22 dB    |
| 2 (C)   | JS8 40   | 600           | 20 Hz      | 6 s    | 20 Hz       | ~160 Hz   | −20 dB    |
| 4 (E)   | SLOW     | 3840          | 3.125 Hz   | 30 s   | 3.125 Hz    | ~25 Hz    | −28 dB    |
| 8 (I)   | JS8 60   | 384           | ~31.25 Hz  | 4 s    | 31.25 Hz    | ~250 Hz   | −18 dB    |

All modes: **12 kHz internal sample rate**, **79 symbols/frame**, **8-FSK**.

### Costas Arrays

Two variants; Normal (A) uses ORIGINAL, all others use MODIFIED.

```
ORIGINAL  — same array for all 3 blocks:
  Block 0,1,2: [4, 2, 5, 6, 1, 3, 0]

MODIFIED  — each block is different:
  Block 0: [0, 6, 2, 3, 5, 4, 1]
  Block 1: [1, 5, 0, 2, 3, 6, 4]
  Block 2: [2, 5, 0, 6, 4, 1, 3]
```

### Frame Layout (79 symbols)

```
Symbol index:  0  1  2  3  4  5  6 | 7 ............. 35 | 36 37 38 39 40 41 42 | 43 ............ 71 | 72 73 74 75 76 77 78
               [--- Costas 0 ---]   [--- 29 data ---]    [------ Costas 1 -----]  [--- 29 data ---]   [------ Costas 2 -----]
               <--- 7 sync ----->   <-- ND/2 data  -->   <------ 7 sync -------->  <-- ND/2 data  -->  <------- 7 sync ------>

NS  = 21 sync symbols   (3 Costas blocks × 7)
ND  = 58 data symbols   (split 29 + 29 around middle Costas block)
NN  = 79 total symbols
```

### Payload

```
87 information bits = 75 message bits + 12 CRC bits

75 message bits = 12 × 6-bit words from 64-char alphabet:
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-+"

3 additional bits (decoded[72..74]) = message type field

Total encoded: LDPC(174,87)
  N = 174 bits  (87 message + 87 parity check)
  K = 87 bits
  M = 87 check nodes
  Code rate = 0.5
```

---

## 2. Decode Pipeline (Linear, Step-by-Step)

### Stage 1 — Audio Input and Downsampling

**Source:** `Detector.cpp` — `writeData()`

```
Input: 48 kHz PCM (int16, stereo or mono) from sound card

Step 1.1  For every 4 input samples (48 kHz → 12 kHz decimation):
          Apply 49-tap FIR lowpass filter using Eigen dot product
          Filter coefficients: pre-designed lowpass at 6 kHz cutoff
          Output: 1 int16 sample at 12 kHz

Step 1.2  Append downsampled sample to d2[] ring buffer
          Buffer size: JS8_RX_SAMPLE_SIZE = NTMAX × 12000
          NTMAX is typically 120 s → 1,440,000 samples

Step 1.3  At quarter-symbol boundaries emit framesWritten signal
          → triggers decode cycle
```

---

### Stage 2 — Copy Audio into Decoder Working Buffer

**Source:** `JS8.cpp` — top of `decode()`

```
Step 2.1  Lock the detector mutex
Step 2.2  Copy d2[] into local float array dd[NMAX]
          dd[] is single-precision float, length = NTXDUR × 12000
          Example (NORMAL mode): 15 × 12000 = 180,000 samples

Step 2.3  Compute baseband FFT over dd[] (computeBasebandFFT)
          FFT size: NDFFT1 = NSPS × NDD
          Example (NORMAL): 1920 × 100 = 192,000-pt real-to-complex FFT
          Output: ds_cx[]  (complex frequency domain, NDFFT1/2+1 bins)
          Purpose: one large FFT used by ALL candidates — computed once
```

---

### Stage 3 — Candidate Detection (`syncjs8`)

**Source:** `JS8.cpp` — `syncjs8(int nfa, int nfb)`

```
Step 3.1  Build short-time power spectra s[freq][time]

          FFT size: NFFT1 = NSPS × 2  (2× oversampled in frequency)
          Step size: NSTEP = NSPS / 4  (quarter-symbol time resolution)
          Number of steps: NHSYM = NMAX/NSTEP - 3

          For each step j = 0 .. NHSYM-1:
            window = dd[j×NSTEP .. j×NSTEP + NFFT1] × Nuttall window
            FFT window → sd[] (complex, NFFT1/2+1 bins)
            s[i][j] = |sd[i]|²   for i = 0 .. NSPS-1
            savg[i] += s[i][j]   (accumulate average spectrum)

Step 3.2  Compute noise baseline (baselinejs8)

          Input: savg[] in linear power scale
          Region: 500–2500 Hz  (indices bmin..bmax)

          a) Convert savg[bmin..bmax] to dB:  10×log10(savg[i])
          b) Sample 10th-percentile values at 6 Chebyshev nodes
             (equally spaced in Chebyshev cosine spacing, avoids Runge's phenomenon)
          c) Fit 5th-degree polynomial via least-squares (Vandermonde + QR)
          d) Evaluate polynomial at every i in [ia, ib] → savg[i] (now = baseline dB)
          e) savg[] outside [ia, ib] is zeroed

Step 3.3  Costas correlation scan (coarse candidates)

          For each frequency bin i in [ia, ib]:
            For each time offset j in [-JZ, +JZ]:
              For each Costas block p = 0,1,2:
                For each column n = 0..6:
                  offset = j + JSTRT + NSSY×n + p×36×NSSY
                  t[0][p] += s[i + 2×Costas[p][n]][offset]  (Costas tone energy)
                  t[1][p] += Σ_freq s[i + 2×freq][offset]   (total energy all 8 tones)

              sync_all = (t[0][0]+t[0][1]+t[0][2]) / ((t[1][0]+t[1][1]+t[1][2] - costas_sum) / 6)
              sync_01  = same for blocks 0+1 only
              sync_12  = same for blocks 1+2 only
              sync_value = max(sync_all, sync_01, sync_12)

            Record: best (j, sync_value) for this bin i
            Emit Sync{freq=i×DF, step=j×TSTEP, sync=sync_value}

Step 3.4  Normalize and prune

          Sort all Sync entries by sync value
          Normalize: divide every sync by the 40th-percentile value
          Remove entries below ASYNCMIN = 1.5
          Remove near-duplicates: keep only the strongest within ±AZ Hz
            AZ ≈ 0.64 × baud_rate  (e.g., NORMAL: ±4 Hz)
          Limit to NMAXCAND = 300 candidates

          Output: vector<Sync>{freq, dt, sync}  sorted by strength
```

---

### Stage 4 — Per-Candidate Fine Sync & Downsample (`js8dec`)

**Source:** `JS8.cpp` — `js8dec()`; run once per candidate

```
Step 4.1  Narrow-band downsample (js8_downsample)

          Uses ds_cx[] from Stage 2 (already computed)
          FFT bin resolution: DF = 12000 / NDFFT1

          a) Identify extraction band around f0:
               ft = f0 + 8.5 × baud     (top edge)
               fb = f0 - 1.5 × baud     (bottom edge)
               ib = round(fb / DF)
               it = round(ft / DF)

          b) Copy ds_cx[ib..it] → cd0[0..it-ib]
             Zero-fill remainder of cd0 to NDFFT2 = NDFFT1/NDOWN

          c) Apply cosine taper (NDD+1 samples) to head and tail
             Reduces spectral leakage at band edges

          d) Cyclic rotate cd0 by (i0 - ib) bins to center f0 at DC

          e) Inverse FFT cd0[NDFFT2] → time-domain complex baseband
             Normalize by 1/sqrt(NDFFT1 × NDFFT2)

          Result: cd0[] = complex baseband at 12000/NDOWN rate
          Example NORMAL: rate = 12000/60 = 200 Hz, NDOWNSPS=32 samples/symbol

Step 4.2  Coarse timing search

          Starting guess: i0 = round((xdt + ASTART) × FS2)
          Search i0 ± NQSYMBOL (quarter-symbol steps)

          For each idt in [i0-NQSYMBOL, i0+NQSYMBOL]:
            sync = syncjs8d(idt, 0.0)
            syncjs8d:
              For each Costas block i=0..2, column j=0..6:
                offset = 36×i×NDOWNSPS + idt + j×NDOWNSPS
                inner product of cd0[offset..offset+NDOWNSPS] with
                  precomputed Costas phasor csyncs[i][j][]
                sync += |inner_product|²
            Pick idt with highest sync → ibest

          xdt2 = ibest / FS2

Step 4.3  Fine frequency search

          i0 = round(xdt2 × FS2)
          For ifr = -NFSRCH .. +NFSRCH  (±2.5 Hz in 0.5 Hz steps):
            delf = ifr × 0.5
            compute syncjs8d(i0, delf)  (rotates each sample by phase delf)
            Track best delfbest

          Apply phase rotation to all cd0[] samples:
            dphi = -delfbest × 2π / FS2
            cd0[i] *= exp(j × i × dphi)

          f1 += delfbest  (refine frequency estimate)

Step 4.4  Per-symbol tone magnitude extraction

          For each symbol k = 0 .. 78:

            i1 = ibest + k × NDOWNSPS + timing_tracker_correction

            a) Copy cd0[i1 .. i1+NDOWNSPS] → csymb[]
            b) Apply frequency tracker correction (phase rotation each sample)
            c) FFT csymb[NDOWNSPS] in-place
            d) s2[tone][k] = |csymb[tone]| / 1000   for tone = 0..7

            At Costas symbol positions (k in 0-6, 36-42, 72-78):
              expectedTone = Costas[block][column]
              Estimate residual frequency error via parabolic interpolation:
                delta_Hz = 0.5 × (|csymb[tone-1]|² - |csymb[tone+1]|²)
                           / (|csymb[tone-1]|² - 2|csymb[tone]|² + |csymb[tone+1]|²)
                           × (FS2 / NDOWNSPS)
              Update frequency tracker EMA with delta_Hz
              Compute Goertzel energies at i1-1, i1, i1+1 for timing gradient
              Update timing tracker with gradient estimate

Step 4.5  Sync quality gate

          Count how many of the 21 Costas symbols have correct max tone:
            nsync = count of k ∈ {Costas positions} where argmax_tone(s2[:,k]) == expectedTone

          Reject candidate if nsync ≤ 6
```

---

### Stage 5 — Soft Symbol Conversion and LLR Generation

**Source:** `JS8.cpp` — `WhiteningProcessor`, then LLR assembly

```
Step 5.1  Extract data symbols into s1[8][58]

          From s2[8][79], skip Costas columns:
            s1[row][0..28]  = s2[row][7..35]    (first data block)
            s1[row][29..57] = s2[row][43..71]   (second data block)

Step 5.2  Find winning (max-power) tone per data symbol

          For each symbol j = 0..57:
            symbolWinners[j] = argmax_i(s1[i][j])

Step 5.3  Whitening / LLR computation  (WhiteningProcessor.h)

          Two LLR arrays are produced: llr0[174] and llr1[174]

          The whitening processor:
          a) Computes log-likelihood ratios from the 8-tone magnitudes
             For each bit b of the 3-bit symbol index:
               LLR[b] = log(P(bit=1) / P(bit=0))
                      derived from the magnitude ratios across 8 tones

          b) Maps 58 × 3-bit symbols → 174 LLR values (58×3=174)

          c) Applies erasure threshold: any LLR with |LLR| < threshold → 0
             threshold configurable, default from ldpcFeedbackEnabled()

Step 5.4  Soft combining  (SoftCombiner.h)

          Key = (submode, round(f1), round(xdt), llr hash)
          Combine: average LLRs with previously stored values for same key
          TTL = 2 × NTXDUR (discard stale entries)
          Purpose: accumulate evidence across multiple decode cycles
                   improves weak-signal decoding
```

---

### Stage 6 — LDPC Belief-Propagation Decode

**Source:** `JS8.cpp` — `bpdecode174()` + multi-pass loop

```
LDPC parameters:
  N = 174  (total codeword bits)
  K = 87   (message bits; output)
  M = 87   (check nodes)
  Mn[174][3]  — for each variable node: which 3 check nodes it participates in
  Nm[87][≤7]  — for each check node: which variable nodes it involves

Step 6.1  Initialize
          toc[M][7] = LLR values for each bit in each check node
          tov[N][3] = 0  (messages from check nodes to variable nodes)
          zn[N] = LLR    (current bit beliefs)

Step 6.2  Iterate (max 30 rounds):

          a) Update beliefs:
               zn[i] = llr[i] + sum(tov[i][0..2])

          b) Hard decision:
               cw[i] = (zn[i] > 0) ? 1 : 0

          c) Syndrome check:
               For each check node i:
                 synd[i] = sum(cw[neighbors]) mod 2
               ncheck = count of failed checks
               If ncheck == 0: SUCCESS → copy cw[87..173] to decoded[0..86]

          d) Early stopping:
               If ncheck hasn't improved for 5 consecutive iterations
               AND iter ≥ 10 AND ncheck > 15: FAIL

          e) Update check → variable messages:
               toc[i][j] = zn[neighbor] - tov[neighbor][k]

          f) Update variable → check messages:
               tanhtoc[i][j] = tanh(-toc[i][j] / 2)
               tov[i][j] = 2 × atanh(- product of tanhtoc for all other neighbors)

Step 6.3  Multi-pass decode (4 passes):

          Pass 1: llr0Combined  (primary LLRs from soft combiner)
          Pass 2: llr1Combined  (secondary LLRs)
          Pass 3: llr0Combined with bits 0..23 zeroed    (erasure assist)
          Pass 4: llr0Combined with bits 24..47 zeroed   (erasure assist)

          Between each pass: optional LDPC-feedback refinement
            refineLlrsWithLdpcFeedback(): examine cw[] vs llr[], flip
            low-confidence bits that disagree with check constraints

Step 6.4  CRC-12 verification

          Pack decoded[0..86] bits into bytes
          Compute CRC-12 with polynomial 0xC06, XOR key 42
          Compare to received CRC in decoded[75..86]
          If mismatch: discard this decode attempt
```

---

### Stage 7 — Message Extraction

**Source:** `JS8.cpp` — `extractmessage174()`

```
Step 7.1  Decode 12 characters from decoded[0..71] (72 bits):
          For i = 0..11:
            word[i] = decoded[i*6+0]<<5 | decoded[i*6+1]<<4 | ... | decoded[i*6+5]
            char[i] = alphabet[word[i]]
          alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-+"

Step 7.2  Extract 3-bit message type:
          i3bit = decoded[72]<<2 | decoded[73]<<1 | decoded[74]

Step 7.3  Compute SNR:
          Re-encode the decoded message to itone[79]
          xsig = sum of s2[itone[k]][k]² for k=0..78
          xbase = 10^(0.1 × (savg[freq_bin] - BASESUB))
          xsnr = 10×log10(xsig/xbase - 1) - 32  (in dB)

Step 7.4  Signal subtraction (if lsubtract=true):
          Reconstruct reference signal from itone[] and f1
          Low-pass filter the cross-correlation (cfilt = dd × conj(cref))
          FFT → apply bandpass filter → IFFT
          dd[] -= 2 × Real(cfilt × cref)
          Purpose: removes decoded signal, allows weaker signals to decode
          Repeat whole pipeline from Stage 3 for up to ~5 passes

Step 7.5  Emit decoded event:
          {utc, snr, xdt, frequency, data (12 chars), type, quality, mode}
```

---

### Stage 8 — Varicode / Protocol Layer

**Source:** `JS8_Main/Varicode.cpp`

```
The 12-character fixed-length string is the inner JS8 frame.
The 3-bit type field selects the message format.
Varicode decodes callsigns, grids, free-text, heartbeats, APRS, etc.
This layer is entirely application-level — not needed for a decode-only monitor.
```

---

## 3. Full Pipeline Diagram

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                     JS8 DECODE PIPELINE                      │
                    └─────────────────────────────────────────────────────────────┘

  48 kHz PCM audio
       │
       ▼
┌─────────────┐
│  FIR LPF    │  49-tap lowpass, decimate 4:1
│  Detector   │  48 kHz → 12 kHz
└──────┬──────┘
       │ int16 @ 12 kHz
       ▼
┌─────────────┐
│  d2[] ring  │  circular buffer, ~120 s @ 12 kHz
│  buffer     │
└──────┬──────┘
       │ float dd[NMAX]  (copy each period)
       ▼
┌─────────────────────┐
│  computeBasebandFFT │  1 large R2C FFT (NDFFT1 = NSPS × NDD)
│  ds_cx[]            │  shared by all candidates this cycle
└──────┬──────────────┘
       │
       ▼
┌──────────────────────────────────────────────┐
│  syncjs8() — CANDIDATE SEARCH                │
│                                              │
│  1. Short-time power spectra                 │
│     NHSYM × NFFT1 FFTs (step = NSPS/4)       │
│     s[NSPS][NHSYM], savg[NSPS]               │
│                                              │
│  2. Baseline (Chebyshev + poly fit)          │
│     savg[] → noise floor                    │
│                                              │
│  3. Costas correlation scan                  │
│     Each freq bin × time offset              │
│     Sync metric = Costas energy / noise      │
│                                              │
│  4. Normalize (40th pct), prune, dedupe      │
│     Output: up to 300 candidates             │
└──────┬───────────────────────────────────────┘
       │ vector<Sync>{freq, dt, sync_strength}
       │
       │  ┌────── For each candidate ──────────────────────────────────┐
       ▼  ▼                                                            │
┌──────────────────────────────────────────────┐                      │
│  js8_downsample() — NARROW BAND EXTRACTION   │                      │
│                                              │                      │
│  Extract ±8.5 baud around f0 from ds_cx[]    │                      │
│  Taper edges → cyclic rotate → IFFT          │                      │
│  cd0[] = complex baseband @ 12000/NDOWN Hz   │                      │
└──────┬───────────────────────────────────────┘                      │
       │                                                               │
       ▼                                                               │
┌──────────────────────────────────────────────┐                      │
│  FINE SYNC (syncjs8d)                        │                      │
│                                              │                      │
│  Coarse timing: search ±NQSYMBOL offsets     │                      │
│  Fine frequency: search ±2.5 Hz @ 0.5 Hz     │                      │
│  Apply phase rotation to cd0[]               │                      │
└──────┬───────────────────────────────────────┘                      │
       │                                                               │
       ▼                                                               │
┌──────────────────────────────────────────────┐                      │
│  PER-SYMBOL FFT (79 symbols)                 │                      │
│                                              │                      │
│  k=0..78: FFT(cd0[i1..i1+NDOWNSPS])         │                      │
│  Apply freq/timing tracker corrections       │                      │
│  s2[8][79] = |FFT[0..7]| / 1000             │                      │
│  Gate: reject if Costas match ≤ 6/21         │                      │
└──────┬───────────────────────────────────────┘                      │
       │                                                               │
       ▼                                                               │
┌──────────────────────────────────────────────┐                      │
│  WHITENING + LLR GENERATION                  │                      │
│                                              │                      │
│  s1[8][58] = s2 minus Costas columns         │                      │
│  s1 → log likelihood ratios → llr0[174]      │                      │
│  Erasure threshold: zero weak LLRs           │                      │
│  Soft combining: accumulate across cycles    │                      │
└──────┬───────────────────────────────────────┘                      │
       │                                                               │
       ▼                                                               │
┌──────────────────────────────────────────────┐                      │
│  LDPC BP DECODE (up to 4 passes)             │                      │
│                                              │                      │
│  bpdecode174(llr[174]) → cw[174], nerr       │                      │
│  30 BP iterations per attempt                │                      │
│  Optional LDPC feedback refinement           │                      │
│  CRC-12 verification                         │                      │
└──────┬───────────────────────────────────────┘                      │
       │                                                               │
       ▼                                                               │
┌──────────────────────────────────────────────┐                      │
│  MESSAGE EXTRACTION                          │                      │
│                                              │                      │
│  decoded[0..71] → 12 chars (6-bit alphabet) │                      │
│  decoded[72..74] → 3-bit type               │                      │
│  Compute SNR from signal energy vs baseline  │                      │
│  Subtract signal from dd[] if lsubtract      │──────────────────────┘
└──────┬───────────────────────────────────────┘    (iterative pass)
       │
       ▼
  Event::Decoded{utc, snr, freq, dt, data[12], type, mode}
```

---

## 4. Comparison: JS8 Normal vs FT8 (granolasdr current)

| Parameter          | FT8                     | JS8 Normal (Mode A)       |
|--------------------|-------------------------|---------------------------|
| Sample rate        | 12 kHz                  | 12 kHz                    |
| Tone spacing       | 6.25 Hz                 | 6.25 Hz                   |
| Samples/symbol     | 1920                    | 1920                      |
| Symbols/frame      | 79                      | 79                        |
| Period             | 15 s                    | 15 s                      |
| Costas arrays      | [4,2,5,6,1,3,0] × 3    | Same (ORIGINAL type)      |
| Costas positions   | 0-6, 36-42, 72-78       | 0-6, 36-42, 72-78         |
| LDPC size          | (174, 91)               | **(174, 87)**             |
| Parity matrix      | ft8_lib/ldpc_174_91     | **different matrix**      |
| Message bits       | 91 (77 + CRC-14)        | **87 (75 + CRC-12)**      |
| Message alphabet   | callsign/grid packed    | **64-char 6-bit words**   |
| Message type field | no (protocol fixed)     | **3-bit type field**      |
| LLR count          | 174                     | 174                       |
| Submodes           | 1                       | 5 (different baud rates)  |

**Key insight for CUDA port:** JS8 Normal shares the exact same FFT size, Costas array, and frame timing as FT8. The GPU candidate search (Costas correlation) is nearly identical — only the sync metric normalization and threshold differ slightly. The LDPC decoder requires a different parity matrix and CRC.

---

## 5. What a Minimal GPU Port Needs

```
Stage 1  (Channelizer)   — REUSE unchanged. JS8 bands fall inside existing HF band slices.
                           No new FFT work needed.

Stage 2  (Copy to GPU)   — REUSE. Same 12 kHz sample rate, same buffer structure.

Stage 3  (Candidate scan) — REUSE Costas scan kernel with these changes:
                             • JS8 Normal: use same Costas [4,2,5,6,1,3,0]
                             • JS8 B/C/E/I: swap in MODIFIED Costas array
                             • Sync metric formula is the same structure
                             • Different NSPS/NFFT1/NSTEP/NHSYM constants

Stage 4  (Downsample)    — REUSE band-extract IFFT logic. Same spectral slice approach.

Stage 5  (Symbol FFT)    — REUSE. NDOWNSPS changes per submode but algorithm is identical.

Stage 6  (LLR)          — MINOR CHANGE. Whitening formula is the same structure.
                           Different LLR bit mapping if JS8 uses different gray coding.

Stage 7  (LDPC)         — NEW parity matrix (Mn/Nm tables) for JS8's (174,87) code.
                           Different CRC: CRC-12 with poly 0xC06 XOR 42
                           Different message unpack: 12 × 6-bit chars not FT8 packed fields

Stage 8  (Output)        — NEW message type decode, but can emit same JSON structure
                           with protocol="JS8" field added.
```

The dominant new work is **Stage 7**: a new BP decoder instantiation with the JS8 parity matrix, and a new CRC-12 check. Everything else is parameter changes to existing kernels.

---

## 6. JS8 Operating Frequencies

Primary bands for monitoring:

| Band | Dial Freq  | Mode       |
|------|-----------|------------|
| 40m  | 7.078 MHz  | JS8 Normal |
| 20m  | 14.078 MHz | JS8 Normal |
| 80m  | 3.578 MHz  | JS8 Normal |
| 30m  | 10.130 MHz | JS8 Normal |
| 17m  | 18.104 MHz | JS8 Normal |
| 15m  | 21.078 MHz | JS8 Normal |

All fall within granolasdr's existing band slice coverage.
