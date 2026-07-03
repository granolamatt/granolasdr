# Design: complex-composite ring for per-candidate refine (FT8Cuda / JS8Cuda)

Path A of the coherent/overlap study (see `coherent_overlap_findings.md`): keep
the (4,4) Costas scan, and add a **per-candidate frequency/time refine** that
recovers +6 decodes over the grid (measured offline, 20m/17m/15m). The refine
needs the **complex** composite (phase), which `MagBlock` discards. This is the
retention buffer that makes the refine possible.

## Requirement

For a candidate, the refine must read the complex samples of its full FT8/JS8
frame (79 symbols × 0.16 s = **12.64 s**) to search fine freq/time and re-extract
LLRs. The live complex composite (`hfBufferPosition`) is only ~16 blocks (~80 ms)
deep. So we need a rolling ring holding ≥ one frame + scan/decode latency + margin.

## Where the current data flows

```
HFChannelizer ── hfBufferPosition ──► MagBlock<200>  ──► DeviceRingBuffer<uint8,200>  (mag ring)
  (complex composite,                  (RFFT+|·|²+u8)        │  read by:
   409.6 kHz, 2048/block, ~5 ms)                            ├─ FT8Cuda   (Costas scan → LLR → LDPC/OSD)
                                                            └─ JS8Cuda<200> (same)
```

Key timing (`MagBlock::doCopy`): reads `length`=2048 complex samples/call into an
accumulator; every `rfft`=65536 samples it emits **one mag slot** = **32 composite
blocks** = one FT8/JS8 symbol (0.16 s). FT8Cuda and JS8Cuda<200> read the SAME
MagBlock<200>, so **one complex ring serves both**.

## Design

Add the complex ring to **MagBlock<200>** (it already reads every composite block
and owns the mag-slot ↔ composite-block correspondence). Expose it as a second
output; FT8Cuda and JS8Cuda take it by const ref, same contract as the mag ring.

```
MagBlock<200>
  ├─ mag ring       DeviceRingBuffer<uint8_t, 200>          (unchanged)
  ├─ complex ring   DeviceRingBuffer<complex<float>, N_CPLX>  (NEW, ~20 s)
  └─ slot_cplx_idx[200]   per-mag-slot snapshot of complex write_idx (NEW)
       │
       ├─ FT8Cuda:      scan (mag ring) → candidates → [refine on complex ring] → LLR → LDPC/OSD
       └─ JS8Cuda<200>: same
```

1. **Ring:** `DeviceRingBuffer<complex<float>, N_CPLX>`, slot = one composite block
   (2048 complex). `N_CPLX = 4096` blocks → ~20 s → **64 MB VRAM** (4096·2048·8).
   Frame is 12.64 s = 2528 blocks; 4096 leaves ~7 s of scan/decode-latency margin.
2. **Producer:** in `doCopy`, alongside the existing accumulator copy, D2D-copy the
   incoming 2048 block into `complex_ring.slot(cplx_wi++)`; record its event. One
   extra `cudaMemcpyAsync` of 16 KB per call. Negligible.
3. **Correspondence:** each time a mag slot `wi_mag` is emitted, store
   `slot_cplx_idx[wi_mag % 200] = cplx_wi`. Exact map from a candidate's mag-slot
   time to its complex-block start — no drift, no arithmetic guess.
4. **Refine (FT8Cuda / JS8Cuda), per candidate `(fo, to)`:**
   a. `cplx_start = slot_cplx_idx[(snap_start+to) % 200]` → complex ring offset.
   b. Read the frame (2528 blocks) from the complex ring; downconvert to the
      candidate's composite frequency (from `fo`) → baseband, decimate.
   c. Fine freq/time search maximizing Costas energy (±½ symbol time, ±½ bin freq).
      The search absorbs any mag↔complex granularity slop, so step 3 need only be
      approximately right.
   d. Re-extract the 174 LLRs at the refined alignment → existing LDPC/OSD/CRC.

## Refine as a fallback (recommended), not a replacement

The pipeline already has an **OSD fallback** for LDPC failures. Slot the refine in
the same way: run the current direct decode first (unchanged, cheap, on the mag
ring); **refine only the strong-Costas candidates that direct + OSD failed to
crack.** The offline win came from signals the direct path misses entirely, so
re-aligning and retrying those recovers the +6 while bounding cost to the failures,
not every candidate. This mirrors the existing fallback architecture and is purely
additive — the working path is untouched.

## Multi-reader / lifetime

FT8Cuda and JS8Cuda<200> read the complex ring with independent cursors (same
pattern as the mag ring). At 20 s depth both refine well before overwrite. The
refine reads recent history (a just-scanned candidate), so it is always well
inside the retained window.

## Wiring (HFRx.cc)

`MagBlock<200>` exposes `getComplexRing()` (and the slot map). Forward construction:
`FT8Cuda(magblock.getRing(), magblock.getComplexRing(), …)` and same for
`JS8Cuda<200>`. No orchestration-level callbacks; contract preserved.

## Open decisions

- **Refine-all vs fallback-only.** Recommend fallback-only (above). Refine-all is
  the max-decode ceiling but refines noise too.
- **GPU vs CPU refine.** GPU, batched across the failed candidates (the freq/time
  search + per-symbol FFTs are GPU-friendly). Confirm it fits the FT8Cuda stream.
- **N_CPLX exact.** 4096 (~20 s) proposed; tighten once max candidate age at
  refine time is measured live.
- **Other JS8 rates (fast/slow/turbo/ultra, JS8Cuda<128>).** Same pattern, own
  complex rings on their MagBlocks, deferred until Normal FT8+JS8 is proven.
- **Absolute gain unverified live.** Offline is vanilla ft8_lib; the live pipeline
  has OSD/multipass, so measure the delta on-air. Second recording still wanted.
