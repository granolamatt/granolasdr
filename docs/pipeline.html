<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>granolasdr Pipeline</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0d1117;
  color: #c9d1d9;
  font-family: 'Cascadia Code', 'Fira Code', monospace;
  font-size: 13px;
  padding: 24px;
}
h1 { color: #58a6ff; font-size: 16px; margin-bottom: 6px; }
.subtitle { color: #6e7681; font-size: 11px; margin-bottom: 28px; }

/* MAIN LAYOUT */
.pipeline {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0;
  max-width: 900px;
  margin: 0 auto;
}

/* STAGE BLOCK */
.stage {
  width: 100%;
  border-radius: 8px;
  border: 1px solid;
  overflow: hidden;
}
.stage-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 14px;
  font-size: 12px;
  font-weight: bold;
}
.stage-body {
  padding: 10px 14px;
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 6px 24px;
}
.stage-body.wide { grid-template-columns: 1fr; }
.step { display: flex; align-items: baseline; gap: 8px; line-height: 1.5; }
.step-num {
  color: #6e7681;
  font-size: 10px;
  min-width: 16px;
  text-align: right;
  flex-shrink: 0;
}
.step-text { font-size: 12px; }
.step-detail { color: #6e7681; font-size: 11px; }
.file-tag {
  font-size: 10px;
  padding: 1px 6px;
  border-radius: 3px;
  background: #161b22;
  border: 1px solid #30363d;
  color: #8b949e;
}

/* THREAD LABELS */
.thread-badge {
  font-size: 10px;
  padding: 2px 7px;
  border-radius: 10px;
  font-weight: normal;
}

/* ARROWS + DATA ANNOTATIONS */
.arrow-row {
  display: flex;
  align-items: center;
  justify-content: center;
  position: relative;
  height: 52px;
  width: 100%;
}
.arrow-line {
  width: 2px;
  height: 36px;
  background: #30363d;
  position: absolute;
  left: 50%;
  top: 8px;
}
.arrow-head {
  width: 0; height: 0;
  border-left: 7px solid transparent;
  border-right: 7px solid transparent;
  border-top: 10px solid #30363d;
  position: absolute;
  left: calc(50% - 7px);
  bottom: 2px;
}
.arrow-label {
  background: #161b22;
  border: 1px solid #30363d;
  border-radius: 4px;
  padding: 3px 10px;
  font-size: 11px;
  color: #8b949e;
  white-space: nowrap;
  position: absolute;
  left: calc(50% + 16px);
  top: 50%;
  transform: translateY(-50%);
}
.arrow-label.left {
  left: auto;
  right: calc(50% + 16px);
}

/* SPLIT SECTION */
.split-row {
  display: flex;
  gap: 16px;
  width: 100%;
  align-items: stretch;
}
.split-col {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 0;
}
.split-arrow {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  padding-top: 4px;
  gap: 2px;
}
.split-arrow .corner {
  color: #30363d;
  font-size: 20px;
  line-height: 1;
}

/* COLORS BY LAYER */
.rx888  { border-color: #388bfd44; }
.rx888 .stage-header  { background: #0d1f3c; color: #79c0ff; }
.rx888 .stage-body    { background: #0a1929; }

.chan   { border-color: #3fb95044; }
.chan .stage-header   { background: #0f2317; color: #56d364; }
.chan .stage-body     { background: #0b1d12; }

.ft8cu  { border-color: #d29922aa; }
.ft8cu .stage-header  { background: #2d1f0a; color: #e3b341; }
.ft8cu .stage-body    { background: #1c1409; }

.scan   { border-color: #ff6b6baa; }
.scan .stage-header   { background: #2a0d0d; color: #ff9999; }
.scan .stage-body     { background: #1c0909; }

.d2h    { border-color: #56cfe1aa; }
.d2h .stage-header    { background: #091e24; color: #70d6e8; }
.d2h .stage-body      { background: #071518; }

.ft8cpu { border-color: #bc8cff44; }
.ft8cpu .stage-header { background: #1a0e2e; color: #d2a8ff; }
.ft8cpu .stage-body   { background: #120a1f; }

.ldpc   { border-color: #ff7b7255; }
.ldpc .stage-header   { background: #2a1010; color: #ffa198; }
.ldpc .stage-body     { background: #1c0c0c; }

.out    { border-color: #56d36444; }
.out .stage-header    { background: #0f2317; color: #7ee787; }
.out .stage-body      { background: #0b1d12; }

/* special highlight */
.hl { color: #e3b341; }
.hl2 { color: #79c0ff; }
.hl3 { color: #56d364; }
.hl4 { color: #ff9999; }
.hl5 { color: #d2a8ff; }
.dim { color: #4a5060; }

/* WF LAYOUT */
.wf-diagram {
  background: #0a0e14;
  border: 1px solid #21262d;
  border-radius: 4px;
  padding: 8px 12px;
  margin-top: 4px;
  font-size: 11px;
}
.wf-row { display: flex; align-items: center; gap: 6px; margin: 2px 0; }
.wf-box {
  display: inline-block;
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 11px;
  white-space: nowrap;
}

.note {
  font-size: 11px;
  color: #6e7681;
  margin-top: 4px;
  padding: 4px 8px;
  border-left: 2px solid #30363d;
}
.future {
  border-color: #388bfd;
  color: #79c0ff;
}
.band-grid {
  display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px;
}
.band {
  font-size: 10px; padding: 2px 6px; border-radius: 3px;
  background: #161b22; border: 1px solid #388bfd44;
  color: #79c0ff;
}
</style>
</head>
<body>
<h1>granolasdr Signal Processing Pipeline</h1>
<div class="subtitle">RX888 → HFChannelizer → FT8Cuda → ft8.cc | block diagram with code locations</div>

<div class="pipeline">

<!-- ===== RX888 ===== -->
<div class="stage rx888">
  <div class="stage-header">
    RX888 ADC
    <span class="thread-badge" style="background:#0d1f3c;border:1px solid #388bfd44;color:#79c0ff">hw thread</span>
    <span class="file-tag">gm/rx888/rx888.h</span>
  </div>
  <div class="stage-body wide">
    <div class="step"><span class="step-num">1</span><span class="step-text">16-bit real ADC samples, <span class="hl2">140 MS/s</span></span></div>
    <div class="step"><span class="step-num">2</span><span class="step-text">DMA into host ring buffer, block = <span class="hl2">524,288 samples</span> = 3.75 ms</span></div>
    <div class="step"><span class="step-num">3</span><span class="step-text">Signals <code>BufferPosition&lt;int16_t&gt;</code> — HFChannelizer thread wakes</span></div>
  </div>
</div>

<div class="arrow-row">
  <div class="arrow-line"></div><div class="arrow-head"></div>
  <div class="arrow-label">int16_t ring · 140 MS/s · 524,288 samples/block</div>
</div>

<!-- ===== HFChannelizer ===== -->
<div class="stage chan">
  <div class="stage-header">
    HFChannelizer — Wideband FFT &amp; Band Select
    <span class="thread-badge" style="background:#0f2317;border:1px solid #3fb95044;color:#56d364">CUDA thread</span>
    <span class="file-tag">gm/cuda/HFChannelizer.cc</span>
  </div>
  <div class="stage-body">
    <div>
      <div class="step"><span class="step-num">1</span><span class="step-text hl3">H2D copy:</span><span class="step-text"> int16 → GPU</span></div>
      <div class="step"><span class="step-num">2</span><span class="step-text hl3">Cast kernel:</span><span class="step-text"> int16 → float32, scale</span></div>
      <div class="step"><span class="step-num">3</span><span class="step-text hl3">Overlap-save wrap:</span><span class="step-text"> copy tail of previous block to start of fftInData_d</span></div>
      <div class="step"><span class="step-num">4</span>
        <span class="step-text"><span class="hl3">R2C FFT: <span class="hl">1,048,576-pt</span></span> real → complex<br>
        <span class="step-detail">bin width = 140,000,000 / 1,048,576 ≈ <span class="hl2">133.5 Hz/bin</span><br>
        output: 524,288 complex bins spanning 0–70 MHz</span>
        </span>
      </div>
    </div>
    <div>
      <div class="step"><span class="step-num">5</span>
        <span class="step-text"><span class="hl3">Band bin select:</span> zero-fill channelData_d[32768], then copy 10 HF bands into composite spectrum:
        <div class="band-grid">
          <span class="band">160m 13480–14980</span>
          <span class="band">80m 26212–29960</span>
          <span class="band">60m 39920–40712</span>
          <span class="band">40m 52424–54676</span>
          <span class="band">30m 75644–76024</span>
          <span class="band">20m 104856–107480</span>
          <span class="band">17m 135324–136076</span>
          <span class="band">15m 157284–160660</span>
          <span class="band">12m 186420–187172</span>
          <span class="band">10m 209712–222448</span>
        </div>
        <span class="step-detail">Total composite bins filled ≈ 28,912 of 32,768</span>
        </span>
      </div>
      <div class="step"><span class="step-num">6</span>
        <span class="step-text"><span class="hl3">C2C IFFT: <span class="hl">32,768-pt</span></span> composite → decimated time domain<br>
        <span class="step-detail">output sample rate = 140,000,000 / 32 = <span class="hl2">4,375,000 S/s</span></span>
        </span>
      </div>
      <div class="step"><span class="step-num">7</span>
        <span class="step-text"><span class="hl3">Overlap-save extract:</span> copy samples [8192..24576] = <span class="hl2">16,384 complex</span> → demodData_d<br>
        <span class="step-detail">discards overlap prefix; remaining are valid decimated samples</span>
        </span>
      </div>
      <div class="step"><span class="step-num">8</span>
        <span class="step-text"><span class="hl3">Stream callback:</span> setPosition() — FT8Cuda thread wakes</span>
      </div>
    </div>
  </div>
</div>

<div class="arrow-row">
  <div class="arrow-line"></div><div class="arrow-head"></div>
  <div class="arrow-label">complex&lt;float&gt; ring · 4.375 MS/s · 16,384 samples/block · all HF bands packed</div>
</div>

<!-- ===== FT8Cuda ===== -->
<div class="stage ft8cu">
  <div class="stage-header">
    FT8Cuda — Multi-offset FFT &amp; Magnitude Ring
    <span class="thread-badge" style="background:#2d1f0a;border:1px solid #d2992244;color:#e3b341">CUDA thread</span>
    <span class="file-tag">gm/cuda/FT8Cuda.cc</span>
  </div>
  <div class="stage-body">
    <div>
      <div class="step"><span class="step-num">1</span>
        <span class="step-text"><span class="hl">Accumulate:</span> D2D copy blocks → demodData_d until <span class="hl">buff_pos &gt; 2 × 698,880 = 1,397,760 samples</span><br>
        <span class="step-detail">≈ 85 HFChannelizer blocks = 320 ms of data</span>
        </span>
      </div>
      <div class="step"><span class="step-num">2</span>
        <span class="step-text"><span class="hl">16 FFTs</span> (4 time offsets × 4 freq offsets):</span>
      </div>
      <div class="step"><span class="step-num"></span>
        <span class="step-text">
          <table style="font-size:11px;border-collapse:collapse;margin-left:20px">
            <tr><td style="color:#6e7681;padding:1px 8px 1px 0">time offset t=0..3</td><td>start sample = t × 174,720</td></tr>
            <tr><td style="color:#6e7681;padding:1px 8px 1px 0">freq offset f=1..3</td><td>freqShift by f × 1.5625 Hz</td></tr>
            <tr><td style="color:#6e7681;padding:1px 8px 1px 0">each FFT</td><td style="color:#e3b341">C2C 698,880-pt → 698,880 bins</td></tr>
            <tr><td style="color:#6e7681;padding:1px 8px 1px 0">bin width</td><td>4,375,000 / 698,880 = <span style="color:#79c0ff">6.25 Hz</span> (= 1 FT8 tone)</td></tr>
            <tr><td style="color:#6e7681;padding:1px 8px 1px 0">output</td><td>demodFT8_d[16 × 698,880] complex</td></tr>
          </table>
        </span>
      </div>
      <div class="step"><span class="step-num">3</span>
        <span class="step-text"><span class="hl">Advance:</span> buff_pos -= 698,880 samples<br>
        <span class="step-detail">= 160 ms = one FT8 symbol period → fires ~6.25×/sec</span>
        </span>
      </div>
    </div>
    <div>
      <div class="step"><span class="step-num">4</span>
        <span class="step-text"><span class="hl">magKernel:</span> complex → uint8 magnitude<br>
        <span class="step-detail">each byte ≈ 20·log₁₀(|z|) + offset (dB-compressed power)</span>
        </span>
      </div>
      <div class="step"><span class="step-num">5</span>
        <span class="step-text"><span class="hl">D2D ring write:</span> 1 block = 4×4×698,880 = <span class="hl">11,182,080 bytes</span> → magFT8_ring_d<br>
        <span class="step-detail">ring wraps at RING_BLOCKS=200 slots = 2.24 GB on GPU</span>
        </span>
      </div>
      <div class="note" style="margin-top:8px">
        <b>Waterfall memory layout</b> (same for GPU ring and CPU decode slot):<br>
        <code style="color:#e3b341">mag[block × block_stride + (ts×4 + fs) × 698880 + fo]</code><br>
        block_stride = time_osr × freq_osr × num_bins = 4×4×698,880 = 11,182,080<br>
        <span style="color:#6e7681">ts = time sub-sample [0..3]  |  fs = freq sub-sample [0..3]  |  fo = freq bin [0..698879]</span>
      </div>
      <div class="step" style="margin-top:8px"><span class="step-num">6</span>
        <span class="step-text"><span class="hl">Trigger</span> @ second 14.7 of each 15s epoch:<br>
        <span class="step-detail">cudaEventRecord(ring_ready, stream)<br>
        → fires scan_stream and transfer_stream, see below</span>
        </span>
      </div>
    </div>
  </div>
</div>

<!-- SPLIT ARROW -->
<div style="width:100%;height:28px;position:relative;display:flex;align-items:flex-end;justify-content:center">
  <div style="position:absolute;left:50%;top:0;width:2px;height:28px;background:#30363d"></div>
  <!-- left arm -->
  <div style="position:absolute;left:25%;top:14px;width:calc(25%);height:2px;background:#30363d"></div>
  <!-- right arm -->
  <div style="position:absolute;left:50%;top:14px;width:calc(25%);height:2px;background:#30363d"></div>
  <!-- arrowheads -->
  <div style="position:absolute;left:calc(25% - 7px);top:14px;width:0;height:0;border-top:7px solid transparent;border-bottom:7px solid transparent;border-left:10px solid #30363d"></div>
  <div style="position:absolute;left:calc(75% - 3px);top:14px;width:0;height:0;border-top:7px solid transparent;border-bottom:7px solid transparent;border-left:10px solid #30363d"></div>
</div>

<!-- SPLIT: GPU SCAN + D2H -->
<div class="split-row">

  <div class="split-col">
    <div class="stage scan">
      <div class="stage-header">
        GPU Sync Scan
        <span class="thread-badge" style="background:#2a0d0d;border:1px solid #ff6b6b44;color:#ff9999">scan_stream</span>
        <span class="file-tag">gm/cuda/FT8ScanCuda.cu</span>
      </div>
      <div class="stage-body wide">
        <div class="step"><span class="step-num">1</span>
          <span class="step-text">Reads <span class="hl4">magFT8_ring_d</span> (GPU ring)<br>
          <span class="step-detail">snap_start = ring_write_idx - 106 blocks</span>
          </span>
        </div>
        <div class="step"><span class="step-num">2</span>
          <span class="step-text"><span class="hl4">ft8SyncScanKernel</span> grid (2730, 480) = 1.31M blocks<br>
          <span class="step-detail">one thread per (to, ts, fs, fo) combination<br>
          to=0..29 × ts=0..3 × fs=0..3 × fo=0..698879</span>
          </span>
        </div>
        <div class="step"><span class="step-num">3</span>
          <span class="step-text">Each thread checks <span class="hl4">Costas pattern</span> at 3 sync groups:<br>
          <span class="step-detail">symbols [0..6], [36..42], [72..78]<br>
          pattern = {3,1,4,0,6,5,2} (tones 0–6 of FT8 alphabet)<br>
          compares against magnitude of neighboring tones<br>
          kCostas[] is __constant__ memory</span>
          </span>
        </div>
        <div class="step"><span class="step-num">4</span>
          <span class="step-text">Passes threshold → <span class="hl4">atomicAdd</span> to candidate list<br>
          <span class="step-detail">outputs ~22,000 candidates (fo, to, ts, fs, score)<br>
          GPU_CAND_MAX = 500,000 safety cap</span>
          </span>
        </div>
        <div class="step"><span class="step-num">5</span>
          <span class="step-text">cudaEventRecord(<span class="hl4">scan_done</span>, scan_stream)</span>
        </div>
        <div class="note future" style="margin-top:6px"><b>CURRENT:</b> validates against CPU scan (VALIDATE_GPU_CANDS)<br><b>NEXT:</b> replace CPU scan entirely with this kernel output</div>
      </div>
    </div>
  </div>

  <div class="split-col">
    <div class="stage d2h">
      <div class="stage-header">
        D2H Snapshot (background thread)
        <span class="thread-badge" style="background:#091e24;border:1px solid #56cfe144;color:#70d6e8">transfer_stream</span>
        <span class="file-tag">gm/cuda/FT8Cuda.cc</span>
      </div>
      <div class="stage-body wide">
        <div class="step"><span class="step-num">1</span>
          <span class="step-text">transfer_stream waits for <span class="hl2">ring_ready</span> event</span>
        </div>
        <div class="step"><span class="step-num">2</span>
          <span class="step-text">1–2 <span class="hl2">cudaMemcpyAsync D2H</span> transfers<br>
          <span class="step-detail">106 × 11.2 MB = <span class="hl2">1.18 GB</span> of pinned magFT8[]<br>
          destination = pinned cudaHostAlloc decode slot<br>
          pure DMA, no CUDA context lock held</span>
          </span>
        </div>
        <div class="step"><span class="step-num">3</span>
          <span class="step-text">cudaStreamSynchronize(transfer_stream)<br>
          <span class="step-detail">waits ~100 ms for PCIe DMA</span>
          </span>
        </div>
        <div class="step"><span class="step-num">4</span>
          <span class="step-text">cudaEventSynchronize(<span class="hl2">scan_done</span>)<br>
          <span class="step-detail">waits for GPU scan kernel to finish</span>
          </span>
        </div>
        <div class="step"><span class="step-num">5</span>
          <span class="step-text">Download ~22K GPU candidates D2H<br>
          <span class="step-detail">fo[], to[], ts[], fs[], score[] arrays</span>
          </span>
        </div>
        <div class="step"><span class="step-num">6</span>
          <span class="step-text">setPosition() → <span class="hl2">ft8.cc wakes</span></span>
        </div>
      </div>
    </div>
  </div>

</div>

<!-- rejoin arrow -->
<div style="width:100%;height:28px;position:relative">
  <!-- right arm in -->
  <div style="position:absolute;left:25%;top:14px;width:calc(25%);height:2px;background:#30363d"></div>
  <!-- right arm in -->
  <div style="position:absolute;left:50%;top:14px;width:calc(25%);height:2px;background:#30363d"></div>
  <!-- drop down -->
  <div style="position:absolute;left:50%;top:0;width:2px;height:28px;background:#30363d"></div>
</div>

<div class="arrow-row" style="height:28px">
  <div class="arrow-line" style="height:20px"></div><div class="arrow-head"></div>
  <div class="arrow-label">uint8_t decode slot (pinned, 1.18 GB) + GpuScanResult (~22K cands)</div>
</div>

<!-- ===== ft8.cc CPU SYNC SCAN ===== -->
<div class="stage ft8cpu">
  <div class="stage-header">
    ft8.cc — CPU Sync Scan (ft8_lib)
    <span class="thread-badge" style="background:#1a0e2e;border:1px solid #bc8cff44;color:#d2a8ff">CPU thread</span>
    <span class="file-tag">gm/hf/ft8.cc · ft8_lib/ft8/decode.h</span>
  </div>
  <div class="stage-body">
    <div>
      <div class="step"><span class="step-num">1</span>
        <span class="step-text"><code style="color:#d2a8ff">mon.wf.mag</code> = pointer to pinned magFT8 decode slot<br>
        <span class="step-detail">106 blocks × block_stride = 1,185,300,480 bytes<br>
        wf.num_bins=698880 · time_osr=4 · freq_osr=4</span>
        </span>
      </div>
      <div class="step"><span class="step-num">2</span>
        <span class="step-text"><span class="hl5">ftx_find_candidates_range()</span> — multi-threaded<br>
        <span class="step-detail">splits freq_offset range across all CPU cores<br>
        each thread scans its slice of 698,880 bins</span>
        </span>
      </div>
      <div class="step"><span class="step-num">3</span>
        <span class="step-text">Inner loop: <span class="hl5">ft8_sync_score()</span><br>
        <span class="step-detail">for each (to, ts, fs, fo): check 21 Costas tones at 3 sync groups<br>
        <span style="color:#d2a8ff">SAME algorithm as GPU kernel — GPU replicates this exactly</span></span>
        </span>
      </div>
    </div>
    <div>
      <div class="step"><span class="step-num">4</span>
        <span class="step-text">Per-thread heap: top <span class="hl5">2,000 candidates</span> by score<br>
        <span class="step-detail">heap cap = CPU's way of limiting work; GPU has no such cap</span>
        </span>
      </div>
      <div class="step"><span class="step-num">5</span>
        <span class="step-text">Merge + sort all threads → global list of <span class="hl5">2,000 candidates</span><br>
        <span class="step-detail">each candidate: freq_offset, time_offset, time_sub, freq_sub, score</span>
        </span>
      </div>
      <div class="step"><span class="step-num">6</span>
        <span class="step-text">⏱ takes <span class="hl5">~1,600 ms</span> on 16-core CPU<br>
        <span class="step-detail">this is why GPU scan exists — target: replace entirely with GPU output</span>
        </span>
      </div>
      <div class="note future" style="margin-top:6px"><b>PLAN:</b> replace this step with GPU scan output (already validated via VALIDATE_GPU_CANDS)</div>
    </div>
  </div>
</div>

<div class="arrow-row">
  <div class="arrow-line"></div><div class="arrow-head"></div>
  <div class="arrow-label">2,000 candidates (fo, to, ts, fs, score)</div>
</div>

<!-- ===== LDPC ===== -->
<div class="stage ldpc">
  <div class="stage-header">
    ft8.cc — LDPC Decode (ft8_lib)
    <span class="thread-badge" style="background:#2a1010;border:1px solid #ff7b7244;color:#ffa198">CPU thread</span>
    <span class="file-tag">ft8_lib/ft8/decode.h · ft8_lib/ldpc/</span>
  </div>
  <div class="stage-body">
    <div>
      <div class="step"><span class="step-num">1</span>
        <span class="step-text">For each of the 2,000 candidates — run in parallel across CPU cores</span>
      </div>
      <div class="step"><span class="step-num">2</span>
        <span class="step-text"><span class="hl4">ftx_decode_candidate():</span><br>
        <span class="step-detail">• Extract 79-symbol log-likelihood ratios from wf.mag at candidate position<br>
        • Map to soft-decision LLRs for LDPC decoder</span>
        </span>
      </div>
      <div class="step"><span class="step-num">3</span>
        <span class="step-text"><span class="hl4">LDPC decode:</span> 25-iteration belief propagation<br>
        <span class="step-detail">• Code: (174,91) — 91 data bits, 83 parity bits<br>
        • Recovers 77-bit message payload + 14-bit CRC</span>
        </span>
      </div>
    </div>
    <div>
      <div class="step"><span class="step-num">4</span>
        <span class="step-text"><span class="hl4">CRC check:</span> 14-bit CRC-14 over 77+83 bits</span>
      </div>
      <div class="step"><span class="step-num">5</span>
        <span class="step-text"><span class="hl4">Message unpack:</span> ftx_message_decode()<br>
        <span class="step-detail">• 77 bits → callsign1 (28b) + callsign2 (28b) + grid/report (15b) + type (3b)<br>
        • Callsign hash table lookup for compressed callsigns</span>
        </span>
      </div>
      <div class="step"><span class="step-num">6</span>
        <span class="step-text">⏱ takes <span class="hl4">~20 ms</span> total<br>
        <span class="step-detail">fast because 2000 candidates run fully parallel on CPU threads</span>
        </span>
      </div>
      <div class="note future" style="margin-top:6px"><b>FUTURE:</b> LDPC decode on GPU — one thread block per candidate, batched over ~22K GPU candidates</div>
    </div>
  </div>
</div>

<div class="arrow-row">
  <div class="arrow-line"></div><div class="arrow-head"></div>
  <div class="arrow-label">decoded FT8 messages (callsign, freq, SNR, time)</div>
</div>

<!-- ===== OUTPUT ===== -->
<div class="stage out">
  <div class="stage-header">
    ft8.cc — Frequency Convert &amp; Publish
    <span class="thread-badge" style="background:#0f2317;border:1px solid #56d36444;color:#7ee787">CPU thread</span>
    <span class="file-tag">gm/hf/ft8.cc</span>
  </div>
  <div class="stage-body">
    <div>
      <div class="step"><span class="step-num">1</span>
        <span class="step-text"><span class="hl3">composite_bin_to_rf_hz():</span> convert FFT bin → RF Hz<br>
        <span class="step-detail">fo (698,880-bin FT8 FFT) → composite IFFT bin (32,768) → wideband bin (1,048,576) → Hz<br>
        uses kBandMap table derived from kHFBands (gm/hf/hf_bands.h)</span>
        </span>
      </div>
      <div class="step"><span class="step-num">2</span>
        <span class="step-text"><span class="hl3">SNR:</span> score - 26 dB (normalise to 2500 Hz reference BW)</span>
      </div>
    </div>
    <div>
      <div class="step"><span class="step-num">3</span>
        <span class="step-text"><span class="hl3">stdout:</span> <code>-07.0 +1778976210.1 +3.00 7076397 ~  WA9TT AB5CC 73</code></span>
      </div>
      <div class="step"><span class="step-num">4</span>
        <span class="step-text"><span class="hl3">ZMQ PUB:</span> JSON on tcp://*:5580<br>
        <span class="step-detail">{"call":"WA9TT","freq":7076397,"snr":-7.0,"unix":…,"offset":3.0}</span>
        </span>
      </div>
    </div>
  </div>
</div>

<!-- TIMING SUMMARY -->
<div style="width:100%;margin-top:28px;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px 18px">
  <div style="color:#58a6ff;font-size:12px;font-weight:bold;margin-bottom:10px">Per-epoch timing (15s window)</div>
  <table style="width:100%;border-collapse:collapse;font-size:11px">
    <tr style="color:#6e7681;border-bottom:1px solid #21262d">
      <th style="text-align:left;padding:3px 8px">Stage</th>
      <th style="text-align:left;padding:3px 8px">Thread</th>
      <th style="text-align:left;padding:3px 8px">Runs for</th>
      <th style="text-align:left;padding:3px 8px">Duration</th>
      <th style="text-align:left;padding:3px 8px">Status</th>
    </tr>
    <tr style="border-bottom:1px solid #21262d">
      <td style="padding:4px 8px;color:#56d364">HFChannelizer FFTs</td>
      <td style="padding:4px 8px;color:#6e7681">CUDA stream</td>
      <td style="padding:4px 8px">every 3.75 ms</td>
      <td style="padding:4px 8px;color:#e3b341">&lt;1 ms GPU</td>
      <td style="padding:4px 8px;color:#3fb950">✓ not a bottleneck</td>
    </tr>
    <tr style="border-bottom:1px solid #21262d">
      <td style="padding:4px 8px;color:#e3b341">FT8Cuda FFT×16 + mag + ring</td>
      <td style="padding:4px 8px;color:#6e7681">main CUDA stream</td>
      <td style="padding:4px 8px">every 160 ms</td>
      <td style="padding:4px 8px;color:#e3b341">~30 ms GPU</td>
      <td style="padding:4px 8px;color:#3fb950">✓ non-blocking</td>
    </tr>
    <tr style="border-bottom:1px solid #21262d">
      <td style="padding:4px 8px;color:#ff9999">GPU Costas scan kernel</td>
      <td style="padding:4px 8px;color:#6e7681">scan_stream</td>
      <td style="padding:4px 8px">once per epoch</td>
      <td style="padding:4px 8px;color:#ff9999">~100–200 ms GPU</td>
      <td style="padding:4px 8px;color:#3fb950">✓ non-blocking to main</td>
    </tr>
    <tr style="border-bottom:1px solid #21262d">
      <td style="padding:4px 8px;color:#70d6e8">D2H snapshot (1.18 GB pinned)</td>
      <td style="padding:4px 8px;color:#6e7681">transfer_stream</td>
      <td style="padding:4px 8px">once per epoch</td>
      <td style="padding:4px 8px;color:#70d6e8">~100 ms PCIe DMA</td>
      <td style="padding:4px 8px;color:#3fb950">✓ non-blocking to main</td>
    </tr>
    <tr style="border-bottom:1px solid #21262d">
      <td style="padding:4px 8px;color:#d2a8ff">CPU sync scan (ft8_lib)</td>
      <td style="padding:4px 8px;color:#6e7681">ft8.cc thread</td>
      <td style="padding:4px 8px">once per epoch</td>
      <td style="padding:4px 8px;color:#d2a8ff">~1,600 ms</td>
      <td style="padding:4px 8px;color:#f0883e">⚠ target for GPU replacement</td>
    </tr>
    <tr>
      <td style="padding:4px 8px;color:#ffa198">CPU LDPC decode (ft8_lib)</td>
      <td style="padding:4px 8px;color:#6e7681">ft8.cc thread</td>
      <td style="padding:4px 8px">once per epoch</td>
      <td style="padding:4px 8px;color:#ffa198">~20 ms</td>
      <td style="padding:4px 8px;color:#f0883e">⚠ future GPU candidate</td>
    </tr>
  </table>
</div>

</div><!-- end pipeline -->
</body>
</html>
