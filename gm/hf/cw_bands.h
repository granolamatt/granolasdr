#ifndef _GM_HF_CW_BANDS_H_
#define _GM_HF_CW_BANDS_H_

#include <stdint.h>

// CW sub-band windows for the wideband CW skimmer, separate from kHFBands
// (which sit on the FT8/JS8 dial frequencies, above the CW portion of each band).
//
// Same wideband-FFT bin indexing as hf_bands.h: bin = freq_Hz / 100 (the
// 1,400,000-pt R2C FFT at 140 MS/s gives exactly 100 Hz/bin).  Windows sized
// from a reversebeacon.net activity skim (2026-06-27) plus band-plan extents
// for bands quiet at sample time.  bw = wb_end - wb_start.
//
// Total bw packs into the CW composite's 8192-pt IFFT (-> 819.2 kHz), exactly
// as kHFBands' 2110 bins pack into the FT8/JS8 4096-pt IFFT (-> 409.6 kHz).
struct CWBand {
    uint32_t    wb_start;   // wideband FFT start bin (freq_Hz / 100)
    uint32_t    wb_end;     // wideband FFT end bin (exclusive)
    uint32_t    bw;         // wb_end - wb_start
    const char* name;
};

static const CWBand kCWBands[] = {
    {  18000,  18400,  400, "160m" },   // 1.800-1.840 MHz
    {  35000,  35600,  600,  "80m" },   // 3.500-3.560 MHz
    {  70000,  70500,  500,  "40m" },   // 7.000-7.050 MHz
    { 101000, 101300,  300,  "30m" },   // 10.100-10.130 MHz
    { 140000, 140700,  700,  "20m" },   // 14.000-14.070 MHz
    { 180680, 180980,  300,  "17m" },   // 18.068-18.098 MHz
    { 210000, 210900,  900,  "15m" },   // 21.000-21.090 MHz
    { 248900, 249200,  300,  "12m" },   // 24.890-24.920 MHz
    { 280000, 280700,  700,  "10m" },   // 28.000-28.070 MHz
};
static const int kNumCWBands = (int)(sizeof(kCWBands) / sizeof(kCWBands[0]));

// CW composite IFFT length: must hold Σ bw (= 4700) with zero-pad headroom.
// 8192 -> 819.2 kHz output (8192 bins × 100 Hz).  819.2 kHz is deliberate: it
// makes the downstream CW MagBlock's 16384-pt FFT a 20 ms window (vs 40 ms at
// the half-rate), short enough to resolve CW keying up to ~30 WPM.
static const uint32_t kCWCompositeFFT = 8192;

#endif // _GM_HF_CW_BANDS_H_
