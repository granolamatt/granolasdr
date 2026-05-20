#ifndef _GM_HF_HF_BANDS_H_
#define _GM_HF_HF_BANDS_H_

#include <stdint.h>

// Single source of truth for RX888 HF band layout.
// Wideband FFT: 1400000-pt R2C at 140 MS/s → bin_hz = 100.0 Hz exactly.
// Each row: wideband bin range occupied by one HF amateur band.
// bw = wb_end - wb_start (number of composite IFFT bins for this band).
// All indices: freq_Hz / 100.
struct HFBand {
    uint32_t wb_start;
    uint32_t wb_end;
    uint32_t bw;
    const char* name;
    uint32_t ft8_dial_bin; // wideband bin for FT8 dial freq (for audio extraction)
};

static const HFBand kHFBands[] = {
    { 18000,  20000,  2000, "160m",  18400},
    { 35000,  40000,  5000,  "80m",  35730},
    { 53300,  54100,   800,  "60m",  53380},
    { 70000,  73000,  3000,  "40m",  70740},
    {101000, 101500,   500,  "30m", 101360},
    {140000, 143500,  3500,  "20m", 140740},
    {180680, 181680,  1000,  "17m", 181000},
    {210000, 214500,  4500,  "15m", 210740},
    {248900, 249900,  1000,  "12m", 249150},
    {280000, 297000, 17000,  "10m", 280740},
};
static const int kNumHFBands = (int)(sizeof(kHFBands) / sizeof(kHFBands[0]));

#endif // _GM_HF_HF_BANDS_H_
