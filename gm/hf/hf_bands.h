#ifndef _GM_HF_HF_BANDS_H_
#define _GM_HF_HF_BANDS_H_

#include <stdint.h>

// Single source of truth for RX888 HF band layout.
// Wideband FFT: 1400000-pt R2C at 140 MS/s → bin_hz = 100.0 Hz exactly.
// Each row: wideband bin window covering both FT8 and JS8 sub-bands, with
// 1 kHz (10-bin) guard on each side.  bw = wb_end - wb_start.
// All indices: freq_Hz / 100.
struct HFBand {
    uint32_t wb_start;
    uint32_t wb_end;
    uint32_t bw;
    const char* name;
    uint32_t ft8_dial_bin; // wideband bin for FT8 dial freq
    uint32_t js8_dial_bin; // wideband bin for JS8 dial freq
};

// Windows sized from observed on-air activity (granolasdr-old messages.txt analysis).
// Total bw = 2110 wideband bins → composite IFFT 4096 (zero-padded), sample rate 409.6 kHz,
// MagBlock FFT 65536-pt at 6.25 Hz/bin.
static const HFBand kHFBands[] = {
    { 18390,  18460,   70, "160m",  18400,  18420},   // FT8 1.840, JS8 1.842 MHz
    { 35720,  35820,  100,  "80m",  35730,  35780},   // FT8 3.573, JS8 3.578 MHz
    { 53560,  53740,  180,  "60m",  53570,  53570},   // FT8/JS8 5.357 MHz + 5.3715 MHz channel
    { 70680,  70820,  140,  "40m",  70740,  70780},   // FT8 7.074, JS8 7.078 MHz; lowered start to 7.068
    {101290, 101420,  130,  "30m", 101360, 101300},   // FT8 10.136, JS8 10.130 MHz
    {140700, 141000,  300,  "20m", 140740, 140780},   // FT8 14.074, JS8 14.078 MHz; 14.070-14.100
    {180700, 181130,  430,  "17m", 181000, 181040},   // FT8 18.100, JS8 18.104 MHz; 18.070-18.113 (JS8 at 18.108+)
    {210720, 210950,  230,  "15m", 210740, 210780},   // FT8 21.074, JS8 21.078 MHz; +21.091 secondary
    {249140, 249310,  170,  "12m", 249150, 249220},   // FT8 24.915, JS8 24.922 MHz; extended to 24.931
    {280730, 281000,  270,  "10m", 280740, 280780},   // FT8 28.074, JS8 28.078 MHz; extended to 28.100
    {503110, 503200,   90,   "6m", 503130, 503180},   // FT8 50.313, JS8 50.318 MHz
};
static const int kNumHFBands = (int)(sizeof(kHFBands) / sizeof(kHFBands[0]));

#endif // _GM_HF_HF_BANDS_H_
