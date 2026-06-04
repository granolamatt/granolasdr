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

// Windows cover [min(ft8,js8)_dial - 10, max(ft8,js8)_dial + 30 + 10] wideband bins.
// FT8/JS8 passband = 0-3000 Hz above dial = 30 bins; 10-bin guard = 1 kHz each side.
// Total bw = 900 wideband bins → composite IFFT 2048, sample rate 204.8 kHz,
// MagBlock FFT 32768-pt at 6.25 Hz/bin.
static const HFBand kHFBands[] = {
    { 18390,  18460,   70, "160m",  18400,  18420},   // FT8 1.840, JS8 1.842 MHz
    { 35720,  35820,  100,  "80m",  35730,  35780},   // FT8 3.573, JS8 3.578 MHz
    { 53560,  53610,   50,  "60m",  53570,  53570},   // FT8 = JS8 5.357 MHz
    { 70730,  70820,   90,  "40m",  70740,  70780},   // FT8 7.074, JS8 7.078 MHz
    {101290, 101400,  110,  "30m", 101360, 101300},   // FT8 10.136, JS8 10.130 MHz
    {140730, 140820,   90,  "20m", 140740, 140780},   // FT8 14.074, JS8 14.078 MHz
    {180990, 181080,   90,  "17m", 181000, 181040},   // FT8 18.100, JS8 18.104 MHz
    {210730, 210820,   90,  "15m", 210740, 210780},   // FT8 21.074, JS8 21.078 MHz
    {249140, 249260,  120,  "12m", 249150, 249220},   // FT8 24.915, JS8 24.922 MHz
    {280730, 280820,   90,  "10m", 280740, 280780},   // FT8 28.074, JS8 28.078 MHz
};
static const int kNumHFBands = (int)(sizeof(kHFBands) / sizeof(kHFBands[0]));

#endif // _GM_HF_HF_BANDS_H_
