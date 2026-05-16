#ifndef _GM_HF_HF_BANDS_H_
#define _GM_HF_HF_BANDS_H_

#include <stdint.h>

// Single source of truth for RX888 HF band layout.
// Wideband FFT: 1048576-pt R2C at 140 MS/s → bin_hz ≈ 133.5 Hz.
// Each row: wideband bin range occupied by one HF amateur band.
// bw = wb_end - wb_start (number of composite IFFT bins for this band).
// Derived from calc_rf.py.
struct HFBand {
    uint32_t wb_start;
    uint32_t wb_end;
    uint32_t bw;
    const char* name;
};

static const HFBand kHFBands[] = {
    {13480,  14980,  1500, "160m"},
    {26212,  29960,  3748,  "80m"},
    {39920,  40712,   792,  "60m"},
    {52424,  54676,  2252,  "40m"},
    {75644,  76024,   380,  "30m"},
    {104856, 107480, 2624,  "20m"},
    {135324, 136076,  752,  "17m"},
    {157284, 160660, 3376,  "15m"},
    {186420, 187172,  752,  "12m"},
    {209712, 222448, 12736, "10m"},
};
static const int kNumHFBands = (int)(sizeof(kHFBands) / sizeof(kHFBands[0]));

#endif // _GM_HF_HF_BANDS_H_
