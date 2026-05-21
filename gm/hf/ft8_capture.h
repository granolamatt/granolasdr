#ifndef _GM_HF_FT8_CAPTURE_H_
#define _GM_HF_FT8_CAPTURE_H_

#include "ft8_lib/ft8/constants.h"

// Total blocks per FT8 decode window: 79 data symbols + 27 guard blocks.
// Rolling ring snapshot (trigger==14) places the on-time signal at block ≈18,
// keeping signal start well within the scan kernel's time_off search range 0..29.
#define FT8_CAPTURE_BLOCKS (FT8_NN + 27)

// Width of the 20m audio extraction window in FT8 FFT bins.
// 1920 bins × 6.25 Hz/bin ≈ 12 kHz, covering the full FT8 passband above dial.
#define FT8_AUDIO_BINS 1920

// Time oversampling: number of FFTs computed per symbol period, each offset by
// 1/FT8_TIME_OSR of a symbol. Must match the waterfall_init time_osr argument in ft8.cc.
#define FT8_TIME_OSR 4

// Frequency oversampling: number of sub-bin frequency shifts applied before each FFT.
// Shifts are 0, 1/freq_osr, 2/freq_osr, ... of one bin (6.25 Hz), reducing worst-case
// frequency misalignment from ±3.1 Hz (freq_osr=1) to ±0.8 Hz (freq_osr=4).
// Must match the waterfall_init freq_osr argument in ft8.cc.
#define FT8_FREQ_OSR 4

#endif // _GM_HF_FT8_CAPTURE_H_
