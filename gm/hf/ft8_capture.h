#ifndef _GM_HF_FT8_CAPTURE_H_
#define _GM_HF_FT8_CAPTURE_H_

#include "ft8_lib/ft8/constants.h"

// Total blocks captured per FT8 epoch: 79 data symbols + 14 guard/pre-roll blocks.
// Must be kept in sync between FT8Cuda (buffer sizing) and ft8.cc (waterfall_init).
// Change this one number to widen the capture window.
#define FT8_CAPTURE_BLOCKS (FT8_NN + 14)

// Time oversampling: number of FFTs computed per symbol period, each offset by
// 1/FT8_TIME_OSR of a symbol. Higher values improve timing resolution for
// candidate detection. Must match the waterfall_init time_osr argument in ft8.cc.
#define FT8_TIME_OSR 9

#endif // _GM_HF_FT8_CAPTURE_H_
