#ifndef _GM_HF_FT8_CAPTURE_H_
#define _GM_HF_FT8_CAPTURE_H_

#include "ft8_lib/ft8/constants.h"

// Total blocks per FT8 decode window: 79 data symbols + 27 guard blocks.
// Rolling ring snapshot (trigger==14) places the on-time signal at block ≈14,
// giving ≈2.2s early tolerance and ≈2.1s late tolerance without every-other-epoch.
#define FT8_CAPTURE_BLOCKS (FT8_NN + 27)

// Time oversampling: number of FFTs computed per symbol period, each offset by
// 1/FT8_TIME_OSR of a symbol. Higher values improve timing resolution for
// candidate detection. Must match the waterfall_init time_osr argument in ft8.cc.
#define FT8_TIME_OSR 9

#endif // _GM_HF_FT8_CAPTURE_H_
