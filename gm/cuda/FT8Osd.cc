#include "gm/cuda/FT8Osd.h"
#include "gm/cuda/OsdCore.h"
#include "ft8_lib/ft8/constants.h"   // kFTX_LDPC_generator, FTX_LDPC_{N,K,M}

namespace {

// Systematic generator G (91 x 174) for FT8: c = msg * G.
//   columns [0..90]   identity (message is systematic in the low positions)
//   columns [91..173] parity i = dot(msg, kFTX_LDPC_generator[i]) (mod 2)
// Built once from the packed parity-generator table FT8 already carries.
uint8_t g_ft8_gen[FTX_LDPC_K][FTX_LDPC_N];

bool build_generator()
{
    for (int m = 0; m < FTX_LDPC_K; ++m) {
        g_ft8_gen[m][m] = 1;                         // identity part
        for (int i = 0; i < FTX_LDPC_M; ++i) {       // parity columns
            int bit = (kFTX_LDPC_generator[i][m >> 3] >> (7 - (m & 7))) & 1;
            g_ft8_gen[m][FTX_LDPC_K + i] = (uint8_t)bit;
        }
    }
    return true;
}

} // namespace

bool ft8_osd_decode(const float* llr, int order, uint8_t* plain174,
                    float* out_soft_dist)
{
    static const bool inited = build_generator();   // thread-safe one-time init
    (void)inited;
    return osd_decode_n174(llr, FTX_LDPC_K, &g_ft8_gen[0][0], order,
                           plain174, out_soft_dist);
}
