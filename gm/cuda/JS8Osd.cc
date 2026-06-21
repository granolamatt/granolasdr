#include "gm/cuda/JS8Osd.h"
#include "gm/cuda/JS8Generator.h"   // g_js8_gen[87][174]
#include "gm/cuda/OsdCore.h"

bool js8_osd_decode(const float* llr, int order, uint8_t* xhat, float* out_soft_dist)
{
    return osd_decode_n174(llr, /*K=*/87, &g_js8_gen[0][0], order, xhat, out_soft_dist);
}
