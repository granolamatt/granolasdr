#include "gm/hf/refine.h"

#include <algorithm>
#include <cmath>
#include <vector>

extern "C" {
#include "ft8_lib/fft/kiss_fft.h"
}

namespace gm {
namespace hf {

const RefineMode kFT8Refine = {{3, 1, 4, 0, 6, 5, 2}, {0, 1, 3, 2, 5, 6, 4, 7}};
const RefineMode kJS8Refine = {{4, 2, 5, 6, 1, 3, 0}, {0, 1, 2, 3, 4, 5, 6, 7}};

static constexpr int   NSPS   = kRefineNsps;
static constexpr int   NSYM   = kRefineNsym;
static const int COS_POS[3]    = {0, 36, 72};
// Data-symbol frame positions (skip the 3 Costas blocks): k + (k<29?7:14).
static int data_pos(int k) { return k + (k < 29 ? 7 : 14); }

// FFT each symbol of `frame` (shifted by dt, mixed by df Hz) -> tone bins [79][8].
// Returns Costas sync energy = Σ over the 21 pilots of |z[pos, costas_tone]|.
static float sym_tones(const std::complex<float>* frame, int flen,
                       int dt, float df, const RefineMode& mode,
                       kiss_fft_cfg cfg, std::complex<float>* Z /*[NSYM*8] or null*/) {
    std::vector<kiss_fft_cpx> in(NSPS), out(NSPS);
    const float w = -2.0f * (float)M_PI * df / kRefineSr;
    float energy = 0.0f;
    for (int p = 0; p < NSYM; ++p) {
        int base = p * NSPS + dt;
        for (int i = 0; i < NSPS; ++i) {
            int idx = base + i;
            std::complex<float> s = (idx >= 0 && idx < flen) ? frame[idx] : std::complex<float>(0, 0);
            if (df != 0.0f) { float ph = w * idx; s *= std::complex<float>(std::cos(ph), std::sin(ph)); }
            in[i].r = s.real(); in[i].i = s.imag();
        }
        kiss_fft(cfg, in.data(), out.data());
        if (Z) for (int j = 0; j < 8; ++j) Z[p * 8 + j] = {out[j].r, out[j].i};
        // accumulate Costas energy if this symbol is a pilot
        for (int b = 0; b < 3; ++b) {
            int rel = p - COS_POS[b];
            if (rel >= 0 && rel < 7) {
                int tone = mode.costas[rel];
                energy += std::hypot(out[tone].r, out[tone].i);
            }
        }
    }
    return energy;
}

static float max4(float a, float b, float c, float d) {
    return std::max(std::max(a, b), std::max(c, d));
}

float refine_llr(const std::complex<float>* frame, int frame_len,
                 const RefineMode& mode, float* log174) {
    kiss_fft_cfg cfg = kiss_fft_alloc(NSPS, 0, nullptr, nullptr);

    // Fine time+freq search maximizing Costas energy (continuous, absorbs the
    // coarse grid's residual offset). ±512 samples, ±3.5 Hz — matches the offline.
    int   best_dt = 0;
    float best_df = 0.0f, best_e = -1.0f;
    for (int dt = -512; dt <= 512; dt += 64) {
        for (int fi = 0; fi < 13; ++fi) {
            float df = -3.5f + 7.0f * fi / 12.0f;
            float e = sym_tones(frame, frame_len, dt, df, mode, cfg, nullptr);
            if (e > best_e) { best_e = e; best_dt = dt; best_df = df; }
        }
    }

    // Re-run at the best alignment, keep the tone bins, extract LLRs.
    std::vector<std::complex<float>> Z(NSYM * 8);
    sym_tones(frame, frame_len, best_dt, best_df, mode, cfg, Z.data());
    kiss_fft_free(cfg);

    for (int k = 0; k < 58; ++k) {
        int p = data_pos(k);
        float s2[8];
        for (int j = 0; j < 8; ++j) {
            const std::complex<float>& z = Z[p * 8 + mode.tone_map[j]];
            s2[j] = std::hypot(z.real(), z.imag());       // non-coherent metric
        }
        log174[3*k+0] = max4(s2[4], s2[5], s2[6], s2[7]) - max4(s2[0], s2[1], s2[2], s2[3]);
        log174[3*k+1] = max4(s2[2], s2[3], s2[6], s2[7]) - max4(s2[0], s2[1], s2[4], s2[5]);
        log174[3*k+2] = max4(s2[1], s2[3], s2[5], s2[7]) - max4(s2[0], s2[2], s2[4], s2[6]);
    }
    return best_e;
}

} // namespace hf
} // namespace gm
