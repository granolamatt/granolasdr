#include "gm/hf/refine.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

extern "C" {
#include "ft8_lib/fft/kiss_fft.h"
}

namespace gm {
namespace hf {

const RefineMode kFT8Refine = {{3, 1, 4, 0, 6, 5, 2}, {0, 1, 3, 2, 5, 6, 4, 7}};
const RefineMode kJS8Refine = {{4, 2, 5, 6, 1, 3, 0}, {0, 1, 2, 3, 4, 5, 6, 7}};

// Experimental de-chirp toggle (OFF by default). Read once. To remove the whole
// de-chirp feature, delete this function, its declaration, the DE-CHIRP block in
// refine_llr, the kSlope* constants, and the callers' dechirp argument.
bool dechirp_enabled() {
    static const bool on = [] {
        const char* e = std::getenv("REFINE_DECHIRP");
        return e && e[0] == '1';
    }();
    return on;
}

static constexpr int   NSPS   = kRefineNsps;
static constexpr int   NSYM   = kRefineNsym;
static const int COS_POS[3]    = {0, 36, 72};
// Data-symbol frame positions (skip the 3 Costas blocks): k + (k<29?7:14).
static int data_pos(int k) { return k + (k < 29 ? 7 : 14); }

// De-chirp constants (EXPERIMENTAL, OFF by default — see the DE-CHIRP block in
// refine_llr). Search a linear frequency drift (slope, Hz/s) so a signal that
// walks across the 12.6 s frame stays in its tone bin. A fixed-df align tolerates
// only ~half a bin end-to-end (~0.5 Hz/s). Grid/margin match the offline proof
// (coherent/dechirp_test.py: baseline 0/24 vs de-chirp 24/24 at 1–3 Hz/s).
static constexpr float kSlopeMaxHz  = 5.0f;   // Hz/s search bound
static constexpr int   kSlopeSteps  = 21;     // 0.5 Hz/s resolution (< half the budget)
static constexpr float kSlopeAccept = 1.05f;  // keep a slope only if it beats flat by 5%

// FFT each symbol of `frame` (shifted by dt, mixed by df Hz) -> tone bins [79][8].
// Returns Costas sync energy = Σ over the 21 pilots of |z[pos, costas_tone]|.
static float sym_tones(const std::complex<float>* frame, int flen,
                       int dt, float df, float slope, const RefineMode& mode,
                       kiss_fft_cfg cfg, std::complex<float>* Z /*[NSYM*8] or null*/) {
    std::vector<kiss_fft_cpx> in(NSPS), out(NSPS);
    float energy = 0.0f;
    for (int p = 0; p < NSYM; ++p) {
        // During the fine-alignment search (Z == null) only the 21 Costas pilots
        // contribute to the energy — skip the 58 data-symbol FFTs entirely.
        bool is_pilot = false;
        for (int b = 0; b < 3; ++b) { int rel = p - COS_POS[b]; if (rel >= 0 && rel < 7) is_pilot = true; }
        if (!Z && !is_pilot) continue;

        int base = p * NSPS + dt;
        for (int i = 0; i < NSPS; ++i) {
            int idx = base + i;
            std::complex<float> s = (idx >= 0 && idx < flen) ? frame[idx] : std::complex<float>(0, 0);
            // De-rotate by the instantaneous correction freq f(t) = df + slope*t,
            // so the applied phase is -2π(df*t + 0.5*slope*t²), t = idx/sr. Double
            // precision: idx² overflows int, and float loses the frame-scale t² term.
            if (df != 0.0f || slope != 0.0f) {
                double t  = (double)idx / kRefineSr;
                double ph = -2.0 * M_PI * ((double)df * t + 0.5 * (double)slope * t * t);
                s *= std::complex<float>((float)std::cos(ph), (float)std::sin(ph));
            }
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
                 const RefineMode& mode, float* log174, bool dechirp) {
    kiss_fft_cfg cfg = kiss_fft_alloc(NSPS, 0, nullptr, nullptr);

    // Fine time+freq alignment maximizing Costas energy (continuous, absorbs the
    // coarse grid's residual offset). ±512 samples, ±3.5 Hz — matches the offline.
    // The energy surface is smooth and near-unimodal over one coarse cell, so
    // coordinate descent (dt, then df, then dt again to catch interaction) finds
    // the same optimum as the full 17x13 joint grid in 17+13+17 evals — ~4.7x
    // cheaper, which is what keeps refine off the scan worker's critical path.
    auto sweep_dt = [&](float df, float slope, int& best_dt, float& best_e) {
        for (int dt = -512; dt <= 512; dt += 64) {
            float e = sym_tones(frame, frame_len, dt, df, slope, mode, cfg, nullptr);
            if (e > best_e) { best_e = e; best_dt = dt; }
        }
    };
    auto sweep_df = [&](int dt, float slope, float& best_df, float& best_e) {
        for (int fi = 0; fi < 13; ++fi) {
            float df = -3.5f + 7.0f * fi / 12.0f;
            float e = sym_tones(frame, frame_len, dt, df, slope, mode, cfg, nullptr);
            if (e > best_e) { best_e = e; best_df = df; }
        }
    };
    int   best_dt = 0;
    float best_df = 0.0f, best_slope = 0.0f, best_e = -1.0f;
    sweep_dt(0.0f, 0.0f, best_dt, best_e);                     // dt at df=0
    best_e = -1.0f; sweep_df(best_dt, 0.0f, best_df, best_e);  // df at best dt (incl. df=0)
    best_e = -1.0f; sweep_dt(best_df, 0.0f, best_dt, best_e);  // dt at best df

    // ================= DE-CHIRP (experimental; OFF by default) =================
    // Corrects a linearly drifting signal by searching a frequency slope. Runs
    // only when the caller passes dechirp=true (REFINE_DECHIRP=1 at the call
    // site). Proven in synthesis (coherent/dechirp_test.py) but NOT yet shown to
    // help on a real capture — no drifters found, and it can cost a marginal
    // decode by over-fitting the pilots — so it stays opt-in until we need it.
    // SELF-CONTAINED / EASY TO REMOVE: delete this block, the kSlope* constants,
    // the `slope` arg to sym_tones/sweep_*, and the `dechirp` parameter.
    if (dechirp) {
        auto sweep_slope = [&](int dt, float df, float& best_slope, float& best_e) {
            for (int si = 0; si < kSlopeSteps; ++si) {
                float slope = -kSlopeMaxHz + 2.0f * kSlopeMaxHz * si / (kSlopeSteps - 1);
                float e = sym_tones(frame, frame_len, dt, df, slope, mode, cfg, nullptr);
                if (e > best_e) { best_e = e; best_slope = slope; }
            }
        };
        float e_flat = best_e;                                 // best energy with no drift
        float cand_slope = 0.0f, e_slope = -1.0f;
        sweep_slope(best_dt, best_df, cand_slope, e_slope);
        if (cand_slope != 0.0f && e_slope > e_flat * kSlopeAccept) {
            best_slope = cand_slope;
            best_e = -1.0f;
            sweep_df(best_dt, best_slope, best_df, best_e);    // re-fit df at the drift
        }
    }
    // =============================== END DE-CHIRP ==============================

    // Re-run at the best alignment, keep the tone bins, extract LLRs.
    std::vector<std::complex<float>> Z(NSYM * 8);
    sym_tones(frame, frame_len, best_dt, best_df, best_slope, mode, cfg, Z.data());
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

void extract_frame(const std::complex<float>* in, int n_in, float freq_hz,
                   int sr_in, std::complex<float>* out, int n_out) {
    const int decim = sr_in / kRefineSr;                 // 32 (409600 -> 12800)
    // Windowed-sinc lowpass, cutoff below the decimated Nyquist (kRefineSr/2).
    const int   half = 4 * decim;                        // 128 -> 257 taps
    const float fc   = 0.40f * (float)kRefineSr / sr_in; // ~5 kHz normalized
    std::vector<float> h(2 * half + 1);
    float hsum = 0.0f;
    for (int t = -half; t <= half; ++t) {
        float sinc = (t == 0) ? 2*fc : std::sin(2*(float)M_PI*fc*t) / ((float)M_PI*t);
        float win  = 0.5f * (1 - std::cos(2*(float)M_PI*(t + half) / (2*half)));
        h[t + half] = sinc * win;
        hsum += h[t + half];
    }
    for (auto& v : h) v /= hsum;                         // unity DC gain

    // Downconvert-then-FIR factors into a single complex-tap FIR plus a per-output
    // phase rotation.  The mix factor is exp(i*w*idx); for output center c,
    //   acc = Σ_t h[t]·in[c+t]·exp(i·w·(c+t)) = exp(i·w·c) · Σ_t g[t]·in[c+t]
    // with combined taps g[t] = h[t]·exp(i·w·t).  This computes the trig ONCE for
    // the taps (2·half+1) instead of once per (output × tap) — the inner loop is
    // then pure complex MACs.  exp(i·w·c) advances incrementally per output.
    const float w = -2.0f * (float)M_PI * freq_hz / sr_in;
    std::vector<std::complex<float>> g(2 * half + 1);
    for (int t = -half; t <= half; ++t) {
        float ph = w * t;
        g[t + half] = h[t + half] * std::complex<float>(std::cos(ph), std::sin(ph));
    }
    const std::complex<float> rot_step(std::cos(w * decim), std::sin(w * decim));
    std::complex<float> phasor(1.0f, 0.0f);              // exp(i·w·c), c = k·decim
    for (int k = 0; k < n_out; ++k) {
        int c  = k * decim;
        int t0 = std::max(-half, -c);
        int t1 = std::min(half, n_in - 1 - c);
        std::complex<float> acc(0, 0);
        for (int t = t0; t <= t1; ++t)
            acc += g[t + half] * in[c + t];
        out[k] = phasor * acc;
        phasor *= rot_step;
        if ((k & 1023) == 0) {                           // guard against drift
            float m = std::abs(phasor);
            if (m > 1e-6f) phasor /= m;
        }
    }
}

} // namespace hf
} // namespace gm
