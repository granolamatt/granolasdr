// ft8decode — pybind11 binding: decode FT8 from real audio at a chosen STFT
// overlap (time_osr, freq_osr), using ft8_lib's proven sync + LDPC + CRC + unpack.
//
// ft8_lib's CPU monitor is stripped in this fork (STFT is done on GPU at runtime),
// so we build the waterfall ourselves from a kiss_fftr STFT into ft8_lib's exact
// layout (mag[block][time_sub][freq_sub][num_bins], uint8 = (dB+120)*2) and hand
// it to ftx_find_candidates + ftx_decode_candidate.
//
// The whole point: time_osr/freq_osr are the experiment knobs. Decode the same
// audio at (4,4) vs (1,1) and compare CRC-valid message sets.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "ft8_lib/fft/kiss_fftr.h"
#include "ft8_lib/ft8/decode.h"
#include "ft8_lib/ft8/message.h"
#include "ft8_lib/ft8/constants.h"
}

namespace py = pybind11;

static constexpr float FT8_SYM_SEC = 0.16f;   // 6.25 baud
static constexpr float TONE_HZ     = 6.25f;

// ---- minimal callsign hash table (for unpacking hashed-callsign messages) ---
static constexpr int HN = 256;
static struct { char call[12]; uint32_t hash; } g_ht[HN];
static void ht_reset() { std::memset(g_ht, 0, sizeof(g_ht)); }
static void ht_save(const char* call, uint32_t n22) {
    uint16_t h10 = (n22 >> 12) & 0x3FF;
    int i = (h10 * 23) % HN;
    for (int n = 0; n < HN; ++n, i = (i + 1) % HN)
        if (g_ht[i].call[0] == '\0') {
            std::strncpy(g_ht[i].call, call, 11); g_ht[i].hash = n22; return;
        }
}
static bool ht_lookup(ftx_callsign_hash_type_t t, uint32_t h, char* call) {
    uint32_t mask = (t == FTX_CALLSIGN_HASH_22_BITS) ? 0x3FFFFF
                  : (t == FTX_CALLSIGN_HASH_12_BITS) ? 0xFFF : 0x3FF;
    uint32_t shift = (t == FTX_CALLSIGN_HASH_22_BITS) ? 0
                   : (t == FTX_CALLSIGN_HASH_12_BITS) ? 10 : 12;
    for (int i = 0; i < HN; ++i)
        if (g_ht[i].call[0] && ((g_ht[i].hash >> shift) & mask) == (h & mask)) {
            std::strcpy(call, g_ht[i].call); return true;
        }
    return false;
}
static ftx_callsign_hash_interface_t g_hash_if = { ht_lookup, ht_save };

// ---- build ft8_lib waterfall from a zero-padded STFT of real audio ----------
struct Waterfall {
    std::vector<uint8_t> mag;
    ftx_waterfall_t wf{};
    int min_bin{0};
};

static Waterfall build_waterfall(const float* x, int n, int sr,
                                 int time_osr, int freq_osr,
                                 float f_min, float f_max) {
    const int block = (int)std::lround(sr * FT8_SYM_SEC);   // samples/symbol
    const int nfft  = block * freq_osr;                     // zero-padded length
    const int hop   = block / time_osr;
    const int half  = nfft / 2 + 1;
    const int min_bin = (int)(f_min / TONE_HZ);             // in 6.25 Hz units
    const int max_bin = (int)(f_max / TONE_HZ);
    const int num_bins = max_bin - min_bin;

    // blocks: last needed sample = (nb-1)*block + (time_osr-1)*hop + block
    int num_blocks = 0;
    while (((num_blocks) * block + (time_osr - 1) * hop + block) <= n) ++num_blocks;
    if (num_blocks < 1) num_blocks = 0;

    // Decode reads up to time_offset(<30) + FT8_NN(79) blocks per candidate;
    // over-allocate zero-padding so edge candidates read zeros, not OOB (the
    // reference decoder does the same via max_blocks > num_blocks).
    const int pad_blocks = num_blocks + FT8_NN + 32;

    Waterfall W;
    W.min_bin = min_bin;
    W.wf.max_blocks = pad_blocks;
    W.wf.num_blocks = num_blocks;
    W.wf.num_bins   = num_bins;
    W.wf.time_osr   = time_osr;
    W.wf.freq_osr   = freq_osr;
    W.wf.block_stride = time_osr * freq_osr * num_bins;
    W.wf.protocol   = FTX_PROTOCOL_FT8;
    if (num_blocks == 0) return W;
    W.mag.assign((size_t)pad_blocks * W.wf.block_stride, 0);

    // Hann window over the symbol (block) samples; zero-pad to nfft.
    std::vector<float> win(block);
    for (int i = 0; i < block; ++i)
        win[i] = 0.5f * (1 - std::cos(2 * (float)M_PI * i / (block - 1)));

    kiss_fftr_cfg cfg = kiss_fftr_alloc(nfft, 0, nullptr, nullptr);
    std::vector<float> buf(nfft, 0.0f);
    std::vector<kiss_fft_cpx> out(half);

    // Pass 1: dB into a float scratch (ft8_lib uses only relative magnitudes,
    // so we reference to the noise floor rather than an absolute full-scale).
    std::vector<float> tdb((size_t)pad_blocks * W.wf.block_stride, -200.0f);
    for (int b = 0; b < num_blocks; ++b) {
        for (int s = 0; s < time_osr; ++s) {
            int start = b * block + s * hop;
            std::fill(buf.begin(), buf.end(), 0.0f);
            for (int i = 0; i < block; ++i) buf[i] = x[start + i] * win[i];
            kiss_fftr(cfg, buf.data(), out.data());
            float* row = tdb.data() + ((size_t)(b * time_osr + s) * freq_osr) * num_bins;
            for (int fs = 0; fs < freq_osr; ++fs) {
                float* dst = row + (size_t)fs * num_bins;
                for (int bin = 0; bin < num_bins; ++bin) {
                    int k = (min_bin + bin) * freq_osr + fs;
                    float re = out[k].r, im = out[k].i;
                    dst[bin] = 10.0f * std::log10(re * re + im * im + 1e-12f);
                }
            }
        }
    }
    kiss_fftr_free(cfg);

    // Noise-floor reference = median over the filled region.
    size_t filled = (size_t)num_blocks * W.wf.block_stride;
    std::vector<float> samp;
    samp.reserve(filled / 7 + 1);
    for (size_t i = 0; i < filled; i += 7) samp.push_back(tdb[i]);
    float ref = 0.0f;
    if (!samp.empty()) {
        std::nth_element(samp.begin(), samp.begin() + samp.size() / 2, samp.end());
        ref = samp[samp.size() / 2];
    }
    // Pass 2: uint8 = (dB - noise_floor)*2 + 40  (0.5 dB/unit, ft8_lib scale).
    for (size_t i = 0; i < tdb.size(); ++i) {
        int v = (int)std::lround((tdb[i] - ref) * 2.0f + 40.0f);
        W.mag[i] = (uint8_t)(v < 0 ? 0 : (v > 255 ? 255 : v));
    }
    return W;
}

// ---- decode one audio window -> list of {text, freq, time, score} -----------
static py::list decode_audio(py::array_t<float, py::array::c_style | py::array::forcecast> audio,
                             int sample_rate, int time_osr, int freq_osr,
                             float f_min, float f_max,
                             int min_score, int max_cand, int ldpc_iters) {
    auto a = audio.unchecked<1>();
    int n = (int)a.shape(0);
    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) x[i] = a(i);

    Waterfall W = build_waterfall(x.data(), n, sample_rate, time_osr, freq_osr, f_min, f_max);
    py::list out;
    if (W.wf.num_blocks == 0) return out;
    W.wf.mag = W.mag.data();   // bind the mag pointer to the (now-final) vector

    ht_reset();
    std::vector<ftx_candidate_t> cands(max_cand);
    int nc = ftx_find_candidates(&W.wf, max_cand, cands.data(), min_score);

    std::vector<std::string> seen;
    for (int i = 0; i < nc; ++i) {
        const ftx_candidate_t* c = &cands[i];
        ftx_message_t msg; ftx_decode_status_t st;
        if (!ftx_decode_candidate(&W.wf, c, ldpc_iters, &msg, &st)) continue;
        char text[64];
        if (ftx_message_decode(&msg, &g_hash_if, text) != FTX_MESSAGE_RC_OK) continue;
        std::string t(text);
        bool dup = false;
        for (auto& s : seen) if (s == t) { dup = true; break; }
        if (dup) continue;
        seen.push_back(t);

        float freq_hz = (W.min_bin + c->freq_offset + (float)c->freq_sub / freq_osr) * TONE_HZ;
        float time_s  = (c->time_offset + (float)c->time_sub / time_osr) * FT8_SYM_SEC;
        py::dict d;
        d["text"] = t; d["freq_hz"] = freq_hz; d["time_sec"] = time_s; d["score"] = (int)c->score;
        out.append(d);
    }
    return out;
}

PYBIND11_MODULE(ft8decode, m) {
    m.doc() = "FT8 decode via ft8_lib at a chosen STFT overlap (time_osr, freq_osr)";
    m.def("decode_audio", &decode_audio,
          py::arg("audio"), py::arg("sample_rate"),
          py::arg("time_osr") = 2, py::arg("freq_osr") = 2,
          py::arg("f_min") = 200.0f, py::arg("f_max") = 3200.0f,
          py::arg("min_score") = 10, py::arg("max_cand") = 440,
          py::arg("ldpc_iters") = 25);
}
