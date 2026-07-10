// test_refine.cc — host-only round-trip for the shared FT8 refine core.
// Encode a message -> FT8 tones -> modulate to a baseband frame with a deliberate
// freq/time offset + noise -> refine_llr -> ft8_lib LDPC/CRC -> check it decodes.
// Validates the fine freq/time search + LLR extraction without any hardware.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <complex>
#include <string>
#include <vector>

#include "gm/hf/refine.h"

extern "C" {
#include "ft8_lib/ft8/encode.h"
#include "ft8_lib/ft8/message.h"
#include "ft8_lib/ft8/decode.h"
}

using gm::hf::refine_llr;
using gm::hf::kFT8Refine;
using gm::hf::kRefineSr;
using gm::hf::kRefineNsps;
using gm::hf::kRefineNsym;

static uint64_t s_rng = 0xD1CE5EED;
static float gauss() {
    auto u = [&]() { s_rng ^= s_rng<<13; s_rng ^= s_rng>>7; s_rng ^= s_rng<<17;
                     return ((s_rng>>11)&0xFFFFFF)/(float)0x1000000 + 1e-7f; };
    return std::sqrt(-2*std::log(u())) * std::cos(6.2831853f*u());
}

static int g_fail = 0;

static void test_one(const char* msg_text, float f_off, int t_off, float snr_db,
                     float drift_hz_s = 0.0f) {
    ftx_message_t msg;
    if (ftx_message_encode(&msg, nullptr, msg_text) != FTX_MESSAGE_RC_OK) {
        printf("  [FAIL] encode %s\n", msg_text); g_fail++; return;
    }
    uint8_t tones[FT8_NN];
    ft8_encode(msg.payload, tones);

    // Modulate to complex baseband @ 12.8 kHz, continuous phase, with offsets,
    // a linear frequency drift (Hz/s), and noise.
    const int flen = kRefineNsym * kRefineNsps + 2048;
    std::vector<std::complex<float>> frame(flen, {0,0});
    const float nstd = std::pow(10.0f, -snr_db/20.0f) * 0.7071f;
    double phase = 0.0;
    for (int p = 0; p < kRefineNsym; ++p) {
        for (int i = 0; i < kRefineNsps; ++i) {
            int sn = p*kRefineNsps + i;                    // signal sample index
            double f = tones[p]*6.25 + f_off + drift_hz_s*(sn/(double)kRefineSr);
            phase += 2*M_PI*f/kRefineSr;
            int idx = t_off + sn;
            if (idx >= 0 && idx < flen)
                frame[idx] = std::complex<float>(std::cos(phase) + nstd*gauss(),
                                                 std::sin(phase) + nstd*gauss());
        }
    }

    float log174[174];
    refine_llr(frame.data(), flen, kFT8Refine, log174);

    ftx_message_t out; ftx_decode_status_t st;
    char text[64] = {0};
    bool ok = ftx_decode_from_llr(log174, 30, &out, &st)
              && ftx_message_decode(&out, nullptr, text) == FTX_MESSAGE_RC_OK
              && std::string(text) == msg_text;
    printf("  [%s] f_off=%+.1fHz t_off=%+d drift=%+.1fHz/s snr=%.0fdB -> \"%s\"\n",
           ok ? "PASS" : "FAIL", f_off, t_off, drift_hz_s, snr_db, text);
    if (!ok) g_fail++;
}

// Full pipeline DSP: modulate at the 409.6 kHz composite rate at a composite
// frequency, then extract_frame (downconvert+decimate) -> refine_llr -> decode.
static void test_extract(const char* msg_text, float f_comp, float snr_db) {
    ftx_message_t msg;
    ftx_message_encode(&msg, nullptr, msg_text);
    uint8_t tones[FT8_NN];
    ft8_encode(msg.payload, tones);

    const int SR_IN = 409600;
    const int NSPS_IN = SR_IN * 16 / 100;                  // 65536 samples/symbol
    const int n_in = kRefineNsym * NSPS_IN + NSPS_IN;
    std::vector<std::complex<float>> raw(n_in, {0,0});
    const float nstd = std::pow(10.0f, -snr_db/20.0f) * 0.7071f;
    double phase = 0.0;
    for (int p = 0; p < kRefineNsym; ++p) {
        double f = f_comp + tones[p] * 6.25;
        for (int i = 0; i < NSPS_IN; ++i) {
            phase += 2*M_PI*f/SR_IN;
            int idx = p*NSPS_IN + i;
            raw[idx] = std::complex<float>(std::cos(phase) + nstd*gauss(),
                                           std::sin(phase) + nstd*gauss());
        }
    }
    std::vector<std::complex<float>> frame(gm::hf::kRefineFrame);
    gm::hf::extract_frame(raw.data(), n_in, f_comp, SR_IN, frame.data(), gm::hf::kRefineFrame);

    float log174[174];
    refine_llr(frame.data(), gm::hf::kRefineFrame, kFT8Refine, log174);
    ftx_message_t out; ftx_decode_status_t st; char text[64] = {0};
    bool ok = ftx_decode_from_llr(log174, 30, &out, &st)
              && ftx_message_decode(&out, nullptr, text) == FTX_MESSAGE_RC_OK
              && std::string(text) == msg_text;
    printf("  [%s] extract f_comp=%.0fHz snr=%.0fdB -> \"%s\"\n",
           ok ? "PASS" : "FAIL", f_comp, snr_db, text);
    if (!ok) g_fail++;
}

int main() {
    const char* MSG = "CQ KF0RRR EM48";
    printf("FT8 refine round-trip\n");
    test_one(MSG,  0.0f,   0, 30.0f);   // clean, aligned
    test_one(MSG,  1.5f, 200, 30.0f);   // freq + time offset (refine must align)
    test_one(MSG, -2.0f,-300, 30.0f);   // negative offsets
    test_one(MSG,  1.0f, 150, 10.0f);   // + noise
    printf("de-chirp (linear frequency drift — old fixed-df fails at >=1 Hz/s):\n");
    test_one(MSG,  0.0f,   0, 30.0f, 0.3f);   // below the ~0.5 Hz/s budget — still fine
    test_one(MSG,  0.0f,   0, 30.0f, 1.0f);   // 2x budget — needs de-chirp
    test_one(MSG,  0.5f, 100, 20.0f, 2.0f);   // strong drift + offsets + noise
    test_one(MSG, -1.0f,-150, 15.0f,-1.5f);   // negative drift + noise
    printf("extract + refine (409.6 kHz composite -> 12.8 kHz -> decode):\n");
    test_extract(MSG,  73000.0f, 30.0f);  // 20m-like composite freq
    test_extract(MSG, 141234.5f, 15.0f);  // off-bin freq + noise
    printf("%s — %d failure(s)\n", g_fail ? "FAILED" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
