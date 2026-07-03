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

static void test_one(const char* msg_text, float f_off, int t_off, float snr_db) {
    ftx_message_t msg;
    if (ftx_message_encode(&msg, nullptr, msg_text) != FTX_MESSAGE_RC_OK) {
        printf("  [FAIL] encode %s\n", msg_text); g_fail++; return;
    }
    uint8_t tones[FT8_NN];
    ft8_encode(msg.payload, tones);

    // Modulate to complex baseband @ 12.8 kHz, continuous phase, with offsets+noise.
    const int flen = kRefineNsym * kRefineNsps + 2048;
    std::vector<std::complex<float>> frame(flen, {0,0});
    const float nstd = std::pow(10.0f, -snr_db/20.0f) * 0.7071f;
    double phase = 0.0;
    for (int p = 0; p < kRefineNsym; ++p) {
        double f = tones[p] * 6.25 + f_off;               // baseband tone freq
        for (int i = 0; i < kRefineNsps; ++i) {
            phase += 2*M_PI*f/kRefineSr;
            int idx = t_off + p*kRefineNsps + i;
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
    printf("  [%s] f_off=%+.1fHz t_off=%+d snr=%.0fdB -> \"%s\"\n",
           ok ? "PASS" : "FAIL", f_off, t_off, snr_db, text);
    if (!ok) g_fail++;
}

int main() {
    const char* MSG = "CQ KF0RRR EM48";
    printf("FT8 refine round-trip\n");
    test_one(MSG,  0.0f,   0, 30.0f);   // clean, aligned
    test_one(MSG,  1.5f, 200, 30.0f);   // freq + time offset (refine must align)
    test_one(MSG, -2.0f,-300, 30.0f);   // negative offsets
    test_one(MSG,  1.0f, 150, 10.0f);   // + noise
    printf("%s — %d failure(s)\n", g_fail ? "FAILED" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
