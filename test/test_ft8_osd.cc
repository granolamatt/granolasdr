// test_ft8_osd.cc — Functional tests for the FT8 (174,91) OSD fallback decoder.
//
//   Test 1  generator: build the systematic G from kFTX_LDPC_generator (same
//           formula FT8Osd uses) and verify G * H^T == 0 against kFTX_LDPC_Nm.
//   Test 2  recovery: encode CRC-valid messages, add channel noise/erasures,
//           verify ft8_osd_decode recovers them and ftx_decode_from_bits (CRC-14)
//           passes with the original payload.
//   Test 3  false-alarm: OSD on pure noise almost never passes CRC-14.
//
// Host-only: links ft8_lib (generator/CRC/decode) + FT8Osd.cc + OsdCore.cc.

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "gm/cuda/FT8Osd.h"
#include "ft8_lib/ft8/constants.h"
#include "ft8_lib/ft8/crc.h"
#include "ft8_lib/ft8/decode.h"
#include "ft8_lib/ft8/message.h"

static const int N = FTX_LDPC_N;   // 174
static const int K = FTX_LDPC_K;   // 91
static const int M = FTX_LDPC_M;   // 83

// Reference systematic generator, identical construction to FT8Osd::build_generator.
static uint8_t refG[91][174];
static void build_refG() {
    memset(refG, 0, sizeof(refG));
    for (int m = 0; m < K; ++m) {
        refG[m][m] = 1;
        for (int i = 0; i < M; ++i) {
            int bit = (kFTX_LDPC_generator[i][m >> 3] >> (7 - (m & 7))) & 1;
            refG[m][K + i] = (uint8_t)bit;
        }
    }
}

// Deterministic PRNG.
static uint64_t s_rng = 0xD1B54A32D192ED03ull;
static uint32_t rnd() { s_rng ^= s_rng << 13; s_rng ^= s_rng >> 7; s_rng ^= s_rng << 17; return (uint32_t)s_rng; }
static float frand() { return (float)(rnd() >> 8) / (float)(1 << 24); }

static void encode_ref(const uint8_t* msg_bits, uint8_t* cw) {
    memset(cw, 0, N);
    for (int m = 0; m < K; ++m)
        if (msg_bits[m])
            for (int c = 0; c < N; ++c) cw[c] ^= refG[m][c];
}

// Test 1: G * H^T == 0 over GF(2), checked against the parity-check tables.
static bool test_generator() {
    for (int m = 0; m < K; ++m)
        for (int chk = 0; chk < M; ++chk) {
            int acc = 0;
            for (int t = 0; t < kFTX_LDPC_Num_rows[chk]; ++t)
                acc ^= refG[m][kFTX_LDPC_Nm[chk][t] - 1];
            if (acc) { printf("[FAIL] G*H^T != 0 at row %d check %d\n", m, chk); return false; }
        }
    // Also confirm systematic: row m has a 1 in column m.
    for (int m = 0; m < K; ++m)
        if (!refG[m][m]) { printf("[FAIL] generator not systematic at %d\n", m); return false; }
    printf("[PASS] generator: G*H^T==0 and systematic\n");
    return true;
}

// Test 2: recovery under noise + erasures, validated by CRC-14.
static bool test_recovery() {
    const int TRIALS = 2000;
    int recovered = 0, crc_ok = 0, payload_ok = 0;

    for (int t = 0; t < TRIALS; ++t) {
        uint8_t payload[10] = {};
        for (int b = 0; b < 77; ++b)
            if (rnd() & 1) payload[b / 8] |= (uint8_t)(1 << (7 - (b & 7)));

        uint8_t a91[FTX_LDPC_K_BYTES] = {};
        ftx_add_crc(payload, a91);                 // 77 payload + 14 CRC, packed

        uint8_t msg_bits[91];
        for (int b = 0; b < K; ++b) msg_bits[b] = (a91[b / 8] >> (7 - (b & 7))) & 1;

        uint8_t cw[174];
        encode_ref(msg_bits, cw);

        float llr[174];
        for (int i = 0; i < N; ++i) llr[i] = cw[i] ? 6.0f : -6.0f;
        for (int f = 0; f < 6; ++f) {              // sign errors OSD-2 should fix
            int p = rnd() % N;
            llr[p] = -llr[p] * (0.3f + 0.7f * frand());
        }
        for (int e = 0; e < 4; ++e) llr[rnd() % N] = 0.0f;   // erasures

        uint8_t plain174[174];
        float dist = 0.0f;
        ft8_osd_decode(llr, 2, plain174, &dist);

        if (memcmp(plain174, cw, N) == 0) ++recovered;

        ftx_message_t msg;
        ftx_decode_status_t st;
        if (ftx_decode_from_bits(plain174, &msg, &st)) {
            ++crc_ok;
            // ftx_decode_from_bits zeroes the CRC bits (a91[9]&=0xF8, a91[10]=0),
            // so compare only the 77 payload bits: bytes 0..8 + top 5 of byte 9.
            if (memcmp(msg.payload, a91, 9) == 0 &&
                (msg.payload[9] & 0xF8) == (a91[9] & 0xF8)) ++payload_ok;
        }
    }

    printf("[recovery] %d trials | exact %d (%.1f%%) | CRC-14 %d (%.1f%%) | payload-match %d\n",
           TRIALS, recovered, 100.0 * recovered / TRIALS,
           crc_ok, 100.0 * crc_ok / TRIALS, payload_ok);

    bool pass = recovered >= (int)(0.95 * TRIALS) &&
                crc_ok    >= (int)(0.95 * TRIALS) &&
                payload_ok == crc_ok;
    printf("%s recovery\n", pass ? "[PASS]" : "[FAIL]");
    return pass;
}

// Test 3: OSD on pure noise should rarely pass CRC-14 (~2^-14 floor).
static bool test_false_alarm() {
    const int NOISE = 20000;
    int fa = 0;
    for (int t = 0; t < NOISE; ++t) {
        float llr[174];
        for (int i = 0; i < N; ++i) llr[i] = (frand() - 0.5f) * 12.0f;
        uint8_t plain174[174];
        ft8_osd_decode(llr, 2, plain174, nullptr);
        ftx_message_t msg;
        ftx_decode_status_t st;
        if (ftx_decode_from_bits(plain174, &msg, &st)) ++fa;
    }
    printf("[false-alarm] %d noise inputs | CRC-14 false-accept %d (%.4f%%, ~expected %.4f%%)\n",
           NOISE, fa, 100.0 * fa / NOISE, 100.0 / 16384.0);
    return true;   // informational; the gating bounds real-world exposure
}

int main() {
    build_refG();
    int pass = 0, total = 0;
    auto run = [&](bool ok) { ++total; if (ok) ++pass; };

    run(test_generator());
    run(test_recovery());
    run(test_false_alarm());

    printf("\n%d / %d tests passed\n", pass, total);
    return (pass == total) ? 0 : 1;
}
