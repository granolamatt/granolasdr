// test_osd.cc — Functional tests for the JS8 (174,87) OSD fallback decoder.
//
// Two modes:
//   (no args)         synthetic: encode random CRC-valid messages with the
//                     generator, add channel noise/erasures to the LLRs, and
//                     verify OSD recovers them and CRC-12 passes.
//   <capture.jsonl>   replay real captured LLRs (js8_llr_*.jsonl) — these were
//                     decoded by BP, so OSD must recover ~all of them.  Also
//                     reports the OSD-on-noise false-accept rate.
//
// Builds host-only (no CUDA): links JS8Osd.cc + JS8Generator.h.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "gm/cuda/JS8Osd.h"
#include "gm/cuda/JS8Generator.h"

// ---- CRC-12 (copied verbatim from gm/hf/js8.cc) ----------------------------
static uint16_t crc12_augmented(const uint8_t* data, int nbytes) {
    uint16_t crc = 0;
    for (int i = 0; i < nbytes; ++i)
        for (int b = 7; b >= 0; --b) {
            int inp = (data[i] >> b) & 1;
            int quotient = (crc >> 11) & 1;
            crc = ((crc << 1) | inp) & 0xFFF;
            if (quotient) crc ^= 0xC06;
        }
    return crc ^ 42;
}
static bool checkCRC12(const uint8_t* bits) {
    uint8_t packed[11] = {};
    for (int i = 0; i < 87; ++i)
        if (bits[i]) packed[i / 8] |= (uint8_t)(1 << (7 - (i & 7)));
    uint16_t rx = ((uint16_t)(packed[9] & 0x1F) << 7) | (packed[10] >> 1);
    packed[9] &= 0xE0; packed[10] = 0x00;
    return rx == crc12_augmented(packed, 11);
}
// Compute and write the 12 CRC bits (positions 75..86) of an 87-bit message.
static void setCRC12(uint8_t* bits) {
    uint8_t packed[11] = {};
    for (int i = 0; i < 75; ++i)
        if (bits[i]) packed[i / 8] |= (uint8_t)(1 << (7 - (i & 7)));
    packed[9] &= 0xE0; packed[10] = 0x00;
    uint16_t crc = crc12_augmented(packed, 11);   // 12 bits
    for (int k = 0; k < 12; ++k)
        bits[75 + k] = (crc >> (11 - k)) & 1;
}

// ---- Deterministic PRNG (no <random> ordering surprises) --------------------
static uint64_t s_rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd() { s_rng ^= s_rng << 13; s_rng ^= s_rng >> 7; s_rng ^= s_rng << 17; return (uint32_t)s_rng; }
static float frand() { return (float)(rnd() >> 8) / (float)(1 << 24); } // [0,1)

// Encode an 87-bit message into a 174-bit codeword via the generator.
static void encode(const uint8_t* msg, uint8_t* cw) {
    memset(cw, 0, 174);
    for (int r = 0; r < 87; ++r)
        if (msg[r])
            for (int c = 0; c < 174; ++c) cw[c] ^= g_js8_gen[r][c];
}

// ---- Synthetic test: recovery under noise + erasures ------------------------
static int synthetic() {
    const int TRIALS = 2000;
    int recovered = 0, crc_ok = 0;

    for (int t = 0; t < TRIALS; ++t) {
        uint8_t msg[87];
        for (int i = 0; i < 75; ++i) msg[i] = rnd() & 1;
        setCRC12(msg);
        uint8_t cw[174];
        encode(msg, cw);
        // sanity: codeword is systematic in the message positions
        for (int i = 0; i < 87; ++i) if (cw[87 + i] != msg[i]) { printf("[FAIL] encode not systematic\n"); return 1; }

        // Channel: strong LLRs, then corrupt a handful of bits and erase a few.
        float llr[174];
        for (int i = 0; i < 174; ++i) llr[i] = cw[i] ? 6.0f : -6.0f;
        int nflip = 6;      // hard-decision errors OSD-2 should fix
        for (int f = 0; f < nflip; ++f) {
            int p = rnd() % 174;
            llr[p] = -llr[p] * (0.3f + 0.7f * frand());  // wrong sign, low reliability
        }
        for (int e = 0; e < 4; ++e) llr[rnd() % 174] = 0.0f;  // erasures

        uint8_t xhat[174];
        float dist = 0.0f;
        js8_osd_decode(llr, 2, xhat, &dist);

        if (memcmp(xhat, cw, 174) == 0) ++recovered;
        if (checkCRC12(xhat + 87)) ++crc_ok;
    }

    printf("[synthetic] %d trials | exact-recover %d (%.1f%%) | CRC-pass %d (%.1f%%)\n",
           TRIALS, recovered, 100.0 * recovered / TRIALS, crc_ok, 100.0 * crc_ok / TRIALS);

    // False-alarm probe: pure-noise LLRs should almost never pass CRC.
    int fa = 0;
    const int NOISE = 20000;
    for (int t = 0; t < NOISE; ++t) {
        float llr[174];
        for (int i = 0; i < 174; ++i) llr[i] = (frand() - 0.5f) * 12.0f;
        uint8_t xhat[174];
        js8_osd_decode(llr, 2, xhat, nullptr);
        if (checkCRC12(xhat + 87)) ++fa;
    }
    printf("[false-alarm] %d noise inputs | CRC false-accept %d (%.4f%%, ~expected %.4f%%)\n",
           NOISE, fa, 100.0 * fa / NOISE, 100.0 / 4096.0);

    bool pass = (recovered >= (int)(0.95 * TRIALS)) && (crc_ok >= (int)(0.95 * TRIALS));
    printf("\n%s synthetic OSD-2 recovery\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

// ---- Capture replay: parse "llr":[...] arrays from a jsonl file -------------
static bool parse_llr(const std::string& line, float* llr) {
    size_t p = line.find("\"llr\"");
    if (p == std::string::npos) return false;
    p = line.find('[', p);
    if (p == std::string::npos) return false;
    size_t end = line.find(']', p);
    if (end == std::string::npos) return false;
    int n = 0;
    const char* s = line.c_str() + p + 1;
    const char* e = line.c_str() + end;
    while (s < e && n < 174) {
        char* next = nullptr;
        float v = strtof(s, &next);
        if (next == s) break;
        llr[n++] = v;
        s = next;
        while (s < e && (*s == ',' || *s == ' ')) ++s;
    }
    return n == 174;
}

static int replay(const char* path) {
    std::ifstream in(path);
    if (!in) { printf("[FAIL] cannot open %s\n", path); return 1; }
    std::string line;
    int total = 0, crc_ok = 0;
    while (std::getline(in, line)) {
        float llr[174];
        if (!parse_llr(line, llr)) continue;
        ++total;
        uint8_t xhat[174];
        js8_osd_decode(llr, 2, xhat, nullptr);
        if (checkCRC12(xhat + 87)) ++crc_ok;
    }
    printf("[replay %s] %d captures | OSD-2 CRC-pass %d (%.1f%%)\n",
           path, total, crc_ok, total ? 100.0 * crc_ok / total : 0.0);
    // Captures were BP-decoded; OSD must recover the large majority.
    bool pass = total > 0 && crc_ok >= (int)(0.90 * total);
    printf("\n%s capture replay\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc > 1) return replay(argv[1]);
    return synthetic();
}
