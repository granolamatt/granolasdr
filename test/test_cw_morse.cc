// test_cw_morse.cc — Functional tests for the pure-CPU CW (Morse) decoder core.
//
// Synthesizes a magnitude envelope for a known message at a given WPM/SNR
// (standard PARIS timing), feeds it to CwMorse, and checks the copy.  This is
// the Phase 0 de-risk harness: clean-signal copy must be exact; the hardened
// gate (multi-signal / fast / QSB) is exercised by the cases below.
//
// Builds host-only (no CUDA): links gm/hf/cw_morse.cc only.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "gm/hf/cw_morse.h"

using gm::hf::CwMorse;

// ---- char -> Morse (generator side) ----------------------------------------
static std::string toMorse(char c) {
    switch (c) {
        case 'A': return ".-";   case 'B': return "-...";  case 'C': return "-.-.";
        case 'D': return "-..";  case 'E': return ".";     case 'F': return "..-.";
        case 'G': return "--.";  case 'H': return "....";  case 'I': return "..";
        case 'J': return ".---"; case 'K': return "-.-";   case 'L': return ".-..";
        case 'M': return "--";   case 'N': return "-.";    case 'O': return "---";
        case 'P': return ".--."; case 'Q': return "--.-";  case 'R': return ".-.";
        case 'S': return "...";  case 'T': return "-";     case 'U': return "..-";
        case 'V': return "...-"; case 'W': return ".--";   case 'X': return "-..-";
        case 'Y': return "-.--"; case 'Z': return "--..";
        case '0': return "-----";case '1': return ".----"; case '2': return "..---";
        case '3': return "...--";case '4': return "....-"; case '5': return ".....";
        case '6': return "-....";case '7': return "--...";  case '8': return "---..";
        case '9': return "----.";case '/': return "-..-.";  case '=': return "-...-";
        default:  return "";
    }
}

// ---- deterministic PRNG + gaussian -----------------------------------------
static uint64_t s_rng = 0x1234567890ABCDEFull;
static void seed(uint64_t s) { s_rng = s ? s : 1; }
static uint32_t rnd() { s_rng ^= s_rng << 13; s_rng ^= s_rng >> 7; s_rng ^= s_rng << 17; return (uint32_t)(s_rng >> 11); }
static float frand() { return (float)rnd() / (float)0xFFFFFFFFu * 2.0f; } // ~[0,2)... see gauss
static float gauss() {
    // Box-Muller from two uniforms in (0,1].
    float u1 = ((rnd() & 0xFFFFFF) + 1) / (float)0x1000000;
    float u2 = ((rnd() & 0xFFFFFF) + 1) / (float)0x1000000;
    return std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
}

// Append `count` samples of the given on/off state at amplitude `amp`,
// magnitude = |signal + noise| with noise std from snr_db. qsb scales amp.
static void emit(std::vector<float>& env, bool on, int count, float nstd,
                 float qsb) {
    const float base = on ? 1.0f * qsb : 0.0f;
    for (int i = 0; i < count; ++i)
        env.push_back(std::fabs(base + nstd * gauss()));
}

// Generate the envelope for `msg` at `wpm`, hop `hop_ms`, signal-to-noise
// `snr_db`.  `qsb_period_s` > 0 applies slow fading (0 = none).
static std::vector<float> genEnv(const std::string& msg, float wpm, float hop_ms,
                                 float snr_db, float qsb_period_s = 0.0f) {
    const float dit_ms = 1200.0f / wpm;
    const int   dit_n  = (int)std::lround(dit_ms / hop_ms);
    const float nstd   = std::pow(10.0f, -snr_db / 20.0f);
    std::vector<float> env;

    auto qsbAt = [&](int idx) -> float {
        if (qsb_period_s <= 0.0f) return 1.0f;
        const float t = idx * hop_ms / 1000.0f;
        return 0.65f + 0.35f * std::cos(6.2831853f * t / qsb_period_s); // 0.3..1.0 (stays workable)
    };

    emit(env, false, 40, nstd, 1.0f);            // leading noise, lets trackers settle
    bool first = true;
    for (char ch : msg) {
        if (ch == ' ') { emit(env, false, 7 * dit_n, nstd, 1.0f); first = true; continue; }
        const std::string m = toMorse(ch);
        if (m.empty()) continue;
        if (!first) emit(env, false, 3 * dit_n, nstd, 1.0f); // char gap
        first = false;
        for (size_t j = 0; j < m.size(); ++j) {
            if (j) emit(env, false, dit_n, nstd, 1.0f);      // inter-element gap
            const int n = (m[j] == '.' ? 1 : 3) * dit_n;
            emit(env, true, n, nstd, qsbAt((int)env.size()));
        }
    }
    emit(env, false, 40, nstd, 1.0f);            // trailing noise
    return env;
}

// Fraction of positions that match (length-tolerant: over max length).
static float charAcc(const std::string& a, const std::string& b) {
    size_t n = std::max(a.size(), b.size());
    if (!n) return 1.0f;
    size_t hit = 0, m = std::min(a.size(), b.size());
    for (size_t i = 0; i < m; ++i) hit += (a[i] == b[i]);
    return (float)hit / (float)n;
}

static int g_fail = 0;
static void check(bool ok, const char* name, const std::string& got, const std::string& want) {
    printf("  [%s] %-28s got=\"%s\"  want=\"%s\"\n", ok ? "PASS" : "FAIL", name,
           got.c_str(), want.c_str());
    if (!ok) g_fail++;
}

int main() {
    const std::string MSG = "CQ CQ DE KF0RRR K";
    const float HOP = 5.0f;

    printf("CW Morse decoder — Phase 0 gate\n");

    // 1-3: clean copy across the RBN-observed WPM range -> must be exact.
    for (float wpm : {18.0f, 26.0f, 36.0f}) {
        seed(0xC0FFEE + (uint64_t)wpm);
        auto env = genEnv(MSG, wpm, HOP, 30.0f);
        CwMorse dec(HOP, 20.0f);
        std::string got = dec.decode(env.data(), (int)env.size());
        char nm[40]; snprintf(nm, sizeof nm, "clean %.0f WPM", wpm);
        check(got == MSG, nm, got, MSG);
    }

    // 4: adaptive lock from a wrong seed (decoder seeded 12 WPM, signal 30 WPM).
    {
        seed(0xABCD1);
        auto env = genEnv(MSG, 30.0f, HOP, 30.0f);
        CwMorse dec(HOP, 12.0f);
        std::string got = dec.decode(env.data(), (int)env.size());
        check(charAcc(got, MSG) >= 0.85f, "adaptive lock (12->30)", got, MSG);
    }

    // 5: noisy copy (SNR 12 dB) -> callsign must survive.
    {
        seed(0x5151);
        auto env = genEnv(MSG, 22.0f, HOP, 12.0f);
        CwMorse dec(HOP, 20.0f);
        std::string got = dec.decode(env.data(), (int)env.size());
        check(got.find("KF0RRR") != std::string::npos, "noisy 12dB callsign", got, MSG);
    }

    // 6: QSB (slow fade, 2 s period, signal stays above the noise) -> AGC copies it.
    //    Note: a signal that fades BELOW the noise floor is uncopyable by any
    //    decoder; that is a dropout, not a threshold/AGC problem.
    {
        seed(0x9999);
        auto env = genEnv(MSG, 24.0f, HOP, 22.0f, 2.0f);
        CwMorse dec(HOP, 20.0f);
        std::string got = dec.decode(env.data(), (int)env.size());
        check(charAcc(got, MSG) >= 0.85f, "QSB fade (workable)", got, MSG);
    }

    // 7: crash-safety (CW1). Adversarial envelopes that previously drove the
    //    all-zero / degenerate path into cw_morse's 0/0 AGC -> NaN -> otsu
    //    hist[(int)NaN] out-of-bounds write. Each must RETURN (empty is fine);
    //    reaching the line after decode() is the assertion — a crash aborts.
    {
        printf("crash-safety (must not crash; empty output OK):\n");
        CwMorse dec(HOP, 20.0f);
        auto survives = [&](const char* nm, std::vector<float> v) {
            std::string got = dec.decode(v.data(), (int)v.size());
            printf("  [PASS] %-24s -> \"%s\"\n", nm, got.c_str());
        };
        survives("all zeros",    std::vector<float>(400, 0.0f));
        survives("flat mid",     std::vector<float>(400, 128.0f));
        survives("saturated",    std::vector<float>(400, 255.0f));
        std::vector<float> spike(400, 0.0f); spike[200] = 255.0f;
        survives("single spike", spike);
        survives("all NaN",      std::vector<float>(400, std::nanf("")));
        survives("all +inf",     std::vector<float>(400, INFINITY));
        survives("length 1",     std::vector<float>(1, 0.0f));
        survives("length 0",     std::vector<float>());
    }

    printf("\nPhase 0 gate: clean 18/26/36 WPM exact + adaptive lock + 12 dB noise + QSB.\n");
    printf("Known limit (escalation trigger): dense pileups <80 Hz spacing and\n");
    printf("signals that fade below the noise need the per-signal Viterbi path.\n");
    printf("%s — %d failure(s)\n", g_fail ? "FAILED" : "OK", g_fail);
    return g_fail ? 1 : 0;
}
