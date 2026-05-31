// Validate JS8 Fast-mode decode chain: synthetic signal → STFT → MODIFIED Costas → LDPC → CRC-12.
//
// Generates a synthetic 10-second signal (≥100 "ring slots" at time_osr=2) from the known
// codeword for "KD9ABCTEST00", decodes it, and asserts at least one correct decode.
//
// Can also read a real WAV file: ./test_js8_fast_wav [file.wav [min_score]]
//
// D6/D8 ring-slot requirement: 10s × 12 kHz / NSTEP(600) = 200 STFT steps = 100 symbols
// (cap_blocks=100 for Fast).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>

extern "C" {
#include "ft8_lib/fft/kiss_fft.h"
}
#include "gm/cuda/JS8LdpcCuda.h"

// ---- Fast-mode constants at 12 kHz ----------------------------------------
static constexpr int   NSPS   = 1200;              // samples/symbol (100 ms)
static constexpr int   NFOS   = 2;                 // freq oversampling
static constexpr int   NSSY   = 2;                 // time oversampling (time_osr=2)
static constexpr int   NFFT   = NSPS * NFOS;       // 2400
static constexpr int   NSTEP  = NSPS / NSSY;       // 600
static constexpr int   NN     = 79;                // symbols per frame
static constexpr int   ND     = 58;                // data symbols
static constexpr int   SR     = 12000;
static constexpr float DF     = (float)SR / NFFT;  // 5.0 Hz/bin
static constexpr float TONE_HZ = (float)SR / NSPS; // 10.0 Hz tone spacing
static constexpr int   BINS_PER_TONE = NFFT / NSPS; // 2

// MODIFIED Costas arrays (Fast/Slow/Turbo/Ultra) — block-dependent.
static const int COSTAS_MODIFIED[3][7] = {
    {0, 6, 2, 3, 5, 4, 1},
    {1, 5, 0, 2, 3, 6, 4},
    {2, 5, 0, 6, 4, 1, 3},
};
static const int COSTAS_POS[3] = {0, 36, 72};

// ---- CRC-12 (poly 0xC06, augmented, XOR 42) --------------------------------
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

static bool checkCRC12(const uint8_t* decoded_bits) {
    uint8_t packed[11] = {};
    for (int i = 0; i < 87; ++i)
        if (decoded_bits[i])
            packed[i / 8] |= (uint8_t)(1 << (7 - (i & 7)));
    uint16_t received = ((uint16_t)(packed[9] & 0x1F) << 7) | (packed[10] >> 1);
    packed[9] &= 0xE0; packed[10] = 0x00;
    return received == crc12_augmented(packed, 11);
}

// ---- Message extract -------------------------------------------------------
static const char kAlphabet[] =
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-+";

static std::string extractMsg(const uint8_t* bits) {
    char out[13] = {};
    for (int i = 0; i < 12; ++i) {
        int w = 0;
        for (int b = 0; b < 6; ++b) w = (w << 1) | (bits[i*6+b] & 1);
        if (w > 63) return {};
        out[i] = kAlphabet[w];
    }
    return std::string(out, 12);
}

// ---- Nuttall window --------------------------------------------------------
static void nuttall(float* w, int n) {
    static const float a[4] = {0.3635819f, 0.4891775f, 0.1365995f, 0.0106411f};
    for (int i = 0; i < n; ++i) {
        float x = (float)(2 * M_PI * i) / (n - 1);
        w[i] = a[0] - a[1]*cosf(x) + a[2]*cosf(2*x) - a[3]*cosf(3*x);
    }
}

// ---- Synthetic signal generation -------------------------------------------
// Generates a 79-symbol JS8 Fast frame from a 174-bit LDPC codeword.
// base_hz: audio frequency of tone 0 (Hz).  Returns mono float samples.
static std::vector<float> genFastFrame(const uint8_t cw174[174], float base_hz) {
    std::vector<float> frame(NN * NSPS, 0.0f);

    // Map JS8 symbol position → tone index, then generate sinusoid.
    for (int sym = 0; sym < NN; ++sym) {
        int tone = -1;
        // Costas blocks
        if (sym < 7)           tone = COSTAS_MODIFIED[0][sym];
        else if (sym < 36)     { int d = sym - 7;  tone = ((cw174[d*3]&1)<<2)|((cw174[d*3+1]&1)<<1)|(cw174[d*3+2]&1); }
        else if (sym < 43)     tone = COSTAS_MODIFIED[1][sym - 36];
        else if (sym < 72)     { int d = 29 + (sym-43); tone = ((cw174[d*3]&1)<<2)|((cw174[d*3+1]&1)<<1)|(cw174[d*3+2]&1); }
        else                   tone = COSTAS_MODIFIED[2][sym - 72];

        float freq = base_hz + tone * TONE_HZ;
        float phase = 0.0f;
        for (int i = 0; i < NSPS; ++i) {
            frame[sym * NSPS + i] = cosf(2.0f * M_PI * freq * i / SR + phase);
        }
    }
    return frame;
}

// Known codeword for "KD9ABCTEST00" (verified by test_js8_wav.cc LDPC self-test).
// Identical codeword valid for all JS8 modes (LDPC(174,87) is mode-independent).
static const uint8_t kKnownCw174[174] = {
    1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,0,1,0,1,
    0,1,0,1,1,0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,
    0,0,1,1,0,1,1,1,1,0,0,1,1,0,1,1,1,1,0,1,
    0,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,1,0,0,0,
    1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,1,1,0,1,0,
    0,1,0,0,1,0,0,1,0,1,0,0,0,1,0,1,1,0,0,1,
    1,0,0,0,1,1,1,0,1,0,0,1,1,1,0,0,1,1,1,0,
    0,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,1,1,1,0,0,1,0,1,0,0,0,
};

// ---- WAV reader (16-bit PCM, mono) -----------------------------------------
static std::vector<float> readWav(const char* path, int* rate_out) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    char riff[4]; fread(riff, 1, 4, f);
    if (memcmp(riff, "RIFF", 4)) { fprintf(stderr, "Not a WAV\n"); exit(1); }
    uint32_t file_sz; fread(&file_sz, 4, 1, f);
    char wave[4]; fread(wave, 1, 4, f);
    uint16_t channels = 0, bits = 0;
    uint32_t rate = 0, data_sz = 0;
    while (!feof(f)) {
        char id[4]; uint32_t sz;
        if (fread(id, 1, 4, f) != 4) break;
        fread(&sz, 4, 1, f);
        if (!memcmp(id, "fmt ", 4)) {
            uint16_t fmt; fread(&fmt, 2, 1, f);
            fread(&channels, 2, 1, f);
            fread(&rate, 4, 1, f);
            uint32_t brate; fread(&brate, 4, 1, f);
            uint16_t align; fread(&align, 2, 1, f);
            fread(&bits, 2, 1, f);
            fseek(f, sz - 16, SEEK_CUR);
        } else if (!memcmp(id, "data", 4)) {
            data_sz = sz;
            break;
        } else {
            fseek(f, sz, SEEK_CUR);
        }
    }
    if (bits != 16 || channels < 1) { fprintf(stderr, "Need 16-bit PCM\n"); exit(1); }
    *rate_out = (int)rate;
    int n = data_sz / (channels * 2);
    std::vector<float> s(n);
    std::vector<int16_t> buf(n * channels);
    fread(buf.data(), 2, n * channels, f);
    fclose(f);
    for (int i = 0; i < n; ++i) s[i] = buf[i * channels] / 32768.0f;
    return s;
}

// ---- Self-test: perfect LLR decode -----------------------------------------
static void test_ldpc_perfect_llr() {
    float llr[174];
    for (int i = 0; i < 174; ++i) llr[i] = kKnownCw174[i] ? 20.0f : -20.0f;
    uint8_t xhat[174];
    bool ok = js8_ldpc_decode_cpu(llr, xhat);
    printf("[SELF-TEST] Perfect LLR: ldpc=%d", (int)ok);
    if (ok) {
        bool crc = checkCRC12(xhat + 87);
        std::string msg = extractMsg(xhat + 87);
        printf("  crc=%d  msg='%s'\n", (int)crc, msg.c_str());
    } else {
        printf("  FAIL\n");
    }
}

// ---- Decode samples (STFT → MODIFIED Costas scan → LLR → LDPC → CRC) ------
static int decodeSamples(const std::vector<float>& samples, float min_score) {
    int total = (int)samples.size();
    int num_steps = (total - NFFT) / NSTEP + 1;
    int num_bins  = NFFT / 2 + 1;
    printf("Audio: %d samples (%.2f s), %d STFT steps, %d bins\n",
           total, (double)total / SR, num_steps, num_bins);
    printf("Ring-slot coverage: %d steps / %d NSSY = %d symbols (need ≥100)\n",
           num_steps, NSSY, num_steps / NSSY);

    // STFT
    std::vector<float> win(NFFT);
    nuttall(win.data(), NFFT);
    kiss_fft_cfg cfg = kiss_fft_alloc(NFFT, 0, nullptr, nullptr);
    std::vector<kiss_fft_cpx> in_buf(NFFT), out_buf(NFFT);
    std::vector<float> mag(num_steps * num_bins, 0.0f);

    for (int step = 0; step < num_steps; ++step) {
        int off = step * NSTEP;
        for (int i = 0; i < NFFT; ++i) {
            in_buf[i].r = (off + i < total) ? samples[off + i] * win[i] : 0.0f;
            in_buf[i].i = 0.0f;
        }
        kiss_fft(cfg, in_buf.data(), out_buf.data());
        for (int b = 0; b < num_bins; ++b) {
            float re = out_buf[b].r, im = out_buf[b].i;
            mag[step * num_bins + b] = sqrtf(re*re + im*im);
        }
    }
    free(cfg);
    printf("STFT done. DF=%.2f Hz/bin, tone spacing=%.1f Hz (%d bins/tone)\n",
           (double)DF, (double)TONE_HZ, BINS_PER_TONE);

    // MODIFIED Costas scan
    int frame_steps = NN * NSSY;
    struct Cand { int t0; int fo_bin; float score; };
    std::vector<Cand> candidates;

    int fo_min = (int)(100.0f / DF);
    int fo_max = (int)(3500.0f / DF);

    for (int t0 = 0; t0 + frame_steps < num_steps; ++t0) {
        for (int fo = fo_min; fo <= fo_max; ++fo) {
            float score = 0.0f;
            bool ok = true;
            for (int c = 0; c < 3 && ok; ++c) {
                int sym_base = t0 + COSTAS_POS[c] * NSSY;
                for (int s = 0; s < 7; ++s) {
                    int step = sym_base + s * NSSY;
                    if (step >= num_steps) { ok = false; break; }
                    int tone_bin = fo + COSTAS_MODIFIED[c][s] * BINS_PER_TONE;
                    if (tone_bin + BINS_PER_TONE >= num_bins) { ok = false; break; }
                    float m = 0.0f;
                    for (int b = 0; b < BINS_PER_TONE; ++b)
                        m = std::max(m, mag[step * num_bins + tone_bin + b]);
                    score += m;
                }
            }
            if (ok && score > min_score)
                candidates.push_back({t0, fo, score});
        }
    }

    printf("MODIFIED Costas scan: %zu raw candidates (min_score=%.1f)\n",
           candidates.size(), (double)min_score);

    std::sort(candidates.begin(), candidates.end(),
              [](const Cand& a, const Cand& b){ return a.score > b.score; });

    // Deduplicate
    std::vector<bool> used(candidates.size(), false);
    std::vector<Cand> deduped;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (used[i]) continue;
        deduped.push_back(candidates[i]);
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (abs(candidates[j].t0 - candidates[i].t0) <= NSSY &&
                abs(candidates[j].fo_bin - candidates[i].fo_bin) <= BINS_PER_TONE)
                used[j] = true;
        }
    }
    printf("After dedup: %zu candidates\n", deduped.size());

    int n_decoded = 0;
    const int MAX_CANDS = std::min((int)deduped.size(), 200);

    auto computeLLR = [&](int fo_bin, int t0, float* llr) {
        for (int k = 0; k < ND; ++k) {
            // Skip Costas positions: 0-6 (block 0), 36-42 (block 1), 72-78 (block 2)
            int sym = (k < 29) ? k + 7 : (k < 58) ? k + 14 : k + 21;
            if (k >= 29) sym = k + 14;
            int step = t0 + sym * NSSY;
            if (step >= num_steps) {
                llr[k*3] = llr[k*3+1] = llr[k*3+2] = 0.0f;
                continue;
            }
            float s[8];
            for (int j = 0; j < 8; ++j) {
                int bin = fo_bin + j * BINS_PER_TONE;
                s[j] = (bin >= 0 && bin < num_bins) ? mag[step * num_bins + bin] : 0.0f;
            }
            // Natural binary: no Gray coding. Max-log LLR.
            auto mx4 = [](float a, float b, float c, float d) {
                return std::max({a, b, c, d});
            };
            llr[k*3+0] = mx4(s[4],s[5],s[6],s[7]) - mx4(s[0],s[1],s[2],s[3]);
            llr[k*3+1] = mx4(s[2],s[3],s[6],s[7]) - mx4(s[0],s[1],s[4],s[5]);
            llr[k*3+2] = mx4(s[1],s[3],s[5],s[7]) - mx4(s[0],s[2],s[4],s[6]);
        }
    };

    for (int ci = 0; ci < MAX_CANDS; ++ci) {
        const Cand& c = deduped[ci];
        float llr[174] = {};
        uint8_t xhat[174] = {};
        bool ldpc_ok = false;
        int best_fo = c.fo_bin;

        for (int dfo = 0; dfo < BINS_PER_TONE; ++dfo) {
            float try_llr[174];
            computeLLR(c.fo_bin + dfo, c.t0, try_llr);
            uint8_t try_xhat[174];
            if (js8_ldpc_decode_cpu(try_llr, try_xhat) && !ldpc_ok) {
                ldpc_ok = true;
                best_fo = c.fo_bin + dfo;
                std::copy(try_llr, try_llr + 174, llr);
                std::copy(try_xhat, try_xhat + 174, xhat);
            }
        }

        if (ci < 5) {
            bool crc_ok = ldpc_ok && checkCRC12(xhat+87);
            printf("  cand %d: ldpc=%d crc=%d  t0=%d fo=%d (%.0fHz) score=%.2f\n",
                   ci, (int)ldpc_ok, (int)crc_ok, c.t0, c.fo_bin,
                   (double)c.fo_bin * DF, (double)c.score);
        }
        if (!ldpc_ok) continue;
        if (!checkCRC12(xhat + 87)) continue;
        std::string msg = extractMsg(xhat + 87);
        if (msg.empty()) continue;
        float freq_hz = best_fo * DF;
        float time_s  = (float)c.t0 * NSTEP / SR;
        printf("DECODED: %-12s  freq=%7.1f Hz  time=%+.3f s  score=%.1f\n",
               msg.c_str(), (double)freq_hz, (double)time_s, (double)c.score);
        ++n_decoded;
    }

    printf("\n%d message(s) decoded from %d candidates checked.\n", n_decoded, MAX_CANDS);
    return n_decoded;
}

// ---- main ------------------------------------------------------------------
int main(int argc, char* argv[]) {
    test_ldpc_perfect_llr();

    std::vector<float> samples;
    float min_score = 5.0f;

    if (argc > 1) {
        // Real WAV file path provided
        int rate = 0;
        samples = readWav(argv[1], &rate);
        if (rate != SR) { fprintf(stderr, "Expected %d Hz, got %d\n", SR, rate); return 1; }
        if (argc > 2) min_score = atof(argv[2]);
        printf("WAV: %s  %d Hz  %.2f s\n", argv[1], rate, (double)samples.size() / SR);
    } else {
        // Synthetic signal: 10 s of silence with one JS8 Fast frame at t=1s, f=1500Hz.
        // 10 s × 12000 Hz = 120000 samples = 200 STFT steps = 100 "ring slots" at time_osr=2.
        printf("Generating synthetic JS8 Fast signal (KD9ABCTEST00 @ 1500 Hz, t=1s)\n");
        const int TOTAL = 120000;  // 10 s at 12 kHz (cap_blocks=100 coverage)
        const float BASE_HZ = 1500.0f;
        const int SIGNAL_OFFSET = SR;  // 1 second silence guard

        samples.resize(TOTAL, 0.0f);
        std::vector<float> frame = genFastFrame(kKnownCw174, BASE_HZ);
        for (int i = 0; i < (int)frame.size() && SIGNAL_OFFSET + i < TOTAL; ++i)
            samples[SIGNAL_OFFSET + i] = frame[i] * 0.5f;
        min_score = 50.0f;  // tight threshold for clean synthetic signal
    }

    int n = decodeSamples(samples, min_score);

    if (n == 0) {
        printf("FAIL — no messages decoded\n");
        return 1;
    }
    printf("PASS — %d message(s) decoded\n", n);
    return 0;
}
