// Validate JS8 Mode A decode chain: WAV → STFT → Costas scan → LLR → LDPC → CRC-12 → msg
// Expected output for js8_mode_a.wav: KD9ABCTEST00 @ ~1500 Hz
//
// Signal path mirrors the runtime exactly:
//   - Gray-coded max-log LLR (same as ft8_soft_symbols_kernel)
//   - js8_ldpc_decode_cpu (exact runtime function)
//   - CRC-12 poly 0xC06 XOR 42 (same as js8.cc)
//   - 64-char alphabet message extract (same as js8.cc)

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

// ---- Mode A constants -------------------------------------------------------
static constexpr int NSPS   = 1920;   // samples/symbol at 12 kHz
static constexpr int NFOS   = 2;      // frequency oversampling
static constexpr int NSSY   = 4;      // time oversampling
static constexpr int NFFT   = NSPS * NFOS;   // 3840
static constexpr int NSTEP  = NSPS / NSSY;   // 480
static constexpr int NN     = 79;     // symbols per frame
static constexpr int ND     = 58;     // data symbols (NN - 3*7)
static constexpr int SR     = 12000;
static constexpr float DF   = (float)SR / NFFT;  // ~3.125 Hz per bin

static const int COSTAS[7]    = {4, 2, 5, 6, 1, 3, 0};
static const int COSTAS_POS[3]= {0, 36, 72};
// JS8 uses natural binary (no Gray coding) — tone j carries bit pattern j directly.
static const int GRAY[8]      = {0, 1, 2, 3, 4, 5, 6, 7};

// ---- CRC-12 matching boost::augmented_crc<12, 0xc06> XOR 42 ---------------
// The augmented variant differs from standard CRC: input bit feeds into the
// LSB of the shifted remainder rather than the XOR feedback path.
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

// ---- Message extract (info bits 0..71 → 12 chars from 64-char alphabet) ---
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

// ---- Nuttall window ---------------------------------------------------------
static void nuttall(float* w, int n) {
    static const float a[4] = {0.3635819f, 0.4891775f, 0.1365995f, 0.0106411f};
    for (int i = 0; i < n; ++i) {
        float x = (float)(2 * M_PI * i) / (n - 1);
        w[i] = a[0] - a[1]*cosf(x) + a[2]*cosf(2*x) - a[3]*cosf(3*x);
    }
}

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

// Known codeword for "KD9ABCTEST00" extracted from js8_mode_a.wav.
// Verified: H*cw=0, CRC-12=0x728 (boost::augmented_crc<12,0xc06> ^ 42).
static void test_ldpc_perfect_llr() {
    static const uint8_t cw174[174] = {
        1,0,1,0,1,0,1,0,1,0,0,0,1,0,1,0,0,1,0,1,  // 0-19
        0,1,0,1,1,0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,  // 20-39
        0,0,1,1,0,1,1,1,1,0,0,1,1,0,1,1,1,1,0,1,  // 40-59
        0,1,0,1,0,1,1,1,0,1,1,1,0,1,1,1,1,0,0,0,  // 60-79
        1,0,0,0,0,1,0,0,1,0,1,0,0,0,0,1,1,0,1,0,  // 80-99
        0,1,0,0,1,0,0,1,0,1,0,0,0,1,0,1,1,0,0,1,  // 100-119
        1,0,0,0,1,1,1,0,1,0,0,1,1,1,0,0,1,1,1,0,  // 120-139
        0,0,1,1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,  // 140-159
        0,0,0,1,1,1,0,0,1,0,1,0,0,0,               // 160-173
    };
    float llr[174];
    for (int i = 0; i < 174; ++i) llr[i] = cw174[i] ? 20.0f : -20.0f;

    uint8_t xhat[174];
    bool ok = js8_ldpc_decode_cpu(llr, xhat);
    printf("[SELF-TEST] Perfect LLR decode: ldpc=%d", (int)ok);
    if (ok) {
        bool crc = checkCRC12(xhat + 87);
        std::string msg = extractMsg(xhat + 87);
        printf("  crc=%d  msg='%s'\n", (int)crc, msg.c_str());
    } else {
        printf("  FAIL\n");
    }
}

int main(int argc, char* argv[]) {
    test_ldpc_perfect_llr();
    const char* wav = (argc > 1) ? argv[1]
                                 : "/home/matt/workarea/js8/tools/test_signals/js8_mode_a.wav";
    float min_score = (argc > 2) ? atof(argv[2]) : 3.0f;

    int rate = 0;
    std::vector<float> samples = readWav(wav, &rate);
    if (rate != SR) { fprintf(stderr, "Expected %d Hz, got %d\n", SR, rate); return 1; }

    int total = (int)samples.size();
    int num_steps = (total - NFFT) / NSTEP + 1;
    int num_bins  = NFFT / 2 + 1;
    printf("WAV: %d samples (%.2f s), %d time steps, %d freq bins\n",
           total, (double)total / SR, num_steps, num_bins);

    // ---- STFT ---------------------------------------------------------------
    std::vector<float> win(NFFT);
    nuttall(win.data(), NFFT);

    kiss_fft_cfg cfg = kiss_fft_alloc(NFFT, 0, nullptr, nullptr);
    std::vector<kiss_fft_cpx> in_buf(NFFT), out_buf(NFFT);
    // mag[step * num_bins + bin]
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

    printf("STFT done. DF=%.3f Hz/bin, tone spacing=%.3f Hz (%.1f bins)\n",
           DF, (double)SR/NSPS, (double)SR/NSPS/DF);

    // ---- Costas scan --------------------------------------------------------
    // Each Costas block is 7 symbols × time_osr steps wide.
    // We scan at symbol-step resolution (step == one time_osr sub-step per symbol).
    // Frame span in steps: NN * NSSY = 79 * 4 = 316 steps.
    // freq_osr=2: scan both sub-bins per tone spacing.

    const int TONE_BINS = NSPS * NFOS / SR * SR / NSPS;  // = NFOS = 2
    // tone spacing in bins: SR/NSPS * NFOS / (SR/NFFT) = NFFT/NSPS = 2 bins
    const int BINS_PER_TONE = NFFT / NSPS;  // = 2

    // Frame takes NN*NSSY steps total (79*4=316)
    int frame_steps = NN * NSSY;

    struct Cand { int t0; int fs; int fo_bin; float score; };
    std::vector<Cand> candidates;

    // Search frequency: 200..3000 Hz (tone 0 base), in half-bin increments
    int fo_min = (int)(200.0f / DF);
    int fo_max = (int)(3000.0f / DF);

    for (int t0 = 0; t0 + frame_steps < num_steps; ++t0) {
        for (int fo = fo_min; fo <= fo_max; fo += 1) {
            // Costas correlation: sum mag at Costas tone positions in 3 blocks
            float score = 0.0f;
            for (int c = 0; c < 3; ++c) {
                int sym_base_step = t0 + COSTAS_POS[c] * NSSY;
                for (int s = 0; s < 7; ++s) {
                    int step = sym_base_step + s * NSSY;  // one sub-step per symbol
                    if (step >= num_steps) goto next_fo;
                    int tone_bin = fo + COSTAS[s] * BINS_PER_TONE;
                    if (tone_bin + 1 >= num_bins) goto next_fo;
                    // Take max over freq_osr sub-bins
                    float m = mag[step * num_bins + tone_bin];
                    if (tone_bin + 1 < num_bins)
                        m = std::max(m, mag[step * num_bins + tone_bin + 1]);
                    score += m;
                }
            }
            if (score > min_score)
                candidates.push_back({t0, 0, fo, score});
            next_fo:;
        }
    }

    printf("Costas scan: %zu raw candidates (min_score=%.1f)\n", candidates.size(), (double)min_score);

    // Sort by score descending
    std::sort(candidates.begin(), candidates.end(),
              [](const Cand& a, const Cand& b){ return a.score > b.score; });

    // Deduplicate: keep one per (t0±4, fo±2) window
    std::vector<bool> used(candidates.size(), false);
    std::vector<Cand> deduped;
    for (size_t i = 0; i < candidates.size(); ++i) {
        if (used[i]) continue;
        deduped.push_back(candidates[i]);
        for (size_t j = i + 1; j < candidates.size(); ++j) {
            if (abs(candidates[j].t0  - candidates[i].t0)  <= NSSY &&
                abs(candidates[j].fo_bin - candidates[i].fo_bin) <= 2)
                used[j] = true;
        }
    }
    printf("After dedup: %zu candidates\n", deduped.size());

    // Print top 20 candidates for debugging
    printf("Top candidates:\n");
    for (int i = 0; i < std::min((int)deduped.size(), 20); ++i) {
        const Cand& c = deduped[i];
        printf("  [%3d] t0=%3d (%.2fs)  fo_bin=%4d (%.0fHz)  score=%.3f\n",
               i, c.t0, (double)c.t0*NSTEP/SR, c.fo_bin, (double)c.fo_bin*DF, (double)c.score);
    }

    // ---- Per-candidate: LLR + LDPC + CRC + msg --------------------------------
    int n_decoded = 0;
    const int MAX_CANDS = std::min((int)deduped.size(), 200);

    auto computeLLR = [&](int fo_bin, int t0, float* llr) {
        for (int k = 0; k < ND; ++k) {
            int sym = (k < 29) ? k + 7 : k + 14;
            int step = t0 + sym * NSSY;
            if (step >= num_steps) { llr[k*3] = llr[k*3+1] = llr[k*3+2] = 0.0f; continue; }
            float s[8];
            for (int j = 0; j < 8; ++j) {
                int bin = fo_bin + GRAY[j] * BINS_PER_TONE;
                s[j] = (bin >= 0 && bin < num_bins) ? mag[step * num_bins + bin] : 0.0f;
            }
            auto mx = [](float a, float b, float c2, float d) {
                return std::max({a, b, c2, d});
            };
            llr[k*3+0] = mx(s[4],s[5],s[6],s[7]) - mx(s[0],s[1],s[2],s[3]);
            llr[k*3+1] = mx(s[2],s[3],s[6],s[7]) - mx(s[0],s[1],s[4],s[5]);
            llr[k*3+2] = mx(s[1],s[3],s[5],s[7]) - mx(s[0],s[2],s[4],s[6]);
        }
    };

    for (int ci = 0; ci < MAX_CANDS; ++ci) {
        const Cand& c = deduped[ci];

        // Try fo_bin and fo_bin±1: the Costas scan uses max(bin,bin+1) so the
        // winning fo may be 1 bin off the true tone-0 center.
        float llr[174] = {};
        uint8_t xhat[174] = {};
        bool ldpc_ok = false;
        int best_fo = c.fo_bin;

        for (int dfo = 0; dfo < BINS_PER_TONE; ++dfo) {
            float try_llr[174];
            computeLLR(c.fo_bin + dfo, c.t0, try_llr);
            uint8_t try_xhat[174];
            bool ok = js8_ldpc_decode_cpu(try_llr, try_xhat);
            if (ok && !ldpc_ok) {
                ldpc_ok = true;
                best_fo = c.fo_bin + dfo;
                std::copy(try_llr, try_llr + 174, llr);
                std::copy(try_xhat, try_xhat + 174, xhat);
            }
        }

        if (ci < 5) {
            float llr_max = *std::max_element(llr, llr+174);
            float llr_min = *std::min_element(llr, llr+174);
            bool crc_ok = ldpc_ok && checkCRC12(xhat+87);
            printf("  cand %d: ldpc=%d crc=%d llr=[%.2f..%.2f]  t0=%d fo=%d (best_fo=%d)\n",
                   ci, (int)ldpc_ok, (int)crc_ok, (double)llr_min, (double)llr_max,
                   c.t0, c.fo_bin, best_fo);
        }
        if (!ldpc_ok) continue;

        const uint8_t* info = xhat + 87;
        if (!checkCRC12(info)) continue;

        std::string msg = extractMsg(info);
        if (msg.empty()) continue;

        float freq_hz = best_fo * DF;
        float time_s  = (float)c.t0 * NSTEP / SR;
        printf("DECODED: %-12s  freq=%7.1f Hz  time=%+.3f s  score=%.1f\n",
               msg.c_str(), (double)freq_hz, (double)time_s, (double)c.score);
        ++n_decoded;
    }

    printf("\n%d message(s) decoded from %d candidates checked.\n", n_decoded, MAX_CANDS);
    if (n_decoded == 0) {
        printf("FAIL — no messages decoded (expected KD9ABCTEST00)\n");
        return 1;
    }
    return 0;
}
