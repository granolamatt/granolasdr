// decode_corpus — CPU FT8 decoder for corpus WAV files saved by hf_rx --jtdx
//
// Usage: ./decode_corpus ft8_corpus/*.wav
//
// Prints one JSON line per decoded message:
//   {"msg":"CQ K1ABC FN42","snr":-8,"freq":892,"time":0.45,"file":"...wav"}

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ft8_lib/common/wave.h"
#include "ft8_lib/ft8/constants.h"
#include "ft8_lib/ft8/decode.h"
#include "ft8_lib/ft8/message.h"
#include "ft8_lib/fft/kiss_fft.h"

#define TIME_OSR    2
#define FREQ_OSR    2
#define MAX_CANDS   300
#define MIN_SCORE   10
#define MAX_ITER    20

// Maximum expected samples: 17s at 12100 Hz
#define MAX_SAMPLES 210000

static float g_samples[MAX_SAMPLES];

static void build_waterfall(const float* samples, int num_samples, int sample_rate,
                             ftx_waterfall_t* wf, uint8_t* mag_buf, int max_blocks)
{
    const int block_size    = (int)roundf(sample_rate * FT8_SYMBOL_PERIOD);
    const int subblock_size = block_size / TIME_OSR;
    const int nfft          = block_size;
    const int num_bins      = nfft / 2;

    wf->max_blocks   = max_blocks;
    wf->num_blocks   = 0;
    wf->num_bins     = num_bins;
    wf->time_osr     = TIME_OSR;
    wf->freq_osr     = FREQ_OSR;
    wf->protocol     = FTX_PROTOCOL_FT8;
    wf->mag          = mag_buf;
    wf->block_stride = TIME_OSR * FREQ_OSR * num_bins;

    float* window = (float*)malloc(nfft * sizeof(float));
    for (int i = 0; i < nfft; i++)
        window[i] = 0.5f - 0.5f * cosf(2.0f * (float)M_PI * i / nfft);

    kiss_fft_cfg cfg = kiss_fft_alloc(nfft, 0, NULL, NULL);
    kiss_fft_cpx* fin  = (kiss_fft_cpx*)malloc(nfft * sizeof(kiss_fft_cpx));
    kiss_fft_cpx* fout = (kiss_fft_cpx*)malloc(nfft * sizeof(kiss_fft_cpx));

    int num_subblocks = (num_samples - nfft) / subblock_size;
    int num_blocks    = num_subblocks / TIME_OSR;
    if (num_blocks > max_blocks) num_blocks = max_blocks;

    for (int block = 0; block < num_blocks; block++) {
        for (int ts = 0; ts < TIME_OSR; ts++) {
            int offset = (block * TIME_OSR + ts) * subblock_size;
            for (int fs = 0; fs < FREQ_OSR; fs++) {
                for (int n = 0; n < nfft; n++) {
                    float s = (offset + n < num_samples) ? samples[offset + n] : 0.0f;
                    float w = window[n];
                    // Frequency shift by fs/FREQ_OSR bins for sub-bin resolution
                    float phase = -2.0f * (float)M_PI * fs * n / ((float)FREQ_OSR * nfft);
                    fin[n].r = s * w * cosf(phase);
                    fin[n].i = s * w * sinf(phase);
                }
                kiss_fft(cfg, fin, fout);

                uint8_t* row = &mag_buf[block * wf->block_stride + (ts * FREQ_OSR + fs) * num_bins];
                for (int bin = 0; bin < num_bins; bin++) {
                    float re = fout[bin].r, im = fout[bin].i;
                    float db = 10.0f * log10f(re*re + im*im + 1e-30f);
                    int v = (int)(2.0f * (db + 120.0f));
                    row[bin] = (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v);
                }
            }
        }
        wf->num_blocks++;
    }

    kiss_fft_free(cfg);
    free(fin);
    free(fout);
    free(window);
}

static void decode_file(const char* path)
{
    int num_samples = MAX_SAMPLES;
    int sample_rate = 0;
    if (load_wav(g_samples, &num_samples, &sample_rate, path) != 0) {
        fprintf(stderr, "decode_corpus: cannot load %s\n", path);
        return;
    }

    const int block_size = (int)roundf(sample_rate * FT8_SYMBOL_PERIOD);
    const int num_bins   = block_size / 2;
    const int max_blocks = FT8_NN + 30;  // 79 symbols + search window

    size_t mag_size = (size_t)max_blocks * TIME_OSR * FREQ_OSR * num_bins;
    uint8_t* mag_buf = (uint8_t*)malloc(mag_size);
    if (!mag_buf) { fprintf(stderr, "decode_corpus: OOM\n"); return; }

    ftx_waterfall_t wf = {};
    build_waterfall(g_samples, num_samples, sample_rate, &wf, mag_buf, max_blocks);

    ftx_candidate_t* cands = (ftx_candidate_t*)malloc(MAX_CANDS * sizeof(ftx_candidate_t));
    int num_cands = ftx_find_candidates(&wf, MAX_CANDS, cands, MIN_SCORE);

    for (int i = 0; i < num_cands; i++) {
        ftx_message_t msg = {};
        ftx_decode_status_t status = {};
        if (!ftx_decode_candidate(&wf, &cands[i], MAX_ITER, &msg, &status))
            continue;

        char text[FTX_MAX_MESSAGE_LENGTH + 1] = {};
        ftx_message_decode(&msg, NULL, text);

        float freq_hz  = (cands[i].freq_offset + (float)cands[i].freq_sub / FREQ_OSR)
                         * (float)sample_rate / block_size;
        float time_sec = (cands[i].time_offset + (float)cands[i].time_sub / TIME_OSR)
                         * FT8_SYMBOL_PERIOD;
        // Approximate SNR from candidate score (rough)
        int snr = (int)(cands[i].score / 4) - 25;

        printf("{\"msg\":\"%s\",\"snr\":%d,\"freq\":%.0f,\"time\":%.2f,\"file\":\"%s\"}\n",
               text, snr, freq_hz, time_sec, path);
    }

    free(cands);
    free(mag_buf);
}

int main(int argc, char* argv[])
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <wav_file> [wav_file ...]\n", argv[0]);
        return 1;
    }
    for (int i = 1; i < argc; i++)
        decode_file(argv[i]);
    return 0;
}
