#define _POSIX_C_SOURCE 200112L
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#include <ft8/decode.h>
#include <ft8/encode.h>
#include <ft8/message.h>

#include <common/common.h>
#include <common/wave.h>
#include <common/monitor.h>
#include <common/audio.h>

#define LOG_LEVEL LOG_INFO
#include <ft8/debug.h>

const int kMin_score = 10; // Minimum sync score threshold for candidates
const int kMax_candidates = 440;
const int kLDPC_iterations = 25;

const int kMax_decoded_messages = 100;

const int kFreq_osr = 2; // Frequency oversampling rate (bin subdivision)
const int kTime_osr = 2; // Time oversampling rate (symbol subdivision)

void usage(const char* error_msg)
{
    if (error_msg != NULL)
    {
        fprintf(stderr, "ERROR: %s\n", error_msg);
    }
    fprintf(stderr, "Usage: decode_ft8 [-list|([-ft4] [INPUT|-dev DEVICE])]\n\n");
    fprintf(stderr, "Decode a 15-second (or slighly shorter) WAV file.\n");
}

#define CALLSIGN_HASHTABLE_SIZE 256

static struct
{
    char callsign[12]; ///> Up to 11 symbols of callsign + trailing zeros (always filled)
    uint32_t hash;     ///> 8 MSBs contain the age of callsign; 22 LSBs contain hash value
} callsign_hashtable[CALLSIGN_HASHTABLE_SIZE];

static int callsign_hashtable_size;

void hashtable_init(void)
{
    callsign_hashtable_size = 0;
    memset(callsign_hashtable, 0, sizeof(callsign_hashtable));
}

void hashtable_cleanup(uint8_t max_age)
{
    for (int idx_hash = 0; idx_hash < CALLSIGN_HASHTABLE_SIZE; ++idx_hash)
    {
        if (callsign_hashtable[idx_hash].callsign[0] != '\0')
        {
            uint8_t age = (uint8_t)(callsign_hashtable[idx_hash].hash >> 24);
            if (age > max_age)
            {
                LOG(LOG_INFO, "Removing [%s] from hash table, age = %d\n", callsign_hashtable[idx_hash].callsign, age);
                // free the hash entry
                callsign_hashtable[idx_hash].callsign[0] = '\0';
                callsign_hashtable[idx_hash].hash = 0;
                callsign_hashtable_size--;
            }
            else
            {
                // increase callsign age
                callsign_hashtable[idx_hash].hash = (((uint32_t)age + 1u) << 24) | (callsign_hashtable[idx_hash].hash & 0x3FFFFFu);
            }
        }
    }
}

void hashtable_add(const char* callsign, uint32_t hash)
{
    uint16_t hash10 = (hash >> 12) & 0x3FFu;
    int idx_hash = (hash10 * 23) % CALLSIGN_HASHTABLE_SIZE;
    while (callsign_hashtable[idx_hash].callsign[0] != '\0')
    {
        if (((callsign_hashtable[idx_hash].hash & 0x3FFFFFu) == hash) && (0 == strcmp(callsign_hashtable[idx_hash].callsign, callsign)))
        {
            // reset age
            callsign_hashtable[idx_hash].hash &= 0x3FFFFFu;
            LOG(LOG_DEBUG, "Found a duplicate [%s]\n", callsign);
            return;
        }
        else
        {
            LOG(LOG_DEBUG, "Hash table clash!\n");
            // Move on to check the next entry in hash table
            idx_hash = (idx_hash + 1) % CALLSIGN_HASHTABLE_SIZE;
        }
    }
    callsign_hashtable_size++;
    strncpy(callsign_hashtable[idx_hash].callsign, callsign, 11);
    callsign_hashtable[idx_hash].callsign[11] = '\0';
    callsign_hashtable[idx_hash].hash = hash;
}

bool hashtable_lookup(ftx_callsign_hash_type_t hash_type, uint32_t hash, char* callsign)
{
    uint8_t hash_shift = (hash_type == FTX_CALLSIGN_HASH_10_BITS) ? 12 : (hash_type == FTX_CALLSIGN_HASH_12_BITS ? 10 : 0);
    uint16_t hash10 = (hash >> (12 - hash_shift)) & 0x3FFu;
    int idx_hash = (hash10 * 23) % CALLSIGN_HASHTABLE_SIZE;
    while (callsign_hashtable[idx_hash].callsign[0] != '\0')
    {
        if (((callsign_hashtable[idx_hash].hash & 0x3FFFFFu) >> hash_shift) == hash)
        {
            strcpy(callsign, callsign_hashtable[idx_hash].callsign);
            return true;
        }
        // Move on to check the next entry in hash table
        idx_hash = (idx_hash + 1) % CALLSIGN_HASHTABLE_SIZE;
    }
    callsign[0] = '\0';
    return false;
}

ftx_callsign_hash_interface_t hash_if = {
    .lookup_hash = hashtable_lookup,
    .save_hash = hashtable_add
};

void decode(const monitor_t* mon, struct tm* tm_slot_start)
{
    const ftx_waterfall_t* wf = &mon->wf;
    // Find top candidates by Costas sync score and localize them in time and frequency
    ftx_candidate_t candidate_list[kMax_candidates];
    int num_candidates = ftx_find_candidates(wf, kMax_candidates, candidate_list, kMin_score);
    printf("Found %d candidates\n",num_candidates);

    // Hash table for decoded messages (to check for duplicates)
    int num_decoded = 0;
    ftx_message_t decoded[kMax_decoded_messages];
    ftx_message_t* decoded_hashtable[kMax_decoded_messages];

    // Initialize hash table pointers
    for (int i = 0; i < kMax_decoded_messages; ++i)
    {
        decoded_hashtable[i] = NULL;
    }

    // Go over candidates and attempt to decode messages
    for (int idx = 0; idx < num_candidates; ++idx)
    {
        const ftx_candidate_t* cand = &candidate_list[idx];

        float freq_hz = (mon->min_bin + cand->freq_offset + (float)cand->freq_sub / wf->freq_osr) / mon->symbol_period;
        float time_sec = (cand->time_offset + (float)cand->time_sub / wf->time_osr) * mon->symbol_period;

#ifdef WATERFALL_USE_PHASE
        // int resynth_len = 12000 * 16;
        // float resynth_signal[resynth_len];
        // for (int pos = 0; pos < resynth_len; ++pos)
        // {
        //     resynth_signal[pos] = 0;
        // }
        // monitor_resynth(mon, cand, resynth_signal);
        // char resynth_path[80];
        // sprintf(resynth_path, "resynth_%04f_%02.1f.wav", freq_hz, time_sec);
        // save_wav(resynth_signal, resynth_len, 12000, resynth_path);
#endif

        ftx_message_t message;
        ftx_decode_status_t status;
        if (!ftx_decode_candidate(wf, cand, kLDPC_iterations, &message, &status))
        {
            if (status.ldpc_errors > 0)
            {
                LOG(LOG_DEBUG, "LDPC decode: %d errors\n", status.ldpc_errors);
            }
            else if (status.crc_calculated != status.crc_extracted)
            {
                LOG(LOG_DEBUG, "CRC mismatch!\n");
            }
            continue;
        }

        LOG(LOG_DEBUG, "Checking hash table for %4.1fs / %4.1fHz [%d]...\n", time_sec, freq_hz, cand->score);
        int idx_hash = message.hash % kMax_decoded_messages;
        bool found_empty_slot = false;
        bool found_duplicate = false;
        do
        {
            if (decoded_hashtable[idx_hash] == NULL)
            {
                LOG(LOG_DEBUG, "Found an empty slot\n");
                found_empty_slot = true;
            }
            else if ((decoded_hashtable[idx_hash]->hash == message.hash) && (0 == memcmp(decoded_hashtable[idx_hash]->payload, message.payload, sizeof(message.payload))))
            {
                LOG(LOG_DEBUG, "Found a duplicate!\n");
                found_duplicate = true;
            }
            else
            {
                LOG(LOG_DEBUG, "Hash table clash!\n");
                // Move on to check the next entry in hash table
                idx_hash = (idx_hash + 1) % kMax_decoded_messages;
            }
        } while (!found_empty_slot && !found_duplicate);

        if (found_empty_slot)
        {
            // Fill the empty hashtable slot
            memcpy(&decoded[idx_hash], &message, sizeof(message));
            decoded_hashtable[idx_hash] = &decoded[idx_hash];
            ++num_decoded;

            char text[FTX_MAX_MESSAGE_LENGTH];
            ftx_message_rc_t unpack_status = ftx_message_decode(&message, &hash_if, text);
            if (unpack_status != FTX_MESSAGE_RC_OK)
            {
                snprintf(text, sizeof(text), "Error [%d] while unpacking!", (int)unpack_status);
            }

            // Fake WSJT-X-like output for now
            float snr = cand->score * 0.5f; // TODO: compute better approximation of SNR
            printf("%02d%02d%02d %+05.1f %+4.2f %4.0f ~  %s\n",
                tm_slot_start->tm_hour, tm_slot_start->tm_min, tm_slot_start->tm_sec,
                snr, time_sec, freq_hz, text);
        }
    }
    LOG(LOG_INFO, "Decoded %d messages, callsign hashtable size %d\n", num_decoded, callsign_hashtable_size);
    hashtable_cleanup(10);
}

static void waterfall_init(ftx_waterfall_t* me, int max_blocks, int num_bins, int time_osr, int freq_osr)
{
    size_t mag_size = max_blocks * time_osr * freq_osr * num_bins * sizeof(me->mag[0]);
    me->max_blocks = max_blocks;
    me->num_blocks = 0;
    me->num_bins = num_bins;
    me->time_osr = time_osr;
    me->freq_osr = freq_osr;
    me->block_stride = (time_osr * freq_osr * num_bins);
    me->mag = (WF_ELEM_T*)malloc(mag_size);
    LOG(LOG_DEBUG, "Waterfall size = %zu\n", mag_size);
}

void load_monitor(monitor_t* me)
{
    float slot_time = FT8_SLOT_TIME;
    float symbol_period = FT8_SYMBOL_PERIOD;
    // Compute DSP parameters that depend on the sample rate
    me->block_size = 698880; // samples corresponding to one FSK symbol
    me->subblock_size = 698880;
    me->nfft = 698880;
    me->fft_norm = 698880;
    // const int len_window = 1.8f * me->block_size; // hand-picked and optimized

    // me->window = (float*)malloc(me->nfft * sizeof(me->window[0]));
    // for (int i = 0; i < me->nfft; ++i)
    // {
    //     // window[i] = 1;
    //     me->window[i] = me->fft_norm * hann_i(i, me->nfft);
    //     // me->window[i] = blackman_i(i, me->nfft);
    //     // me->window[i] = hamming_i(i, me->nfft);
    //     // me->window[i] = (i < len_window) ? hann_i(i, len_window) : 0;
    // }
    // me->last_frame = (float*)calloc(me->nfft, sizeof(me->last_frame[0]));

    LOG(LOG_INFO, "Block size = %d\n", me->block_size);
    LOG(LOG_INFO, "Subblock size = %d\n", me->subblock_size);

    // size_t fft_work_size = 0;
    // kiss_fftr_alloc(me->nfft, 0, 0, &fft_work_size);
    // me->fft_work = malloc(698880*8);
    // me->fft_cfg = kiss_fftr_alloc(me->nfft, 0, me->fft_work, &fft_work_size);

    // LOG(LOG_INFO, "N_FFT = %d\n", me->nfft);
    // LOG(LOG_DEBUG, "FFT work area = %zu\n", fft_work_size);

    // Allocate enough blocks to fit the entire FT8/FT4 slot in memory
    const int max_blocks = (int)(slot_time / symbol_period);
    // Keep only FFT bins in the specified frequency range (f_min/f_max)
    me->min_bin = 0;
    me->max_bin = 698880;
    const int num_bins = me->max_bin - me->min_bin;

    waterfall_init(&me->wf, max_blocks, num_bins, 1, 1);
    me->wf.protocol = FTX_PROTOCOL_FT8;

    me->symbol_period = symbol_period;

    me->max_mag = -120.0f;
}

int main(int argc, char** argv)
{
    // Accepted arguments
    const char* wav_path = NULL;
    const char* dev_name = NULL;
    ftx_protocol_t protocol = FTX_PROTOCOL_FT8;
    float time_shift = 0.8;

    // Parse arguments one by one
    int arg_idx = 1;
    while (arg_idx < argc)
    {
        // Check if the current argument is an option (-xxx)
        if (argv[arg_idx][0] == '-')
        {
            // Check agaist valid options
            if (0 == strcmp(argv[arg_idx], "-ft4"))
            {
                protocol = FTX_PROTOCOL_FT4;
            }
            else if (0 == strcmp(argv[arg_idx], "-list"))
            {
                audio_init();
                audio_list();
                return 0;
            }
            else if (0 == strcmp(argv[arg_idx], "-dev"))
            {
                if (arg_idx + 1 < argc)
                {
                    ++arg_idx;
                    dev_name = argv[arg_idx];
                }
                else
                {
                    usage("Expected an audio device name after -dev");
                    return -1;
                }
            }
            else
            {
                usage("Unknown command line option");
                return -1;
            }
        }
        else
        {
            if (wav_path == NULL)
            {
                wav_path = argv[arg_idx];
            }
            else
            {
                usage("Multiple positional arguments");
                return -1;
            }
        }
        ++arg_idx;
    }
    // Check if all mandatory arguments have been received
    if (wav_path == NULL && dev_name == NULL)
    {
        usage("Expected either INPUT file path or DEVICE name");
        return -1;
    }

    float slot_period = ((protocol == FTX_PROTOCOL_FT8) ? FT8_SLOT_TIME : FT4_SLOT_TIME);
    int sample_rate = 698880;// / slot_period;
    int num_samples = slot_period * sample_rate;
    float signal[num_samples];
    bool is_live = true;


    // hashtable_init();
    monitor_t mon;
    load_monitor(&mon);
    unsigned char buffer[sample_rate*8];
    float* samps = (float*)buffer;
    FILE *ptr;
    int cnt, cc;

    ptr = fopen("ft8.bin","rb");  // r for read, b for binary
    // fread(buffer,8,4096,ptr); // read 10 bytes to our buffer

    struct tm tm_slot_start = { 0 };

    
    while (is_live) {

        struct timespec spec;
        clock_gettime(CLOCK_REALTIME, &spec);
        double time = (double)spec.tv_sec + (spec.tv_nsec / 1e9);
        double time_within_slot = fmod(time - time_shift, slot_period);
        time_t time_slot_start = (time_t)(time - time_within_slot);
        gmtime_r((const time_t *)&time_slot_start, (struct tm*)&tm_slot_start);
        // LOG(LOG_INFO, "Time within slot %02d%02d%02d: %.3f s\n", tm_slot_start.tm_hour,
            // tm_slot_start.tm_min, tm_slot_start.tm_sec, time_within_slot);

        int num = fread(buffer,8,sample_rate,ptr); // read 10 bytes to our buffer
        is_live = (num > 0);
        int offset = mon.wf.num_blocks * mon.wf.block_stride;
        // printf("Read in %d offset is %d\n", num, offset);


        for (cc=0; cc< 698880; cc++) {
            float real = samps[2*cc] / 100e6;
            float imag = samps[2*cc+1] / 100e6;
            float mag2 = real*real + imag*imag;
            float db = 10.0f * log10f(1E-12f + mag2);
            int scaled = (int)(2 * db + 240);
            mon.wf.mag[offset] = (scaled < 0) ? 0 : ((scaled > 255) ? 255 : scaled);
            // printf("Put in %d db %f\n", mon.wf.mag[cc], db);
            if (db > mon.max_mag)
                mon.max_mag = db;
            offset += 1;
        }
        ++mon.wf.num_blocks;
        if (mon.wf.num_blocks % FT8_NN == 0) {
            // Decode accumulated data (containing slightly less than a full time slot)
            decode(&mon, &tm_slot_start);
            monitor_reset(&mon);
            // next message segment is at 65624832
            // read in 10413312

            for (cc=0; cc<21; cc++)
            fread(buffer,8,495872,ptr);

        }
        // fprintf(stderr, "\n");
        // LOG(LOG_DEBUG, "Waterfall accumulated %d symbols\n", mon.wf.num_blocks);
        // LOG(LOG_INFO, "Max magnitude: %.1f on %d dB\n", mon.max_mag, cnt);


    }

    free(mon.wf.mag);

    return 0;
}
