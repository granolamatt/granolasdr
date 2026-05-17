
#include <unistd.h>
#include <iostream>
#include <string.h>
#include <complex>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <numeric>
#include <zmq.hpp>

#include "gm/hf/ft8.h"
#include "gm/hf/ft8_capture.h"
#include "gm/hf/hf_bands.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/cuda/FT8Cuda.h"

#include "ft8_lib/ft8/decode.h"
#include "ft8_lib/ft8/encode.h"
#include "ft8_lib/ft8/message.h"
#include "ft8_lib/common/common.h"
#include "ft8_lib/common/monitor.h"

#define LOG_LEVEL LOG_INFO
#include "ft8_lib/ft8/debug.h"


// ---- Composite-to-RF frequency conversion --------------------------------- //
// The HFChannelizer packs HF bands into a 32768-bin composite IFFT, then
// outputs 16384 complex samples at 4.375 MS/s. The FT8 FFT (698880 points) has
// bin_hz = 4,375,000 / 698,880 ≈ 6.25 Hz. freq_offset is a bin in that composite
// FFT, not an RF frequency. Convert it back using the HFChannelizer bin table.
//
// Mapping: composite_ifft_bin = round(freq_offset * IFFT_SIZE / FT8_FFT_SIZE)
//          Then look up composite_ifft_bin in kBandMap to get the wideband bin,
//          then rf_hz = wb_bin * 140e6 / 1048576.
//
// kBandMap is built at startup from kHFBands (gm/hf/hf_bands.h) — single source of truth.
static const int kBandMapSize = kNumHFBands;
static struct { int ifft_start; int ifft_end; int wb_start; } kBandMap[kNumHFBands];

static void init_band_map() {
    int offset = 0;
    for (int i = 0; i < kNumHFBands; ++i) {
        kBandMap[i].ifft_start = offset;
        kBandMap[i].ifft_end   = offset + (int)kHFBands[i].bw;
        kBandMap[i].wb_start   = (int)kHFBands[i].wb_start;
        offset += (int)kHFBands[i].bw;
    }
}

static const int kIfftSize        = 32768;
static const int kFt8FftSize      = 698880;
static const int kWidebandFftSize = 1048576; // 2 * NLARGE
static const float kWbSampleRate  = 140000000.0f;

// Returns actual RF frequency in Hz, or the raw bin number if no band matches.
static float composite_bin_to_rf_hz(int freq_offset) {
    int ifft_bin = (int)roundf((float)freq_offset * kIfftSize / kFt8FftSize);
    // Wrap negative-frequency half (bins >= kIfftSize/2 represent negative freqs)
    if (ifft_bin < 0) ifft_bin += kIfftSize;
    for (int i = 0; i < kBandMapSize; ++i) {
        if (ifft_bin >= kBandMap[i].ifft_start && ifft_bin < kBandMap[i].ifft_end) {
            int wb_bin = kBandMap[i].wb_start + (ifft_bin - kBandMap[i].ifft_start);
            return (float)wb_bin * kWbSampleRate / kWidebandFftSize;
        }
    }
    return (float)freq_offset; // fallback
}

const int kMin_score = 5; // Minimum sync score threshold for candidates
const int kMax_candidates = 2000;
const int kLDPC_iterations = 25;

const int kMax_decoded_messages = 200;

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

void decode(const monitor_t* mon, double tm_slot_start, gm::hf::FT8* publisher,
            const gm::cuda::GpuScanResult* gpu = nullptr)
{
    const ftx_waterfall_t* wf = &mon->wf;

    auto t_find_start = std::chrono::steady_clock::now();

    std::vector<ftx_candidate_t> cand_storage;
    ftx_candidate_t* candidate_list = nullptr;
    int num_candidates = 0;
    bool used_gpu = false;

    if (gpu && gpu->count > 0) {
        // GPU scan already found candidates — sort by score and use directly.
        used_gpu = true;
        uint32_t ngpu = gpu->count;
        std::vector<uint32_t> sidx(ngpu);
        std::iota(sidx.begin(), sidx.end(), 0u);
        std::sort(sidx.begin(), sidx.end(), [&](uint32_t a, uint32_t b){
            return gpu->score[a] > gpu->score[b];
        });
        cand_storage.resize(ngpu);
        for (uint32_t i = 0; i < ngpu; ++i) {
            uint32_t gi = sidx[i];
            cand_storage[i].freq_offset = gpu->fo[gi];
            cand_storage[i].time_offset = gpu->to[gi];
            cand_storage[i].time_sub    = gpu->ts[gi];
            cand_storage[i].freq_sub    = gpu->fs[gi];
            cand_storage[i].score       = gpu->score[gi];
        }
        candidate_list = cand_storage.data();
        num_candidates = (int)ngpu;
    } else {
        // CPU fallback: parallel scan split across threads.
        int nthreads = (int)std::thread::hardware_concurrency();
        if (nthreads < 1) nthreads = 1;
        int bins_per_thread = (wf->num_bins + nthreads - 1) / nthreads;
        std::vector<std::vector<ftx_candidate_t>> per_thread_heaps(nthreads);
        std::vector<int> per_thread_sizes(nthreads, 0);
        {
            std::vector<std::thread> search_workers;
            search_workers.reserve(nthreads);
            for (int t = 0; t < nthreads; ++t) {
                per_thread_heaps[t].resize(kMax_candidates);
                search_workers.emplace_back([&, t]() {
                    int freq_start = t * bins_per_thread;
                    int freq_end   = std::min(freq_start + bins_per_thread, wf->num_bins);
                    per_thread_sizes[t] = ftx_find_candidates_range(
                        wf, kMax_candidates, per_thread_heaps[t].data(), kMin_score,
                        freq_start, freq_end);
                });
            }
            for (auto& w : search_workers) w.join();
        }
        cand_storage.reserve(nthreads * kMax_candidates);
        for (int t = 0; t < nthreads; ++t)
            for (int i = 0; i < per_thread_sizes[t]; ++i)
                cand_storage.push_back(per_thread_heaps[t][i]);
        std::sort(cand_storage.begin(), cand_storage.end(),
                  [](const ftx_candidate_t& a, const ftx_candidate_t& b) {
                      return a.score > b.score;
                  });
        if ((int)cand_storage.size() > kMax_candidates)
            cand_storage.resize(kMax_candidates);
        candidate_list = cand_storage.data();
        num_candidates = (int)cand_storage.size();
    }

    auto t_find_end = std::chrono::steady_clock::now();

    auto t_ldpc_start = std::chrono::steady_clock::now();
    // Parallel LDPC decode — each candidate is independent (read-only waterfall).
    struct CandResult {
        ftx_message_t message;
        ftx_decode_status_t status;
        bool ok;
    };
    std::vector<CandResult> results(num_candidates);

    int num_threads = std::min(num_candidates,
                               (int)std::thread::hardware_concurrency());
    if (num_threads < 1) num_threads = 1;

    std::vector<std::thread> workers;
    workers.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            // Round-robin: thread t handles candidates t, t+num_threads, ...
            for (int idx = t; idx < num_candidates; idx += num_threads) {
                results[idx].ok = ftx_decode_candidate(
                    wf, &candidate_list[idx], kLDPC_iterations,
                    &results[idx].message, &results[idx].status);
            }
        });
    }
    for (auto& w : workers) w.join();
    auto t_ldpc_end = std::chrono::steady_clock::now();

    // Sequential: dedup and print (callsign hashtable is not thread-safe).
    int num_decoded = 0;
    int band_counts[kNumHFBands] = {};
    ftx_message_t decoded[kMax_decoded_messages];
    ftx_message_t* decoded_hashtable[kMax_decoded_messages];
    for (int i = 0; i < kMax_decoded_messages; ++i)
        decoded_hashtable[i] = NULL;

    for (int idx = 0; idx < num_candidates; ++idx)
    {
        if (!results[idx].ok) continue;

        const ftx_candidate_t* cand = &candidate_list[idx];
        const ftx_message_t& message = results[idx].message;
        float freq_hz = composite_bin_to_rf_hz(mon->min_bin + cand->freq_offset);
        float time_sec = (cand->time_offset + (float)cand->time_sub / wf->time_osr) * mon->symbol_period;

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
                idx_hash = (idx_hash + 1) % kMax_decoded_messages;
            }
        } while (!found_empty_slot && !found_duplicate);

        if (found_empty_slot)
        {
            memcpy(&decoded[idx_hash], &message, sizeof(message));
            decoded_hashtable[idx_hash] = &decoded[idx_hash];
            ++num_decoded;
            int ifft_bin = (int)roundf((float)(mon->min_bin + cand->freq_offset) * kIfftSize / kFt8FftSize);
            if (ifft_bin < 0) ifft_bin += kIfftSize;
            for (int bi = 0; bi < kBandMapSize; ++bi)
                if (ifft_bin >= kBandMap[bi].ifft_start && ifft_bin < kBandMap[bi].ifft_end)
                    { band_counts[bi]++; break; }

            char text[FTX_MAX_MESSAGE_LENGTH];
            ftx_message_rc_t unpack_status = ftx_message_decode(&message, &hash_if, text);
            if (unpack_status != FTX_MESSAGE_RC_OK)
            {
                snprintf(text, sizeof(text), "Error [%d] while unpacking!", (int)unpack_status);
            }

            // Each uint8_t waterfall byte encodes 20*log10(amplitude)+offset, so
            // byte differences are already power-dB. Subtract 10*log10(2500/6.25)=26 dB
            // to normalise to the standard 2500 Hz reference bandwidth.
            float snr = (float)cand->score - 26.0f;
            printf("%+05.1f %+05.1f %+4.2f %4.0f ~  %s\n",
                snr, tm_slot_start, time_sec, freq_hz, text);
            printf("DECODED: %s time_offset=%.3fs freq=%.1fHz snr=%.1f unix=%.0f\n", text, time_sec, freq_hz, snr, tm_slot_start);
            if (unpack_status == FTX_MESSAGE_RC_OK && publisher)
                publisher->publishDecoded(text, freq_hz, snr, tm_slot_start, time_sec);
        }
    }
    char band_summary[256] = "";
    int bspos = 0;
    for (int bi = 0; bi < kNumHFBands; ++bi)
        if (band_counts[bi] > 0)
            bspos += snprintf(band_summary + bspos, (int)sizeof(band_summary) - bspos,
                              " %s:%d", kHFBands[bi].name, band_counts[bi]);
    printf("EPOCH: %d decoded / %d candidates (%s) |%s | find=%.1fms ldpc=%.1fms\n",
           num_decoded, num_candidates,
           used_gpu ? "gpu" : "cpu",
           band_summary[0] ? band_summary : " (none)",
           std::chrono::duration<double, std::milli>(t_find_end - t_find_start).count(),
           std::chrono::duration<double, std::milli>(t_ldpc_end - t_ldpc_start).count());

    fflush(stdout);
    LOG(LOG_INFO, "Decoded %d messages, callsign hashtable size %d\n", num_decoded, callsign_hashtable_size);
   
    int numfound = 0;
    for(int cc=0; cc < callsign_hashtable_size; cc++) {
	    if (callsign_hashtable[cc].callsign[0]) {
		    numfound = 1;
		    break;
	    }
    }
    if (numfound) {
      printf("Callsign list");
      for(int cc=0; cc < callsign_hashtable_size; cc++) {
        if (callsign_hashtable[cc].callsign[0])
            printf(" %s",callsign_hashtable[cc].callsign);
      }
      printf("\n");
      fflush(stdout);
    }

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
    if (!me->mag) {
        fprintf(stderr, "waterfall_init: malloc failed for mag (%zu bytes) — out of memory\n", mag_size);
        exit(1);
    }
    printf("Waterfall size %zu\n", mag_size);
    LOG(LOG_DEBUG, "Waterfall size = %zu\n", mag_size);
}

void load_monitor(monitor_t* me)
{
    float symbol_period = FT8_SYMBOL_PERIOD;
    // Compute DSP parameters that depend on the sample rate
    me->block_size = 698880; // samples corresponding to one FSK symbol
    me->subblock_size = 698880;
    me->nfft = 698880;
    me->fft_norm = 698880;

    LOG(LOG_INFO, "Block size = %d\n", me->block_size);
    LOG(LOG_INFO, "Subblock size = %d\n", me->subblock_size);

    // Allocate enough blocks to fit the capture window (FT8_CAPTURE_BLOCKS in ft8_capture.h)
    const int max_blocks = FT8_CAPTURE_BLOCKS;
    // Keep only FFT bins in the specified frequency range (f_min/f_max)
    me->min_bin = 0;
    me->max_bin = 698880;
    const int num_bins = me->max_bin - me->min_bin;

    waterfall_init(&me->wf, max_blocks, num_bins, FT8_TIME_OSR, FT8_FREQ_OSR);
    me->wf.protocol = FTX_PROTOCOL_FT8;
    size_t block_bytes = (size_t)FT8_TIME_OSR * FT8_FREQ_OSR * num_bins;
    printf("FT8: time_osr=%d freq_osr=%d block_bytes=%zu ring=%.1fGB\n",
           FT8_TIME_OSR, FT8_FREQ_OSR, block_bytes,
           (double)(200 * block_bytes) / 1e9);

    me->symbol_period = symbol_period;

    me->max_mag = -120.0f;
}

namespace gm {
namespace hf {

    FT8::FT8(gm::buffer::BufferPosition<uint8_t>* inP,
             gm::cuda::FT8Cuda* ft8cuda_in,
             int zmq_port) :
      inPos(inP),
      ft8cuda(ft8cuda_in),
      zmq_ctx(1),
      zmq_pub(zmq_ctx, ZMQ_PUB) {
        init_band_map();
        hashtable_init();
        load_monitor(&mon);
        demodFT8 = inPos->getBuffer();

        std::string endpoint = "tcp://*:" + std::to_string(zmq_port);
        zmq_pub.bind(endpoint);
        printf("FT8 ZMQ publisher bound to %s\n", endpoint.c_str());
    }

    FT8::~FT8() {
        // waterfall destroy
    }

    void FT8::publishDecoded(const char* callsign, float freq_hz, float snr,
                              double unix_time, float time_offset)
    {
        char buf[256];
        int len = snprintf(buf, sizeof(buf),
            "{\"call\":\"%s\",\"freq\":%.0f,\"snr\":%.1f,\"unix\":%.0f,\"offset\":%.3f}",
            callsign, (double)freq_hz, (double)snr, unix_time, (double)time_offset);
        zmq::message_t msg(buf, len);
        auto result = zmq_pub.send(msg, zmq::send_flags::dontwait);
        if (!result)
            fprintf(stderr, "ZMQ send: queue full, decode dropped for %s\n", callsign);
    }

    void FT8::run() {
        uint64_t now = inPos->getNow(1);

        while(isRunning()) {
            uint64_t next = inPos->getPosition(now+1, 1);
            while(now < next) {
                uint64_t length = next - now;
                if (length > 4) {
                    std::cout << "Error Falling Behind in FT8, Dropping Data" << std::endl;
                    now = next;
                    break;
                }
                int offset = mon.wf.num_blocks * mon.wf.block_stride;
                int buff = now % inPos->getShape()[0];
                uint8_t* src = demodFT8 + inPos->getShape()[1] * buff;
                memcpy(mon.wf.mag + offset, src, inPos->getShape()[1]);
                mon.wf.num_blocks += FT8_CAPTURE_BLOCKS;
                auto nowsec = std::chrono::system_clock::now();
                auto duration = nowsec.time_since_epoch();
                double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
                printf("Processing %f\n", seconds);
                const gm::cuda::GpuScanResult* gpu_res =
                    ft8cuda ? &ft8cuda->getGpuScanResult(buff) : nullptr;
                decode(&mon, seconds, this, gpu_res);
                monitor_reset(&mon);
                now += 1;
            }
        }
    }
}

}

