
#include <unistd.h>
#include <fcntl.h>
#include <iostream>
#include <string.h>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <atomic>
#include <mutex>
#include <zmq.hpp>

#include "gm/hf/ft8.h"
#include "gm/hf/ft8_capture.h"
#include "gm/hf/hf_bands.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/cuda/FT8Cuda.h"

#include "ft8_lib/ft8/decode.h"
#include "ft8_lib/ft8/constants.h"
#include "ft8_lib/ft8/message.h"
#include "ft8_lib/common/common.h"
#include "ft8_lib/common/monitor.h"

#define LOG_LEVEL LOG_INFO
#include "ft8_lib/ft8/debug.h"


#include "gm/hf/band_map.h"

const int kMin_score = 5; // Minimum sync score threshold for candidates
const int kMax_candidates = 2000;
const int kLDPC_iterations = 25;

const int kMax_decoded_messages = 200;

#define CALLSIGN_HASHTABLE_SIZE 2048

static thread_local struct
{
    char callsign[12]; ///> Up to 11 symbols of callsign + trailing zeros (always filled)
    uint32_t hash;     ///> 8 MSBs contain the age of callsign; 22 LSBs contain hash value
} callsign_hashtable[CALLSIGN_HASHTABLE_SIZE];

static thread_local int callsign_hashtable_size;

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
    for (int probe = 0; probe < CALLSIGN_HASHTABLE_SIZE; ++probe)
    {
        if (callsign_hashtable[idx_hash].callsign[0] == '\0')
            break;
        if (((callsign_hashtable[idx_hash].hash & 0x3FFFFFu) == hash) && (0 == strcmp(callsign_hashtable[idx_hash].callsign, callsign)))
        {
            callsign_hashtable[idx_hash].hash &= 0x3FFFFFu;
            return;
        }
        idx_hash = (idx_hash + 1) % CALLSIGN_HASHTABLE_SIZE;
    }
    if (callsign_hashtable[idx_hash].callsign[0] != '\0')
        return; // table full, drop silently
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
    for (int probe = 0; probe < CALLSIGN_HASHTABLE_SIZE; ++probe)
    {
        if (callsign_hashtable[idx_hash].callsign[0] == '\0')
            break;
        if (((callsign_hashtable[idx_hash].hash & 0x3FFFFFu) >> hash_shift) == hash)
        {
            strcpy(callsign, callsign_hashtable[idx_hash].callsign);
            return true;
        }
        idx_hash = (idx_hash + 1) % CALLSIGN_HASHTABLE_SIZE;
    }
    callsign[0] = '\0';
    return false;
}

ftx_callsign_hash_interface_t hash_if = {
    .lookup_hash = hashtable_lookup,
    .save_hash = hashtable_add
};

// Continuous path: discards hashtable writes so the epoch path owns the table.
// ftx_message_decode calls save_hash unconditionally (no null check in ft8_lib),
// so nullptr crashes; use a no-op stub instead.
static void noop_save_hash(const char*, uint32_t) {}

ftx_callsign_hash_interface_t cont_hash_if = {
    .lookup_hash = hashtable_lookup,
    .save_hash = noop_save_hash
};

static std::atomic<int> window_decode_count{0};
static time_t           window_start_ts{0};

// LLR capture: set LDPC_CAPTURE=/path/to/file.bin to save raw float32 LLR vectors
// (174 floats each) for every unique decoded candidate. Load in Python with:
//   np.fromfile(path, dtype=np.float32).reshape(-1, 174)
// See tools/analyze_ldpc_captures.py for analysis.
static int    g_ldpc_capture_fd    = -1;
static int    g_ldpc_capture_count = 0;
constexpr int kLdpcCaptureMax      = 500;

// Score log: set SCORE_LOG=/path/to/score_ldpc.csv to write per-candidate
// (epoch_unix, score, parity, crc_ok) rows for offline threshold analysis.
static FILE*            g_score_log      = nullptr;
static std::once_flag   s_score_log_init;

void decode(const monitor_t* mon, double tm_slot_start, gm::hf::FT8* publisher,
            const char* label,
            const gm::cuda::GpuScanResult* gpu = nullptr, bool use_gpu_ldpc = false)
{
    // One-time capture file init (env-var gated, no overhead when unset).
    static std::once_flag s_cap_init;
    std::call_once(s_cap_init, []() {
        const char* path = getenv("LDPC_CAPTURE");
        if (path) {
            g_ldpc_capture_fd = open(path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
            if (g_ldpc_capture_fd >= 0)
                fprintf(stderr, "[LDPC CAPTURE] Writing up to %d LLR vectors to %s\n",
                        kLdpcCaptureMax, path);
        }
    });

    // One-time score log init: SCORE_LOG=/path writes per-candidate
    // (epoch_unix, costas_score, parity_pass, crc_ok) for threshold analysis.
    std::call_once(s_score_log_init, []() {
        const char* path = getenv("SCORE_LOG");
        if (path) {
            g_score_log = fopen(path, "a");
            if (g_score_log) {
                fseek(g_score_log, 0, SEEK_END);
                if (ftell(g_score_log) == 0)
                    fprintf(g_score_log, "epoch_unix,score,parity,crc_ok\n");
                fprintf(stderr, "[SCORE LOG] Writing per-candidate score data to %s\n", path);
            }
        }
    });

    // Per-15min decode count logging.
    time_t now_ts = (time_t)tm_slot_start;
    if (window_start_ts == 0) window_start_ts = now_ts - (now_ts % 900);
    if (now_ts >= window_start_ts + 900) {
        printf("[STATS] 15-min decodes: %d\n", window_decode_count.load());
        window_decode_count.store(0);
        window_start_ts += 900;
    }
    const ftx_waterfall_t* wf = &mon->wf;

    auto t_find_start = std::chrono::steady_clock::now();

    std::vector<ftx_candidate_t> cand_storage;
    ftx_candidate_t* candidate_list = nullptr;
    int num_candidates = 0;

    if (gpu && gpu->count > 0) {
        uint32_t ngpu = gpu->count;
        cand_storage.resize(ngpu);
        for (uint32_t i = 0; i < ngpu; ++i) {
            cand_storage[i].freq_offset = gpu->fo[i];
            cand_storage[i].time_offset = gpu->to[i];
            cand_storage[i].time_sub    = gpu->ts[i];
            cand_storage[i].freq_sub    = gpu->fs[i];
            cand_storage[i].score       = gpu->score[i];
        }
        candidate_list = cand_storage.data();
        num_candidates = (int)ngpu;
    }

    auto t_find_end = std::chrono::steady_clock::now();

#ifdef VALIDATE_SOFT_SYMBOLS
    // Compare GPU LLRs against CPU reference. Requires wf->mag populated (pre-T7 build).
    if (gpu && !gpu->log174.empty() && wf->mag != NULL) {
        float max_delta = 0.0f;
        float cpu_llr[FTX_LDPC_N];
        for (int idx = 0; idx < num_candidates; ++idx) {
            ftx_get_ft8_llr(wf, &candidate_list[idx], cpu_llr);
            const float* gpu_llr = gpu->log174.data() + (size_t)idx * FTX_LDPC_N;
            for (int i = 0; i < FTX_LDPC_N; ++i) {
                float d = fabsf(cpu_llr[i] - gpu_llr[i]);
                if (d > max_delta) max_delta = d;
            }
        }
        printf("VALIDATE_SOFT_SYMBOLS: max_delta=%.6f over %d candidates\n",
               max_delta, num_candidates);
    }
#endif

    auto t_ldpc_start = std::chrono::steady_clock::now();
    struct CandResult {
        ftx_message_t message;
        ftx_decode_status_t status;
        bool ok;
    };
    std::vector<CandResult> results(num_candidates);

    bool using_gpu_ldpc = use_gpu_ldpc
                          && gpu
                          && !gpu->x_hat.empty()
                          && !gpu->parity.empty()
                          && (int)gpu->parity.size() >= num_candidates;

    if (using_gpu_ldpc) {
        // GPU already decoded all LDPC frames. For candidates that passed parity,
        // skip LDPC entirely and run CRC only via ftx_decode_from_bits().
        // Candidates that failed parity get results[idx].ok = false without CRC.
        int num_threads = std::min(num_candidates,
                                   (int)std::thread::hardware_concurrency());
        if (num_threads < 1) num_threads = 1;
        std::vector<std::thread> workers;
        workers.reserve(num_threads);
        std::atomic<int> n_parity_pass{0}, n_allzero{0};
        for (int t = 0; t < num_threads; ++t) {
            workers.emplace_back([&, t]() {
                for (int idx = t; idx < num_candidates; idx += num_threads) {
                    if (!gpu->parity[idx]) {
                        results[idx].ok = false;
                        results[idx].status.ldpc_errors = 1;
                        continue;
                    }
                    n_parity_pass.fetch_add(1, std::memory_order_relaxed);
                    const uint8_t* bits = gpu->x_hat.data() + (size_t)idx * FTX_LDPC_N;
                    bool allzero = true;
                    for (int b = 0; b < FTX_LDPC_N && allzero; ++b)
                        if (bits[b]) allzero = false;
                    if (allzero) n_allzero.fetch_add(1, std::memory_order_relaxed);
                    results[idx].ok = ftx_decode_from_bits(
                        bits, &results[idx].message, &results[idx].status);
                }
            });
        }
        for (auto& w : workers) w.join();
        int pp = n_parity_pass.load(), az = n_allzero.load();

        // Per-epoch score histogram: buckets [5-9],[10-14],[15-19],[20-24],[25+].
        // Shows parity-pass and crc-ok counts per bucket so we can see whether
        // low-score candidates ever decode and whether the threshold is worth lowering.
        constexpr int N_BKT = 5;
        int bkt_total[N_BKT] = {}, bkt_pass[N_BKT] = {}, bkt_crc[N_BKT] = {};
        int n_crc_ok = 0;
        for (int idx = 0; idx < num_candidates; ++idx) {
            int s = (int)gpu->score[idx];
            int b = (s < 10) ? 0 : (s < 15) ? 1 : (s < 20) ? 2 : (s < 25) ? 3 : 4;
            bkt_total[b]++;
            if (gpu->parity[idx]) bkt_pass[b]++;
            if (results[idx].ok)  { bkt_crc[b]++; n_crc_ok++; }
            if (g_score_log)
                fprintf(g_score_log, "%.0f,%d,%d,%d\n",
                        tm_slot_start, s,
                        (int)gpu->parity[idx], (int)results[idx].ok);
        }
        if (g_score_log) fflush(g_score_log);

        const char* bnames[N_BKT] = {"[5-9]","[10-14]","[15-19]","[20-24]","[25+]"};
        fprintf(stderr, "[GPU LDPC] n=%d  parity=%d  crc=%d  zeros=%d  |",
                num_candidates, pp, n_crc_ok, az);
        for (int b = 0; b < N_BKT; ++b)
            if (bkt_total[b] > 0)
                fprintf(stderr, "  %s p=%d/%d crc=%d",
                        bnames[b], bkt_pass[b], bkt_total[b], bkt_crc[b]);
        fprintf(stderr, "\n");
    } else {
        int num_threads = std::min(num_candidates,
                                   (int)std::thread::hardware_concurrency());
        if (num_threads < 1) num_threads = 1;
        std::vector<std::thread> workers;
        workers.reserve(num_threads);
        for (int t = 0; t < num_threads; ++t) {
            workers.emplace_back([&, t, gpu]() {
                for (int idx = t; idx < num_candidates; idx += num_threads) {
                    results[idx].ok = ftx_decode_from_llr(
                        gpu->log174.data() + (size_t)idx * FTX_LDPC_N,
                        kLDPC_iterations,
                        &results[idx].message, &results[idx].status);
                }
            });
        }
        for (auto& w : workers) w.join();
    }
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
            int bi = composite_bin_to_band_idx(mon->min_bin + cand->freq_offset);
            if (bi >= 0) band_counts[bi]++;

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
            // Save raw LLRs for offline ADMM analysis (tools/analyze_ldpc_captures.py).
            if (g_ldpc_capture_fd >= 0 && g_ldpc_capture_count < kLdpcCaptureMax &&
                gpu && (size_t)(idx + 1) * FTX_LDPC_N <= gpu->log174.size()) {
                const float* llr = gpu->log174.data() + (size_t)idx * FTX_LDPC_N;
                write(g_ldpc_capture_fd, llr, FTX_LDPC_N * sizeof(float));
                ++g_ldpc_capture_count;
            }
            if (unpack_status == FTX_MESSAGE_RC_OK && publisher) {
                publisher->publishDecoded(text, freq_hz, snr, tm_slot_start, time_sec);
                window_decode_count.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    char band_summary[256] = "";
    int bspos = 0;
    for (int bi = 0; bi < kNumHFBands; ++bi)
        if (band_counts[bi] > 0)
            bspos += snprintf(band_summary + bspos, (int)sizeof(band_summary) - bspos,
                              " %s:%d", kHFBands[bi].name, band_counts[bi]);
    printf("[%s] EPOCH: %d decoded / %d candidates (gpu) |%s | find=%.1fms %s=%.1fms\n",
           label, num_decoded, num_candidates,
           band_summary[0] ? band_summary : " (none)",
           std::chrono::duration<double, std::milli>(t_find_end - t_find_start).count(),
           using_gpu_ldpc ? "ldpc_gpu" : "ldpc",
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

void load_monitor(monitor_t* me)
{
    float symbol_period = FT8_SYMBOL_PERIOD;
    me->block_size    = 1048576;
    me->subblock_size = 1048576;
    me->nfft          = 1048576;
    me->fft_norm      = 1048576;

    me->min_bin = 0;
    me->max_bin = 1048576;
    const int num_bins = me->max_bin - me->min_bin;

    // Phase 4: skip waterfall mag alloc — GPU computes LLRs from ring buffer.
    // wf.mag = NULL; metadata set directly so time_osr/freq_osr/num_bins are available.
    me->wf.max_blocks   = FT8_CAPTURE_BLOCKS;
    me->wf.num_blocks   = 0;
    me->wf.num_bins     = num_bins;
    me->wf.time_osr     = FT8_TIME_OSR;
    me->wf.freq_osr     = FT8_FREQ_OSR;
    me->wf.block_stride = FT8_TIME_OSR * FT8_FREQ_OSR * num_bins;
    me->wf.mag          = NULL;
    me->wf.protocol     = FTX_PROTOCOL_FT8;

    me->symbol_period = symbol_period;
    me->max_mag       = -120.0f;

    printf("FT8: time_osr=%d freq_osr=%d num_bins=%d (GPU LLR mode)\n",
           FT8_TIME_OSR, FT8_FREQ_OSR, num_bins);
}

namespace gm {
namespace hf {

    FT8::FT8(gm::buffer::BufferPosition<uint8_t>* inP,
             gm::cuda::FT8Cuda* ft8cuda_in,
             int zmq_port,
             bool use_gpu_ldpc_in) :
      inPos(inP),
      ft8cuda(ft8cuda_in),
      use_gpu_ldpc(use_gpu_ldpc_in),
      zmq_port_(zmq_port),
      zmq_ctx(1),
      zmq_pub(zmq_ctx, ZMQ_PUB),
      timing_log_(nullptr) {
        init_band_map();
        hashtable_init();
        load_monitor(&mon);

        if (zmq_port > 0) {
            std::string endpoint = "tcp://localhost:" + std::to_string(zmq_port);
            zmq_pub.connect(endpoint);
            printf("FT8 ZMQ publisher → %s  topic=ft8/decode\n", endpoint.c_str());
        } else {
            printf("FT8 ZMQ disabled (stdout only)\n");
        }

        timing_log_ = fopen("ft8_timing.csv", "a");
        if (timing_log_) {
            fseek(timing_log_, 0, SEEK_END);
            if (ftell(timing_log_) == 0)
                fprintf(timing_log_, "wall_clock,epoch_time,epoch_mod15,dt_sec,freq_hz,snr,callsign\n");
            printf("FT8 timing log: ft8_timing.csv\n");
        } else {
            perror("FT8: cannot open ft8_timing.csv");
        }
    }

    FT8::~FT8() {
        if (timing_log_) fclose(timing_log_);
    }

    void FT8::publishDecoded(const char* callsign, float freq_hz, float snr,
                              double unix_time, float time_offset)
    {
        char buf[256];
        int len = snprintf(buf, sizeof(buf),
            "{\"call\":\"%s\",\"freq\":%.0f,\"snr\":%.1f,\"unix\":%.0f,\"offset\":%.3f}",
            callsign, (double)freq_hz, (double)snr, unix_time, (double)time_offset);
        {
            std::lock_guard<std::mutex> lk(zmq_mutex_);
            if (zmq_port_ > 0) {
                zmq::message_t topic("ft8/decode", 10);
                zmq::message_t payload(buf, len);
                zmq_pub.send(topic, zmq::send_flags::sndmore);
                auto result = zmq_pub.send(payload, zmq::send_flags::dontwait);
                if (!result)
                    fprintf(stderr, "[FT8] ZMQ send queue full, dropped: %s\n", callsign);
            }
        }

        if (timing_log_) {
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            double wall = ts.tv_sec + ts.tv_nsec * 1e-9;
            fprintf(timing_log_, "%.3f,%.3f,%.3f,%.3f,%.0f,%.1f,%s\n",
                    wall, unix_time, fmod(unix_time, 15.0),
                    (double)time_offset, (double)freq_hz, (double)snr, callsign);
            fflush(timing_log_);
        }
    }

    void FT8::decodeAndPublishContinuous(gm::cuda::ContScanResult& r) {
        uint32_t n = std::min(*r.count, gm::cuda::CONT_CAND_MAX);
        if (n == 0) return;

        for (uint32_t i = 0; i < n; ++i) {
            const float* llr = r.log174 + (size_t)i * FTX_LDPC_N;
            ftx_message_t msg;
            ftx_decode_status_t st;
            if (!ftx_decode_from_llr(llr, kLDPC_iterations, &msg, &st)) continue;

            char text[FTX_MAX_MESSAGE_LENGTH];
            ftx_message_rc_t rc = ftx_message_decode(&msg, &cont_hash_if, text);
            if (rc != FTX_MESSAGE_RC_OK) continue;

            // Inform epoch scan of where this real signal is so it can align its triggers.
            if (ft8cuda)
                ft8cuda->reportDecoded(r.snap_start + (uint64_t)r.to[i]);

            float freq_hz  = composite_bin_to_rf_hz(r.fo[i]);
            float time_sec = (r.to[i] + (float)r.ts[i] / FT8_TIME_OSR) * FT8_SYMBOL_PERIOD;
            float snr      = (float)r.score[i] - 26.0f;

            // Dedup: print once per 20 s per (text, freq-100Hz bucket).
            std::string dedup_key = std::string(text) + "|" +
                                    std::to_string((int)(freq_hz / 100));
            double& last = cont_dedup_[dedup_key];
            if (r.timestamp - last >= 20.0) {
                printf("[FT8 CONT] %s  freq=%.0f Hz  snr=%+.1f  offset=%+.3fs\n",
                       text, (double)freq_hz, (double)snr, (double)time_sec);
                last = r.timestamp;
            }
            publishDecoded(text, freq_hz, snr, r.timestamp, time_sec);
            window_decode_count.fetch_add(1, std::memory_order_relaxed);
        }
    }

    void FT8::run() {
        uint64_t now = inPos->getNow(1);

        while(isRunning()) {
            uint64_t next = inPos->getPosition(now+1, 1);
            while(now < next) {
                uint64_t length = next - now;
                if (length > 4) {
                    std::cerr << "Error Falling Behind in FT8, Dropping Data" << std::endl;
                    now = next;
                    break;
                }
                int buff = now % inPos->getShape()[0];
                auto nowsec = std::chrono::system_clock::now();
                auto duration = nowsec.time_since_epoch();
                double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
                printf("Processing %f\n", seconds);
                const gm::cuda::GpuScanResult* gpu_res =
                    ft8cuda ? &ft8cuda->getGpuScanResult(buff) : nullptr;
                char label[16];
                snprintf(label, sizeof(label), ":%d", zmq_port_);
                decode(&mon, seconds, this, label, gpu_res, use_gpu_ldpc);
                monitor_reset(&mon);
                now += 1;
            }
        }
    }
}

}

