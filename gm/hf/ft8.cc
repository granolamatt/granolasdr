
#include <string.h>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <atomic>
#include <cstdlib>
#include <vector>
#include <zmq.hpp>

#include "gm/hf/ft8.h"
#include "gm/hf/ft8_capture.h"
#include "gm/hf/band_map.h"
#include "gm/cuda/FT8Cuda.h"
#include "gm/cuda/FT8Osd.h"
#include "wsdict.h"
#include "gm/hf/decode_stats.h"
#include "third_party/nlohmann_json.hpp"

#include "ft8_lib/ft8/decode.h"
#include "ft8_lib/ft8/constants.h"
#include "ft8_lib/ft8/message.h"
#include "ft8_lib/common/common.h"

#define LOG_LEVEL LOG_INFO
#include "ft8_lib/ft8/debug.h"

const int kLDPC_iterations = 25;

// Run body(i) for i in [0,count) across up to `nthreads` workers with dynamic
// (atomic-counter) scheduling, so uneven per-item costs (OSD) self-balance. body
// must be reentrant and write only per-index state; the caller serializes the
// shared publish/dedup work afterward, in order, so output stays deterministic.
namespace {
template <typename Fn>
void parallel_for(size_t count, int nthreads, Fn&& body) {
    if (count == 0) return;
    nthreads = std::max(1, std::min(nthreads, (int)count));
    if (nthreads == 1) { for (size_t i = 0; i < count; ++i) body(i); return; }
    std::atomic<size_t> next{0};
    std::vector<std::thread> pool;
    pool.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t)
        pool.emplace_back([&] {
            for (size_t j; (j = next.fetch_add(1, std::memory_order_relaxed)) < count; )
                body(j);
        });
    for (auto& th : pool) th.join();
}
}  // namespace

// Continuous-scan host-decode cap.  The GPU Costas scan can emit tens of
// thousands of candidates under heavy QRN / crowded low bands, most of them
// noise.  BP-decoding all of them serially in the scan callback made the
// consumer fall behind, filling the slot ring and dropping blocks ("[CONT]
// slots full").  We host-decode only the top-K candidates by sync score — real
// signals score highest, so this keeps essentially every true decode while
// bounding per-slot work.  Inert whenever the candidate count is below the cap.
const uint32_t kContTopKCandidates = 4096;

// Refine is ~50 ms/candidate (coordinate-descent FFT search + D2H); the scan
// runs every ~1 s on top of BP+OSD, so cap refines per scan to keep the decode
// callback well under budget.  A transmission reappears across ~12 scans, so the
// strong failures still get refined over time — the cap just spreads the work.
const int kRefineMaxPerCycle = 2;

#define CALLSIGN_HASHTABLE_SIZE 2048

static thread_local struct
{
    char callsign[12];
    uint32_t hash;
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
            if (age >= max_age)
            {
                callsign_hashtable[idx_hash].callsign[0] = '\0';
                callsign_hashtable[idx_hash].hash = 0;
                callsign_hashtable_size--;
            }
            else
            {
                callsign_hashtable[idx_hash].hash = (((uint32_t)age + 1u) << 24) | (callsign_hashtable[idx_hash].hash & 0x3FFFFFu);
            }
        }
    }
}

void hashtable_add(const char* callsign, uint32_t hash)
{
    int idx_hash = (int)(hash % CALLSIGN_HASHTABLE_SIZE);
    // Bounded probe (at most SIZE steps). The original `while(true)` never exited
    // on a FULL table (no empty slot, no match) -> infinite loop while holding
    // decode_mu_, which wedged the continuous scan after ~24 h of accumulated
    // callsigns (the table is thread_local and the refine worker's copy is never
    // cleaned). On a full table we fall through and the guard below skips the add.
    for (int probe = 0; probe < CALLSIGN_HASHTABLE_SIZE; ++probe) {
        if (callsign_hashtable[idx_hash].callsign[0] == '\0')
            break;
        if (((callsign_hashtable[idx_hash].hash & 0x3FFFFFu) == hash) && (0 == strcmp(callsign_hashtable[idx_hash].callsign, callsign)))
        {
            callsign_hashtable[idx_hash].hash &= 0x3FFFFFu;
            return;
        }
        idx_hash = (idx_hash + 1) % CALLSIGN_HASHTABLE_SIZE;
    }
    if (callsign_hashtable[idx_hash].callsign[0] != '\0') return;   // table full: skip
    callsign_hashtable_size++;
    strncpy(callsign_hashtable[idx_hash].callsign, callsign, 11);
    callsign_hashtable[idx_hash].callsign[11] = '\0';
    callsign_hashtable[idx_hash].hash = hash;
}

bool hashtable_lookup(ftx_callsign_hash_type_t hash_type, uint32_t hash, char* callsign)
{
    int hash_shift = (hash_type == FTX_CALLSIGN_HASH_22_BITS) ? 0 :
                     (hash_type == FTX_CALLSIGN_HASH_12_BITS) ? 10 : 12;
    int idx_hash = (int)((hash << hash_shift) % CALLSIGN_HASHTABLE_SIZE);
    // Bounded probe: a full table would otherwise loop forever here (same hang).
    for (int probe = 0; probe < CALLSIGN_HASHTABLE_SIZE &&
                        callsign_hashtable[idx_hash].callsign[0] != '\0'; ++probe) {
        if (((callsign_hashtable[idx_hash].hash & 0x3FFFFFu) >> hash_shift) == hash)
        {
            strcpy(callsign, callsign_hashtable[idx_hash].callsign);
            return true;
        }
        idx_hash = (idx_hash + 1) % CALLSIGN_HASHTABLE_SIZE;
    }
    return false;
}

static ftx_callsign_hash_interface_t hash_if = {
    .lookup_hash = hashtable_lookup,
    .save_hash   = hashtable_add
};

namespace gm {
namespace hf {

    FT8::FT8(gm::cuda::FT8Cuda* ft8cuda_in, int zmq_port, int wsdict_port)
      : ft8cuda_(ft8cuda_in)
      , zmq_port_(zmq_port)
      , zmq_ctx_(1)
      , zmq_pub_(zmq_ctx_, ZMQ_PUB)
      , timing_log_(nullptr) {

        hashtable_init();

        // Parallel decode workers for the continuous-scan Pass-1 BP / Pass-2 OSD
        // (the per-slot CPU bottleneck a strong antenna exposes). Default leaves
        // headroom for the GPU/CW/ZMQ threads; FT8_DECODE_THREADS overrides.
        {
            unsigned hw = std::thread::hardware_concurrency();
            int def = hw > 4 ? std::min(16, (int)hw - 2) : 1;
            const char* e = std::getenv("FT8_DECODE_THREADS");
            decode_threads_ = e ? std::max(1, atoi(e)) : std::max(1, def);
            printf("FT8 continuous-scan decode threads: %d\n", decode_threads_);
        }

        ft8cuda_->setDecodeCallback([this](gm::cuda::ContScanResult& r) {
            decodeAndPublishContinuous(r);
        });
        ft8cuda_->startContinuousScan();

        if (zmq_port > 0) {
            std::string endpoint = "tcp://localhost:" + std::to_string(zmq_port);
            zmq_pub_.connect(endpoint);
            printf("FT8 ZMQ publisher → %s  topic=ft8/decode\n", endpoint.c_str());
        } else {
            printf("FT8 ZMQ disabled (stdout only)\n");
        }

        if (wsdict_port > 0) {
            try {
                ws_client_ = std::make_unique<WsDictClient>("127.0.0.1", wsdict_port);
                printf("FT8 wsdict → port=%d  key=granolasdr:ft8:decode\n", wsdict_port);
            } catch (const std::exception& e) {
                fprintf(stderr, "[FT8] wsdict connect failed: %s\n", e.what());
            }
        }

        timing_log_ = fopen("ft8_timing.csv", "a");
        if (timing_log_) {
            fseek(timing_log_, 0, SEEK_END);
            if (ftell(timing_log_) == 0)
                fprintf(timing_log_, "wall_clock,epoch_time,epoch_mod15,dt_sec,freq_hz,snr,callsign\n");
            printf("FT8 timing log: ft8_timing.csv\n");
        }

        llr_capture_.openIfEnabled("FT8_CAPTURE_LLR", "ft8", /*mode_tag=*/nullptr);
    }

    FT8::~FT8() {
        refine_run_.store(false);
        refine_cv_.notify_all();
        if (refine_thread_.joinable()) refine_thread_.join();
        if (timing_log_) fclose(timing_log_);
    }

    void FT8::run() {
        while (isRunning())
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    void FT8::publishDecoded(const char* callsign, float freq_hz, float snr,
                              double unix_time, float time_offset)
    {
        printf("DECODED: %s time_offset=%.3fs freq=%.0fHz snr=%.1f unix=%.0f\n",
               callsign, (double)time_offset, (double)freq_hz, (double)snr, unix_time);
        gm::hf::decode_stats_count("FT8");

        nlohmann::json v;
        v["call"]   = callsign;
        v["freq"]   = (double)freq_hz;
        v["snr"]    = (double)snr;
        v["unix"]   = unix_time;
        v["offset"] = (double)time_offset;
        // dump() with error_handler_t::replace ensures valid UTF-8 even if the
        // decoder emits a false-positive with non-ASCII bytes in the callsign.
        std::string json_str = v.dump(-1, ' ', false,
                                      nlohmann::json::error_handler_t::replace);
        {
            std::lock_guard<std::mutex> lk(zmq_mutex_);
            if (zmq_port_ > 0) {
                zmq::message_t topic("ft8/decode", 10);
                zmq::message_t payload(json_str.data(), json_str.size());
                zmq_pub_.send(topic, zmq::send_flags::sndmore);
                auto result = zmq_pub_.send(payload, zmq::send_flags::dontwait);
                if (!result)
                    fprintf(stderr, "[FT8] ZMQ send queue full, dropped: %s\n", callsign);
            }
            if (ws_client_) {
                try {
                    v["mode"] = "FT8";
                    // One key per callsign; '/' → '-' so key doesn't contain glob separator.
                    // 900s (15 min) TTL matches the display window and survives page refresh.
                    std::string call_key = callsign;
                    std::replace(call_key.begin(), call_key.end(), '/', '-');
                    ws_client_->set_with_ttl("granolasdr:ft8:heard:" + call_key, v, 900000);
                } catch (const std::exception& e) {
                    fprintf(stderr, "[FT8] wsdict publish: %s\n", e.what());
                    ws_client_.reset();
                }
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

    void FT8::flushWindow(double now)
    {
        std::unordered_map<std::string, WindowSpot> buf;
        {
            std::lock_guard<std::mutex> lk(window_mu_);
            buf.swap(window_buf_);
            window_start_ = now;
        }
        printf("[FT8] 15s window: %zu unique callsigns\n", buf.size());
        for (auto& kv : buf)
            publishDecoded(kv.first.c_str(),
                           kv.second.freq_hz, kv.second.snr,
                           kv.second.unix_time, kv.second.time_offset);
        fflush(stdout);
        hashtable_cleanup(10);
    }

    void FT8::decodeAndPublishContinuous(gm::cuda::ContScanResult& r) {
        const uint32_t raw = *r.count;
        uint32_t n = std::min(raw, gm::cuda::CONT_CAND_MAX);

        // Candidate-load readout: accumulate per-scan Costas-candidate counts and
        // print a summary every ~15 s, so --min-score can be tuned against how hard
        // the scan is actually pushed (avg/peak vs the top-K cap and the 100000 max).
        ++cand_scans_;
        cand_sum_ += raw;
        if (raw > cand_peak_) cand_peak_ = raw;
        if (raw >= gm::cuda::CONT_CAND_MAX) ++cand_sat_;
        if (cand_report_t0_ == 0.0) cand_report_t0_ = r.timestamp;
        if (r.timestamp - cand_report_t0_ >= 15.0) {
            printf("[CONT] FT8 candidates/scan avg=%llu peak=%u  (top-K %u, max %u; "
                   "%d scans/%.0fs, %d hit max)\n",
                   (unsigned long long)(cand_sum_ / (unsigned)std::max(1, cand_scans_)),
                   cand_peak_, kContTopKCandidates, gm::cuda::CONT_CAND_MAX,
                   cand_scans_, r.timestamp - cand_report_t0_, cand_sat_);
            cand_report_t0_ = r.timestamp;
            cand_sum_ = 0; cand_peak_ = 0; cand_scans_ = 0; cand_sat_ = 0;
        }

        if (n == 0) return;

        // Initialise window start on first call.
        if (window_start_ == 0.0)
            window_start_ = r.timestamp;

        // Flush when a 15-second window expires.
        if (r.timestamp - window_start_ >= 15.0)
            flushWindow(r.timestamp);

        // Forwarders to the shared members (also used by the refine worker).
        auto publish_spot = [&](uint32_t i, const char* text) {
            publishSpot(text, r.fo[i], (int)r.to[i], (int)r.ts[i], r.score[i], r.timestamp);
        };
        auto capture = [&](uint32_t i, const char* status, const char* text,
                           const float* dist, const float* llr_override = nullptr) {
            const float* llr = llr_override ? llr_override
                                            : r.log174 + (size_t)i * FTX_LDPC_N;
            captureCand(status, llr, r.fo[i], (int)r.to[i], (int)r.ts[i],
                        r.score[i], r.timestamp, text, dist);
        };

        // ftx_message_decode mutates a shared callsign hashtable — guard it since
        // the refine worker thread also decodes concurrently.
        auto decode_msg = [&](ftx_message_t& msg, char* text) {
            std::lock_guard<std::mutex> lk(decode_mu_);
            return ftx_message_decode(&msg, &hash_if, text) == FTX_MESSAGE_RC_OK;
        };

        // Frequency bins that produced a valid decode this scan (pass or osd);
        // kept out of the "fail" set and used to dedup captures by bin.
        std::unordered_set<int32_t> decoded_bins;

        // Bound per-slot work: host-decode only the top-K candidates by sync
        // score.  Under QRN the GPU can hand us tens of thousands of (mostly
        // noise) candidates; decoding all of them stalls the continuous-scan
        // consumer and drops blocks.  Real signals score highest, so the cap
        // costs no real decodes and only engages when the count is pathological.
        std::vector<uint32_t> order(n);
        for (uint32_t i = 0; i < n; ++i) order[i] = i;
        if (n > kContTopKCandidates) {
            std::nth_element(order.begin(), order.begin() + kContTopKCandidates,
                             order.end(),
                             [&](uint32_t a, uint32_t b) { return r.score[a] > r.score[b]; });
            order.resize(kContTopKCandidates);   // load reported by the periodic readout above
        }

        // Pass 1: BP-decode every selected candidate. The BP is the per-slot CPU
        // bottleneck under a strong antenna and is independent per candidate, so
        // run it across the decode-thread pool; the shared publish/dedup work then
        // runs serially and IN ORDER below (output is identical to the serial
        // version regardless of thread timing). vector<uint8_t>, not vector<bool>,
        // so concurrent writes don't race on a shared bitset.
        std::vector<uint8_t>       bp_ok(order.size(), 0);
        std::vector<ftx_message_t> bp_msg(order.size());
        parallel_for(order.size(), decode_threads_, [&](size_t j) {
            const float* llr = r.log174 + (size_t)order[j] * FTX_LDPC_N;
            ftx_decode_status_t st;
            bp_ok[j] = ftx_decode_from_llr(llr, kLDPC_iterations, &bp_msg[j], &st) ? 1 : 0;
        });

        std::vector<uint32_t> osd_fail;
        for (size_t j = 0; j < order.size(); ++j) {
            const uint32_t i = order[j];
            if (bp_ok[j]) {
                char text[FTX_MAX_MESSAGE_LENGTH];
                if (decode_msg(bp_msg[j], text)) {
                    publish_spot(i, text);
                    if (!decoded_bins.count(r.fo[i])) capture(i, "pass", text, nullptr);
                    decoded_bins.insert(r.fo[i]);
                }
            } else if (osd_enable_ && (float)r.score[i] >= osd_score_floor_) {
                osd_fail.push_back(i);
            }
        }

        if (!osd_enable_ || osd_fail.empty()) return;

        // Pass 2: OSD fallback on the strongest BP failures.  Dedup by frequency
        // bin (keep best sync score), sort by score, cap per cycle — this gating,
        // not CRC-14 alone, holds the OSD-on-noise false-accept rate negligible.
        std::unordered_map<int32_t, uint32_t> best_by_bin;
        for (uint32_t i : osd_fail) {
            auto it = best_by_bin.find(r.fo[i]);
            if (it == best_by_bin.end() || r.score[i] > r.score[it->second])
                best_by_bin[r.fo[i]] = i;
        }
        std::vector<uint32_t> cands;
        cands.reserve(best_by_bin.size());
        for (const auto& kv : best_by_bin) cands.push_back(kv.second);
        std::sort(cands.begin(), cands.end(),
                  [&](uint32_t a, uint32_t b) { return r.score[a] > r.score[b]; });
        if ((int)cands.size() > osd_max_per_cycle_) cands.resize(osd_max_per_cycle_);

        // Parallel OSD compute (order-2 OSD is the other per-slot hot spot), then
        // serial in-order CRC / publish / refine-enqueue below.
        struct OsdOut { uint8_t ok; float dist; uint8_t plain174[FTX_LDPC_N]; };
        std::vector<OsdOut> osd(cands.size());
        parallel_for(cands.size(), decode_threads_, [&](size_t k) {
            const float* llr = r.log174 + (size_t)cands[k] * FTX_LDPC_N;
            osd[k].dist = 0.0f;
            osd[k].ok = (ft8_osd_decode(llr, osd_order_, osd[k].plain174, &osd[k].dist) &&
                         (osd_soft_thresh_ <= 0.0f || osd[k].dist <= osd_soft_thresh_)) ? 1 : 0;
        });

        int refine_used = 0;
        for (size_t k = 0; k < cands.size(); ++k) {
            const uint32_t i = cands[k];
            float dist = osd[k].dist;
            bool decoded = false;
            if (osd[k].ok) {
                ftx_message_t msg;
                ftx_decode_status_t st;
                if (ftx_decode_from_bits(osd[k].plain174, &msg, &st)) {  // CRC-14 gate
                    char text[FTX_MAX_MESSAGE_LENGTH];
                    if (decode_msg(msg, text)) {
                        publish_spot(i, text);
                        if (!decoded_bins.count(r.fo[i])) capture(i, "osd", text, &dist);
                        decoded_bins.insert(r.fo[i]);
                        decoded = true;
                    }
                }
            }

            // Strong-Costas candidate the OSD gate tried but no decoder cracked.
            if (!decoded && !decoded_bins.count(r.fo[i]))
                capture(i, "fail", nullptr, &dist);

            // Refine fallback runs OFF this thread: enqueue the failed strong
            // candidate for the refine worker (bounded, drop-oldest) so the heavy
            // FFT search never blocks the scan callback.
            if (!decoded && refine_enable_ && refine_used < kRefineMaxPerCycle) {
                ++refine_used;
                std::lock_guard<std::mutex> lk(refine_mu_);
                if (refine_q_.size() >= kRefineQueueMax) refine_q_.pop_front();
                refine_q_.push_back({r.fo[i], (int)r.to[i], (int)r.ts[i],
                                     r.snap_start, r.score[i], r.timestamp});
                refine_cv_.notify_one();
            }
        }
    }

    // Shared window/capture helpers (scan callback + refine worker).
    void FT8::publishSpot(const char* text, int32_t fo, int to, int ts,
                          int16_t score, double timestamp) {
        float freq_hz  = composite_bin_to_rf_hz(fo);
        float time_sec = (to + (float)ts / FT8_TIME_OSR) * FT8_SYMBOL_PERIOD;
        float snr      = (float)score - 26.0f;
        std::lock_guard<std::mutex> lk(window_mu_);
        auto it = window_buf_.find(text);
        if (it == window_buf_.end() || snr > it->second.snr)
            window_buf_[text] = {snr, freq_hz, timestamp, time_sec};
    }

    void FT8::captureCand(const char* status, const float* llr, int32_t fo, int to,
                          int ts, int16_t score, double timestamp,
                          const char* text, const float* dist) {
        if (!llr_capture_.enabled()) return;
        float freq_hz = composite_bin_to_rf_hz(fo);
        float snr     = (float)score - 26.0f;
        std::lock_guard<std::mutex> lk(capture_mu_);
        llr_capture_.write("FT8", status, llr, FTX_LDPC_N, fo, to, ts, (int)score,
                           (double)freq_hz, (double)snr, timestamp, text, dist);
    }

    // Refine worker: drains the job queue, re-extracts LLRs from the retained
    // complex frame with continuous freq/time alignment, retries BP then OSD, and
    // publishes/captures any decode the discrete-grid path missed.  Runs on its
    // own thread + CUDA stream, so it never stalls the continuous scan.
    void FT8::refineWorker() {
        uint8_t plain174[FTX_LDPC_N];
        while (refine_run_.load(std::memory_order_acquire)) {
            RefineJob job;
            {
                std::unique_lock<std::mutex> lk(refine_mu_);
                refine_cv_.wait(lk, [&] {
                    return !refine_run_.load(std::memory_order_acquire) || !refine_q_.empty();
                });
                if (!refine_run_.load(std::memory_order_acquire)) break;
                job = refine_q_.front();
                refine_q_.pop_front();
            }

            float rllr[FTX_LDPC_N];
            if (!ft8cuda_->refineCandidate(job.fo, job.to, job.snap_start, rllr))
                continue;

            ftx_message_t msg;
            ftx_decode_status_t st;
            float rdist = 0.0f;
            bool cracked = ftx_decode_from_llr(rllr, kLDPC_iterations, &msg, &st);
            if (!cracked && osd_enable_ &&
                ft8_osd_decode(rllr, osd_order_, plain174, &rdist) &&
                (osd_soft_thresh_ <= 0.0f || rdist <= osd_soft_thresh_))
                cracked = ftx_decode_from_bits(plain174, &msg, &st);   // CRC-14 gate
            if (!cracked) continue;

            char text[FTX_MAX_MESSAGE_LENGTH];
            {
                std::lock_guard<std::mutex> lk(decode_mu_);
                bool ok = (ftx_message_decode(&msg, &hash_if, text) == FTX_MESSAGE_RC_OK);
                // Age THIS worker's own thread_local callsign table periodically.
                // flushWindow's cleanup runs on the contWorker thread and never
                // touches this one, so without this it fills over ~24 h (and, pre-
                // fix, hung). Bounded scan of the table, so it's cheap under the lock.
                if (++refine_decode_cnt_ >= 128) { refine_decode_cnt_ = 0; hashtable_cleanup(10); }
                if (!ok) continue;
            }
            publishSpot(text, job.fo, job.to, job.ts, job.score, job.timestamp);
            captureCand("refine", rllr, job.fo, job.to, job.ts, job.score,
                        job.timestamp, text, &rdist);
        }
    }

    void FT8::setOsdConfig(bool enable, int order, float score_floor,
                           int max_per_cycle, float soft_thresh)
    {
        osd_enable_        = enable;
        osd_order_         = order;
        osd_score_floor_   = score_floor;
        osd_max_per_cycle_ = max_per_cycle;
        osd_soft_thresh_   = soft_thresh;
    }

}
}
