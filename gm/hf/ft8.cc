
#include <string.h>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <zmq.hpp>

#include "gm/hf/ft8.h"
#include "gm/hf/ft8_capture.h"
#include "gm/hf/band_map.h"
#include "gm/cuda/FT8Cuda.h"
#include "gm/cuda/FT8Osd.h"
#include "wsdict.h"
#include "third_party/nlohmann_json.hpp"

#include "ft8_lib/ft8/decode.h"
#include "ft8_lib/ft8/constants.h"
#include "ft8_lib/ft8/message.h"
#include "ft8_lib/common/common.h"

#define LOG_LEVEL LOG_INFO
#include "ft8_lib/ft8/debug.h"

const int kLDPC_iterations = 25;

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
    while (true) {
        if (callsign_hashtable[idx_hash].callsign[0] == '\0')
            break;
        if (((callsign_hashtable[idx_hash].hash & 0x3FFFFFu) == hash) && (0 == strcmp(callsign_hashtable[idx_hash].callsign, callsign)))
        {
            callsign_hashtable[idx_hash].hash &= 0x3FFFFFu;
            return;
        }
        idx_hash = (idx_hash + 1) % CALLSIGN_HASHTABLE_SIZE;
    }
    if (callsign_hashtable[idx_hash].callsign[0] != '\0') return;
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
    while (callsign_hashtable[idx_hash].callsign[0] != '\0') {
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
        uint32_t n = std::min(*r.count, gm::cuda::CONT_CAND_MAX);
        if (n == 0) return;

        // Initialise window start on first call.
        if (window_start_ == 0.0)
            window_start_ = r.timestamp;

        // Flush when a 15-second window expires.
        if (r.timestamp - window_start_ >= 15.0)
            flushWindow(r.timestamp);

        // Keep best SNR per callsign within the current 15s window.
        auto publish_spot = [&](uint32_t i, const char* text) {
            float freq_hz  = composite_bin_to_rf_hz(r.fo[i]);
            float time_sec = (r.to[i] + (float)r.ts[i] / FT8_TIME_OSR) * FT8_SYMBOL_PERIOD;
            float snr      = (float)r.score[i] - 26.0f;
            std::lock_guard<std::mutex> lk(window_mu_);
            auto it = window_buf_.find(text);
            if (it == window_buf_.end() || snr > it->second.snr)
                window_buf_[text] = {snr, freq_hz, r.timestamp, time_sec};
        };

        // Labeled LLR training capture (one line per frequency bin per scan).
        auto capture = [&](uint32_t i, const char* status,
                           const char* text, const float* dist) {
            if (!llr_capture_.enabled()) return;
            const float* llr = r.log174 + (size_t)i * FTX_LDPC_N;
            float freq_hz = composite_bin_to_rf_hz(r.fo[i]);
            float snr     = (float)r.score[i] - 26.0f;
            llr_capture_.write("FT8", status, llr, FTX_LDPC_N,
                               r.fo[i], (int)r.to[i], (int)r.ts[i], (int)r.score[i],
                               (double)freq_hz, (double)snr, r.timestamp, text, dist);
        };

        // Frequency bins that produced a valid decode this scan (pass or osd);
        // kept out of the "fail" set and used to dedup captures by bin.
        std::unordered_set<int32_t> decoded_bins;

        // Pass 1: BP-decode every candidate; collect BP failures for OSD.
        std::vector<uint32_t> osd_fail;
        for (uint32_t i = 0; i < n; ++i) {
            const float* llr = r.log174 + (size_t)i * FTX_LDPC_N;
            ftx_message_t msg;
            ftx_decode_status_t st;
            if (ftx_decode_from_llr(llr, kLDPC_iterations, &msg, &st)) {
                char text[FTX_MAX_MESSAGE_LENGTH];
                if (ftx_message_decode(&msg, &hash_if, text) == FTX_MESSAGE_RC_OK) {
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

        uint8_t plain174[FTX_LDPC_N];
        for (uint32_t i : cands) {
            const float* llr = r.log174 + (size_t)i * FTX_LDPC_N;
            float dist = 0.0f;
            bool decoded = false;
            if (ft8_osd_decode(llr, osd_order_, plain174, &dist) &&
                (osd_soft_thresh_ <= 0.0f || dist <= osd_soft_thresh_)) {
                ftx_message_t msg;
                ftx_decode_status_t st;
                if (ftx_decode_from_bits(plain174, &msg, &st)) {        // CRC-14 gate
                    char text[FTX_MAX_MESSAGE_LENGTH];
                    if (ftx_message_decode(&msg, &hash_if, text) == FTX_MESSAGE_RC_OK) {
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
