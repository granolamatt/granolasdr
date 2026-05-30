
#include <string.h>
#include <chrono>
#include <mutex>
#include <thread>
#include <zmq.hpp>

#include "gm/hf/ft8.h"
#include "gm/hf/ft8_capture.h"
#include "gm/hf/band_map.h"
#include "gm/cuda/FT8Cuda.h"

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

    FT8::FT8(gm::cuda::FT8Cuda* ft8cuda_in, int zmq_port)
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

        timing_log_ = fopen("ft8_timing.csv", "a");
        if (timing_log_) {
            fseek(timing_log_, 0, SEEK_END);
            if (ftell(timing_log_) == 0)
                fprintf(timing_log_, "wall_clock,epoch_time,epoch_mod15,dt_sec,freq_hz,snr,callsign\n");
            printf("FT8 timing log: ft8_timing.csv\n");
        }
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
        char buf[256];
        int len = snprintf(buf, sizeof(buf),
            "{\"call\":\"%s\",\"freq\":%.0f,\"snr\":%.1f,\"unix\":%.0f,\"offset\":%.3f}",
            callsign, (double)freq_hz, (double)snr, unix_time, (double)time_offset);
        {
            std::lock_guard<std::mutex> lk(zmq_mutex_);
            if (zmq_port_ > 0) {
                zmq::message_t topic("ft8/decode", 10);
                zmq::message_t payload(buf, len);
                zmq_pub_.send(topic, zmq::send_flags::sndmore);
                auto result = zmq_pub_.send(payload, zmq::send_flags::dontwait);
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

        for (uint32_t i = 0; i < n; ++i) {
            const float* llr = r.log174 + (size_t)i * FTX_LDPC_N;
            ftx_message_t msg;
            ftx_decode_status_t st;
            if (!ftx_decode_from_llr(llr, kLDPC_iterations, &msg, &st)) continue;

            char text[FTX_MAX_MESSAGE_LENGTH];
            ftx_message_rc_t rc = ftx_message_decode(&msg, &hash_if, text);
            if (rc != FTX_MESSAGE_RC_OK) continue;

            float freq_hz  = composite_bin_to_rf_hz(r.fo[i]);
            float time_sec = (r.to[i] + (float)r.ts[i] / FT8_TIME_OSR) * FT8_SYMBOL_PERIOD;
            float snr      = (float)r.score[i] - 26.0f;

            // Keep best SNR per callsign within the current 15s window.
            std::lock_guard<std::mutex> lk(window_mu_);
            auto it = window_buf_.find(text);
            if (it == window_buf_.end() || snr > it->second.snr) {
                window_buf_[text] = {snr, freq_hz, r.timestamp, time_sec};
            }
        }
    }

}
}
