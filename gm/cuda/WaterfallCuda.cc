#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <thread>
#include "gm/cuda/WaterfallCuda.h"
#include "gm/cuda/WaterfallKernel.h"
#include "gm/cuda/HostCuda.h"
#include "wsdict.h"

namespace gm {
namespace cuda {

template<int N>
WaterfallCuda<N>::WaterfallCuda(
    const gm::buffer::DeviceRingBuffer<uint8_t, N>& ring,
    int bin_start, int bin_end, int out_bins, int ws_port,
    const char* key_prefix, float full_rate_hz, int min_publish_ms)
    : ring_(ring)
    , bin_start_(bin_start)
    , bin_end_(bin_end)
    , out_bins_(out_bins)
    , ws_port_(ws_port)
    , key_prefix_(key_prefix)
    , full_rate_hz_(full_rate_hz)
    , min_publish_ms_(min_publish_ms)
{
    cuda_check_error(cudaStreamCreate(&stream_));
    cuda_check_error(cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming));

    // Shared render controls (all waterfalls): contrast, noise-floor lift, and the
    // percentile used to auto-track the noise floor.
    if (const char* g = std::getenv("WF_GAIN")) {
        wf_gain_ = (float)std::atof(g);
        if (wf_gain_ <= 0.0f) wf_gain_ = 1.0f;
    }
    if (const char* o = std::getenv("WF_OFFSET")) wf_offset_ = std::atoi(o);
    if (const char* p = std::getenv("WF_NOISE_PCT")) {
        noise_pct_ = std::atoi(p);
        if (noise_pct_ < 1)  noise_pct_ = 1;
        if (noise_pct_ > 99) noise_pct_ = 99;
    }

    size_t rgba_bytes = (size_t)ROWS_PER_SLOT * out_bins_ * 4;
    cuda_check_error(cudaMalloc((void**)&rgba_d_, rgba_bytes));
    cuda_check_error(cudaHostAlloc((void**)&rgba_h_, rgba_bytes, cudaHostAllocDefault));
    cuda_check_error(cudaMalloc((void**)&hist_d_, 256 * sizeof(unsigned int)));

    float hz_per_bin = full_rate_hz_ / (float)ring_.num_bins;
    printf("Waterfall[%s]: bins [%d, %d) = %.0f–%.0f Hz, %d pixels × %d rows (%.1f Hz/px)\n",
           key_prefix_.c_str(), bin_start_, bin_end_,
           bin_start_ * hz_per_bin, bin_end_ * hz_per_bin,
           out_bins_, ROWS_PER_SLOT,
           (float)(bin_end_ - bin_start_) * hz_per_bin / out_bins_);
}

template<int N>
WaterfallCuda<N>::~WaterfallCuda()
{
    if (rgba_d_) cudaFree(rgba_d_);
    if (rgba_h_) cudaFreeHost(rgba_h_);
    if (hist_d_) cudaFree(hist_d_);
    cudaEventDestroy(ready_);
    cudaStreamDestroy(stream_);
}

template<int N>
void WaterfallCuda<N>::run()
{
    std::unique_ptr<WsDictClient> ws;
    if (ws_port_ > 0) {
        for (int attempt = 0; attempt < 20 && !ws; ++attempt) {
            try {
                ws = std::make_unique<WsDictClient>("127.0.0.1", ws_port_);
            } catch (const std::exception& e) {
                fprintf(stderr, "[Waterfall] wsdict connect: %s — retrying\n", e.what());
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
        }
        if (!ws)
            fprintf(stderr, "[Waterfall] wsdict unavailable — rows will not be published\n");
    }

    const size_t rgba_bytes = (size_t)ROWS_PER_SLOT * out_bins_ * 4;
    int key_idx   = 0;
    uint64_t last_wi = 0;
    bool pending  = false;
    uint64_t row_count = 0;
    auto last_pub = std::chrono::steady_clock::now();

    while (isRunning()) {
        // Deliver completed frame if one is in flight.
        if (pending && cudaEventQuery(ready_) == cudaSuccess) {
            if (ws) {
                std::string key = key_prefix_ + std::to_string(key_idx);
                std::string b64 = wsdict_base64_encode(rgba_h_, rgba_bytes);
                try { ws->set(key, nlohmann::json(b64)); }
                catch (const std::exception& e) {
                    fprintf(stderr, "[Waterfall] wsdict set: %s\n", e.what());
                    ws.reset();
                }
                key_idx = (key_idx + 1) % RING_KEYS;
            }

            // Auto noise-floor: the noise_pct-th percentile of this frame's magnitude
            // histogram, EMA-smoothed.  Each waterfall tracks its own floor, so the
            // shared WF_GAIN/WF_OFFSET behave identically on CW and FT8.
            uint64_t total = 0;
            for (int i = 0; i < 256; ++i) total += hist_h_[i];
            if (total > 0) {
                uint64_t target = total * (uint64_t)noise_pct_ / 100;
                uint64_t cum = 0; int pctl = 0;
                for (int i = 0; i < 256; ++i) { cum += hist_h_[i]; if (cum >= target) { pctl = i; break; } }
                float est = (float)pctl;
                if (!noise_init_) { noise_ref_ = est; noise_init_ = true; }
                else              { noise_ref_ = 0.9f * noise_ref_ + 0.1f * est; }
            }

            row_count += ROWS_PER_SLOT;
            if (row_count % 100 < ROWS_PER_SLOT) {
                printf("[Waterfall %s] noise_ref=%.1f gain=%.1f offset=%d\n",
                       key_prefix_.c_str(), (double)noise_ref_, (double)wf_gain_, wf_offset_);
                fflush(stdout);
            }
            pending = false;
        }

        uint64_t wi = ring_.write_idx.load(std::memory_order_acquire);
        if (wi <= last_wi || pending) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // Publish-rate throttle: the CW ring writes far faster than the FT8 ring,
        // so publishing every slot floods the wsdict WebSocket (periodic reconnects
        // = blank rows).  Cap the rate; when throttled, hold last_wi so the next
        // publish still renders the latest slot (decimation, not backlog).
        if (min_publish_ms_ > 0) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_pub).count()
                    < min_publish_ms_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }
            last_pub = now;
        }
        last_wi = wi;

        uint64_t slot_idx = (wi - 1) % N;
        const uint8_t* slot_base = ring_.base_d + slot_idx * ring_.slot_bytes;

        cudaStreamWaitEvent(stream_, ring_.ready, 0);
        // Row stride = elements per time sub-array = freq_osr * num_bins, taken
        // from the ring's real geometry (slot_bytes / ROWS_PER_SLOT).  This is 4x
        // smaller for the CW ring (freq_osr=1) than the FT8 ring (freq_osr=4);
        // the kernel previously hardcoded the FT8 value and over-strode CW rows.
        const int row_stride = (int)(ring_.slot_bytes / ROWS_PER_SLOT);

        // Histogram this slot for the noise-floor estimate (consumed at next publish,
        // so noise_ref lags one frame — fine for a slow-adapting floor).
        cudaMemsetAsync(hist_d_, 0, 256 * sizeof(unsigned int), stream_);
        waterfall_histogram(slot_base, bin_start_, bin_end_, row_stride, hist_d_, stream_);
        cudaMemcpyAsync(hist_h_, hist_d_, 256 * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost, stream_);

        waterfall_rgba(slot_base, bin_start_, bin_end_,
                       rgba_d_, out_bins_, (int)ring_.num_bins, row_stride,
                       noise_ref_, wf_gain_, wf_offset_, stream_);

        cudaError_t kerr = cudaGetLastError();
        if (kerr != cudaSuccess)
            fprintf(stderr, "[WF] kernel error: %s\n", cudaGetErrorString(kerr));

        cudaMemcpyAsync(rgba_h_, rgba_d_, rgba_bytes,
                        cudaMemcpyDeviceToHost, stream_);
        cudaEventRecord(ready_, stream_);
        pending = true;
    }
}

// Explicit instantiations: FT8/JS8 composite ring (200) and CW composite ring (128).
template class WaterfallCuda<200>;
template class WaterfallCuda<128>;

} // namespace cuda
} // namespace gm
