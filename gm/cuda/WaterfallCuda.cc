#include <cstdio>
#include <cstring>
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
    uint8_t wf_floor, uint8_t wf_ceil,
    const char* key_prefix, float full_rate_hz, int min_publish_ms)
    : ring_(ring)
    , bin_start_(bin_start)
    , bin_end_(bin_end)
    , out_bins_(out_bins)
    , ws_port_(ws_port)
    , wf_floor_(wf_floor)
    , wf_ceil_(wf_ceil)
    , key_prefix_(key_prefix)
    , full_rate_hz_(full_rate_hz)
    , min_publish_ms_(min_publish_ms)
{
    cuda_check_error(cudaStreamCreate(&stream_));
    cuda_check_error(cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming));

    size_t rgba_bytes = (size_t)ROWS_PER_SLOT * out_bins_ * 4;
    cuda_check_error(cudaMalloc((void**)&rgba_d_, rgba_bytes));
    cuda_check_error(cudaHostAlloc((void**)&rgba_h_, rgba_bytes, cudaHostAllocDefault));

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

            row_count += ROWS_PER_SLOT;
            if (row_count % 100 < ROWS_PER_SLOT) {
                uint8_t vmin = 255, vmax = 0;
                for (size_t i = 0; i < rgba_bytes; i += 4) {
                    // Recover approximate magnitude from R channel as a sanity check.
                    if (rgba_h_[i] < vmin) vmin = rgba_h_[i];
                    if (rgba_h_[i] > vmax) vmax = rgba_h_[i];
                }
                printf("[Waterfall] row %llu: R min=%u max=%u bin=[%d,%d)\n",
                       (unsigned long long)row_count, vmin, vmax, bin_start_, bin_end_);
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
        waterfall_rgba(slot_base, bin_start_, bin_end_,
                       rgba_d_, out_bins_, (int)ring_.num_bins, row_stride,
                       wf_floor_, wf_ceil_, stream_);

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
