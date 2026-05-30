#include <chrono>
#include <cstdio>
#include <algorithm>
#include "gm/cuda/JS8Cuda.h"
#include "gm/cuda/JS8ScanCuda.h"
#include "gm/cuda/JS8SoftCuda.h"
#include "gm/hf/ft8_capture.h"

namespace gm {
namespace cuda {

JS8Cuda::JS8Cuda(FT8Cuda* ft8, float min_score)
    : ft8_(ft8), min_score_(min_score)
{
    cudaStreamCreate(&js8_scan_stream_);

    const size_t log174_bytes = (size_t)CAND_MAX * kFtxLdpcN * sizeof(float);
    cudaMalloc((void**)&cand_count_d_, sizeof(uint32_t));
    cudaMalloc((void**)&cand_fo_d_,    CAND_MAX * sizeof(int32_t));
    cudaMalloc((void**)&cand_to_d_,    CAND_MAX * sizeof(uint8_t));
    cudaMalloc((void**)&cand_ts_d_,    CAND_MAX * sizeof(uint8_t));
    cudaMalloc((void**)&cand_fs_d_,    CAND_MAX * sizeof(uint8_t));
    cudaMalloc((void**)&cand_score_d_, CAND_MAX * sizeof(int16_t));
    cudaMalloc((void**)&log174_d_,     log174_bytes);

    allocSlots();
}

JS8Cuda::~JS8Cuda()
{
    stop();

    if (cand_count_d_) cudaFree(cand_count_d_);
    if (cand_fo_d_)    cudaFree(cand_fo_d_);
    if (cand_to_d_)    cudaFree(cand_to_d_);
    if (cand_ts_d_)    cudaFree(cand_ts_d_);
    if (cand_fs_d_)    cudaFree(cand_fs_d_);
    if (cand_score_d_) cudaFree(cand_score_d_);
    if (log174_d_)     cudaFree(log174_d_);
    freeSlots();

    cudaStreamDestroy(js8_scan_stream_);
}

void JS8Cuda::allocSlots()
{
    const size_t log174_bytes = (size_t)CAND_MAX * kFtxLdpcN * sizeof(float);
    for (int i = 0; i < CONTINUOUS_SLOTS; ++i) {
        ContScanResult& s = cont_slots_[i];
        cudaMalloc((void**)&s.count_d,  sizeof(uint32_t));
        cudaMalloc((void**)&s.fo_d,     CAND_MAX * sizeof(int32_t));
        cudaMalloc((void**)&s.to_d,     CAND_MAX * sizeof(uint8_t));
        cudaMalloc((void**)&s.ts_d,     CAND_MAX * sizeof(uint8_t));
        cudaMalloc((void**)&s.fs_d,     CAND_MAX * sizeof(uint8_t));
        cudaMalloc((void**)&s.score_d,  CAND_MAX * sizeof(int16_t));
        cudaMalloc((void**)&s.log174_d, log174_bytes);
        cudaHostAlloc((void**)&s.count,  sizeof(uint32_t),           cudaHostAllocDefault);
        cudaHostAlloc((void**)&s.fo,     CAND_MAX * sizeof(int32_t), cudaHostAllocDefault);
        cudaHostAlloc((void**)&s.to,     CAND_MAX * sizeof(uint8_t), cudaHostAllocDefault);
        cudaHostAlloc((void**)&s.ts,     CAND_MAX * sizeof(uint8_t), cudaHostAllocDefault);
        cudaHostAlloc((void**)&s.fs,     CAND_MAX * sizeof(uint8_t), cudaHostAllocDefault);
        cudaHostAlloc((void**)&s.score,  CAND_MAX * sizeof(int16_t), cudaHostAllocDefault);
        cudaHostAlloc((void**)&s.log174, log174_bytes,               cudaHostAllocDefault);
        cudaEventCreateWithFlags(&s.event, cudaEventDisableTiming);
    }
}

void JS8Cuda::freeSlots()
{
    for (int i = 0; i < CONTINUOUS_SLOTS; ++i) {
        ContScanResult& s = cont_slots_[i];
        if (s.count_d)  cudaFree(s.count_d);
        if (s.fo_d)     cudaFree(s.fo_d);
        if (s.to_d)     cudaFree(s.to_d);
        if (s.ts_d)     cudaFree(s.ts_d);
        if (s.fs_d)     cudaFree(s.fs_d);
        if (s.score_d)  cudaFree(s.score_d);
        if (s.log174_d) cudaFree(s.log174_d);
        if (s.count)    cudaFreeHost(s.count);
        if (s.fo)       cudaFreeHost(s.fo);
        if (s.to)       cudaFreeHost(s.to);
        if (s.ts)       cudaFreeHost(s.ts);
        if (s.fs)       cudaFreeHost(s.fs);
        if (s.score)    cudaFreeHost(s.score);
        if (s.log174)   cudaFreeHost(s.log174);
        if (s.event)    cudaEventDestroy(s.event);
    }
}

void JS8Cuda::start()
{
    running_.store(true, std::memory_order_release);
    scan_thread_   = std::thread([this]() { scanLoop(); });
    worker_thread_ = std::thread([this]() { workerLoop(); });
}

void JS8Cuda::stop()
{
    if (!running_.load(std::memory_order_acquire)) return;
    running_.store(false, std::memory_order_release);
    if (scan_thread_.joinable())   scan_thread_.join();
    if (worker_thread_.joinable()) worker_thread_.join();
}

void JS8Cuda::scanLoop()
{
    const int RING_BLOCKS    = ft8_->getRingBlocks();
    const int num_bins       = ft8_->getRingNumBins();
    constexpr int STRIDE     = 6;
    constexpr int CAP_BLOCKS = FT8_CAPTURE_BLOCKS;

    uint64_t last_triggered = 0;

    while (running_.load(std::memory_order_acquire)) {
        uint64_t wi = ft8_->getRingWriteIdx();

        if (wi < (uint64_t)CAP_BLOCKS || wi <= last_triggered ||
            wi % STRIDE != 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        uint64_t ri = cont_read_idx_.load(std::memory_order_acquire);
        uint64_t cw = cont_write_idx_.load(std::memory_order_relaxed);
        if (cw - ri >= (uint64_t)CONTINUOUS_SLOTS) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        last_triggered = wi;

        ContScanResult& slot  = cont_slots_[cw % CONTINUOUS_SLOTS];
        int snap_start        = (int)((wi - CAP_BLOCKS) % RING_BLOCKS);
        const uint8_t* ring   = ft8_->getRingPtr();

        cudaStreamWaitEvent(js8_scan_stream_, ft8_->getRingReadyEvent(), 0);

        js8_gpu_scan(
            ring, snap_start, RING_BLOCKS,
            slot.fo_d, slot.to_d, slot.ts_d, slot.fs_d,
            slot.score_d, slot.count_d, CAND_MAX,
            num_bins, CAP_BLOCKS,
            FT8_TIME_OSR, FT8_FREQ_OSR, min_score_,
            js8_scan_stream_);

        js8_soft_symbols(
            ring, snap_start, RING_BLOCKS,
            slot.fo_d, slot.to_d, slot.ts_d, slot.fs_d,
            slot.count_d, slot.log174_d,
            num_bins, CAP_BLOCKS,
            FT8_TIME_OSR, FT8_FREQ_OSR, (int)CAND_MAX,
            js8_scan_stream_);

        cudaMemcpyAsync(slot.count,  slot.count_d,  sizeof(uint32_t),           cudaMemcpyDeviceToHost, js8_scan_stream_);
        cudaMemcpyAsync(slot.fo,     slot.fo_d,     CAND_MAX * sizeof(int32_t), cudaMemcpyDeviceToHost, js8_scan_stream_);
        cudaMemcpyAsync(slot.to,     slot.to_d,     CAND_MAX * sizeof(uint8_t), cudaMemcpyDeviceToHost, js8_scan_stream_);
        cudaMemcpyAsync(slot.ts,     slot.ts_d,     CAND_MAX * sizeof(uint8_t), cudaMemcpyDeviceToHost, js8_scan_stream_);
        cudaMemcpyAsync(slot.fs,     slot.fs_d,     CAND_MAX * sizeof(uint8_t), cudaMemcpyDeviceToHost, js8_scan_stream_);
        cudaMemcpyAsync(slot.score,  slot.score_d,  CAND_MAX * sizeof(int16_t), cudaMemcpyDeviceToHost, js8_scan_stream_);
        cudaMemcpyAsync(slot.log174, slot.log174_d,
                        (size_t)CAND_MAX * kFtxLdpcN * sizeof(float),
                        cudaMemcpyDeviceToHost, js8_scan_stream_);

        cudaEventRecord(slot.event, js8_scan_stream_);
        slot_dispatch_ns_[cw % CONTINUOUS_SLOTS] =
            std::chrono::steady_clock::now().time_since_epoch().count();
        slot.dispatched.store(true, std::memory_order_release);
        cont_write_idx_.fetch_add(1, std::memory_order_relaxed);
    }
}

void JS8Cuda::workerLoop()
{
    uint64_t ri = 0;
    while (running_.load(std::memory_order_acquire)) {
        uint64_t wi = cont_write_idx_.load(std::memory_order_acquire);
        if (ri == wi) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        ContScanResult& slot = cont_slots_[ri % CONTINUOUS_SLOTS];
        while (!slot.dispatched.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        cudaError_t ev;
        while ((ev = cudaEventQuery(slot.event)) == cudaErrorNotReady) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        if (ev == cudaSuccess) {
            int64_t now_ns = std::chrono::steady_clock::now().time_since_epoch().count();
            int64_t dispatch_ns = slot_dispatch_ns_[ri % CONTINUOUS_SLOTS];
            float scan_ms = dispatch_ns ? (now_ns - dispatch_ns) * 1e-6f : 0.0f;

            uint32_t n = std::min(*slot.count, CAND_MAX);
            if (n > 0) {
                fprintf(stderr, "[JS8] scan: %u candidates  %.1f ms\n", n, (double)scan_ms);
                if (decode_callback_) {
                    *slot.count = n;
                    try {
                        decode_callback_(slot);
                    } catch (...) {
                        fprintf(stderr, "[JS8] decode_callback threw\n");
                    }
                }
                if (timing_callback_) timing_callback_(scan_ms, 0.0f, n);
            } else if (ri % 60 == 0) {
                fprintf(stderr, "[JS8] alive: %lu scans, 0 candidates\n", (unsigned long)ri);
            }
        } else {
            fprintf(stderr, "[JS8] CUDA event error at slot %lu: %s\n",
                    (unsigned long)(ri % CONTINUOUS_SLOTS), cudaGetErrorString(ev));
        }

        slot.dispatched.store(false, std::memory_order_release);
        cont_read_idx_.store(++ri, std::memory_order_release);
    }
}

} // namespace cuda
} // namespace gm
