#include <unistd.h>
#include <cstdio>
#include <chrono>
#include <thread>
#include "gm/cuda/FT8Cuda.h"
#include "gm/cuda/FT8ScanCuda.h"
#include "gm/cuda/FT8SoftCuda.h"
#include "gm/cuda/HostCuda.h"
#include "gm/hf/ft8_capture.h"
#include "ft8_lib/ft8/constants.h"

namespace gm {
namespace cuda {

FT8Cuda::FT8Cuda(const gm::buffer::DeviceRingBuffer<uint8_t, 200>& ring,
                 float min_score, const std::string& tag, int zmq_port)
    : ring_(ring)
    , tag_(tag)
    , min_score_(min_score)
    , zmq_ctx_(1)
    , zmq_pub_(zmq_ctx_, ZMQ_PUB)
{
    if (zmq_port > 0)
        zmq_pub_.connect("tcp://localhost:" + std::to_string(zmq_port));

    cuda_check_error(cudaSetDevice(0));
    cuda_check_error(cudaStreamCreate(&cont_scan_stream_));

    allocContSlots();

    printf("FT8: time_osr=%d freq_osr=%d num_bins=%zu (GPU LLR mode)\n",
           FT8_TIME_OSR, FT8_FREQ_OSR, ring_.num_bins);
}

FT8Cuda::~FT8Cuda()
{
    cont_scan_active_.store(false, std::memory_order_release);
    if (cont_worker_thread_.joinable()) cont_worker_thread_.join();
    freeContSlots();
    cudaStreamDestroy(cont_scan_stream_);
}

void FT8Cuda::pubJson(const char* topic, const char* json)
{
    std::lock_guard<std::mutex> lk(zmq_mu_);
    zmq::message_t t(topic, strlen(topic));
    zmq::message_t d(json,  strlen(json));
    zmq_pub_.send(t, zmq::send_flags::sndmore);
    zmq_pub_.send(d, zmq::send_flags::none);
}

void FT8Cuda::setDecodeCallback(std::function<void(ContScanResult&)> cb)
{
    decode_callback_ = std::move(cb);
}

void FT8Cuda::startContinuousScan()
{
    cont_scan_active_.store(true, std::memory_order_release);
    cont_worker_thread_ = std::thread([this]() { contWorker(); });
}

void FT8Cuda::launchContScan(uint64_t wi)
{
    uint64_t cw = cont_write_idx_.load(std::memory_order_relaxed);
    uint64_t ri = cont_read_idx_.load(std::memory_order_acquire);
    if (cw - ri >= (uint64_t)CONTINUOUS_SLOTS) {
        fprintf(stderr, "[CONT] slots full, dropping block %llu\n",
                (unsigned long long)wi);
        return;
    }

    ContScanResult& slot = cont_slots_[cw % CONTINUOUS_SLOTS];
    uint64_t cont_snap   = wi - FT8_CAPTURE_BLOCKS;
    int dev_cont_snap    = (int)(cont_snap % 200);
    slot.snap_start      = cont_snap;

    cudaStreamWaitEvent(cont_scan_stream_, ring_.ready, 0);

    ft8_gpu_scan(
        ring_.base_d,
        dev_cont_snap, 200,
        slot.fo_d, slot.to_d, slot.ts_d, slot.fs_d,
        slot.score_d, slot.count_d, CONT_CAND_MAX,
        (int)ring_.num_bins, FT8_CAPTURE_BLOCKS,
        FT8_TIME_OSR, FT8_FREQ_OSR, min_score_,
        cont_scan_stream_);

    ft8_soft_symbols(
        ring_.base_d,
        dev_cont_snap, 200,
        slot.fo_d, slot.to_d, slot.ts_d, slot.fs_d,
        slot.count_d, slot.log174_d,
        (int)ring_.num_bins, FT8_CAPTURE_BLOCKS,
        FT8_TIME_OSR, FT8_FREQ_OSR, CONT_CAND_MAX,
        cont_scan_stream_);

    cudaMemcpyAsync(slot.count,  slot.count_d,  sizeof(uint32_t),
                   cudaMemcpyDeviceToHost, cont_scan_stream_);
    cudaMemcpyAsync(slot.fo,     slot.fo_d,     CONT_CAND_MAX * sizeof(int32_t),
                   cudaMemcpyDeviceToHost, cont_scan_stream_);
    cudaMemcpyAsync(slot.to,     slot.to_d,     CONT_CAND_MAX * sizeof(uint8_t),
                   cudaMemcpyDeviceToHost, cont_scan_stream_);
    cudaMemcpyAsync(slot.ts,     slot.ts_d,     CONT_CAND_MAX * sizeof(uint8_t),
                   cudaMemcpyDeviceToHost, cont_scan_stream_);
    cudaMemcpyAsync(slot.fs,     slot.fs_d,     CONT_CAND_MAX * sizeof(uint8_t),
                   cudaMemcpyDeviceToHost, cont_scan_stream_);
    cudaMemcpyAsync(slot.score,  slot.score_d,  CONT_CAND_MAX * sizeof(int16_t),
                   cudaMemcpyDeviceToHost, cont_scan_stream_);
    cudaMemcpyAsync(slot.log174, slot.log174_d,
                   (size_t)CONT_CAND_MAX * FTX_LDPC_N * sizeof(float),
                   cudaMemcpyDeviceToHost, cont_scan_stream_);

    cudaEventRecord(slot.event, cont_scan_stream_);
    slot.timestamp = std::chrono::duration_cast<std::chrono::duration<double>>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    slot.dispatched.store(true, std::memory_order_release);
    cont_write_idx_.fetch_add(1, std::memory_order_relaxed);
}

void FT8Cuda::run()
{
    uint64_t last_cont_scan = 0;

    while (isRunning()) {
        uint64_t wi = ring_.write_idx.load(std::memory_order_acquire);

        if (cont_scan_active_.load(std::memory_order_acquire) &&
            wi >= (uint64_t)FT8_CAPTURE_BLOCKS &&
            wi > last_cont_scan &&
            wi % (uint64_t)cont_stride_ == 0) {
            last_cont_scan = wi;
            launchContScan(wi);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void FT8Cuda::allocContSlots()
{
    const size_t log174_bytes = (size_t)CONT_CAND_MAX * FTX_LDPC_N * sizeof(float);
    for (int i = 0; i < CONTINUOUS_SLOTS; ++i) {
        ContScanResult& s = cont_slots_[i];
        cuda_check_error(cudaMalloc((void**)&s.count_d,  sizeof(uint32_t)));
        cuda_check_error(cudaMalloc((void**)&s.fo_d,     CONT_CAND_MAX * sizeof(int32_t)));
        cuda_check_error(cudaMalloc((void**)&s.to_d,     CONT_CAND_MAX * sizeof(uint8_t)));
        cuda_check_error(cudaMalloc((void**)&s.ts_d,     CONT_CAND_MAX * sizeof(uint8_t)));
        cuda_check_error(cudaMalloc((void**)&s.fs_d,     CONT_CAND_MAX * sizeof(uint8_t)));
        cuda_check_error(cudaMalloc((void**)&s.score_d,  CONT_CAND_MAX * sizeof(int16_t)));
        cuda_check_error(cudaMalloc((void**)&s.log174_d, log174_bytes));
        cuda_check_error(cudaHostAlloc((void**)&s.count,  sizeof(uint32_t),               cudaHostAllocDefault));
        cuda_check_error(cudaHostAlloc((void**)&s.fo,     CONT_CAND_MAX * sizeof(int32_t), cudaHostAllocDefault));
        cuda_check_error(cudaHostAlloc((void**)&s.to,     CONT_CAND_MAX * sizeof(uint8_t), cudaHostAllocDefault));
        cuda_check_error(cudaHostAlloc((void**)&s.ts,     CONT_CAND_MAX * sizeof(uint8_t), cudaHostAllocDefault));
        cuda_check_error(cudaHostAlloc((void**)&s.fs,     CONT_CAND_MAX * sizeof(uint8_t), cudaHostAllocDefault));
        cuda_check_error(cudaHostAlloc((void**)&s.score,  CONT_CAND_MAX * sizeof(int16_t), cudaHostAllocDefault));
        cuda_check_error(cudaHostAlloc((void**)&s.log174, log174_bytes,                    cudaHostAllocDefault));
        cuda_check_error(cudaEventCreateWithFlags(&s.event, cudaEventDisableTiming));
    }
    double mb = (double)CONTINUOUS_SLOTS * (log174_bytes + CONT_CAND_MAX * (4+1+1+1+2)) / 1e6;
    printf("Continuous scan: %d slots x %.1f MB = %.1f MB device + %.1f MB pinned\n",
           CONTINUOUS_SLOTS, mb / CONTINUOUS_SLOTS, mb, mb);
}

void FT8Cuda::freeContSlots()
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

void FT8Cuda::contWorker()
{
    uint64_t ri = 0;
    while (cont_scan_active_.load(std::memory_order_acquire)) {
        uint64_t wi = cont_write_idx_.load(std::memory_order_acquire);
        if (ri == wi) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        ContScanResult& slot = cont_slots_[ri % CONTINUOUS_SLOTS];
        while (!slot.dispatched.load(std::memory_order_acquire))
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        cudaError_t ev;
        while ((ev = cudaEventQuery(slot.event)) == cudaErrorNotReady)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

        if (ev != cudaSuccess) {
            fprintf(stderr, "[CONT] CUDA event error at slot %llu: %s\n",
                    (unsigned long long)(ri % CONTINUOUS_SLOTS),
                    cudaGetErrorString(ev));
        } else {
            uint32_t n = std::min(*slot.count, CONT_CAND_MAX);
            if (n > 0 && decode_callback_) {
                *slot.count = n;
                try {
                    decode_callback_(slot);
                } catch (...) {
                    fprintf(stderr, "[CONT] decode_callback threw\n");
                }
            }
        }
        slot.dispatched.store(false, std::memory_order_release);
        cont_read_idx_.store(++ri, std::memory_order_release);
    }
}

} // namespace cuda
} // namespace gm
