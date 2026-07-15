#include <chrono>
#include <cstdio>
#include <algorithm>
#include "gm/cuda/JS8Cuda.h"
#include "gm/cuda/JS8SoftCuda.h"
#include "gm/cuda/JS8LdpcCuda.h"
#include "gm/cuda/HostCuda.h"
#include "gm/hf/ft8_capture.h"
#include "gm/hf/refine.h"

namespace gm {
namespace cuda {

template<int N>
JS8Cuda<N>::JS8Cuda(const gm::buffer::DeviceRingBuffer<uint8_t, N>& ring,
                    float min_score, int zmq_port,
                    JS8ScanFn scan_fn, int time_osr, int freq_osr, int cap_blocks,
                    const char* label, bool legacy_costas,
                    const gm::buffer::DeviceRingBuffer<std::complex<float>,
                          gm::buffer::kComplexCompositeBlocks>* cplx_ring,
                    const uint64_t* slot_cplx_idx)
    : ring_(ring), min_score_(min_score), legacy_costas_(legacy_costas),
      cplx_ring_(cplx_ring), slot_cplx_idx_(slot_cplx_idx),
      scan_fn_(scan_fn), time_osr_(time_osr), freq_osr_(freq_osr), cap_blocks_(cap_blocks),
      label_(label),
      zmq_ctx_(1), zmq_pub_(zmq_ctx_, ZMQ_PUB)
{
    if (zmq_port > 0)
        zmq_pub_.connect("tcp://localhost:" + std::to_string(zmq_port));
    cudaStreamCreate(&js8_scan_stream_);
    if (cplx_ring_) cudaStreamCreate(&refine_stream_);
    js8_ldpc_init_constants();

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

template<int N>
JS8Cuda<N>::~JS8Cuda()
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
    if (refine_stream_) cudaStreamDestroy(refine_stream_);
}

template<int N>
void JS8Cuda<N>::allocSlots()
{
    const size_t log174_bytes = (size_t)CAND_MAX * kFtxLdpcN * sizeof(float);
    const size_t x_hat_bytes  = (size_t)CAND_MAX * kFtxLdpcN * sizeof(uint8_t);
    const size_t parity_bytes = (size_t)CAND_MAX * sizeof(bool);
    for (int i = 0; i < CONTINUOUS_SLOTS; ++i) {
        ContScanResult& s = cont_slots_[i];
        cudaMalloc((void**)&s.count_d,  sizeof(uint32_t));
        cudaMalloc((void**)&s.fo_d,     CAND_MAX * sizeof(int32_t));
        cudaMalloc((void**)&s.to_d,     CAND_MAX * sizeof(uint8_t));
        cudaMalloc((void**)&s.ts_d,     CAND_MAX * sizeof(uint8_t));
        cudaMalloc((void**)&s.fs_d,     CAND_MAX * sizeof(uint8_t));
        cudaMalloc((void**)&s.score_d,  CAND_MAX * sizeof(int16_t));
        cudaMalloc((void**)&s.log174_d, log174_bytes);
        cudaMalloc((void**)&s.x_hat_d,  x_hat_bytes);
        cudaMalloc((void**)&s.parity_d, parity_bytes);
        cudaHostAlloc((void**)&s.count,  sizeof(uint32_t),           cudaHostAllocDefault);
        cudaHostAlloc((void**)&s.fo,     CAND_MAX * sizeof(int32_t), cudaHostAllocDefault);
        cudaHostAlloc((void**)&s.to,     CAND_MAX * sizeof(uint8_t), cudaHostAllocDefault);
        cudaHostAlloc((void**)&s.ts,     CAND_MAX * sizeof(uint8_t), cudaHostAllocDefault);
        cudaHostAlloc((void**)&s.fs,     CAND_MAX * sizeof(uint8_t), cudaHostAllocDefault);
        cudaHostAlloc((void**)&s.score,  CAND_MAX * sizeof(int16_t), cudaHostAllocDefault);
        cudaHostAlloc((void**)&s.log174, log174_bytes,               cudaHostAllocDefault);
        cudaHostAlloc((void**)&s.x_hat,  x_hat_bytes,                cudaHostAllocDefault);
        cudaHostAlloc((void**)&s.parity, parity_bytes,               cudaHostAllocDefault);
        cudaEventCreateWithFlags(&s.event, cudaEventDisableTiming);
    }
}

template<int N>
void JS8Cuda<N>::freeSlots()
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
        if (s.x_hat_d)  cudaFree(s.x_hat_d);
        if (s.parity_d) cudaFree(s.parity_d);
        if (s.count)    cudaFreeHost(s.count);
        if (s.fo)       cudaFreeHost(s.fo);
        if (s.to)       cudaFreeHost(s.to);
        if (s.ts)       cudaFreeHost(s.ts);
        if (s.fs)       cudaFreeHost(s.fs);
        if (s.score)    cudaFreeHost(s.score);
        if (s.log174)   cudaFreeHost(s.log174);
        if (s.x_hat)    cudaFreeHost(s.x_hat);
        if (s.parity)   cudaFreeHost(s.parity);
        if (s.event)    cudaEventDestroy(s.event);
    }
}

template<int N>
void JS8Cuda<N>::start()
{
    running_.store(true, std::memory_order_release);
    scan_thread_   = std::thread([this]() { scanLoop(); });
    worker_thread_ = std::thread([this]() { workerLoop(); });
}

template<int N>
void JS8Cuda<N>::stop()
{
    if (!running_.load(std::memory_order_acquire)) return;
    running_.store(false, std::memory_order_release);
    if (scan_thread_.joinable())   scan_thread_.join();
    if (worker_thread_.joinable()) worker_thread_.join();
}

template<int N>
void JS8Cuda<N>::scanLoop()
{
    constexpr int STRIDE = 6;

    uint64_t last_triggered = 0;

    while (running_.load(std::memory_order_acquire)) {
        uint64_t wi = ring_.write_idx.load(std::memory_order_acquire);

        if (wi < (uint64_t)cap_blocks_ || wi <= last_triggered ||
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

        ContScanResult& slot = cont_slots_[cw % CONTINUOUS_SLOTS];
        int snap_start       = (int)((wi - cap_blocks_) % N);
        slot.snap_start      = wi - cap_blocks_;   // absolute, for refine's ring map

        { std::lock_guard<std::mutex> lk(ring_.ready_mu);
          cudaStreamWaitEvent(js8_scan_stream_, ring_.ready, 0); }

        scan_fn_(
            ring_.base_d, snap_start, N,
            slot.fo_d, slot.to_d, slot.ts_d, slot.fs_d,
            slot.score_d, slot.count_d, CAND_MAX,
            (int)ring_.num_bins, cap_blocks_,
            time_osr_, freq_osr_, min_score_,
            js8_scan_stream_, legacy_costas_);

        js8_soft_symbols(
            ring_.base_d, snap_start, N,
            slot.fo_d, slot.to_d, slot.ts_d, slot.fs_d,
            slot.count_d, slot.log174_d,
            (int)ring_.num_bins, cap_blocks_,
            time_osr_, freq_osr_, (int)CAND_MAX,
            js8_scan_stream_);

        js8_sp_flooded_decode_batch(
            slot.count_d, CAND_MAX, slot.log174_d, slot.x_hat_d, slot.parity_d,
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
        cudaMemcpyAsync(slot.x_hat,  slot.x_hat_d,
                        (size_t)CAND_MAX * kFtxLdpcN * sizeof(uint8_t),
                        cudaMemcpyDeviceToHost, js8_scan_stream_);
        cudaMemcpyAsync(slot.parity, slot.parity_d,
                        (size_t)CAND_MAX * sizeof(bool),
                        cudaMemcpyDeviceToHost, js8_scan_stream_);

        cudaEventRecord(slot.event, js8_scan_stream_);
        slot_dispatch_ns_[cw % CONTINUOUS_SLOTS] =
            std::chrono::steady_clock::now().time_since_epoch().count();
        slot.dispatched.store(true, std::memory_order_release);
        cont_write_idx_.fetch_add(1, std::memory_order_relaxed);
    }
}

template<int N>
void JS8Cuda<N>::workerLoop()
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
                if (decode_callback_) {
                    *slot.count = n;
                    try {
                        decode_callback_(slot);
                    } catch (...) {
                        fprintf(stderr, "[%s] decode_callback threw\n", label_);
                    }
                }
                {
                    char buf[96];
                    snprintf(buf, sizeof(buf), "{\"scan_ms\":%.1f,\"n\":%u}", scan_ms, n);
                    std::lock_guard<std::mutex> lk(zmq_mu_);
                    zmq::message_t t("js8/timing", 10);
                    zmq::message_t d(buf, strlen(buf));
                    zmq_pub_.send(t, zmq::send_flags::sndmore);
                    zmq_pub_.send(d, zmq::send_flags::none);
                }
            } else if (ri % 60 == 0) {
                fprintf(stderr, "[%s] alive: %lu scans, 0 candidates\n", label_, (unsigned long)ri);
            }
        } else {
            fprintf(stderr, "[%s] CUDA event error at slot %lu: %s\n", label_,
                    (unsigned long)(ri % CONTINUOUS_SLOTS), cudaGetErrorString(ev));
        }

        slot.dispatched.store(false, std::memory_order_release);
        cont_read_idx_.store(++ri, std::memory_order_release);
    }
}

template<int N>
bool JS8Cuda<N>::refineCandidate(int32_t fo, int to, uint64_t snap_start, float* log174)
{
    if (!cplx_ring_ || !slot_cplx_idx_) return false;

    // One composite block = cplx_ring_->num_bins samples; one symbol (ring_.num_bins
    // samples) spans that many blocks; a frame is 79 symbols. Mirrors FT8Cuda.
    const int BLK            = (int)cplx_ring_->num_bins;
    const int blocks_per_sym = (int)(ring_.num_bins / cplx_ring_->num_bins);
    const int FRAME_BLOCKS   = gm::hf::kRefineNsym * blocks_per_sym;
    const int n_in           = FRAME_BLOCKS * BLK;

    const uint64_t mag_slot   = snap_start + (uint64_t)to;
    const uint64_t cplx_start = slot_cplx_idx_[mag_slot % N];
    const uint64_t cwi        = cplx_ring_->write_idx.load(std::memory_order_acquire);

    if (cplx_start + (uint64_t)FRAME_BLOCKS > cwi) return false;                 // incomplete
    if (cwi - cplx_start > (uint64_t)gm::buffer::kComplexCompositeBlocks) return false; // aged out

    if ((int)refine_host_.size() != n_in) refine_host_.resize(n_in);
    const uint64_t s0    = cplx_start % (uint64_t)gm::buffer::kComplexCompositeBlocks;
    const int      first = (int)std::min<uint64_t>(
        (uint64_t)FRAME_BLOCKS, (uint64_t)gm::buffer::kComplexCompositeBlocks - s0);
    cuda_check_error(cudaMemcpyAsync(
        refine_host_.data(), cplx_ring_->base_d + s0 * (uint64_t)BLK,
        (size_t)first * BLK * sizeof(std::complex<float>),
        cudaMemcpyDeviceToHost, refine_stream_));
    if (first < FRAME_BLOCKS) {
        cuda_check_error(cudaMemcpyAsync(
            refine_host_.data() + (size_t)first * BLK, cplx_ring_->base_d,
            (size_t)(FRAME_BLOCKS - first) * BLK * sizeof(std::complex<float>),
            cudaMemcpyDeviceToHost, refine_stream_));
    }
    cuda_check_error(cudaStreamSynchronize(refine_stream_));

    if ((int)refine_decim_.size() != gm::hf::kRefineFrame)
        refine_decim_.resize(gm::hf::kRefineFrame);
    const float freq_hz = (float)fo * (409600.0f / (float)ring_.num_bins);
    gm::hf::extract_frame(refine_host_.data(), n_in, freq_hz, 409600,
                          refine_decim_.data(), gm::hf::kRefineFrame);
    gm::hf::refine_llr(refine_decim_.data(), gm::hf::kRefineFrame,
                       gm::hf::kJS8Refine, log174, gm::hf::dechirp_enabled());
    return true;
}

template class JS8Cuda<200>;
template class JS8Cuda<128>;

} // namespace cuda
} // namespace gm
