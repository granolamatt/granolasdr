#include <unistd.h>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <chrono>
#include <algorithm>
#include <thread>
#include "gm/cuda/FT8Cuda.h"
#include "gm/cuda/FT8ScanCuda.h"
#include "gm/cuda/FT8SoftCuda.h"
#include "gm/cuda/FT8LdpcCuda.h"
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

    // Scan streams
    cuda_check_error(cudaStreamCreate(&scan_stream_));
    cuda_check_error(cudaStreamCreate(&transfer_stream_));
    cuda_check_error(cudaStreamCreate(&cont_scan_stream_));
    {
        int lo_pri, hi_pri;
        cudaDeviceGetStreamPriorityRange(&lo_pri, &hi_pri);
        cuda_check_error(cudaStreamCreateWithPriority(
            &ldpc_stream_, cudaStreamNonBlocking, lo_pri));
    }

    cuda_check_error(cudaEventCreateWithFlags(&scan_done_, cudaEventDisableTiming));
    cuda_check_error(cudaEventCreateWithFlags(&ldpc_done_, cudaEventDisableTiming));

    // Soft LLR buffers (epoch path)
    const size_t log174_bytes = (size_t)FT8_GPU_CAND_MAX * FTX_LDPC_N * sizeof(float);
    cuda_check_error(cudaMalloc((void**)&log174_d_, log174_bytes));
    cuda_check_error(cudaHostAlloc((void**)&log174_, log174_bytes, cudaHostAllocDefault));
    printf("log174 alloc: %.0f MB device + %.0f MB pinned\n",
           (double)log174_bytes / 1.0e6, (double)log174_bytes / 1.0e6);

    // GPU candidate buffers (epoch path)
    cuda_check_error(cudaMalloc((void**)&gpu_cand_fo_d_,    FT8_GPU_CAND_MAX * sizeof(int32_t)));
    cuda_check_error(cudaMalloc((void**)&gpu_cand_to_d_,    FT8_GPU_CAND_MAX * sizeof(uint8_t)));
    cuda_check_error(cudaMalloc((void**)&gpu_cand_ts_d_,    FT8_GPU_CAND_MAX * sizeof(uint8_t)));
    cuda_check_error(cudaMalloc((void**)&gpu_cand_fs_d_,    FT8_GPU_CAND_MAX * sizeof(uint8_t)));
    cuda_check_error(cudaMalloc((void**)&gpu_cand_score_d_, FT8_GPU_CAND_MAX * sizeof(int16_t)));
    cuda_check_error(cudaMalloc((void**)&gpu_cand_count_d_, sizeof(uint32_t)));

    // GPU LDPC output buffers
    cuda_check_error(cudaMalloc((void**)&x_hat_d_,
        (size_t)FT8_LDPC_BATCH * FTX_LDPC_N * sizeof(uint8_t)));
    cuda_check_error(cudaMalloc((void**)&parity_d_,
        (size_t)FT8_LDPC_BATCH * sizeof(bool)));

    // 1-byte dummy keeps rt8BufferPosition_ pointer non-null
    cuda_check_error(cudaHostAlloc((void**)&magFT8_dummy_, 1, cudaHostAllocDefault));
    memset(magFT8_dummy_, 0, 1);
    rt8BufferPosition_.setBuffer(magFT8_dummy_, {BUFFERS, 1});

    ft8_ldpc_init_constants();
    allocContSlots();

    printf("FT8: time_osr=%d freq_osr=%d num_bins=%zu (GPU LLR mode)\n",
           FT8_TIME_OSR, FT8_FREQ_OSR, ring_.num_bins);
    printf("FT8 ZMQ publisher -> tcp://localhost:%d  topic=ft8/decode\n", zmq_port);
}

FT8Cuda::~FT8Cuda()
{
    cont_scan_active_.store(false, std::memory_order_release);
    if (cont_worker_thread_.joinable()) cont_worker_thread_.join();
    if (last_snapshot_thread_.joinable()) last_snapshot_thread_.join();

    if (log174_d_)          cudaFree(log174_d_);
    if (log174_)            cudaFreeHost(log174_);
    if (gpu_cand_fo_d_)     cudaFree(gpu_cand_fo_d_);
    if (gpu_cand_to_d_)     cudaFree(gpu_cand_to_d_);
    if (gpu_cand_ts_d_)     cudaFree(gpu_cand_ts_d_);
    if (gpu_cand_fs_d_)     cudaFree(gpu_cand_fs_d_);
    if (gpu_cand_score_d_)  cudaFree(gpu_cand_score_d_);
    if (gpu_cand_count_d_)  cudaFree(gpu_cand_count_d_);
    if (x_hat_d_)           cudaFree(x_hat_d_);
    if (parity_d_)          cudaFree(parity_d_);
    if (magFT8_dummy_)      cudaFreeHost(magFT8_dummy_);
    freeContSlots();

    cudaEventDestroy(scan_done_);
    cudaEventDestroy(ldpc_done_);
    cudaStreamDestroy(scan_stream_);
    cudaStreamDestroy(transfer_stream_);
    cudaStreamDestroy(ldpc_stream_);
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

void FT8Cuda::launchEpochScan(uint64_t wi, double seconds)
{
    static uint64_t s_prev_ring = 0;
    static double   s_prev_sec  = 0.0;
    if (s_prev_sec > 0.0) {
        double dt   = seconds - s_prev_sec;
        double rate = (double)(wi - s_prev_ring) / dt;
        printf("TIMING: epoch=%.3f mod15=%.3f ring=%llu blocks/s=%.2f (nominal 6.25)\n",
               seconds, fmod(seconds, 15.0), (unsigned long long)wi, rate);
    }
    s_prev_ring = wi;
    s_prev_sec  = seconds;

    printf("Processing buffer %u\n", buffer_number_);

    int      decode_slot    = buffer_number_ % BUFFERS;
    uint64_t snap_start     = wi - FT8_CAPTURE_BLOCKS;
    int      next_buf       = buffer_number_ + 1;
    int      dev_snap_start = (int)(snap_start % 200);

    cudaStreamWaitEvent(scan_stream_,     ring_.ready, 0);
    cudaStreamWaitEvent(transfer_stream_, ring_.ready, 0);

    ft8_gpu_scan(
        ring_.base_d,
        dev_snap_start, 200,
        gpu_cand_fo_d_, gpu_cand_to_d_, gpu_cand_ts_d_, gpu_cand_fs_d_,
        gpu_cand_score_d_, gpu_cand_count_d_, FT8_GPU_CAND_MAX,
        (int)ring_.num_bins, FT8_CAPTURE_BLOCKS,
        FT8_TIME_OSR, FT8_FREQ_OSR, min_score_,
        scan_stream_);

    ft8_soft_symbols(
        ring_.base_d,
        dev_snap_start, 200,
        gpu_cand_fo_d_, gpu_cand_to_d_, gpu_cand_ts_d_, gpu_cand_fs_d_,
        gpu_cand_count_d_, log174_d_,
        (int)ring_.num_bins, FT8_CAPTURE_BLOCKS,
        FT8_TIME_OSR, FT8_FREQ_OSR, FT8_GPU_CAND_MAX,
        scan_stream_);

    cudaEventRecord(scan_done_, scan_stream_);

    if (last_snapshot_thread_.joinable())
        last_snapshot_thread_.join();

    double snap_seconds = seconds;
    last_snapshot_thread_ = std::thread([this, decode_slot, next_buf, snap_seconds]() {
        auto t_thread_start = std::chrono::steady_clock::now();
        cudaEventSynchronize(scan_done_);
        auto t_scan_done = std::chrono::steady_clock::now();
        double scan_ms = std::chrono::duration<double, std::milli>(
            t_scan_done - t_thread_start).count();

        uint32_t n = 0;
        cudaMemcpy(&n, gpu_cand_count_d_, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        uint32_t n_raw = n;
        n = std::min(n, (uint32_t)FT8_GPU_CAND_MAX);
        if (n_raw > FT8_GPU_CAND_MAX)
            fprintf(stderr, "[EPOCH] WARNING: candidate count %u clamped to %u\n",
                    n_raw, (uint32_t)FT8_GPU_CAND_MAX);

        bool ldpc_launched = (n > 0);
        if (ldpc_launched) {
            ft8_bp_decode_batch(n, log174_d_, x_hat_d_, parity_d_, ldpc_stream_);
            cudaEventRecord(ldpc_done_, ldpc_stream_);
        }

        bool   ldpc_ok = false;
        double ldpc_ms = 0.0;
        if (ldpc_launched) {
            auto t_ldpc_start = std::chrono::steady_clock::now();
            auto deadline     = t_ldpc_start + std::chrono::milliseconds(500);
            cudaError_t ev;
            while ((ev = cudaEventQuery(ldpc_done_)) == cudaErrorNotReady) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    ldpc_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t_ldpc_start).count();
                    fprintf(stderr,
                            "[EPOCH TIMING][%s] scan=%.1fms ldpc=TIMEOUT(%.1fms) n=%u\n",
                            tag_.c_str(), scan_ms, ldpc_ms, n);
                    goto ldpc_timeout;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            ldpc_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t_ldpc_start).count();
            if (ev != cudaSuccess)
                fprintf(stderr, "[EPOCH] CUDA ldpc_done error: %s\n",
                        cudaGetErrorString(ev));
            else
                ldpc_ok = true;
            ldpc_timeout:;
        }

        bool timed_out = ldpc_launched && !ldpc_ok;
        if (timed_out) {
            if (++consecutive_timeouts_ > 2)
                fprintf(stderr, "[EPOCH] WARNING: %d consecutive LDPC timeouts\n",
                        consecutive_timeouts_);
        } else {
            consecutive_timeouts_ = 0;
        }

        fprintf(stderr, "[EPOCH TIMING][%s] scan=%.1fms ldpc=%.1fms n=%u ldpc_ok=%d\n",
                tag_.c_str(), scan_ms, ldpc_ms, n, (int)ldpc_ok);
        {
            char buf[128];
            snprintf(buf, sizeof(buf),
                     "{\"scan_ms\":%.1f,\"ldpc_ms\":%.1f,\"n\":%u}",
                     (float)scan_ms, (float)ldpc_ms, n);
            pubJson("ft8/timing", buf);
        }

        GpuScanResult& res = gpu_results_[decode_slot];
        res.count = n;
        if (n > 0) {
            res.fo.resize(n); res.to.resize(n);
            res.ts.resize(n); res.fs.resize(n);
            res.score.resize(n);
            res.log174.resize((size_t)n * FTX_LDPC_N);
            cudaMemcpy(res.fo.data(),    gpu_cand_fo_d_,    n * sizeof(int32_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(res.to.data(),    gpu_cand_to_d_,    n * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(res.ts.data(),    gpu_cand_ts_d_,    n * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(res.fs.data(),    gpu_cand_fs_d_,    n * sizeof(uint8_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(res.score.data(), gpu_cand_score_d_, n * sizeof(int16_t), cudaMemcpyDeviceToHost);
            cudaMemcpy(log174_, log174_d_,
                       (size_t)n * FTX_LDPC_N * sizeof(float), cudaMemcpyDeviceToHost);
            memcpy(res.log174.data(), log174_,
                   (size_t)n * FTX_LDPC_N * sizeof(float));

            if (ldpc_ok) {
                res.x_hat.resize((size_t)n * FTX_LDPC_N);
                res.parity.resize(n);
                cudaMemcpy(res.x_hat.data(), x_hat_d_,
                           (size_t)n * FTX_LDPC_N * sizeof(uint8_t),
                           cudaMemcpyDeviceToHost);
                bool* par_stage = new bool[n];
                cudaMemcpy(par_stage, parity_d_, n * sizeof(bool),
                           cudaMemcpyDeviceToHost);
                for (uint32_t i = 0; i < n; ++i) res.parity[i] = par_stage[i];
                delete[] par_stage;
            } else {
                res.x_hat.clear();
                res.parity.clear();
            }
        } else {
            res.log174.clear();
            res.x_hat.clear();
            res.parity.clear();
        }

        rt8BufferPosition_.setPosition(next_buf, 1);
    });

    buffer_number_++;
}

void FT8Cuda::run()
{
    uint64_t last_cont_scan     = 0;

    while (isRunning()) {
        uint64_t wi = ring_.write_idx.load(std::memory_order_acquire);

        // Continuous Costas scan: fire every cont_stride_ ring blocks.
        if (cont_scan_active_.load(std::memory_order_acquire) &&
            wi >= (uint64_t)FT8_CAPTURE_BLOCKS &&
            wi > last_cont_scan &&
            wi % (uint64_t)cont_stride_ == 0) {
            last_cont_scan = wi;
            launchContScan(wi);
        }

        // Epoch trigger: wall-clock mod-15 == 14, fractional part > 0.7.
        auto nowsec = std::chrono::system_clock::now();
        double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(
            nowsec.time_since_epoch()).count();
        uint64_t trig_mod = (uint64_t)(seconds) % 15;
        if (trig_mod == 14 &&
            (seconds - std::trunc(seconds)) > 0.7 &&
            (uint64_t)(seconds) != last_trigger_second_ &&
            wi >= (uint64_t)FT8_CAPTURE_BLOCKS) {
            last_trigger_second_ = (uint64_t)(seconds);
            launchEpochScan(wi, seconds);
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
