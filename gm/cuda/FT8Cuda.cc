#include <unistd.h>
#include <iostream>
#include <thread>
#include <cstring>
#include <cuda.h>
#include <complex>
#include <chrono>
#include <algorithm>

#include "gm/cuda/FT8Cuda.h"
#include "gm/cuda/FT8ScanCuda.h"
#include "gm/cuda/FT8SoftCuda.h"
#include "gm/cuda/HostCuda.h"
#include "ft8_lib/ft8/constants.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/rx888/rx888.h"
#include "gm/hf/ft8_capture.h"

#include "ft8_lib/common/monitor.h"

namespace gm {
namespace cuda {

FT8Cuda::FT8Cuda(gm::buffer::BufferPosition<std::complex<float>>* inP) :
inPos(inP),
buff_pos{0},
ring_write_idx(0),
last_trigger_second(0),
demodData_d(NULL),
demodFT8_d(NULL),
demodShift_d(NULL),
magFT8_d(NULL),
magFT8(NULL),
magFT8_ring_d(NULL),
log174_d(NULL),
log174(NULL),
gpu_cand_fo_d(NULL),
gpu_cand_to_d(NULL),
gpu_cand_ts_d(NULL),
gpu_cand_fs_d(NULL),
gpu_cand_score_d(NULL),
gpu_cand_count_d(NULL),
buffer_number(0) {
    inShape = inPos->getShape();
    inData_d = (std::complex<float>*)inPos->getBuffer();

    try {
        cuda_check_error(cudaSetDevice(0));
        cuda_check_error(cudaStreamCreate(&stream));
        cuda_check_error(cudaStreamCreate(&scan_stream));
        cuda_check_error(cudaStreamCreate(&transfer_stream));
        cuda_check_error(cudaEventCreateWithFlags(&scan_done,   cudaEventDisableTiming));
        cuda_check_error(cudaEventCreateWithFlags(&ring_ready,  cudaEventDisableTiming));

        cuda_h = gm::cuda::device::HostCuda(stream);
        rfft_length = 698880; // half off for some reason

        // 1-byte dummy keeps magFT8 non-null so rt8BufferPosition pointer stays valid.
        // The waterfall D2H is gone (Phase 4); this pointer is never written after init.
        cuda_check_error(cudaHostAlloc((void**)&magFT8, 1, cudaHostAllocDefault));
        memset(magFT8, 0, 1);
        rt8BufferPosition.setBuffer(magFT8, {BUFFERS, 1});

        // Two so we can use it as a buffer also
        cuda_check_error(cudaMalloc((void**)&demodData_d, 4*rfft_length*sizeof(std::complex<float>) + 1024));
        printf("Total rfft length is %u\n", rfft_length);

        cuda_check_error(cudaMalloc((void**)&demodFT8_d, FT8_TIME_OSR*FT8_FREQ_OSR*rfft_length*sizeof(std::complex<float>) + 1024));
        cuda_check_error(cudaMalloc((void**)&magFT8_d, FT8_TIME_OSR*FT8_FREQ_OSR*rfft_length*sizeof(uint8_t) + 1024));
        cuda_check_error(cudaMalloc((void**)&demodShift_d, rfft_length*sizeof(std::complex<float>) + 1024));
        if (!demodShift_d) {
            fprintf(stderr, "FT8Cuda: cudaMalloc failed for demodShift_d — out of GPU memory\n");
            exit(1);
        }

        // Device-side mag ring: RING_BLOCKS slots, wraps independently of FT8_CAPTURE_BLOCKS.
        const size_t dev_ring_bytes =
            (size_t)RING_BLOCKS * FT8_TIME_OSR * FT8_FREQ_OSR * rfft_length;
        cuda_check_error(cudaMalloc((void**)&magFT8_ring_d, dev_ring_bytes));
        cuda_check_error(cudaMemset(magFT8_ring_d, 0, dev_ring_bytes));
        printf("GPU mag ring: %.1f MB\n", (double)dev_ring_bytes / 1e6);

        // Soft symbol LLR output: FT8_GPU_CAND_MAX × 174 floats each on device and pinned host.
        const size_t log174_bytes = (size_t)FT8_GPU_CAND_MAX * FTX_LDPC_N * sizeof(float);
        cuda_check_error(cudaMalloc((void**)&log174_d, log174_bytes));
        cuda_check_error(cudaHostAlloc((void**)&log174, log174_bytes, cudaHostAllocDefault));
        printf("log174 alloc: %.0f MB device + %.0f MB pinned\n",
               (double)log174_bytes / 1.0e6, (double)log174_bytes / 1.0e6);

        // Candidate output buffers for GPU scan.
        cuda_check_error(cudaMalloc((void**)&gpu_cand_fo_d,    FT8_GPU_CAND_MAX * sizeof(int32_t)));
        cuda_check_error(cudaMalloc((void**)&gpu_cand_to_d,    FT8_GPU_CAND_MAX * sizeof(uint8_t)));
        cuda_check_error(cudaMalloc((void**)&gpu_cand_ts_d,    FT8_GPU_CAND_MAX * sizeof(uint8_t)));
        cuda_check_error(cudaMalloc((void**)&gpu_cand_fs_d,    FT8_GPU_CAND_MAX * sizeof(uint8_t)));
        cuda_check_error(cudaMalloc((void**)&gpu_cand_score_d, FT8_GPU_CAND_MAX * sizeof(int16_t)));
        cuda_check_error(cudaMalloc((void**)&gpu_cand_count_d, sizeof(uint32_t)));

        // Now for the sub channels
        cufftResult fftRes = cufftPlan1d(&rplan, rfft_length, CUFFT_C2C, 1);
        if (fftRes) {
            printf("Error: exit for now\n");
        }
        fftRes = cufftSetStream(rplan, stream);
        if (fftRes) {
            printf("Error: exit for now\n");
        }

    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error after cudaSetDevice: " << e.what() << std::endl;
    }
}

FT8Cuda::~FT8Cuda() {
    if (last_snapshot_thread.joinable()) last_snapshot_thread.join();
    if (demodData_d) cudaFree(demodData_d);
    if (demodFT8_d) cudaFree(demodFT8_d);
    if (demodShift_d) cudaFree(demodShift_d);
    if (magFT8_d) cudaFree(magFT8_d);
    if (magFT8_ring_d) cudaFree(magFT8_ring_d);
    if (log174_d) cudaFree(log174_d);
    if (log174) cudaFreeHost(log174);
    if (gpu_cand_fo_d) cudaFree(gpu_cand_fo_d);
    if (gpu_cand_to_d) cudaFree(gpu_cand_to_d);
    if (gpu_cand_ts_d) cudaFree(gpu_cand_ts_d);
    if (gpu_cand_fs_d) cudaFree(gpu_cand_fs_d);
    if (gpu_cand_score_d) cudaFree(gpu_cand_score_d);
    if (gpu_cand_count_d) cudaFree(gpu_cand_count_d);
    if (magFT8) cudaFreeHost(magFT8);
    cufftDestroy(rplan);
    cudaEventDestroy(scan_done);
    cudaEventDestroy(ring_ready);
    cudaStreamDestroy(stream);
    cudaStreamDestroy(scan_stream);
    cudaStreamDestroy(transfer_stream);
}

int FT8Cuda::doCopy(uint64_t now) {
    try {
        size_t length = inShape[1];

        cudaMemcpyAsync(&demodData_d[buff_pos], &inData_d[length * (now % inShape[0])],
                        length * sizeof(std::complex<float>), cudaMemcpyDeviceToDevice, stream);
        buff_pos += length;

        if (buff_pos > 2*rfft_length) {
            auto nowsec = std::chrono::system_clock::now();
            auto duration = nowsec.time_since_epoch();
            double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
            uint64_t trigger = (uint64_t)(seconds) % 15;
            lastsecond = seconds;

            for (int t = 0; t < FT8_TIME_OSR; t++) {
                for (int f = 0; f < FT8_FREQ_OSR; f++) {
                    std::complex<float>* input = &demodData_d[t * rfft_length / FT8_TIME_OSR];
                    if (f > 0) {
                        cuda_h.freqShift(input, demodShift_d, rfft_length,
                                         f * 6.25f / FT8_FREQ_OSR);
                        input = demodShift_d;
                    }
                    cufftResult rval = cufftExecC2C(rplan,
                        (cufftComplex *)input,
                        (cufftComplex *)&demodFT8_d[(t * FT8_FREQ_OSR + f) * rfft_length],
                        CUFFT_FORWARD);
                    if (rval) {
                        printf("Error in fft (t=%d f=%d)\n", t, f);
                        return 0;
                    }
                }
            }
            buff_pos -= rfft_length;
            cuda_check_error(cudaMemcpyAsync(&demodData_d[0],
                &demodData_d[rfft_length],
                buff_pos * sizeof(std::complex<float>), cudaMemcpyDeviceToDevice, stream));

            // Compute magnitude and append to device ring.
            const size_t block_bytes = (size_t)FT8_TIME_OSR * FT8_FREQ_OSR * rfft_length;
            cuda_h.magKernel(&demodFT8_d[0], &magFT8_d[0], FT8_TIME_OSR*FT8_FREQ_OSR*rfft_length);
            cuda_check_error(cudaMemcpyAsync(
                &magFT8_ring_d[(ring_write_idx % RING_BLOCKS) * block_bytes],
                &magFT8_d[0],
                block_bytes * sizeof(uint8_t), cudaMemcpyDeviceToDevice, stream));
            ring_write_idx++;

            // On trigger: snapshot the last FT8_CAPTURE_BLOCKS blocks into a decode slot.
            // Background thread does the ring→slot copy so doCopy never blocks.
            uint64_t trigger_second = (uint64_t)(seconds);
            bool gotime = (trigger == 14 &&
                           (seconds - trunc(seconds)) > 0.7 &&
                           trigger_second != last_trigger_second &&
                           ring_write_idx >= (uint64_t)FT8_CAPTURE_BLOCKS);
            if (gotime) {
                last_trigger_second = trigger_second;
                printf("Processing buffer %u\n", buffer_number);

                int decode_slot = buffer_number % BUFFERS;
                uint64_t snap_start = ring_write_idx - FT8_CAPTURE_BLOCKS;
                int next_buf = buffer_number + 1;
                int dev_snap_start = (int)(snap_start % RING_BLOCKS);

                // Commit all device ring writes up to this point.
                cudaEventRecord(ring_ready, stream);

                // scan_stream and transfer_stream both wait for ring writes before proceeding.
                cudaStreamWaitEvent(scan_stream,    ring_ready, 0);
                cudaStreamWaitEvent(transfer_stream, ring_ready, 0);

                // Launch GPU sync scan on scan_stream (non-blocking for doCopy hot path).
                ft8_gpu_scan(
                    magFT8_ring_d,
                    dev_snap_start, RING_BLOCKS,
                    gpu_cand_fo_d, gpu_cand_to_d, gpu_cand_ts_d, gpu_cand_fs_d,
                    gpu_cand_score_d, gpu_cand_count_d, FT8_GPU_CAND_MAX,
                    (int)rfft_length, FT8_CAPTURE_BLOCKS,
                    FT8_TIME_OSR, FT8_FREQ_OSR, /*min_score=*/5,
                    scan_stream);

                // Soft symbols kernel runs after Costas scan on the same stream.
                // scan_done fires after both kernels complete.
                ft8_soft_symbols(
                    magFT8_ring_d,
                    dev_snap_start, RING_BLOCKS,
                    gpu_cand_fo_d, gpu_cand_to_d, gpu_cand_ts_d, gpu_cand_fs_d,
                    gpu_cand_count_d, log174_d,
                    (int)rfft_length, FT8_CAPTURE_BLOCKS,
                    FT8_TIME_OSR, FT8_FREQ_OSR,
                    scan_stream);
                cudaEventRecord(scan_done, scan_stream);

                if (last_snapshot_thread.joinable())
                    last_snapshot_thread.join();

                last_snapshot_thread = std::thread([this, decode_slot, next_buf]() {
                    // Wait for Costas scan + soft symbols kernel.
                    cudaEventSynchronize(scan_done);

                    uint32_t n = 0;
                    cudaMemcpy(&n, gpu_cand_count_d, sizeof(uint32_t), cudaMemcpyDeviceToHost);
                    n = std::min(n, (uint32_t)FT8_GPU_CAND_MAX);

                    GpuScanResult& res = gpu_results[decode_slot];
                    res.count = n;
                    if (n > 0) {
                        res.fo.resize(n);
                        res.to.resize(n);
                        res.ts.resize(n);
                        res.fs.resize(n);
                        res.score.resize(n);
                        res.log174.resize((size_t)n * FTX_LDPC_N);
                        cudaMemcpy(res.fo.data(),    gpu_cand_fo_d,    n * sizeof(int32_t), cudaMemcpyDeviceToHost);
                        cudaMemcpy(res.to.data(),    gpu_cand_to_d,    n * sizeof(uint8_t), cudaMemcpyDeviceToHost);
                        cudaMemcpy(res.ts.data(),    gpu_cand_ts_d,    n * sizeof(uint8_t), cudaMemcpyDeviceToHost);
                        cudaMemcpy(res.fs.data(),    gpu_cand_fs_d,    n * sizeof(uint8_t), cudaMemcpyDeviceToHost);
                        cudaMemcpy(res.score.data(), gpu_cand_score_d, n * sizeof(int16_t), cudaMemcpyDeviceToHost);
                        cudaMemcpy(log174, log174_d, (size_t)n * FTX_LDPC_N * sizeof(float), cudaMemcpyDeviceToHost);
                        memcpy(res.log174.data(), log174, (size_t)n * FTX_LDPC_N * sizeof(float));
                    } else {
                        res.log174.clear();
                    }

                    // Signal ft8.cc: GPU scan + soft LLRs are ready.
                    rt8BufferPosition.setPosition(next_buf, 1);
                });

                buffer_number++;
            }

            lastepoch = seconds;
        }

        return 1;
    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error after cudaCopy: " << e.what() << std::endl;
    }
    return 0;
}

void FT8Cuda::run() {
    uint64_t now = inPos->getNow(1) + 1;
    
    while(isRunning()) {
        uint64_t next = inPos->getPosition(now+1, 1);
        while(now < next) {
            uint64_t length = next - now;
            if (length > 4) {
                std::cout << "Error Falling Behind in FT8Cuda Copy, Dropping Data" << std::endl;
                now = next;
                break;
            }
            int numCopied = doCopy(now);
            if (!numCopied) exit(-200);
	        now += numCopied;
        }
    }
}

}
}
