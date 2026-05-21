#include <unistd.h>
#include <iostream>
#include <thread>
#include <cstring>
#include <cuda.h>
#include <complex>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <sys/stat.h>
#include "gm/cuda/FT8Cuda.h"
#include "gm/cuda/FT8ScanCuda.h"
#include "gm/cuda/FT8SoftCuda.h"
#include "gm/cuda/HostCuda.h"
#include "ft8_lib/ft8/constants.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/rx888/rx888.h"
#include "gm/hf/ft8_capture.h"
#include "gm/hf/hf_bands.h"

#include "ft8_lib/common/monitor.h"
#include "ft8_lib/common/wave.h"
#include "ft8_lib/fft/kiss_fft.h"

namespace gm {
namespace cuda {

FT8Cuda::FT8Cuda(gm::buffer::BufferPosition<std::complex<float>>* inP, bool corpus) :
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
buffer_number(0),
enable_corpus(corpus),
audio_ft8_bin(0),
audio_ft8_bin_10m(0),
audio_sample_rate(0),
audioBins_host(NULL),
audioBins_host_10m(NULL) {
    inShape = inPos->getShape();
    inData_d = (std::complex<float>*)inPos->getBuffer();

    try {
        cuda_check_error(cudaSetDevice(0));
        cuda_check_error(cudaStreamCreate(&stream));
        cuda_check_error(cudaStreamCreate(&scan_stream));
        cuda_check_error(cudaStreamCreate(&transfer_stream));
        cuda_check_error(cudaEventCreateWithFlags(&scan_done,  cudaEventDisableTiming));
        cuda_check_error(cudaEventCreateWithFlags(&ring_ready, cudaEventDisableTiming));

        cuda_h = gm::cuda::device::HostCuda(stream);
        rfft_length = 1048576; // 2^20; 6553600 Hz composite / 6.25 Hz/bin

        // Compute FT8 FFT bins for 20m (14.074 MHz) and 10m (28.074 MHz) dials.
        // Chain: wb_bin = round(freq * 1400000 / 140e6) = round(freq / 100)
        //        ifft_bin = (sum of bw[0..band-1]) + (wb_bin - wb_start_band)
        //        ft8_bin  = round(ifft_bin * rfft_length / 65536)
        {
            const int kWbFftSize = 1400000;
            const int kIfftSize  = 65536;

            // 20m: kHFBands index 5
            {
                int wb_bin = (int)roundf(14074000.0f * kWbFftSize / 140000000.0f);
                int ifft_start = 0;
                for (int i = 0; i < 5; ++i) ifft_start += (int)kHFBands[i].bw;
                int ifft_bin  = ifft_start + (wb_bin - (int)kHFBands[5].wb_start);
                audio_ft8_bin = (int)roundf((float)ifft_bin * rfft_length / kIfftSize);
                audio_sample_rate = (int)roundf(6553600.0f * FT8_AUDIO_BINS / rfft_length);
                printf("20m audio: wb_bin=%d ifft_bin=%d ft8_bin=%d audio_rate=%d Hz\n",
                       wb_bin, ifft_bin, audio_ft8_bin, audio_sample_rate);
            }

            // 10m: kHFBands index 9
            {
                int wb_bin = (int)roundf(28074000.0f * kWbFftSize / 140000000.0f);
                int ifft_start = 0;
                for (int i = 0; i < 9; ++i) ifft_start += (int)kHFBands[i].bw;
                int ifft_bin  = ifft_start + (wb_bin - (int)kHFBands[9].wb_start);
                audio_ft8_bin_10m = (int)roundf((float)ifft_bin * rfft_length / kIfftSize);
                printf("10m audio: wb_bin=%d ifft_bin=%d ft8_bin=%d\n",
                       wb_bin, ifft_bin, audio_ft8_bin_10m);
            }
        }

        // Pinned audio rings for corpus capture (enabled with --jtdx): one each for 20m and 10m.
        if (enable_corpus) {
            const size_t audio_ring_bytes =
                (size_t)AUDIO_RING_SLOTS * FT8_AUDIO_BINS * sizeof(std::complex<float>);
            cuda_check_error(cudaHostAlloc((void**)&audioBins_host,
                                           audio_ring_bytes, cudaHostAllocDefault));
            cuda_check_error(cudaHostAlloc((void**)&audioBins_host_10m,
                                           audio_ring_bytes, cudaHostAllocDefault));
            printf("Corpus audio rings: %.1f MB pinned (20m + 10m)\n",
                   (double)audio_ring_bytes * 2 / 1e6);
        }

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
    if (audioBins_host) cudaFreeHost(audioBins_host);
    if (audioBins_host_10m) cudaFreeHost(audioBins_host_10m);
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

            // Corpus: async D2H of 1920 complex bins at 20m and 10m dial frequencies (sub=0 FFT).
            if (enable_corpus) {
                uint64_t audio_slot = ring_write_idx % (uint64_t)AUDIO_RING_SLOTS;
                cuda_check_error(cudaMemcpyAsync(
                    &audioBins_host[audio_slot * FT8_AUDIO_BINS],
                    &demodFT8_d[audio_ft8_bin],
                    FT8_AUDIO_BINS * sizeof(std::complex<float>),
                    cudaMemcpyDeviceToHost, stream));
                cuda_check_error(cudaMemcpyAsync(
                    &audioBins_host_10m[audio_slot * FT8_AUDIO_BINS],
                    &demodFT8_d[audio_ft8_bin_10m],
                    FT8_AUDIO_BINS * sizeof(std::complex<float>),
                    cudaMemcpyDeviceToHost, stream));
            }

            ring_write_idx++;

            // Commit all device ring writes before epoch path reads.
            cudaEventRecord(ring_ready, stream);

            // On trigger: snapshot the last FT8_CAPTURE_BLOCKS blocks into a decode slot.
            // Background thread does the ring→slot copy so doCopy never blocks.
            uint64_t trigger_second = (uint64_t)(seconds);
            bool gotime = (trigger == 14 &&
                           (seconds - trunc(seconds)) > 0.7 &&
                           trigger_second != last_trigger_second &&
                           ring_write_idx >= (uint64_t)FT8_CAPTURE_BLOCKS);
            if (gotime) {
                last_trigger_second = trigger_second;
                {
                    static uint64_t s_prev_ring = 0;
                    static double   s_prev_sec  = 0.0;
                    if (s_prev_sec > 0.0) {
                        double dt = seconds - s_prev_sec;
                        double rate = (double)(ring_write_idx - s_prev_ring) / dt;
                        printf("TIMING: epoch=%.3f mod15=%.3f ring=%llu blocks/s=%.2f (nominal 6.25)\n",
                               seconds, fmod(seconds, 15.0), (unsigned long long)ring_write_idx, rate);
                    }
                    s_prev_ring = ring_write_idx;
                    s_prev_sec  = seconds;
                }
                printf("Processing buffer %u\n", buffer_number);

                int decode_slot = buffer_number % BUFFERS;
                uint64_t snap_start = ring_write_idx - FT8_CAPTURE_BLOCKS;
                int next_buf = buffer_number + 1;
                int dev_snap_start = (int)(snap_start % RING_BLOCKS);

                cudaStreamWaitEvent(scan_stream,    ring_ready, 0);
                cudaStreamWaitEvent(transfer_stream, ring_ready, 0);

                // Launch epoch Costas scan on scan_stream.
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
                    FT8_GPU_CAND_MAX,
                    scan_stream);
                cudaEventRecord(scan_done, scan_stream);

                if (last_snapshot_thread.joinable())
                    last_snapshot_thread.join();

                double snap_seconds = seconds;
                last_snapshot_thread = std::thread([this, decode_slot, next_buf, snap_start, snap_seconds]() {
                    // Wait for Costas scan + soft symbols kernel (also ensures audio D2H done).
                    cudaError_t ev;
                    while ((ev = cudaEventQuery(scan_done)) == cudaErrorNotReady)
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    if (ev != cudaSuccess)
                        fprintf(stderr, "[EPOCH] CUDA event error: %s\n", cudaGetErrorString(ev));

                    // Snapshot audio rings immediately after sync (before doCopy can overwrite).
                    std::vector<std::complex<float>> audio_snap, audio_snap_10m;
                    if (enable_corpus) {
                        const size_t snap_bins = (size_t)FT8_CAPTURE_BLOCKS * FT8_AUDIO_BINS;
                        audio_snap.resize(snap_bins);
                        audio_snap_10m.resize(snap_bins);
                        for (int b = 0; b < FT8_CAPTURE_BLOCKS; ++b) {
                            uint64_t slot = (snap_start + b) % (uint64_t)AUDIO_RING_SLOTS;
                            memcpy(&audio_snap[(size_t)b * FT8_AUDIO_BINS],
                                   &audioBins_host[slot * FT8_AUDIO_BINS],
                                   FT8_AUDIO_BINS * sizeof(std::complex<float>));
                            memcpy(&audio_snap_10m[(size_t)b * FT8_AUDIO_BINS],
                                   &audioBins_host_10m[slot * FT8_AUDIO_BINS],
                                   FT8_AUDIO_BINS * sizeof(std::complex<float>));
                        }
                    }

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

                    // Corpus WAV write: IFFT each block's 1920 bins → real PCM → save_wav.
                    if (enable_corpus && !audio_snap.empty()) {
                        mkdir("ft8_corpus", 0755);
                        time_t t = (time_t)snap_seconds;
                        struct tm* tm_info = gmtime(&t);
                        int total_samples = FT8_CAPTURE_BLOCKS * FT8_AUDIO_BINS;

                        kiss_fft_cfg cfg = kiss_fft_alloc(FT8_AUDIO_BINS, 1, NULL, NULL);
                        kiss_fft_cpx in_buf[FT8_AUDIO_BINS], out_buf[FT8_AUDIO_BINS];

                        auto write_band_wav = [&](const std::vector<std::complex<float>>& snap,
                                                  const char* band_prefix) {
                            std::vector<float> pcm(total_samples);
                            for (int b = 0; b < FT8_CAPTURE_BLOCKS; ++b) {
                                const std::complex<float>* src = &snap[(size_t)b * FT8_AUDIO_BINS];
                                for (int k = 0; k < FT8_AUDIO_BINS; ++k) {
                                    in_buf[k].r = src[k].real();
                                    in_buf[k].i = src[k].imag();
                                }
                                kiss_fft(cfg, in_buf, out_buf);
                                for (int k = 0; k < FT8_AUDIO_BINS; ++k)
                                    pcm[(size_t)b * FT8_AUDIO_BINS + k] = out_buf[k].r / FT8_AUDIO_BINS;
                            }
                            float peak = 1e-10f;
                            for (float v : pcm) if (fabsf(v) > peak) peak = fabsf(v);
                            for (float& v : pcm) v /= peak;

                            char path[256];
                            snprintf(path, sizeof(path),
                                     "ft8_corpus/%s_%04d%02d%02d_%02d%02d%02d.wav",
                                     band_prefix,
                                     tm_info->tm_year + 1900, tm_info->tm_mon + 1, tm_info->tm_mday,
                                     tm_info->tm_hour, tm_info->tm_min, tm_info->tm_sec);
                            if (save_wav(pcm.data(), total_samples, audio_sample_rate, path) == 0)
                                printf("Corpus: saved %s (%d samples @ %d Hz)\n",
                                       path, total_samples, audio_sample_rate);
                            else
                                fprintf(stderr, "Corpus: save_wav failed for %s\n", path);
                        };

                        write_band_wav(audio_snap,     "20m");
                        write_band_wav(audio_snap_10m, "10m");
                        kiss_fft_free(cfg);
                    }
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
                std::cerr << "Error Falling Behind in FT8Cuda Copy, Dropping Data" << std::endl;
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
