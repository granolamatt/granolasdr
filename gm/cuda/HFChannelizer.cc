#include <unistd.h>
#include <iostream>
#include <cuda.h>
#include <complex>
#include <chrono>
#include <cstring>
#include <immintrin.h>

#include "gm/cuda/HFChannelizer.h"
#include "gm/cuda/HostCuda.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/rx888/rx888.h"

#include "ft8_lib/fft/kiss_fft.h"
#include "gm/hf/hf_bands.h"

namespace gm {
namespace cuda {

HFChannelizer::HFChannelizer(gm::buffer::BufferPosition<int16_t>* inP) :
inPos(inP),
inData_d(NULL),
fftInData_d(NULL),
fftData_d(NULL),
demodData_d(NULL),
channelData_d(NULL),
num_blocks(0),
buffer_number(0),
audio_pinned(NULL),
audio_zmq_ctx(1) {
    inShape = inPos->getShape();
    inData = (int16_t*)inPos->getBuffer();

    for (int i = 0; i < AUDIO_BANDS; i++)
        audio_sockets[i] = nullptr;

    try {
        cuda_check_error(cudaMalloc((void**)&inData_d, inPos->getByteSize()));
        cuda_check_error(cudaMalloc((void**)&fftInData_d,
            sizeof(float)*(gm::rx888::rx888::NLARGE*gm::rx888::rx888::BUFFERS + gm::rx888::rx888::NLARGE)));
        // R2C of 1,400,000-pt FFT → 700,001 complex output bins; pad to 2*NLARGE+1024.
        cuda_check_error(cudaMalloc((void**)&fftData_d,
            sizeof(std::complex<float>)*(2*gm::rx888::rx888::NLARGE + 1024)));

        cuda_check_error(cudaSetDevice(0));
        int lo, hi;
        cudaDeviceGetStreamPriorityRange(&lo, &hi);
        cuda_check_error(cudaStreamCreateWithPriority(&stream, cudaStreamDefault, hi));
        cufftResult fftRes = cufftPlan1d(&plan, gm::rx888::rx888::NLARGE*2, CUFFT_R2C, 1);
        if (fftRes) printf("HFChannelizer: cufftPlan1d (R2C) failed\n");
        fftRes = cufftSetStream(plan, stream);
        if (fftRes) printf("HFChannelizer: cufftSetStream (R2C) failed\n");

        cuda_h = gm::cuda::device::HostCuda(stream);
        bins.resize(kNumHFBands);
        for (int i = 0; i < kNumHFBands; ++i)
            bins[i] = {kHFBands[i].wb_start, kHFBands[i].wb_end, kHFBands[i].bw};
        fft_length = 65536;

        cuda_check_error(cudaMalloc((void**)&demodData_d, BUFFERS * fft_length / 2 * sizeof(std::complex<float>) + 1024));

        hfBufferPosition.setBuffer(demodData_d, {BUFFERS, fft_length / 2});

        cuda_check_error(cudaMalloc((void**)&channelData_d, fft_length*sizeof(std::complex<float>) + 1024));
        printf("Total fft length is %u\n", fft_length);

        fftRes = cufftPlan1d(&iplan, fft_length, CUFFT_C2C, 1);
        if (fftRes) printf("HFChannelizer: cufftPlan1d (C2C) failed\n");
        fftRes = cufftSetStream(iplan, stream);
        if (fftRes) printf("HFChannelizer: cufftSetStream (C2C) failed\n");

        // Audio extraction: pinned ring buffer for 10-band FFT bin snapshots.
        const size_t audio_ring_bytes =
            (size_t)AUDIO_RING * AUDIO_BANDS * AUDIO_BINS * sizeof(std::complex<float>);
        cuda_check_error(cudaHostAlloc((void**)&audio_pinned,
                                       audio_ring_bytes, cudaHostAllocDefault));
        printf("Audio ring: %.1f MB pinned (%d slots × %d bands × %d bins)\n",
               (double)audio_ring_bytes / 1e6, AUDIO_RING, AUDIO_BANDS, AUDIO_BINS);

        // ZMQ PUB sockets for audio: one per band, ports 5581-5590.
        for (int i = 0; i < AUDIO_BANDS; i++) {
            audio_sockets[i] = new zmq::socket_t(audio_zmq_ctx, ZMQ_PUB);
            char ep[64];
            snprintf(ep, sizeof(ep), "tcp://*:%d", 5581 + i);
            audio_sockets[i]->bind(ep);
            printf("Audio ZMQ: band %s → port %d\n", kHFBands[i].name, 5581 + i);
        }

        audio_thread = std::thread(&HFChannelizer::audioWorker, this);

    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error in HFChannelizer constructor: " << e.what() << std::endl;
    }
}

HFChannelizer::~HFChannelizer() {
    if (audio_thread.joinable()) audio_thread.join();
    for (int i = 0; i < AUDIO_BANDS; i++) {
        delete audio_sockets[i];
        audio_sockets[i] = nullptr;
    }
    if (audio_pinned) cudaFreeHost(audio_pinned);
    if (inData_d) cudaFree(inData_d);
    if (fftInData_d) cudaFree(fftInData_d);
    if (fftData_d) cudaFree(fftData_d);
    if (channelData_d) cudaFree(channelData_d);
    if (demodData_d) cudaFree(demodData_d);
    cufftDestroy(plan);
    cufftDestroy(iplan);
    cudaStreamDestroy(stream);
}

int HFChannelizer::doCopy(uint64_t now) {
    try {
        static gm::buffer::BufferPosition<std::complex<float>>* bpos = &hfBufferPosition;
        size_t length = inShape[1];
        int in_position = (now % inShape[0]) * length;

        // Host→device for new rx888 block.
        cuda_check_error(cudaMemcpyAsync(&inData_d[in_position], &inData[in_position],
            length*inPos->getElementSize(), cudaMemcpyHostToDevice, stream));
        cuda_h.copyKernel(&fftInData_d[in_position + length], &inData_d[in_position], length);

        if (!(now % gm::rx888::rx888::BUFFERS)) {
            cuda_check_error(cudaMemcpyAsync(fftInData_d,
                            &fftInData_d[gm::rx888::rx888::BUFFERS*length],
                            length*sizeof(float), cudaMemcpyDeviceToDevice, stream));
        }

        // Front-end 1,400,000-pt R2C FFT → fftData_d (100 Hz/bin).
        cufftResult_t rval = cufftExecR2C(plan, &fftInData_d[in_position], (cufftComplex*)fftData_d);
        if (rval) {
            printf("Error in R2C FFT\n");
            return 0;
        }

        // Audio extraction: async D2H of AUDIO_BINS complex bins per band.
        // Enqueued in main stream immediately after the R2C FFT completes.
        // The callback at the end of doCopy signals the audio worker once done.
        {
            uint64_t produce = audio_produce_idx.load(std::memory_order_relaxed);
            uint64_t consume = audio_consume_idx.load(std::memory_order_acquire);
            if (produce - consume < (uint64_t)AUDIO_RING) {
                uint64_t slot = produce % (uint64_t)AUDIO_RING;
                std::complex<float>* dst = audio_pinned + slot * (size_t)AUDIO_BANDS * AUDIO_BINS;
                for (int i = 0; i < AUDIO_BANDS; i++) {
                    cuda_check_error(cudaMemcpyAsync(
                        dst + i * AUDIO_BINS,
                        &fftData_d[kHFBands[i].ft8_dial_bin],
                        AUDIO_BINS * sizeof(std::complex<float>),
                        cudaMemcpyDeviceToHost, stream));
                }
                // Second callback signals audio worker after D2H complete.
                static std::atomic<uint64_t>* s_audio_produce = &audio_produce_idx;
                cudaStreamAddCallback(stream,
                    [](cudaStream_t, cudaError_t, void*) {
                        s_audio_produce->fetch_add(1, std::memory_order_release);
                    }, nullptr, 0);
            }
        }

        // Composite assembly: pack all band bins into channelData_d[0..total_bw-1].
        uint32_t offset = 0;
        cuda_check_error(cudaMemsetAsync(channelData_d, 0, fft_length*sizeof(std::complex<float>), stream));
        for (const std::vector<uint32_t>& b : bins) {
            cuda_check_error(cudaMemcpyAsync(&channelData_d[offset],
                &fftData_d[b[0]],
                b[2]*sizeof(std::complex<float>), cudaMemcpyDeviceToDevice, stream));
            offset += b[2];
        }

        // 65,536-pt composite IFFT → 32,768 valid samples at 6.5536 MS/s.
        rval = cufftExecC2C(iplan, (cufftComplex*)&channelData_d[0],
            (cufftComplex*)&channelData_d[0], CUFFT_INVERSE);
        if (rval) {
            printf("Error in composite IFFT\n");
            return 0;
        }

        cuda_check_error(cudaMemcpyAsync(&demodData_d[(buffer_number % BUFFERS) * fft_length / 2],
            &channelData_d[fft_length/4],
            fft_length / 2 * sizeof(std::complex<float>), cudaMemcpyDeviceToDevice, stream));

        cudaStreamAddCallback(stream,
        [](cudaStream_t mstream, cudaError_t status, void *data) {
            uint64_t* pos = (uint64_t*)data;
            bpos->setPosition(*pos, 1);
            *pos += 1;
        }
        , (void *)&buffer_number, 0);

        return 1;
    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error after cudaCopy: " << e.what() << std::endl;
    }
    return 0;
}

void HFChannelizer::audioWorker() {
    kiss_fft_cfg cfg = kiss_fft_alloc(AUDIO_BINS, 1, NULL, NULL);
    std::vector<kiss_fft_cpx> in_buf(AUDIO_BINS), out_buf(AUDIO_BINS);

    // Frame header: [band_id: uint32][seq: uint32][samples: float32 × AUDIO_VALID]
    const size_t frame_bytes = 2 * sizeof(uint32_t) + AUDIO_VALID * sizeof(float);
    std::vector<uint8_t> frame(frame_bytes);
    float* pcm = reinterpret_cast<float*>(frame.data() + 2 * sizeof(uint32_t));

    uint64_t seq = 0;

    while (isRunning()) {
        uint64_t produce = audio_produce_idx.load(std::memory_order_acquire);
        uint64_t consume = audio_consume_idx.load(std::memory_order_relaxed);
        if (produce <= consume) {
            _mm_pause();
            continue;
        }

        uint64_t slot = consume % (uint64_t)AUDIO_RING;
        const std::complex<float>* src = audio_pinned + slot * (size_t)AUDIO_BANDS * AUDIO_BINS;

        for (int band = 0; band < AUDIO_BANDS; band++) {
            const std::complex<float>* bins_in = src + band * AUDIO_BINS;
            for (int k = 0; k < AUDIO_BINS; k++) {
                in_buf[k].r = bins_in[k].real();
                in_buf[k].i = bins_in[k].imag();
            }
            kiss_fft(cfg, in_buf.data(), out_buf.data());

            // Last AUDIO_VALID samples are valid (overlap-save second half).
            const float norm = 1.0f / AUDIO_BINS;
            for (int k = 0; k < AUDIO_VALID; k++)
                pcm[k] = out_buf[AUDIO_BINS - AUDIO_VALID + k].r * norm;

            memcpy(frame.data(), &band, sizeof(uint32_t));
            uint32_t seq32 = (uint32_t)seq;
            memcpy(frame.data() + sizeof(uint32_t), &seq32, sizeof(uint32_t));

            zmq::message_t msg(frame_bytes);
            memcpy(msg.data(), frame.data(), frame_bytes);
            audio_sockets[band]->send(msg, zmq::send_flags::dontwait);
        }

        seq++;
        audio_consume_idx.fetch_add(1, std::memory_order_release);
    }

    kiss_fft_free(cfg);
}

void HFChannelizer::run() {
    uint64_t now = inPos->getNow(1) + 1;
    uint64_t call_count = 0;
    double total_wait_ms = 0, total_copy_ms = 0, max_copy_ms = 0;

    while(isRunning()) {
        auto t0 = std::chrono::steady_clock::now();
        uint64_t next = inPos->getPosition(now+1, 1);
        auto t1 = std::chrono::steady_clock::now();
        total_wait_ms += std::chrono::duration<double,std::milli>(t1-t0).count();

        while(now < next) {
            uint64_t length = next - now;
            if (length > 4) {
                std::cerr << "Error Falling Behind in Channelizer, Dropping Data" << std::endl;
                now = next;
                break;
            }
            auto tc0 = std::chrono::steady_clock::now();
            int numCopied = doCopy(now);
            auto tc1 = std::chrono::steady_clock::now();
            if (!numCopied) exit(-200);
            double dt = std::chrono::duration<double,std::milli>(tc1-tc0).count();
            total_copy_ms += dt;
            if (dt > max_copy_ms) max_copy_ms = dt;
            call_count++;
            if (call_count % 200 == 0) {
                printf("HFChannelizer: avg_wait=%.2fms avg_copy=%.2fms max_copy=%.2fms (budget=5ms)\n",
                       total_wait_ms / 200, total_copy_ms / 200, max_copy_ms);
                total_wait_ms = total_copy_ms = max_copy_ms = 0;
            }
	        now += numCopied;
        }
    }
}

}
}
