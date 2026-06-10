#include <unistd.h>
#include <iostream>
#include <fstream>
#include <cuda.h>
#include <complex>
#include <chrono>
#include <cstring>

#include "gm/cuda/HFChannelizer.h"
#include "gm/cuda/HostCuda.h"
#include "gm/cuda/SpectrumNorm.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/rx888/rx888.h"

#include "gm/hf/hf_bands.h"
#include "third_party/nlohmann_json.hpp"
#include "wsdict.h"
#include "tci_server.h"
#include <cmath>

namespace gm {
namespace cuda {

static const struct { const char* name; uint32_t freq_hz; } kFT8Presets[] = {
    {"160m",  1840000},
    {"80m",   3573000},
    {"60m",   5357000},
    {"40m",   7074000},
    {"30m",  10136000},
    {"20m",  14074000},
    {"17m",  18100000},
    {"15m",  21074000},
    {"12m",  24915000},
    {"10m",  28074000},
    {"6m",  50313000},
};
static const int kNumFT8Presets = (int)(sizeof(kFT8Presets) / sizeof(kFT8Presets[0]));

HFChannelizer::HFChannelizer(gm::buffer::BufferPosition<int16_t>* inP,
                              int wsdict_port) :
inPos(inP),
inData_d(NULL),
fftInData_d(NULL),
fftData_d(NULL),
demodData_d(NULL),
channelData_d(NULL),
audioBins_d(NULL),
num_blocks(0),
buffer_number(0),
audio_pinned(NULL),
audio_zmq_ctx(1),
wsdict_port_(wsdict_port) {
    inShape = inPos->getShape();
    inData = (int16_t*)inPos->getBuffer();

    // Default sinks: 20m, 40m, 80m, 10m FT8 dial frequencies
    sink_bins[0].store(140740, std::memory_order_relaxed);
    sink_bins[1].store(70740,  std::memory_order_relaxed);
    sink_bins[2].store(35730,  std::memory_order_relaxed);
    sink_bins[3].store(280740, std::memory_order_relaxed);
    sink_labels[0] = "20m FT8";
    sink_labels[1] = "40m FT8";
    sink_labels[2] = "80m FT8";
    sink_labels[3] = "10m FT8";

    for (int i = 0; i < NUM_SINKS; i++)
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
        fft_length = 4096;

        // Spectral normalization setup: allocate per-composite-bin gain buffers.
        norm_total_bins_ = 0;
        for (const auto& b : bins) norm_total_bins_ += (int)b[2];
        norm_gains_h_.assign(norm_total_bins_, 1.0f);
        norm_snap_h_.resize(norm_total_bins_);
        norm_ema_h_.assign(norm_total_bins_, 0.0f);
        norm_logmag_h_.resize(norm_total_bins_);
        cuda_check_error(cudaMalloc((void**)&norm_gains_d_,
                                    norm_total_bins_ * sizeof(float)));
        cuda_check_error(cudaMemcpy(norm_gains_d_, norm_gains_h_.data(),
                                    norm_total_bins_ * sizeof(float),
                                    cudaMemcpyHostToDevice));

        cuda_check_error(cudaMalloc((void**)&demodData_d, BUFFERS * fft_length / 2 * sizeof(std::complex<float>) + 1024));

        hfBufferPosition.setBuffer(demodData_d, {BUFFERS, fft_length / 2});

        cuda_check_error(cudaMalloc((void**)&channelData_d, fft_length*sizeof(std::complex<float>) + 1024));
        printf("Total fft length is %u\n", fft_length);

        fftRes = cufftPlan1d(&iplan, fft_length, CUFFT_C2C, 1);
        if (fftRes) printf("HFChannelizer: cufftPlan1d (C2C) failed\n");
        fftRes = cufftSetStream(iplan, stream);
        if (fftRes) printf("HFChannelizer: cufftSetStream (C2C) failed\n");

        // Batched audio IFFT: NUM_SINKS × AUDIO_BINS C2C on GPU.
        cuda_check_error(cudaMalloc((void**)&audioBins_d,
            (size_t)NUM_SINKS * AUDIO_BINS * sizeof(std::complex<float>)));
        {
            int audio_n[1] = {AUDIO_BINS};
            fftRes = cufftPlanMany(&audio_plan, 1, audio_n,
                nullptr, 1, AUDIO_BINS,
                nullptr, 1, AUDIO_BINS,
                CUFFT_C2C, NUM_SINKS);
            if (fftRes) printf("HFChannelizer: cufftPlanMany (audio IFFT) failed\n");
            fftRes = cufftSetStream(audio_plan, stream);
            if (fftRes) printf("HFChannelizer: cufftSetStream (audio) failed\n");
        }

        // Pinned ring: AUDIO_RING × NUM_SINKS × AUDIO_BINS complex floats for D2H→worker handoff.
        const size_t audio_ring_bytes =
            (size_t)AUDIO_RING * NUM_SINKS * AUDIO_BINS * sizeof(std::complex<float>);
        cuda_check_error(cudaHostAlloc((void**)&audio_pinned,
                                       audio_ring_bytes, cudaHostAllocDefault));
        printf("Audio ring: %.1f MB pinned (%d slots × %d sinks × %d bins)\n",
               (double)audio_ring_bytes / 1e6, AUDIO_RING, NUM_SINKS, AUDIO_BINS);

        for (int i = 0; i < NUM_SINKS; i++) {
            audio_sockets[i] = new zmq::socket_t(audio_zmq_ctx, ZMQ_PUB);
            char ep[64];
            snprintf(ep, sizeof(ep), "tcp://*:%d", 5581 + i);
            audio_sockets[i]->bind(ep);
            printf("Audio ZMQ: sink %d (%s) → port %d\n", i, sink_labels[i].c_str(), 5581 + i);
        }

        cmd_thread_ = std::thread(&HFChannelizer::cmdWorker, this);
        cmd_thread_.detach();

    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error in HFChannelizer constructor: " << e.what() << std::endl;
    }
}

HFChannelizer::~HFChannelizer() {
    if (tci_vfo_thread_.joinable()) tci_vfo_thread_.join();
    if (audio_thread.joinable()) audio_thread.join();
    for (int i = 0; i < NUM_SINKS; i++) {
        delete audio_sockets[i];
        audio_sockets[i] = nullptr;
    }
    if (audio_pinned) cudaFreeHost(audio_pinned);
    if (audioBins_d) cudaFree(audioBins_d);
    if (norm_gains_d_) cudaFree(norm_gains_d_);
    if (inData_d) cudaFree(inData_d);
    if (fftInData_d) cudaFree(fftInData_d);
    if (fftData_d) cudaFree(fftData_d);
    if (channelData_d) cudaFree(channelData_d);
    if (demodData_d) cudaFree(demodData_d);
    cufftDestroy(plan);
    cufftDestroy(iplan);
    cufftDestroy(audio_plan);
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

        // Audio extraction: D2D gather bins → audioBins_d, batched GPU IFFT, D2H to pinned ring.
        {
            uint64_t produce = audio_produce_idx.load(std::memory_order_relaxed);
            uint64_t consume = audio_consume_idx.load(std::memory_order_acquire);
            if (produce - consume < (uint64_t)AUDIO_RING) {
                uint64_t slot = produce % (uint64_t)AUDIO_RING;
                std::complex<float>* dst = audio_pinned + slot * (size_t)NUM_SINKS * AUDIO_BINS;
                for (int i = 0; i < NUM_SINKS; i++) {
                    uint32_t start_bin = sink_bins[i].load(std::memory_order_acquire);
                    cuda_check_error(cudaMemcpyAsync(
                        audioBins_d + i * AUDIO_BINS,
                        &fftData_d[start_bin],
                        AUDIO_BINS * sizeof(std::complex<float>),
                        cudaMemcpyDeviceToDevice, stream));
                }
                cufftResult_t arval = cufftExecC2C(audio_plan,
                    (cufftComplex*)audioBins_d, (cufftComplex*)audioBins_d, CUFFT_INVERSE);
                if (arval) printf("Error in audio IFFT\n");
                cuda_check_error(cudaMemcpyAsync(
                    dst, audioBins_d,
                    (size_t)NUM_SINKS * AUDIO_BINS * sizeof(std::complex<float>),
                    cudaMemcpyDeviceToHost, stream));
                static std::atomic<uint64_t>* s_audio_produce = &audio_produce_idx;
                cudaStreamAddCallback(stream,
                    [](cudaStream_t, cudaError_t, void*) {
                        s_audio_produce->fetch_add(1, std::memory_order_release);
                    }, nullptr, 0);
            }
        }

        // Noise-floor normalization: every NORM_INTERVAL frames, snapshot the wideband
        // FFT magnitudes for each band, update a slow EMA (α≈0.03 → ~5-min time
        // constant at 10 s/update), fit a per-band polynomial, and recompute gains.
        // Gains are applied to channelData_d after composite assembly, before the IFFT,
        // so both the waterfall and the decoders receive a spectrally flattened signal.
        if (++norm_frame_ % NORM_INTERVAL == 0) {
            cudaStreamSynchronize(stream);  // wait for R2C FFT to finish

            // D2H: collect complex band bins from the wideband FFT output.
            int snap_off = 0;
            for (const auto& b : bins) {
                int bw = (int)b[2];
                cudaMemcpy(&norm_snap_h_[snap_off], &fftData_d[b[0]],
                           bw * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);
                snap_off += bw;
            }

            // Asymmetric EMA: track noise floor downward quickly (signals drop away),
            // drift upward very slowly (genuine band noise increase takes ~30 min).
            // Warm-up phase uses faster rates for both directions to seed the estimate.
            bool warm = (norm_update_count_ < 5);
            float a_down = warm ? 0.30f : 0.10f;
            float a_up   = warm ? 0.10f : 0.005f;
            for (int i = 0; i < norm_total_bins_; i++) {
                float re = norm_snap_h_[i].real(), im = norm_snap_h_[i].imag();
                float mag = std::sqrt(re*re + im*im);
                if (norm_update_count_ == 0) {
                    norm_ema_h_[i] = mag;  // cold-start: seed directly
                } else {
                    float a = (mag < norm_ema_h_[i]) ? a_down : a_up;
                    norm_ema_h_[i] = (1.0f - a) * norm_ema_h_[i] + a * mag;
                }
            }
            norm_update_count_++;

            // Fit polynomial to log10(EMA) per band; compute equalization gains.
            int comp_off = 0;
            for (const auto& b : bins) {
                int bw = (int)b[2];
                for (int i = 0; i < bw; i++)
                    norm_logmag_h_[comp_off + i] = std::log10(norm_ema_h_[comp_off + i] + 1e-30f);
                norm_fit_gains(norm_logmag_h_.data() + comp_off, bw, NORM_POLY_DEG,
                               norm_gains_h_.data() + comp_off);
                comp_off += bw;
            }

            // Cross-band leveling: compute the geometric mean of all band-center EMA
            // magnitudes and add a per-band scalar so every band arrives at the same
            // brightness.  Without this, active bands (more signals → higher EMA) remain
            // brighter than quiet bands even after intra-band flattening.
            {
                double global_log = 0.0;
                int off = 0;
                for (const auto& b : bins) {
                    int bw = (int)b[2];
                    global_log += std::log10(norm_ema_h_[off + bw / 2] + 1e-30f);
                    off += bw;
                }
                global_log /= (double)bins.size();

                off = 0;
                for (const auto& b : bins) {
                    int bw = (int)b[2];
                    double band_log = std::log10(norm_ema_h_[off + bw / 2] + 1e-30f);
                    float scale = (float)std::pow(10.0, global_log - band_log);
                    for (int i = 0; i < bw; i++)
                        norm_gains_h_[off + i] *= scale;
                    off += bw;
                }
            }

            // H2D the updated gains (async — ordered before apply_gains_kernel below).
            cudaMemcpyAsync(norm_gains_d_, norm_gains_h_.data(),
                            norm_total_bins_ * sizeof(float), cudaMemcpyHostToDevice, stream);
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

        // Apply per-bin equalization gains to the packed composite (before IFFT).
        norm_apply_gains((cufftComplex*)channelData_d, norm_gains_d_,
                         norm_total_bins_, stream);

        // 4096-pt composite C2C IFFT → 2048 valid samples (center half) at 409.6 kHz.
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
    // Frame: [sink_id: u32][seq: u32][AUDIO_VALID × f32] = 968 bytes
    const size_t frame_bytes = 2 * sizeof(uint32_t) + AUDIO_VALID * sizeof(float);
    std::vector<uint8_t> frame(frame_bytes);
    float* pcm = reinterpret_cast<float*>(frame.data() + 2 * sizeof(uint32_t));

    uint64_t seq = 0;

    while (isRunning()) {
        uint64_t produce = audio_produce_idx.load(std::memory_order_acquire);
        uint64_t consume = audio_consume_idx.load(std::memory_order_relaxed);
        if (produce <= consume) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }

        uint64_t slot = consume % (uint64_t)AUDIO_RING;
        const std::complex<float>* src = audio_pinned + slot * (size_t)NUM_SINKS * AUDIO_BINS;

        for (int sink = 0; sink < NUM_SINKS; sink++) {
            const std::complex<float>* ifft_out = src + sink * AUDIO_BINS;
            // Last AUDIO_VALID samples are valid (overlap-save second half).
            const float norm = 1.0f / AUDIO_BINS;
            for (int k = 0; k < AUDIO_VALID; k++)
                pcm[k] = ifft_out[AUDIO_BINS - AUDIO_VALID + k].real() * norm;

            uint32_t sink32 = (uint32_t)sink;
            memcpy(frame.data(), &sink32, sizeof(uint32_t));
            uint32_t seq32 = (uint32_t)seq;
            memcpy(frame.data() + sizeof(uint32_t), &seq32, sizeof(uint32_t));

            zmq::message_t msg(frame_bytes);
            memcpy(msg.data(), frame.data(), frame_bytes);
            audio_sockets[sink]->send(msg, zmq::send_flags::dontwait);

            tci_push_audio((uint32_t)sink, pcm, AUDIO_VALID);
        }

        seq++;

        // S-meter: every 100 frames (~1 Hz at 10 ms/frame).
        // Source: norm_ema_h_[composite_bin] — wideband RF noise floor EMA,
        // independent of modulation depth (D11). Read without lock; benign
        // data race on float (x86 aligned reads are atomic, value changes slowly).
        if (seq % 100 == 0) {
            for (int s = 0; s < NUM_SINKS; s++) {
                uint32_t wb = sink_bins[s].load(std::memory_order_relaxed);
                float ema_val = 0.0f;
                int cum = 0;
                for (int b = 0; b < kNumHFBands; b++) {
                    if (wb >= kHFBands[b].wb_start && wb < kHFBands[b].wb_end) {
                        int idx = cum + (int)(wb - kHFBands[b].wb_start);
                        if (idx >= 0 && idx < (int)norm_ema_h_.size())
                            ema_val = norm_ema_h_[idx];
                        break;
                    }
                    cum += (int)kHFBands[b].bw;
                }
                float dbfs = (ema_val > 1e-10f) ? 20.0f * log10f(ema_val) : -100.0f;
                tci_push_smeter((uint32_t)s, dbfs);
            }
        }

        audio_consume_idx.fetch_add(1, std::memory_order_release);
    }
}

void HFChannelizer::cmdWorker() {
    using json = nlohmann::json;

    // Publisher connection: writes granolasdr:audio:status:N keys so the browser
    // can see current sink tuning state immediately on connect.
    WsDictClient pub_ws("127.0.0.1", wsdict_port_);
    auto publishSink = [&](int i) {
        uint32_t bin = sink_bins[i].load(std::memory_order_acquire);
        pub_ws.set("granolasdr:audio:status:" + std::to_string(i),
                   json{{"id", i}, {"freq_hz", (uint64_t)bin * 100}, {"label", sink_labels[i]}});
    };
    for (int i = 0; i < NUM_SINKS; i++) publishSink(i);

    // Subscriber connection: receives tune commands sent by the browser as
    // {"type":"set","key":"granolasdr:audio:cmd","value":{sink, freq_hz, label}}.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    WsDictClient sub_ws("127.0.0.1", wsdict_port_);
    sub_ws.subscribe({"granolasdr:audio:cmd"});

    while (isRunning()) {
        try {
            json msg = sub_ws.recv_message();
            if (msg.value("type", "") != "update") continue;

            json val = msg["value"];
            if (!val.contains("sink") || !val.contains("freq_hz")) continue;

            int sink = val["sink"].get<int>();
            uint64_t freq_hz = val["freq_hz"].get<uint64_t>();
            if (sink < 0 || sink >= NUM_SINKS) continue;
            if (freq_hz > 69952000ULL) continue;

            uint32_t new_bin = (uint32_t)(freq_hz / 100);
            std::string label = val.value("label", std::to_string(freq_hz / 1000) + " kHz");
            sink_bins[sink].store(new_bin, std::memory_order_release);
            sink_labels[sink] = label;
            printf("Tune: sink %d → %s (bin %u, %.3f MHz)\n",
                   sink, label.c_str(), new_bin, (double)freq_hz / 1e6);

            pub_ws.set("granolasdr:audio:status:" + std::to_string(sink),
                       json{{"id", sink}, {"freq_hz", freq_hz}, {"label", label}});
        } catch (...) {
            break;
        }
    }
}

void HFChannelizer::tciVfoWorker() {
    // Drain TCI VFO commands and apply them to sink_bins[].
    // Does NOT write sink_labels[] — non-atomic std::string, avoids data race (D3).
    while (isRunning()) {
        uint32_t trx;
        uint64_t freq_hz;
        while (tci_poll_vfo(&trx, &freq_hz)) {
            if ((int)trx < NUM_SINKS) {
                uint32_t new_bin = (uint32_t)(freq_hz / 100);
                sink_bins[trx].store(new_bin, std::memory_order_release);
                printf("[TCI] VFO: sink %u → %.3f MHz (bin %u)\n",
                       trx, (double)freq_hz / 1e6, new_bin);
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

gm::buffer::BufferFileParams HFChannelizer::getBufferFileParams() const {
    gm::buffer::BufferFileParams p;
    p.block_samples     = fft_length / 2;
    p.block_interval_ns = (uint64_t)(
        (double)gm::rx888::rx888::NLARGE / gm::rx888::rx888::rx_samplerate * 1e9);
    p.sample_rate_hz    = (uint32_t)(
        ((double)(fft_length / 2)) * 1e9 / (double)p.block_interval_ns);
    p.rx_sample_rate    = gm::rx888::rx888::rx_samplerate;
    return p;
}

void HFChannelizer::run() {
    // Start workers here so isRunning() is guaranteed true when they first check.
    audio_thread    = std::thread(&HFChannelizer::audioWorker,    this);
    tci_vfo_thread_ = std::thread(&HFChannelizer::tciVfoWorker,   this);

    uint64_t now = inPos->getNow(1) + 1;

    while(isRunning()) {
        uint64_t next = inPos->getPosition(now+1, 1);

        while(now < next) {
            uint64_t length = next - now;
            if (length > 4) {
                std::cerr << "Error Falling Behind in Channelizer, Dropping Data" << std::endl;
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
