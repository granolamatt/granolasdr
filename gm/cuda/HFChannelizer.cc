#include <unistd.h>
#include <iostream>
#include <fstream>
#include <cuda.h>
#include <complex>
#include <chrono>
#include <cstring>

#include "gm/cuda/HFChannelizer.h"
#include "gm/cuda/HostCuda.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/rx888/rx888.h"

#include "gm/hf/hf_bands.h"
#include "third_party/httplib.h"
#include "third_party/nlohmann_json.hpp"

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
};
static const int kNumFT8Presets = (int)(sizeof(kFT8Presets) / sizeof(kFT8Presets[0]));

HFChannelizer::HFChannelizer(gm::buffer::BufferPosition<int16_t>* inP,
                              const std::string& ctrl_host,
                              int ctrl_port) :
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
ctrl_host_(ctrl_host),
ctrl_port_(ctrl_port) {
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
        fft_length = 65536;

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

        ctrl_thread  = std::thread(&HFChannelizer::controlWorker, this);
        ctrl_thread.detach();

    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error in HFChannelizer constructor: " << e.what() << std::endl;
    }
}

HFChannelizer::~HFChannelizer() {
    if (audio_thread.joinable()) audio_thread.join();
    for (int i = 0; i < NUM_SINKS; i++) {
        delete audio_sockets[i];
        audio_sockets[i] = nullptr;
    }
    if (audio_pinned) cudaFreeHost(audio_pinned);
    if (audioBins_d) cudaFree(audioBins_d);
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
        }

        seq++;
        audio_consume_idx.fetch_add(1, std::memory_order_release);
    }
}

void HFChannelizer::controlWorker() {
    using json = nlohmann::json;
    httplib::Server svr;

    svr.Get("/api/status", [&](const httplib::Request&, httplib::Response& res) {
        json sinks = json::array();
        for (int i = 0; i < NUM_SINKS; i++) {
            uint32_t bin = sink_bins[i].load(std::memory_order_acquire);
            sinks.push_back({
                {"id", i},
                {"freq_hz", (uint64_t)bin * 100},
                {"label", sink_labels[i]}
            });
        }
        res.set_content(json{{"sinks", sinks}}.dump(), "application/json");
    });

    svr.Post("/api/tune", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid JSON\"}", "application/json");
            return;
        }
        if (!body.contains("sink") || !body.contains("freq_hz")) {
            res.status = 422;
            res.set_content("{\"error\":\"sink and freq_hz required\"}", "application/json");
            return;
        }
        int sink = body["sink"].get<int>();
        uint64_t freq_hz = body["freq_hz"].get<uint64_t>();
        if (sink < 0 || sink >= NUM_SINKS) {
            res.status = 422;
            res.set_content("{\"error\":\"sink out of range\"}", "application/json");
            return;
        }
        // Max freq: (700000 - AUDIO_BINS) bins × 100 Hz = 69,952,000 Hz
        if (freq_hz > 69952000ULL) {
            res.status = 422;
            res.set_content("{\"error\":\"freq_hz out of range (max 69952000)\"}", "application/json");
            return;
        }
        uint32_t new_bin = (uint32_t)(freq_hz / 100);
        std::string label = body.value("label", std::to_string(freq_hz / 1000) + " kHz");
        sink_bins[sink].store(new_bin, std::memory_order_release);
        sink_labels[sink] = label;
        printf("Tune: sink %d → %s (bin %u, %.3f MHz)\n",
               sink, label.c_str(), new_bin, (double)freq_hz / 1e6);
        res.set_content(json{{"ok",true},{"sink",sink},{"freq_hz",freq_hz}}.dump(),
                        "application/json");
    });

    svr.Post("/api/preset", [&](const httplib::Request& req, httplib::Response& res) {
        json body;
        try { body = json::parse(req.body); }
        catch (...) {
            res.status = 400;
            res.set_content("{\"error\":\"invalid JSON\"}", "application/json");
            return;
        }
        if (!body.contains("sink") || !body.contains("preset")) {
            res.status = 422;
            res.set_content("{\"error\":\"sink and preset required\"}", "application/json");
            return;
        }
        int sink = body["sink"].get<int>();
        std::string preset = body["preset"].get<std::string>();
        if (sink < 0 || sink >= NUM_SINKS) {
            res.status = 422;
            res.set_content("{\"error\":\"sink out of range\"}", "application/json");
            return;
        }
        uint32_t freq_hz = 0;
        bool found = false;
        for (int i = 0; i < kNumFT8Presets; i++) {
            if (preset == kFT8Presets[i].name) {
                freq_hz = kFT8Presets[i].freq_hz;
                found = true;
                break;
            }
        }
        if (!found) {
            res.status = 422;
            res.set_content("{\"error\":\"unknown preset\"}", "application/json");
            return;
        }
        uint32_t new_bin = freq_hz / 100;
        std::string label = preset + " FT8";
        sink_bins[sink].store(new_bin, std::memory_order_release);
        sink_labels[sink] = label;
        printf("Preset: sink %d → %s (%.3f MHz)\n", sink, label.c_str(), (double)freq_hz / 1e6);
        res.set_content(json{{"ok",true},{"sink",sink},{"freq_hz",(uint64_t)freq_hz},{"label",label}}.dump(),
                        "application/json");
    });

    svr.Get("/api/presets", [](const httplib::Request&, httplib::Response& res) {
        using json = nlohmann::json;
        json presets = json::array();
        for (int i = 0; i < kNumFT8Presets; i++)
            presets.push_back({{"name", kFT8Presets[i].name}, {"freq_hz", kFT8Presets[i].freq_hz}});
        res.set_content(json{{"presets", presets}}.dump(), "application/json");
    });

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        std::ifstream f("control/index.html");
        if (!f.is_open()) {
            res.set_content("<h1>granolasdr control</h1><p>control/index.html not found</p>", "text/html");
            return;
        }
        std::string html((std::istreambuf_iterator<char>(f)),
                          std::istreambuf_iterator<char>());
        res.set_content(html, "text/html");
    });

    printf("Control server: http://%s:%d/\n", ctrl_host_.c_str(), ctrl_port_);
    svr.listen(ctrl_host_.c_str(), ctrl_port_);
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
    // Start audio worker here so isRunning() is guaranteed true when it checks.
    audio_thread = std::thread(&HFChannelizer::audioWorker, this);

    uint64_t now = inPos->getNow(1) + 1;

    while(isRunning()) {
        uint64_t next = inPos->getPosition(now+1, 1);

        while(now < next) {
            uint64_t length = next - now;
            if (length > 4) {
                std::cerr << "Error Falling Behind in Channelizer " << length << std::endl;
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
