#ifndef _GM_CUDA_HFCHANNELIZER_H_
#define _GM_CUDA_HFCHANNELIZER_H_

#include <atomic>
#include <string>
#include <thread>
#include <vector>
#include <cuda.h>
#include <cufft.h>
#include <zmq.hpp>
#include "gm/cuda/HostCuda.h"
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"

namespace gm {
namespace cuda {


class HFChannelizer : public Thread {
public:
    HFChannelizer(gm::buffer::BufferPosition<int16_t>* inP,
                  const std::string& ctrl_host = "127.0.0.1",
                  int ctrl_port = 8080);
    ~HFChannelizer();
    void run();
    void stop() {
        setRunning(false);
    }
    gm::buffer::BufferPosition<std::complex<float>>* getBuffer() {
        return &hfBufferPosition;
    }


private:
    cudaStream_t stream;
    cufftHandle plan;
    cufftHandle iplan;
    cufftHandle audio_plan;  // batched NUM_SINKS × AUDIO_BINS C2C IFFT
    double lastepoch;

    const static int BUFFERS = 16;

    int num_blocks;
    uint64_t buffer_number;

    gm::buffer::BufferPosition<std::complex<float>> hfBufferPosition;

    gm::cuda::device::HostCuda cuda_h;
    std::vector<size_t> inShape;
    int16_t* inData_d;
    int16_t* inData;
    float* fftInData_d;
    std::complex<float>* fftData_d;
    gm::buffer::BufferPosition<int16_t>* inPos;
    int doCopy(uint64_t now);
    std::complex<float>* channelData_d;
    std::complex<float>* demodData_d;
    std::complex<float>* audioBins_d;  // device staging: NUM_SINKS × AUDIO_BINS for batched IFFT
    std::vector<std::vector<uint32_t>> bins;
    uint32_t fft_length;
    uint32_t nTune;
    uint32_t nChannels;

    // Audio extraction: NUM_SINKS tunable 48 kHz channels from front FFT.
    // Each sink is independently tunable at runtime via REST API.
    // AUDIO_BINS bins per sink → AUDIO_VALID valid PCM samples/block at 48 kHz.
    static const int NUM_SINKS  = 4;
    static const int AUDIO_BINS = 480;  // bins from front FFT (100 Hz/bin, 48 kHz BW)
    static const int AUDIO_VALID = 240; // valid samples per block (overlap-save, 48 kHz)
    static const int AUDIO_RING  = 16;  // ring depth for D2H→worker handoff

    // Per-sink tuning state: start bin in wideband FFT (freq_hz / 100).
    // Written by controlWorker, read in doCopy — atomic for wait-free access.
    std::atomic<uint32_t> sink_bins[NUM_SINKS];
    std::string           sink_labels[NUM_SINKS];

    // Pinned host ring: AUDIO_RING × NUM_SINKS × AUDIO_BINS complex floats
    std::complex<float>* audio_pinned;

    std::atomic<uint64_t> audio_produce_idx{0};
    std::atomic<uint64_t> audio_consume_idx{0};
    std::thread audio_thread;
    void audioWorker();

    // REST control server + SSE event broadcast (cpp-httplib, runs in its own thread)
    std::string ctrl_host_;
    int         ctrl_port_;
    std::thread ctrl_thread;
    void controlWorker();


    zmq::context_t audio_zmq_ctx;
    zmq::socket_t* audio_sockets[NUM_SINKS];

    // Recording state — writes channelizer output blocks to a binary file.
    static const int RECORD_RING = 4;
    std::atomic<bool>            recording_{false};
    FILE*                        record_fp_{nullptr};
    std::complex<float>*         record_pinned_{nullptr};
    std::atomic<uint64_t>        record_produce_{0};
    std::atomic<uint64_t>        record_consume_{0};
    std::thread                  record_thread_;
    void recordWorker();

public:
    // Open filename and start writing channelizer output blocks.  Safe to call
    // after start().  Stop with stopRecording() or object destruction.
    void startRecording(const std::string& filename);
    void stopRecording();
};
}
}

#endif // _GM_CUDA_HFCHANNELIZER_H_
