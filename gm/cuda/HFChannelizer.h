#ifndef _GM_CUDA_HFCHANNELIZER_H_
#define _GM_CUDA_HFCHANNELIZER_H_

#include <atomic>
#include <thread>
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
    HFChannelizer(gm::buffer::BufferPosition<int16_t>* inP);
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
    std::vector<std::vector<uint32_t>> bins;
    uint32_t fft_length;
    uint32_t nTune;
    uint32_t nChannels;

    // Audio extraction: 120 bins × 10 bands from front FFT → 12 kHz ZMQ audio.
    // Each block: async D2H 120 bins/band → worker thread IFFTs + ZMQ publishes.
    static const int AUDIO_BANDS = 10;
    static const int AUDIO_BINS  = 120;  // bins from front FFT (100 Hz/bin, 12 kHz BW)
    static const int AUDIO_VALID = 60;   // valid samples per block (overlap-save, 12 kHz)
    static const int AUDIO_RING  = 16;   // ring depth for D2H→worker handoff

    // Pinned host ring: AUDIO_RING × AUDIO_BANDS × AUDIO_BINS complex floats
    std::complex<float>* audio_pinned;

    std::atomic<uint64_t> audio_produce_idx{0};
    std::atomic<uint64_t> audio_consume_idx{0};
    std::thread audio_thread;
    void audioWorker();

    zmq::context_t audio_zmq_ctx;
    zmq::socket_t* audio_sockets[AUDIO_BANDS];
};
}
}

#endif // _GM_CUDA_HFCHANNELIZER_H_
