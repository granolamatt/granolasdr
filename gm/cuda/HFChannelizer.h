#ifndef _GM_CUDA_HFCHANNELIZER_H_
#define _GM_CUDA_HFCHANNELIZER_H_

#include <fstream>
#include <iostream>
#include <cuda.h>
#include <cufft.h>
#include "gm/cuda/HostCuda.h"
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"

namespace gm {
namespace cuda {

class HFChannelizer : public Runnable {
public:
    HFChannelizer(gm::buffer::BufferPosition<int16_t>* inP);
    ~HFChannelizer();
    void run();
    void stop() {
        setRunning(false);
    }
    const static int oversample = 1;
        
private:
    // bool running;
    cudaStream_t stream;
    cufftHandle plan;
    cufftHandle iplan;
    cufftHandle rplan;
    double lastepoch;
    const static int BUFFERS = 16;

    std::ofstream fs;
    bool startcap;
    int num_blocks;
    int buffer_number;

    gm::buffer::BufferPosition<std::complex<float>> rt8BufferPosition;

    gm::cuda::device::HostCuda cuda_h;
    std::vector<size_t> inShape;
    int16_t* inData_d; // cuda copy of rx data
    int16_t* inData; // rx data
    float* fftInData_d;
    double freqsperbin;
    std::complex<float>* fftData_d;
    gm::buffer::BufferPosition<int16_t>* inPos;
    int doCopy(uint64_t now);
    std::complex<float>* channelData_d;
    std::complex<float>* demodData_d;
    std::complex<float>* demodFT8_d;
    std::complex<float>* demodFT8;
    char* pixel_d;
    std::vector<std::vector<uint32_t>> bins;
    uint32_t fft_length;
    uint32_t rfft_length;
    uint32_t nTune;
    uint32_t nChannels;

    uint32_t buff_pos;


};
}
}

#endif // _GM_CUDA_HFCHANNELIZER_H_
