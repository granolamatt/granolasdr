#ifndef _GM_CUDA_FT8CUDA_H_
#define _GM_CUDA_FT8CUDA_H_

#include <fstream>
#include <iostream>
#include <cuda.h>
#include <cufft.h>
#include "gm/cuda/HostCuda.h"
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"

namespace gm {
namespace cuda {

class FT8Cuda : public Thread {
public:
    FT8Cuda(gm::buffer::BufferPosition<std::complex<float>>* inP);
    ~FT8Cuda();
    void run();
    void stop() {
        setRunning(false);
    }
    gm::buffer::BufferPosition<uint8_t>* getBuffer() {
        return &rt8BufferPosition;
    }
private:
    // bool running;
    cudaStream_t stream;
    cufftHandle rplan;
    double lastepoch;
    const static int BUFFERS = 16;

    std::ofstream fs;
    bool startcap;
    int num_blocks;
    int buffer_number;

    gm::buffer::BufferPosition<uint8_t> rt8BufferPosition;

    gm::cuda::device::HostCuda cuda_h;
    std::vector<size_t> inShape;
    double freqsperbin;
    std::complex<float>* fftData_d;
    gm::buffer::BufferPosition<std::complex<float>>* inPos;
    std::complex<float> *inData_d;
    int doCopy(uint64_t now);
    std::complex<float>* demodData_d;
    std::complex<float>* demodFT8_d;
    uint8_t* magFT8_d;
    uint8_t* magFT8;
    std::vector<std::vector<uint32_t>> bins;
    size_t rfft_length;
    uint32_t nTune;
    uint32_t nChannels;
    double lastsecond;

    uint32_t buff_pos;


};
}
}

#endif // _GM_CUDA_FT8CUDA_H_
