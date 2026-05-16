#ifndef _GM_CUDA_FT8CUDA_H_
#define _GM_CUDA_FT8CUDA_H_

#include <fstream>
#include <iostream>
#include <thread>
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
    const static int BUFFERS = 2;      // number of decode slots for ft8.cc
    const static int RING_BLOCKS = 200; // rolling magnitude ring size in FT8 blocks

    std::ofstream fs;
    int buffer_number;
    uint64_t ring_write_idx;
    uint64_t last_trigger_second;
    std::thread last_snapshot_thread;

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
    std::complex<float>* demodShift_d;
    uint8_t* magFT8_d;
    uint8_t* magFT8;
    uint8_t* magFT8_ring;
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
