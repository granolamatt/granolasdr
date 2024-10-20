#ifndef _GM_CUDA_HFCHANNELIZER_H_
#define _GM_CUDA_HFCHANNELIZER_H_

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
        
private:
    // bool running;
    cudaStream_t stream;
    cufftHandle plan;
    cufftHandle iplan;

    gm::cuda::device::HostCuda cuda_h;
    std::vector<size_t> inShape;
    int16_t* inData_d; // cuda copy of rx data
    int16_t* inData; // rx data
    float* fftInData_d;
    std::complex<float>* fftData_d;
    gm::buffer::BufferPosition<int16_t>* inPos;
    int doCopy(uint64_t now);
    std::complex<float>* channelData_d;
    std::vector<std::vector<uint32_t>> bins;
    std::vector<std::vector<uint32_t>> getBins();
    uint32_t fft_length;
};
}
}

#endif // _GM_CUDA_HFCHANNELIZER_H_
