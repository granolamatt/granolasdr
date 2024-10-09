#ifndef _GM_CUDA_UHFFMDEMOD_H_
#define _GM_CUDA_UHFFMDEMOD_H_

#include <cuda.h>
#include <cufft.h>
#include <thrust/complex.h>
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/cuda/CudaCopy.h"
#include "gm/cuda/FMDemodLong.h"
#include "gm/zmqcode/zmqworker.h"


namespace gm {
namespace cuda {
class UHFFMDemod : public Runnable {
public:
    const static int NLARGE = (1048576);
    const static int NSMALL = 4096;
    const static int NTUNE = 512;
    const static int NEPOCH = (NLARGE/2);
    const static int NSBUFFERS = 4;
    const static int NSTREAMS = 4;
    const static int NTOTAL = (NLARGE/NTUNE);
    const static int NCHANNELS = (NTOTAL - 10);
    UHFFMDemod(gm::buffer::BufferPosition<std::complex<short>>* inP);
    ~UHFFMDemod();
    void run();
    void stop() {
        setRunning(false);
    }
    gm::buffer::BufferPosition<float>* getOutputBufferPos() {
        return &outPos;
    }
    float* getSquelchData() {
        return squelch;
    }
    float* getAveData() {
        return aveData;
    }
private:
    thrust::complex<float>* fftData_d;
    float* answer;
    float* squelch;
    float* answer_d;
    float* squelch_d;
    float* aveData_d;
    float* aveData;
    thrust::complex<short>* inData_d;
    int dmaSize;
    long epoch;
    int streamNum;
    int bufferNum;
    //thrust::complex<short>* inData;
    gm::buffer::BufferPosition<std::complex<short>>* inPos;
    gm::buffer::BufferPosition<float> outPos;
    CudaCopy* cudaCopy[NSBUFFERS];
    FMDemodLong* fmDemod[NSBUFFERS];
    cufftHandle plan[NSTREAMS];
    cufftHandle iplan[NSTREAMS];
    cufftHandle dplan[NSTREAMS];
    cufftHandle diplan[NSTREAMS];
    cudaStream_t stream[NSTREAMS];
    gm::zmqcode::func_t processSamples();
    gm::zmqcode::func_t setFreqRange();
    gm::zmqcode::func_t getFreqs();
};
}
}

#endif //_GM_CUDA_UHFFMDEMOD_H_
