#include <unistd.h>
#include <iostream>
#include <cuda.h>
#include <complex>
#include "gm/cuda/HFChannelizer.h"
#include "gm/cuda/HostCuda.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/rx888/rx888.h"

namespace gm {
namespace cuda {

HFChannelizer::HFChannelizer(gm::buffer::BufferPosition<int16_t>* inP) :
inPos(inP), inData_d(NULL), fftInData_d(NULL), fftData_d(NULL) {
    inShape = inPos->getShape();
    inData = (int16_t*)inPos->getBuffer();
    try {
        //cuda_check_error(cudaHostAlloc((void**)&inData_d, inPos->getByteSize(), cudaHostAllocPortable));
        cuda_check_error(cudaMalloc((void**)&inData_d, inPos->getByteSize()));
        cuda_check_error(cudaMalloc((void**)&fftInData_d, 
            sizeof(float)*(gm::rx888::rx888::NLARGE*gm::rx888::rx888::BUFFERS + gm::rx888::rx888::NLARGE)));
        cuda_check_error(cudaMalloc((void**)&fftData_d, 
            sizeof(std::complex<float>)*(2*gm::rx888::rx888::NLARGE + 1024))); // 1024 is pad for the extra data from realfft

        cuda_check_error(cudaSetDevice(0));
        cuda_check_error(cudaStreamCreate(&stream));
        cufftResult fftRes = cufftPlan1d(&plan, gm::rx888::rx888::NLARGE*2, CUFFT_R2C, 1);
        cuda_h = gm::cuda::device::HostCuda(stream);
    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error after cudaSetDevice: " << e.what() << std::endl;
    }

}

HFChannelizer::~HFChannelizer() {
    if (inData_d) cudaFree(inData_d);
    if (fftInData_d) cudaFree(fftInData_d);
    if (fftData_d) cudaFree(fftData_d);
    cufftDestroy(plan);
    cudaStreamDestroy(stream);
}

int HFChannelizer::doCopy(uint64_t now) {
    try {
        size_t length = inShape[1];
        int in_position = (now % inShape[0]) * length;
        // first copy the data into the device
        cuda_check_error(cudaMemcpyAsync(&inData_d[in_position], &inData[in_position], 
            length*inPos->getElementSize(), cudaMemcpyHostToDevice, stream));
        // now cast it to floats and scale it
        cuda_h.copyKernel(&fftInData_d[in_position + gm::rx888::rx888::NLARGE], &inData_d[in_position], length);
        // if we are on first buffer then do the wrap
        if (!(now % gm::rx888::rx888::BUFFERS)) {
            cudaMemcpyAsync(&fftInData_d, 
                            &fftInData_d[in_position + gm::rx888::rx888::NLARGE], 
                            inShape[1]*sizeof(float),cudaMemcpyDeviceToDevice, stream);
        }
        cufftResult_t rval = cufftExecR2C(plan, &fftInData_d[in_position], (cufftComplex *) fftData_d);
        if (rval) {
            printf("Error in fft\n");
            return 0;
        }

        return 1;
    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error after cudaCopy: " << e.what() << std::endl;
    }
    return 0;
}

void HFChannelizer::run() {
    uint64_t now = inPos->getNow(1) + 1;
    
    while(isRunning()) {
        uint64_t next = inPos->getPosition(now+1, 1);
        printf("Got epoch %d\n", next);
        while(now < next) {
            uint64_t length = next - now;
            if (length > 4) {
                std::cout << "Error Falling Behind in Cuda Copy, Dropping Data" << std::endl;
                now = next;
                break;
            }
            printf("Going copy on epoch %d\n", now);
            int numCopied = doCopy(now);
            if (!numCopied) exit(-200);
            //outPos.setPosition(now, 1);
	    now += numCopied;
        }
    }
}

}
}
