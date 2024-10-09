#include <unistd.h>
#include <iostream>
#include <cuda.h>
#include <complex>
#include "gm/cuda/CopyBuffer.h"
#include "gm/cuda/HostCuda.h"
#include "gm/buffer/BufferPosition.h"

namespace gm {
namespace cuda {
CopyBuffer::CopyBuffer(gm::buffer::BufferPosition<std::complex<short>>* inP) :
inPos(inP), outPos(), sampleSize(131072), outSize(64*1024*1024) {
    inSize = inPos->getBufferSize();
    //outSize = inSize;
    inData = (std::complex<short>*)inPos->getBuffer();
}

CopyBuffer::~CopyBuffer() {
    
}

int CopyBuffer::doCopy(long now, long length) {
        int out_position = (int)(now % outSize);
        int in_position = (int)(now % inSize);
        
        int out_remaining = outSize - out_position;
        int in_remaining = inSize - in_position;
        
        int firstCopy = in_remaining < out_remaining ? in_remaining : out_remaining;
        
        //Everything fits in buffer
        if (firstCopy > length) {
            cuda_check_error(cudaMemcpy(&outData_d[out_position], &inData[in_position], 
                (size_t)length*inPos->getElementSize(), cudaMemcpyHostToDevice));
            //std::cout << "Copying from " << in_position << " to " << out_position << " size " << length << std::endl;
            return (int) length;
        } else {
            cuda_check_error(cudaMemcpy(&outData_d[out_position], &inData[in_position], 
                firstCopy*inPos->getElementSize(), cudaMemcpyHostToDevice));
            //std::cout << "Copying from " << in_position << " to " << out_position << " size " << firstCopy << std::endl;
            return firstCopy;
        }
}

void CopyBuffer::run() {
    try {
        //cuda_check_error(cudaHostAlloc((void**)&outData_d, sizeof(std::complex<short>) * outSize, cudaHostAllocPortable));
        cuda_check_error(cudaMalloc((void**)&outData_d, sizeof(std::complex<short>) * outSize));
    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error after cudaSetDevice: " << e.what() << std::endl;
    }
    outPos.setBuffer(outData_d, outSize);
    long now = inPos->getNow();
    
    while(isRunning()) {
        long next = inPos->getPosition(now + sampleSize);
        
        while(now < next) {
            long length = next - now;
            if (length > outSize || length > inSize) {
                std::cout << "Error Falling Behind in Cuda Copy, Dropping Data" << std::endl;
                now = next;
                outPos.setPosition(now);
                break;
            }
            int numCopied = doCopy(now, length);
            now += numCopied;
            outPos.setPosition(now);
        }
        
    }
}

}
}
