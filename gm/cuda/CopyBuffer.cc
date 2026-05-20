#include <unistd.h>
#include <iostream>
#include <cuda.h>
#include <complex>
#include "gm/cuda/CopyBuffer.h"
#include "gm/cuda/HostCuda.h"
#include "gm/buffer/BufferPosition.h"

namespace gm {
namespace cuda {

template<class T>
CopyBuffer<T>::CopyBuffer(gm::buffer::BufferPosition<T>* inP) :
inPos(inP), outPos(), sampleSize(131072), outSize(64*1024*1024), outData_d(NULL) {
    inSize = inPos->getBufferSize();
    inData = (T*)inPos->getBuffer();
}

template<class T>
CopyBuffer<T>::~CopyBuffer() {
    if (outData_d) cudaFree(outData_d);
}

template<class T>
int CopyBuffer<T>::doCopy(long now, long length) {
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
            // wrap issue handle at layer above
            cuda_check_error(cudaMemcpy(&outData_d[out_position], &inData[in_position], 
                firstCopy*inPos->getElementSize(), cudaMemcpyHostToDevice));
            //std::cout << "Copying from " << in_position << " to " << out_position << " size " << firstCopy << std::endl;
            return firstCopy;
        }
}

template<class T>
void CopyBuffer<T>::run() {
    try {
        //cuda_check_error(cudaHostAlloc((void**)&outData_d, sizeof(T) * outSize, cudaHostAllocPortable));
        cuda_check_error(cudaMalloc((void**)&outData_d, sizeof(T) * outSize));
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
                std::cerr << "Error Falling Behind in Cuda Copy, Dropping Data" << std::endl;
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


void TemporaryFunction()
{
    // needed to put the compile time objects into the object table for proper linking
    CopyBuffer<std::complex<short>> TempObjCS((gm::buffer::BufferPosition<std::complex<short>>*) NULL);
    CopyBuffer<std::complex<float>> TempObjCF((gm::buffer::BufferPosition<std::complex<float>>*) NULL);
    CopyBuffer<short> TempObjS((gm::buffer::BufferPosition<short>*) NULL);
}

}
}
