#include <unistd.h>
#include <iostream>
#include <cuda.h>
#include <complex>
#include "gm/cuda/CopyEpoch.h"
#include "gm/cuda/HostCuda.h"
#include "gm/buffer/BufferPosition.h"

namespace gm {
namespace cuda {

template<class T>
CopyEpoch<T>::CopyEpoch(gm::buffer::BufferPosition<T>* inP) :
inPos(inP), outPos(), outData_d(NULL) {
    inShape = inPos->getShape();
    inData = (T*)inPos->getBuffer();
}

template<class T>
CopyEpoch<T>::~CopyEpoch() {
    if (outData_d) cudaFree(outData_d);
}

template<class T>
int CopyEpoch<T>::doCopy(long now, long length) {
        // int out_remaining = outSize - out_position;
        // int in_remaining = inSize - in_position;
        
        // int firstCopy = in_remaining < out_remaining ? in_remaining : out_remaining;
        
        // //Everything fits in buffer
        // if (firstCopy > length) {
        //     cuda_check_error(cudaMemcpy(&outData_d[out_position], &inData[in_position], 
        //         (size_t)length*inPos->getElementSize(), cudaMemcpyHostToDevice));
        //     //std::cout << "Copying from " << in_position << " to " << out_position << " size " << length << std::endl;
        //     return (int) length;
        // } else {
        //     // wrap issue handle at layer above
        //     cuda_check_error(cudaMemcpy(&outData_d[out_position], &inData[in_position], 
        //         firstCopy*inPos->getElementSize(), cudaMemcpyHostToDevice));
        //     //std::cout << "Copying from " << in_position << " to " << out_position << " size " << firstCopy << std::endl;
        //     return firstCopy;
        // }
    return 0;
}

template<class T>
void CopyEpoch<T>::run() {
    try {
        //cuda_check_error(cudaHostAlloc((void**)&outData_d, sizeof(T) * outSize, cudaHostAllocPortable));
        cuda_check_error(cudaMalloc((void**)&outData_d, inPos->getByteSize()));
    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error after cudaSetDevice: " << e.what() << std::endl;
    }
    outPos.setBuffer(outData_d, inShape);
    long now = inPos->getNow(1) + 1;
    
    while(isRunning()) {
        long next = inPos->getPosition(now, 1);        
        while(now < next) {
            long length = next - now;
            if (length > 4) {
                std::cout << "Error Falling Behind in Cuda Copy, Dropping Data" << std::endl;
                now = next;
                outPos.setPosition(now, 1);
                break;
            }
            //int numCopied = doCopy(now, length);
            now += 1;
            outPos.setPosition(now, 1);
        }
    }
}


void TemporaryFunction()
{
    // needed to put the compile time objects into the object table for proper linking
    CopyEpoch<std::complex<short>> TempObjCS((gm::buffer::BufferPosition<std::complex<short>>*) NULL);
    CopyEpoch<std::complex<float>> TempObjCF((gm::buffer::BufferPosition<std::complex<float>>*) NULL);
    CopyEpoch<short> TempObjS((gm::buffer::BufferPosition<short>*) NULL);
}

}
}
