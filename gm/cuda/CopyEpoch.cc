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
int CopyEpoch<T>::doCopy(uint64_t now) {
    size_t length = inShape[1];
    int in_position = (now % inShape[0]) * length;

    cuda_check_error(cudaMemcpy(&outData_d[in_position], &inData[in_position], 
        length*inPos->getElementSize(), cudaMemcpyHostToDevice));
    return 1;
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
    uint64_t now = inPos->getNow(1) + 1;
    
    while(isRunning()) {
        uint64_t next = inPos->getPosition(now+1, 1);
        // printf("Got epoch %d\n", next); 
        while(now < next) {
            uint64_t length = next - now;
            if (length > 4) {
                std::cout << "Error Falling Behind in Cuda Copy, Dropping Data" << std::endl;
                now = next;
                outPos.setPosition(now, 1);
                break;
            }
            int numCopied = doCopy(now);
            outPos.setPosition(now, 1);
	    now += 1;
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
