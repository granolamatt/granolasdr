#include <cuda.h>
#include <cufft.h>
#include <thrust/complex.h>
#include "gm/cuda/HostCuda.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/cuda/CudaCopy.h"
#include "gm/cuda/OverlapSave.h"

namespace gm {
namespace cuda {
OverlapSave::OverlapSave(gm::buffer::BufferPosition<std::complex<short>>* inP) :
    inPos(inP),
    inData((thrust::complex<short>*)inP->getBuffer()),
    dmaSize(inP->getBufferSize()),
    outPos()
{
    try {
//        cuda_check_error(cudaHostRegister(inData, inP->getElementSize()*dmaSize, 0));
        cuda_check_error(cudaMalloc((void**)&inData_d, sizeof(thrust::complex<short>) * NLARGE * NSTREAMS));
        cuda_check_error(cudaMalloc((void**)&fftData_d, sizeof(thrust::complex<float>) * NLARGE * NSBUFFERS));
        outPos.setBuffer((std::complex<float>*)fftData_d, NLARGE * NSBUFFERS);

        for (int cnt = 0; cnt < NSTREAMS; cnt++) {
            cuda_check_error(cudaStreamCreate(&stream[cnt]));
            cufftResult fftRes = cufftPlan1d(&plan[cnt], NLARGE, CUFFT_C2C, 1);
            if (fftRes) {
                printf("Error: exit for now\n");
            }
            fftRes = cufftSetStream(plan[cnt], stream[cnt]);
            if (fftRes) {
                printf("Error: exit for now\n");
            }
        }
        for (int cnt = 0; cnt < NSBUFFERS; cnt++) {
            int in_pos = (cnt % NSTREAMS) * NLARGE;
            int out_pos = (cnt % NSBUFFERS) * NLARGE;
            cudaCopy[cnt] = new CudaCopy(stream[(cnt % NSTREAMS)]);
            cudaCopy[cnt]->setInput((std::complex<short>*)&inData_d[in_pos]);
            cudaCopy[cnt]->setOutput((std::complex<float>*)&fftData_d[out_pos]);
            cudaCopy[cnt]->setSize(NEPOCH);
        }
    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error after cudaSetDevice: " << e.what() << std::endl;
    }
}


OverlapSave::~OverlapSave() {
    for (int cnt = 0; cnt < NSBUFFERS; cnt++) {
        delete(cudaCopy[cnt]);
    }
    cudaFree((void*)inData_d);
    cudaFree((void*)fftData_d);
    //XXX Need to clean up streams and cufft

}

void OverlapSave::run() {
    try
    {
        running = true;
        printf("Starting host cuda runnable\n");
        cuda_check_error(cudaSetDevice(0));
        //		cuda_check_error(cudaDeviceSetLimit(cudaLimitMallocHeapSize,20000000*sizeof(thrust::complex<float>)));

        long position = inPos->getNow();
        epoch = position / NEPOCH + 1;
        static gm::buffer::BufferPosition<std::complex<float>>* bpos = &outPos;
        while(running) {
            printf("On epoch %ld\n", epoch);
            position = epoch * NEPOCH;
            const int length = inPos->getElementSize() * NEPOCH;
            inPos->getPosition(position);
            int dma_position = (int)(position % dmaSize);
            streamNum = epoch % NSTREAMS;
            bufferNum = epoch % NSBUFFERS;
            int lastStreamNum = streamNum - 1;
            if (lastStreamNum < 0) {
                lastStreamNum += NSTREAMS;
            }
            cudaStream_t myStream = stream[streamNum];
            cufftHandle myPlan = plan[streamNum];
            thrust::complex<short>* data_h = &inData[dma_position];
            thrust::complex<short>* data_d = &inData_d[streamNum*NLARGE];
            thrust::complex<short>* last_data_d = &inData_d[lastStreamNum*NLARGE];
            thrust::complex<float>* fft_d = &fftData_d[bufferNum*NLARGE];

            const int first = dmaSize - dma_position;
            if (first >= NEPOCH) {
                cudaMemcpy(&data_d[NEPOCH], data_h, length, cudaMemcpyHostToDevice);
            } else {
                cudaMemcpy(&data_d[NEPOCH], data_h, first * inPos->getElementSize(), cudaMemcpyHostToDevice);
                cudaMemcpy(&data_d[NEPOCH + first], inData, length - first * inPos->getElementSize(), cudaMemcpyHostToDevice);
            }
            cudaMemcpy(data_d, &last_data_d[NEPOCH], length, cudaMemcpyDeviceToDevice);
            cudaCopy[bufferNum]->copyKernel();
            cuda_check_error(cudaPeekAtLastError());
            cufftResult_t rval = cufftExecC2C(myPlan, (cufftComplex *)fft_d, (cufftComplex *) fft_d, CUFFT_FORWARD);
            if (rval) {
                printf("Error in fft\n");
            }
            //cudaStreamAddCallback(myStream, MyCallback, (void *)&outPos, 0);
            cudaStreamAddCallback(myStream,
            [](cudaStream_t stream, cudaError_t status, void *data) {
                bpos->setPosition((long)data);
                long position = bpos->getNow();
                printf("callback now %ld\n", position);
            }
            , (void *)epoch, 0);
            epoch++;
        }
    }
    catch (thrust::system_error &e)
    {
        std::cerr << "CUDA error after cudaSetDevice: " << e.what() << std::endl;
    }
}

}
}


