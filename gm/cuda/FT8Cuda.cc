#include <unistd.h>
#include <iostream>
#include <cuda.h>
#include <complex>
#include <chrono>

#include "gm/cuda/FT8Cuda.h"
#include "gm/cuda/HostCuda.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/rx888/rx888.h"

#include "ft8_lib/common/monitor.h"

namespace gm {
namespace cuda {

FT8Cuda::FT8Cuda(gm::buffer::BufferPosition<std::complex<float>>* inP) :
inPos(inP),
buff_pos{0},
startcap(false),
demodData_d(NULL),
demodFT8_d(NULL),
num_blocks(0),
buffer_number(0) {
    inShape = inPos->getShape();
    inData_d = (std::complex<float>*)inPos->getBuffer();

    try {
        cuda_check_error(cudaSetDevice(0));
        cuda_check_error(cudaStreamCreate(&stream));

        cuda_h = gm::cuda::device::HostCuda(stream);

        cuda_h = gm::cuda::device::HostCuda(stream);
        rfft_length = 698880; // half off for some reason

        demodFT8 = (std::complex<float>*)calloc(sizeof(std::complex<float>), (FT8_NN + 14)*BUFFERS*rfft_length);
        
        rt8BufferPosition.setBuffer(demodFT8, {BUFFERS,rfft_length*(FT8_NN + 14)});

        // Two so we can use it as a buffer also
        cuda_check_error(cudaMalloc((void**)&demodData_d, 2*rfft_length*sizeof(std::complex<float>) + 1024));
        printf("Total rfft length is %u\n", rfft_length);

        // Two so we can use it as a buffer also
        cuda_check_error(cudaMalloc((void**)&demodFT8_d, rfft_length*sizeof(std::complex<float>) + 1024));
        printf("Total rfft length is %u\n", rfft_length);

        // Now for the sub channels
        cufftResult fftRes = cufftPlan1d(&rplan, rfft_length, CUFFT_C2C, 1);
        if (fftRes) {
            printf("Error: exit for now\n");
        }
        fftRes = cufftSetStream(rplan, stream);
        if (fftRes) {
            printf("Error: exit for now\n");
        }

    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error after cudaSetDevice: " << e.what() << std::endl;
    }
}

FT8Cuda::~FT8Cuda() {
    if (demodData_d) cudaFree(demodData_d);
    if (demodFT8_d) cudaFree(demodFT8_d);
    cufftDestroy(rplan);
    cudaStreamDestroy(stream);
}

int FT8Cuda::doCopy(uint64_t now) {
    try {
        size_t length = inShape[1];


        cudaMemcpyAsync(&demodData_d[buff_pos], &inData_d[length * (now % BUFFERS)], 
                        length * sizeof(std::complex<float>), cudaMemcpyDeviceToDevice, stream);
        buff_pos += length;

        if (buff_pos > rfft_length) {
            auto nowsec = std::chrono::system_clock::now();
            auto duration = nowsec.time_since_epoch();
            double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
            uint64_t trigger = (uint64_t)(seconds) % 15;
            lastsecond = seconds;
            
            bool gotime = (trigger == 14 && trunc(seconds) > 0.81);
            if (gotime && ~startcap) {
                startcap = true;
            }

            cufftResult rval = cufftExecC2C(rplan, (cufftComplex *)&demodData_d[0],
                (cufftComplex *)&demodFT8_d[0], CUFFT_FORWARD);
            if (rval) {
                printf("Error in fft\n");
                return 0;
            }
            buff_pos -= rfft_length / oversample;
            // I think this will not stomp on the data
            cuda_check_error(cudaMemcpyAsync(&demodData_d[0], 
                &demodData_d[rfft_length / oversample],
                buff_pos * sizeof(float),cudaMemcpyDeviceToDevice, stream));
            if (startcap) {
                // printf("Do the bb fft %u delta %f\n", buff_pos, seconds - lastepoch);
                int buffnum = buffer_number % BUFFERS;
                // cuda_h.magKernel(&demodFT8_d[0], &demodFT8[num_blocks*rfft_length + buffnum*rfft_length*(FT8_NN + 14)], rfft_length);
                cuda_check_error(cudaMemcpyAsync(&demodFT8[num_blocks*rfft_length + buffnum*rfft_length*(FT8_NN + 14)], 
                    &demodFT8_d[0],
                    rfft_length * sizeof(std::complex<float>),cudaMemcpyDeviceToHost, stream));
                num_blocks++;
                // fs.write(reinterpret_cast<const char*>(demodFT8), rfft_length * sizeof(std::complex<float>));
                if (num_blocks >= (FT8_NN + 14)) {
                    printf("Processing buffer %u\n", buffer_number);
                    cudaStreamSynchronize(stream);
                    buffer_number++;

                    rt8BufferPosition.setPosition(buffer_number, 1);
                    // Decode accumulated data (containing slightly less than a full time slot)
                    //decode(&mon, seconds);
                    startcap = false;
                    num_blocks = 0;
                }

            }
            lastepoch = seconds;
            
        }

        return 1;
    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error after cudaCopy: " << e.what() << std::endl;
    }
    return 0;
}

void FT8Cuda::run() {
    uint64_t now = inPos->getNow(1) + 1;
    
    while(isRunning()) {
        uint64_t next = inPos->getPosition(now+1, 1);
        while(now < next) {
            uint64_t length = next - now;
            if (length > 4) {
                std::cout << "Error Falling Behind in Cuda Copy, Dropping Data" << std::endl;
                now = next;
                break;
            }
            int numCopied = doCopy(now);
            if (!numCopied) exit(-200);
	        now += numCopied;
        }
    }
}

}
}
