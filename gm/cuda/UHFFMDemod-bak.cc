#include <cuda.h>
#include <cufft.h>
#include <thrust/complex.h>
#include "gm/cuda/HostCuda.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/cuda/CudaCopy.h"
#include "gm/cuda/FMDemodLong.h"
#include "gm/cuda/UHFFMDemod.h"

namespace gm {
namespace cuda {
UHFFMDemod::UHFFMDemod(gm::buffer::BufferPosition<std::complex<short>>* inP) :
    inPos(inP),
    inData_d((thrust::complex<short>*)inP->getBuffer()),
    dmaSize(inP->getBufferSize()),
    outPos()
{
    try {
        cuda_check_error(cudaMalloc((void**)&fftData_d, sizeof(thrust::complex<float>) * NLARGE * NSBUFFERS));
        cuda_check_error(cudaMalloc((void**)&fftDatao_d, sizeof(thrust::complex<float>) * NLARGE * NSBUFFERS));
        cuda_check_error(cudaMalloc((void**)&aveData_d, sizeof(float) * NSMALL));
        outPos.setBuffer((std::complex<float>*)fftData_d, NLARGE * NSBUFFERS);

        for (int cnt = 0; cnt < NSTREAMS; cnt++) {
            cuda_check_error(cudaStreamCreate(&stream[cnt]));
            cufftResult fftRes = cufftPlan1d(&plan[cnt], NLARGE, CUFFT_C2C, 1);
            //cufftResult fftRes = cufftPlan1d(&plan[cnt], NLARGE/2048, CUFFT_C2C, 2048);
            if (fftRes) {
                printf("Error: exit for now\n");
            }
            fftRes = cufftSetStream(plan[cnt], stream[cnt]);
            if (fftRes) {
                printf("Error: exit for now\n");
            }
            fftRes = cufftPlan1d(&iplan[cnt], 2048, CUFFT_C2C, 400);
            if (fftRes) {
                printf("Error: exit for now\n");
            }
            fftRes = cufftSetStream(iplan[cnt], stream[cnt]);
            if (fftRes) {
                printf("Error: exit for now\n");
            }
        }
        for (int cnt = 0; cnt < NSBUFFERS; cnt++) {
            int in_pos = (cnt % NSBUFFERS) * NEPOCH;
            int out_pos = (cnt % NSBUFFERS) * NLARGE;
            cudaCopy[cnt] = new CudaCopy(stream[(cnt % NSTREAMS)]);
            cudaCopy[cnt]->setInput(&inData_d[in_pos]);
            cudaCopy[cnt]->setOutput(&fftData_d[out_pos]);
            cudaCopy[cnt]->setSize(NLARGE);
            fmDemod[cnt] = new FMDemodLong(stream[(cnt % NSTREAMS)]);
            fmDemod[cnt]->setSize(NEPOCH);
            fmDemod[cnt]->setOutput(&fftData_d[out_pos]);
            fmDemod[cnt]->setOutputO(&fftDatao_d[out_pos]);
        }
        cudaCopy[NSBUFFERS-1]->setSize(NEPOCH);
        cudaCopy[0]->setAve(&aveData_d[0]);
    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error after cudaSetDevice: " << e.what() << std::endl;
    }
    
	gm::zmqcode::zmqWorker().getFuncMap().insert({"getData", processSamples()});
	gm::zmqcode::zmqWorker().getFuncMap().insert({"freqRange", setFreqRange()});
	gm::zmqcode::zmqWorker().getFuncMap().insert({"getFreqs", getFreqs()});
    
}


UHFFMDemod::~UHFFMDemod() {
    for (int cnt = 0; cnt < NSBUFFERS; cnt++) {
        delete cudaCopy[cnt];
        delete fmDemod[cnt];
    }
    cudaFree((void*)inData_d);
    cudaFree((void*)fftData_d);
    cudaFree((void*)fftDatao_d);
    //XXX Need to clean up streams and cufft

}

void UHFFMDemod::run() {
    try
    {
        running = true;
        printf("Starting host cuda runnable\n");
        cuda_check_error(cudaSetDevice(0));
        //      cuda_check_error(cudaDeviceSetLimit(cudaLimitMallocHeapSize,20000000*sizeof(thrust::complex<float>)));
        long position = inPos->getNow();
        epoch = position / NEPOCH + 10;
        static gm::buffer::BufferPosition<std::complex<float>>* bpos = &outPos;
        

        while (running) {
            position = epoch * NEPOCH;
            //const int length = inPos->getElementSize() * NEPOCH;
            //printf("On epoch %ld at %ld not %ld \n", epoch, position, inPos->getNow());
            inPos->getPosition(position + NLARGE);
            streamNum = epoch % NSTREAMS;
            bufferNum = epoch % NSBUFFERS;
            cudaStream_t myStream = stream[streamNum];
            cufftHandle myPlan = plan[streamNum];
            cufftHandle myiPlan = iplan[streamNum];
            thrust::complex<float>* fft_d = &fftData_d[bufferNum * NLARGE];
            thrust::complex<float>* ffto_d = &fftDatao_d[bufferNum * NLARGE];

            if (bufferNum == NSBUFFERS-1) {
                int in_pos = bufferNum * NEPOCH;
                cudaCopy[bufferNum]->setInput(&inData_d[in_pos]);
                cudaCopy[bufferNum]->copyKernel();
                cudaCopy[bufferNum]->setInput(&inData_d[0]);
                cudaCopy[bufferNum]->copyKernel();
            } else {
                cudaCopy[bufferNum]->copyKernel();
            }
            cuda_check_error(cudaPeekAtLastError());
            cufftResult_t rval = cufftExecC2C(myPlan, (cufftComplex *)fft_d, (cufftComplex *) fft_d, CUFFT_FORWARD);
            if (rval) {
                printf("Error in fft\n");
            }
            if(bufferNum == 0) {
                cudaCopy[0]->averageKernel();
                cudaMemcpy(aveData, aveData_d, NSMALL*sizeof(float), cudaMemcpyDeviceToHost);
            }
            // rval = cufftExecC2C(myiPlan, (cufftComplex *)&fft_d[2048*50], (cufftComplex *) &ffto_d[2048*50], CUFFT_INVERSE);
            // if (rval) {
            //     printf("Error in fft\n");
            // }
            // rval = cufftExecC2C(myiPlan, (cufftComplex *)&fft_d[2048*50+1024], (cufftComplex *) &fft_d[2048*50], CUFFT_INVERSE);
            // if (rval) {
            //     printf("Error in fft\n");
            // }
            // fmDemod[bufferNum]->polarDescriminator();
            // cuda_check_error(cudaPeekAtLastError());

            cudaStreamAddCallback(myStream,
            [&](cudaStream_t stream, cudaError_t status, void *data) {
                bpos->setPosition((long)data);
                long position = bpos->getNow();
                //printf("callback now %ld\n", position);
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

gm::zmqcode::func_t UHFFMDemod::processSamples() {
	return [&](zmq::socket_t* socket, zmq::message_t* identity) {
		float mag[NSMALL];
		memset(&mag[0], 0, sizeof(mag));

		for (int cnt = 0; cnt < NSMALL; cnt++) {
			float val = std::abs(aveData[cnt]);
			int idx =  cnt;
			mag[idx] += val;
		}
		socket->send(*identity, ZMQ_SNDMORE);
		std::string func("NO ERROR GETDATA");
		// Wrap this in proto buffers
		socket->send(func.c_str(), func.size(), ZMQ_SNDMORE);
		socket->send(&mag[0], 4000*4);
	};
}

gm::zmqcode::func_t UHFFMDemod::getFreqs() {
	return [&](zmq::socket_t* socket, zmq::message_t* identity) {
		float startf = (160000000 - 12500000) / 1000000.0f;
		float stopf = (442000000 + 12500000 - 2014) / 1000000.0f;
		socket->send(*identity, ZMQ_SNDMORE);
		std::string func("NO ERROR GETFREQS");
		// Wrap this in proto buffers
		std::stringstream ss;
		ss << startf;
		std::string start_freq = ss.str();
		std::stringstream st;
		st << stopf;
		std::string stop_freq = st.str();

		socket->send(func.c_str(), func.size(), ZMQ_SNDMORE);
		socket->send(start_freq.c_str(), start_freq.size(), ZMQ_SNDMORE);
		socket->send(stop_freq.c_str(), stop_freq.size());
	};
}

gm::zmqcode::func_t UHFFMDemod::setFreqRange() {
	return [&](zmq::socket_t* socket, zmq::message_t* identity) {
		zmq::message_t message;
		int more;
		size_t more_size = sizeof (more);
		zmq::message_t start_message;
		zmq::message_t stop_message;

		socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
		float freqs[2];
		freqs[0] = 0;
		freqs[1] = 0;

		if (more) {
			socket->recv(&start_message);
			std::string startf = std::string(static_cast<char*>(start_message.data()), start_message.size());
			socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
			if (more) {
				socket->recv(&stop_message);
				std::string stopf = std::string(static_cast<char*>(stop_message.data()), stop_message.size());
				freqs[0] = atof(startf.c_str());
				freqs[1] = atof(stopf.c_str());
			}
		}
		socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
		socket->send(*identity, ZMQ_SNDMORE);
		std::string func("NO ERROR FREQRANGE");
		socket->send(func.c_str(), func.size());
	};
}

}
}


