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
        cuda_check_error(cudaMalloc((void**)&fftData_d, sizeof(thrust::complex<float>) * NLARGE * NSBUFFERS * 4));
        cuda_check_error(cudaMalloc((void**)&aveData_d, sizeof(float) * NSMALL));
        cuda_check_error(cudaMalloc((void**)&answer_d, sizeof(float) * (NLARGE * NSBUFFERS * 4 + 4 * NTOTAL * NSBUFFERS + 2048)));
        cuda_check_error(cudaMalloc((void**)&squelch_d, sizeof(float) * (4 * NTOTAL * NSBUFFERS)));
        cuda_check_error(cudaMallocHost((void**)&answer, sizeof(float) * (NLARGE * NSBUFFERS * 4)));
        cuda_check_error(cudaMallocHost((void**)&squelch, sizeof(float) * (4 * NTOTAL * NSBUFFERS)));
        cuda_check_error(cudaMallocHost((void**)&aveData, sizeof(float) * NSMALL));
        outPos.setBuffer((float*)answer, NLARGE * NSBUFFERS * 4);
        
        void* zeros = calloc(sizeof(float), NSMALL * NSBUFFERS);
        cudaMemcpy(aveData_d, zeros, sizeof(float) * NSMALL * NSBUFFERS, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        free(zeros);

        for (int cnt = 0; cnt < NSTREAMS; cnt++) {
            cuda_check_error(cudaStreamCreate(&stream[cnt]));
            cufftResult fftRes = cufftPlan1d(&plan[cnt], NLARGE, CUFFT_C2C, 1);
            //cufftResult fftRes = cufftPlan1d(&plan[cnt], NLARGE/NTUNE, CUFFT_C2C, NTUNE);
            if (fftRes) {
                printf("Error: exit for now\n");
            }
            fftRes = cufftSetStream(plan[cnt], stream[cnt]);
            if (fftRes) {
                printf("Error: exit for now\n");
            }
            fftRes = cufftPlan1d(&iplan[cnt], NTUNE, CUFFT_C2C, NCHANNELS);
            if (fftRes) {
                printf("Error: exit for now\n");
            }
            fftRes = cufftSetStream(iplan[cnt], stream[cnt]);
            if (fftRes) {
                printf("Error: exit for now\n");
            }
            fftRes = cufftPlan1d(&dplan[cnt], NTUNE, CUFFT_R2C, NTOTAL*4);
            if (fftRes) {
                printf("Error: exit for now\n");
            }
            fftRes = cufftSetStream(dplan[cnt], stream[cnt]);
            if (fftRes) {
                printf("Error: exit for now\n");
            }
            fftRes = cufftPlan1d(&diplan[cnt], NTUNE, CUFFT_C2R, NTOTAL*4);
            if (fftRes) {
                printf("Error: exit for now\n");
            }
            fftRes = cufftSetStream(diplan[cnt], stream[cnt]);
            if (fftRes) {
                printf("Error: exit for now\n");
            }
        }
        for (int cnt = 0; cnt < NSBUFFERS; cnt++) {
            int in_pos = (cnt % NSBUFFERS) * NEPOCH;
            int out_pos = (cnt % NSBUFFERS) * NLARGE * 4;
            cudaCopy[cnt] = new CudaCopy(stream[(cnt % NSTREAMS)]);
            cudaCopy[cnt]->setInput((std::complex<short>*)&inData_d[in_pos]);
            cudaCopy[cnt]->setOutput((std::complex<float>*)&fftData_d[out_pos]);
            cudaCopy[cnt]->setSize(NLARGE);
            cudaCopy[cnt]->setAve(&aveData_d[(cnt % NSBUFFERS) * NSMALL]);
            fmDemod[cnt] = new FMDemodLong(stream[(cnt % NSTREAMS)]);
            fmDemod[cnt]->setSize(NLARGE * 4, NTUNE);
            fmDemod[cnt]->makeFilter();
            fmDemod[cnt]->makeDeEmphasis();
            fmDemod[cnt]->setOutput((std::complex<float>*)&fftData_d[out_pos]);
            fmDemod[cnt]->setAnswer(&answer_d[out_pos + (cnt % NSBUFFERS) * NTOTAL * 4]);
            fmDemod[cnt]->setSquelch(&squelch_d[(cnt % NSBUFFERS) * NTOTAL * 4]);
        }
        cudaCopy[NSBUFFERS-1]->setSize(NEPOCH);
    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error after cudaSetDevice: " << e.what() << std::endl;
    }
    
//	gm::zmqcode::zmqWorker().getFuncMap().insert({"getData", processSamples()});
//	gm::zmqcode::zmqWorker().getFuncMap().insert({"freqRange", setFreqRange()});
//	gm::zmqcode::zmqWorker().getFuncMap().insert({"getFreqs", getFreqs()});
    
}


UHFFMDemod::~UHFFMDemod() {
    for (int cnt = 0; cnt < NSBUFFERS; cnt++) {
        delete cudaCopy[cnt];
        delete fmDemod[cnt];
    }
    cudaFree((void*)inData_d);
    cudaFree((void*)fftData_d);
    cudaFree((void*)answer_d);
    //XXX Need to clean up streams and cufft

}

void UHFFMDemod::run() {
    try
    {
        printf("Starting host cuda runnable\n");
        cuda_check_error(cudaSetDevice(0));
        //      cuda_check_error(cudaDeviceSetLimit(cudaLimitMallocHeapSize,20000000*sizeof(thrust::complex<float>)));
        long position = inPos->getNow();
        epoch = position / NEPOCH + 10;
        static gm::buffer::BufferPosition<float>* bpos = &outPos;
        

        while (isRunning()) {
            position = epoch * NEPOCH;
            //const int length = inPos->getElementSize() * NEPOCH;
            //printf("On epoch %ld at %ld not %ld \n", epoch, position, inPos->getNow());
            inPos->getPosition(position + NLARGE);
            streamNum = epoch % NSTREAMS;
            bufferNum = epoch % NSBUFFERS;
            cudaStream_t myStream = stream[streamNum];
            cufftHandle myPlan = plan[streamNum];
            cufftHandle myiPlan = iplan[streamNum];
            cufftHandle mydPlan = dplan[streamNum];
            cufftHandle mydiPlan = diplan[streamNum];
            thrust::complex<float>* fft_d = &fftData_d[bufferNum * NLARGE * 4];
            float* ans_d = &answer_d[bufferNum * NLARGE * 4 + bufferNum * NTOTAL * 4];
            //thrust::complex<float>* ffto_d = &fftDatao_d[bufferNum * NLARGE];

            if (bufferNum == NSBUFFERS-1) {
                int in_pos = bufferNum * NEPOCH;
                int out_pos = bufferNum * NLARGE * 4;
                cudaCopy[bufferNum]->setInput((std::complex<short>*)&inData_d[in_pos]);
                cudaCopy[bufferNum]->setOutput((std::complex<float>*)&fftData_d[out_pos]);
                cudaCopy[bufferNum]->copyKernel();
                cudaCopy[bufferNum]->setInput((std::complex<short>*)&inData_d[0]);
                cudaCopy[bufferNum]->setOutput((std::complex<float>*)&fftData_d[out_pos + NEPOCH]);
                cudaCopy[bufferNum]->copyKernel();
                cudaCopy[bufferNum]->setOutput((std::complex<float>*)&fftData_d[out_pos]);
            } else {
                cudaCopy[bufferNum]->copyKernel();
            }
            cuda_check_error(cudaPeekAtLastError());
            cufftResult_t rval = cufftExecC2C(myPlan, (cufftComplex *)fft_d, (cufftComplex *) fft_d, CUFFT_FORWARD);
            if (rval) {
                printf("Error in fft\n");
            }
            
            if (bufferNum == 0) {
            cudaCopy[bufferNum]->averageKernel();
            cudaMemcpyAsync(&aveData[bufferNum * NSMALL], &aveData_d[bufferNum * NSMALL], NSMALL*sizeof(float), cudaMemcpyDeviceToHost, myStream);
            }
            fmDemod[bufferNum]->doDemod(myiPlan, (cufftComplex *)&fft_d[NTUNE*5 + 3*NTUNE/4], (cufftComplex *) &fft_d[NTUNE*5 + 3*NLARGE]);
            fmDemod[bufferNum]->doDemod(myiPlan, (cufftComplex *)&fft_d[NTUNE*5 + NTUNE/2], (cufftComplex *) &fft_d[NTUNE*5 + 2*NLARGE]);
            fmDemod[bufferNum]->doDemod(myiPlan, (cufftComplex *)&fft_d[NTUNE*5 + NTUNE/4], (cufftComplex *) &fft_d[NTUNE*5 + 1*NLARGE]);
            fmDemod[bufferNum]->doDemod(myiPlan, (cufftComplex *)&fft_d[NTUNE*5], (cufftComplex *) &fft_d[NTUNE*5]);
            
            fmDemod[bufferNum]->doSquelch();
            fmDemod[bufferNum]->polarDescriminator();
            cuda_check_error(cudaPeekAtLastError());
            rval = cufftExecR2C(mydPlan, (cufftReal *)ans_d, (cufftComplex *)ans_d);
            if (rval) {
                printf("Error in fft\n");
            }
            
            fmDemod[bufferNum]->deEmphasis();
            rval = cufftExecC2R(mydiPlan, (cufftComplex *)ans_d, (cufftReal *)ans_d);
            if (rval) {
                printf("Error in fft\n");
            }
            
            cudaMemcpyAsync(&answer[bufferNum * NLARGE * 4], ans_d, 4 * NLARGE*sizeof(float), cudaMemcpyDeviceToHost, myStream);
            cudaMemcpyAsync(&squelch[bufferNum * NTOTAL * 4], &squelch_d[bufferNum * NTOTAL * 4], 4 * NTOTAL * sizeof(float), cudaMemcpyDeviceToHost, myStream);
            cudaStreamAddCallback(myStream,
            [](cudaStream_t stream, cudaError_t status, void *data) {
                bpos->setPosition((long)data);
                //long position = bpos->getNow();
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
			mag[idx] = val;
		}
		socket->send(*identity, ZMQ_SNDMORE);
		std::string func("NO ERROR GETDATA");
		// Wrap this in proto buffers
		socket->send(func.c_str(), func.size(), ZMQ_SNDMORE);
		socket->send(&mag[0], NSMALL*4);
	};
}

// gm::zmqcode::func_t UHFFMDemod::getFreqs() {
// 	return [&](zmq::socket_t* socket, zmq::message_t* identity) {
// 		float startf = (160000000 - 12500000) / 1000000.0f;
// 		float stopf = (442000000 + 12500000 - 2014) / 1000000.0f;
// 		socket->send(*identity, ZMQ_SNDMORE);
// 		std::string func("NO ERROR GETFREQS");
// 		// Wrap this in proto buffers
// 		std::stringstream ss;
// 		ss << startf;
// 		std::string start_freq = ss.str();
// 		std::stringstream st;
// 		st << stopf;
// 		std::string stop_freq = st.str();

// 		socket->send(func.c_str(), func.size(), ZMQ_SNDMORE);
// 		socket->send(start_freq.c_str(), start_freq.size(), ZMQ_SNDMORE);
// 		socket->send(stop_freq.c_str(), stop_freq.size());
// 	};
// }

// gm::zmqcode::func_t UHFFMDemod::setFreqRange() {
// 	return [&](zmq::socket_t* socket, zmq::message_t* identity) {
// 		zmq::message_t message;
// 		int more;
// 		size_t more_size = sizeof (more);
// 		zmq::message_t start_message;
// 		zmq::message_t stop_message;

// 		socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
// 		float freqs[2];
// 		freqs[0] = 0;
// 		freqs[1] = 0;

// 		if (more) {
// 			socket->recv(&start_message);
// 			std::string startf = std::string(static_cast<char*>(start_message.data()), start_message.size());
// 			socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
// 			if (more) {
// 				socket->recv(&stop_message);
// 				std::string stopf = std::string(static_cast<char*>(stop_message.data()), stop_message.size());
// 				freqs[0] = atof(startf.c_str());
// 				freqs[1] = atof(stopf.c_str());
// 			}
// 		}
// 		socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
// 		socket->send(*identity, ZMQ_SNDMORE);
// 		std::string func("NO ERROR FREQRANGE");
// 		socket->send(func.c_str(), func.size());
// 	};
// }

}
}


