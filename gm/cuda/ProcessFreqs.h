#ifndef _GM_CUDA_PROCESSFREQS_
#define _GM_CUDA_PROCESSFREQS_

#include <string>
#include <zmq.hpp>
#include <complex.h>
#include "gm/buffer/BufferPosition.h"
#include "gm/cuda/CudaCopy.h"
#include "gm/zmqcode/zmqworker.h"


namespace gm {
namespace cuda {

class ProcessFreqs {
public:
	ProcessFreqs(gm::buffer::BufferPosition<std::complex<float>>* inP);
	~ProcessFreqs();
	void addToMap(std::string name);

private:
	gm::zmqcode::func_t processSamples();
	gm::buffer::BufferPosition<std::complex<float>>* inputPosition;
	std::complex<float>* samples;
};

}
}

#endif //_GM_CUDA_PROCESSFREQS_