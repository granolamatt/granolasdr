#include <string.h>
#include <iostream>
#include <stdio.h>
#include <zmq.hpp>
#include <complex.h>
#include "gm/cuda/ProcessFreqs.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/zmqcode/zmqworker.h"
#include "gm/cuda/FreqView.h"

namespace gm {
namespace cuda {

gm::zmqcode::func_t ProcessFreqs::processSamples() {
	return [&](zmq::socket_t* socket, zmq::message_t* identity) {
		const int buffSize = gm::cuda::FreqView::NLARGE;
		int pos = (int)(inputPosition->getNow()
		                % gm::cuda::FreqView::NSBUFFERS ) * buffSize;
		long epoch = inputPosition->getNow();
		float mag[1024];
		//std::cout << "On epoch " << epoch << " sending " << sizeof(mag) << std::endl;
		std::complex<float>* buffer = &inputPosition->getBuffer()[pos];

		memset(&mag[0], 0, sizeof(mag));

		for (int cnt = 0; cnt < buffSize; cnt++) {
			float val = std::abs(buffer[cnt]);
			int idx =  cnt / 1024;
			mag[idx] += val;
		}
		//std::cout << "here" << std::endl;
		socket->send(*identity, ZMQ_SNDMORE);
		std::string func("NO ERROR");
		// Wrap this in proto buffers
		socket->send(func.c_str(), func.size(), ZMQ_SNDMORE);
		socket->send(&mag[0], sizeof(mag));
	};
}

void ProcessFreqs::addToMap(std::string name) {
	gm::zmqcode::zmqWorker().getFuncMap().insert({name, processSamples()});
}


ProcessFreqs::ProcessFreqs(gm::buffer::BufferPosition<std::complex<float>>* inP)
{
	inputPosition = inP;
	
}

ProcessFreqs::~ProcessFreqs() {

}

}
}