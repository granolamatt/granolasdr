#include <string.h>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <zmq.hpp>
#include "gm/cuda/SumFreqs.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/zmqcode/zmqworker.h"
#include "gm/cuda/FreqScan.h"
#include "gm/Thread.h"

namespace gm {
namespace cuda {

gm::zmqcode::func_t SumFreqs::processSamples() {
	return [&](zmq::socket_t* socket, zmq::message_t* identity) {
		// We have 4096 bins and it is incrementing at 15Mhz.  This means 1280 bins on each
		const int buffSize = gm::cuda::FreqScan::NSMALL;

		float mag[4000];
		synchronized(*this, [&] {
			float totalBW = getStopFreq() - getStartFreq();
			float stepf = (totalBW / 4000.0f) / (25000000.0f / (float)gm::cuda::FreqScan::NSMALL);
			//std::cout << "Step is " << stepf << " start " << getStartFreq()
			//<< " stop " << getStopFreq() << std::endl;

			int offset = gm::cuda::FreqScan::NSMALL / 2 - 2560 / 2;
			for (int cnt = 0; cnt < 4000; cnt++) {
				int index = (int)(stepf * cnt);
				int bufferNumber = (index) / 2560;
				int sampleNumber = (index) % 2560;
				float* buffer = &buffPos->getBuffer()[bufferNumber * gm::cuda::FreqScan::NSMALL];
				mag[cnt] = buffer[sampleNumber + offset];
			}
		});
		socket->send(*identity, ZMQ_SNDMORE);
		std::string func("NO ERROR GETDATA");
		// Wrap this in proto buffers
		socket->send(func.c_str(), func.size(), ZMQ_SNDMORE);
		socket->send(&mag[0], sizeof(mag));
	};
}

gm::zmqcode::func_t SumFreqs::getFreqs() {
	return [&](zmq::socket_t* socket, zmq::message_t* identity) {
		float startf = getStartFreq() / 1000000.0f;
		float stopf = getStopFreq() / 1000000.0f;
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

bool SumFreqs::isFreqsOK(float freqs[2]) {
	if (freqs[0] >= freqs[1]) {
		return false;
	}
	if (freqs[0] < 8) {
		return false;
	}
	if (freqs[1] > 3300) {
		return false;
	}
	return true;
}

gm::zmqcode::func_t SumFreqs::setFreqRange() {
	return [&](zmq::socket_t* socket, zmq::message_t* identity) {
		zmq::message_t message;
		int more;
		size_t more_size = sizeof (more);
		zmq::message_t start_message;
		zmq::message_t stop_message;
//		std::string::size_type sz;     // alias of size_t

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
				// try {
//				 	start_freq = std::stof (startf, &sz);
//				 	stop_freq = std::stof (stopf, &sz);
				// } catch (...) {

				// }
			}
		}
		if (!isFreqsOK(freqs)) {
			socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
			socket->send(*identity, ZMQ_SNDMORE);
			std::string func("INVALID FREQS");
			socket->send(func.c_str(), func.size());
			return;
		}
		//std::cout<<"Got a freq start " << startFreq << " stop " << stopFreq << std::endl;
		synchronized(*this, [&] {
			startFreq = freqs[0] * 1000000.0f;
			stopFreq = freqs[1] * 1000000.0f;
			calculateNumbers();
		});
		//socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
		socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
		socket->send(*identity, ZMQ_SNDMORE);
		std::string func("NO ERROR FREQRANGE");
		socket->send(func.c_str(), func.size());
	};
}

void SumFreqs::calculateNumbers() {
	totalFreq = stopFreq - startFreq;
	setTotalPasses(totalFreq / 15000000);
	if ((float)totalFreq / 15000000.0f > getTotalPasses()) {
		setTotalPasses(getTotalPasses() + 1);
	}
	if (getTotalPasses() > 200) {
		setTotalPasses(200);
	}
	skipped = 200 - getTotalPasses();
}

void SumFreqs::addDataToList(std::string getData, std::string freqRange) {
	gm::zmqcode::zmqWorker().getFuncMap().insert({getData, processSamples()});
	gm::zmqcode::zmqWorker().getFuncMap().insert({freqRange, setFreqRange()});
}

SumFreqs::SumFreqs(gm::buffer::BufferPosition<float>* inP) :
	startFreq(400000000), stopFreq(600000000), buffPos(inP)
{
	calculateNumbers();
	gm::zmqcode::zmqWorker().getFuncMap().insert({"getData", processSamples()});
	gm::zmqcode::zmqWorker().getFuncMap().insert({"freqRange", setFreqRange()});
	gm::zmqcode::zmqWorker().getFuncMap().insert({"getFreqs", getFreqs()});
}

SumFreqs::~SumFreqs() {

}

}
}