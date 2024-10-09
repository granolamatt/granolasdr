#include <vector>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <complex>
#include <unistd.h>
#include <zmq.hpp>
#include "gm/sound/SoundMonitor.h"
#include "gm/cuda/UHFFMDemod.h"
#include "gm/zmqcode/zmqworker.h"

namespace gm {
namespace sound {

SoundMonitor::SoundMonitor(gm::buffer::BufferPosition<float>* inP, float* fmsquelch, float* myave) : inputPos(inP), squelch(fmsquelch),ave(myave) {
 	gm::zmqcode::zmqWorker().getFuncMap().insert({"getActiveChannels", getActiveChannels()});   
}

SoundMonitor::~SoundMonitor() {
    
}

bool SoundMonitor::isChannel(SignalCheck* check, int bufferNum, int channel) {
    float sq = squelch[bufferNum * gm::cuda::UHFFMDemod::NTOTAL * 4];
    int overlap = 0;
    for (int sqt = 1; sqt < 4; sqt++) {
        float sqtest = squelch[bufferNum * gm::cuda::UHFFMDemod::NTOTAL * 4 + gm::cuda::UHFFMDemod::NTOTAL * sqt + channel];
        if (sqtest > sq) {
            sq = sqtest;
            overlap = sqt;
        }
    }
    check->squelch = sq;
    check->channel = channel;
    check->overlap = overlap;
    // Good squelch, now check adjacent channels
    if (check->squelch > 600) {
	//std::cout << "Good squelch " << check->squelch << std::endl;
        SignalCheck before;
        bool test_before = isChannel(&before, bufferNum, channel - 1);
        if (test_before) {
            if (before.squelch > check->squelch) {
                //std::cout << "Channel before was greater " << channel << std::endl;
                check->squelch = before.squelch;
                check->overlap = before.overlap;
                check->channel = before.channel;
            }
        }
        return true;
    }
    
    return false;
}

void SoundMonitor::run() {
    
    std::cout << "Sound Thread Started running " << isRunning() << std::endl;

    zmq::context_t context(1);
    std::vector<zmq::socket_t> publisher;
    const int numChannels = gm::cuda::UHFFMDemod::NCHANNELS > 500 ? 500 : gm::cuda::UHFFMDemod::NCHANNELS;
    
    for (int cnt = 0; cnt < numChannels; cnt++) {
        std::stringstream ss;
        ss << "tcp://*:" << (cnt+16000);
        std::string constring = ss.str();
        publisher.push_back(zmq::socket_t(context, ZMQ_PUB));
        publisher[cnt].bind(constring);
    }    
    
//    long start = inputPos->getNow();
    long start = 50;
    
    const int nTune = gm::cuda::UHFFMDemod::NTUNE;
    const int midOffset = (gm::cuda::UHFFMDemod::NLARGE / 2) - (numChannels / 2) * gm::cuda::UHFFMDemod::NTUNE;
    const int startingOffset = midOffset + nTune/4; // Starting channel, one fourth through
    float* data = inputPos->getBuffer();
    int count = 0;
    
    while(isRunning()) {
        inputPos->getPosition(start);
        int bufferNum = (int)(start % gm::cuda::UHFFMDemod::NSBUFFERS);
        int offset = bufferNum * gm::cuda::UHFFMDemod::NLARGE * 4  + gm::cuda::UHFFMDemod::NLARGE * 2 + startingOffset;
        float magreal = 0;
        float magimag = 0;
        int magchan = start % numChannels;
        for (int cnt = 0; cnt < numChannels; cnt++) {
            float sum = 0;
            float* mychannel = &data[offset + cnt*nTune];
            publisher[cnt].send(mychannel, sizeof(float) * nTune/2);

        }
        for (int cnt = 5; cnt < 2043; cnt++) {
            if (cnt != 1023) {
                SignalCheck sqcheck;
                
                // Add it to a vector or hashmap and start recording
                if (isChannel(&sqcheck, bufferNum, cnt)) {
                    // First check previous and adjacent channels
                    
                    bool found = false;
                 synchronized(waveset, [&] {
                    for(int cc = 0; cc < waveset.size(); cc++) {
                        FMSound* value = waveset[cc];
                        if (value->getChannelNum() == sqcheck.channel) {
                            found = true;
                            if (!value->isRunning()) {
                                std::cout << "!!!!!!!!!!!!!!! Channel " << sqcheck.channel << " is not longer running" << std::endl;
                                waveset.erase(waveset.begin() + cc);
                                delete value;
                            } else {
                                value->updateTime();
                            }
                            break;
                        }
                    }
                    if (!found) {
                        std::cout << "It was not found";
                        std::cout << "Tuner " << sqcheck.channel << " val " << sqcheck.squelch << ":" << sqcheck.overlap << std::endl;
                        FMSound* input = new FMSound(inputPos);
                        waveset.push_back(input);
                        waveset.back()->setChannel(sqcheck.channel);
                        waveset.back()->setOverlap(sqcheck.overlap);
                        waveset.back()->start();
                        waveset.back()->setRunning(true);
                    }
                });
                }
            }
        }
        synchronized(waveset, [&] {
            for(int cc = 0; cc < waveset.size(); cc++) {
                FMSound* value = waveset[cc];
                if (!value->isRunning()) {
                    std::cout << "!!!!!!!!!!!!!!! Channel " << cc << " is not longer running" << std::endl;
                    waveset.erase(waveset.begin() + cc);
                    delete value;
                }
            }
        });
        start++;
        count++;
    }
    
    std::cout << "Sound is done " << isRunning() << std::endl;
 }


gm::zmqcode::func_t SoundMonitor::getActiveChannels() {
	return [&](zmq::socket_t* socket, zmq::message_t* identity) {
	    
	    synchronized(waveset, [&] {
		socket->send(*identity, ZMQ_SNDMORE);
		
		std::stringstream st;
		
		st << std::setprecision(3);
		
		st << "{\"channels\":[";
		for (int cnt = 0; cnt < waveset.size(); cnt++) {
		    st << "{\"freq\":";
		    st << "\"" << waveset[cnt]->getFreqMHz() << "\"";
		    st << ",\"fname\":\"" << waveset[cnt]->getFileName();
		    st << "\"}";
		    if (cnt < waveset.size() - 1) {
		        st << ",";
		    }
		}
		st << "]}";

//        std::cout << "Sending " << st.str() << std::endl;
		// Wrap this in proto buffers
        std::string func("NO ERROR GETACTIVECHANNELS");
        socket->send(func.c_str(), func.size(), ZMQ_SNDMORE);
        std::string signals = st.str();
        socket->send(signals.c_str(), signals.size());
        
	    });
	};
}


}
}
