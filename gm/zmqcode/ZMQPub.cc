#include <vector>
#include <sstream>
#include <iostream>
#include <complex>
#include <unistd.h>
#include <zmq.hpp>
#include "gm/zmqcode/ZMQPub.h"
#include "gm/cuda/UHFFMDemod.h"

namespace gm {
namespace zmqcode {

ZMQPub::ZMQPub(gm::buffer::BufferPosition<float>* inP, float* fmsquelch, float* myave) : inputPos(inP), squelch(fmsquelch),ave(myave) {
    
}

ZMQPub::~ZMQPub() {
    
}

void ZMQPub::run() {
    
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
            //if (cnt == magchan) {
                
            // for (int cc = 0; cc < nTune/2; cc++) {
            //     magreal += abs(mychannel[cc]);
            //     // magreal += mychannel[cc].real();
            //     // magimag += mychannel[cc].imag();
            // }
            //}

        }
        for (int cnt = 5; cnt < 2043; cnt++) {
            // float magg = abs(ave[bufferNum * gm::cuda::UHFFMDemod::NSMALL + 2*cnt] 
            // - ave[bufferNum * gm::cuda::UHFFMDemod::NSMALL + 2*cnt - 1]
            // - ave[bufferNum * gm::cuda::UHFFMDemod::NSMALL + 2*cnt + 1]);
            float sq = squelch[bufferNum * gm::cuda::UHFFMDemod::NTOTAL * 4 + gm::cuda::UHFFMDemod::NTOTAL * 2 + cnt];
            float sq1 = squelch[bufferNum * gm::cuda::UHFFMDemod::NTOTAL * 4 + gm::cuda::UHFFMDemod::NTOTAL * 1 + cnt];
            float sq2 = squelch[bufferNum * gm::cuda::UHFFMDemod::NTOTAL * 4 + cnt];
            float sq3 = squelch[bufferNum * gm::cuda::UHFFMDemod::NTOTAL * 4 + gm::cuda::UHFFMDemod::NTOTAL * 3 + cnt];
            //if (magg > 8) {
                // float sig = squelch[bufferNum * gm::cuda::UHFFMDemod::NTOTAL * 4 + gm::cuda::UHFFMDemod::NTOTAL * 2 + cnt];
                // float sig2 = squelch[bufferNum * gm::cuda::UHFFMDemod::NTOTAL * 4 + cnt];
                // float sig3 = squelch[bufferNum * gm::cuda::UHFFMDemod::NTOTAL * 4 + gm::cuda::UHFFMDemod::NTOTAL + cnt];
                // float sig4 = squelch[bufferNum * gm::cuda::UHFFMDemod::NTOTAL * 4 + gm::cuda::UHFFMDemod::NTOTAL * 3 + cnt];
                // Add it to a vector or hashmap and start recording
                 if (sq > 100)
                     std::cout << "Tuner " << cnt << " val " << sq << ":" << sq1 << ":" << sq2 << ":" << sq3 << std::endl;
            //}
        }
        //magreal = squelch[bufferNum * gm::cuda::UHFFMDemod::NTOTAL * 4 + gm::cuda::UHFFMDemod::NTOTAL * 2 + magchan + 1024 - 250];
        // magreal = abs(ave[bufferNum * gm::cuda::UHFFMDemod::NSMALL + 2*magchan + 2048 - 500] 
        //     - ave[bufferNum * gm::cuda::UHFFMDemod::NSMALL + 2*magchan + 2048 - 500 - 1]
        //     - ave[bufferNum * gm::cuda::UHFFMDemod::NSMALL + 2*magchan + 2048 - 500 + 1]);
        //if (magreal < 1e6)
        //    std::cout << "Pos " << inputPos->getNow() << " on " << start << " Magchan " << magchan << " :" << magreal << std::endl;
        start++;
        count++;
    }
    
    std::cout << "Sound is done " << isRunning() << std::endl;
 }



}
}
