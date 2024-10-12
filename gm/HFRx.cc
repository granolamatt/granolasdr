#include <stdio.h>
#include <unistd.h>
#include <complex>
#include <cmath>
#include <iostream>
#include <sstream>

#include "gm/rx888/rx888.h"
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/cuda/HostCuda.h"
#include "gm/cuda/CopyBuffer.h"
#include "gm/cuda/UHFFMDemod.h"


int main() {

    gm::rx888::rx888 mydsp;
    mydsp.start_card();
    
    gm::cuda::CopyBuffer<int16_t> cpybuffer(mydsp.getRxBufferPosition());
    cpybuffer.setSize(gm::cuda::UHFFMDemod::NEPOCH * gm::cuda::UHFFMDemod::NSBUFFERS);
    gm::Thread nvcpy(cpybuffer);
    nvcpy.start();
    // wait until at least an epoch has gone by
    cpybuffer.getOutputBufferPos()->getPosition(gm::cuda::UHFFMDemod::NEPOCH);

    // Now we need to channelize the HF and put it into eight bands


    // gm::cuda::UHFFMDemod myhost(cpybuffer.getOutputBufferPos());
    // gm::Thread tester(myhost);
    // tester.start();
    
    // // Make sure it is started
    // myhost.getOutputBufferPos()->getPosition(10);

    // gm::sound::SoundMonitor fmsound(myhost.getOutputBufferPos(), myhost.getSquelchData(), myhost.getAveData());
    // gm::Thread soundthread(fmsound);
    // soundthread.start();
    // std::cout<<"FMSound is running" << std::endl;

    while (true) {
        usleep(1000000);
    }

    return 0;
}
