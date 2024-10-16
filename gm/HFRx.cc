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

    gm::buffer::BufferPosition<int16_t>* bpos = mydsp.getRxBufferPosition();
    
    // gm::cuda::CopyBuffer<int16_t> cpybuffer(mydsp.getRxBufferPosition());
    // cpybuffer.setSize(gm::cuda::UHFFMDemod::NEPOCH * gm::cuda::UHFFMDemod::NSBUFFERS);
    // gm::Thread nvcpy(cpybuffer);
    // nvcpy.start();
    // // wait until at least an epoch has gone by
    // cpybuffer.getOutputBufferPos()->getPosition(gm::cuda::UHFFMDemod::NEPOCH);

    // Now we need to channelize the HF and put it into eight bands
    uint64_t now = 1;

    while (now < 10000) {
        uint64_t posnow = bpos->getPosition(now);
        now += 1;
        printf("Got a new position %lu now %lu\n", posnow, now);
    }

    return 0;
}
