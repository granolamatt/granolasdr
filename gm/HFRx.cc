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
#include "gm/cuda/HFChannelizer.h"
#include "gm/hf/ft8.cc"


int main() {

    gm::rx888::rx888 mydsp;
    mydsp.start_card();

    gm::cuda::HFChannelizer epochbuffer(mydsp.getRxBufferPosition());

    gm::Thread nvcpy(epochbuffer);
    nvcpy.start();
    // wait until at least an epoch has gone by
    // gm::buffer::BufferPosition<int16_t>* bpos = epochbuffer.getOutputBufferPos();
    // bpos->getPosition(1,1);

    gm::hf::FT8 ft8(epochbuffer.getBuffer());
    gm::Thread nvft8(ft8);
    nvcpy.start();

    while (true) {
        usleep(1000000);
        // uint64_t posnow = bpos->getPosition(now, 1);
	    // uint64_t azpos = bpos->getNow();
        // now += 1;
        // printf("Got a new position %lu now %lu element %f\n", posnow, now, (double)azpos / 140e6);
    }

    return 0;
}
