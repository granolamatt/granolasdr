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
#include "gm/cuda/CopyEpoch.h"


int main() {

    gm::rx888::rx888 mydsp;
    mydsp.start_card();

    gm::cuda::CopyEpoch<int16_t> epochbuffer(mydsp.getRxBufferPosition());

    gm::Thread nvcpy(epochbuffer);
    nvcpy.start();
    // wait until at least an epoch has gone by
    gm::buffer::BufferPosition<int16_t>* bpos = epochbuffer.getOutputBufferPos();
    bpos->getPosition(1,1);

    // Now we need to channelize the HF and put it into eight bands
    uint64_t now = 1;

    while (now < 10000) {
        uint64_t posnow = bpos->getPosition(now, 1);
	    uint64_t azpos = bpos->getNow();
        now += 1;
        printf("Got a new position %lu now %lu element %f\n", posnow, now, (double)azpos / 140e6);
    }

    return 0;
}
