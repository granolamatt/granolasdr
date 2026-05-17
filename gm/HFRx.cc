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
#include "gm/cuda/FT8Cuda.h"
#include "gm/hf/ft8.h"


int main() {

    gm::rx888::rx888 mydsp;
    mydsp.start_card();

    gm::cuda::HFChannelizer epochbuffer(mydsp.getRxBufferPosition());
    epochbuffer.start();

    gm::cuda::FT8Cuda ft8channel(epochbuffer.getBuffer());
    ft8channel.start();

    gm::hf::FT8 ft8(ft8channel.getBuffer(), &ft8channel);
    ft8.start();

    while (true) {
        usleep(1000000);
        // printf("pos is %lu \n", bpos->getNow(1));
        // uint64_t posnow = bpos->getPosition(now, 1);
	    // uint64_t azpos = bpos->getNow();
        // now += 1;
        // printf("Got a new position %lu now %lu element %f\n", posnow, now, (double)azpos / 140e6);
    }

    return 0;
}
