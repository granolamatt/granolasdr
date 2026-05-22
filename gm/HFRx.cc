#include <stdio.h>
#include <unistd.h>
#include <cstring>
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


int main(int argc, char* argv[]) {

    bool enable_corpus = false;
    bool use_gpu_ldpc  = false;
    std::string ctrl_host = "127.0.0.1";
    int ctrl_port = 8080;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--jtdx") == 0) {
            enable_corpus = true;
        } else if (strcmp(argv[i], "--gpu-ldpc") == 0) {
            use_gpu_ldpc = true;
        } else if (strcmp(argv[i], "--control-host") == 0 && i + 1 < argc) {
            ctrl_host = argv[++i];
        } else if (strcmp(argv[i], "--control-port") == 0 && i + 1 < argc) {
            ctrl_port = std::stoi(argv[++i]);
        }
    }

    gm::rx888::rx888 mydsp;
    mydsp.start_card();

    gm::cuda::HFChannelizer epochbuffer(mydsp.getRxBufferPosition(), ctrl_host, ctrl_port);
    epochbuffer.start();

    gm::cuda::FT8Cuda ft8channel(epochbuffer.getBuffer(), enable_corpus);
    ft8channel.start();

    gm::hf::FT8 ft8(ft8channel.getBuffer(), &ft8channel, 5580, use_gpu_ldpc);

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
