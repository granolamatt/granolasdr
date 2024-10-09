#include <stdio.h>
#include <unistd.h>
#include <complex>
#include <cmath>
#include <iostream>
#include <sstream>

#include "gm/bladerf/bladerf.h"
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/cuda/HostCuda.h"
#include "gm/cuda/CopyBuffer.h"
#include "gm/cuda/SumFreqs.h"
#include "gm/cuda/UHFFMDemod.h"
#include "gm/zmqcode/zmqserver.h"
#include "gm/zmqcode/zmqworker.h"
//#include "gm/zmqcode/ZMQPub.h"
#include "gm/sound/SoundMonitor.h"


int main() {

    gm::bladerf::bladeRF myblade;
    myblade.setRxSampleRate(25000000);
    myblade.start_card();

//    myblade.change_rx_freq(162000000 + 2000);
//    myblade.change_rx_freq(440000000);
    myblade.change_rx_freq(150000000);
    
    gm::cuda::CopyBuffer cpybuffer(myblade.getRxBufferPosition());
    cpybuffer.setSize(gm::cuda::UHFFMDemod::NEPOCH * gm::cuda::UHFFMDemod::NSBUFFERS);
    gm::Thread nvcpy(cpybuffer);
    nvcpy.start();

    cpybuffer.getOutputBufferPos()->getPosition(10000000);

    gm::cuda::UHFFMDemod myhost(cpybuffer.getOutputBufferPos());
    gm::Thread tester(myhost);
    tester.start();

    
    // Make sure it is started
    myhost.getOutputBufferPos()->getPosition(10);

    gm::sound::SoundMonitor fmsound(myhost.getOutputBufferPos(), myhost.getSquelchData(), myhost.getAveData());
    gm::Thread soundthread(fmsound);
    soundthread.start();
    std::cout<<"FMSound is running" << std::endl;
    
    // gm::sound::FMSound fmsound(myhost.getOutputBufferPos());
    // gm::Thread soundthread(fmsound);
    // soundthread.start();
    // std::cout<<"FMSound is running" << std::endl;

    // gm::cuda::UHFFMDemod myhost(myblade.getRxBufferPosition());
    // gm::Thread tester(&myhost);
    // tester.start();
    
    gm::zmqcode::zmqServer* myServer = new gm::zmqcode::zmqServer();
    myServer->start();
    printf("Server started\n");

    for (int cnt = 0; cnt < 4; cnt++) {
        gm::zmqcode::zmqWorker* worker = new gm::zmqcode::zmqWorker();
        myServer->add_poll_item(worker);
    }

    gm::zmqcode::zmqWorker().getFuncMap().insert({"Tester", [](zmq::socket_t* socket, zmq::message_t* identity) {
        zmq::message_t message;
        int more;
        size_t more_size = sizeof (more);
        socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);;
        while (more) {
            socket->recv(&message);
            socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
        }
        socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
        std::cout << "Hello Again " << more << std::endl;
        socket->send(*identity, ZMQ_SNDMORE);
        std::string func("NO ERROR");
        socket->send(func.c_str(), func.size());

    }
                                                 });

    std::cout << "Done with setup\n" << std::endl;

    while (true) {
//        myhost.capture();
        usleep(1000000);
    }

    return 0;
}
