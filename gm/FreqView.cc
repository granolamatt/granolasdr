#include <stdio.h>
#include <unistd.h>
#include <complex>
#include <cmath>
#include <sstream>

#include "gm/bladerf/bladerf.h"
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/cuda/HostCuda.h"
#include "gm/cuda/OverlapSave.h"
#include "gm/cuda/ProcessFreqs.h"
#include "gm/cuda/FreqView.h"
#include "gm/zmqcode/zmqserver.h"
#include "gm/zmqcode/zmqworker.h"



int main() {

     gm::bladerf::bladeRF myblade;
     myblade.start_card();

     gm::cuda::FreqView myhost(myblade.getRxBufferPosition());
     gm::Thread tester([&] {
        //printf("Tester here\n");
        // When tester falls out of scope the destructor will be called and stop the while loop
        while (tester.isRunning()) {
            myhost.capture();
        }
        printf("fell out of scope so told to stop\n");
    });
    tester.start();

     gm::cuda::ProcessFreqs processFreqs(myhost.getOutputBufferPos());
     processFreqs.addToMap("changeFreq");

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

    while (true) {
//        myhost.capture();
        usleep(1000000);
    }

    return 0;
}
