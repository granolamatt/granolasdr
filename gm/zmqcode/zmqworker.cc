#include <iostream>
//#include <sstream>
#include <stdio.h>
#include <unistd.h>
#include <zmq.hpp>
#include "gm/zmqcode/zmqworker.h"

namespace gm {
namespace zmqcode {
zmqWorker::zmqWorker() :
    ctx_(NULL),
    worker_(NULL)
{

}

zmqWorker::~zmqWorker() {
    if (worker_) {
        delete worker_;
    }
}

// void zmqWorker::getMessage() {

// }

void zmqWorker::doWork() {
    try {
        while (isRunning()) {
            // zmq::message_t copied_id;
            // zmq::message_t copied_msg;
//            std::lock_guard<std::mutex> lock(workmutex);
            zmq::message_t identity;
            zmq::message_t func;
//            std::cout << "In the lock" << std::endl;
            worker_->recv(&identity);
            int more;
            size_t more_size = sizeof (more);
            worker_->getsockopt(ZMQ_RCVMORE, &more, &more_size);
            if (more) {
                worker_->recv(&func);
                std::string func_name = std::string(static_cast<char*>(func.data()), func.size());
                //std::cout << "Got a string " << func_name << std::endl;
                func_t dofunc = NULL;
                synchronized(getFuncMap(), [&] {
                    dofunc = getFuncMap()[func_name];
                });
                if (dofunc) {
                    //std::cout << "function is not null" << std::endl;
                    dofunc(worker_, &identity);
                } else {
                    //std::cout << "function is null" << std::endl;
                    std::string error("ERROR NOT FOUND");
                    worker_->send(identity, ZMQ_SNDMORE);
                    worker_->send(error.c_str(), error.size());
                }
            }

            // worker_->recv(&msg);
            // for (int reply = 0; reply < 5; reply++) {
            //     //usleep(1000000);
            //     copied_id.copy(&identity);
            //     copied_msg.copy(&msg);
            //     worker_->send(copied_id, ZMQ_SNDMORE);
            //     worker_->send(copied_msg);
            // }
        }
    }
    catch (std::exception &e) {}
}

void zmqWorker::run() {
    if (worker_) {
        printf("Worker started\n");
        worker_->connect("inproc://backend");
        doWork();
    }
}

int zmqWorker::connect(zmq::context_t* ctx, int sock_type) {
    if (worker_) {
        delete worker_;
    }
    worker_ = new zmq::socket_t(*ctx, sock_type);
    start();
    return 0;
}

}
}