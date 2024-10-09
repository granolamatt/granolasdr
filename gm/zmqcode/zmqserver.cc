#include <vector>
#include <string>
#include <stdio.h>
#include <unistd.h>

#include <zmq.hpp>
#include "gm/Thread.h"
#include "gm/zmqcode/zmqserver.h"

namespace gm {
namespace zmqcode {
zmqServer::zmqServer()
    : ctx_(1),
      frontend_(ctx_, ZMQ_ROUTER),
      backend_(ctx_, ZMQ_DEALER),
      running(false)
{
}

void zmqServer::stop() {
    if (running) {
        printf("Stopping the proxy\n");
        frontend_.close();
        backend_.close();
        join();
    }
}

zmqServer::~zmqServer() {
    stop();
    while(workers.size()) {
        zmqWorker* del = workers.back();
        workers.pop_back();
        delete del;
    }
}

void zmqServer::run() {
    running = true;
    printf("Starting ZMQ Proxy\n");
    frontend_.bind("tcp://*:15070");
    // std::string ss("tester");
    // frontend_.setsockopt(ZMQ_IDENTITY, ss.c_str(), ss.length());
    backend_.bind("inproc://backend");
    try {
        zmq::proxy(static_cast<void *>(frontend_), static_cast<void *>(backend_), NULL);
    }
    catch (std::exception &e) {
        printf("Exception happened in proxy\n");
    }
    printf("ZMQ Proxy Finished\n");
    running = false;
    // std::vector<server_worker *> worker;
    // std::vector<std::thread *> worker_thread;
}



int zmqServer::add_poll_item(zmqWorker* worker) {
    int err = worker->connect(&ctx_, ZMQ_DEALER);
    workers.push_back(worker);
    return 0;
}

}
}
