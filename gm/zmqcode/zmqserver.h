#ifndef GM_ZMQ_SERVER_H_
#define GM_ZMQ_SERVER_H_

#include <vector>
#include <zmq.hpp>
#include "gm/Thread.h"
#include "gm/zmqcode/zmqworker.h"

namespace gm {
namespace zmqcode {
class zmqServer : public gm::Thread {
private:
    zmq::context_t ctx_;
    zmq::socket_t frontend_;
    zmq::socket_t backend_;
    std::vector<zmqWorker *> workers;
    //pthread_t server;
    bool running;
    //static void* startServer(void* server);
public:
    zmqServer();
    ~zmqServer();
    virtual void run();
    void stop();
    int add_poll_item(zmqWorker* worker);
};
}
}


#endif //GM_ZMQ_SERVER_H_

