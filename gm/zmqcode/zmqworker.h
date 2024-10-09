#ifndef _GM_ZMQ_WORKER_H
#define _GM_ZMQ_WORKER_H

#include <iostream>
#include <mutex>
#include <unordered_map>
#include <zmq.hpp>
//#include <initializer_list>

#include "gm/Thread.h"

namespace gm {
namespace zmqcode {
//typedef std::function<void(zmq::socket_t* socket, std::initializer_list<zmq::message_t> message)> func_t;
typedef std::function<void(zmq::socket_t* socket, zmq::message_t* identity)> func_t;
class zmqWorker : public Thread {
private:
	static std::unordered_map<std::string, func_t> & FuncMap () {
		static std::unordered_map<std::string, func_t> func_;
		return func_;
	}

//	std::function <void()>work_function;
	zmq::context_t* ctx_;
	zmq::socket_t* worker_;
	std::mutex workmutex;
	void doWork();
	void run();
	
public:
	zmq::socket_t* getSocket() {return worker_;}
	zmqWorker();
	~zmqWorker();
	int connect(zmq::context_t* ctx, int sock_type);
	std::unordered_map<std::string, func_t> & getFuncMap () {  return FuncMap();  }

};
}
}
#endif //_GM_ZMQ_WORKER_H