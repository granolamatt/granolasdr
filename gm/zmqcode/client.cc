#include <vector>
#include <thread>
#include <memory>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <functional>

#include <zmq.hpp>

#define within(num) (int) ((float) (num) * random () / (RAND_MAX + 1.0))

class client_task {
public:
    client_task()
        : ctx_(1),
          client_socket_(ctx_, ZMQ_DEALER)
    {}

//  Receives all message parts from socket, prints neatly
//
    static void
    s_dump (zmq::socket_t & socket)
    {
        std::cout << "----------------------------------------" << std::endl;

        while (1) {
            //  Process all parts of the message
            zmq::message_t message;
            socket.recv(&message);

            //  Dump the message as text or binary
            int size = message.size();
            std::string data(static_cast<char*>(message.data()), size);

            bool is_text = true;

            int char_nbr;
            unsigned char byte;
            for (char_nbr = 0; char_nbr < size; char_nbr++) {
                byte = data [char_nbr];
                if (byte < 32 || byte > 127)
                    is_text = false;
            }
            std::cout << "[" << std::setfill('0') << std::setw(3) << size << "]";
            for (char_nbr = 0; char_nbr < size; char_nbr++) {
                if (is_text)
                    std::cout << (char)data [char_nbr];
                else
                    std::cout << std::setfill('0') << std::setw(2)
                              << std::hex << (unsigned int) data [char_nbr];
            }
            std::cout << std::endl;

            int more = 0;           //  Multipart detection
            size_t more_size = sizeof (more);
            socket.getsockopt (ZMQ_RCVMORE, &more, &more_size);
            if (!more)
                break;              //  Last message part
        }
    }

    void start() {
        // generate random identity
        char identity[10] = {};
        sprintf(identity, "%04X-%04X", within(0x10000), within(0x10000));
        printf("%s\n", identity);
        client_socket_.setsockopt(ZMQ_IDENTITY, identity, strlen(identity));
        client_socket_.connect("tcp://localhost:5570");

        zmq::pollitem_t items[] = {{static_cast<void *>(client_socket_), 0, ZMQ_POLLIN, 0}};
        //std::vector<zmq::pollitem_t> items = {{static_cast<void *>(client_socket_), 0, ZMQ_POLLIN, 0}};
        int request_nbr = 0;
        try {
            while (true) {
                for (int i = 0; i < 100; ++i) {
                    // 10 milliseconds
                    zmq::poll(items, 1, 10);
                    if (items[0].revents & ZMQ_POLLIN) {
                        printf("\n%s ", identity);
                        s_dump(client_socket_);
                    }
                }
                std::string func("getData");
                client_socket_.send(func.c_str(), func.size(), ZMQ_SNDMORE);
                char request_string[16] = {};
                sprintf(request_string, "request #%d", ++request_nbr);
                client_socket_.send(request_string, strlen(request_string), ZMQ_SNDMORE);
                client_socket_.send(request_string, strlen(request_string));
            }
        }
        catch (std::exception &e) {}
    }

private:
    zmq::context_t ctx_;
    zmq::socket_t client_socket_;
};

int main() {
    client_task ct1;

    ct1.start();

    return 0;
}