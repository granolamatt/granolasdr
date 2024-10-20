
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include "gm/rx888/rx888.h"
#include "gm/zmqcode/zmqserver.h"
#include "gm/zmqcode/zmqworker.h"

int testBlade() {
    gm::rx888::rx888 mydsp;
    mydsp.start_card();

    gm::buffer::BufferPosition<short>* bp = mydsp.getRxBufferPosition();
    

    long ask = 10000000L;
    printf("Asking now\n");
    struct timespec now;

    while (1) {
        bp->getPosition(ask);
	clock_gettime(CLOCK_REALTIME, &now);
        printf("Got the position %ld at %ld.%f\n", ask, now.tv_sec, (double)now.tv_nsec/1e9);
        ask += 140000000L;
    }

    printf("Program started\n");
    gm::zmqcode::zmqServer* myServer = new gm::zmqcode::zmqServer();
    myServer->start();
    printf("Server started\n");

    for (int cnt = 0; cnt < 10; cnt++) {
        gm::zmqcode::zmqWorker* worker = new gm::zmqcode::zmqWorker();
        myServer->add_poll_item(worker);
    }

    usleep(10000000);

    mydsp.stop_card();
    delete myServer;
    printf("Server is stopped");
    
//    unmap_shared_memory((char *)tuner_data, sizeof(RawTunerStruct));

    return 0;

}

int main(int argc, char *argv[])
{
    printf("Hello world\n");
    testBlade();
    return 0;
}
