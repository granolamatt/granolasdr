
#include <stdio.h>
#include <unistd.h>
#include "gm/common/tuner_mem.h"
#include "gm/bladerf/bladerf.h"
#include "gm/zmqcode/zmqserver.h"
#include "gm/zmqcode/zmqworker.h"

int testBlade() {
    gm::bladerf::bladeRF myblade;
    myblade.start_card();

    gm::buffer::BufferPosition<std::complex<short>>* bp = myblade.getRxBufferPosition();
    
//    printf("Mapping memory to %s \n", "tuner_output\0");
//    RawTunerStruct *tuner_data = (RawTunerStruct *)map_shared_memory("tuner_output", sizeof(RawTunerStruct));
    RawTunerStruct *tuner_data = (RawTunerStruct *)calloc(1,sizeof(RawTunerStruct));
    if (!tuner_data)
    {
        printf("Could not allocate shared memory\n");
        exit(4);
    }
    strncpy(tuner_data->version, "version 0.1\0", STRINGSIZE);
    strncpy(tuner_data->description, "IQ Samples from bladeRF\0", STRINGSIZE);
    tuner_data->buffer_size = bp->getBufferSize();  // in samples
    tuner_data->frequency = (double)myblade.get_rx_freq();
    tuner_data->sample_rate = myblade.getRxSampleRate();
    tuner_data->current_sample = 0;
    for (int cnt = 0; cnt < BUFFERSIZE; cnt++)
    {
        tuner_data->sample_holder.samples[cnt].x = (short)0;
        tuner_data->sample_holder.samples[cnt].y = (short)0;
    }

    long ask = 10000000L;
    printf("Asking now\n");
    while (ask < 100000000) {
        bp->getPosition(ask);
        printf("Got the position %ld\n", ask);
        ask += 10000000L;
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

    myblade.stop_card();
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
