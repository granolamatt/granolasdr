#include <unistd.h>
#include <iostream>
#include <complex>
#include <cmath>

#include "gm/bladerf/bladerf.h"
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/cuda/CopyBuffer.h"

int main() {

    int16_t *rxBuffer = (int16_t*)calloc(10,sizeof(int16_t));
    
    gm::buffer::BufferPosition<int16_t> vectorBufferPosition;
    gm::buffer::BufferPosition<int16_t> rxBufferPosition;

    int16_t vdata[1000];
    rxBufferPosition.setBuffer(rxBuffer, 10);
    vectorBufferPosition.setBuffer(vdata, {10,100});

    for (int cnt=0; cnt<10; cnt++) rxBuffer[cnt] = cnt;

    // printf("Buffer size is %ld\n", rxBuffer.size());
    printf("Element size is %ld buffer size %ld byte size %ld\n", rxBufferPosition.getElementSize(), 
                rxBufferPosition.getBufferSize(), rxBufferPosition.getByteSize());
    printf("Element size is %ld buffer size %ld byte size %ld\n", vectorBufferPosition.getElementSize(), 
                vectorBufferPosition.getBufferSize(), vectorBufferPosition.getByteSize());


    int16_t* backing = rxBufferPosition.getBuffer();
    for (int cnt=0; cnt< rxBufferPosition.getBufferSize(); cnt++) {
        printf("Value %ld is %d\n", cnt, backing[cnt]);
    }
    printf("Sleeping\n");
    usleep(1000000);

    free(rxBuffer);

    
    // gm::cuda::CopyBuffer myhost(myblade.getRxBufferPosition());
    // gm::Thread tester(myhost);
    // tester.start();
    
}
