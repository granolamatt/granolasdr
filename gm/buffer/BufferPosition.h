#ifndef _H_GM_BUFFERPOSITION_
#define _H_GM_BUFFERPOSITION_

#include <complex>
#include <cmath>
#include "gm/Thread.h"

namespace gm {
namespace buffer {

//Need to templete this
template<class T>
class BufferPosition {
private:
    long buffPosition;
    int lastWait;
    bool running;
    T* buffer;
    int bufferSize;
public:
    BufferPosition() : lastWait(0), buffPosition(0), buffer(NULL), running(false) {}
    ~BufferPosition() {
        if (buffer) {
            free(buffer);
        }
    }
    void setPosition(long position) {
        buffPosition = position;
        running=true;
        notifyAll(*this);
    }
    void release() {
        notifyAll(*this);
    }
    bool isRunning() {
        return running;
    }
    void setRunning(bool nrun) {
        running=nrun;
    }
    long getNow() {
        return buffPosition;
    }
    int getLastWait() {
        return lastWait;
    }
    long getPosition(long desired) {
//	printf("Asked for %ld\n", desired);
        if (desired < 0) {
            printf("Can't be less than one\n");
            return -1L;
        }
        int waitCount = 0;
        while (buffPosition < desired) {
//		printf("Waiting on %ld\n", buffPosition);
            wait(*this);
            waitCount++;
        }
        if (waitCount == 0) {
            if (lastWait >= 100) {
                printf("Falling behind\n");
            }
            lastWait++;
        } else {
            lastWait = 0;
        }
        return buffPosition;
    }
    void setBuffer(T* buff, int size) {
        buffer = buff;
        bufferSize = size;
    }
    T* getBuffer() {
        return buffer;
    }
    int getBufferSize() {
        return bufferSize;
    }
    int getByteSize() {
        return sizeof(T)*bufferSize;
    }
    int getElementSize() {
        return sizeof(T);
    }
};

}
}


#endif //_H_GM_BUFFERPOSITION_