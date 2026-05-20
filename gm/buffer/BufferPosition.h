#ifndef _H_GM_BUFFERPOSITION_
#define _H_GM_BUFFERPOSITION_

#include <complex>
#include <algorithm>
#include <cmath>
#include "gm/Thread.h"

namespace gm {
namespace buffer {

//Need to templete this
template<class T>
class BufferPosition {
private:
    using value_type = T;
    using pointer = T*;

    uint64_t buffPosition;
    int lastWait;
    bool running;
    pointer buffer;
    int bufferSize;
    std::vector<size_t> shape;
    std::vector<size_t> stride;
public:
    BufferPosition() : lastWait(0), 
                    buffPosition(0), 
                    buffer(NULL), 
                    running(false), 
                    shape({0}), 
                    stride({sizeof(value_type)}) {}
    ~BufferPosition() {
        // {100, 1000, 1000}, // shape
        // {1000*1000*8, 1000*8, 8}, // C-style contiguous strides for double
    }
    void setPosition(uint64_t position, int axis=0) {
        if (axis && axis < shape.size()) {
            // The buffer fills in last shape then moves up
            uint64_t nPosition = shape[shape.size()-1] * position;
            for (int cnt = 1; cnt < axis; cnt++) {
                nPosition *= shape[shape.size()-1-cnt];
            }
            auto& sync = getSynchro(*this);
            {
                std::lock_guard<std::mutex> lk(sync.mutex);
                if (nPosition != buffPosition) {
                    buffPosition = nPosition;
                    running = true;
                }
            }
            sync.cv.notify_all();
        } else {
            auto& sync = getSynchro(*this);
            {
                std::lock_guard<std::mutex> lk(sync.mutex);
                buffPosition = position;
                running = true;
            }
            sync.cv.notify_all();
        }
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
    uint64_t getNow(int axis=0) {
        if (axis && axis < shape.size()) {
            // The buffer fills in last shape then moves up
            int div = shape[shape.size()-1];
            for (int cnt = 1; cnt < axis; cnt++) {
                div *= shape[shape.size()-1-cnt];
            }
            return buffPosition / div;
        }
        return buffPosition;
    }
    int getLastWait() {
        return lastWait;
    }

    uint64_t getPosition(uint64_t desired_o, int axis=0) {
        uint64_t desired = desired_o;
        if (axis && axis < shape.size()) {
            // The buffer fills in last shape then moves up
            desired = shape[shape.size()-1] * desired_o;
            for (int cnt = 1; cnt < axis; cnt++) {
                desired *= shape[shape.size()-1-cnt];
            }

        }
        int waitCount = 0;
        auto& sync = getSynchro(*this);
        {
            std::unique_lock<std::mutex> lk(sync.mutex);
            while (buffPosition < desired) {
                sync.cv.wait(lk);
                waitCount++;
            }
        }
        if (waitCount == 0) {
            if (lastWait >= 100) {
                fprintf(stderr, "Falling behind\n");
            }
            lastWait++;
        } else {
            lastWait = 0;
        }
        return getNow(axis);
    }
    
    void setBuffer(pointer buff, size_t size) {
        buffer = buff;
        shape = {size};
        stride = {sizeof(value_type)};
    }
    void setBuffer(pointer buff, std::vector<size_t> shape) {
        buffer = buff;
        this->shape = shape;
        stride.clear();
        size_t lastshape = sizeof(value_type);
        stride.push_back(lastshape);

        for (int cnt = 1; cnt < shape.size(); cnt++) {
            lastshape *= shape[shape.size() - cnt];
            stride.push_back(lastshape);
            std::rotate(stride.begin(), stride.begin() + 1, stride.end());
        }
    }

    pointer getBuffer() {
        return buffer;
    }
    std::vector<size_t> getShape() {
        return shape;
    }

    std::vector<size_t> getStride() {
        return stride;
    }

    size_t getBufferSize() {
        size_t ret = 0;
        if (shape.size()) {
            ret = shape[0];
            for (int cnt=1; cnt < shape.size(); cnt++) {
                ret *= shape[cnt];
            }
        }
        return ret;
    }
    size_t getByteSize() {
        size_t ret = 0;
        if (shape.size()) {
            ret = shape[0]*sizeof(value_type);
            for (int cnt=1; cnt < shape.size(); cnt++) {
                ret *= shape[cnt];
            }
        }
        return ret;
    }
    size_t getElementSize() {
        return sizeof(value_type);
    }
};

}
}


#endif //_H_GM_BUFFERPOSITION_
