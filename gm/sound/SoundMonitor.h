#ifndef _GM_SOUND_SOUNDMONITOR_H_
#define _GM_SOUND_SOUNDMONITOR_H_

//#include <zmq.hpp>
#include <unordered_set>
#include <complex>
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/sound/FMSound.h"
#include "gm/zmqcode/zmqworker.h"

namespace gm {
namespace sound {

struct SignalCheck {
    float squelch;
    int channel;
    int overlap;
};

class SoundMonitor : public Runnable {
public :
    SoundMonitor(gm::buffer::BufferPosition<float>* inP, float* fmsquelch, float* ave);
    ~SoundMonitor();
    void run();
    bool isChannel(SignalCheck* check, int bufferNum, int channel);
private :
    gm::buffer::BufferPosition<float>* inputPos;
    float* squelch;
    float* ave;
    // std::unordered_set <FMSound> waveset;
    std::vector<FMSound*> waveset;
    gm::zmqcode::func_t getActiveChannels();
};

}
}

#endif //_GM_SOUND_SOUNDMONITOR_H_