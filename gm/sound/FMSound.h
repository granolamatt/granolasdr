#ifndef _GM_SOUND_FMSOUND_H_
#define _GM_SOUND_FMSOUND_H_

#include <string>
#include <functional>
#include <complex>
#include <time.h>
#include "gm/Thread.h"
#include "gm/buffer/BufferPosition.h"

namespace gm {
namespace sound {

class FMSound : public Thread {
public :
    FMSound(gm::buffer::BufferPosition<float>* inP);
    ~FMSound();
    void run();
    void setChannel(int start) {channelNum = start; setFilename();}
    void setOverlap(int over) {overlap = over; setFilename();}
    void updateTime();
    int getChannelNum() {
        return channelNum;
    }
    int getOverlap() {
        return overlap;
    }
    float getFreqMHz() {
        return (channelNum * (25.0f/2048.0f)) - 25.0f/2.0f;
    }
    std::string getFileName() {
        return filename;
    }
    // bool operator==(const FMSound &other) const;
    // std::size_t operator()(const FMSound& k) const;
private :
    std::string filename;
    int channelNum;
    int overlap;
    gm::buffer::BufferPosition<float>* inputPos;
    int bufferSize;
    int position;
    bool bufferFull;
    short* soundBuffer;
    struct timespec lupdate;
    const static int timeout = 60;
    bool checkTime();
    void writeFile();
    void setFilename();
};

}
}

// // custom specialization of std::hash can be injected in namespace std
// namespace std
// {
//     template<> struct hash<gm::sound::FMSound>
//     {
//         typedef gm::sound::FMSound argument_type;
//         typedef std::size_t result_type;
//         result_type operator()(argument_type const& s) const
//         {
//             // result_type const h1 ( std::hash<int>()(s.getOverlap()) );
//             // result_type const h2 ( std::hash<int>()(s.getChannelNum()) );
//             // return h1 ^ (h2 << 1);
//             return s.operator()(s);
//         }
//     };
// }


#endif // _GM_SOUND_FMSOUND_H_