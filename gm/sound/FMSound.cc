#include <string.h>
#include <iostream>
#include <complex>
#include <unistd.h>
#include "gm/sound/FMSound.h"
#include "gm/cuda/UHFFMDemod.h"
#include "gm/wavfile/wavfile.h"

namespace gm {
namespace sound {

FMSound::FMSound(gm::buffer::BufferPosition<float>* inP) : 
    inputPos(inP), position(0), bufferFull(false), soundBuffer(NULL) {
    float epochTime = (float)gm::cuda::UHFFMDemod::NEPOCH / 25e6f;
    float numEpochBuffer = 300.0f / epochTime;
    bufferSize = (int)(numEpochBuffer * gm::cuda::UHFFMDemod::NTUNE/2);
    soundBuffer = (short*)calloc(bufferSize, sizeof(short));
    updateTime();
}



FMSound::~FMSound() {
    if (soundBuffer) {
        free(soundBuffer);
    }
}

void FMSound::updateTime() { 
    clock_gettime(CLOCK_REALTIME, &lupdate);
}

bool FMSound::checkTime() { 
    struct timespec tcheck;
    clock_gettime(CLOCK_REALTIME, &tcheck);
    return tcheck.tv_sec > timeout + lupdate.tv_sec;
}

void FMSound::setFilename() {

    filename = "chan" 
        + std::to_string(channelNum)
        + "over"
        + std::to_string(overlap) + ".wav";
}

void FMSound::run() {
    std::cout << "Sound Thread Started running " << isRunning() << std::endl;
    
    long start = inputPos->getNow();
    //long start = 50;
    const int numChannels = gm::cuda::UHFFMDemod::NCHANNELS * 4;
    position = 0;
    
    const int nTune = gm::cuda::UHFFMDemod::NTUNE;
    const int startingOffset = nTune/4; // Starting channel, one fourth through
    float* data = inputPos->getBuffer();
    int count = 0;

    while(isRunning()) {
        //std::cout << "In the loop " << start << std::endl;
        if (checkTime()) {
            std::cout << "Timeout passed, signal is done" << std::endl;
            // Now run the callback
            break;
        }
        //std::cout << "Getting position " << start << std::endl;
        inputPos->getPosition(start);
        int bufferNum = (int)(start % gm::cuda::UHFFMDemod::NSBUFFERS);
        int offset = bufferNum * gm::cuda::UHFFMDemod::NLARGE * 4  + gm::cuda::UHFFMDemod::NLARGE * overlap + startingOffset;
        float sum = 0;
        //std::cout << "Reading data" << std::endl;
        float* mychannel = &data[offset + channelNum*nTune];
        for (int cc = 0; cc <  nTune/2; cc++) {
            soundBuffer[position] = (short)(mychannel[cc] * 8.0f);
            position++;
            if (position >= bufferSize) {
                bufferFull = true;
                position = 0;
            }
        }
        if (count > 300) {
            count = 0;
            writeFile();
        }
        //std::cout << "Pos " << inputPos->getNow() << " on " << bufferNum << std::endl;
        start++;
        count++;
    }
    writeFile();
    std::cout << "Sound is done " << getChannelNum() << std::endl;
    setRunning(false);
}

void FMSound::writeFile() {
	std::string fullname = "/home/matt/sounds/" + filename;
	//std::string fullname = "/srv/http/usr/share/nginx/html/sound/" + filename;
	std::cout << "Writing to file " << fullname << std::endl;
	FILE * f = wavfile_open(fullname.c_str());
	if(!f) {
		printf("couldn't open sound.wav for writing: %s",strerror(errno));
	}
	
	std::cout << "File open" << std::endl;
	if (bufferFull) {
	    std::cout << "Buffer is full" << std::endl;
	    int pos = 0;
	    short tempBuffer[bufferSize];
	    memcpy(&tempBuffer[0], &soundBuffer[position], bufferSize - position);
	    memcpy(&tempBuffer[bufferSize - position], &soundBuffer[0], position);
	    wavfile_write(f,tempBuffer,bufferSize);
	} else {
	    wavfile_write(f,soundBuffer,position);
	}
	wavfile_close(f);

}

// bool FMSound::operator==(const FMSound &other) const {
//     //std::cout << "Equals called " << other.inputPos << std::endl;
//     return (channelNum == other.channelNum
//             && overlap == other.overlap);
// }

// std::size_t FMSound::operator()(const FMSound& k) const {
//       using std::size_t;
//       using std::hash;
//       using std::string;

//       // Compute individual hash values for first,
//       // second and third and combine them using XOR
//       // and bit shifting:

//       return hash<int>()(k.overlap)
//               ^ (hash<int>()(k.channelNum) << 1);
// }

}
}
