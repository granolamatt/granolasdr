#ifndef _GM_CUDA_SUMFREQS_
#define _GM_CUDA_SUMFREQS_

#include <string>
#include <zmq.hpp>
#include "gm/buffer/BufferPosition.h"
#include "gm/cuda/CudaCopy.h"
#include "gm/zmqcode/zmqworker.h"


namespace gm {
namespace cuda {

class SumFreqs {
public:
	SumFreqs(gm::buffer::BufferPosition<float>* inP);
	~SumFreqs();

	long getStartFreq() {return startFreq;}
	void setStartFreq(long f) {startFreq = f; calculateNumbers();}
	long getStopFreq() {return stopFreq;}
	void setStopFreq(long f) {stopFreq = f; calculateNumbers();}
	long getTotalFreq() {return totalFreq;}
	long getTotalPasses() {return tpasses;}
	void setTotalPasses(long passes) {tpasses = passes;}
	long getSkipped() {return skipped;}
	void addDataToList(std::string getData, std::string freqRange);

private:
	void calculateNumbers();
	gm::zmqcode::func_t processSamples();
	gm::zmqcode::func_t getFreqs();
	gm::zmqcode::func_t setFreqRange();
	bool isFreqsOK(float freqs[2]);

	gm::buffer::BufferPosition<float>* buffPos;
	float* samples;
	long startFreq;
	long stopFreq;
	long totalFreq;
	long skipped;
	long tpasses;
};

}
}

#endif //_GM_CUDA_SUMFREQS_