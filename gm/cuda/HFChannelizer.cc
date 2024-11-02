#include <unistd.h>
#include <iostream>
#include <cuda.h>
#include <complex>
#include <chrono>

#include "ft8_lib/ft8/decode.h"
#include "ft8_lib/ft8/encode.h"
#include "ft8_lib/ft8/message.h"
#include "ft8_lib/common/common.h"
#include "ft8_lib/common/monitor.h"

#define LOG_LEVEL LOG_INFO
#include "ft8_lib/ft8/debug.h"

#include "gm/cuda/HFChannelizer.h"
#include "gm/cuda/HostCuda.h"
#include "gm/buffer/BufferPosition.h"
#include "gm/rx888/rx888.h"

const int kMin_score = 10; // Minimum sync score threshold for candidates
const int kMax_candidates = 140;
const int kLDPC_iterations = 25;

const int kMax_decoded_messages = 50;

const int kFreq_osr = 2; // Frequency oversampling rate (bin subdivision)
const int kTime_osr = 2; // Time oversampling rate (symbol subdivision)

#define CALLSIGN_HASHTABLE_SIZE 256

static struct
{
    char callsign[12]; ///> Up to 11 symbols of callsign + trailing zeros (always filled)
    uint32_t hash;     ///> 8 MSBs contain the age of callsign; 22 LSBs contain hash value
} callsign_hashtable[CALLSIGN_HASHTABLE_SIZE];

static int callsign_hashtable_size;

void hashtable_init(void)
{
    callsign_hashtable_size = 0;
    memset(callsign_hashtable, 0, sizeof(callsign_hashtable));
}

void hashtable_cleanup(uint8_t max_age)
{
    for (int idx_hash = 0; idx_hash < CALLSIGN_HASHTABLE_SIZE; ++idx_hash)
    {
        if (callsign_hashtable[idx_hash].callsign[0] != '\0')
        {
            uint8_t age = (uint8_t)(callsign_hashtable[idx_hash].hash >> 24);
            if (age > max_age)
            {
                LOG(LOG_INFO, "Removing [%s] from hash table, age = %d\n", callsign_hashtable[idx_hash].callsign, age);
                // free the hash entry
                callsign_hashtable[idx_hash].callsign[0] = '\0';
                callsign_hashtable[idx_hash].hash = 0;
                callsign_hashtable_size--;
            }
            else
            {
                // increase callsign age
                callsign_hashtable[idx_hash].hash = (((uint32_t)age + 1u) << 24) | (callsign_hashtable[idx_hash].hash & 0x3FFFFFu);
            }
        }
    }
}

void hashtable_add(const char* callsign, uint32_t hash)
{
    uint16_t hash10 = (hash >> 12) & 0x3FFu;
    int idx_hash = (hash10 * 23) % CALLSIGN_HASHTABLE_SIZE;
    while (callsign_hashtable[idx_hash].callsign[0] != '\0')
    {
        if (((callsign_hashtable[idx_hash].hash & 0x3FFFFFu) == hash) && (0 == strcmp(callsign_hashtable[idx_hash].callsign, callsign)))
        {
            // reset age
            callsign_hashtable[idx_hash].hash &= 0x3FFFFFu;
            LOG(LOG_DEBUG, "Found a duplicate [%s]\n", callsign);
            return;
        }
        else
        {
            LOG(LOG_DEBUG, "Hash table clash!\n");
            // Move on to check the next entry in hash table
            idx_hash = (idx_hash + 1) % CALLSIGN_HASHTABLE_SIZE;
        }
    }
    callsign_hashtable_size++;
    strncpy(callsign_hashtable[idx_hash].callsign, callsign, 11);
    callsign_hashtable[idx_hash].callsign[11] = '\0';
    callsign_hashtable[idx_hash].hash = hash;
}

bool hashtable_lookup(ftx_callsign_hash_type_t hash_type, uint32_t hash, char* callsign)
{
    uint8_t hash_shift = (hash_type == FTX_CALLSIGN_HASH_10_BITS) ? 12 : (hash_type == FTX_CALLSIGN_HASH_12_BITS ? 10 : 0);
    uint16_t hash10 = (hash >> (12 - hash_shift)) & 0x3FFu;
    int idx_hash = (hash10 * 23) % CALLSIGN_HASHTABLE_SIZE;
    while (callsign_hashtable[idx_hash].callsign[0] != '\0')
    {
        if (((callsign_hashtable[idx_hash].hash & 0x3FFFFFu) >> hash_shift) == hash)
        {
            strcpy(callsign, callsign_hashtable[idx_hash].callsign);
            return true;
        }
        // Move on to check the next entry in hash table
        idx_hash = (idx_hash + 1) % CALLSIGN_HASHTABLE_SIZE;
    }
    callsign[0] = '\0';
    return false;
}

ftx_callsign_hash_interface_t hash_if = {
    .lookup_hash = hashtable_lookup,
    .save_hash = hashtable_add
};

void decode(const monitor_t* mon, double tm_slot_start)
{
    const ftx_waterfall_t* wf = &mon->wf;
    // Find top candidates by Costas sync score and localize them in time and frequency
    ftx_candidate_t candidate_list[kMax_candidates];
    int num_candidates = ftx_find_candidates(wf, kMax_candidates, candidate_list, kMin_score);
    printf("Found %d candidates\n",num_candidates);

    // Hash table for decoded messages (to check for duplicates)
    int num_decoded = 0;
    ftx_message_t decoded[kMax_decoded_messages];
    ftx_message_t* decoded_hashtable[kMax_decoded_messages];

    // Initialize hash table pointers
    for (int i = 0; i < kMax_decoded_messages; ++i)
    {
        decoded_hashtable[i] = NULL;
    }

    // Go over candidates and attempt to decode messages
    for (int idx = 0; idx < num_candidates; ++idx)
    {
        const ftx_candidate_t* cand = &candidate_list[idx];

        float freq_hz = (mon->min_bin + cand->freq_offset + (float)cand->freq_sub / wf->freq_osr) / mon->symbol_period;
        float time_sec = (cand->time_offset + (float)cand->time_sub / wf->time_osr) * mon->symbol_period;

        ftx_message_t message;
        ftx_decode_status_t status;
        if (!ftx_decode_candidate(wf, cand, kLDPC_iterations, &message, &status))
        {
            if (status.ldpc_errors > 0)
            {
                LOG(LOG_DEBUG, "LDPC decode: %d errors\n", status.ldpc_errors);
            }
            else if (status.crc_calculated != status.crc_extracted)
            {
                LOG(LOG_DEBUG, "CRC mismatch!\n");
            }
            continue;
        }

        LOG(LOG_DEBUG, "Checking hash table for %4.1fs / %4.1fHz [%d]...\n", time_sec, freq_hz, cand->score);
        int idx_hash = message.hash % kMax_decoded_messages;
        bool found_empty_slot = false;
        bool found_duplicate = false;
        do
        {
            if (decoded_hashtable[idx_hash] == NULL)
            {
                LOG(LOG_DEBUG, "Found an empty slot\n");
                found_empty_slot = true;
            }
            else if ((decoded_hashtable[idx_hash]->hash == message.hash) && (0 == memcmp(decoded_hashtable[idx_hash]->payload, message.payload, sizeof(message.payload))))
            {
                LOG(LOG_DEBUG, "Found a duplicate!\n");
                found_duplicate = true;
            }
            else
            {
                LOG(LOG_DEBUG, "Hash table clash!\n");
                // Move on to check the next entry in hash table
                idx_hash = (idx_hash + 1) % kMax_decoded_messages;
            }
        } while (!found_empty_slot && !found_duplicate);

        if (found_empty_slot)
        {
            // Fill the empty hashtable slot
            memcpy(&decoded[idx_hash], &message, sizeof(message));
            decoded_hashtable[idx_hash] = &decoded[idx_hash];
            ++num_decoded;

            char text[FTX_MAX_MESSAGE_LENGTH];
            ftx_message_rc_t unpack_status = ftx_message_decode(&message, &hash_if, text);
            if (unpack_status != FTX_MESSAGE_RC_OK)
            {
                snprintf(text, sizeof(text), "Error [%d] while unpacking!", (int)unpack_status);
            }

            // Fake WSJT-X-like output for now
            float snr = cand->score * 0.5f; // TODO: compute better approximation of SNR
            printf("%+05.1f %+05.1f %+4.2f %4.0f ~  %s\n",
                snr, tm_slot_start, time_sec, freq_hz, text);
        }
    }
    LOG(LOG_INFO, "Decoded %d messages, callsign hashtable size %d\n", num_decoded, callsign_hashtable_size);
    hashtable_cleanup(10);
}

static void waterfall_init(ftx_waterfall_t* me, int max_blocks, int num_bins, int time_osr, int freq_osr)
{
    size_t mag_size = max_blocks * time_osr * freq_osr * num_bins * sizeof(me->mag[0]);
    me->max_blocks = max_blocks;
    me->num_blocks = 0;
    me->num_bins = num_bins;
    me->time_osr = time_osr;
    me->freq_osr = freq_osr;
    me->block_stride = (time_osr * freq_osr * num_bins);
    me->mag = (WF_ELEM_T*)malloc(mag_size);
    LOG(LOG_DEBUG, "Waterfall size = %zu\n", mag_size);
}

void load_monitor(monitor_t* me)
{
    float slot_time = FT8_SLOT_TIME;
    float symbol_period = FT8_SYMBOL_PERIOD;
    // Compute DSP parameters that depend on the sample rate
    me->block_size = 698880; // samples corresponding to one FSK symbol
    me->subblock_size = 698880;
    me->nfft = 698880;
    me->fft_norm = 698880;

    LOG(LOG_INFO, "Block size = %d\n", me->block_size);
    LOG(LOG_INFO, "Subblock size = %d\n", me->subblock_size);

    // Allocate enough blocks to fit the entire FT8/FT4 slot in memory
    const int max_blocks = (int)(slot_time / symbol_period);
    // Keep only FFT bins in the specified frequency range (f_min/f_max)
    me->min_bin = 0;
    me->max_bin = 698880;
    const int num_bins = me->max_bin - me->min_bin;

    waterfall_init(&me->wf, max_blocks, num_bins, 1, 1);
    me->wf.protocol = FTX_PROTOCOL_FT8;

    me->symbol_period = symbol_period;

    me->max_mag = -120.0f;
}

namespace gm {
namespace cuda {

HFChannelizer::HFChannelizer(gm::buffer::BufferPosition<int16_t>* inP) :
inPos(inP),
buff_pos{0},
inData_d(NULL),
startcap(false),
fftInData_d(NULL), 
fftData_d(NULL),
channelData_d(NULL),
demodData_d(NULL),
demodFT8_d(NULL),
pixel_d(NULL) {
    inShape = inPos->getShape();
    inData = (int16_t*)inPos->getBuffer();
    fs = std::ofstream("ft8.bin", std::ios::out | std::ios::binary | std::ios::app);
    try {
        //cuda_check_error(cudaHostAlloc((void**)&inData_d, inPos->getByteSize(), cudaHostAllocPortable));
        cuda_check_error(cudaMalloc((void**)&inData_d, inPos->getByteSize()));
        cuda_check_error(cudaMalloc((void**)&fftInData_d,
            sizeof(float)*(gm::rx888::rx888::NLARGE*gm::rx888::rx888::BUFFERS + gm::rx888::rx888::NLARGE)));
        cuda_check_error(cudaMalloc((void**)&fftData_d, 
            sizeof(std::complex<float>)*(2*gm::rx888::rx888::NLARGE + 1024))); // 1024 is pad for the extra data from realfft

        cuda_check_error(cudaSetDevice(0));
        cuda_check_error(cudaStreamCreate(&stream));
        cufftResult fftRes = cufftPlan1d(&plan, gm::rx888::rx888::NLARGE*2, CUFFT_R2C, 1);
        if (fftRes) {
            printf("Error: exit for now\n");
        }
        fftRes = cufftSetStream(plan, stream);
        if (fftRes) {
            printf("Error: exit for now\n");
        }
        cuda_h = gm::cuda::device::HostCuda(stream);
        // From calc_rf.py
        bins = {{13480,14980,1500},
                {26212,29960,3748},
                {39920,40712,792},
                {52424,54676,2252},
                {75644,76024,380},
                {104856,107480,2624},
                {135324,136076,752},
                {157284,160660,3376},
                {186420,187172,752},
                {209712,222448,12736},
            };
        fft_length = 32768;
        rfft_length = 698880; // half off for some reason

        demodFT8 = (std::complex<float>*)malloc(rfft_length*sizeof(std::complex<float>));


        cuda_check_error(cudaMalloc((void**)&channelData_d, fft_length*sizeof(std::complex<float>) + 1024));
        printf("Total fft length is %u\n", fft_length);

        // Two so we can use it as a buffer also
        cuda_check_error(cudaMalloc((void**)&demodData_d, 2*rfft_length*sizeof(std::complex<float>) + 1024));
        printf("Total rfft length is %u\n", rfft_length);

        // Two so we can use it as a buffer also
        cuda_check_error(cudaMalloc((void**)&demodFT8_d, rfft_length*sizeof(std::complex<float>) + 1024));
        printf("Total rfft length is %u\n", rfft_length);

        // Now for the sub channels
        fftRes = cufftPlan1d(&iplan, fft_length, CUFFT_C2C, 1);
        if (fftRes) {
            printf("Error: exit for now\n");
        }
        fftRes = cufftSetStream(iplan, stream);
        if (fftRes) {
            printf("Error: exit for now\n");
        }

        // Now for the sub channels
        fftRes = cufftPlan1d(&rplan, rfft_length, CUFFT_C2C, 1);
        if (fftRes) {
            printf("Error: exit for now\n");
        }
        fftRes = cufftSetStream(rplan, stream);
        if (fftRes) {
            printf("Error: exit for now\n");
        }

        hashtable_init();
        load_monitor(&mon);

    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error after cudaSetDevice: " << e.what() << std::endl;
    }
}

HFChannelizer::~HFChannelizer() {
    if (inData_d) cudaFree(inData_d);
    if (fftInData_d) cudaFree(fftInData_d);
    if (fftData_d) cudaFree(fftData_d);
    if (channelData_d) cudaFree(channelData_d);
    if (demodData_d) cudaFree(demodData_d);
    if (demodFT8_d) cudaFree(demodFT8_d);
    // if (pixel_d) cudaFree(pixel_d);
    cufftDestroy(plan);
    cufftDestroy(iplan);
    cufftDestroy(rplan);
    cudaStreamDestroy(stream);
}

int HFChannelizer::doCopy(uint64_t now) {
    try {
        size_t length = inShape[1];
        int in_position = (now % inShape[0]) * length;
        // first copy the data into the device
        cuda_check_error(cudaMemcpyAsync(&inData_d[in_position], &inData[in_position], 
            length*inPos->getElementSize(), cudaMemcpyHostToDevice, stream));
        // now cast it to floats and scale it
        cuda_h.copyKernel(&fftInData_d[in_position + length], &inData_d[in_position], length);
        // if we are on first buffer then do the wrap

        if (!(now % gm::rx888::rx888::BUFFERS)) {
            cudaMemcpyAsync(&fftInData_d, 
                            &fftInData_d[gm::rx888::rx888::BUFFERS*length], 
                            length*sizeof(float),cudaMemcpyDeviceToDevice, stream);
        }
        
        cufftResult_t rval = cufftExecR2C(plan, &fftInData_d[in_position], (cufftComplex *) fftData_d);
        if (rval) {
            printf("Error in fft\n");
            return 0;
        }
        
        uint32_t offset = 0;
        cuda_check_error(cudaMemsetAsync(channelData_d, 0, fft_length*sizeof(std::complex<float>), stream));
        // Does the USB really belong here??  It is really
        // the same just inverted spectrum
        // copy it in backwards maybe
        for(const std::vector<uint32_t>& b : bins) {
            cuda_check_error(cudaMemcpyAsync(&channelData_d[offset], 
                &fftData_d[b[0]],
                b[2]*sizeof(std::complex<float>),cudaMemcpyDeviceToDevice, stream));
            offset += b[2];
        }

        // now time data of all our freqs
        rval = cufftExecC2C(iplan, (cufftComplex *)&channelData_d[0],
            (cufftComplex *)&channelData_d[0], CUFFT_INVERSE);        
        if (rval) {
            printf("Error in fft\n");
            return 0;
        }
        
        // we are done so put it in a buffer so others can use it??
        // but it stays in cuda so reuse the stream and have events??
        // I guess for now lets do ft8 here to make sure we have a concept
        

        cuda_check_error(cudaMemcpyAsync(&demodData_d[buff_pos], 
            &channelData_d[fft_length/4],
            fft_length / 2 * sizeof(float),cudaMemcpyDeviceToDevice, stream));
        buff_pos += fft_length / 2;

        if (buff_pos > rfft_length) {
            auto now = std::chrono::system_clock::now();
            auto duration = now.time_since_epoch();
            double seconds = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
            uint64_t trigger = (uint64_t)(seconds) % 15;
            
            bool gotime = (trigger == 14 && trunc(seconds) > 0.99);
            if (gotime && ~startcap) {
                startcap = true;
            }
            rval = cufftExecC2C(rplan, (cufftComplex *)&demodData_d[0],
                (cufftComplex *)&demodFT8_d[0], CUFFT_FORWARD);
            if (rval) {
                printf("Error in fft\n");
                return 0;
            }
            cuda_check_error(cudaMemcpyAsync(&demodFT8[0], 
                &demodFT8_d[0],
                rfft_length * sizeof(float),cudaMemcpyDeviceToHost, stream));

            buff_pos -= rfft_length / oversample;
                // I think this will not stomp on the data
                cuda_check_error(cudaMemcpyAsync(&demodData_d[0], 
                    &demodData_d[rfft_length / oversample],
                    buff_pos * sizeof(float),cudaMemcpyDeviceToDevice, stream));
                printf("Do the bb fft %u delta %f\n", buff_pos, seconds - lastepoch);
                lastepoch = seconds;

                cuda_check_error(cudaMemcpyAsync(&demodFT8[0], 
                    &demodFT8_d[0],
                    rfft_length * sizeof(std::complex<float>),cudaMemcpyDeviceToHost, stream));       
                cudaStreamSynchronize(stream);
	    if (startcap) {

                // fs.write(reinterpret_cast<const char*>(demodFT8), rfft_length * sizeof(std::complex<float>));

                // XXX Do this instead
                // int num = fread(buffer,8,sample_rate,ptr); // read 10 bytes to our buffer
                // is_live = (num > 0);
                int offset = mon.wf.num_blocks * mon.wf.block_stride;

                for (int cc=0; cc< 698880; cc++) {
                //for (int cc=0; cc< 0; cc++) {
                    float real = demodFT8[cc].real() / 100e6;
                    float imag = demodFT8[cc].imag() / 100e6;
                    float mag2 = real*real + imag*imag;
                    float db = 10.0f * log10f(1E-12f + mag2);
                    int scaled = (int)(2 * db + 240);
                    mon.wf.mag[offset] = (scaled < 0) ? 0 : ((scaled > 255) ? 255 : scaled);
                    if (db > mon.max_mag)
                        mon.max_mag = db;
                    offset += 1;
                }
                ++mon.wf.num_blocks;
                if (mon.wf.num_blocks % FT8_NN == 0) {
                    printf("Processing\n");
                    // Decode accumulated data (containing slightly less than a full time slot)
                    //decode(&mon, seconds);
                    monitor_reset(&mon);
                    startcap = false;
                }

            }
            

        }

        return 1;
    } catch (thrust::system_error &e) {
        std::cerr << "CUDA error after cudaCopy: " << e.what() << std::endl;
    }
    return 0;
}

void HFChannelizer::run() {
    uint64_t now = inPos->getNow(1) + 1;
    
    while(isRunning()) {
        uint64_t next = inPos->getPosition(now+1, 1);
        while(now < next) {
            uint64_t length = next - now;
            if (length > 4) {
                std::cout << "Error Falling Behind in Cuda Copy, Dropping Data" << std::endl;
                now = next;
                break;
            }
            int numCopied = doCopy(now);
            if (!numCopied) exit(-200);
            //outPos.setPosition(now, 1);
	    now += numCopied;
        }
    }
}

}
}
