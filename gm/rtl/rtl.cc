#include <string>
#include <string.h>
#include <complex>
#include <fftw3.h>
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <libbladeRF.h>
#include "gm/zmqcode/zmqworker.h"
#include "gm/bladerf/bladerf.h"
#include "gm/zmqcode/zmqworker.h"
#include "gm/Thread.h"



namespace gm {
namespace rtl {

bladeRF::bladeRF() :
    rx_samplerate(25000000),
    rx_frequency(160000000),
    tx_enable(false),
    tx_samplerate(10000000),
    tx_frequency(440000000),
    device_str("\0"),
    dev(NULL),
    samplerate_actual(0),
    frequency_actual(0),
    bandwidth_actual(0),
    rxBufferPosition(),
    txBufferPosition() {
    rxThread = new rxTask(this);
    txThread = new txTask(this);
	gm::zmqcode::zmqWorker().getFuncMap().insert({"setRxGain", setRxGain()});
	gm::zmqcode::zmqWorker().getFuncMap().insert({"setRxFreq", setRxFreq()});
	gm::zmqcode::zmqWorker().getFuncMap().insert({"getRxFreq", getRxFreq()});
	gm::zmqcode::zmqWorker().getFuncMap().insert({"getRxSamplePeek", rxSamplePeek()});
	gm::zmqcode::zmqWorker().getFuncMap().insert({"setRxCalibration", setRxCalibration()});
}

bladeRF::~bladeRF() {
    delete(rxThread);
    delete(txThread);
    fftw_destroy_plan(plan);
    fftw_free(fft_in); 
    fftw_free(fft_out);
}


gm::zmqcode::func_t bladeRF::rxSamplePeek() {
    return [&](zmq::socket_t* socket, zmq::message_t* identity) {
        int sampLength = N * 100;
        std::complex<short> samples[sampLength];
        float mag[N];
        memset(&mag[0], 0, sizeof(mag));

        long position = getRxBufferPosition()->getNow();
        int memPosition = (int)(position % getRxBufferPosition()->getBufferSize());
        memPosition = (getRxBufferPosition()->getBufferSize() - memPosition > sampLength) ? memPosition : getRxBufferPosition()->getBufferSize() - sampLength - 1;
        memcpy(&samples[0], &getRxBufferPosition()->getBuffer()[memPosition], sampLength*sizeof(std::complex<short>));
        
        for (int cc = 0; cc < 100; cc++) {
            double inv = 1.0;
            for (int cnt = 0; cnt < N; cnt++) {
                fft_in[cnt][0] = inv * (double)samples[cnt + N*cc].real() / 8192.0;
                fft_in[cnt][1] = inv * (double)samples[cnt + N*cc].imag() / 8192.0;
                inv = -inv;
            }
            fftw_execute(plan);

            for (int cnt = 0; cnt < N; cnt++) {
                //std::complex<float> cval = samples[cnt];
                double real = fft_out[cnt][0];
                double imag = fft_out[cnt][1];
                float val = (float)(real * real + imag * imag);
                if (val == 0) {
                    val = 1e-6;
                }
                mag[cnt] += val;
            }
        }
        
        printf("Sending data\n");
        socket->send(*identity, ZMQ_SNDMORE);
        std::string func("NO ERROR GETRXSAMPLEPEEK");
        // Wrap this in proto buffers
        socket->send(func.c_str(), func.size(), ZMQ_SNDMORE);
        socket->send(&mag[0], N*4);
    };
}

gm::zmqcode::func_t bladeRF::setRxGain() {
	return [&](zmq::socket_t* socket, zmq::message_t* identity) {
		zmq::message_t message;
		int more;
		size_t more_size = sizeof (more);
		zmq::message_t lna_message;
		zmq::message_t vga1_message;
		zmq::message_t vga2_message;

		socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
		float gain[3];
		gain[0] = 0;
		gain[1] = 0;
		gain[2] = 0;
		
		if (more) {
			socket->recv(&lna_message);
//			std::cout << "Got lna" << std::endl;
			std::string lna = std::string(static_cast<char*>(lna_message.data()), lna_message.size());
			socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
			if (more) {
				socket->recv(&vga1_message);
//				std::cout << "Got vga1" << std::endl;
				std::string vga1 = std::string(static_cast<char*>(vga1_message.data()), vga1_message.size());
				socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
				if (more) {
				    socket->recv(&vga2_message);
//				    std::cout << "Got vga2" << std::endl;
				    std::string vga2 = std::string(static_cast<char*>(vga2_message.data()), vga2_message.size());
//				    std::cout << "Got gains lna " << lna << " vga1 " << vga1 << " vga2 " << vga2 << std::endl;
				    gain[0] = atof(lna.c_str());
				    gain[1] = atof(vga1.c_str());
				    gain[2] = atof(vga2.c_str());
				}
			}
		}
		if (gain[0] < 3) {
            bladerf_set_lna_gain(dev, BLADERF_LNA_GAIN_BYPASS);
		} else if (gain[0] < 6) {
		    bladerf_set_lna_gain(dev, BLADERF_LNA_GAIN_MID);
		} else {
		    bladerf_set_lna_gain(dev, BLADERF_LNA_GAIN_MAX);
		}
        bladerf_set_rxvga1(dev, gain[1]);
        bladerf_set_rxvga2(dev, gain[2]);
		
		socket->send(*identity, ZMQ_SNDMORE);
		std::string func("NO ERROR SETRXGAIN");
		socket->send(func.c_str(), func.size());
	};
}

gm::zmqcode::func_t bladeRF::getRxFreq() {
    return [&](zmq::socket_t* socket, zmq::message_t* identity) {
        float centerf = (rx_frequency) / 1000000.0f;
        float bw = (rx_samplerate) / 1000000.0f;
        socket->send(*identity, ZMQ_SNDMORE);
        std::string func("NO ERROR GETRXFREQ");
        // Wrap this in proto buffers
        std::stringstream cs;
        cs << centerf;
        std::string center_freq = cs.str();
        std::stringstream bws;
        bws << bw;
        std::string bandwidth = bws.str();
        std::cout << "Sending back freqs" << std::endl;
        socket->send(func.c_str(), func.size(), ZMQ_SNDMORE);
        socket->send(center_freq.c_str(), center_freq.size(), ZMQ_SNDMORE);
        socket->send(bandwidth.c_str(), bandwidth.size());
    };
}

gm::zmqcode::func_t bladeRF::setRxFreq() {
    return [&](zmq::socket_t* socket, zmq::message_t* identity) {
        zmq::message_t message;
        int more;
        size_t more_size = sizeof (more);
        zmq::message_t center_message;

        socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
        float cf;
        int status = -1;
        if (more) {
            socket->recv(&center_message);
            std::string centerf = std::string(static_cast<char*>(center_message.data()), center_message.size());
            socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
            cf = atof(centerf.c_str());
            if (cf > 10.0f && cf < 3600.0f) {
                unsigned int nfreq = (unsigned int)(cf * 1000000.0f);
                
                //status = change_rx_freq(nfreq);
            }
        }
        if (!status) {
            socket->send(*identity, ZMQ_SNDMORE);
            std::string func("NO ERROR SETRXFREQ");
            socket->send(func.c_str(), func.size());
        } else {
            socket->send(*identity, ZMQ_SNDMORE);
            std::string func("BLADERF ERROR SETRXFREQ");
            socket->send(func.c_str(), func.size());
        }
    };
}

gm::zmqcode::func_t bladeRF::setRxCalibration() {
    return [&](zmq::socket_t* socket, zmq::message_t* identity) {
        zmq::message_t message;
        int more;
        size_t more_size = sizeof (more);
        zmq::message_t dci_message;
        zmq::message_t dcq_message;
        zmq::message_t gain_message;
        zmq::message_t phase_message;

        socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
        
        int status = -1;
        // DCI, DCQ,GAIN, PHASE
        int16_t dci, dcq, gain, phase;
        if (more) {
            socket->recv(&dci_message);
            std::string dci_i = std::string(static_cast<char*>(dci_message.data()), dci_message.size());
            socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
            dci = atoi(dci_i.c_str());
            if (more) {
                socket->recv(&dcq_message);
                std::string dcq_i = std::string(static_cast<char*>(dcq_message.data()), dcq_message.size());
                socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
                dcq = atoi(dcq_i.c_str());
                if (more) {
                    socket->recv(&gain_message);
                    std::string gain_i = std::string(static_cast<char*>(gain_message.data()), gain_message.size());
                    socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
                    gain = atoi(gain_i.c_str());
                    if (more) {
                        socket->recv(&phase_message);
                        std::string phase_i = std::string(static_cast<char*>(phase_message.data()), phase_message.size());
                        socket->getsockopt(ZMQ_RCVMORE, &more, &more_size);
                        phase = atoi(phase_i.c_str());
                        std::cout << "Setting gain dci: " << dci << " dcq: " << dcq << " gain " << gain << " phase " << phase << std::endl;
                        status = bladerf_set_correction(dev, BLADERF_MODULE_RX, BLADERF_CORR_LMS_DCOFF_I, dci);
                        status |= bladerf_set_correction(dev, BLADERF_MODULE_RX, BLADERF_CORR_LMS_DCOFF_Q, dcq);
                        status |= bladerf_set_correction(dev, BLADERF_MODULE_RX, BLADERF_CORR_FPGA_GAIN, gain);
                        status |= bladerf_set_correction(dev, BLADERF_MODULE_RX, BLADERF_CORR_FPGA_PHASE, phase);
                    }
                } 
            }
        }
        if (!status) {
            socket->send(*identity, ZMQ_SNDMORE);
            std::string func("NO ERROR SETRXCALIBRATION");
            socket->send(func.c_str(), func.size());
        } else {
            socket->send(*identity, ZMQ_SNDMORE);
            std::string func("BLADERF ERROR SETRXCALIBRATION");
            socket->send(func.c_str(), func.size());
        }
    };
}

int bladeRF::init_module(struct bladerf *dev, unsigned int samplerate, unsigned int frequency,
                         bladerf_module m)
{
    int status;

    status = bladerf_set_sample_rate(dev, m, samplerate, &samplerate_actual);

    if (status != 0) {
        printf("Failed to set samplerate: %s\n",
               bladerf_strerror(status));
        return status;
    }

    status = bladerf_set_bandwidth(dev, m, 20000000, &bandwidth_actual);

    if (status != 0) {
        printf("Failed to set bandwidth: %s\n",
               bladerf_strerror(status));
        return status;
    }

    std::cout << "Bandwidth is " << bandwidth_actual << std::endl;

    //bladerf_log_set_verbosity(BLADERF_LOG_LEVEL_DEBUG);

    status = bladerf_set_frequency(dev, m, frequency);

    if (status != 0) {
        printf("Failed to set frequency: %s\n",
               bladerf_strerror(status));
        return status;
    }

    status = bladerf_get_frequency(dev, m, &frequency_actual);
    if (status != 0) {
        printf("Failed to read back frequency: %s\n",
               bladerf_strerror(status));
        return status;
    }

    printf("Frequency = %u, Samplerate = %u\n",
           frequency_actual, samplerate_actual);

    bladerf_lna_gain lna_gain;
    int rx1gain, rx2gain;
    //look at gain
    bladerf_get_lna_gain(dev, &lna_gain);
    bladerf_get_rxvga1(dev, &rx1gain);
    bladerf_get_rxvga2(dev, &rx2gain);
    std::cout << "LNA gain " << lna_gain << " rxvga1 " << rx1gain << " rxvga2 " << rx2gain << std::endl;
    //bladerf_set_gain(dev, m, 0);
    rx1gain = 0;
    rx2gain = 0;

    bladerf_set_lna_gain(dev, BLADERF_LNA_GAIN_BYPASS);
    bladerf_set_rxvga1(dev, 20);
    bladerf_set_rxvga2(dev, 0);

    bladerf_get_lna_gain(dev, &lna_gain);
    bladerf_get_rxvga1(dev, &rx1gain);
    bladerf_get_rxvga2(dev, &rx2gain);
    std::cout << "LNA gain " << lna_gain << " rxvga1 " << rx1gain << " rxvga2 " << rx2gain << std::endl;


    // gm::zmqcode::zmqWorker().getFuncMap().insert({"changeFreq", [&](zmq::socket_t* socket, zmq::message_t* identity) {
    //     std::cout << "Change freq was called" << std::endl;
    //     float freq;
    //     int floatSize = sizeof(freq);
    //     int more = 0;           //  Multipart detection
    //     size_t more_size = sizeof (more);
    //     socket->getsockopt (ZMQ_RCVMORE, &more, &more_size);
    //     int status = -1;
    //     if (more) {
    //         socket->recv(&freq, floatSize);
    //         unsigned int nfreq = (unsigned int)(freq * 1000000.0f);
    //         status = change_rx_freq(nfreq);
    //     }

    // }
    //                                              });

    return status;
}

struct bladerf* bladeRF::initialize_device()
{
    fft_in = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    fft_out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);    
    plan = fftw_plan_dft_1d(N, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);

    
    try {
        int status = bladerf_open(&dev, device_str.c_str());
        int fpga_loaded;

        printf("Dev string %s\n", device_str.c_str());

        if (status != 0) {
            printf("Failed to open device: %s\n",
                   bladerf_strerror(status));
            return NULL;
        }

        fpga_loaded = bladerf_is_fpga_configured(dev);
        if (fpga_loaded < 0) {
            printf("Failed to check FPGA state: %s\n",
                   bladerf_strerror(fpga_loaded));
            status = -1;
            throw "Failed to check FPGA state:\n";
        } else if (fpga_loaded == 0) {
            printf("The device's FPGA is not loaded.\n");
            status = -1;
            throw "The device's FPGA is not loaded.\n";
        }

        status = init_module(dev, rx_samplerate, rx_frequency, BLADERF_MODULE_RX);
        if (status != 0) {
            printf("Failed to init RX module: %s\n",
                   bladerf_strerror(status));
            throw "Failed to init RX module:\n";
        }

        std::complex<short>* rxBuffer = (std::complex<short>*)calloc(rx_buffer_size, sizeof(std::complex<short>));
        rxBufferPosition.setBuffer(rxBuffer, rx_buffer_size);

        if (tx_enable) {
            status = init_module(dev, tx_samplerate, tx_frequency, BLADERF_MODULE_TX);
            if (status != 0) {
                printf("Failed to init TX module: %s\n",
                       bladerf_strerror(status));
                throw "Failed to init TX module:\n";
            }
            std::complex<short>* txBuffer = (std::complex<short>*)calloc(tx_buffer_size, sizeof(std::complex<short>));
            txBufferPosition.setBuffer(txBuffer, tx_buffer_size);
        }
        
        bladerf_set_correction(dev,BLADERF_MODULE_RX,BLADERF_CORR_LMS_DCOFF_I,1568);
        bladerf_set_correction(dev,BLADERF_MODULE_RX,BLADERF_CORR_LMS_DCOFF_Q,96);
        
    } catch (const char* error_string) {
        bladerf_close(dev);
        dev = NULL;
    }

    return dev;
}

int bladeRF::change_rx_freq(unsigned int freq)
{
    int status;
    //printf("Bladerf got a change freq %u\n", freq);
    synchronized(*this, [&] {
        bladerf_select_band(dev, BLADERF_MODULE_RX, freq);
        status = bladerf_set_frequency(dev, BLADERF_MODULE_RX, freq);

    });
    rx_frequency = freq;
    return status;
}

unsigned int bladeRF::get_rx_freq()
{
    int status;
    unsigned int freq;
    status = bladerf_get_frequency(dev, BLADERF_MODULE_RX, &freq);
    if (status != 0) {
        printf("Failed to read back frequency: %s\n",
         bladerf_strerror(status));
        throw "Failed to read back frequency";
    }
    return freq;
}

rxTask::rxTask(bladeRF* device)
    :  device(device), done(false)
{

}

rxTask::~rxTask()
{

}

void rxTask::run()
{
    int status;
    std::complex<short> *mem_base;
    std::complex<short> *samples;
    long position = 0;
    int mem_size;
    gm::buffer::BufferPosition<std::complex<short>>* bPos = device->getRxBufferPosition();
    // struct task_args *task = (struct task_args*) args;
    // struct test_params *p = task->p;

    samples = (std::complex<short> *)calloc(device->getBlockSize(), sizeof(std::complex<short>));
    mem_size = sizeof(std::complex<short>) * device->getBlockSize();
    mem_base = device->getRxBuffer();
    int stream_buffer_count = 32;
    int stream_buffer_size = 8192;
    int num_xfers = 2;
    int timeout_ms = 0;
    printf("Starting bladerf config count: %d size: %d num: %d timeout %d\n", stream_buffer_count,
           stream_buffer_size,
           num_xfers, timeout_ms);

    status = bladerf_sync_config(device->getDev(),
                                 BLADERF_MODULE_RX,
                                 BLADERF_FORMAT_SC16_Q11,
                                 stream_buffer_count,
                                 stream_buffer_size,
                                 num_xfers,
                                 timeout_ms);

    if (status != 0) {
        printf("Failed to initialize RX sync handle: %s\n",
               bladerf_strerror(status));
        return;
    }
    printf("blade is loaded block size %d\n", device->getBlockSize());

    synchronized(device, [&] {
        status = bladerf_enable_module(device->getDev(), BLADERF_MODULE_RX, true);
    });
    if (status != 0) {
        printf("Failed to enable RX module: %s\n", bladerf_strerror(status));
        done = true;
    }

    /* This assumption is made with the below cast */
    // long lastp = 0;
    while (!done) {
        status = bladerf_sync_rx(device->getDev(), samples, device->getBlockSize(), NULL, 0);

        if (status != 0) {
            printf("RX failed: %s\n", bladerf_strerror(status));
            done = true;
        } else {
            position += (long)device->getBlockSize();
            memcpy(&mem_base[(int)(position % device->rx_buffer_size)], samples, mem_size);
            bPos->setPosition(position);
            // if (position > lastp) {
            //     printf("TIK\n");
            //     lastp = position + device->getRxSampleRate();
            // }
        }
    }
    free(samples);

    synchronized(device, [&] {
        status = bladerf_enable_module(device->getDev(), BLADERF_MODULE_RX, false);
    });
    if (status != 0) {
        printf("Failed to disable RX module: %s\n", bladerf_strerror(status));
    }

}

txTask::txTask(bladeRF* device)
    :  device(device), done(false)
{

}

txTask::~txTask() {

}

void txTask::run()
{
    int status;
    std::complex<short> *mem_base;
    std::complex<short> *samples;
    long position = 0;
    int mem_size;
    gm::buffer::BufferPosition<std::complex<short>>* bPos = device->getTxBufferPosition();
    // struct task_args *task = (struct task_args*) args;
    // struct test_params *p = task->p;

    samples = (std::complex<short> *)calloc(device->getBlockSize(), sizeof(std::complex<short>));
    mem_size = sizeof(std::complex<short>) * device->getBlockSize();
    mem_base = device->getTxBuffer();
    int stream_buffer_count = 32;
    int stream_buffer_size = 8192;
    int num_xfers = 2;
    int timeout_ms = 0;
    printf("Starting bladerf config count: %d size: %d num: %d timeout %d\n", stream_buffer_count,
           stream_buffer_size,
           num_xfers, timeout_ms);

    status = bladerf_sync_config(device->getDev(),
                                 BLADERF_MODULE_TX,
                                 BLADERF_FORMAT_SC16_Q11,
                                 stream_buffer_count,
                                 stream_buffer_size,
                                 num_xfers,
                                 timeout_ms);

    if (status != 0) {
        printf("Failed to initialize TX sync handle: %s\n",
               bladerf_strerror(status));
        return;
    }
    printf("blade tx is loaded block size %d\n", device->getBlockSize());

    synchronized(device, [&] {
        status = bladerf_enable_module(device->getDev(), BLADERF_MODULE_TX, true);
    });
    if (status != 0) {
        printf("Failed to enable TX module: %s\n", bladerf_strerror(status));
        done = true;
    }

    /* This assumption is made with the below cast */
    long lastp = 0;
    while (!done) {
        status = bladerf_sync_tx(device->getDev(), samples, device->getBlockSize(), NULL, 0);

        if (status != 0) {
            printf("TX failed: %s\n", bladerf_strerror(status));
            done = true;
        } else {
            position += (long)device->getBlockSize();
            memcpy(samples, &mem_base[(int)(position % device->tx_buffer_size)], mem_size);
            bPos->setPosition(position);
        }

    }
    free(samples);

    synchronized(device, [&] {
        status = bladerf_enable_module(device->getDev(), BLADERF_MODULE_TX, false);
    });
    if (status != 0) {
        printf("Failed to disable TX module: %s\n", bladerf_strerror(status));
    }
}



/* For the sake of simplicity, we'll just bounce back and forth between
 * RX and TX when doing both. */
int bladeRF::start_card()
{
    dev = initialize_device();
    if (dev == NULL) {
        printf("Cant initialize device \n");
        return -1;
    }

    printf("device initialized\n");

    if (tx_enable) {
        printf("Starting TX task\n");
        txThread->start();
    }

    bladerf_expansion_attach(dev, BLADERF_XB_200);
    bladerf_xb200_set_filterbank(dev, BLADERF_MODULE_RX, BLADERF_XB200_AUTO_3DB);
    bladerf_xb200_set_path(dev, BLADERF_MODULE_RX, BLADERF_XB200_MIX);
//    bladerf_xb200_set_path(dev, BLADERF_MODULE_RX, BLADERF_XB200_BYPASS);

    printf("Starting RX task\n");
    rxThread->start();

    printf("Running...\n");

    return 0;

}

int bladeRF::stop_card() {
    rxThread->setDone();
    if (tx_enable) {
        txThread->setDone();
        txThread->join();
    }
    rxThread->join();
}


}
}


// int bladerf_common::set_dc_offset(bladerf_module module, const std::complex<double> &offset, size_t chan)
// {
//     int ret = 0;
//     int16_t val_i, val_q;

//     val_i = (int16_t)(offset.real() * DCOFF_SCALE);
//     val_q = (int16_t)(offset.imag() * DCOFF_SCALE);

//     ret  = bladerf_set_correction(_dev.get(), module, BLADERF_CORR_LMS_DCOFF_I, val_i);
//     ret |= bladerf_set_correction(_dev.get(), module, BLADERF_CORR_LMS_DCOFF_Q, val_q);

//     return ret;
// }

// int bladerf_common::set_iq_balance(bladerf_module module, const std::complex<double> &balance, size_t chan)
// {
//     int ret = 0;
//     int16_t val_gain, val_phase;

//     val_gain = (int16_t)(balance.real() * GAIN_SCALE);
//     val_phase = (int16_t)(balance.imag() * PHASE_SCALE);

//     ret  = bladerf_set_correction(_dev.get(), module, BLADERF_CORR_FPGA_GAIN, val_gain);
//     ret |= bladerf_set_correction(_dev.get(), module, BLADERF_CORR_FPGA_PHASE, val_phase);

//     return ret;
// }

