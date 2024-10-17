cmake_minimum_required (VERSION 3.5)

project("GronolaSDR")

set (CMAKE_BUILD_TYPE Debug)
#set (CMAKE_BUILD_TYPE RelWithDebInfo)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 50 70 75)
endif()

set(LIBUSB usb pthread rt)
set(LIBLOG)
set(LIBLOC "-L/opt/cuda/lib64")
include_directories("/opt/cuda/include")

set(GM_CUDA_SOURCES
    "gm/cuda/CudaCopy.cu"
    "gm/cuda/HostCuda.cu"
    "gm/cuda/FMDemodLong.cu"
    "gm/cuda/OverlapSave.cc"
    "gm/cuda/UHFFMDemod.cc"
    "gm/cuda/FreqView.cc"
    "gm/cuda/FreqScan.cc"
    "gm/cuda/ProcessFreqs.cc"
    "gm/cuda/SumFreqs.cc"
    "gm/cuda/CopyBuffer.cc"
    "gm/cuda/CopyEpoch.cc"
    )

enable_language("CUDA")

execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
                OUTPUT_VARIABLE GCC_VERSION)


add_library (gmcuda ${GM_CUDA_SOURCES})
target_link_libraries (gmcuda ${LIBLOC} cufft)

set(GM_SOURCES
    "gm/Thread.cc"
    "gm/bladerf/bladerf.cc"
    "gm/rx888/rx888.cc"
    "gm/zmqcode/zmqserver.cc" 
    "gm/zmqcode/zmqworker.cc"
    "gm/zmqcode/ZMQPub.cc"
    "gm/sound/FMSound.cc"
    "gm/sound/SoundMonitor.cc")

set(GM_LIBS ${LIBLOC} ${LIBLOG} bladeRF sddc ${LIBUSB} zmq)

add_library(lib_granolamatt ${GM_SOURCES})
target_link_libraries (lib_granolamatt ${GM_LIBS})

add_library(lib_wav "gm/wavfile/wavfile.c")

add_executable(blade_capture "gm/main.cc")
target_link_libraries (blade_capture lib_granolamatt ${GM_LIBS})

add_executable(rx888_capture "gm/rx888_test.cc")
target_link_libraries (rx888_capture lib_granolamatt ${GM_LIBS})

add_executable(hf_rx "gm/HFRx.cc")
target_link_libraries (hf_rx lib_granolamatt gmcuda ${GM_LIBS})

add_executable(zmqclient gm/zmqcode/client.cc)
target_link_libraries (zmqclient ${LIBLOC} zmq)

add_executable(cudatest gm/CudaHostTest.cc)
target_link_libraries (cudatest lib_granolamatt gmcuda ${GM_LIBS})

add_executable(buffertest gm/test_bufferpos.cc)
target_link_libraries (buffertest lib_granolamatt gmcuda ${GM_LIBS})

add_executable(uhf_rx gm/UHFRx.cc)
target_link_libraries (uhf_rx lib_granolamatt gmcuda ${GM_LIBS} lib_wav)

add_executable(copytest gm/copytest.cc)
target_link_libraries (copytest lib_granolamatt gmcuda ${GM_LIBS})

#adb push cudatest /data/local/develop/
