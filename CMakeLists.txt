cmake_minimum_required (VERSION 2.8)

#option(ENABLE_ANDROID
#       "Build for android"
#       ON
#)

#set (CMAKE_CXX_FLAGS "--std=c++11 ${CMAKE_CXX_FLAGS}")
set (CMAKE_BUILD_TYPE Debug)
#set (CMAKE_BUILD_TYPE RelWithDebInfo)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#IF(FFTW_INCLUDE_DIRS)
#FIND_PATH(FFTW_INCLUDE_DIR fftw3.h ${FFTW_INCLUDE_DIRS})
#FIND_LIBRARY(FFTW_LIBRARY fftw3 ${FFTW_LIBRARY_DIRS})
#ELSE(FFTW_INCLUDE_DIRS)
#FIND_PATH(FFTW_INCLUDE_DIR fftw3.h ${QMC_INCLUDE_PATHS})
#FIND_LIBRARY(FFTW_LIBRARIES fftw3 ${QMC_LIBRARY_PATHS})
#ENDIF(FFTW_INCLUDE_DIRS)

#SET(FFTW_FOUND FALSE)
#IF(FFTW_INCLUDE_DIR AND FFTW_LIBRARIES)
#MESSAGE(STATUS "FFTW_INCLUDE_DIR=${FFTW_INCLUDE_DIR}")
#MESSAGE(STATUS "FFTW_LIBRARIES=${FFTW_LIBRARIES}")
#SET(FFTW_FOUND TRUE)
#ENDIF()

#MARK_AS_ADVANCED(
#FFTW_INCLUDE_DIR
#FFTW_LIBRARIES
#FFTW_FOUND
#)

# ugly hacks
if(ENABLE_ANDROID)
    message("Android is enabled")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -fPIE -pie")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC -fPIE -pie")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -fPIC")
    include_directories(${CMAKE_CURRENT_SOURCE_DIR} $ENV{ANDROIDPATH}/include)
    set(LIBUSB usb1.0)
    set(LIBLOG log)
    set(LIBLOC "-L$ENV{ANDROIDPATH}/lib" 
        "-L/home/matt/NVPACK/cuda-android-7.0/aarch64-linux-androideabi/lib64")
    set (SM_ARCH 53)
else()
    message("Android is NOT enabled")
    set(LIBUSB usb pthread rt)
    set(LIBLOG)
    set(LIBLOC "-L/opt/cuda/lib64")
    set (SM_ARCH 52)
endif()

#set_property(TARGET myTarget PROPERTY CUDA_ARCHITECTURES 50)

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_50,code=sm_50")

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
    )

#set (HOST_COMPILER $ENV{CXX})
find_package(CUDA REQUIRED)
# set(CUDA_HOST_COMPILATION_CPP ON)
# set(CUDA_SEPARABLE_COMPILATION ON)
# set(CUDA_HOST_COMPILER gcc)

enable_language("CUDA")

include_directories("${CUDA_INCLUDE_DIRS}")

execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
                OUTPUT_VARIABLE GCC_VERSION)
#if (GCC_VERSION VERSION_GREATER 5.0 OR GCC_VERSION VERSION_EQUAL 5.0)
#        message(STATUS "GCC Version >= 5.0")
#        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_${SM_ARCH},code=sm_${SM_ARCH})
#else()
#        message(STATUS "Version < 5.0 Using std=c++11 flag in nvcc")
#        set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 --std=c++11 -D__float128=void -gencode arch=compute_${SM_ARCH},code=sm_${SM_ARCH})
#endif()

#cuda_add_executable (gmcuda ${GM_CUDA_SOURCES})

if(ENABLE_ANDROID)
    # for android we have to remove -lrt and -rdl before cuda_add_library
    set(CUDA_LIBRARIES ${CUDA_cudart_static_LIBRARY})
endif()

add_library (gmcuda ${GM_CUDA_SOURCES})
#CUDA_ADD_CUFFT_TO_TARGET(gmcuda)
target_link_libraries (gmcuda ${LIBLOC} cufft)

set(GM_SOURCES
    "gm/Thread.cc"
    "gm/bladerf/bladerf.cc"
    "gm/zmqcode/zmqserver.cc" 
    "gm/zmqcode/zmqworker.cc"
    "gm/zmqcode/ZMQPub.cc"
    "gm/sound/FMSound.cc"
    "gm/sound/SoundMonitor.cc")

set(GM_LIBS ${LIBLOC} ${LIBLOG} bladeRF ${LIBUSB} zmq)

add_library(lib_granolamatt ${GM_SOURCES})
target_link_libraries (lib_granolamatt ${GM_LIBS})

add_library(lib_wav "gm/wavfile/wavfile.c")

add_executable(blade_capture "gm/main.cc")
target_link_libraries (blade_capture lib_granolamatt ${GM_LIBS})

add_executable(zmqclient gm/zmqcode/client.cc)
target_link_libraries (zmqclient ${LIBLOC} zmq)

add_executable(cudatest gm/CudaHostTest.cc)
target_link_libraries (cudatest lib_granolamatt gmcuda ${GM_LIBS})

add_executable(uhf_rx gm/UHFRx.cc)
target_link_libraries (uhf_rx lib_granolamatt gmcuda ${GM_LIBS} lib_wav)

add_executable(copytest gm/copytest.cc)
target_link_libraries (copytest lib_granolamatt gmcuda ${GM_LIBS})

#adb push cudatest /data/local/develop/
