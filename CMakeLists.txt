cmake_minimum_required (VERSION 3.5)

project("GronolaSDR")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT PYTHON_EXECUTABLE)
   find_program(PYTHON_EXECUTABLE python)
endif(NOT PYTHON_EXECUTABLE)

execute_process(
	COMMAND ${PYTHON_EXECUTABLE} "-c" 
	"import pybind11 as _; [_] = _.__path__; print(_)"
	RESULT_VARIABLE _pybind11_status
	OUTPUT_VARIABLE _pybind11_location
	OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NOT _pybind11_status)
	set(pybind11_DIR ${_pybind11_location}/share/cmake/pybind11)
endif()

find_package(Python COMPONENTS Interpreter Development)
find_package(PythonLibs)
find_package(pybind11 REQUIRED)

set (CMAKE_BUILD_TYPE Debug)
#set (CMAKE_BUILD_TYPE RelWithDebInfo)
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${PYTHON_INCLUDE_DIRS} "/opt/cuda/include" "/opt/cuda/targets/x86_64-linux/include/cccl")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

set(LIBUSB usb pthread rt)
set(LIBLOG)
set(LIBLOC "-L/opt/cuda/lib64")

set(GM_CUDA_SOURCES
    "gm/cuda/HostCuda.cu"
    "gm/cuda/OverlapSave.cc"
    "gm/cuda/CopyBuffer.cc"
    "gm/cuda/HFChannelizer.cc"
    "gm/cuda/FT8Cuda.cc"
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
    "gm/hf/ft8.cc"
    "gm/zmqcode/ZMQPub.cc")

set(GM_LIBS ${LIBLOC} ${LIBLOG} bladeRF sddc ${LIBUSB} zmq)

add_library(lib_granolamatt ${GM_SOURCES})
target_link_libraries (lib_granolamatt ${GM_LIBS})

add_executable(blade_capture "gm/main.cc")
target_link_libraries (blade_capture lib_granolamatt ${GM_LIBS})

add_executable(rx888_capture "gm/rx888_test.cc")
target_link_libraries (rx888_capture lib_granolamatt ${GM_LIBS})

# -fsanitize=address

add_executable(hf_rx "gm/HFRx.cc")
target_link_libraries (hf_rx -lm lib_granolamatt gmcuda ${CMAKE_CURRENT_SOURCE_DIR}/ft8_lib/libft8.a ${GM_LIBS})

add_executable(zmqclient gm/zmqcode/client.cc)
target_link_libraries (zmqclient ${LIBLOC} zmq)

add_executable(buffertest gm/test_bufferpos.cc)
target_link_libraries (buffertest lib_granolamatt gmcuda ${GM_LIBS})

add_executable(copytest gm/copytest.cc)
target_link_libraries (copytest lib_granolamatt gmcuda ${GM_LIBS})

#adb push cudatest /data/local/develop/

pybind11_add_module(granolasdr gm/PythonModel.cc)
target_link_libraries(granolasdr PRIVATE lib_granolamatt ${GM_LIBS})
