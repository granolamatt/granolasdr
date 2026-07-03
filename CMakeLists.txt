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
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/third_party ${PYTHON_INCLUDE_DIRS} "/opt/cuda/include" "/opt/cuda/targets/x86_64-linux/include/cccl")

# wsserver — Rust WebSocket dict server, built as a static library
set(WSSERVER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/wsserver)
set(WSSERVER_LIB ${WSSERVER_DIR}/target/release/libwsserver.a)
# Use ALL so CMake always delegates to cargo; cargo's own fingerprinting
# handles incremental rebuilds, so this is a no-op when nothing has changed.
add_custom_target(wsserver_lib ALL
    COMMAND cargo build --release --lib
    WORKING_DIRECTORY ${WSSERVER_DIR}
    BYPRODUCTS ${WSSERVER_LIB}
    COMMENT "Building wsserver Rust static library"
)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70;75;80;86;89;90;120)
endif()

set(LIBUSB usb-1.0 pthread rt)
set(LIBLOG)
set(LIBLOC "-L/opt/cuda/lib64")

set(GM_CUDA_SOURCES
    "gm/cuda/HostCuda.cu"
    "gm/cuda/ChiFilter.cu"
    "gm/cuda/FT8ScanCuda.cu"
    "gm/cuda/FT8SoftCuda.cu"
    "gm/cuda/FT8LdpcCuda.cu"
    "gm/cuda/JS8ScanCuda.cu"
    "gm/cuda/JS8FastScanCuda.cu"
    "gm/cuda/JS8LdpcCuda.cu"
    "gm/cuda/OsdCore.cc"
    "gm/cuda/JS8Osd.cc"
    "gm/cuda/FT8Osd.cc"
    "gm/cuda/JS8SoftCuda.cu"
    "gm/cuda/WaterfallKernel.cu"
    "gm/cuda/WaterfallCuda.cc"
    "gm/cuda/SpectrumNorm.cu"
    "gm/cuda/OverlapSave.cc"
    "gm/cuda/CopyBuffer.cc"
    "gm/buffer/BufferFile.cc"
    "gm/cuda/HFChannelizer.cc"
    "gm/cuda/MagBlock.cc"
    "gm/cuda/FT8Cuda.cc"
    "gm/cuda/JS8Cuda.cc"
    "ft8_lib/fft/kiss_fft.c"
    )

enable_language("CUDA")

execute_process(COMMAND ${CMAKE_C_COMPILER} -dumpversion
                OUTPUT_VARIABLE GCC_VERSION)


add_library (gmcuda ${GM_CUDA_SOURCES})
target_link_libraries (gmcuda ${LIBLOC} cufft)

set(GM_SOURCES
    "gm/Thread.cc"
    "gm/rx888/rx888.cc"
    "gm/zmqcode/zmqserver.cc"
    "gm/zmqcode/zmqworker.cc"
    "gm/hf/band_map.cc"
    "gm/hf/llr_capture.cc"
    "gm/hf/ft8.cc"
    "gm/hf/js8.cc"
    "gm/hf/jsc.cc"
    "gm/hf/jsc_map.cc"
    "gm/hf/jsc_list.cc"
    "gm/zmqcode/ZMQPub.cc")

set(GM_LIBS ${LIBLOC} ${LIBLOG} sddc ${LIBUSB} zmq)

add_library(lib_granolamatt ${GM_SOURCES})
target_link_libraries (lib_granolamatt ${GM_LIBS})

add_executable(rx888_capture "gm/rx888_test.cc")
target_link_libraries (rx888_capture lib_granolamatt ${GM_LIBS})

# -fsanitize=address

add_executable(hf_rx "gm/HFRx.cc")
add_dependencies(hf_rx wsserver_lib)
target_link_libraries (hf_rx -lm lib_granolamatt gmcuda ${CMAKE_CURRENT_SOURCE_DIR}/ft8_lib/libft8.a ${GM_LIBS} ${WSSERVER_LIB} dl gcc_s util rt)

# Keep the control/ static files in sync with the build directory.
add_custom_command(TARGET hf_rx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/control
        ${CMAKE_CURRENT_BINARY_DIR}/control
    COMMENT "Syncing control/ static files to build directory"
)

add_executable(zmqclient gm/zmqcode/client.cc)
target_link_libraries (zmqclient ${LIBLOC} zmq)

add_executable(decode_corpus decode_corpus.cc ft8_lib/fft/kiss_fft.c)
target_link_libraries (decode_corpus ${CMAKE_CURRENT_SOURCE_DIR}/ft8_lib/libft8.a -lm)

# Offline coherent-demod experiment harness (host-only, no CUDA).
add_executable(coherent_offline coherent_offline.cc)
target_link_libraries (coherent_offline -lm)

add_executable(test_ldpc_sign test/test_ldpc_sign.cc)
target_link_libraries(test_ldpc_sign gmcuda ${CMAKE_CURRENT_SOURCE_DIR}/ft8_lib/libft8.a -lm cudart)

# Host-only OSD test: compiles JS8Osd.cc directly, no CUDA link needed.
add_executable(test_osd test/test_osd.cc gm/cuda/JS8Osd.cc gm/cuda/OsdCore.cc)
target_link_libraries(test_osd -lm)

# Host-only CW Morse decoder test (Phase 0 gate): pure CPU, no CUDA link.
add_executable(test_cw_morse test/test_cw_morse.cc gm/hf/cw_morse.cc)
target_link_libraries(test_cw_morse -lm)

# FT8 OSD test: links ft8_lib for the generator table, CRC-14 and ftx_decode_from_bits.
add_executable(test_ft8_osd test/test_ft8_osd.cc gm/cuda/FT8Osd.cc gm/cuda/OsdCore.cc)
target_link_libraries(test_ft8_osd ${CMAKE_CURRENT_SOURCE_DIR}/ft8_lib/libft8.a -lm)

add_executable(test_ldpc_captures test/test_ldpc_captures.c)
target_link_libraries(test_ldpc_captures ${CMAKE_CURRENT_SOURCE_DIR}/ft8_lib/libft8.a -lm)

add_executable(test_js8_wav test/test_js8_wav.cc ft8_lib/fft/kiss_fft.c)
target_link_libraries(test_js8_wav gmcuda ${CMAKE_CURRENT_SOURCE_DIR}/ft8_lib/libft8.a -lm cudart)

add_executable(test_js8_fast_gpu_scan test/test_js8_fast_gpu_scan.cc)
target_link_libraries(test_js8_fast_gpu_scan gmcuda -lm cudart)


#adb push cudatest /data/local/develop/

pybind11_add_module(granolasdr gm/PythonModel.cc)
target_link_libraries(granolasdr PRIVATE lib_granolamatt ${GM_LIBS})

# Offline FT8 decode binding for the coherent/overlap experiments (host-only).
# Compile the ft8_lib sources in directly (libft8.a is non-PIC, can't link into a .so).
pybind11_add_module(ft8decode coherent/ft8decode.cc
    ft8_lib/ft8/decode.c ft8_lib/ft8/message.c ft8_lib/ft8/ldpc.c
    ft8_lib/ft8/crc.c ft8_lib/ft8/constants.c ft8_lib/ft8/text.c
    ft8_lib/fft/kiss_fftr.c ft8_lib/fft/kiss_fft.c)
target_include_directories(ft8decode PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/ft8_lib/ft8 ${CMAKE_CURRENT_SOURCE_DIR}/ft8_lib)
set_target_properties(ft8decode PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/coherent)
