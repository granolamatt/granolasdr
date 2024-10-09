#!/bin/sh

export ANDROID_NDK_HOME=${NVPACK_ROOT}/android-ndk-r10e
export ANDROID_STANDALONE_TOOLCHAIN=${NVPACK_ROOT}/standalone-v10-ndk
export CC=${NVPACK_ROOT}/standalone-v10-ndk/bin/aarch64-linux-android-gcc
export CXX=${NVPACK_ROOT}/standalone-v10-ndk/bin/aarch64-linux-android-g++
export PATH=${NVPACK_ROOT}/cuda-7.0/bin:${PATH}
export ANDROIDPATH=/home/matt/shield
