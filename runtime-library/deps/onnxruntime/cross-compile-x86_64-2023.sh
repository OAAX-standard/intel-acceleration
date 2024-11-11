#!/bin/bash
# Go through this first: https://docs.openvino.ai/2024/get-started/install-openvino/install-openvino-archive-linux.html
set -e

BASE_DIR="$(cd "$(dirname "$0")"; pwd)";
cd $BASE_DIR

ort_version="1.16.3"
ort_folder_name="onnxruntime-${ort_version}"
cd $ort_folder_name

rm -rf build

export CPATH="$BASE_DIR/../nlohmann/single_include"

export CROSS_ROOT="/opt/x86_64-unknown-linux-gnu-gcc-9.5.0"
export COMPILER_PREFIX="x86_64-unknown-linux-gnu-"
export PATH=$CROSS_ROOT/bin:/snap/bin:/bin:/usr/bin:/usr/local/bin
export SYSROOT="/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/x86_64-unknown-linux-gnu/sysroot"

export CPPFLAGS="-I${CROSS_ROOT}/include -L${CROSS_ROOT}/lib"
export CFLAGS="-I${CROSS_ROOT}/include -L${CROSS_ROOT}/lib"
export AR=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ar
export AS=${CROSS_ROOT}/bin/${COMPILER_PREFIX}as
export LD=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld
export RANLIB=${CROSS_ROOT}/bin/${COMPILER_PREFIX}ranlib
export CC=${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc
export CXX=${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++
export NM=${CROSS_ROOT}/bin/${COMPILER_PREFIX}nm

source /opt/intel/openvino_2023/setupvars.sh


./build.sh --config Release --allow_running_as_root --use_openvino --build_shared_lib  --parallel --skip_tests \
  --cmake_extra_defines \
  CMAKE_POSITION_INDEPENDENT_CODE=ON \
  onnxruntime_USE_CUDA=OFF \
  onnxruntime_BUILD_UNIT_TESTS=OFF \
  onnxruntime_USE_NSYNC=OFF  \
  onnxruntime_ENABLE_CPUINFO=ON

cd ..

rm -rf X86_64-2023 || true
mkdir X86_64-2023

cp -rf ./$ort_folder_name/include ./X86_64-2023 || true
RELEASE="./$ort_folder_name/build/Linux/Release"
cp $RELEASE/*.a ./X86_64-2023 2>>/dev/null || true
cp $RELEASE/*.so ./X86_64-2023 2>>/dev/null  || true
shopt -s globstar
cp -rf $RELEASE/_deps/*-build/**/*.a ./X86_64-2023 2>>/dev/null || true
cp -rf $RELEASE/_deps/*-build/**/*.so ./X86_64-2023 2>>/dev/null  || true

cp /opt/intel/openvino_2023/runtime/lib/intel64/libopenvino_c.so.2310 ./X86_64-2023 || true
cp /opt/intel/openvino_2023/runtime/lib/intel64/libopenvino_onnx_frontend.so.2310 ./X86_64-2023 || true
cp /opt/intel/openvino_2023/runtime/lib/intel64/libopenvino.so.2310 ./X86_64-2023 || true
cp /opt/intel/openvino_2023/runtime/lib/intel64/libopenvino_ir_frontend.so.2310 ./X86_64-2023 || true
cp /opt/intel/openvino_2023/runtime/lib/intel64/libopenvino_intel_cpu_plugin.so ./X86_64-2023 || true
# ubuntu 20 supporting libs
cp __openvino_support_libs__/* ./X86_64-2023 || true

printf "\nDone :) \n"
