#!/bin/bash
set -e

cd "$(dirname "$0")"

BUILD_DIR="$(pwd)/build"
ARTIFACTS_DIR="$(pwd)/artifacts"
ROOT_DIR="$(pwd)/.."

mkdir -p $BUILD_DIR
rm -rf $ARTIFACTS_DIR
mkdir -p $ARTIFACTS_DIR

VERSION_FILE="$ROOT_DIR/VERSION"
RUNTIME_VERSION="$(cat $VERSION_FILE)"

echo "Building OpenVINO Native Runtime for version: $RUNTIME_VERSION"

# Detect OpenVINO installation
OPENVINO_DIR="${OPENVINO_DIR:-/usr/local/lib/python3.10/dist-packages/openvino}"

if [ ! -f "$OPENVINO_DIR/include/openvino/openvino.hpp" ]; then
    echo "Error: OpenVINO not found at $OPENVINO_DIR"
    echo "Please set OPENVINO_DIR environment variable or install OpenVINO"
    exit 1
fi

echo "Using OpenVINO from: $OPENVINO_DIR"

cd ${BUILD_DIR}

rm -rf *
cmake .. \
    -DPLATFORM=X86_64 \
    -DCMAKE_BUILD_TYPE=Release \
    -DRUNTIME_VERSION="$RUNTIME_VERSION" \
    -DOPENVINO_DIR="$OPENVINO_DIR"

make -j$(nproc)

echo "Build complete. The following shared libraries were created:"
ls ./*.so

# Clear executable stack bit for security
if command -v execstack &> /dev/null; then
    execstack -c ./*.so* || true
fi

echo "Copying shared libraries to artifacts directory..."
mkdir -p ${ARTIFACTS_DIR}/X86_64
cp ./*.so* ${ARTIFACTS_DIR}/X86_64

cd ${ARTIFACTS_DIR}/X86_64
tar czf ${ARTIFACTS_DIR}/runtime-library-X86_64.tar.gz ./*

echo "Shared libraries have been copied to ${ARTIFACTS_DIR}"
echo "Archive created: ${ARTIFACTS_DIR}/runtime-library-X86_64.tar.gz"
