#!/bin/bash
# Build script for OpenVINO Conversion Toolchain Docker image
set -e

IMAGE_NAME="${IMAGE_NAME:-openvino-converter}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

echo "========================================="
echo "Building OpenVINO Conversion Toolchain"
echo "========================================="
echo ""
echo "Image: ${FULL_IMAGE}"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Build the Docker image
echo "Building Docker image..."
docker build -t "${FULL_IMAGE}" .

# Verify the build
echo ""
echo "Verifying build..."
docker run --rm "${FULL_IMAGE}" --help

echo ""
echo "========================================="
echo "Build successful!"
echo "========================================="
echo ""
echo "Image: ${FULL_IMAGE}"
echo ""
echo "Usage:"
echo "  docker run -v \$(pwd)/input:/input -v \$(pwd)/output:/output \\"
echo "             ${FULL_IMAGE} /input/model.zip /output"
echo ""
