#!/bin/bash
# Build script for OpenVINO Conversion Toolchain Docker image
set -e

IMAGE_NAME="${IMAGE_NAME:-openvino-converter}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# Always build from the repo root so that both VERSION and
# conversion-toolchain/ are available in the Docker build context.
cd "$(dirname "$0")/.."

VERSION=$(cat VERSION)

echo "========================================="
echo "Building OpenVINO Conversion Toolchain"
echo "========================================="
echo ""
echo "Image   : ${FULL_IMAGE}"
echo "Version : ${VERSION}"
echo ""

# Build the Docker image
echo "Building Docker image..."
docker build \
    -f conversion-toolchain/Dockerfile \
    --build-arg VERSION="${VERSION}" \
    -t "${FULL_IMAGE}" \
    .

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
