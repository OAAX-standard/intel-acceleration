#!/usr/bin/env bash
# deploy.sh — copy the built runtime library to the local inference engine directory
#
# Usage:
#   bash deploy.sh [--dest <path>]
#
# Default destination: ~/witness-ai-manager/inference_engine/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT}/runtime-library/build"
DEST="${1:-${HOME}/witness-ai-manager/inference_engine}"

# Allow --dest <path> flag
if [[ "${1:-}" == "--dest" && -n "${2:-}" ]]; then
    DEST="$2"
fi

if [[ ! -d "${BUILD_DIR}" ]]; then
    echo "ERROR: build directory not found: ${BUILD_DIR}"
    echo "       Run 'bash runtime-library/build-runtimes.sh' first."
    exit 1
fi

if [[ ! -d "${DEST}" ]]; then
    echo "ERROR: destination directory not found: ${DEST}"
    exit 1
fi

echo "Deploying runtime library to ${DEST} ..."

RUNTIME_LIB="${BUILD_DIR}/libRuntimeLibrary.so"
if [[ ! -f "${RUNTIME_LIB}" ]]; then
    echo "ERROR: libRuntimeLibrary.so not found in ${BUILD_DIR}"
    exit 1
fi

deploy_file() {
    local src="$1"
    local dst
    dst="${DEST}/$(basename "${src}")"
    # Remove first (directory is world-writable even when files are root-owned)
    rm -f "${dst}"
    cp -v "${src}" "${dst}"
}

deploy_file "${RUNTIME_LIB}"

# Also sync OpenVINO .so files that may have been updated
for so in "${BUILD_DIR}"/libopenvino*.so*; do
    [[ -f "${so}" ]] && deploy_file "${so}"
done

echo "Done."
