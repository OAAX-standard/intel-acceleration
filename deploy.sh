#!/usr/bin/env bash
# deploy.sh — build (optionally with profiling) and copy the runtime library
#             to the local inference engine directory.
#
# Usage:
#   bash deploy.sh [--dest <path>] [--profile]
#
# Options:
#   --profile   Build with OAAX_PROFILE=ON (per-request timing logs)
#   --dest      Override destination directory
#
# Default destination: ~/witness-ai-manager/inference_engine/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT}/runtime-library/build"
DEST="${HOME}/witness-ai-manager/inference_engine"
PROFILE=0

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest) DEST="$2"; shift 2 ;;
        --profile) PROFILE=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Build ─────────────────────────────────────────────────────────────────────
OPENVINO_DIR="${OPENVINO_DIR:-/opt/intel/openvino/runtime}"
VERSION_FILE="${ROOT}/VERSION"
RUNTIME_VERSION="$(cat "${VERSION_FILE}")"

mkdir -p "${BUILD_DIR}"

CMAKE_ARGS=(
    "-DPLATFORM=X86_64"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DRUNTIME_VERSION=${RUNTIME_VERSION}"
    "-DOPENVINO_DIR=${OPENVINO_DIR}"
)
if [[ "${PROFILE}" == "1" ]]; then
    CMAKE_ARGS+=("-DOAAX_PROFILE=ON")
    echo "=== Building with OAAX_PROFILE=ON ==="
else
    echo "=== Building runtime library ==="
fi

cmake "${ROOT}/runtime-library" "${CMAKE_ARGS[@]}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" -j "$(nproc)"

if [[ ! -d "${BUILD_DIR}" ]]; then
    echo "ERROR: build directory not found: ${BUILD_DIR}"
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
