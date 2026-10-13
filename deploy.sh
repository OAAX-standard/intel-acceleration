#!/usr/bin/env bash
# deploy.sh — build and copy the runtime library to a local inference engine directory.
#
# Usage:
#   bash deploy.sh --dest <path>
#   OAAX_DEPLOY_DEST=<path> bash deploy.sh
#
# Options:
#   --dest      Destination directory (or set OAAX_DEPLOY_DEST); required

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${ROOT}/runtime-library/build"
DEST="${OAAX_DEPLOY_DEST:-}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dest) DEST="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

[[ -n "${DEST}" ]] || { echo "ERROR: no destination — pass --dest <path> or set OAAX_DEPLOY_DEST"; exit 1; }

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
echo "=== Building runtime library ==="

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
