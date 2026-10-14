#!/bin/bash
# Build the C++ runtime tests against the prebuilt runtime library.
# Run build-runtimes.sh first so runtime-library/build/ is populated.
set -e

cd "$(dirname "$0")"

RUNTIME_LIB_DIR="${RUNTIME_LIB_DIR:-$(pwd)/../../runtime-library/build}"
BUILD_DIR="$(pwd)/build"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. -DRUNTIME_LIB_DIR="$RUNTIME_LIB_DIR" -DCMAKE_BUILD_TYPE=Release
make -j "$(nproc)"

# Symlink shared libs into the build dir so $ORIGIN RPATH resolves at runtime
for lib in "$RUNTIME_LIB_DIR"/*.so*; do
    [ -e "$lib" ] || continue
    ln -sf "$lib" .
done

echo "Tests built in: $BUILD_DIR"
