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

echo "Building for runtime version: $RUNTIME_VERSION"

cd ${BUILD_DIR}

rm -rf *
cmake .. -DPLATFORM=X86_64 -DCMAKE_BUILD_TYPE=Release -DRUNTIME_VERSION="$RUNTIME_VERSION"
make -j
echo "Build complete. The following shared libraries were created:"
ls ./*.so
echo "Copying shared libraries to artifacts directory..."
mkdir -p ${ARTIFACTS_DIR}/X86_64
cp ./*.so* ${ARTIFACTS_DIR}/X86_64
cd ${ARTIFACTS_DIR}/X86_64
tar czf ${ARTIFACTS_DIR}/runtime-library-X86_64.tar.gz ./*
echo "Shared libraries have been copied to ${ARTIFACTS_DIR}"
