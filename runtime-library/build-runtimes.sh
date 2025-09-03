set -e

cd "$(dirname "$0")"

BUILD_DIR="$(pwd)/build"
ARTIFACTS_DIR="$(pwd)/artifacts"
mkdir -p $BUILD_DIR
rm -rf $ARTIFACTS_DIR
mkdir -p $ARTIFACTS_DIR

cd ${BUILD_DIR}

rm -rf *
cmake .. -DPLATFORM=X86_64
make -j
echo "Build complete. The following shared libraries were created:"
ls ./*.so
echo "Copying shared libraries to artifacts directory..."
mkdir -p ${ARTIFACTS_DIR}/X86_64
cp ./*.so* ${ARTIFACTS_DIR}/X86_64
cd ${ARTIFACTS_DIR}/X86_64
tar czf ${ARTIFACTS_DIR}/runtime-library-X86_64.tar.gz ./*
echo "Shared libraries have been copied to ${ARTIFACTS_DIR}"
