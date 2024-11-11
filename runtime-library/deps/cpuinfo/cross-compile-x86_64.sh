BASE_DIR="$(
  cd "$(dirname "$0")"
  pwd
)"
cd $BASE_DIR
cd source || exit 1


export PATH=/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/bin:/usr/bin:/usr/local/bin:/snap/bin
export CC=/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/bin/x86_64-unknown-linux-gnu-gcc
export CXX=/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/bin/x86_64-unknown-linux-gnu-g++

rm -rf build
mkdir build
cd build

cmake .. -DCMAKE_SYSTEM_PROCESSOR=x86_64 \
-DCPUINFO_TARGET_PROCESSOR=x86 \
-DCPUINFO_BUILD_BENCHMARKS=OFF \
-DCPUINFO_RUNTIME_TYPE=static \
-DCMAKE_SYSTEM_NAME=Linux \
-DCMAKE_SYSROOT=/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/x86_64-unknown-linux-gnu/sysroot \
-DCMAKE_C_COMPILER=/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/bin/x86_64-unknown-linux-gnu-gcc \
-DCMAKE_CXX_COMPILER=/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/bin/x86_64-unknown-linux-gnu-g++ \
-DCMAKE_FIND_ROOT_PATH_MODE_PROGRAM=NEVER \
-DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY \
-DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=ONLY \
-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ONLY \
-DCMAKE_POSITION_INDEPENDENT_CODE=ON

make -j

cd ../..

rm -rf X86_64
mkdir X86_64
rm -rf include
mkdir include

cp ./source/build/*.a ./X86_64
cp ./source/include/*.h ./include
