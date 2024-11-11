export PATH=/snap/bin:/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/bin:/usr/bin:/usr/local/bin
export CC=/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/bin/x86_64-unknown-linux-gnu-gcc
export CXX=/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/bin/x86_64-unknown-linux-gnu-g++

cd source
rm -rf build
mkdir build
cd build

cmake .. -DCMAKE_POSITION_INDEPENDENT_CODE=ON

make -j
sudo make

cd ../..
rm -rf X86_64 include 
mkdir X86_64
mkdir include

cp ./source/build/libre2.a ./X86_64
cp -rf ./source/re2 ./include
