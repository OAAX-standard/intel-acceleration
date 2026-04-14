#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Change the working directory to the directory of the script
cd "$(dirname "$0")"

# Install ubuntu packages
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y wget curl build-essential execstack patchelf python3-pip s3cmd

# Install uv (Python package manager used by the test suite)
pip3 install uv

# Define an array of URLs for downloading toolchains
toolchain_urls=(
    "https://oaax.nbg1.your-objectstorage.com/toolchains/x86_64-unknown-linux-gnu-gcc-9.5.0.tar.gz" # check-secrets-ignore: public download URL, no credentials
)

# Function to extract the filename from a given URL
function get_filename_from_url() {
    local url=$1
    echo "${url##*/}" # Extract the part of the URL after the last '/'
}

# Iterate over all toolchain URLs to download and extract them
for url in "${toolchain_urls[@]}"; do
    filename=$(get_filename_from_url "$url") # Get the filename from the URL
    echo wget -nv -c "$url"                  # Print the wget command for logging
    wget -nv -c "$url"                       # Download the file with minimal output and resume capability
    # Extract the downloaded file to /opt, trying both gzip and non-gzip formats
    tar xzf "$filename" -C /opt 2>/dev/null || tar xf "$filename" -C /opt
    # Remove the downloaded file after extraction
    rm -rf "$filename" || true
    # Log the successful extraction of the file
    echo ">>>>>>>>>>> extracted: $filename"
done

# Install cmake
host_platform=$(uname -m)
wget "https://cmake.org/files/v3.31/cmake-3.31.7-linux-${host_platform}.sh" \
    -q -O /tmp/cmake-install.sh &&
    chmod u+x /tmp/cmake-install.sh &&
    mkdir /opt/cmake-3.31.7 &&
    /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-3.31.7 &&
    rm /tmp/cmake-install.sh &&
    ln -fs /opt/cmake-3.31.7/bin/* /usr/local/bin

# Install OpenVINO runtime archive
OV_ARCHIVE="openvino_toolkit_ubuntu22_2026.1.0.21367.63e31528c62_x86_64.tgz" # check-secrets-ignore: public download URL, no credentials
wget -q "https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.1/linux/${OV_ARCHIVE}" \
    -O /tmp/openvino.tgz
mkdir -p /opt/intel/openvino
tar -xzf /tmp/openvino.tgz -C /opt/intel/openvino --strip-components=1
rm /tmp/openvino.tgz
echo ">>>>>>>>>>> OpenVINO installed to /opt/intel/openvino"
