# OpenVINO Conversion Toolchain Docker Image
# Converts ONNX models to OpenVINO IR format with FP16/INT8 optimizations
#
# Usage:
#   docker build -t openvino-converter .
#   docker run -v $(pwd)/input:/input -v $(pwd)/output:/output \
#              openvino-converter /input/model.zip /output

ARG DEBIAN_FRONTEND=noninteractive
FROM python:3.10-slim

LABEL maintainer="OAAX Standard"
LABEL description="OpenVINO ONNX to IR Conversion Toolchain"
LABEL version="0.3.0"

# Install system dependencies
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get update -qq -y --fix-missing && \
    apt-get install -qq -y gcc g++ curl make wget git cmake build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv --version
ENV PATH="/root/.local/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy only necessary files for installation (layer caching)
COPY pyproject.toml /app/

# Copy source code
COPY conversion_toolchain /app/conversion_toolchain

# Install Python dependencies
RUN uv pip install --system -e . && \
    uv cache clean

# Create standard directories for mounting
RUN mkdir -p /input /output /workspace

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV OV_LOG_LEVEL=WARNING

# Health check (verify conversion_toolchain is available)
RUN conversion_toolchain --help || exit 1

# Use direct Python entrypoint (not bash script)
ENTRYPOINT ["conversion_toolchain"]
CMD ["--help"]

# Document expected volumes
VOLUME ["/input", "/output"]

# Document that this runs as root (can be overridden)
USER root
