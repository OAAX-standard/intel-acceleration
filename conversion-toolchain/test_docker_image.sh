#!/bin/bash
# Docker Image Validation Script
# Tests the oaax-intel-toolchain Docker image

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "Docker Image Validation for oaax-intel-toolchain"
echo "========================================="
echo ""

# Function to print success
success() {
    echo -e "${GREEN}✓${NC} $1"
}

# Function to print error
error() {
    echo -e "${RED}✗${NC} $1"
    exit 1
}

# Function to print info
info() {
    echo -e "${YELLOW}→${NC} $1"
}

# 1. Check Docker
info "Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    error "Docker is not running or not accessible"
fi
success "Docker is running"

# 2. Build image
echo ""
info "Building Docker image (this may take a few minutes)..."
if docker build -t oaax-intel-toolchain:latest . > /tmp/docker_build.log 2>&1; then
    success "Image built successfully"
else
    error "Image build failed (check /tmp/docker_build.log for details)"
fi

# 3. Test help command
echo ""
info "Testing help command..."
if docker run --rm oaax-intel-toolchain:latest --help > /dev/null 2>&1; then
    success "Help command works"
else
    error "Help command failed"
fi

# 4. Prepare test data
echo ""
info "Preparing test data..."
mkdir -p bundles output 2>/dev/null || true

if [ ! -f "bundles/squeezenet.onnx" ]; then
    info "Downloading test model (SqueezeNet)..."

    # Try multiple download sources
    DOWNLOADED=false

    # Try validated path first
    if wget -q --timeout=30 \
        "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-7.onnx" \
        -O bundles/squeezenet.onnx 2>/dev/null; then
        DOWNLOADED=true
    # Try old path
    elif wget -q --timeout=30 \
        "https://github.com/onnx/models/raw/main/vision/classification/squeezenet/model/squeezenet1.0-7.onnx" \
        -O bundles/squeezenet.onnx 2>/dev/null; then
        DOWNLOADED=true
    fi

    if [ "$DOWNLOADED" = true ]; then
        success "Test model downloaded"
    else
        info "Could not download from GitHub, trying alternative..."

        # Try using existing test models from pytest
        if [ -d "tests/test_models" ]; then
            EXISTING_MODEL=$(find tests/test_models -name "*.onnx" -type f | head -1)
            if [ -n "$EXISTING_MODEL" ] && [ -f "$EXISTING_MODEL" ]; then
                cp "$EXISTING_MODEL" bundles/squeezenet.onnx
                success "Using existing test model: $(basename $EXISTING_MODEL)"
                DOWNLOADED=true
            fi
        fi

        # If still no model, try downloading with Python
        if [ "$DOWNLOADED" = false ]; then
            info "Attempting to download with Python..."
            if python3 tests/download_test_models.py --output-dir tests/test_models 2>/dev/null; then
                EXISTING_MODEL=$(find tests/test_models -name "*.onnx" -type f | head -1)
                if [ -n "$EXISTING_MODEL" ] && [ -f "$EXISTING_MODEL" ]; then
                    cp "$EXISTING_MODEL" bundles/squeezenet.onnx
                    success "Downloaded test model with Python"
                    DOWNLOADED=true
                fi
            fi
        fi

        if [ "$DOWNLOADED" = false ]; then
            error "Failed to download test model. Options:\n  1. Download manually from https://github.com/onnx/models\n  2. Run: python tests/download_test_models.py\n  3. Place any ONNX model in bundles/squeezenet.onnx"
        fi
    fi
else
    success "Test model already exists"
fi

# Create bundle
cd bundles
if zip -q squeezenet.zip squeezenet.onnx 2>/dev/null; then
    cd ..
    success "Test bundle created"
else
    cd ..
    error "Failed to create test bundle"
fi

# 5. Run conversion
echo ""
info "Running model conversion..."
rm -f output/squeezenet.zip output/logs.json 2>/dev/null || true

if docker run --rm \
    -v $(pwd)/bundles:/input \
    -v $(pwd)/output:/output \
    oaax-intel-toolchain:latest /input/squeezenet.zip /output > /tmp/conversion.log 2>&1; then
    success "Conversion completed successfully"
else
    error "Conversion failed (check /tmp/conversion.log and output/logs.json)"
fi

# 6. Verify output
echo ""
info "Verifying output files..."

if [ -f "output/squeezenet.zip" ]; then
    SIZE=$(du -h output/squeezenet.zip | cut -f1)
    success "Output zip created (size: $SIZE)"
else
    error "Output zip not found"
fi

if [ -f "output/logs.json" ]; then
    success "Logs file created"
else
    error "Logs file not found"
fi

# Check zip contents
if unzip -l output/squeezenet.zip | grep -q "squeezenet.xml" && \
   unzip -l output/squeezenet.zip | grep -q "squeezenet.bin"; then
    success "Zip contains .xml and .bin files"
else
    error "Zip contents invalid"
fi

# 7. Test error handling
echo ""
info "Testing error handling..."

# Test nonexistent file (should return exit code 1)
if docker run --rm \
    -v $(pwd)/bundles:/input \
    -v $(pwd)/output:/output \
    oaax-intel-toolchain:latest /input/nonexistent.zip /output > /dev/null 2>&1; then
    error "Should have failed for nonexistent file"
else
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 1 ]; then
        success "Error handling works (exit code: $EXIT_CODE)"
    else
        error "Wrong exit code: $EXIT_CODE (expected 1)"
    fi
fi

# Test invalid zip (should return exit code 2)
echo "not a zip" > bundles/invalid.zip
if docker run --rm \
    -v $(pwd)/bundles:/input \
    -v $(pwd)/output:/output \
    oaax-intel-toolchain:latest /input/invalid.zip /output > /dev/null 2>&1; then
    error "Should have failed for invalid zip"
else
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 2 ]; then
        success "Invalid zip handling works (exit code: $EXIT_CODE)"
    else
        error "Wrong exit code: $EXIT_CODE (expected 2)"
    fi
fi

# 8. Test with resource limits
echo ""
info "Testing with resource limits..."
if docker run --rm \
    --memory=2g \
    --cpus=1 \
    -v $(pwd)/bundles:/input \
    -v $(pwd)/output:/output \
    oaax-intel-toolchain:latest /input/squeezenet.zip /output > /dev/null 2>&1; then
    success "Works with resource limits"
else
    error "Failed with resource limits"
fi

# 9. Summary
echo ""
echo "========================================="
echo -e "${GREEN}All Validation Tests Passed!${NC}"
echo "========================================="
echo ""
echo "Image Details:"
docker images oaax-intel-toolchain:latest --format "  Name: {{.Repository}}:{{.Tag}}\n  Size: {{.Size}}\n  Created: {{.CreatedSince}}"
echo ""
echo "Next Steps:"
echo "  1. Run Docker integration tests: pytest tests/test_docker.py -v"
echo "  2. Test with your own models"
echo "  3. Deploy to production"
echo ""
echo "Usage:"
echo "  docker run --rm \\"
echo "    -v \$(pwd)/bundles:/input \\"
echo "    -v \$(pwd)/output:/output \\"
echo "    oaax-intel-toolchain:latest /input/your_model.zip /output"
echo ""
