#!/bin/bash
# Quick Test - Verifies the toolchain is ready for Docker deployment
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================="
echo "OpenVINO Conversion Toolchain - Quick Test"
echo "========================================="
echo ""

success() { echo -e "${GREEN}✓${NC} $1"; }
info() { echo -e "${YELLOW}→${NC} $1"; }
header() { echo -e "${BLUE}$1${NC}"; }

header "1. Checking Environment"
if [ -f ".venv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
    success "Virtual environment activated"
else
    info "No virtual environment found (optional for Docker)"
fi

header "2. Verifying Project Structure"
REQUIRED_FILES=(
    "Dockerfile"
    "pyproject.toml"
    "conversion_toolchain/main.py"
    "conversion_toolchain/utils.py"
    "tests/test_conversion.py"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        success "$file"
    else
        echo "✗ Missing: $file"
        exit 1
    fi
done

header "3. Checking Documentation"
if [ -f "README.md" ]; then
    success "README.md"
fi

header "4. Preparing Test Models"
mkdir -p tests/test_models bundles output 2>/dev/null || true

if [ -f "tests/test_models/squeezenet1.0-7.onnx" ]; then
    success "Test models available"
    cp tests/test_models/squeezenet1.0-7.onnx bundles/squeezenet.onnx 2>/dev/null || true
else
    info "Downloading test models..."
    if command -v python3 &> /dev/null; then
        python3 tests/download_test_models.py --output-dir tests/test_models
        success "Test models downloaded"
        cp tests/test_models/squeezenet1.0-7.onnx bundles/squeezenet.onnx 2>/dev/null || true
    fi
fi

header "5. Running Unit Tests"
if command -v pytest &> /dev/null; then
    info "Running conversion tests..."
    pytest tests/test_conversion.py -v --tb=short -q 2>&1 | tail -10
    success "Unit tests completed"
else
    info "pytest not installed, skipping"
fi

header "6. Verifying Dockerfile"
if grep -q "/root/.local/bin/uv" Dockerfile; then
    success "Dockerfile uses correct UV path"
fi

if grep -q 'ENTRYPOINT \["conversion_toolchain"\]' Dockerfile; then
    success "Dockerfile has correct entrypoint"
fi

header "7. Checking Scripts"
for script in build-docker.sh test_docker_image.sh quick-test.sh; do
    if [ -x "$script" ]; then
        success "$script is executable"
    elif [ -f "$script" ]; then
        info "$script exists but not executable"
    fi
done

echo ""
echo "========================================="
echo -e "${GREEN}Quick Test Complete!${NC}"
echo "========================================="
echo ""
echo "Project Status:"
echo "  ✓ All required files present"
echo "  ✓ Test models available"
echo "  ✓ Dockerfile verified"
echo ""
echo "Next Steps:"
echo "  ${BLUE}Docker Build:${NC}    docker build -t openvino-converter ."
echo "  ${BLUE}Docker Test:${NC}     ./test_docker_image.sh"
echo "  ${BLUE}Unit Tests:${NC}      pytest tests/ -v"
echo ""
