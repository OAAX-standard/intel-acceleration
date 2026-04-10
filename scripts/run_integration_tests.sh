#!/bin/bash
# End-to-end integration tests: toolchain (Python) + runtime (C++)
# Tests YOLOv8n and YOLOv11n through the full OAAX pipeline.
#
# Usage:
#   bash scripts/run_integration_tests.sh [--skip-runtime] [--device CPU|GPU|NPU]
#
# Requirements:
#   - uv installed
#   - C++ runtime built: cd runtime-library && bash build-runtimes.sh
#
set -e
cd "$(dirname "$0")/.."

ROOT_DIR="$(pwd)"
RUNTIME_DIR="$ROOT_DIR/runtime-library"
CONVERTED_DIR="/tmp/oaax_integration_converted"
DEVICE="CPU"
SKIP_RUNTIME=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-runtime) SKIP_RUNTIME=1; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

pass()   { echo -e "${GREEN}✓${NC} $1"; }
fail()   { echo -e "${RED}✗${NC} $1"; exit 1; }
header() { echo -e "\n${BLUE}=== $1 ===${NC}"; }

# ── 1. Set up environment ─────────────────────────────────────────────────────

header "Step 1: Setting up environment"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi
source .venv/bin/activate

echo "Syncing dependencies..."
uv sync --extra integration --extra quantization -q

pass "Environment ready"

# ── 2. Python integration tests ───────────────────────────────────────────────

header "Step 2: Python integration tests (YOLOv8n + YOLOv11n)"

pytest tests/test_yolo_integration.py -v --tb=short

pass "Python integration tests passed"

# ── 3. Convert models for C++ runtime tests ───────────────────────────────────

if [ "$SKIP_RUNTIME" -eq 1 ]; then
    echo "Skipping C++ runtime tests (--skip-runtime)"
    exit 0
fi

header "Step 3: Converting models for C++ runtime tests"

mkdir -p "$CONVERTED_DIR"

python3 - <<EOF
import sys
sys.path.insert(0, "$ROOT_DIR")
from tests.models import download_model
from conversion_toolchain.utils import convert_to_ir
from conversion_toolchain.logger import Logs
import zipfile
from pathlib import Path

converted_dir = Path("$CONVERTED_DIR")

for model_name in ["yolov8n", "yolo11n"]:
    print(f"Preparing {model_name}...")
    models_dir = Path("$ROOT_DIR/tests/test_models")
    models_dir.mkdir(exist_ok=True)
    onnx_path = download_model(model_name, str(models_dir))

    out_dir = converted_dir / model_name
    logs = Logs()
    zip_path = convert_to_ir(onnx_path, str(out_dir), logs)

    ir_dir = out_dir / "ir"
    ir_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(ir_dir)

    xml = next(ir_dir.glob("*.xml"))
    print(f"  ✓ IR ready: {xml}")
EOF

pass "Models converted"

# ── 4. C++ runtime integration tests ─────────────────────────────────────────

header "Step 4: C++ runtime integration tests (device=$DEVICE)"

BUILD_DIR="$RUNTIME_DIR/build"
if [ ! -f "$BUILD_DIR/yolo_test" ]; then
    echo "Building test binaries..."
    OPENVINO_DIR="${OPENVINO_DIR:-/usr/local/lib/python3.10/dist-packages/openvino}"
    CMAKE_BIN="${CMAKE_BIN:-/usr/bin/cmake}"

    # The OpenVINO Python package ships only versioned .so files (e.g. libopenvino.so.2450).
    # The linker needs unversioned symlinks (libopenvino.so). Create them in a local dir.
    LINK_DIR="$BUILD_DIR/openvino_links"
    mkdir -p "$LINK_DIR"
    for lib in "$OPENVINO_DIR/libs/"*.so.*; do
        base=$(basename "$lib")
        # libopenvino.so.2450 -> libopenvino.so  |  libtbb.so.12 -> libtbb.so
        unversioned="${base%%\.*}.so"
        ln -sf "$lib" "$LINK_DIR/$unversioned" 2>/dev/null || true
    done

    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
    "$CMAKE_BIN" .. \
        -DPLATFORM=X86_64 \
        -DCMAKE_BUILD_TYPE=Release \
        -DRUNTIME_VERSION="$(cat "$ROOT_DIR/VERSION")" \
        -DOPENVINO_DIR="$OPENVINO_DIR" \
        -DOPENVINO_LINK_DIR="$LINK_DIR"
    make -j"$(nproc)" yolo_test simple_test
    cd "$ROOT_DIR"
fi

pass_count=0
fail_count=0

for model_name in yolov8n yolo11n; do
    xml_path=$(find "$CONVERTED_DIR/$model_name/ir" -name "*.xml" | head -1)
    if [ -z "$xml_path" ]; then
        echo "Skipping $model_name: converted model not found"
        continue
    fi

    echo "Testing $model_name on $DEVICE..."
    if (cd "$BUILD_DIR" && LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
            ./yolo_test "$xml_path" "$DEVICE"); then
        pass "$model_name on $DEVICE"
        pass_count=$((pass_count + 1))
    else
        echo -e "${RED}✗${NC} $model_name FAILED"
        fail_count=$((fail_count + 1))
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────

header "Results"
echo "  C++ tests passed : $pass_count"
echo "  C++ tests failed : $fail_count"

[ "$fail_count" -gt 0 ] && fail "$fail_count C++ test(s) failed"

pass "All integration tests passed"
