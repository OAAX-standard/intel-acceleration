#!/usr/bin/env bash
# Stage 2: Benchmark and runtime validation using compiled models from Stage 1.
#
# Requires tests/compiled_models/ to be populated by stage1_compile.sh first.
# Runs:
#   - OpenVINO benchmark_app (max throughput) on all compiled models
#   - C++ yolo_test (runtime library validation + latency benchmark)
#
# Usage:
#   bash scripts/stage2_run.sh [--devices CPU,GPU.0] [--duration 10] [--skip-runtime]

set -e
cd "$(dirname "$0")/.."

ROOT_DIR="$(pwd)"
RUNTIME_DIR="$ROOT_DIR/runtime-library"
COMPILED_DIR="tests/compiled_models"
DEVICES=(CPU)
DURATION=10
SKIP_RUNTIME=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices)      IFS=',' read -ra DEVICES <<< "$2"; shift 2 ;;
        --duration)     DURATION="$2";                      shift 2 ;;
        --skip-runtime) SKIP_RUNTIME=1;                     shift   ;;
        *) echo "Usage: $0 [--devices CPU,GPU.0] [--duration 10] [--skip-runtime]"; exit 1 ;;
    esac
done

GREEN='\033[0;32m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'
pass()   { echo -e "${GREEN}✓${NC} $1"; }
fail()   { echo -e "${RED}✗${NC} $1"; exit 1; }
header() { echo -e "\n${BLUE}=== $1 ===${NC}"; }

# ── Guard: compiled models must exist ─────────────────────────────────────────

if [[ ! -d "$COMPILED_DIR" ]] || ! ls "$COMPILED_DIR"/*/FP32/*.xml &>/dev/null; then
    fail "tests/compiled_models/ not found or empty — run stage1_compile.sh first"
fi

# ── Activate venv ─────────────────────────────────────────────────────────────

[[ ! -f .venv/bin/activate ]] && fail ".venv not found — run stage1_compile.sh first"
source .venv/bin/activate

# ── 1. benchmark_app: throughput across all models × variants × devices ───────

header "Step 1: Throughput benchmark (benchmark_app, hint=throughput, ${DURATION}s)"

printf "  %-10s  %-6s  %-8s  %12s\n" "Model" "Prec" "Device" "Throughput"
printf "  %-10s  %-6s  %-8s  %12s\n" "----------" "------" "--------" "------------"

for xml in "$COMPILED_DIR"/*/FP32/*.xml \
           "$COMPILED_DIR"/*/FP16/*.xml \
           "$COMPILED_DIR"/*/INT8/*.xml; do
    [[ -f "$xml" ]] || continue
    variant=$(basename "$(dirname "$xml")")
    model=$(basename "$xml" .xml)
    for device in "${DEVICES[@]}"; do
        output=$(benchmark_app -m "$xml" -d "$device" -hint throughput -t "$DURATION" 2>&1)
        fps=$(echo "$output" | grep "Throughput:" | awk '{print $(NF-1) " FPS"}')
        printf "  %-10s  %-6s  %-8s  %12s\n" "$model" "$variant" "$device" "${fps:-FAILED}"
    done
done
echo ""
pass "Throughput benchmark done"

# ── 2. C++ runtime: build if needed + run yolo_test ──────────────────────────

[[ "$SKIP_RUNTIME" -eq 1 ]] && { echo "Skipping C++ runtime tests."; exit 0; }

header "Step 2: C++ runtime tests (yolo_test)"

BUILD_DIR="$RUNTIME_DIR/build"

if [[ ! -f "$BUILD_DIR/yolo_test" ]]; then
    echo "  Building C++ runtime..."
    OPENVINO_DIR="${OPENVINO_DIR:-/usr/local/lib/python3.10/dist-packages/openvino}"
    CMAKE_BIN="${CMAKE_BIN:-/usr/bin/cmake}"

    LINK_DIR="$BUILD_DIR/openvino_links"
    mkdir -p "$LINK_DIR"
    for lib in "$OPENVINO_DIR/libs/"*.so.*; do
        base=$(basename "$lib")
        ln -sf "$lib" "$LINK_DIR/${base%%\.*}.so" 2>/dev/null || true
    done

    mkdir -p "$BUILD_DIR" && cd "$BUILD_DIR"
    "$CMAKE_BIN" .. \
        -DPLATFORM=X86_64 \
        -DCMAKE_BUILD_TYPE=Release \
        -DRUNTIME_VERSION="$(cat "$ROOT_DIR/VERSION")" \
        -DOPENVINO_DIR="$OPENVINO_DIR" \
        -DOPENVINO_LINK_DIR="$(realpath "$LINK_DIR")"
    make -j"$(nproc)" yolo_test simple_test
    cd "$ROOT_DIR"
fi

pass_count=0; fail_count=0

for xml in "$COMPILED_DIR"/*/FP32/*.xml; do
    [[ -f "$xml" ]] || continue
    model=$(basename "$xml" .xml)
    xml_abs="$(realpath "$xml")"
    for device in "${DEVICES[@]}"; do
        echo "  $model  FP32  $device"
        if (cd "$BUILD_DIR" && LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
                ./yolo_test "$xml_abs" "$device" --warmup 3 --runs 10); then
            pass "$model FP32 on $device"
            pass_count=$((pass_count + 1))
        else
            echo -e "${RED}✗${NC} $model FP32 on $device FAILED"
            fail_count=$((fail_count + 1))
        fi
    done
done

header "Stage 2 results"
echo "  Runtime tests passed : $pass_count"
echo "  Runtime tests failed : $fail_count"

[[ "$fail_count" -gt 0 ]] && fail "$fail_count runtime test(s) failed"
pass "Stage 2 complete"
