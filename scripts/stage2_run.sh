#!/usr/bin/env bash
# Stage 2: Benchmark and runtime validation using compiled models from Stage 1.
#
# Requires tests/compiled_models/ to be populated by stage1_compile.sh first.
# Runs:
#   - OpenVINO benchmark_app (latency hint, p95) on all compiled models
#   - C++ yolo_test (runtime library validation + latency benchmark)
#
# Usage:
#   bash scripts/stage2_run.sh [--devices CPU,GPU.0] [--duration 10]
#                              [--csv results.csv] [--skip-runtime]
#
# CSV columns: timestamp,tool,model,variant,device,avg_ms,min_ms,p95_ms,throughput_fps

set -e
cd "$(dirname "$0")/.."

ROOT_DIR="$(pwd)"
RUNTIME_DIR="$ROOT_DIR/runtime-library"
COMPILED_DIR="tests/compiled_models"
DEVICES=(CPU)
DURATION=10
SKIP_RUNTIME=0
CSV_FILE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --devices)      IFS=',' read -ra DEVICES <<< "$2"; shift 2 ;;
        --duration)     DURATION="$2";                      shift 2 ;;
        --csv)          CSV_FILE="$2";                       shift 2 ;;
        --skip-runtime) SKIP_RUNTIME=1;                      shift   ;;
        *) echo "Usage: $0 [--devices CPU,GPU.0] [--duration 10] [--csv file.csv] [--skip-runtime]"
           exit 1 ;;
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

# ── CSV helpers ───────────────────────────────────────────────────────────────

csv_init() {
    [[ -z "$CSV_FILE" ]] && return
    if [[ ! -f "$CSV_FILE" ]]; then
        echo "timestamp,tool,model,variant,device,avg_ms,min_ms,p95_ms,throughput_fps" > "$CSV_FILE"
        echo "  CSV: $CSV_FILE"
    fi
}

csv_row() {
    # csv_row <tool> <model> <variant> <device> <avg_ms> <min_ms> <p95_ms> <fps>
    [[ -z "$CSV_FILE" ]] && return
    local ts; ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$ts" "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" >> "$CSV_FILE"
}

csv_init

# ── 1. benchmark_app: latency + throughput across all models × variants × devices

header "Step 1: benchmark_app  (hint=latency, p95, ${DURATION}s per run)"

printf "  %-10s  %-6s  %-8s  %8s  %8s  %8s  %12s\n" \
    "Model" "Prec" "Device" "avg_ms" "min_ms" "p95_ms" "Throughput"
printf "  %-10s  %-6s  %-8s  %8s  %8s  %8s  %12s\n" \
    "----------" "------" "--------" "--------" "--------" "--------" "------------"

for xml in "$COMPILED_DIR"/*/FP32/*.xml \
           "$COMPILED_DIR"/*/FP16/*.xml \
           "$COMPILED_DIR"/*/INT8/*.xml; do
    [[ -f "$xml" ]] || continue
    variant=$(basename "$(dirname "$xml")")
    model=$(basename "$xml" .xml)
    for device in "${DEVICES[@]}"; do
        out=$(benchmark_app -m "$xml" -d "$device" \
              -hint latency -latency_percentile 95 -t "$DURATION" 2>&1)

        avg=$(echo "$out" | awk '/Average:/{print $(NF-1)}')
        min=$(echo "$out" | awk '/   Min:/{print $(NF-1)}')
        p95=$(echo "$out" | awk '/percentile:/{print $(NF-1)}')
        fps=$(echo "$out" | awk '/Throughput:/{print $(NF-1)}')

        printf "  %-10s  %-6s  %-8s  %7sms  %7sms  %7sms  %10s FPS\n" \
            "$model" "$variant" "$device" \
            "${avg:--}" "${min:--}" "${p95:--}" "${fps:--}"

        csv_row "benchmark_app" "$model" "$variant" "$device" \
                "${avg:-}" "${min:-}" "${p95:-}" "${fps:-}"
    done
done
echo ""
pass "benchmark_app done"

# ── 2. C++ runtime: build if needed + run yolo_test ──────────────────────────

[[ "$SKIP_RUNTIME" -eq 1 ]] && { echo "Skipping C++ runtime tests."; exit 0; }

header "Step 2: yolo_test  (OAAX runtime library, FP32 models)"

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

printf "\n  %-10s  %-6s  %-8s  %8s  %8s  %8s  %12s\n" \
    "Model" "Prec" "Device" "avg_ms" "min_ms" "p95_ms" "Throughput"
printf "  %-10s  %-6s  %-8s  %8s  %8s  %8s  %12s\n" \
    "----------" "------" "--------" "--------" "--------" "--------" "------------"

pass_count=0; fail_count=0

for xml in "$COMPILED_DIR"/*/FP32/*.xml; do
    [[ -f "$xml" ]] || continue
    model=$(basename "$xml" .xml)
    xml_abs="$(realpath "$xml")"
    for device in "${DEVICES[@]}"; do
        out=$(cd "$BUILD_DIR" && LD_LIBRARY_PATH="$BUILD_DIR:$LD_LIBRARY_PATH" \
              ./yolo_test "$xml_abs" "$device" --warmup 5 --runs 30 2>&1)

        if echo "$out" | grep -q "=== Results ==="; then
            avg=$(echo "$out" | awk '/  Avg /{print $(NF-1)}')
            min=$(echo "$out" | awk '/  Min /{print $(NF-1)}')
            p95=$(echo "$out" | awk '/  p95 /{print $(NF-1)}')
            fps=$(echo "$avg" | awk '{printf "%.2f", 1000/$1}')

            printf "  %-10s  %-6s  %-8s  %7sms  %7sms  %7sms  %10s FPS\n" \
                "$model" "FP32" "$device" "$avg" "$min" "$p95" "$fps"

            csv_row "yolo_test" "$model" "FP32" "$device" "$avg" "$min" "$p95" "$fps"
            pass_count=$((pass_count + 1))
        else
            printf "  %-10s  %-6s  %-8s  %s\n" "$model" "FP32" "$device" "FAILED"
            fail_count=$((fail_count + 1))
        fi
    done
done
echo ""

header "Stage 2 results"
[[ -n "$CSV_FILE" ]] && echo "  CSV saved to : $CSV_FILE"
echo "  Runtime tests passed : $pass_count"
echo "  Runtime tests failed : $fail_count"

[[ "$fail_count" -gt 0 ]] && fail "$fail_count runtime test(s) failed"
pass "Stage 2 complete"
