#!/usr/bin/env bash
# run_stage2.sh — wrapper around tests/stage2.py with convenient defaults
#
# All variables can be set from the environment before running, e.g.:
#   DEVICES="CPU GPU.0" MODELS="yolo11n yolo11s" PRECISIONS="FP32" bash tests/run_stage2.sh
#
# Available variables:
#   DEVICES       Space or comma-separated device list      (default: CPU)
#   PERF_HINTS    Space or comma-separated hint list        (default: throughput)
#   RUNS          Inference runs for yolo_test              (default: 100)
#   WARMUP        Warmup runs for yolo_test                 (default: 5)
#   DURATION      benchmark_app duration in seconds         (default: 10)
#   MODELS        Space-separated model name filter         (default: all)
#                 e.g. "yolo11n yolo11n_320"
#   PRECISIONS    Space-separated precision filter          (default: all)
#                 e.g. "FP32 FP16"
#   CSV           Path for CSV output                       (default: none)
#   IN_FLIGHT     Max parallel in-flight requests            (default: 5)
#   LOG_LEVEL     Runtime log level 0=trace..4=err           (default: 2/info)
#   SKIP_BENCH    Set to 1 to skip benchmark_app section    (default: 0)
#   SKIP_RUNTIME  Set to 1 to skip yolo_test section        (default: 0)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_BUILD_DIR="${ROOT}/tests/runtime/build"
VENV="${ROOT}/.venv"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEVICES="${DEVICES:-CPU}"
PERF_HINTS="${PERF_HINTS:-throughput}"
RUNS="${RUNS:-2000}"
WARMUP="${WARMUP:-100}"
DURATION="${DURATION:-5}"
MODELS="${MODELS:-yolo11n_320}"
PRECISIONS="${PRECISIONS:-INT8}"
CSV="${CSV:-}"
IN_FLIGHT="${IN_FLIGHT:-100}"
LOG_LEVEL="${LOG_LEVEL:-2}"
SKIP_BENCH="${SKIP_BENCH:-0}"
SKIP_RUNTIME="${SKIP_RUNTIME:-0}"

# ── Print config ───────────────────────────────────────────────────────────────
echo "=== run_stage2 ==="
echo "  DEVICES     : ${DEVICES}"
echo "  PERF_HINTS  : ${PERF_HINTS}"
echo "  RUNS        : ${RUNS}"
echo "  WARMUP      : ${WARMUP}"
echo "  DURATION    : ${DURATION}s"
echo "  MODELS      : ${MODELS:-<all>}"
echo "  PRECISIONS  : ${PRECISIONS:-<all>}"
echo "  CSV         : ${CSV:-<none>}"
echo "  IN_FLIGHT   : ${IN_FLIGHT}"
echo "  LOG_LEVEL   : ${LOG_LEVEL}"
echo "  SKIP_BENCH  : ${SKIP_BENCH}"
echo "  SKIP_RUNTIME: ${SKIP_RUNTIME}"
echo ""

# ── Activate venv if present ──────────────────────────────────────────────────
if [[ -f "${VENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV}/bin/activate"
fi

# ── Build yolo_test if needed ─────────────────────────────────────────────────
if [[ "${SKIP_RUNTIME}" != "1" && ! -f "${TEST_BUILD_DIR}/yolo_test" ]]; then
    echo "[run_stage2] Building yolo_test..."
    bash "${ROOT}/tests/runtime/build-tests.sh"
fi

# ── Convert space-separated values to comma-separated for stage2.py ───────────
to_csv() { echo "$1" | tr ' ' ','; }

# ── Build argument list ───────────────────────────────────────────────────────
ARGS=(
    "--devices"    "$(to_csv "${DEVICES}")"
    "--perf-hints" "$(to_csv "${PERF_HINTS}")"
    "--runs"       "${RUNS}"
    "--warmup"     "${WARMUP}"
    "--duration"   "${DURATION}"
)

[[ -n "${MODELS}"     ]] && ARGS+=("--models"     "$(to_csv "${MODELS}")")
[[ -n "${PRECISIONS}" ]] && ARGS+=("--precisions" "$(to_csv "${PRECISIONS}")")
[[ -n "${CSV}"        ]] && ARGS+=("--csv"        "${CSV}")
ARGS+=("--in-flight" "${IN_FLIGHT}")
ARGS+=("--log-level" "${LOG_LEVEL}")
[[ "${SKIP_RUNTIME}" == "1" ]] && ARGS+=("--skip-runtime")
[[ "${SKIP_BENCH}"   == "1" ]] && ARGS+=("--skip-bench")

cd "${ROOT}"
python3 tests/stage2.py "${ARGS[@]}"
