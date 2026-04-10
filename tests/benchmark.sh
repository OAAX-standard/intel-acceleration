#!/usr/bin/env bash
# Benchmark YOLO models across precisions and devices using OpenVINO benchmark_app.
# Converts models to FP32/FP16/INT8 IR if not already cached.
#
# Usage:
#   bash tests/benchmark.sh [--models yolov8n,yolo11n] [--devices CPU,GPU.0] [--duration 10]
#
# Defaults: models=yolov8n,yolo11n  devices=CPU  duration=10 (seconds)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS=(yolov8n yolo11n)
DEVICES=(CPU)
DURATION=10
VARIANTS=(FP32 FP16 INT8)
CACHE_DIR="tests/benchmark_models"
CALIB_DIR="tests/benchmark_calibration"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)   IFS=',' read -ra MODELS   <<< "$2"; shift 2 ;;
        --devices)  IFS=',' read -ra DEVICES  <<< "$2"; shift 2 ;;
        --duration) DURATION="$2";                       shift 2 ;;
        *) echo "Usage: $0 [--models m1,m2] [--devices d1,d2] [--duration N]"; exit 1 ;;
    esac
done

cd "$REPO_ROOT"

if [[ ! -f .venv/bin/activate ]]; then
    echo "Error: .venv not found. Run: uv venv && uv sync --extra integration --extra quantization"
    exit 1
fi
source .venv/bin/activate

# ── Convert models (cached in $CACHE_DIR) ─────────────────────────────────────
export BENCH_MODELS="${MODELS[*]}"
export BENCH_CACHE_DIR="$CACHE_DIR"
export BENCH_CALIB_DIR="$CALIB_DIR"

python - <<"PYEOF"
import os, sys, zipfile
from pathlib import Path

sys.path.insert(0, ".")

from conversion_toolchain.utils import convert_to_ir
from conversion_toolchain.config import OptimizationConfig
from conversion_toolchain.logger import Logs
from conversion_toolchain.quantization import is_nncf_available
from tests.models import download_model, download_calibration_images

MODELS    = os.environ["BENCH_MODELS"].split()
CACHE_DIR = Path(os.environ["BENCH_CACHE_DIR"])
CALIB_DIR = Path(os.environ["BENCH_CALIB_DIR"])

VARIANTS = {
    "FP32": OptimizationConfig({"optimization": {"fp16_compression": False}}),
    "FP16": OptimizationConfig({"optimization": {"fp16_compression": True}}),
    "INT8": OptimizationConfig({
        "optimization": {
            "fp16_compression": False,
            "quantization": {"enabled": True, "preset": "mixed", "subset_size": 128},
        }
    }),
}

calib_dir = None
if is_nncf_available():
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    calib_dir = download_calibration_images(str(CALIB_DIR))
else:
    print("NNCF not available — skipping INT8")
    del VARIANTS["INT8"]

for model_name in MODELS:
    onnx = download_model(model_name, str(CACHE_DIR / "onnx"))
    for variant, config in VARIANTS.items():
        out_dir = CACHE_DIR / model_name / variant
        xml = out_dir / f"{model_name}.xml"
        if xml.exists():
            print(f"  ✓ {model_name}/{variant} already cached")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Converting {model_name} → {variant}...")
        zip_path = convert_to_ir(onnx, str(out_dir / "zip"), Logs(), config,
                                 calib_dir if variant == "INT8" else None)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(out_dir)
        print(f"  ✓ {model_name}/{variant} done")
PYEOF

# ── Benchmark ─────────────────────────────────────────────────────────────────
echo ""
printf "  %-10s  %-6s  %-8s  %12s\n" "Model" "Prec" "Device" "Throughput"
printf "  %-10s  %-6s  %-8s  %12s\n" "----------" "------" "--------" "------------"

for model in "${MODELS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
        xml="$CACHE_DIR/$model/$variant/$model.xml"
        if [[ ! -f "$xml" ]]; then
            printf "  %-10s  %-6s  %-8s  %12s\n" "$model" "$variant" "—" "(skipped)"
            continue
        fi
        for device in "${DEVICES[@]}"; do
            output=$(benchmark_app -m "$xml" -d "$device" -hint throughput -t "$DURATION" 2>&1)
            fps=$(echo "$output" | grep "Throughput:" | awk '{print $(NF-1) " FPS"}')
            printf "  %-10s  %-6s  %-8s  %12s\n" "$model" "$variant" "$device" "${fps:-FAILED}"
        done
    done
    echo ""
done
