"""
Inference benchmark: FP32 / FP16 / INT8 × CPU / GPU

Converts YOLO models to all three precision variants (if not already cached),
then measures latency on each available OpenVINO device.

Usage:
    uv run pytest tests/benchmark.py -v -s
  or standalone:
    uv run python tests/benchmark.py [--models yolov8n yolo11n] [--devices CPU GPU.0]
                                     [--runs 30] [--warmup 5]
"""
import argparse
import time
import zipfile
from pathlib import Path

import numpy as np
import openvino as ov

from conversion_toolchain.utils import convert_to_ir
from conversion_toolchain.config import OptimizationConfig
from conversion_toolchain.logger import Logs
from conversion_toolchain.quantization import is_nncf_available
from tests.models import download_model, download_calibration_images

CACHE_DIR  = Path("tests/benchmark_models")
CALIB_DIR  = Path("tests/benchmark_calibration")

YOLO_MODELS = ["yolov8n", "yolo11n"]
INPUT_SHAPE = (1, 3, 640, 640)

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


def get_or_convert(model_name: str, variant: str, config: OptimizationConfig,
                   calib_dir: str = None) -> Path:
    """Return path to the .xml IR file, converting if not cached."""
    out_dir = CACHE_DIR / model_name / variant
    xml = out_dir / f"{model_name}.xml"
    if xml.exists():
        return xml

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = download_model(model_name, str(CACHE_DIR / "onnx"))
    zip_path = convert_to_ir(onnx_path, str(out_dir / "zip"), Logs(), config, calib_dir)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    return next(out_dir.glob("*.xml"))


def benchmark(xml: Path, device: str, warmup: int, runs: int) -> dict:
    """Compile model on device and return latency stats (ms)."""
    core = ov.Core()
    model = core.compile_model(core.read_model(str(xml)), device)
    req = model.create_infer_request()
    inp = np.random.rand(*INPUT_SHAPE).astype(np.float32)

    for _ in range(warmup):
        req.set_input_tensor(ov.Tensor(inp))
        req.infer()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        req.set_input_tensor(ov.Tensor(inp))
        req.infer()
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "avg": float(np.mean(times)),
        "min": float(np.min(times)),
        "p95": float(np.percentile(times, 95)),
    }


def run(model_names=None, devices=None, warmup=5, runs=30):
    model_names = model_names or YOLO_MODELS
    core = ov.Core()
    available = core.available_devices
    devices = [d for d in (devices or available) if d in available]

    if not devices:
        print("No matching devices available.")
        return

    # Download calibration images once if INT8 is needed and NNCF is available
    calib_dir = None
    if is_nncf_available():
        CALIB_DIR.mkdir(parents=True, exist_ok=True)
        calib_dir = download_calibration_images(str(CALIB_DIR))
    else:
        print("NNCF not available — skipping INT8 variants")

    variants_to_run = {k: v for k, v in VARIANTS.items()
                       if k != "INT8" or calib_dir is not None}

    # Print header
    col = 8
    header = f"  {'Model':<10} {'Variant':<6} {'Device':<{col}}  {'avg':>7}  {'min':>7}  {'p95':>7}"
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))

    for model_name in model_names:
        for variant, config in variants_to_run.items():
            xml = get_or_convert(model_name, variant, config,
                                 calib_dir if variant == "INT8" else None)
            for device in devices:
                stats = benchmark(xml, device, warmup, runs)
                print(f"  {model_name:<10} {variant:<6} {device:<{col}}"
                      f"  {stats['avg']:>6.1f}ms"
                      f"  {stats['min']:>6.1f}ms"
                      f"  {stats['p95']:>6.1f}ms")
        print()


# ── pytest entry point ────────────────────────────────────────────────────────

def test_benchmark(request):
    """Run benchmark as a pytest test (use -s to see output)."""
    devices = request.config.getoption("--devices", default=None)
    runs    = int(request.config.getoption("--runs", default=30))
    warmup  = int(request.config.getoption("--warmup", default=5))
    run(devices=devices.split(",") if devices else None, warmup=warmup, runs=runs)


def pytest_addoption(parser):
    parser.addoption("--devices", help="Comma-separated devices, e.g. CPU,GPU.0")
    parser.addoption("--runs",    default="30", help="Number of inference runs")
    parser.addoption("--warmup",  default="5",  help="Number of warmup runs")


# ── standalone entry point ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenVINO inference benchmark")
    parser.add_argument("--models",  nargs="+", default=YOLO_MODELS,
                        help="Models to benchmark")
    parser.add_argument("--devices", nargs="+", default=None,
                        help="Devices to test (default: all available)")
    parser.add_argument("--runs",    type=int, default=30,
                        help="Number of inference runs (default: 30)")
    parser.add_argument("--warmup",  type=int, default=5,
                        help="Number of warmup runs (default: 5)")
    args = parser.parse_args()
    run(model_names=args.models, devices=args.devices,
        warmup=args.warmup, runs=args.runs)
