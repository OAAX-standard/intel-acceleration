#!/usr/bin/env python3
"""Batch-size throughput sweep for YOLOX models on OpenVINO.

Builds the official YOLOX ONNX with explicit batch sizes (1/2/4/8), converts to
OpenVINO IR, then benchmarks with both detector_test and benchmark_app.

Usage:
    python tests/benchmark_batch_sweep.py [--model yolox_nano] [--device CPU]
                                           [--batches 1,2,4,8]
                                           [--runs 100] [--duration 15]
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import openvino as ov

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))  # so `tests.models` imports when run as `python tests/benchmark_batch_sweep.py`

from tests.models import YOLOX_BASE, adapt_yolox, download_yolox_base  # noqa: E402

BUILD_DIR = ROOT / "tests" / "runtime" / "build"
IS_WINDOWS = platform.system() == "Windows"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="yolox_nano", choices=sorted(YOLOX_BASE), help="YOLOX model")
    p.add_argument("--device", default="CPU", help="OpenVINO device (CPU, GPU.0, GPU, …)")
    p.add_argument("--batches", default="1,2,4,8", help="Comma-separated batch sizes")
    p.add_argument("--runs", type=int, default=100, help="detector_test inference runs per config")
    p.add_argument("--duration", type=int, default=15, help="benchmark_app duration in seconds")
    return p.parse_args()


def find_benchmark_app() -> Path | None:
    name = "benchmark_app.exe" if IS_WINDOWS else "benchmark_app"
    found = shutil.which(name)
    if found:
        return Path(found)
    return None


def detector_test_path() -> Path:
    if IS_WINDOWS:
        return BUILD_DIR / "Release" / "detector_test.exe"
    return BUILD_DIR / "detector_test"


def export_ir(model: str, batch: int, out_dir: Path) -> Path:
    """Build the YOLOX ONNX at `batch`, convert to OpenVINO IR, and zip it for the OAAX runtime."""
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / f"{model}_b{batch}.xml"
    zip_path = xml_path.with_suffix(".zip")

    if zip_path.exists():
        return zip_path

    print(f"  Building {model} batch={batch}...", flush=True)
    onnx_path = out_dir / f"{model}_b{batch}.onnx"
    adapt_yolox(str(download_yolox_base(model, out_dir)), str(onnx_path), batch, YOLOX_BASE[model][1])

    ov.save_model(ov.convert_model(str(onnx_path)), str(xml_path))
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(xml_path, arcname=xml_path.name)
        zf.write(xml_path.with_suffix(".bin"), arcname=xml_path.with_suffix(".bin").name)

    print(f"    → {zip_path.name}", flush=True)
    return zip_path


def run_process(cmd: list, env=None, timeout: int = 300) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
        return r.stdout + r.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [error] {e}")
        return None


def parse_field(text: str, pattern: str) -> str:
    m = re.search(pattern, text)
    return m.group(1) if m else ""


def run_detector_test(model_zip: Path, device: str, batch: int, runs: int, imgsz: int) -> tuple | None:
    binary = detector_test_path()
    if not binary.exists():
        return None
    env = os.environ.copy()
    if not IS_WINDOWS:
        env["LD_LIBRARY_PATH"] = f"{BUILD_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
    text = run_process(
        [
            str(binary),
            str(model_zip),
            device,
            "--batch",
            str(batch),
            "--imgsz",
            str(imgsz),
            "--runs",
            str(runs),
            "--warmup",
            "10",
            "--perf-hint",
            "throughput",
        ],
        env=env,
        timeout=runs * 5,
    )
    if not text or "=== Results ===" not in text:
        return None
    return (
        parse_field(text, r"Min latency:\s+([\d.]+)"),
        parse_field(text, r"Throughput\s*:\s*([\d.]+)"),
    )


def run_benchmark_app(xml: Path, device: str, batch: int, duration: int) -> tuple | None:
    bench = find_benchmark_app()
    if bench is None:
        return None
    text = run_process(
        [str(bench), "-m", str(xml), "-d", device, "-hint", "throughput", "-t", str(duration)],
        timeout=duration * 4,
    )
    if not text:
        return None
    return (
        parse_field(text, r"Min:\s+([\d.]+)"),
        parse_field(text, r"Throughput:\s+([\d.]+)"),
    )


def main() -> None:
    args = parse_args()
    batches = [int(b) for b in args.batches.split(",")]
    imgsz = YOLOX_BASE[args.model][1]

    if not detector_test_path().exists():
        print(f"ERROR: detector_test binary not found at {detector_test_path()} (run tests/runtime/build-tests.sh)")
        sys.exit(1)

    cache_dir = ROOT / "tests" / "compiled_models" / "_batch_sweep"

    print("\n=== Batch-size throughput sweep ===")
    print(f"Model   : {args.model}")
    print(f"Device  : {args.device}")
    print(f"Batches : {batches}")
    print()

    # Export all IR models first
    print("[1] Exporting OpenVINO IR for each batch size...")
    ir_paths: dict[int, Path] = {}
    for b in batches:
        ir_paths[b] = export_ir(args.model, b, cache_dir)

    # Benchmark
    print()
    print("[2] Benchmarking...")
    has_bench = find_benchmark_app() is not None

    hdr = f"  {'Batch':>5}  {'detector_test min_ms':>20}  {'detector_test FPS':>17}  {'bench_app FPS':>13}"
    sep = "  " + "-" * 5 + "  " + "-" * 20 + "  " + "-" * 17 + "  " + "-" * 13
    print(hdr)
    print(sep)

    for b in batches:
        zip_path = ir_paths[b]
        yt = run_detector_test(zip_path, args.device, b, args.runs, imgsz)
        ba = run_benchmark_app(zip_path.with_suffix(".xml"), args.device, b, args.duration) if has_bench else None

        yt_min = f"{yt[0]} ms" if yt and yt[0] else "FAIL"
        yt_fps = yt[1] if yt and yt[1] else "FAIL"
        ba_fps = ba[1] if ba and ba[1] else ("n/a" if not has_bench else "FAIL")

        print(f"  {b:>5}  {yt_min:>20}  {yt_fps:>17}  {ba_fps:>13}")

    print()


if __name__ == "__main__":
    main()
