#!/usr/bin/env python3
"""Batch-size throughput sweep for YOLO models on OpenVINO.

Exports YOLO .pt → ONNX with explicit batch sizes (1/2/4/8), converts to
OpenVINO IR, then benchmarks with both yolo_test and benchmark_app.

Usage:
    python tests/benchmark_batch_sweep.py [--model yolov8n] [--device CPU]
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
import tempfile
from pathlib import Path

import openvino as ov
from ultralytics import YOLO

ROOT = Path(__file__).parent.parent
BUILD_DIR = ROOT / "runtime-library" / "build"
IS_WINDOWS = platform.system() == "Windows"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="yolov8n", help="Model name (stem of .pt file in repo root)")
    p.add_argument("--device", default="CPU", help="OpenVINO device (CPU, GPU.0, GPU, …)")
    p.add_argument("--batches", default="1,2,4,8", help="Comma-separated batch sizes")
    p.add_argument("--runs", type=int, default=100, help="yolo_test inference runs per config")
    p.add_argument("--duration", type=int, default=15, help="benchmark_app duration in seconds")
    return p.parse_args()


def find_benchmark_app() -> Path | None:
    name = "benchmark_app.exe" if IS_WINDOWS else "benchmark_app"
    found = shutil.which(name)
    if found:
        return Path(found)
    return None


def yolo_test_path() -> Path:
    if IS_WINDOWS:
        return BUILD_DIR / "Release" / "yolo_test.exe"
    return BUILD_DIR / "yolo_test"


def export_ir(pt_path: Path, batch: int, out_dir: Path) -> Path:
    """Export .pt → ONNX with explicit batch, then convert to OpenVINO IR."""
    out_dir.mkdir(parents=True, exist_ok=True)
    xml_path = out_dir / f"model_b{batch}.xml"

    if xml_path.exists():
        return xml_path

    print(f"  Exporting batch={batch} from {pt_path.name}...", flush=True)

    # Export to ONNX with explicit batch size via ultralytics
    with tempfile.TemporaryDirectory():
        model = YOLO(str(pt_path))
        onnx_path = model.export(format="onnx", batch=batch, imgsz=640, simplify=True)
        onnx_path = Path(onnx_path)

        # Convert ONNX → OpenVINO IR
        ov_model = ov.convert_model(str(onnx_path))
        ov.save_model(ov_model, str(xml_path))

    print(f"    → {xml_path.name}", flush=True)
    return xml_path


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


def run_yolo_test(xml: Path, device: str, batch: int, runs: int) -> tuple | None:
    binary = yolo_test_path()
    if not binary.exists():
        return None
    env = os.environ.copy()
    if not IS_WINDOWS:
        env["LD_LIBRARY_PATH"] = f"{BUILD_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
    text = run_process(
        [
            str(binary),
            str(xml),
            device,
            "--batch",
            str(batch),
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
    pt_path = ROOT / f"{args.model}.pt"

    if not pt_path.exists():
        print(f"ERROR: {pt_path} not found")
        sys.exit(1)

    if not yolo_test_path().exists():
        print(f"ERROR: yolo_test binary not found at {yolo_test_path()}")
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
        ir_paths[b] = export_ir(pt_path, b, cache_dir)

    # Benchmark
    print()
    print("[2] Benchmarking...")
    has_bench = find_benchmark_app() is not None

    hdr = f"  {'Batch':>5}  {'yolo_test min_ms':>16}  {'yolo_test FPS':>13}  {'bench_app FPS':>13}"
    sep = "  " + "-" * 5 + "  " + "-" * 16 + "  " + "-" * 13 + "  " + "-" * 13
    print(hdr)
    print(sep)

    for b in batches:
        xml = ir_paths[b]
        yt = run_yolo_test(xml, args.device, b, args.runs)
        ba = run_benchmark_app(xml, args.device, b, args.duration) if has_bench else None

        yt_min = f"{yt[0]} ms" if yt and yt[0] else "FAIL"
        yt_fps = yt[1] if yt and yt[1] else "FAIL"
        ba_fps = ba[1] if ba and ba[1] else ("n/a" if not has_bench else "FAIL")

        print(f"  {b:>5}  {yt_min:>16}  {yt_fps:>13}  {ba_fps:>13}")

    print()


if __name__ == "__main__":
    main()
