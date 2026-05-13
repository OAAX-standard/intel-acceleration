#!/usr/bin/env python3
"""Stage 2: Benchmark and runtime validation using compiled models from Stage 1.

Requires tests/compiled_models/ to be populated by stage1 first.
Runs benchmark_app (if available) and yolo_test across all compiled models.

Usage:
    python tests/stage2.py [--devices CPU,GPU.0] [--duration 10]
                           [--perf-hints latency,throughput]
                           [--csv results.csv] [--skip-runtime]
                           [--runs 300] [--warmup 5]
"""

import argparse
import csv
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
COMPILED_DIR = ROOT / "tests" / "compiled_models"
RUNTIME_BUILD_DIR = ROOT / "runtime-library" / "build"
TEST_BUILD_DIR = ROOT / "tests" / "runtime" / "build"
IS_WINDOWS = platform.system() == "Windows"

# Match the models compiled by stage1 / test_yolo_integration.py.
YOLO_MODELS = {
    "yolov8n",
    "yolo11n",
    "yolo11s",
    "yolo11n_b4",
    "yolo11s_b4",
    "yolo11n_320",
    "yolo11s_320",
    "yolo11n_320_b4",
    "yolo11s_320_b4",
    "yolo26s",
    "yolo26s_320",
    "yolo26m",
    "yolo26m_320",
    "yolo26s_b4",
    "yolo26m_b2",
}


def header(title: str) -> None:
    print(f"\n\033[34m=== {title} ===\033[0m")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--devices", default="CPU", help="Comma-separated device list (default: CPU)")
    p.add_argument("--duration", type=int, default=10, help="benchmark_app duration in seconds")
    p.add_argument("--runs", type=int, default=300, help="yolo_test inference runs")
    p.add_argument("--warmup", type=int, default=5, help="yolo_test warmup runs")
    p.add_argument("--in-flight", type=int, default=5, help="max parallel in-flight requests (yolo_test --in-flight)")
    p.add_argument("--log-level", type=int, default=2, help="runtime log level: 0=trace 1=debug 2=info 3=warn 4=err")
    p.add_argument(
        "--perf-hints",
        default="throughput",
        help="Comma-separated perf hints to test (default: throughput)",
    )
    p.add_argument("--csv", default="", help="Path to output CSV file")
    p.add_argument("--skip-runtime", action="store_true", help="Skip yolo_test section")
    p.add_argument("--skip-bench", action="store_true", help="Skip benchmark_app section")
    p.add_argument("--models", default="", help="Comma-separated model names to run (default: all)")
    p.add_argument(
        "--precisions",
        default="",
        help="Comma-separated precisions to run: FP32,FP16,INT8,INT8_NPU,FP32_U8 (default: all)",
    )
    return p.parse_args()


# ── Binary discovery ───────────────────────────────────────────────────────────


def find_benchmark_app() -> Path | None:
    name = "benchmark_app.exe" if IS_WINDOWS else "benchmark_app"
    found = shutil.which(name)
    if found:
        return Path(found)
    openvino_dir = os.environ.get("OPENVINO_DIR", "")
    if openvino_dir:
        for candidate in [
            Path(openvino_dir) / ".." / "bin" / "intel64" / name,
            Path(openvino_dir) / "bin" / "intel64" / name,
        ]:
            if candidate.exists():
                return candidate.resolve()
    return None


def yolo_test_path() -> Path:
    if IS_WINDOWS:
        return TEST_BUILD_DIR / "Release" / "yolo_test.exe"
    return TEST_BUILD_DIR / "yolo_test"


# ── Build yolo_test ────────────────────────────────────────────────────────────


def build_yolo_test() -> bool:
    """Build yolo_test from tests/runtime/CMakeLists.txt. Returns True on success."""
    cmake = os.environ.get("CMAKE_BIN", shutil.which("cmake") or "cmake")
    runtime_lib_dir = str(RUNTIME_BUILD_DIR / "Release") if IS_WINDOWS else str(RUNTIME_BUILD_DIR)

    TEST_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [cmake, "..", f"-DRUNTIME_LIB_DIR={runtime_lib_dir}", "-DCMAKE_BUILD_TYPE=Release"],
            cwd=TEST_BUILD_DIR,
            check=True,
        )
        subprocess.run(
            [cmake, "--build", ".", "--config", "Release", "--target", "yolo_test", "-j", str(os.cpu_count() or 4)],
            cwd=TEST_BUILD_DIR,
            check=True,
        )
        # Make shared libs available alongside the test binary ($ORIGIN RPATH / Windows DLL search)
        if IS_WINDOWS:
            dst_dir = TEST_BUILD_DIR / "Release"
            dst_dir.mkdir(parents=True, exist_ok=True)
            for dll in (RUNTIME_BUILD_DIR / "Release").glob("*.dll"):
                dst = dst_dir / dll.name
                if not dst.exists():
                    shutil.copy2(dll, dst)
        else:
            for so in RUNTIME_BUILD_DIR.glob("*.so*"):
                dst = TEST_BUILD_DIR / so.name
                if not dst.exists():
                    dst.symlink_to(so)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Build failed: {e}")
        return False


# ── Model discovery ────────────────────────────────────────────────────────────


def get_compiled_models(
    model_filter: set[str] | None = None, precision_filter: set[str] | None = None
) -> list[tuple[Path, str, str]]:
    allowed_models = model_filter or YOLO_MODELS
    allowed_precisions = precision_filter or {"FP32", "FP16", "INT8", "INT8_NPU", "FP32_U8"}
    return [
        (zip_path, zip_path.stem, zip_path.parent.name)
        for variant in ("FP32", "FP16", "INT8", "INT8_NPU", "FP32_U8")
        if variant in allowed_precisions
        for zip_path in sorted(COMPILED_DIR.glob(f"*/{variant}/*.zip"))
        if zip_path.stem in allowed_models
    ]


# ── Output parsing ─────────────────────────────────────────────────────────────


def parse_field(text: str, pattern: str) -> str:
    m = re.search(pattern, text)
    return m.group(1) if m else ""


def run_process(cmd: list, cwd=None, env=None, timeout: int = 120) -> str | None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            env=env,
            timeout=timeout,
        )
        if result.returncode != 0:
            print(f"  [exit {result.returncode}] {Path(cmd[0]).name}")
        return result.stdout + result.stderr
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [error] {e}")
        return None


def run_benchmark_app(xml: Path, device: str, duration: int, perf_hint: str = "throughput") -> tuple | None:
    bench = find_benchmark_app()
    if bench is None:
        return None
    text = run_process(
        [
            str(bench),
            "-m",
            str(xml),
            "-d",
            device,
            "-hint",
            perf_hint,
            "-latency_percentile",
            "95",
            "-t",
            str(duration),
        ],
        # 120s base covers GPU JIT compilation before measurement starts.
        timeout=120 + duration * 4,
    )
    if not text:
        return None
    return (
        parse_field(text, r"Average:\s+([\d.]+)"),
        parse_field(text, r"Min:\s+([\d.]+)"),
        parse_field(text, r"percentile:\s+([\d.]+)"),
        parse_field(text, r"Throughput:\s+([\d.]+)"),
    )


def run_yolo_test(
    zip_path: Path,
    device: str,
    warmup: int,
    runs: int,
    batch: int = 1,
    perf_hint: str = "throughput",
    imgsz: int = 640,
    in_flight: int = 5,
    log_level: int = 2,
    no_validate: bool = False,
    input_dtype: str = "f32",
) -> tuple | None:
    binary = yolo_test_path()
    if not binary.exists():
        print(f"  [error] yolo_test binary not found: {binary}")
        return None
    env = os.environ.copy()
    if not IS_WINDOWS:
        env["LD_LIBRARY_PATH"] = f"{TEST_BUILD_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
    cmd = [
        str(binary),
        str(zip_path),
        device,
        "--warmup",
        str(warmup),
        "--runs",
        str(runs),
        "--perf-hint",
        perf_hint,
    ]
    if batch > 1:
        cmd += ["--batch", str(batch)]
    if imgsz != 640:
        cmd += ["--imgsz", str(imgsz)]
    if in_flight != 5:
        cmd += ["--in-flight", str(in_flight)]
    if log_level != 2:
        cmd += ["--log-level", str(log_level)]
    if no_validate:
        cmd += ["--no-validate"]
    if input_dtype != "f32":
        cmd += ["--input-dtype", input_dtype]
    text = run_process(
        cmd,
        cwd=binary.parent,  # run from binary dir so DLLs are found on Windows
        env=env,
        # 600s overhead covers cold model compilation (e.g. yolo11s ~3 min uncached).
        timeout=600 + runs * 2,
    )
    if not text or "=== Results ===" not in text:
        if text:
            print(f"  [output] {text[:400]}")
        return None
    result = (
        parse_field(text, r"Avg latency:\s+([\d.]+)"),
        parse_field(text, r"Min latency:\s+([\d.]+)"),
        parse_field(text, r"p95 latency:\s+([\d.]+)"),
        parse_field(text, r"Throughput\s*:\s*([\d.]+)"),
    )
    if not any(result):
        print(f"  [output] {text[:400]}")
        return None
    return result


# ── Formatting ─────────────────────────────────────────────────────────────────

_HDR = "  {:<10}  {:<6}  {:<8}  {:>8}  {:>8}  {:>8}  {:>12}"
_ROW = "  {:<10}  {:<6}  {:<8}  {:>7}ms  {:>7}ms  {:>7}ms  {:>10} FPS"


def print_table_header() -> None:
    print(_HDR.format("Model", "Prec", "Device", "avg_ms", "min_ms", "p95_ms", "Throughput"))
    print(_HDR.format("-" * 10, "-" * 6, "-" * 8, "-" * 8, "-" * 8, "-" * 8, "-" * 12))


def write_csv_row(writer, tool: str, model: str, variant: str, device: str, perf_hint: str, r: tuple) -> None:
    if writer:
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tool": tool,
                "model": model,
                "variant": variant,
                "device": device,
                "perf_hint": perf_hint,
                "avg_ms": r[0],
                "min_ms": r[1],
                "p95_ms": r[2],
                "throughput_fps": r[3],
            }
        )


# ── Main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    devices = [d.strip() for d in args.devices.split(",")]
    perf_hints = [h.strip() for h in args.perf_hints.split(",")]
    model_filter = {m.strip() for m in args.models.split(",") if m.strip()} or None
    precision_filter = {p.strip() for p in args.precisions.split(",") if p.strip()} or None

    if not COMPILED_DIR.exists() or not any(COMPILED_DIR.rglob("*.xml")):
        print("ERROR: tests/compiled_models/ not found or empty — run stage1 first")
        sys.exit(1)

    models = get_compiled_models(model_filter, precision_filter)

    csv_file = None
    csv_writer = None
    if args.csv:
        csv_file = open(args.csv, "w", newline="")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "timestamp",
                "tool",
                "model",
                "variant",
                "device",
                "perf_hint",
                "avg_ms",
                "min_ms",
                "p95_ms",
                "throughput_fps",
            ],
        )
        csv_writer.writeheader()

    try:
        # ── benchmark_app ──────────────────────────────────────────────────────
        bench = None if args.skip_bench else find_benchmark_app()
        if bench:
            for hint in perf_hints:
                header(f"Step 1: benchmark_app  (hint={hint}, {args.duration}s per run)")
                print_table_header()
                for zip_path, model, variant in models:
                    xml = zip_path.with_suffix(".xml")
                    for device in devices:
                        r = run_benchmark_app(xml, device, args.duration, perf_hint=hint)
                        if r:
                            print(_ROW.format(model, variant, device, *r))
                            write_csv_row(csv_writer, "benchmark_app", model, variant, device, hint, r)
                        else:
                            print(f"  {model:<10}  {variant:<6}  {device:<8}  FAILED")
        else:
            print("benchmark_app not found — skipping")

        if args.skip_runtime:
            return

        # ── yolo_test ──────────────────────────────────────────────────────────
        if not yolo_test_path().exists():
            print(f"  yolo_test not found at {yolo_test_path()}, building...")
            if not build_yolo_test():
                print("  Build failed — skipping runtime tests")
                sys.exit(1)

        pass_count = fail_count = 0
        for hint in perf_hints:
            header(f"Step 2: yolo_test  (OAAX runtime, hint={hint}, warmup={args.warmup}, runs={args.runs})")
            print_table_header()
            for zip_path, model, variant in models:
                batch = 4 if model.endswith("_b4") else 2 if model.endswith("_b2") else 1
                imgsz = 320 if "_320" in model else 640
                # yolo26 models have a non-standard [batch,300,6] output; skip the shape check
                no_validate = model.startswith("yolo26")
                # Match the input dtype baked into the IR by the toolchain:
                # INT8/INT8_NPU → u8, FP16 → f16, FP32_U8 → u8 (u8 with ÷255 baked in), FP32 → f32
                if variant in ("INT8", "INT8_NPU", "FP32_U8"):
                    input_dtype = "u8"
                elif variant == "FP16":
                    input_dtype = "f16"
                else:
                    input_dtype = "f32"
                for device in devices:
                    r = run_yolo_test(
                        zip_path,
                        device,
                        args.warmup,
                        args.runs,
                        batch=batch,
                        perf_hint=hint,
                        imgsz=imgsz,
                        in_flight=args.in_flight,
                        log_level=args.log_level,
                        no_validate=no_validate,
                        input_dtype=input_dtype,
                    )
                    if r:
                        print(_ROW.format(model, variant, device, *r))
                        write_csv_row(csv_writer, "yolo_test", model, variant, device, hint, r)
                        pass_count += 1
                    else:
                        print(f"  {model:<10}  {variant:<6}  {device:<8}  FAILED")
                        fail_count += 1

        header("Stage 2 results")
        if args.csv:
            print(f"  CSV saved to: {args.csv}")
        print(f"  Runtime tests passed: {pass_count}")
        print(f"  Runtime tests failed: {fail_count}")

        if fail_count > 0:
            sys.exit(1)

    finally:
        if csv_file:
            csv_file.close()


if __name__ == "__main__":
    main()
