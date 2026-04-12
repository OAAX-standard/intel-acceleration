#!/usr/bin/env python3
"""Stage 2: Benchmark and runtime validation using compiled models from Stage 1.

Requires tests/compiled_models/ to be populated by stage1 first.
Runs benchmark_app (if available) and yolo_test across all compiled models.

Usage:
    python tests/stage2.py [--devices CPU,GPU.0] [--duration 10]
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
BUILD_DIR = ROOT / "runtime-library" / "build"
IS_WINDOWS = platform.system() == "Windows"


def header(title: str) -> None:
    print(f"\n\033[34m=== {title} ===\033[0m")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--devices", default="CPU", help="Comma-separated device list (default: CPU)")
    p.add_argument("--duration", type=int, default=10, help="benchmark_app duration in seconds")
    p.add_argument("--runs", type=int, default=300, help="yolo_test inference runs")
    p.add_argument("--warmup", type=int, default=5, help="yolo_test warmup runs")
    p.add_argument("--csv", default="", help="Path to output CSV file")
    p.add_argument("--skip-runtime", action="store_true", help="Skip yolo_test section")
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
        return BUILD_DIR / "Release" / "yolo_test.exe"
    return BUILD_DIR / "yolo_test"


# ── Build yolo_test ────────────────────────────────────────────────────────────


def build_yolo_test() -> bool:
    """Build yolo_test using cmake. Returns True on success."""
    cmake = os.environ.get("CMAKE_BIN", shutil.which("cmake") or "cmake")
    openvino_dir = os.environ.get("OPENVINO_DIR", "")
    runtime_version = (ROOT / "VERSION").read_text().strip()

    cmake_args = [
        cmake,
        "..",
        f"-DRUNTIME_VERSION={runtime_version}",
        f"-DOPENVINO_DIR={openvino_dir}",
        "-DCMAKE_BUILD_TYPE=Release",
    ]

    if not IS_WINDOWS:
        cmake_args.append("-DPLATFORM=X86_64")
        # Create unversioned .so symlinks required by the linker
        link_dir = BUILD_DIR / "openvino_links"
        link_dir.mkdir(parents=True, exist_ok=True)
        libs_dir = Path(openvino_dir) / "libs"
        if not libs_dir.exists():
            libs_dir = Path(openvino_dir) / "lib" / "intel64"
        for lib in libs_dir.glob("*.so.*"):
            stem = lib.name.split(".so")[0]
            target = link_dir / f"{stem}.so"
            if not target.exists():
                target.symlink_to(lib)
        cmake_args.append(f"-DOPENVINO_LINK_DIR={link_dir}")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(cmake_args, cwd=BUILD_DIR, check=True)
        subprocess.run(
            [cmake, "--build", ".", "--config", "Release", "--target", "yolo_test", "-j", str(os.cpu_count() or 4)],
            cwd=BUILD_DIR,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Build failed: {e}")
        return False


# ── Model discovery ────────────────────────────────────────────────────────────


def get_compiled_models() -> list[tuple[Path, str, str]]:
    return [
        (xml, xml.stem, xml.parent.name)
        for variant in ("FP32", "FP16", "INT8")
        for xml in sorted(COMPILED_DIR.glob(f"*/{variant}/*.xml"))
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


def run_benchmark_app(xml: Path, device: str, duration: int) -> tuple | None:
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
            "throughput",
            "-latency_percentile",
            "95",
            "-t",
            str(duration),
        ],
        timeout=duration * 4,
    )
    if not text:
        return None
    return (
        parse_field(text, r"Average:\s+([\d.]+)"),
        parse_field(text, r"Min:\s+([\d.]+)"),
        parse_field(text, r"percentile:\s+([\d.]+)"),
        parse_field(text, r"Throughput:\s+([\d.]+)"),
    )


def run_yolo_test(xml: Path, device: str, warmup: int, runs: int) -> tuple | None:
    binary = yolo_test_path()
    if not binary.exists():
        print(f"  [error] yolo_test binary not found: {binary}")
        return None
    env = os.environ.copy()
    if not IS_WINDOWS:
        env["LD_LIBRARY_PATH"] = f"{BUILD_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
    text = run_process(
        [str(binary), str(xml), device, "--warmup", str(warmup), "--runs", str(runs), "--perf-hint", "throughput"],
        cwd=binary.parent,  # run from binary dir so DLLs are found on Windows
        env=env,
        timeout=runs * 2,
    )
    if not text or "=== Results ===" not in text:
        if text:
            print(f"  [output] {text[:400]}")
        return None
    result = (
        parse_field(text, r"Avg\s*:\s*([\d.]+)"),
        parse_field(text, r"Min\s*:\s*([\d.]+)"),
        parse_field(text, r"p95\s*:\s*([\d.]+)"),
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


def write_csv_row(writer, tool: str, model: str, variant: str, device: str, r: tuple) -> None:
    if writer:
        writer.writerow(
            {
                "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "tool": tool,
                "model": model,
                "variant": variant,
                "device": device,
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

    if not COMPILED_DIR.exists() or not any(COMPILED_DIR.rglob("*.xml")):
        print("ERROR: tests/compiled_models/ not found or empty — run stage1 first")
        sys.exit(1)

    models = get_compiled_models()

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
                "avg_ms",
                "min_ms",
                "p95_ms",
                "throughput_fps",
            ],
        )
        csv_writer.writeheader()

    try:
        # ── benchmark_app ──────────────────────────────────────────────────────
        bench = find_benchmark_app()
        if bench:
            header(f"Step 1: benchmark_app  (hint=throughput, {args.duration}s per run)")
            print_table_header()
            for xml, model, variant in models:
                for device in devices:
                    r = run_benchmark_app(xml, device, args.duration)
                    if r:
                        print(_ROW.format(model, variant, device, *r))
                        write_csv_row(csv_writer, "benchmark_app", model, variant, device, r)
                    else:
                        print(f"  {model:<10}  {variant:<6}  {device:<8}  FAILED")
        else:
            print("benchmark_app not found — skipping")

        if args.skip_runtime:
            return

        # ── yolo_test ──────────────────────────────────────────────────────────
        header(f"Step 2: yolo_test  (OAAX runtime, warmup={args.warmup}, runs={args.runs})")

        if not yolo_test_path().exists():
            print(f"  yolo_test not found at {yolo_test_path()}, building...")
            if not build_yolo_test():
                print("  Build failed — skipping runtime tests")
                sys.exit(1)

        print_table_header()
        pass_count = fail_count = 0
        for xml, model, variant in models:
            for device in devices:
                r = run_yolo_test(xml, device, args.warmup, args.runs)
                if r:
                    print(_ROW.format(model, variant, device, *r))
                    write_csv_row(csv_writer, "yolo_test", model, variant, device, r)
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
