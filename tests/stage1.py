#!/usr/bin/env python3
"""Stage 1: Run conversion tests and compile YOLO models to OpenVINO IR.

Output: tests/compiled_models/<model>/<variant>/<model>.xml + .bin
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent


def header(title: str) -> None:
    print(f"\n\033[34m=== {title} ===\033[0m")


def main() -> None:
    header("Step 1: Conversion unit tests")
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_conversion.py", "-v", "--tb=short"],
        cwd=ROOT,
        check=True,
    )

    header("Step 2: YOLO 640x640 integration tests (FP32 / FP16 / INT8)")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_yolo_integration.py",
            "-v",
            "--tb=short",
            "-k",
            "not b4 and not 320",
        ],
        cwd=ROOT,
        check=True,
    )

    header("Step 3: YOLO 640x640 batch=4 integration tests (FP32 / FP16 / INT8)")
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_yolo_integration.py", "-v", "--tb=short", "-k", "b4 and not 320"],
        cwd=ROOT,
        check=True,
    )

    header("Step 4: YOLO 320x320 integration tests (FP32 / FP16 / INT8)")
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_yolo_integration.py", "-v", "--tb=short", "-k", "320 and not b4"],
        cwd=ROOT,
        check=True,
    )

    header("Step 5: YOLO 320x320 batch=4 integration tests (FP32 / FP16 / INT8)")
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_yolo_integration.py", "-v", "--tb=short", "-k", "320 and b4"],
        cwd=ROOT,
        check=True,
    )

    header("Step 6: YOLO u8 preprocessing integration tests")
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_yolo_integration.py", "-v", "--tb=short", "-k", "u8"],
        cwd=ROOT,
        check=True,
    )

    header("Step 7: YOLO26s 640x640 and 320x320 integration tests (FP32 / FP16 / INT8)")
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_yolo_integration.py", "-v", "--tb=short", "-k", "26s"],
        cwd=ROOT,
        check=True,
    )

    header("Step 8: YOLO26 batched integration tests (yolo26s batch=4, yolo26m batch=2)")
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_yolo_integration.py", "-v", "--tb=short", "-k", "26_batch"],
        cwd=ROOT,
        check=True,
    )

    print("\nStage 1 complete — compiled models saved to tests/compiled_models/")


if __name__ == "__main__":
    main()
