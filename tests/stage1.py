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

    header("Step 2: YOLO integration tests (FP32 / FP16 / INT8)")
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_yolo_integration.py", "-v", "--tb=short"],
        cwd=ROOT,
        check=True,
    )

    print("\nStage 1 complete — compiled models saved to tests/compiled_models/")


if __name__ == "__main__":
    main()
