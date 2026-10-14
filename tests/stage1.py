#!/usr/bin/env python3
"""Stage 1: Run conversion tests and compile YOLOX models to OpenVINO IR.

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

    steps = [
        ("Step 2: YOLOX native-resolution integration tests (FP32 / FP16 / INT8)", "not b4 and not 320"),
        ("Step 3: YOLOX native-resolution batch=4 integration tests (FP32 / FP16 / INT8)", "b4 and not 320"),
        ("Step 4: YOLOX 320x320 integration tests (FP32 / FP16 / INT8)", "320 and not b4"),
        ("Step 5: YOLOX 320x320 batch=4 integration tests (FP32 / FP16 / INT8)", "320 and b4"),
        ("Step 6: YOLOX u8 preprocessing integration tests", "u8"),
    ]
    for title, selector in steps:
        header(title)
        subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_detector_integration.py", "-v", "--tb=short", "-k", selector],
            cwd=ROOT,
            check=True,
        )

    print("\nStage 1 complete — compiled models saved to tests/compiled_models/")


if __name__ == "__main__":
    main()
