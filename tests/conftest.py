"""
Shared session fixtures for the OAAX intel-acceleration test suite.

compiled_yolo_models converts all YOLO variants (FP32/FP16/INT8) once per
session, caching to tests/compiled_models/. Stage 1 populates this cache;
Stage 2 reads from it without re-converting.
"""
import zipfile
import pytest
from pathlib import Path

from conversion_toolchain.utils import convert_to_ir
from conversion_toolchain.config import OptimizationConfig
from conversion_toolchain.logger import Logs
from conversion_toolchain.quantization import is_nncf_available
from tests.models import download_model, download_calibration_images

COMPILED_DIR = Path(__file__).parent / "compiled_models"

_VARIANTS = {
    "FP32": OptimizationConfig({"optimization": {"fp16_compression": False}}),
    "FP16": OptimizationConfig({"optimization": {"fp16_compression": True}}),
    "INT8": OptimizationConfig({
        "optimization": {
            "fp16_compression": False,
            "quantization": {"enabled": True, "preset": "mixed", "subset_size": 128},
        }
    }),
}

YOLO_MODELS = ["yolov8n", "yolo11n"]


@pytest.fixture(scope="session")
def calibration_dir():
    """Return path to COCO128 calibration images, or None if NNCF unavailable."""
    if not is_nncf_available():
        return None
    d = COMPILED_DIR / "calibration"
    d.mkdir(parents=True, exist_ok=True)
    return download_calibration_images(str(d))


@pytest.fixture(scope="session")
def compiled_yolo_models(calibration_dir):
    """
    Convert all YOLO models (FP32/FP16/INT8) once, caching to tests/compiled_models/.
    Returns {(model_name, variant): Path-to-xml}.  INT8 omitted if NNCF unavailable.
    """
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        pytest.skip("ultralytics not installed — run: uv sync --extra integration",
                    allow_module_level=False)

    onnx_dir = COMPILED_DIR / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    variants = {k: v for k, v in _VARIANTS.items()
                if k != "INT8" or calibration_dir is not None}

    result = {}
    for model_name in YOLO_MODELS:
        onnx = download_model(model_name, str(onnx_dir))
        for variant, config in variants.items():
            xml = COMPILED_DIR / model_name / variant / f"{model_name}.xml"
            if xml.exists():
                result[(model_name, variant)] = xml
                continue
            out_dir = COMPILED_DIR / model_name / variant
            out_dir.mkdir(parents=True, exist_ok=True)
            logs = Logs()
            zip_path = convert_to_ir(
                onnx, str(out_dir / "zip"), logs, config,
                calibration_dir if variant == "INT8" else None,
            )
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(out_dir)
            result[(model_name, variant)] = xml

    return result
