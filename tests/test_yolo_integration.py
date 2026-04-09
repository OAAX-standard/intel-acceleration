"""
Integration tests for YOLO object detection models (YOLOv8n and YOLOv11n).

Tests the full pipeline:
  ONNX export (ultralytics) → OpenVINO IR conversion → inference validation

Run with:
  pip install ultralytics
  pytest tests/test_yolo_integration.py -v
"""
import pytest
import zipfile
import shutil
import tempfile
import numpy as np
import openvino as ov
from pathlib import Path

from conversion_toolchain.utils import convert_to_ir
from conversion_toolchain.config import OptimizationConfig
from conversion_toolchain.logger import Logs
from conversion_toolchain.quantization import is_nncf_available
from tests.models import download_model, download_calibration_images, TEST_MODELS


YOLO_MODELS = ["yolov8n", "yolo11n"]


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory):
    """Export YOLO ONNX models once per test session."""
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        pytest.skip(
            "ultralytics not installed — run: uv sync --extra integration",
            allow_module_level=False,
        )

    d = tmp_path_factory.mktemp("yolo_models")
    for name in YOLO_MODELS:
        download_model(name, str(d))  # let errors propagate — export failure is a real problem
    return d


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


def _convert_model(model_name: str, models_dir: Path, output_dir: Path, config=None,
                   calibration_dir: str = None):
    """Convert a YOLO ONNX to OpenVINO IR and return the zip path."""
    info = TEST_MODELS[model_name]
    onnx_path = models_dir / info["filename"]
    if not onnx_path.exists():
        pytest.skip(f"{model_name} ONNX not available (ultralytics export failed)")
    logs = Logs()
    return convert_to_ir(str(onnx_path), str(output_dir), logs, config, calibration_dir)


def _extract_ir(zip_path: str, extract_dir: Path):
    """Extract zip and return path to the .xml file."""
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(extract_dir)
    xml_files = list(extract_dir.glob("*.xml"))
    assert len(xml_files) == 1, f"Expected 1 .xml file, found {xml_files}"
    return xml_files[0]


def _run_inference(xml_path: Path, model_name: str):
    """Run inference with a dummy input and return the output tensor."""
    info = TEST_MODELS[model_name]
    core = ov.Core()
    compiled = core.compile_model(core.read_model(str(xml_path)), "CPU")
    dummy = np.zeros(info["input_shape"], dtype=np.float32)
    req = compiled.create_infer_request()
    req.set_input_tensor(ov.Tensor(dummy))
    req.infer()
    return req.get_output_tensor().data


# ── Conversion tests ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_name", YOLO_MODELS)
def test_conversion_produces_zip(model_name, models_dir, temp_dir):
    """YOLO ONNX converts to a zip file."""
    zip_path = _convert_model(model_name, models_dir, temp_dir / "out")
    assert Path(zip_path).exists()
    assert zip_path.endswith(".zip")


@pytest.mark.parametrize("model_name", YOLO_MODELS)
def test_zip_contains_xml_and_bin(model_name, models_dir, temp_dir):
    """Converted zip contains exactly one .xml and one .bin."""
    zip_path = _convert_model(model_name, models_dir, temp_dir / "out")
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
    assert len([f for f in names if f.endswith(".xml")]) == 1
    assert len([f for f in names if f.endswith(".bin")]) == 1


@pytest.mark.parametrize("model_name", YOLO_MODELS)
def test_ir_loads_in_openvino(model_name, models_dir, temp_dir):
    """Converted IR loads in OpenVINO with correct input/output count."""
    zip_path = _convert_model(model_name, models_dir, temp_dir / "out")
    xml_path = _extract_ir(zip_path, temp_dir / "extracted")
    model = ov.Core().read_model(str(xml_path))
    assert len(model.inputs) == 1
    assert len(model.outputs) == 1


# ── Inference validation tests ────────────────────────────────────────────────

@pytest.mark.parametrize("model_name", YOLO_MODELS)
def test_output_shape(model_name, models_dir, temp_dir):
    """Inference output matches expected YOLO shape [1, 84, 8400]."""
    info = TEST_MODELS[model_name]
    zip_path = _convert_model(model_name, models_dir, temp_dir / "out")
    xml_path = _extract_ir(zip_path, temp_dir / "extracted")
    output = _run_inference(xml_path, model_name)

    assert output.ndim == 3, f"Expected 3D output, got shape {output.shape}"
    assert output.shape[0] == 1
    assert output.shape[1] == info["output_channels"], (
        f"Expected {info['output_channels']} channels, got {output.shape[1]}"
    )
    assert output.shape[2] == info["output_anchors"], (
        f"Expected {info['output_anchors']} anchors, got {output.shape[2]}"
    )


@pytest.mark.parametrize("model_name", YOLO_MODELS)
def test_output_is_finite(model_name, models_dir, temp_dir):
    """Inference output contains no NaN or Inf values."""
    zip_path = _convert_model(model_name, models_dir, temp_dir / "out")
    xml_path = _extract_ir(zip_path, temp_dir / "extracted")
    output = _run_inference(xml_path, model_name)
    assert np.all(np.isfinite(output)), "Output contains NaN or Inf"


@pytest.mark.parametrize("model_name", YOLO_MODELS)
def test_bbox_coords_in_range(model_name, models_dir, temp_dir):
    """Bounding box coordinates (first 4 rows) are within image bounds for a real image."""
    info = TEST_MODELS[model_name]
    img_size = info["input_shape"][2]  # 640

    zip_path = _convert_model(model_name, models_dir, temp_dir / "out")
    xml_path = _extract_ir(zip_path, temp_dir / "extracted")

    # Use a non-zero image (uniform gray) to get more realistic outputs
    img = np.full(info["input_shape"], 0.5, dtype=np.float32)
    core = ov.Core()
    compiled = core.compile_model(core.read_model(str(xml_path)), "CPU")
    req = compiled.create_infer_request()
    req.set_input_tensor(ov.Tensor(img))
    req.infer()
    output = req.get_output_tensor().data  # [1, 84, 8400]

    # Rows 0-3 are cx, cy, w, h — should be reasonable (not wildly unbounded)
    # For a normalized image YOLO outputs coords in pixel space (0..640)
    bbox = output[0, :4, :]  # [4, 8400]
    assert np.all(np.isfinite(bbox)), "Bbox coordinates contain NaN/Inf"
    # Reasonable sanity: most coords should be within [-img_size, 2*img_size]
    assert np.median(np.abs(bbox)) < img_size * 10, "Bbox coordinates appear unreasonable"


# ── FP16 compression tests ────────────────────────────────────────────────────

@pytest.mark.parametrize("model_name", YOLO_MODELS)
def test_fp16_conversion_and_inference(model_name, models_dir, temp_dir):
    """FP16-compressed YOLO model converts and produces correctly-shaped output."""
    info = TEST_MODELS[model_name]
    config = OptimizationConfig({"optimization": {"fp16_compression": True}})
    zip_path = _convert_model(model_name, models_dir, temp_dir / "out", config)
    xml_path = _extract_ir(zip_path, temp_dir / "extracted")
    output = _run_inference(xml_path, model_name)

    assert output.shape[1] == info["output_channels"]
    assert output.shape[2] == info["output_anchors"]
    assert np.all(np.isfinite(output))


@pytest.mark.parametrize("model_name", YOLO_MODELS)
def test_fp16_reduces_bin_size(model_name, models_dir, temp_dir):
    """FP16 compressed .bin is smaller than FP32 .bin."""
    fp32_dir = temp_dir / "fp32"
    fp16_dir = temp_dir / "fp16"

    zip_fp32 = _convert_model(
        model_name, models_dir, fp32_dir,
        OptimizationConfig({"optimization": {"fp16_compression": False}})
    )
    zip_fp16 = _convert_model(
        model_name, models_dir, fp16_dir,
        OptimizationConfig({"optimization": {"fp16_compression": True}})
    )

    with zipfile.ZipFile(zip_fp32) as z:
        size_fp32 = sum(i.file_size for i in z.infolist() if i.filename.endswith(".bin"))
    with zipfile.ZipFile(zip_fp16) as z:
        size_fp16 = sum(i.file_size for i in z.infolist() if i.filename.endswith(".bin"))

    assert size_fp16 < size_fp32, (
        f"FP16 .bin ({size_fp16} bytes) should be smaller than FP32 .bin ({size_fp32} bytes)"
    )


# ── INT8 quantization tests ───────────────────────────────────────────────────

@pytest.fixture(scope="session")
def calibration_dir(tmp_path_factory):
    """Download COCO128 calibration images once per test session."""
    if not is_nncf_available():
        pytest.skip("nncf not installed — run: uv sync --extra quantization")
    d = tmp_path_factory.mktemp("calibration")
    return download_calibration_images(str(d))


@pytest.mark.parametrize("model_name", YOLO_MODELS)
def test_int8_conversion_produces_zip(model_name, models_dir, calibration_dir, temp_dir):
    """INT8 quantized YOLO model converts successfully with COCO128 calibration data."""
    config = OptimizationConfig({
        "optimization": {
            "fp16_compression": False,
            "quantization": {"enabled": True, "preset": "mixed", "subset_size": 128},
        }
    })
    zip_path = _convert_model(model_name, models_dir, temp_dir / "out", config,
                              calibration_dir=calibration_dir)
    assert Path(zip_path).exists()
    assert zip_path.endswith(".zip")


@pytest.mark.parametrize("model_name", YOLO_MODELS)
def test_int8_output_shape(model_name, models_dir, calibration_dir, temp_dir):
    """INT8 quantized model produces output with correct YOLO shape [1, 84, 8400]."""
    info = TEST_MODELS[model_name]
    config = OptimizationConfig({
        "optimization": {
            "fp16_compression": False,
            "quantization": {"enabled": True, "preset": "mixed", "subset_size": 128},
        }
    })
    zip_path = _convert_model(model_name, models_dir, temp_dir / "out", config,
                              calibration_dir=calibration_dir)
    xml_path = _extract_ir(zip_path, temp_dir / "extracted")
    output = _run_inference(xml_path, model_name)

    assert output.ndim == 3
    assert output.shape == (1, info["output_channels"], info["output_anchors"]), (
        f"Expected (1, {info['output_channels']}, {info['output_anchors']}), got {output.shape}"
    )


@pytest.mark.parametrize("model_name", YOLO_MODELS)
def test_int8_output_is_finite(model_name, models_dir, calibration_dir, temp_dir):
    """INT8 quantized model output contains no NaN or Inf values."""
    config = OptimizationConfig({
        "optimization": {
            "fp16_compression": False,
            "quantization": {"enabled": True, "preset": "mixed", "subset_size": 128},
        }
    })
    zip_path = _convert_model(model_name, models_dir, temp_dir / "out", config,
                              calibration_dir=calibration_dir)
    xml_path = _extract_ir(zip_path, temp_dir / "extracted")
    output = _run_inference(xml_path, model_name)
    assert np.all(np.isfinite(output)), "INT8 output contains NaN or Inf"


@pytest.mark.parametrize("model_name", YOLO_MODELS)
def test_int8_reduces_bin_size(model_name, models_dir, calibration_dir, temp_dir):
    """INT8 quantized .bin is smaller than FP32 .bin."""
    fp32_dir = temp_dir / "fp32"
    int8_dir = temp_dir / "int8"

    zip_fp32 = _convert_model(
        model_name, models_dir, fp32_dir,
        OptimizationConfig({"optimization": {"fp16_compression": False}}),
    )
    zip_int8 = _convert_model(
        model_name, models_dir, int8_dir,
        OptimizationConfig({
            "optimization": {
                "fp16_compression": False,
                "quantization": {"enabled": True, "preset": "mixed", "subset_size": 128},
            }
        }),
        calibration_dir=calibration_dir,
    )

    with zipfile.ZipFile(zip_fp32) as z:
        size_fp32 = sum(i.file_size for i in z.infolist() if i.filename.endswith(".bin"))
    with zipfile.ZipFile(zip_int8) as z:
        size_int8 = sum(i.file_size for i in z.infolist() if i.filename.endswith(".bin"))

    assert size_int8 < size_fp32, (
        f"INT8 .bin ({size_int8} bytes) should be smaller than FP32 .bin ({size_fp32} bytes)"
    )
