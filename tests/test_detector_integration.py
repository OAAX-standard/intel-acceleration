"""
Stage 1 integration tests: YOLOX model compilation and IR validation.

Uses the compiled_detector_models* session fixtures (conftest.py) which convert
all variants (FP32/FP16/INT8/INT8_NPU) once and cache them to tests/compiled_models/.

Run via Stage 1:
    python tests/stage1.py
Or directly:
    pytest tests/test_detector_integration.py -v
"""

from pathlib import Path

import numpy as np
import openvino as ov
import pytest

from tests.conftest import YOLOX_MODEL_U8, YOLOX_MODELS, YOLOX_MODELS_320, YOLOX_MODELS_320_B4, YOLOX_MODELS_B4
from tests.models import TEST_MODELS

VARIANTS = ["FP32", "FP16", "INT8", "INT8_NPU"]
ALL_MODELS = YOLOX_MODELS + YOLOX_MODELS_B4 + YOLOX_MODELS_320 + YOLOX_MODELS_320_B4


def _fixture_name(model_name: str) -> str:
    """Session fixture that compiles `model_name` (see conftest.py)."""
    if model_name.endswith("_320_b4"):
        return "compiled_detector_models_320_batch4"
    if model_name.endswith("_b4"):
        return "compiled_detector_models_batch4"
    if "_320" in model_name:
        return "compiled_detector_models_320"
    return "compiled_detector_models"


@pytest.fixture
def get_xml(request):
    """Return a lookup (model_name, variant) -> IR .xml path; skips if not compiled (e.g. no NNCF)."""

    def get(model_name: str, variant: str) -> Path:
        compiled = request.getfixturevalue(_fixture_name(model_name))
        xml = compiled.get((model_name, variant))
        if xml is None:
            pytest.skip(f"{variant} not available for {model_name}")
        return xml

    return get


def _infer(xml: Path, model_name: str, img_value: float = 0.0) -> np.ndarray:
    info = TEST_MODELS[model_name]
    core = ov.Core()
    compiled = core.compile_model(core.read_model(str(xml)), "CPU")
    input_dtype = compiled.input(0).element_type.to_dtype()
    img = np.full(info["input_shape"], img_value, dtype=input_dtype)
    req = compiled.create_infer_request()
    req.set_input_tensor(ov.Tensor(img))
    req.infer()
    return req.get_output_tensor().data


# ── IR file presence ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_name", ALL_MODELS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_ir_files_exist(get_xml, model_name, variant):
    """Compiled .xml and .bin files are present on disk."""
    xml = get_xml(model_name, variant)
    assert xml.exists(), f"{xml} not found"
    assert xml.with_suffix(".bin").exists(), f"{xml.with_suffix('.bin')} not found"


# ── OpenVINO loading ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_name", ALL_MODELS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_ir_loads_in_openvino(get_xml, model_name, variant):
    """Compiled IR is readable by OpenVINO Core with one input and one output."""
    model = ov.Core().read_model(str(get_xml(model_name, variant)))
    assert len(model.inputs) == 1
    assert len(model.outputs) == 1


# ── Output shape ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_name", ALL_MODELS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_output_shape(get_xml, model_name, variant):
    """Inference output matches the YOLOX shape [batch, anchors, 85]."""
    expected = tuple(TEST_MODELS[model_name]["output_shape"])
    output = _infer(get_xml(model_name, variant), model_name)
    assert output.shape == expected, f"Expected {expected}, got {output.shape}"


# ── Numerical sanity ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_name", ALL_MODELS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_output_is_finite(get_xml, model_name, variant):
    """Inference output contains no NaN or Inf values."""
    output = _infer(get_xml(model_name, variant), model_name)
    assert np.all(np.isfinite(output)), f"{variant} output has NaN/Inf"


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_bbox_coords_in_range(get_xml, model_name):
    """Decoded box columns (cx, cy, w, h) are within a reasonable pixel range."""
    output = _infer(get_xml(model_name, "FP32"), model_name, img_value=0.5)
    bbox = output[..., :4]
    assert np.all(np.isfinite(bbox))
    assert np.median(np.abs(bbox)) < TEST_MODELS[model_name]["input_shape"][2] * 10


# ── Model size: compressed variants must be smaller than FP32 ─────────────────


@pytest.mark.parametrize("model_name", ALL_MODELS)
def test_fp16_bin_smaller_than_fp32(get_xml, model_name):
    """FP16 .bin is smaller than FP32 .bin."""
    fp32 = get_xml(model_name, "FP32").with_suffix(".bin")
    fp16 = get_xml(model_name, "FP16").with_suffix(".bin")
    assert (
        fp16.stat().st_size < fp32.stat().st_size
    ), f"FP16 ({fp16.stat().st_size} B) not smaller than FP32 ({fp32.stat().st_size} B)"


@pytest.mark.parametrize("model_name", ALL_MODELS)
@pytest.mark.parametrize("variant", ["INT8", "INT8_NPU"])
def test_int8_bin_smaller_than_fp32(get_xml, model_name, variant):
    """INT8/INT8_NPU .bin is smaller than FP32 .bin."""
    fp32 = get_xml(model_name, "FP32").with_suffix(".bin")
    quant = get_xml(model_name, variant).with_suffix(".bin")
    assert (
        quant.stat().st_size < fp32.stat().st_size
    ), f"{variant} ({quant.stat().st_size} B) not smaller than FP32 ({fp32.stat().st_size} B)"


# ── Batch correctness ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_name", ["yolox_nano", "yolox_s"])
def test_b4_matches_batch1(get_xml, model_name):
    """Batch=4 output equals four independent batch=1 runs (guards the batch patch in models.adapt_yolox)."""
    info = TEST_MODELS[f"{model_name}_b4"]
    x = np.random.RandomState(0).rand(*info["input_shape"]).astype(np.float32)

    core = ov.Core()
    c1 = core.compile_model(core.read_model(str(get_xml(model_name, "FP32"))), "CPU")
    c4 = core.compile_model(core.read_model(str(get_xml(f"{model_name}_b4", "FP32"))), "CPU")

    expected = np.concatenate([c1(x[i : i + 1])[0] for i in range(4)])
    np.testing.assert_allclose(c4(x)[0], expected, rtol=1e-4, atol=1e-3)


# ── U8 preprocessing tests ────────────────────────────────────────────────────


@pytest.fixture
def u8_xml(compiled_detector_models_u8) -> Path:
    xml = compiled_detector_models_u8.get((YOLOX_MODEL_U8, "FP32_U8"))
    if xml is None:
        pytest.skip("FP32_U8 variant not compiled")
    return xml


def _infer_u8(xml: Path, model_name: str) -> np.ndarray:
    """Run inference with raw uint8 inputs [0, 255]."""
    info = TEST_MODELS[model_name]
    core = ov.Core()
    compiled = core.compile_model(core.read_model(str(xml)), "CPU")
    img = np.random.randint(0, 256, info["input_shape"], dtype=np.uint8)
    req = compiled.create_infer_request()
    req.set_input_tensor(ov.Tensor(img))
    req.infer()
    return req.get_output_tensor().data


def test_u8_ir_input_type(u8_xml):
    """FP32_U8 IR has u8 as its input boundary type."""
    model = ov.Core().read_model(str(u8_xml))
    assert (
        model.inputs[0].get_element_type() == ov.Type.u8
    ), f"Expected u8 input, got {model.inputs[0].get_element_type()}"


def test_u8_output_shape(u8_xml):
    """FP32_U8 model produces the correct YOLOX output shape from u8 input."""
    expected = tuple(TEST_MODELS[YOLOX_MODEL_U8]["output_shape"])
    output = _infer_u8(u8_xml, YOLOX_MODEL_U8)
    assert output.shape == expected, f"Unexpected output shape: {output.shape}"


def test_u8_output_is_finite(u8_xml):
    """FP32_U8 model output contains no NaN or Inf values."""
    output = _infer_u8(u8_xml, YOLOX_MODEL_U8)
    assert np.all(np.isfinite(output)), "FP32_U8 output has NaN/Inf"


def test_u8_matches_f32_baseline(get_xml, u8_xml):
    """FP32_U8 model fed raw u8 pixels produces the same output as the FP32 baseline
    fed manually-normalized f32 inputs (÷255), within float32 tolerance."""
    np.random.seed(0)
    info = TEST_MODELS[YOLOX_MODEL_U8]
    pixels_u8 = np.random.randint(0, 256, info["input_shape"], dtype=np.uint8)
    pixels_f32 = pixels_u8.astype(np.float32) / 255.0

    core = ov.Core()

    compiled_fp32 = core.compile_model(core.read_model(str(get_xml(YOLOX_MODEL_U8, "FP32"))), "CPU")
    req = compiled_fp32.create_infer_request()
    req.set_input_tensor(ov.Tensor(pixels_f32))
    req.infer()
    ref = req.get_output_tensor().data

    compiled_u8 = core.compile_model(core.read_model(str(u8_xml)), "CPU")
    req = compiled_u8.create_infer_request()
    req.set_input_tensor(ov.Tensor(pixels_u8))
    req.infer()
    out = req.get_output_tensor().data

    np.testing.assert_allclose(ref, out, rtol=1e-4, atol=1e-4, err_msg="FP32_U8 output should match FP32 baseline")
