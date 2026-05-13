"""
Stage 1 integration tests: YOLO model compilation and IR validation.

Uses the compiled_yolo_models session fixture (conftest.py) which converts
all variants (FP32/FP16/INT8) once and caches them to tests/compiled_models/.

Run via Stage 1:
    python tests/stage1.py
Or directly:
    pytest tests/test_yolo_integration.py -v
"""

from pathlib import Path

import numpy as np
import openvino as ov
import pytest

from tests.models import TEST_MODELS

YOLO_MODELS = ["yolov8n", "yolo11n", "yolo11s"]
YOLO_MODELS_B4 = ["yolo11n_b4", "yolo11s_b4"]
YOLO_MODELS_320 = ["yolo11n_320", "yolo11s_320"]
YOLO_MODELS_320_B4 = ["yolo11n_320_b4", "yolo11s_320_b4"]
YOLO_MODELS_26S = ["yolo26s", "yolo26s_320", "yolo26m", "yolo26m_320"]
YOLO_MODELS_26_BATCH = ["yolo26s_b4", "yolo26m_b2"]


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


def _get(compiled_yolo_models, model_name, variant):
    """Return xml path or skip if variant was not compiled (e.g. no NNCF)."""
    xml = compiled_yolo_models.get((model_name, variant))
    if xml is None:
        pytest.skip(f"{variant} not available (NNCF not installed)")
    return xml


# ── IR file presence ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_name", YOLO_MODELS)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_ir_files_exist(compiled_yolo_models, model_name, variant):
    """Compiled .xml and .bin files are present on disk."""
    xml = _get(compiled_yolo_models, model_name, variant)
    assert xml.exists(), f"{xml} not found"
    assert xml.with_suffix(".bin").exists(), f"{xml.with_suffix('.bin')} not found"


# ── OpenVINO loading ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_name", YOLO_MODELS)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_ir_loads_in_openvino(compiled_yolo_models, model_name, variant):
    """Compiled IR is readable by OpenVINO Core with one input and one output."""
    xml = _get(compiled_yolo_models, model_name, variant)
    model = ov.Core().read_model(str(xml))
    assert len(model.inputs) == 1
    assert len(model.outputs) == 1


# ── Output shape ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_name", YOLO_MODELS)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_output_shape(compiled_yolo_models, model_name, variant):
    """Inference output matches expected YOLO shape [1, 84, 8400]."""
    xml = _get(compiled_yolo_models, model_name, variant)
    info = TEST_MODELS[model_name]
    output = _infer(xml, model_name)
    assert output.shape == (
        1,
        info["output_channels"],
        info["output_anchors"],
    ), f"Expected (1, {info['output_channels']}, {info['output_anchors']}), got {output.shape}"


# ── Numerical sanity ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_name", YOLO_MODELS)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_output_is_finite(compiled_yolo_models, model_name, variant):
    """Inference output contains no NaN or Inf values."""
    xml = _get(compiled_yolo_models, model_name, variant)
    output = _infer(xml, model_name)
    assert np.all(np.isfinite(output)), f"{variant} output has NaN/Inf"


@pytest.mark.parametrize("model_name", YOLO_MODELS)
def test_bbox_coords_in_range(compiled_yolo_models, model_name):
    """Bounding-box coordinates (rows 0-3) are within a reasonable range."""
    xml = compiled_yolo_models[(model_name, "FP32")]
    info = TEST_MODELS[model_name]
    output = _infer(xml, model_name, img_value=0.5)
    bbox = output[0, :4, :]  # [4, 8400]
    assert np.all(np.isfinite(bbox))
    assert np.median(np.abs(bbox)) < info["input_shape"][2] * 10


# ── Model size: compressed variants must be smaller than FP32 ─────────────────


@pytest.mark.parametrize("model_name", YOLO_MODELS)
def test_fp16_bin_smaller_than_fp32(compiled_yolo_models, model_name):
    """FP16 .bin is smaller than FP32 .bin."""
    fp32 = compiled_yolo_models[(model_name, "FP32")].with_suffix(".bin")
    fp16 = compiled_yolo_models[(model_name, "FP16")].with_suffix(".bin")
    assert (
        fp16.stat().st_size < fp32.stat().st_size
    ), f"FP16 ({fp16.stat().st_size} B) not smaller than FP32 ({fp32.stat().st_size} B)"


@pytest.mark.parametrize("model_name", YOLO_MODELS)
@pytest.mark.parametrize("variant", ["INT8", "INT8_NPU"])
def test_int8_bin_smaller_than_fp32(compiled_yolo_models, model_name, variant):
    """INT8/INT8_NPU .bin is smaller than FP32 .bin."""
    xml = _get(compiled_yolo_models, model_name, variant)
    fp32 = compiled_yolo_models[(model_name, "FP32")].with_suffix(".bin")
    quant = xml.with_suffix(".bin")
    assert (
        quant.stat().st_size < fp32.stat().st_size
    ), f"{variant} ({quant.stat().st_size} B) not smaller than FP32 ({fp32.stat().st_size} B)"


# ── Batch=4 tests ─────────────────────────────────────────────────────────────


def _get_b4(compiled_yolo_models_batch4, model_name, variant):
    xml = compiled_yolo_models_batch4.get((model_name, variant))
    if xml is None:
        pytest.skip(f"{variant} batch=4 not available")
    return xml


@pytest.mark.parametrize("model_name", YOLO_MODELS_B4)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_b4_ir_files_exist(compiled_yolo_models_batch4, model_name, variant):
    """Batch=4 compiled .xml and .bin files are present on disk."""
    xml = _get_b4(compiled_yolo_models_batch4, model_name, variant)
    assert xml.exists(), f"{xml} not found"
    assert xml.with_suffix(".bin").exists(), f"{xml.with_suffix('.bin')} not found"


@pytest.mark.parametrize("model_name", YOLO_MODELS_B4)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_b4_ir_loads_in_openvino(compiled_yolo_models_batch4, model_name, variant):
    """Batch=4 IR is readable by OpenVINO Core with one input and one output."""
    xml = _get_b4(compiled_yolo_models_batch4, model_name, variant)
    model = ov.Core().read_model(str(xml))
    assert len(model.inputs) == 1
    assert len(model.outputs) == 1


@pytest.mark.parametrize("model_name", YOLO_MODELS_B4)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_b4_output_shape(compiled_yolo_models_batch4, model_name, variant):
    """Batch=4 inference output matches expected shape [4, 84, 8400]."""
    xml = _get_b4(compiled_yolo_models_batch4, model_name, variant)
    info = TEST_MODELS[model_name]
    output = _infer(xml, model_name)
    assert output.shape == (
        4,
        info["output_channels"],
        info["output_anchors"],
    ), f"Expected (4, {info['output_channels']}, {info['output_anchors']}), got {output.shape}"


@pytest.mark.parametrize("model_name", YOLO_MODELS_B4)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_b4_output_is_finite(compiled_yolo_models_batch4, model_name, variant):
    """Batch=4 inference output contains no NaN or Inf values."""
    xml = _get_b4(compiled_yolo_models_batch4, model_name, variant)
    output = _infer(xml, model_name)
    assert np.all(np.isfinite(output)), f"{variant} batch=4 output has NaN/Inf"


@pytest.mark.parametrize("model_name", YOLO_MODELS_B4)
@pytest.mark.parametrize("variant", ["INT8", "INT8_NPU"])
def test_b4_int8_bin_smaller_than_fp32(compiled_yolo_models_batch4, model_name, variant):
    """Batch=4 INT8/INT8_NPU .bin is smaller than FP32 .bin."""
    fp32 = compiled_yolo_models_batch4[(model_name, "FP32")].with_suffix(".bin")
    quant = _get_b4(compiled_yolo_models_batch4, model_name, variant).with_suffix(".bin")
    assert (
        quant.stat().st_size < fp32.stat().st_size
    ), f"{variant} ({quant.stat().st_size} B) not smaller than FP32 ({fp32.stat().st_size} B)"


# ── 320x320 tests ─────────────────────────────────────────────────────────────


def _get_320(compiled_yolo_models_320, model_name, variant):
    xml = compiled_yolo_models_320.get((model_name, variant))
    if xml is None:
        pytest.skip(f"{variant} 320x320 not available")
    return xml


def _get_320_b4(compiled_yolo_models_320_batch4, model_name, variant):
    xml = compiled_yolo_models_320_batch4.get((model_name, variant))
    if xml is None:
        pytest.skip(f"{variant} 320x320 batch=4 not available")
    return xml


@pytest.mark.parametrize("model_name", YOLO_MODELS_320)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_320_ir_files_exist(compiled_yolo_models_320, model_name, variant):
    """320x320 compiled .xml and .bin files are present on disk."""
    xml = _get_320(compiled_yolo_models_320, model_name, variant)
    assert xml.exists()
    assert xml.with_suffix(".bin").exists()


@pytest.mark.parametrize("model_name", YOLO_MODELS_320)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_320_output_shape(compiled_yolo_models_320, model_name, variant):
    """320x320 output matches expected shape [1, 84, 2100]."""
    xml = _get_320(compiled_yolo_models_320, model_name, variant)
    info = TEST_MODELS[model_name]
    output = _infer(xml, model_name)
    assert output.shape == (
        1,
        info["output_channels"],
        info["output_anchors"],
    ), f"Expected (1, {info['output_channels']}, {info['output_anchors']}), got {output.shape}"


@pytest.mark.parametrize("model_name", YOLO_MODELS_320)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_320_output_is_finite(compiled_yolo_models_320, model_name, variant):
    """320x320 output contains no NaN or Inf values."""
    xml = _get_320(compiled_yolo_models_320, model_name, variant)
    output = _infer(xml, model_name)
    assert np.all(np.isfinite(output))


@pytest.mark.parametrize("model_name", YOLO_MODELS_320)
def test_320_fp16_smaller_than_fp32(compiled_yolo_models_320, model_name):
    """320x320 FP16 .bin is smaller than FP32 .bin."""
    fp32 = compiled_yolo_models_320[(model_name, "FP32")].with_suffix(".bin")
    fp16 = compiled_yolo_models_320[(model_name, "FP16")].with_suffix(".bin")
    assert fp16.stat().st_size < fp32.stat().st_size


@pytest.mark.parametrize("model_name", YOLO_MODELS_320)
@pytest.mark.parametrize("variant", ["INT8", "INT8_NPU"])
def test_320_int8_smaller_than_fp32(compiled_yolo_models_320, model_name, variant):
    """320x320 INT8/INT8_NPU .bin is smaller than FP32 .bin."""
    fp32 = compiled_yolo_models_320[(model_name, "FP32")].with_suffix(".bin")
    quant = _get_320(compiled_yolo_models_320, model_name, variant).with_suffix(".bin")
    assert quant.stat().st_size < fp32.stat().st_size


@pytest.mark.parametrize("model_name", YOLO_MODELS_320_B4)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_320_b4_ir_files_exist(compiled_yolo_models_320_batch4, model_name, variant):
    """320x320 batch=4 compiled .xml and .bin files are present on disk."""
    xml = _get_320_b4(compiled_yolo_models_320_batch4, model_name, variant)
    assert xml.exists()
    assert xml.with_suffix(".bin").exists()


@pytest.mark.parametrize("model_name", YOLO_MODELS_320_B4)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_320_b4_output_shape(compiled_yolo_models_320_batch4, model_name, variant):
    """320x320 batch=4 output matches expected shape [4, 84, 2100]."""
    xml = _get_320_b4(compiled_yolo_models_320_batch4, model_name, variant)
    info = TEST_MODELS[model_name]
    output = _infer(xml, model_name)
    assert output.shape == (
        4,
        info["output_channels"],
        info["output_anchors"],
    ), f"Expected (4, {info['output_channels']}, {info['output_anchors']}), got {output.shape}"


@pytest.mark.parametrize("model_name", YOLO_MODELS_320_B4)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_320_b4_output_is_finite(compiled_yolo_models_320_batch4, model_name, variant):
    """320x320 batch=4 output contains no NaN or Inf values."""
    xml = _get_320_b4(compiled_yolo_models_320_batch4, model_name, variant)
    output = _infer(xml, model_name)
    assert np.all(np.isfinite(output))


# ── U8 preprocessing tests ────────────────────────────────────────────────────


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


def test_u8_ir_input_type(compiled_yolo_models_u8):
    """FP32_U8 IR has u8 as its input boundary type."""
    xml = compiled_yolo_models_u8.get(("yolo11n", "FP32_U8"))
    if xml is None:
        pytest.skip("FP32_U8 variant not compiled")
    model = ov.Core().read_model(str(xml))
    assert (
        model.inputs[0].get_element_type() == ov.Type.u8
    ), f"Expected u8 input, got {model.inputs[0].get_element_type()}"


def test_u8_output_shape(compiled_yolo_models_u8):
    """FP32_U8 model produces correct YOLO output shape [1, 84, 8400] from u8 input."""
    xml = compiled_yolo_models_u8.get(("yolo11n", "FP32_U8"))
    if xml is None:
        pytest.skip("FP32_U8 variant not compiled")
    info = TEST_MODELS["yolo11n"]
    output = _infer_u8(xml, "yolo11n")
    assert output.shape == (
        1,
        info["output_channels"],
        info["output_anchors"],
    ), f"Unexpected output shape: {output.shape}"


def test_u8_output_is_finite(compiled_yolo_models_u8):
    """FP32_U8 model output contains no NaN or Inf values."""
    xml = compiled_yolo_models_u8.get(("yolo11n", "FP32_U8"))
    if xml is None:
        pytest.skip("FP32_U8 variant not compiled")
    output = _infer_u8(xml, "yolo11n")
    assert np.all(np.isfinite(output)), "FP32_U8 output has NaN/Inf"


# ── yolo26s tests (640x640 and 320x320) ──────────────────────────────────────
# yolo26s has a different output format: [batch, 300, 6] instead of [batch, 84, anchors]


def _get_26s(compiled_yolo_models_26s, model_name, variant):
    xml = compiled_yolo_models_26s.get((model_name, variant))
    if xml is None:
        pytest.skip(f"{variant} not available for {model_name}")
    return xml


@pytest.mark.parametrize("model_name", YOLO_MODELS_26S)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_26s_ir_files_exist(compiled_yolo_models_26s, model_name, variant):
    """yolo26s compiled .xml and .bin files are present on disk."""
    xml = _get_26s(compiled_yolo_models_26s, model_name, variant)
    assert xml.exists()
    assert xml.with_suffix(".bin").exists()


@pytest.mark.parametrize("model_name", YOLO_MODELS_26S)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_26s_ir_loads_in_openvino(compiled_yolo_models_26s, model_name, variant):
    """yolo26s IR is readable by OpenVINO Core with one input and one output."""
    xml = _get_26s(compiled_yolo_models_26s, model_name, variant)
    model = ov.Core().read_model(str(xml))
    assert len(model.inputs) == 1
    assert len(model.outputs) == 1


@pytest.mark.parametrize("model_name", YOLO_MODELS_26S)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_26s_output_shape(compiled_yolo_models_26s, model_name, variant):
    """yolo26s output matches expected shape [1, 300, 6]."""
    xml = _get_26s(compiled_yolo_models_26s, model_name, variant)
    info = TEST_MODELS[model_name]
    output = _infer(xml, model_name)
    assert output.shape == tuple(info["output_shape"]), f"Expected {info['output_shape']}, got {list(output.shape)}"


@pytest.mark.parametrize("model_name", YOLO_MODELS_26S)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_26s_output_is_finite(compiled_yolo_models_26s, model_name, variant):
    """yolo26s output contains no NaN or Inf values."""
    xml = _get_26s(compiled_yolo_models_26s, model_name, variant)
    output = _infer(xml, model_name)
    assert np.all(np.isfinite(output))


@pytest.mark.parametrize("model_name", YOLO_MODELS_26S)
def test_26s_fp16_smaller_than_fp32(compiled_yolo_models_26s, model_name):
    """yolo26s FP16 .bin is smaller than FP32 .bin."""
    fp32 = compiled_yolo_models_26s[(model_name, "FP32")].with_suffix(".bin")
    fp16 = compiled_yolo_models_26s[(model_name, "FP16")].with_suffix(".bin")
    assert fp16.stat().st_size < fp32.stat().st_size


@pytest.mark.parametrize("model_name", YOLO_MODELS_26S)
@pytest.mark.parametrize("variant", ["INT8", "INT8_NPU"])
def test_26s_int8_smaller_than_fp32(compiled_yolo_models_26s, model_name, variant):
    """yolo26s INT8/INT8_NPU .bin is smaller than FP32 .bin."""
    fp32 = compiled_yolo_models_26s[(model_name, "FP32")].with_suffix(".bin")
    quant = _get_26s(compiled_yolo_models_26s, model_name, variant).with_suffix(".bin")
    assert quant.stat().st_size < fp32.stat().st_size


# ── yolo26 batch tests (yolo26s batch=4, yolo26m batch=2) ────────────────────


def _get_26_batch(compiled_yolo_models_26_batch, model_name, variant):
    xml = compiled_yolo_models_26_batch.get((model_name, variant))
    if xml is None:
        pytest.skip(f"{variant} not available for {model_name}")
    return xml


@pytest.mark.parametrize("model_name", YOLO_MODELS_26_BATCH)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_26_batch_ir_files_exist(compiled_yolo_models_26_batch, model_name, variant):
    """yolo26 batched compiled .xml and .bin files are present on disk."""
    xml = _get_26_batch(compiled_yolo_models_26_batch, model_name, variant)
    assert xml.exists()
    assert xml.with_suffix(".bin").exists()


@pytest.mark.parametrize("model_name", YOLO_MODELS_26_BATCH)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_26_batch_ir_loads_in_openvino(compiled_yolo_models_26_batch, model_name, variant):
    """yolo26 batched IR is readable by OpenVINO Core with one input and one output."""
    xml = _get_26_batch(compiled_yolo_models_26_batch, model_name, variant)
    model = ov.Core().read_model(str(xml))
    assert len(model.inputs) == 1
    assert len(model.outputs) == 1


@pytest.mark.parametrize("model_name", YOLO_MODELS_26_BATCH)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_26_batch_output_shape(compiled_yolo_models_26_batch, model_name, variant):
    """yolo26 batched output matches expected shape [batch, 300, 6]."""
    xml = _get_26_batch(compiled_yolo_models_26_batch, model_name, variant)
    info = TEST_MODELS[model_name]
    output = _infer(xml, model_name)
    assert output.shape == tuple(info["output_shape"]), f"Expected {info['output_shape']}, got {list(output.shape)}"


@pytest.mark.parametrize("model_name", YOLO_MODELS_26_BATCH)
@pytest.mark.parametrize("variant", ["FP32", "FP16", "INT8", "INT8_NPU"])
def test_26_batch_output_is_finite(compiled_yolo_models_26_batch, model_name, variant):
    """yolo26 batched output contains no NaN or Inf values."""
    xml = _get_26_batch(compiled_yolo_models_26_batch, model_name, variant)
    output = _infer(xml, model_name)
    assert np.all(np.isfinite(output))


@pytest.mark.parametrize("model_name", YOLO_MODELS_26_BATCH)
def test_26_batch_fp16_smaller_than_fp32(compiled_yolo_models_26_batch, model_name):
    """yolo26 batched FP16 .bin is smaller than FP32 .bin."""
    fp32 = compiled_yolo_models_26_batch[(model_name, "FP32")].with_suffix(".bin")
    fp16 = compiled_yolo_models_26_batch[(model_name, "FP16")].with_suffix(".bin")
    assert fp16.stat().st_size < fp32.stat().st_size


@pytest.mark.parametrize("model_name", YOLO_MODELS_26_BATCH)
@pytest.mark.parametrize("variant", ["INT8", "INT8_NPU"])
def test_26_batch_int8_smaller_than_fp32(compiled_yolo_models_26_batch, model_name, variant):
    """yolo26 batched INT8/INT8_NPU .bin is smaller than FP32 .bin."""
    fp32 = compiled_yolo_models_26_batch[(model_name, "FP32")].with_suffix(".bin")
    quant = _get_26_batch(compiled_yolo_models_26_batch, model_name, variant).with_suffix(".bin")
    assert quant.stat().st_size < fp32.stat().st_size


def test_u8_matches_f32_baseline(compiled_yolo_models, compiled_yolo_models_u8):
    """FP32_U8 model fed raw u8 pixels produces the same output as the FP32 baseline
    fed manually-normalized f32 inputs (÷255), within float32 tolerance."""
    xml_fp32 = compiled_yolo_models.get(("yolo11n", "FP32"))
    xml_u8 = compiled_yolo_models_u8.get(("yolo11n", "FP32_U8"))
    if xml_fp32 is None or xml_u8 is None:
        pytest.skip("FP32 or FP32_U8 variant not compiled")

    np.random.seed(0)
    info = TEST_MODELS["yolo11n"]
    pixels_u8 = np.random.randint(0, 256, info["input_shape"], dtype=np.uint8)
    pixels_f32 = pixels_u8.astype(np.float32) / 255.0

    core = ov.Core()

    compiled_fp32 = core.compile_model(core.read_model(str(xml_fp32)), "CPU")
    req = compiled_fp32.create_infer_request()
    req.set_input_tensor(ov.Tensor(pixels_f32))
    req.infer()
    ref = req.get_output_tensor().data

    compiled_u8 = core.compile_model(core.read_model(str(xml_u8)), "CPU")
    req = compiled_u8.create_infer_request()
    req.set_input_tensor(ov.Tensor(pixels_u8))
    req.infer()
    out = req.get_output_tensor().data

    np.testing.assert_allclose(ref, out, rtol=1e-4, atol=1e-4, err_msg="FP32_U8 output should match FP32 baseline")
