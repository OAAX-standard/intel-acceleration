"""
Tensor-level accuracy comparison between FP32, FP16, and INT8 OpenVINO models.

For each model variant, runs the same 128 COCO128 images through the FP32
reference and each compressed variant, then reports:
  - Cosine similarity   (1.0 = identical direction)
  - Mean absolute error (MAE)
  - Max absolute error

A cosine similarity > 0.99 indicates the quantization had negligible effect on
the output distribution.  This is a proxy metric — not mAP — but it catches
catastrophic quantization failures immediately and is fast to run.
"""

from pathlib import Path

import numpy as np
import openvino as ov
import pytest
from PIL import Image

COMPILED_DIR = Path(__file__).parent / "compiled_models"
CALIB_IMAGES = COMPILED_DIR / "calibration" / "coco128" / "images" / "train2017"

INPUT_NAME = "images"
INPUT_SHAPE = (1, 3, 640, 640)  # NCHW

# Cosine similarity must stay above this for the test to pass
COSINE_SIMILARITY_THRESHOLD = 0.99


# ── helpers ──────────────────────────────────────────────────────────────────


def preprocess(path: Path) -> np.ndarray:
    """Load an image → [1, 3, 640, 640] float32 in [0, 1]."""
    img = Image.open(path).convert("RGB").resize((640, 640), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC [0,1]
    return arr.transpose(2, 0, 1)[np.newaxis]  # → NCHW


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel(), b.ravel()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 1.0


def run_model(compiled: ov.CompiledModel, image: np.ndarray) -> np.ndarray:
    req = compiled.create_infer_request()
    req.set_tensor(INPUT_NAME, ov.Tensor(image))
    req.infer()
    return req.get_output_tensor(0).data.copy()


def load_compiled(xml_path: Path) -> ov.CompiledModel:
    core = ov.Core()
    return core.compile_model(str(xml_path), "CPU", {ov.properties.hint.performance_mode(): "LATENCY"})


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def images():
    if not CALIB_IMAGES.exists():
        pytest.skip("Calibration images not found — run stage1_compile.sh first")
    paths = sorted(CALIB_IMAGES.glob("*.jpg"))[:128]
    if not paths:
        pytest.skip("No .jpg images in calibration directory")
    return [preprocess(p) for p in paths]


# ── tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("model_name", ["yolo11n", "yolov8n"])
@pytest.mark.parametrize("variant", ["FP16", "INT8"])
def test_tensor_similarity_vs_fp32(model_name, variant, images):
    fp32_xml = COMPILED_DIR / model_name / "FP32" / f"{model_name}.xml"
    cmp_xml = COMPILED_DIR / model_name / variant / f"{model_name}.xml"

    if not fp32_xml.exists():
        pytest.skip(f"FP32 model not found: {fp32_xml}")
    if not cmp_xml.exists():
        pytest.skip(f"{variant} model not found: {cmp_xml}")

    fp32_model = load_compiled(fp32_xml)
    cmp_model = load_compiled(cmp_xml)

    cos_sims, maes, max_errs = [], [], []

    for img in images:
        out_fp32 = run_model(fp32_model, img).astype(np.float32)
        out_cmp = run_model(cmp_model, img).astype(np.float32)

        cos_sims.append(cosine_similarity(out_fp32, out_cmp))
        diff = np.abs(out_fp32 - out_cmp)
        maes.append(float(diff.mean()))
        max_errs.append(float(diff.max()))

    mean_cos = float(np.mean(cos_sims))
    min_cos = float(np.min(cos_sims))
    mean_mae = float(np.mean(maes))
    mean_max = float(np.mean(max_errs))

    print(f"\n{model_name} FP32 vs {variant} — {len(images)} images")
    print(f"  Cosine similarity : mean={mean_cos:.6f}  min={min_cos:.6f}")
    print(f"  MAE               : mean={mean_mae:.6f}")
    print(f"  Max abs error     : mean={mean_max:.6f}")

    assert mean_cos >= COSINE_SIMILARITY_THRESHOLD, (
        f"{model_name} {variant}: mean cosine similarity {mean_cos:.6f} "
        f"is below threshold {COSINE_SIMILARITY_THRESHOLD}"
    )
