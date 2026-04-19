"""
Shared session fixtures for the OAAX intel-acceleration test suite.

compiled_yolo_models converts all YOLO variants (FP32/FP16/INT8) once per
session via the Docker toolchain image, caching to tests/compiled_models/.
Stage 1 populates this cache; Stage 2 reads from it without re-converting.
"""

import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pytest

from tests.models import download_calibration_images, download_model

COMPILED_DIR = Path(__file__).parent / "compiled_models"
DOCKER_IMAGE = "oaax-intel-toolchain:latest"
YOLO_MODELS = ["yolov8n", "yolo11n", "yolo11s"]
YOLO_MODELS_B4 = ["yolo11n_b4", "yolo11s_b4"]

_CONFIGS = {
    "FP32": {"optimization": {"fp16_compression": False}},
    "FP16": {"optimization": {"fp16_compression": True}},
    "INT8": {
        "optimization": {
            "fp16_compression": False,
            "quantization": {"enabled": True, "preset": "mixed", "subset_size": 128},
        }
    },
}

_CONFIGS_B4 = {
    "FP32": {"optimization": {"fp16_compression": False}},
    "FP16": {"optimization": {"fp16_compression": True}},
    "INT8": {
        "optimization": {
            "fp16_compression": False,
            "quantization": {"enabled": True, "preset": "mixed", "subset_size": 128},
        }
    },
}


def _docker_image_available() -> bool:
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        if r.returncode != 0:
            return False
        r = subprocess.run(
            ["docker", "images", "-q", DOCKER_IMAGE],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


def _convert_with_docker(
    model_name: str,
    variant: str,
    onnx_path: Path,
    out_dir: Path,
    config: dict,
    calib_dir: Path | None,
) -> None:
    """Convert one model+variant using the Docker toolchain image."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        bundle = tmp_path / "bundle.zip"
        docker_out = tmp_path / "output"
        docker_out.mkdir()

        with zipfile.ZipFile(bundle, "w") as z:
            # Name the ONNX after the model so the IR files are named accordingly
            z.write(onnx_path, arcname=f"{model_name}.onnx")
            z.writestr("config.json", json.dumps(config))
            if calib_dir:
                for img in sorted(calib_dir.glob("*.jpg")):
                    z.write(img, arcname=f"calibration/{img.name}")

        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{tmp_path}:/input",
                "-v",
                f"{docker_out}:/output",
                DOCKER_IMAGE,
                "/input/bundle.zip",
                "/output",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Docker conversion failed for {model_name} {variant} "
                f"(exit {result.returncode}):\n{result.stdout}\n{result.stderr}"
            )

        output_zips = list(docker_out.glob("*.zip"))
        if not output_zips:
            raise RuntimeError(f"No output zip produced for {model_name} {variant}")

        out_dir.mkdir(parents=True, exist_ok=True)
        # Keep the toolchain zip alongside the extracted IR so the C++ runtime
        # can load it directly.
        dest_zip = out_dir / f"{model_name}.zip"
        import shutil as _shutil

        _shutil.copy2(output_zips[0], dest_zip)
        with zipfile.ZipFile(output_zips[0]) as z:
            z.extractall(out_dir)


@pytest.fixture(scope="session")
def calibration_dir() -> Path:
    """Download COCO128 calibration images and return their directory."""
    d = COMPILED_DIR / "calibration"
    d.mkdir(parents=True, exist_ok=True)
    return Path(download_calibration_images(str(d)))


@pytest.fixture(scope="session")
def compiled_yolo_models(calibration_dir: Path) -> dict:
    """
    Convert all YOLO models (FP32/FP16/INT8) once via the Docker toolchain image,
    caching results to tests/compiled_models/.

    Returns {(model_name, variant): Path-to-xml}.
    """
    if not _docker_image_available():
        pytest.skip(
            f"Docker image '{DOCKER_IMAGE}' not available. "
            f"Build with: IMAGE_NAME=oaax-intel-toolchain bash conversion-toolchain/build-toolchain.sh"
        )

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        pytest.skip("ultralytics not installed — run: uv sync --extra integration", allow_module_level=False)

    onnx_dir = COMPILED_DIR / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for model_name in YOLO_MODELS:
        onnx = Path(download_model(model_name, str(onnx_dir)))
        for variant, config in _CONFIGS.items():
            xml = COMPILED_DIR / model_name / variant / f"{model_name}.xml"
            if xml.exists():
                # Ensure a zip exists alongside the extracted IR for the C++ runtime.
                zip_path = xml.with_suffix(".zip")
                if not zip_path.exists() and xml.with_suffix(".bin").exists():
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                        zf.write(xml, arcname=xml.name)
                        zf.write(xml.with_suffix(".bin"), arcname=xml.with_suffix(".bin").name)
                result[(model_name, variant)] = xml
                continue
            _convert_with_docker(
                model_name,
                variant,
                onnx,
                COMPILED_DIR / model_name / variant,
                config,
                calibration_dir if variant == "INT8" else None,
            )
            result[(model_name, variant)] = xml

    return result


@pytest.fixture(scope="session")
def compiled_yolo_models_batch4(calibration_dir: Path) -> dict:
    """
    Export yolo11n and yolo11s with batch=4, then convert (FP32/FP16) via
    the Docker toolchain image, caching to tests/compiled_models/.

    Returns {(model_name, variant): Path-to-xml}.
    """
    if not _docker_image_available():
        pytest.skip(
            f"Docker image '{DOCKER_IMAGE}' not available. "
            f"Build with: IMAGE_NAME=oaax-intel-toolchain bash conversion-toolchain/build-toolchain.sh"
        )

    try:
        import ultralytics  # noqa: F401
    except ImportError:
        pytest.skip("ultralytics not installed — run: uv sync --extra integration", allow_module_level=False)

    onnx_dir = COMPILED_DIR / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    result = {}
    for model_name in YOLO_MODELS_B4:
        onnx = Path(download_model(model_name, str(onnx_dir)))
        for variant, config in _CONFIGS_B4.items():
            xml = COMPILED_DIR / model_name / variant / f"{model_name}.xml"
            if xml.exists():
                zip_path = xml.with_suffix(".zip")
                if not zip_path.exists() and xml.with_suffix(".bin").exists():
                    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                        zf.write(xml, arcname=xml.name)
                        zf.write(xml.with_suffix(".bin"), arcname=xml.with_suffix(".bin").name)
                result[(model_name, variant)] = xml
                continue
            _convert_with_docker(
                model_name,
                variant,
                onnx,
                COMPILED_DIR / model_name / variant,
                config,
                calibration_dir if variant == "INT8" else None,
            )
            result[(model_name, variant)] = xml

    return result
