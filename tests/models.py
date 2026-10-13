"""
Download real ONNX models for testing.

Downloads popular models from ONNX Model Zoo and other sources:
- ResNet-18 (Image Classification)
- MobileNetV2 (Image Classification)
- SqueezeNet (Image Classification)
- YOLOX nano / tiny / s (Object Detection, Apache-2.0) - official ONNX release, adapted by
  adapt_yolox() to the batch size and resolution a test needs

Only permissively licensed models are used (no AGPL dependencies such as ultralytics).
"""

import hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

YOLOX_RELEASE_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"
# COCO images are fetched individually; the host only serves plain HTTP.
COCO_IMAGE_URL = "http://images.cocodataset.org/train2017/{image_id}.jpg"
CALIBRATION_IDS_FILE = Path(__file__).with_name("calibration_images.txt")

# Official release asset -> (sha256, native input size). Pinned so a changed upstream file fails loudly.
YOLOX_BASE = {
    "yolox_nano": ("c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d", 416),
    "yolox_tiny": ("427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7", 416),
    "yolox_s": ("c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063", 640),
}

# Test model name -> (base model, input size, batch size)
_YOLOX_VARIANTS = {
    "yolox_nano": ("yolox_nano", 416, 1),
    "yolox_tiny": ("yolox_tiny", 416, 1),
    "yolox_s": ("yolox_s", 640, 1),
    "yolox_nano_b4": ("yolox_nano", 416, 4),
    "yolox_s_b4": ("yolox_s", 640, 4),
    "yolox_nano_320": ("yolox_nano", 320, 1),
    "yolox_s_320": ("yolox_s", 320, 1),
    "yolox_nano_320_b4": ("yolox_nano", 320, 4),
    "yolox_s_320_b4": ("yolox_s", 320, 4),
}

_STRIDES = (8, 16, 32)
_YOLOX_COLUMNS = 85  # cx, cy, w, h, objectness, 80 COCO class scores


def _yolox_entry(base: str, imgsz: int, batch: int, name: str) -> dict:
    anchors = sum((imgsz // s) ** 2 for s in _STRIDES)
    return {
        "filename": f"{name}.onnx",
        "task": "object_detection",
        "input_shape": [batch, 3, imgsz, imgsz],
        "input_name": "images",
        "output_channels": _YOLOX_COLUMNS,
        "output_anchors": anchors,
        "output_shape": [batch, anchors, _YOLOX_COLUMNS],
    }


TEST_MODELS = {
    "resnet18": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx",
        "filename": "resnet18-v1-7.onnx",
        "task": "image_classification",
        "input_shape": [1, 3, 224, 224],
    },
    "mobilenetv2": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx",
        "filename": "mobilenetv2-7.onnx",
        "task": "image_classification",
        "input_shape": [1, 3, 224, 224],
    },
    "squeezenet": {
        "url": "https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-7.onnx",
        "filename": "squeezenet1.0-7.onnx",
        "task": "image_classification",
        "input_shape": [1, 3, 224, 224],
    },
    # YOLOX models: downloaded from the official release and adapted (see adapt_yolox below)
    **{name: _yolox_entry(*variant, name) for name, variant in _YOLOX_VARIANTS.items()},
}


def adapt_yolox(src: str, dst: str, batch: int, imgsz: int) -> None:
    """
    Turn an official YOLOX ONNX (batch 1, 0-255 input) into a test model.

    - Batch: the three head Reshape nodes hardcode a leading batch of 1, which folds extra
      batches into the anchor axis. Setting that dim to 0 ("copy from input") makes the
      graph batch-correct (batch=N output is identical to N independent batch=1 runs).
    - Resolution: grids are computed from the input, so only the declared input shape changes.
    - Input range: a x255 is prepended so the model takes [0, 1] floats like the toolchain's
      calibration loader and the u8 preprocessing (scale_values=255) expect.
    """
    import numpy as np
    import onnx
    from onnx import helper
    from onnx import numpy_helper as nh

    model = onnx.load(src)
    graph = model.graph
    inits = {i.name: i for i in graph.initializer}

    patched = 0
    for node in graph.node:
        if node.op_type != "Reshape" or node.input[1] not in inits:
            continue
        shape = nh.to_array(inits[node.input[1]])
        if shape.ndim == 1 and shape.tolist()[:1] == [1] and len(shape) == 3:
            shape = shape.copy()
            shape[0] = 0
            inits[node.input[1]].CopyFrom(nh.from_array(shape, node.input[1]))
            patched += 1
    if patched != len(_STRIDES):
        raise RuntimeError(f"Unexpected YOLOX graph: patched {patched} head reshapes, expected {len(_STRIDES)}")

    dims = graph.input[0].type.tensor_type.shape.dim
    dims[0].dim_value, dims[2].dim_value, dims[3].dim_value = batch, imgsz, imgsz
    for out in graph.output:  # stale static shapes; let the runtime infer them
        del out.type.tensor_type.shape.dim[:]
    del graph.value_info[:]

    input_name = graph.input[0].name
    scaled = f"{input_name}_x255"
    for node in graph.node:
        node.input[:] = [scaled if i == input_name else i for i in node.input]
    graph.initializer.append(nh.from_array(np.float32(255.0), "x255_scale"))
    graph.node.insert(0, helper.make_node("Mul", [input_name, "x255_scale"], [scaled]))

    onnx.save(model, dst)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_yolox_base(base: str, output_dir: Path) -> Path:
    """Download (once) and verify the official YOLOX ONNX for `base`."""
    dest = output_dir / f"{base}.official.onnx"
    expected = YOLOX_BASE[base][0]
    if not dest.exists():
        url = f"{YOLOX_RELEASE_URL}/{base}.onnx"
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, str(dest))
    actual = _sha256(dest)
    if actual != expected:
        dest.unlink()
        raise RuntimeError(f"{base}.onnx checksum mismatch (got {actual}, expected {expected})")
    return dest


def export_yolox_model(model_name: str, output_dir: str) -> str:
    """
    Build a YOLOX test model (see _YOLOX_VARIANTS) as ONNX.

    Args:
        model_name: key from TEST_MODELS / _YOLOX_VARIANTS (e.g. 'yolox_nano', 'yolox_s_b4')
        output_dir: Directory to save the ONNX

    Returns:
        Path to the ONNX file
    """
    if model_name not in _YOLOX_VARIANTS:
        raise ValueError(f"Unknown YOLOX model: {model_name}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dest = output_path / TEST_MODELS[model_name]["filename"]
    if dest.exists():
        print(f"✓ Model already exists: {dest}")
        return str(dest)

    base, imgsz, batch = _YOLOX_VARIANTS[model_name]
    adapt_yolox(str(download_yolox_base(base, output_path)), str(dest), batch, imgsz)
    print(f"✓ Built {model_name} (batch={batch}, imgsz={imgsz}): {dest}")
    return str(dest)


def download_model(model_name: str, output_dir: str = "test_models") -> str:
    """
    Download or build a test model.

    ONNX Zoo models are downloaded via HTTP.
    YOLOX models are built from the official release (see export_yolox_model).

    Args:
        model_name: Model key from TEST_MODELS
        output_dir: Directory to save the model

    Returns:
        Path to the model file
    """
    if model_name not in TEST_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(TEST_MODELS.keys())}")

    model_info = TEST_MODELS[model_name]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / model_info["filename"]

    if model_path.exists():
        print(f"✓ Model already exists: {model_path}")
        return str(model_path)

    # YOLOX models: official release + adaptation
    if model_name in _YOLOX_VARIANTS:
        return export_yolox_model(model_name, output_dir)

    # Classification models: download from ONNX Model Zoo
    print(f"Downloading {model_name} from ONNX Model Zoo...")
    print(f"  URL: {model_info['url']}")
    print(f"  Destination: {model_path}")

    try:
        urllib.request.urlretrieve(model_info["url"], str(model_path))
        print(f"✓ Downloaded successfully: {model_path}")
        return str(model_path)
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")
        raise


def download_all_models(output_dir: str = "test_models"):
    """Download/build all test models."""
    print("Preparing test models...")
    print("=" * 60)

    for model_name in TEST_MODELS.keys():
        try:
            download_model(model_name, output_dir)
        except Exception as e:
            print(f"Warning: Failed to prepare {model_name}: {e}")

    print("=" * 60)
    print("Done!")


def download_calibration_images(output_dir: str) -> str:
    """
    Download a 128-image COCO subset and return the path to its images directory.

    The image ids are listed in calibration_images.txt; images come straight from cocodataset.org.

    Args:
        output_dir: Directory to download into

    Returns:
        Path to the directory containing the .jpg calibration images
    """
    ids = CALIBRATION_IDS_FILE.read_text().split()
    images_dir = Path(output_dir) / "coco_calibration"
    images_dir.mkdir(parents=True, exist_ok=True)

    def fetch(image_id: str) -> None:
        dest = images_dir / f"{image_id}.jpg"
        if dest.exists():
            return
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(COCO_IMAGE_URL.format(image_id=image_id), str(tmp))
        tmp.rename(dest)  # never leave a truncated image behind

    missing = [i for i in ids if not (images_dir / f"{i}.jpg").exists()]
    if missing:
        print(f"Downloading {len(missing)} COCO calibration images...")
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fetch, missing))

    print(f"✓ Calibration images ready: {images_dir} ({len(ids)} images)")
    return str(images_dir)


def get_model_info(model_name: str) -> dict:
    """Get information about a test model."""
    if model_name not in TEST_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return TEST_MODELS[model_name].copy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download ONNX test models")
    parser.add_argument(
        "--model", choices=list(TEST_MODELS.keys()) + ["all"], default="all", help="Model to download (default: all)"
    )
    parser.add_argument("--output-dir", default="test_models", help="Output directory (default: test_models)")

    args = parser.parse_args()

    if args.model == "all":
        download_all_models(args.output_dir)
    else:
        download_model(args.model, args.output_dir)
