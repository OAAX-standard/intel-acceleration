"""
Download real ONNX models for testing.

Downloads popular models from ONNX Model Zoo and other sources:
- ResNet-18 (Image Classification)
- MobileNetV2 (Image Classification)
- SqueezeNet (Image Classification)
- YOLOv8n (Object Detection) - exported via ultralytics
- YOLOv11n (Object Detection) - exported via ultralytics
"""

import shutil
import urllib.request
import zipfile
from pathlib import Path

COCO128_URL = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"

# Maps model name → (ultralytics .pt name, export batch size, imgsz)
_YOLO_EXPORTS = {
    "yolov8n": ("yolov8n.pt", 1, 640),
    "yolo11n": ("yolo11n.pt", 1, 640),
    "yolo11s": ("yolo11s.pt", 1, 640),
    "yolo11n_b4": ("yolo11n.pt", 4, 640),
    "yolo11s_b4": ("yolo11s.pt", 4, 640),
    "yolo11n_320": ("yolo11n.pt", 1, 320),
    "yolo11s_320": ("yolo11s.pt", 1, 320),
    "yolo11n_320_b4": ("yolo11n.pt", 4, 320),
    "yolo11s_320_b4": ("yolo11s.pt", 4, 320),
    "yolo26s": ("yolo26s.pt", 1, 640),
    "yolo26s_320": ("yolo26s.pt", 1, 320),
    "yolo26m": ("yolo26m.pt", 1, 640),
    "yolo26m_320": ("yolo26m.pt", 1, 320),
    "yolo26s_b4": ("yolo26s.pt", 4, 640),
    "yolo26m_b2": ("yolo26m.pt", 2, 640),
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
    # YOLO models are exported via ultralytics (see export_yolo_model below)
    "yolov8n": {
        "filename": "yolov8n.onnx",
        "task": "object_detection",
        "input_shape": [1, 3, 640, 640],
        "input_name": "images",
        "output_channels": 84,  # 4 bbox + 80 COCO classes
        "output_anchors": 8400,  # 20x20 + 40x40 + 80x80
    },
    "yolo11n": {
        "filename": "yolo11n.onnx",
        "task": "object_detection",
        "input_shape": [1, 3, 640, 640],
        "input_name": "images",
        "output_channels": 84,
        "output_anchors": 8400,
    },
    "yolo11s": {
        "filename": "yolo11s.onnx",
        "task": "object_detection",
        "input_shape": [1, 3, 640, 640],
        "input_name": "images",
        "output_channels": 84,
        "output_anchors": 8400,
    },
    "yolo11n_b4": {
        "filename": "yolo11n_b4.onnx",
        "task": "object_detection",
        "input_shape": [4, 3, 640, 640],
        "input_name": "images",
        "output_channels": 84,
        "output_anchors": 8400,
    },
    "yolo11s_b4": {
        "filename": "yolo11s_b4.onnx",
        "task": "object_detection",
        "input_shape": [4, 3, 640, 640],
        "input_name": "images",
        "output_channels": 84,
        "output_anchors": 8400,
    },
    "yolo11n_320": {
        "filename": "yolo11n_320.onnx",
        "task": "object_detection",
        "input_shape": [1, 3, 320, 320],
        "input_name": "images",
        "output_channels": 84,
        "output_anchors": 2100,  # 10x10 + 20x20 + 40x40
    },
    "yolo11s_320": {
        "filename": "yolo11s_320.onnx",
        "task": "object_detection",
        "input_shape": [1, 3, 320, 320],
        "input_name": "images",
        "output_channels": 84,
        "output_anchors": 2100,
    },
    "yolo11n_320_b4": {
        "filename": "yolo11n_320_b4.onnx",
        "task": "object_detection",
        "input_shape": [4, 3, 320, 320],
        "input_name": "images",
        "output_channels": 84,
        "output_anchors": 2100,
    },
    "yolo11s_320_b4": {
        "filename": "yolo11s_320_b4.onnx",
        "task": "object_detection",
        "input_shape": [4, 3, 320, 320],
        "input_name": "images",
        "output_channels": 84,
        "output_anchors": 2100,
    },
    # yolo26 models use a detection-head output [batch, 300, 6] (fixed regardless of input size)
    "yolo26s_b4": {
        "filename": "yolo26s_b4.onnx",
        "task": "object_detection",
        "input_shape": [4, 3, 640, 640],
        "input_name": "images",
        "output_shape": [4, 300, 6],
    },
    "yolo26m_b2": {
        "filename": "yolo26m_b2.onnx",
        "task": "object_detection",
        "input_shape": [2, 3, 640, 640],
        "input_name": "images",
        "output_shape": [2, 300, 6],
    },
    "yolo26s": {
        "filename": "yolo26s.onnx",
        "task": "object_detection",
        "input_shape": [1, 3, 640, 640],
        "input_name": "images",
        "output_shape": [1, 300, 6],
    },
    "yolo26s_320": {
        "filename": "yolo26s_320.onnx",
        "task": "object_detection",
        "input_shape": [1, 3, 320, 320],
        "input_name": "images",
        "output_shape": [1, 300, 6],
    },
    "yolo26m": {
        "filename": "yolo26m.onnx",
        "task": "object_detection",
        "input_shape": [1, 3, 640, 640],
        "input_name": "images",
        "output_shape": [1, 300, 6],
    },
    "yolo26m_320": {
        "filename": "yolo26m_320.onnx",
        "task": "object_detection",
        "input_shape": [1, 3, 320, 320],
        "input_name": "images",
        "output_shape": [1, 300, 6],
    },
}


def export_yolo_model(model_name: str, output_dir: str) -> str:
    """
    Export a YOLO model to ONNX using ultralytics.

    Args:
        model_name: key from TEST_MODELS / _YOLO_EXPORTS (e.g. 'yolo11n', 'yolo11s_b4')
        output_dir: Directory to save the exported ONNX

    Returns:
        Path to the exported ONNX file
    """
    if model_name not in _YOLO_EXPORTS:
        raise ValueError(f"Unknown YOLO export model: {model_name}")

    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise ImportError("ultralytics is required for YOLO models. Install with: pip install ultralytics") from e

    model_info = TEST_MODELS[model_name]
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dest = output_path / model_info["filename"]

    if dest.exists():
        print(f"✓ Model already exists: {dest}")
        return str(dest)

    ultralytics_name, batch, imgsz = _YOLO_EXPORTS[model_name]
    print(f"Exporting {model_name} to ONNX via ultralytics (batch={batch}, imgsz={imgsz})...")
    model = YOLO(ultralytics_name)
    export_kwargs = dict(format="onnx", imgsz=imgsz, opset=12, simplify=False)
    if batch > 1:
        export_kwargs["batch"] = batch
    exported = model.export(**export_kwargs)
    shutil.move(str(exported), str(dest))
    print(f"✓ Exported to: {dest}")
    return str(dest)


def download_model(model_name: str, output_dir: str = "test_models") -> str:
    """
    Download or export a test model.

    ONNX Zoo models are downloaded via HTTP.
    YOLO models are exported via ultralytics.

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

    # YOLO models: export via ultralytics
    if model_name in _YOLO_EXPORTS:
        return export_yolo_model(model_name, output_dir)

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
    """Download/export all test models."""
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
    Download COCO128 and return the path to its images directory.

    COCO128 is a 128-image subset of COCO used for testing and calibration.
    Source: https://github.com/ultralytics/assets

    Args:
        output_dir: Directory to download and extract into

    Returns:
        Path to the directory containing the .jpg calibration images
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images_dir = output_path / "coco128" / "images" / "train2017"
    if images_dir.exists() and any(images_dir.glob("*.jpg")):
        print(f"✓ Calibration images already available: {images_dir} ({len(list(images_dir.glob('*.jpg')))} images)")
        return str(images_dir)

    zip_path = output_path / "coco128.zip"
    print("Downloading COCO128 calibration images (~7 MB)...")
    urllib.request.urlretrieve(COCO128_URL, str(zip_path))

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(output_path)
    zip_path.unlink()

    n = len(list(images_dir.glob("*.jpg")))
    print(f"✓ Calibration images ready: {images_dir} ({n} images)")
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
