"""
Download real ONNX models for testing.

Downloads popular models from ONNX Model Zoo and other sources:
- ResNet-18 (Image Classification)
- MobileNetV2 (Image Classification)
- YOLOv8n (Object Detection)
"""

import urllib.request
import os
from pathlib import Path


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
}


def download_model(model_name: str, output_dir: str = "test_models") -> str:
    """
    Download a test model if it doesn't exist.

    Args:
        model_name: Name of the model to download (e.g., 'resnet18')
        output_dir: Directory to save the model

    Returns:
        Path to the downloaded model file
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
    """Download all test models."""
    print("Downloading test models from ONNX Model Zoo...")
    print("=" * 60)

    for model_name in TEST_MODELS.keys():
        try:
            download_model(model_name, output_dir)
        except Exception as e:
            print(f"Warning: Failed to download {model_name}: {e}")

    print("=" * 60)
    print("Download complete!")


def get_model_info(model_name: str) -> dict:
    """Get information about a test model."""
    if model_name not in TEST_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return TEST_MODELS[model_name].copy()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download ONNX test models")
    parser.add_argument(
        "--model",
        choices=list(TEST_MODELS.keys()) + ["all"],
        default="all",
        help="Model to download (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        default="test_models",
        help="Output directory (default: test_models)"
    )

    args = parser.parse_args()

    if args.model == "all":
        download_all_models(args.output_dir)
    else:
        download_model(args.model, args.output_dir)
