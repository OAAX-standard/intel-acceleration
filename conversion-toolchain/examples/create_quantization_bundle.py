#!/usr/bin/env python3
"""
Example script to create an input bundle for quantized model conversion.

This creates a zip file containing:
- ONNX model
- config.json with quantization settings
- Calibration images (generated synthetically for demo)
"""

import json
import zipfile
from pathlib import Path
import numpy as np
from PIL import Image

def create_synthetic_calibration_images(output_dir: Path, num_images: int = 50):
    """Create synthetic calibration images for demonstration"""
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        # Create random RGB image
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        img.save(output_dir / f"calib_{i:03d}.jpg")

    print(f"Created {num_images} synthetic calibration images in {output_dir}")


def create_config_json(output_path: Path, quantization_enabled: bool = True):
    """Create configuration JSON file"""
    config = {
        "model_file": "model.onnx",
        "optimization": {
            "fp16_compression": False,  # Disable FP16 when using INT8
            "quantization": {
                "enabled": quantization_enabled,
                "mode": "int8",
                "calibration_data": "calibration/",
                "preset": "mixed",
                "subset_size": 50
            }
        }
    }

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Created config.json at {output_path}")


def create_bundle(
    onnx_model_path: str,
    output_bundle: str,
    with_quantization: bool = True,
    num_calib_images: int = 50
):
    """
    Create input bundle zip file

    Args:
        onnx_model_path: Path to existing ONNX model
        output_bundle: Path for output zip file
        with_quantization: Include quantization configuration
        num_calib_images: Number of calibration images to generate
    """
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create config.json
        config_path = temp_path / "config.json"
        create_config_json(config_path, with_quantization)

        # Create calibration images if quantization is enabled
        if with_quantization:
            calib_dir = temp_path / "calibration"
            create_synthetic_calibration_images(calib_dir, num_calib_images)

        # Create zip bundle
        print(f"\nCreating bundle: {output_bundle}")
        with zipfile.ZipFile(output_bundle, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add ONNX model
            zipf.write(onnx_model_path, arcname="model.onnx")
            print(f"  Added: model.onnx")

            # Add config
            zipf.write(config_path, arcname="config.json")
            print(f"  Added: config.json")

            # Add calibration images
            if with_quantization:
                calib_dir = temp_path / "calibration"
                for img_file in calib_dir.glob("*.jpg"):
                    arcname = f"calibration/{img_file.name}"
                    zipf.write(img_file, arcname=arcname)
                print(f"  Added: {num_calib_images} calibration images")

        print(f"\n✅ Bundle created successfully: {output_bundle}")
        print(f"\nTo convert:")
        print(f"  conversion_toolchain --input-zip {output_bundle} --output-dir ./output")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create model conversion bundle")
    parser.add_argument("--onnx-model", required=True, help="Path to ONNX model file")
    parser.add_argument("--output", default="model_bundle.zip", help="Output bundle path")
    parser.add_argument("--no-quantization", action="store_true", help="Disable quantization")
    parser.add_argument("--num-images", type=int, default=50, help="Number of calibration images")

    args = parser.parse_args()

    create_bundle(
        onnx_model_path=args.onnx_model,
        output_bundle=args.output,
        with_quantization=not args.no_quantization,
        num_calib_images=args.num_images
    )
