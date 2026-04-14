#!/usr/bin/env python3
"""
Simple script to create a basic input bundle from an ONNX model.

Creates a zip file containing just the ONNX model (with default FP16 compression).
For quantization, use create_quantization_bundle.py instead.
"""

import argparse
import zipfile
from pathlib import Path


def create_simple_bundle(onnx_model_path: str, output_bundle: str):
    """
    Create a simple input bundle with just the ONNX model.

    Args:
        onnx_model_path: Path to existing ONNX model
        output_bundle: Path for output zip file
    """
    onnx_path = Path(onnx_model_path)

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_model_path}")

    print(f"Creating simple bundle: {output_bundle}")

    with zipfile.ZipFile(output_bundle, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add ONNX model
        zipf.write(onnx_path, arcname="model.onnx")
        print(f"  Added: {onnx_path.name} -> model.onnx")

    print(f"\n✅ Bundle created successfully: {output_bundle}")
    print("\nThis bundle will use default settings:")
    print("  - FP16 compression: ✅ Enabled")
    print("  - INT8 quantization: ❌ Disabled")
    print("\nTo convert:")
    print(f"  conversion_toolchain {output_bundle} ./output")
    print("\nTo customize settings, add a config.json to the zip or use create_quantization_bundle.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a simple ONNX model bundle for conversion")
    parser.add_argument("onnx_model", help="Path to ONNX model file")
    parser.add_argument("-o", "--output", default=None, help="Output bundle path (default: <model_name>_bundle.zip)")

    args = parser.parse_args()

    # Generate output filename if not provided
    if args.output is None:
        model_name = Path(args.onnx_model).stem
        output = f"{model_name}_bundle.zip"
    else:
        output = args.output

    create_simple_bundle(args.onnx_model, output)
