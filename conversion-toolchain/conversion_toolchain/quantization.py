"""
Model quantization using NNCF (Neural Network Compression Framework)
"""
import numpy as np
from pathlib import Path
from typing import List, Optional
from PIL import Image
import openvino as ov
import openvino.runtime as _ov_runtime

# Patch: OpenVINO 2024.6+ moved Node out of the top-level namespace.
# NNCF references ov.Node in class-level type annotations (evaluated eagerly),
# so we restore the alias before NNCF is imported.
if not hasattr(ov, 'Node'):
    ov.Node = _ov_runtime.Node

try:
    import nncf
    NNCF_AVAILABLE = True
except ImportError:
    NNCF_AVAILABLE = False


class CalibrationDataLoader:
    """Loads and preprocesses calibration images"""

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    def __init__(self, calibration_dir: str, input_shape: Optional[List[int]] = None):
        """
        Initialize calibration data loader

        Args:
            calibration_dir: Directory containing calibration images
            input_shape: Expected input shape [batch, channels, height, width]
                        If None, will infer from first image
        """
        self.calibration_dir = Path(calibration_dir)
        self.input_shape = input_shape
        self.image_files = self._find_images()

        if not self.image_files:
            raise ValueError(f"No calibration images found in {calibration_dir}")

    def _find_images(self) -> List[Path]:
        """Find all supported image files in calibration directory"""
        images = []
        for ext in self.SUPPORTED_FORMATS:
            images.extend(self.calibration_dir.glob(f"*{ext}"))
            images.extend(self.calibration_dir.glob(f"*{ext.upper()}"))
        return sorted(images)

    def _preprocess_image(self, image_path: Path) -> np.ndarray:
        """
        Preprocess a single image to match model input requirements

        Args:
            image_path: Path to image file

        Returns:
            Preprocessed image as numpy array
        """
        # Load image
        img = Image.open(image_path).convert('RGB')

        # Determine target size
        if self.input_shape and len(self.input_shape) == 4:
            # Shape is [batch, channels, height, width]
            target_height = self.input_shape[2]
            target_width = self.input_shape[3]
        else:
            # Default to common size
            target_height = 224
            target_width = 224

        # Resize image
        img = img.resize((target_width, target_height), Image.Resampling.BILINEAR)

        # Convert to numpy array
        img_array = np.array(img, dtype=np.float32)

        # Normalize to [0, 1]
        img_array = img_array / 255.0

        # Convert HWC to CHW (Height, Width, Channels -> Channels, Height, Width)
        img_array = np.transpose(img_array, (2, 0, 1))

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def get_data_generator(self, subset_size: int = 300):
        """
        Create a generator for calibration data

        Args:
            subset_size: Maximum number of images to use

        Yields:
            Preprocessed images as numpy arrays
        """
        num_images = min(subset_size, len(self.image_files))

        for image_path in self.image_files[:num_images]:
            try:
                yield self._preprocess_image(image_path)
            except Exception as e:
                # Skip problematic images
                print(f"Warning: Failed to load {image_path}: {e}")
                continue

    def __len__(self) -> int:
        """Return number of calibration images"""
        return len(self.image_files)


def quantize_model(
    model: ov.Model,
    calibration_dir: str,
    preset: str = "mixed",
    subset_size: int = 300,
    logs = None
) -> ov.Model:
    """
    Quantize OpenVINO model to INT8 using NNCF

    Args:
        model: OpenVINO model to quantize
        calibration_dir: Directory containing calibration images
        preset: NNCF quantization preset ('performance', 'mixed', or 'accuracy')
        subset_size: Number of calibration samples to use
        logs: Logger instance for tracking progress

    Returns:
        Quantized OpenVINO model

    Raises:
        ImportError: If NNCF is not installed
        ValueError: If calibration data is invalid
    """
    if not NNCF_AVAILABLE:
        error_msg = (
            "NNCF is not installed. Install it with: uv sync --extra quantization"
        )
        if logs:
            logs.add_message("Quantization Error", {"Error": error_msg})
        raise ImportError(error_msg)

    if logs:
        logs.add_message("Starting INT8 quantization", {
            "Preset": preset,
            "Calibration directory": calibration_dir,
            "Subset size": subset_size
        })

    # Get model input shape
    input_shape = None
    if model.inputs:
        input_shape = model.inputs[0].get_partial_shape()
        if input_shape.is_static:
            input_shape = list(input_shape.get_shape())

    # Load calibration data
    try:
        data_loader = CalibrationDataLoader(calibration_dir, input_shape)
        if logs:
            logs.add_message("Calibration data loaded", {
                "Total images found": len(data_loader),
                "Images to use": min(subset_size, len(data_loader))
            })
    except Exception as e:
        if logs:
            logs.add_message("Failed to load calibration data", {"Error": str(e)})
        raise

    # Create calibration dataset
    calibration_dataset = nncf.Dataset(
        data_loader.get_data_generator(subset_size)
    )

    # Map preset to NNCF preset
    preset_mapping = {
        "performance": nncf.QuantizationPreset.PERFORMANCE,
        "mixed": nncf.QuantizationPreset.MIXED,
    }

    nncf_preset = preset_mapping.get(preset, nncf.QuantizationPreset.MIXED)

    if logs:
        logs.add_message("Running NNCF quantization (this may take several minutes)")

    # Quantize model
    try:
        quantized_model = nncf.quantize(
            model,
            calibration_dataset,
            preset=nncf_preset,
            # Additional settings for better quality
            model_type=nncf.ModelType.TRANSFORMER if "bert" in str(model.get_friendly_name()).lower() else None,
        )

        if logs:
            logs.add_message("INT8 quantization completed successfully", {
                "Preset used": preset
            })

        return quantized_model

    except Exception as e:
        if logs:
            logs.add_message("Quantization failed", {"Error": str(e)})
        raise


def is_nncf_available() -> bool:
    """Check if NNCF is available"""
    return NNCF_AVAILABLE
