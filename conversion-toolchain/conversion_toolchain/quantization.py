"""
Model quantization using NNCF (Neural Network Compression Framework)
"""

from pathlib import Path

import numpy as np
import openvino as ov
from PIL import Image

# Patch: OpenVINO 2024.6 moved Node out of the top-level namespace into
# openvino.runtime. NNCF 2.14 references ov.Node in class-level annotations
# (evaluated eagerly at import time), so we restore the alias before NNCF loads.
# OpenVINO 2026.1+ restores ov.Node directly, so the patch is a no-op there.
if not hasattr(ov, "Node"):
    try:
        import openvino.runtime as _ov_runtime

        ov.Node = _ov_runtime.Node
    except ModuleNotFoundError:
        pass

try:
    import nncf

    NNCF_AVAILABLE = True
except ImportError:
    NNCF_AVAILABLE = False


class CalibrationDataLoader:
    """Loads and preprocesses calibration images"""

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(self, calibration_dir: str, input_shape: list[int] | None = None, input_dtype: str = "f32"):
        """
        Initialize calibration data loader

        Args:
            calibration_dir: Directory containing calibration images
            input_shape: Expected input shape [batch, channels, height, width]
                        If None, will infer from first image
            input_dtype: Element type expected by the model boundary ("f32" or "u8").
                        When "u8", images are returned as raw uint8 [0,255] — the model
                        is responsible for normalization (e.g. via baked-in PPP).
        """
        self.calibration_dir = Path(calibration_dir)
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        self.image_files = self._find_images()

        if not self.image_files:
            raise ValueError(f"No calibration images found in {calibration_dir}")

    def _find_images(self) -> list[Path]:
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
        img = Image.open(image_path).convert("RGB")

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

        # Convert HWC to CHW; keep as u8 or normalise to [0,1] f32
        # depending on what the model's input boundary expects.
        if self.input_dtype == "u8":
            img_array = np.array(img, dtype=np.uint8)
        else:
            img_array = np.array(img, dtype=np.float32) / 255.0

        img_array = np.transpose(img_array, (2, 0, 1))

        # Add batch dimension; tile to match model's fixed batch size if > 1
        img_array = np.expand_dims(img_array, axis=0)
        batch_size = (
            self.input_shape[0] if self.input_shape and len(self.input_shape) == 4 and self.input_shape[0] > 1 else 1
        )
        if batch_size > 1:
            img_array = np.repeat(img_array, batch_size, axis=0)

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
    logs=None,
    input_dtype: str = "f32",
    target_device: str = "any",
) -> ov.Model:
    """
    Quantize OpenVINO model to INT8 using NNCF

    Args:
        model: OpenVINO model to quantize
        calibration_dir: Directory containing calibration images
        preset: NNCF quantization preset ('performance', 'mixed', or 'accuracy')
        subset_size: Number of calibration samples to use
        logs: Logger instance for tracking progress
        input_dtype: Element type expected by the model boundary
        target_device: Deployment target ('any', 'cpu', 'gpu', 'npu'). When 'npu',
            preset is forced to 'performance' and attention/DFL subgraphs are kept
            in FP16 via IgnoredScope to avoid NPU requantization overhead.

    Returns:
        Quantized OpenVINO model

    Raises:
        ImportError: If NNCF is not installed
        ValueError: If calibration data is invalid
    """
    if not NNCF_AVAILABLE:
        error_msg = "NNCF is not installed. Install it with: uv sync --extra quantization"
        if logs:
            logs.add_message("Quantization Error", {"Error": error_msg})
        raise ImportError(error_msg)

    if logs:
        logs.add_message(
            "Starting INT8 quantization",
            {
                "Preset": preset,
                "Target device": target_device,
                "Calibration directory": calibration_dir,
                "Subset size": subset_size,
            },
        )

    # Get model input shape
    input_shape = None
    if model.inputs:
        input_shape = model.inputs[0].get_partial_shape()
        if input_shape.is_static:
            input_shape = list(input_shape.get_shape())

    # Load calibration data
    try:
        data_loader = CalibrationDataLoader(calibration_dir, input_shape, input_dtype)
        if logs:
            logs.add_message(
                "Calibration data loaded",
                {"Total images found": len(data_loader), "Images to use": min(subset_size, len(data_loader))},
            )
    except Exception as e:
        if logs:
            logs.add_message("Failed to load calibration data", {"Error": str(e)})
        raise

    # Create calibration dataset
    calibration_dataset = nncf.Dataset(data_loader.get_data_generator(subset_size))

    preset_mapping = {
        "performance": nncf.QuantizationPreset.PERFORMANCE,
        "mixed": nncf.QuantizationPreset.MIXED,
    }
    device_mapping = {
        "npu": nncf.TargetDevice.NPU,
        "cpu": nncf.TargetDevice.CPU,
        "gpu": nncf.TargetDevice.GPU,
        "any": nncf.TargetDevice.ANY,
    }

    nncf_device = device_mapping.get(target_device.lower(), nncf.TargetDevice.ANY)

    # NPU requires symmetric weights (PERFORMANCE preset) and cannot fuse
    # FakeQuantize boundaries around attention/DFL ops — keep those in FP16.
    is_npu = target_device.lower() == "npu"
    if is_npu and preset != "performance":
        if logs:
            logs.add_message(
                f"NPU target: overriding preset '{preset}' → 'performance' "
                "(symmetric weights required for VPUX full fusion)"
            )
        preset = "performance"

    nncf_preset = preset_mapping.get(preset, nncf.QuantizationPreset.MIXED)

    ignored_scope = (
        nncf.IgnoredScope(
            patterns=[r".*/attn/.*", r".*/dfl/.*"],
            types=["MatMul"],
            validate=False,
        )
        if is_npu
        else None
    )

    if logs:
        logs.add_message("Running NNCF quantization (this may take several minutes)")

    # Quantize model
    try:
        quantized_model = nncf.quantize(
            model,
            calibration_dataset,
            preset=nncf_preset,
            target_device=nncf_device,
            ignored_scope=ignored_scope,
            fast_bias_correction=True,
            model_type=nncf.ModelType.TRANSFORMER if "bert" in str(model.get_friendly_name()).lower() else None,
        )

        if logs:
            logs.add_message("INT8 quantization completed successfully", {"Preset used": preset})

        return quantized_model

    except Exception as e:
        if logs:
            logs.add_message("Quantization failed", {"Error": str(e)})
        raise


def is_nncf_available() -> bool:
    """Check if NNCF is available"""
    return NNCF_AVAILABLE
