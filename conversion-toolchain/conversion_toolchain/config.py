"""
Configuration parser for model optimization settings
"""

import json
from typing import Any


class OptimizationConfig:
    """Configuration for model optimization"""

    DEFAULT_CONFIG = {
        "model_file": "model.onnx",
        "optimization": {
            "fp16_compression": True,
            "quantization": {
                "enabled": False,
                "mode": "int8",
                "calibration_data": "calibration/",
                "preset": "mixed",
                "subset_size": 300,
                "target_device": "any",
            },
        },
        "preprocessing": {
            "input_dtype": None,  # "u8", "f16", "f32", or null (no change)
            "mean_values": None,  # per-channel means to subtract, e.g. [123.675, 116.28, 103.53]
            "scale_values": None,  # per-channel values to divide by, e.g. [255.0, 255.0, 255.0]
        },
        "advanced": {
            "input_shape": None,
            "input_names": None,
            "output_names": None,
            "batch_size": 1,
            "dynamic_batch": False,
        },
    }

    def __init__(self, config_dict: dict[str, Any] | None = None):
        """
        Initialize configuration from dictionary

        Args:
            config_dict: Configuration dictionary. If None, uses defaults.
        """
        if config_dict is None:
            config_dict = {}

        # Merge with defaults
        self.config = self._merge_configs(self.DEFAULT_CONFIG, config_dict)

        # Validate configuration
        self._validate()

    @classmethod
    def from_file(cls, config_path: str) -> "OptimizationConfig":
        """
        Load configuration from JSON file

        Args:
            config_path: Path to config.json file

        Returns:
            OptimizationConfig instance
        """
        with open(config_path) as f:
            config_dict = json.load(f)
        return cls(config_dict)

    @classmethod
    def from_default(cls) -> "OptimizationConfig":
        """Create configuration with all defaults"""
        return cls(None)

    def _merge_configs(self, default: dict, user: dict) -> dict:
        """Recursively merge user config with defaults"""
        result = default.copy()

        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _validate(self):
        """Validate configuration values"""
        # Validate quantization mode
        quant_mode = self.get_quantization_mode()
        if quant_mode not in ["int8", "int4"]:
            raise ValueError(f"Invalid quantization mode: {quant_mode}. Must be 'int8' or 'int4'")

        # Validate quantization preset
        preset = self.get_quantization_preset()
        if preset not in ["performance", "mixed", "accuracy"]:
            raise ValueError(f"Invalid quantization preset: {preset}. Must be 'performance', 'mixed', or 'accuracy'")

        # Validate quantization target device
        target_device = self.get_quantization_target_device()
        if target_device not in ["any", "cpu", "gpu", "npu"]:
            raise ValueError(
                f"Invalid quantization target_device: '{target_device}'. Must be 'any', 'cpu', 'gpu', or 'npu'"
            )

        # Validate subset size
        subset_size = self.get_quantization_subset_size()
        if not isinstance(subset_size, int) or subset_size <= 0:
            raise ValueError(f"Invalid subset_size: {subset_size}. Must be a positive integer")

        # Validate batch size
        batch_size = self.get_batch_size()
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {batch_size}. Must be a positive integer")

        # dynamic_batch and batch_size > 1 are mutually exclusive
        if self.config["advanced"].get("dynamic_batch") and batch_size > 1:
            raise ValueError("dynamic_batch=true and batch_size > 1 are mutually exclusive")

        # Validate preprocessing
        input_dtype = self.get_preprocessing_input_dtype()
        if input_dtype is not None and input_dtype not in ("u8", "f16", "f32"):
            raise ValueError(f"Invalid preprocessing.input_dtype: '{input_dtype}'. Must be 'u8', 'f16', or 'f32'")
        for field in ("mean_values", "scale_values"):
            val = self.config["preprocessing"][field]
            if val is not None and (not isinstance(val, list) or not all(isinstance(v, int | float) for v in val)):
                raise ValueError(f"Invalid preprocessing.{field}: must be a list of numbers or null")

    # Getters for configuration values

    def get_model_file(self) -> str:
        """Get the model filename"""
        return self.config["model_file"]

    def get_fp16_compression(self) -> bool:
        """Get FP16 compression setting"""
        return self.config["optimization"]["fp16_compression"]

    def is_quantization_enabled(self) -> bool:
        """Check if quantization is enabled"""
        return self.config["optimization"]["quantization"]["enabled"]

    def get_quantization_mode(self) -> str:
        """Get quantization mode ('int8' or 'int4')"""
        return self.config["optimization"]["quantization"]["mode"]

    def get_calibration_data_path(self) -> str:
        """Get calibration data path"""
        return self.config["optimization"]["quantization"]["calibration_data"]

    def get_quantization_preset(self) -> str:
        """Get quantization preset"""
        return self.config["optimization"]["quantization"]["preset"]

    def get_quantization_subset_size(self) -> int:
        """Get calibration subset size"""
        return self.config["optimization"]["quantization"]["subset_size"]

    def get_quantization_target_device(self) -> str:
        """Get quantization target device ('any', 'cpu', 'gpu', or 'npu')"""
        return self.config["optimization"]["quantization"].get("target_device", "any")

    def get_preprocessing_input_dtype(self) -> str | None:
        """Get preprocessing input dtype override ('u8', 'f16', 'f32', or None)"""
        return self.config["preprocessing"]["input_dtype"]

    def get_mean_values(self) -> list | None:
        """Get per-channel mean values to subtract (or None)"""
        return self.config["preprocessing"]["mean_values"]

    def get_scale_values(self) -> list | None:
        """Get per-channel scale values to divide by (or None)"""
        return self.config["preprocessing"]["scale_values"]

    def has_preprocessing(self) -> bool:
        """Return True if any preprocessing step is configured"""
        return any(self.config["preprocessing"][k] is not None for k in ("input_dtype", "mean_values", "scale_values"))

    def with_u8_preprocessing(self, scale_values: list) -> "OptimizationConfig":
        """Return a copy with u8 input dtype and the given scale_values set."""
        import copy

        new = copy.deepcopy(self)
        new.config["preprocessing"]["input_dtype"] = "u8"
        new.config["preprocessing"]["scale_values"] = scale_values
        return new

    def with_input_dtype(self, dtype: str) -> "OptimizationConfig":
        """Return a copy with just the input_dtype set (no scale/mean changes)."""
        import copy

        new = copy.deepcopy(self)
        new.config["preprocessing"]["input_dtype"] = dtype
        return new

    def get_batch_size(self) -> int:
        """Get batch size (default 1)"""
        return self.config["advanced"]["batch_size"]

    def is_dynamic_batch(self) -> bool:
        """Return True if batch dimension should be kept dynamic in the IR"""
        return bool(self.config["advanced"].get("dynamic_batch", False))

    def get_input_shape(self) -> list | None:
        """Get input shape override"""
        return self.config["advanced"]["input_shape"]

    def get_input_names(self) -> list | None:
        """Get input tensor names"""
        return self.config["advanced"]["input_names"]

    def get_output_names(self) -> list | None:
        """Get output tensor names"""
        return self.config["advanced"]["output_names"]

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary"""
        return self.config.copy()

    def to_json(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, "w") as f:
            json.dump(self.config, f, indent=2)

    def __repr__(self) -> str:
        return f"OptimizationConfig({json.dumps(self.config, indent=2)})"
