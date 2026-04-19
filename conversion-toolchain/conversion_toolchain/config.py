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
            },
        },
        "advanced": {"input_shape": None, "input_names": None, "output_names": None, "batch_size": 1},
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

        # Validate subset size
        subset_size = self.get_quantization_subset_size()
        if not isinstance(subset_size, int) or subset_size <= 0:
            raise ValueError(f"Invalid subset_size: {subset_size}. Must be a positive integer")

        # Validate batch size
        batch_size = self.get_batch_size()
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(f"Invalid batch_size: {batch_size}. Must be a positive integer")

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

    def get_batch_size(self) -> int:
        """Get batch size (default 1)"""
        return self.config["advanced"]["batch_size"]

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
