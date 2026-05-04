"""
Phase 1 Unit Tests: Conversion Toolchain with Real ONNX Models
Tests using production models: ResNet-18, MobileNetV2, SqueezeNet
"""

import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import openvino as ov
import pytest

from conversion_toolchain.config import OptimizationConfig
from conversion_toolchain.logger import Logs
from conversion_toolchain.quantization import is_nncf_available
from conversion_toolchain.utils import convert_to_ir, extract_input_bundle, md5_hash
from tests.models import TEST_MODELS, download_model, get_model_info


class TestRealModels:
    """Test suite using real production ONNX models"""

    @pytest.fixture(scope="class")
    def test_models_dir(self):
        """Setup: Download test models once for all tests"""
        models_dir = Path(__file__).parent / "test_models"
        models_dir.mkdir(exist_ok=True)

        # Download models if not already present
        print("\nDownloading test models...")
        for model_name in TEST_MODELS.keys():
            try:
                download_model(model_name, str(models_dir))
            except Exception as e:
                print(f"Warning: Could not download {model_name}: {e}")

        return str(models_dir)

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture(params=["resnet18", "mobilenetv2", "squeezenet"])
    def real_model(self, request, test_models_dir):
        """Parametrized fixture providing each real model"""
        model_name = request.param
        model_info = get_model_info(model_name)
        model_path = Path(test_models_dir) / model_info["filename"]

        if not model_path.exists():
            pytest.skip(f"Model {model_name} not available")

        return {"name": model_name, "path": str(model_path), "info": model_info}

    def test_real_model_conversion_creates_zip(self, real_model, temp_dir):
        """Test that real models convert and create zip files"""
        logs = Logs()
        output_dir = Path(temp_dir) / "output"

        zip_path = convert_to_ir(real_model["path"], str(output_dir), logs)

        # Assert zip file exists
        assert Path(zip_path).exists(), f"Zip file not found: {zip_path}"
        assert zip_path.endswith(".zip"), "Output file should have .zip extension"
        print(f"✓ {real_model['name']}: Zip created successfully")

    def test_real_model_zip_contents(self, real_model, temp_dir):
        """Test that real model zip contains .xml and .bin files"""
        logs = Logs()
        output_dir = Path(temp_dir) / "output"

        zip_path = convert_to_ir(real_model["path"], str(output_dir), logs)

        # Check zip contents
        with zipfile.ZipFile(zip_path, "r") as zipf:
            namelist = zipf.namelist()

        # Should contain exactly 2 files
        assert len(namelist) == 2, f"Zip should contain 2 files, found {len(namelist)}"

        # Should have one .xml and one .bin
        xml_files = [f for f in namelist if f.endswith(".xml")]
        bin_files = [f for f in namelist if f.endswith(".bin")]

        assert len(xml_files) == 1, "Zip should contain exactly 1 .xml file"
        assert len(bin_files) == 1, "Zip should contain exactly 1 .bin file"
        print(f"✓ {real_model['name']}: Zip contains correct files")

    def test_real_model_openvino_loading(self, real_model, temp_dir):
        """Test that converted real models load in OpenVINO"""
        logs = Logs()
        output_dir = Path(temp_dir) / "output"

        zip_path = convert_to_ir(real_model["path"], str(output_dir), logs)

        # Extract zip
        extract_dir = Path(temp_dir) / "extracted"
        extract_dir.mkdir()

        with zipfile.ZipFile(zip_path, "r") as zipf:
            zipf.extractall(extract_dir)

        # Find XML file
        xml_files = list(extract_dir.glob("*.xml"))
        assert len(xml_files) == 1, "Should have exactly one .xml file"

        xml_path = xml_files[0]

        # Load with OpenVINO
        core = ov.Core()
        model = core.read_model(str(xml_path))

        # Basic validation
        assert model is not None, "Failed to load model"
        assert len(model.inputs) > 0, "Model should have at least one input"
        assert len(model.outputs) > 0, "Model should have at least one output"

        print(f"✓ {real_model['name']}: Model loaded in OpenVINO")
        print(f"  Inputs: {len(model.inputs)}, Outputs: {len(model.outputs)}")

    def test_real_model_inference(self, real_model, temp_dir):
        """Test that converted real models can run inference"""
        logs = Logs()
        output_dir = Path(temp_dir) / "output"

        zip_path = convert_to_ir(real_model["path"], str(output_dir), logs)

        # Extract zip
        extract_dir = Path(temp_dir) / "extracted"
        extract_dir.mkdir()

        with zipfile.ZipFile(zip_path, "r") as zipf:
            zipf.extractall(extract_dir)

        xml_path = list(extract_dir.glob("*.xml"))[0]

        # Load and compile model
        core = ov.Core()
        model = core.read_model(str(xml_path))
        compiled_model = core.compile_model(model, "CPU")

        # Get input shape from model info
        input_shape = real_model["info"]["input_shape"]

        # Create dummy input
        input_data = np.random.randn(*input_shape).astype(np.float32)

        # Run inference
        infer_request = compiled_model.create_infer_request()
        infer_request.set_input_tensor(ov.Tensor(input_data))
        infer_request.infer()

        # Get output
        output = infer_request.get_output_tensor().data

        # Verify output exists and has correct type
        assert output is not None, "Inference should produce output"
        assert output.dtype == np.float32, "Output should be float32"
        assert output.size > 0, "Output should not be empty"

        print(f"✓ {real_model['name']}: Inference successful")
        print(f"  Input shape: {input_shape}, Output shape: {output.shape}")

    def test_fp16_compression(self, real_model, temp_dir):
        """Test FP16 compression on real models"""
        logs = Logs()
        output_dir = Path(temp_dir) / "output"

        # Create config with FP16 enabled
        config = OptimizationConfig({"optimization": {"fp16_compression": True}})

        zip_path = convert_to_ir(real_model["path"], str(output_dir), logs, config)

        # Check that conversion succeeded
        assert Path(zip_path).exists()

        # Verify logs mention FP16 (check entire log structure, not just messages)
        logs_str = str(logs).lower()
        assert "fp16" in logs_str, "Logs should mention FP16 compression"

        print(f"✓ {real_model['name']}: FP16 compression applied")

    def test_fp32_no_compression(self, real_model, temp_dir):
        """Test FP32 (no compression) on real models"""
        logs = Logs()
        output_dir = Path(temp_dir) / "output"

        # Create config with FP16 disabled
        config = OptimizationConfig({"optimization": {"fp16_compression": False}})

        zip_path = convert_to_ir(real_model["path"], str(output_dir), logs, config)

        # Check that conversion succeeded
        assert Path(zip_path).exists()

        print(f"✓ {real_model['name']}: FP32 (no compression) conversion successful")


class TestBundleWorkflow:
    """Test the complete bundle workflow with real models"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def resnet18_model(self):
        """Get ResNet-18 model path"""
        models_dir = Path(__file__).parent / "test_models"
        models_dir.mkdir(exist_ok=True)

        model_path = download_model("resnet18", str(models_dir))
        if not Path(model_path).exists():
            pytest.skip("ResNet-18 not available")

        return model_path

    def test_bundle_creation_and_conversion(self, resnet18_model, temp_dir):
        """Test creating a bundle and converting it"""
        # Create bundle
        bundle_path = Path(temp_dir) / "test_bundle.zip"

        with zipfile.ZipFile(bundle_path, "w") as zipf:
            zipf.write(resnet18_model, arcname="model.onnx")

        assert bundle_path.exists(), "Bundle should be created"

        # Extract and convert
        output_dir = Path(temp_dir) / "output"
        logs = Logs()

        with tempfile.TemporaryDirectory() as extract_dir:
            onnx_path, config_path, calibration_dir = extract_input_bundle(str(bundle_path), extract_dir, logs)

            assert onnx_path is not None, "ONNX model should be extracted"
            assert Path(onnx_path).exists(), "Extracted ONNX should exist"

            # Convert
            zip_path = convert_to_ir(onnx_path, str(output_dir), logs)
            assert Path(zip_path).exists(), "Conversion should produce zip"

        print("✓ Bundle workflow test passed")

    def test_bundle_with_config(self, resnet18_model, temp_dir):
        """Test bundle with custom config"""
        # Create bundle with config
        bundle_path = Path(temp_dir) / "test_bundle.zip"
        config_data = {"optimization": {"fp16_compression": False}}

        config_file = Path(temp_dir) / "config.json"
        with open(config_file, "w") as f:
            json.dump(config_data, f)

        with zipfile.ZipFile(bundle_path, "w") as zipf:
            zipf.write(resnet18_model, arcname="model.onnx")
            zipf.write(config_file, arcname="config.json")

        # Extract and verify config
        logs = Logs()
        with tempfile.TemporaryDirectory() as extract_dir:
            onnx_path, config_path, calibration_dir = extract_input_bundle(str(bundle_path), extract_dir, logs)

            assert config_path is not None, "Config should be found"

            config = OptimizationConfig.from_file(config_path)
            assert config.get_fp16_compression() is False, "Config should disable FP16"

        print("✓ Bundle with config test passed")


class TestUtilities:
    """Test utility functions"""

    def test_md5_hash_consistency(self):
        """Test MD5 hash function"""
        import tempfile

        # Create temp file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            hash1 = md5_hash(temp_path)
            hash2 = md5_hash(temp_path)

            assert hash1 == hash2, "Hash should be consistent"
            assert len(hash1) == 32, "MD5 hash should be 32 characters"
        finally:
            Path(temp_path).unlink()

    def test_logger_functionality(self):
        """Test logger"""
        logs = Logs()
        logs.add_message("Test message", {"key": "value"})

        assert len(logs.messages) == 1
        assert logs.messages[0].message == "Test message"
        assert logs.messages[0].data == {"key": "value"}

    def test_config_defaults(self):
        """Test configuration defaults"""
        config = OptimizationConfig.from_default()

        assert config.get_fp16_compression() is True, "FP16 should be enabled by default"
        assert config.is_quantization_enabled() is False, "Quantization should be disabled by default"


class TestErrorHandling:
    """Test error handling for invalid inputs and failures"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    def test_nonexistent_input_file(self, temp_dir):
        """Test error handling when input file doesn't exist"""
        logs = Logs()
        nonexistent_file = "/tmp/nonexistent_file_12345.zip"

        with pytest.raises(FileNotFoundError) as exc_info:
            extract_input_bundle(nonexistent_file, temp_dir, logs)

        assert "does not exist" in str(exc_info.value)
        # Check that error was logged
        assert any("Error" in msg.message for msg in logs.messages)

    def test_empty_zip_file(self, temp_dir):
        """Test error handling for empty zip file"""
        logs = Logs()
        empty_zip = Path(temp_dir) / "empty.zip"
        empty_zip.touch()  # Create empty file

        with pytest.raises(ValueError) as exc_info:
            extract_input_bundle(str(empty_zip), temp_dir, logs)

        assert "empty" in str(exc_info.value).lower()

    def test_invalid_zip_file(self, temp_dir):
        """Test error handling for invalid (corrupted) zip file"""
        logs = Logs()
        invalid_zip = Path(temp_dir) / "invalid.zip"

        # Create a file that's not a valid zip
        with open(invalid_zip, "w") as f:
            f.write("This is not a valid zip file content")

        with pytest.raises(zipfile.BadZipFile):
            extract_input_bundle(str(invalid_zip), temp_dir, logs)

    def test_zip_without_onnx_model(self, temp_dir):
        """Test error handling when zip doesn't contain ONNX model"""
        logs = Logs()
        bundle_path = Path(temp_dir) / "no_model.zip"

        # Create zip with only a text file
        with zipfile.ZipFile(bundle_path, "w") as zipf:
            zipf.writestr("readme.txt", "No model here")

        with pytest.raises(ValueError) as exc_info:
            extract_input_bundle(str(bundle_path), temp_dir, logs)

        assert "No ONNX model" in str(exc_info.value)
        # Check that error details were logged
        log_messages = [msg.message for msg in logs.messages]
        assert any("Model validation failed" in msg for msg in log_messages)

    def test_directory_instead_of_file(self, temp_dir):
        """Test error handling when input is a directory not a file"""
        logs = Logs()
        directory_path = Path(temp_dir) / "not_a_file"
        directory_path.mkdir()

        with pytest.raises(ValueError) as exc_info:
            extract_input_bundle(str(directory_path), temp_dir, logs)

        assert "not a file" in str(exc_info.value)

    def test_zip_with_empty_onnx_file(self, temp_dir):
        """Test error handling when ONNX file in zip is empty"""
        logs = Logs()
        bundle_path = Path(temp_dir) / "empty_model.zip"

        # Create zip with empty ONNX file
        with zipfile.ZipFile(bundle_path, "w") as zipf:
            zipf.writestr("model.onnx", "")

        with pytest.raises(ValueError) as exc_info:
            extract_input_bundle(str(bundle_path), temp_dir, logs)

        assert "empty" in str(exc_info.value).lower()

    def test_corrupted_onnx_model(self, temp_dir):
        """Test error handling for corrupted/invalid ONNX model"""
        logs = Logs()
        bundle_path = Path(temp_dir) / "corrupted_model.zip"

        # Create zip with invalid ONNX file
        with zipfile.ZipFile(bundle_path, "w") as zipf:
            zipf.writestr("model.onnx", "This is not a valid ONNX model")

        # Extract should succeed
        with tempfile.TemporaryDirectory() as extract_dir:
            onnx_path, _, _ = extract_input_bundle(str(bundle_path), extract_dir, logs)

            # But conversion should fail
            output_dir = Path(temp_dir) / "output"
            output_dir.mkdir()

            with pytest.raises(RuntimeError) as exc_info:
                convert_to_ir(onnx_path, str(output_dir), logs)

            assert "Failed to convert" in str(exc_info.value)
            # Check that error was logged
            log_messages = [msg.message for msg in logs.messages]
            assert any("Error" in msg for msg in log_messages)

    def test_nonexistent_onnx_path_in_convert(self, temp_dir):
        """Test convert_to_ir with nonexistent ONNX path"""
        logs = Logs()
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir()

        with pytest.raises(FileNotFoundError) as exc_info:
            convert_to_ir("/tmp/nonexistent_model.onnx", str(output_dir), logs)

        assert "does not exist" in str(exc_info.value)

    def test_multiple_onnx_files_warning(self, temp_dir):
        """Test that multiple ONNX files generate a warning but still work"""
        logs = Logs()
        bundle_path = Path(temp_dir) / "multi_model.zip"

        # Download a real model for testing
        models_dir = Path(__file__).parent / "test_models"
        models_dir.mkdir(exist_ok=True)
        real_model = download_model("squeezenet", str(models_dir))

        if not Path(real_model).exists():
            pytest.skip("SqueezeNet not available")

        # Create zip with multiple ONNX files
        with zipfile.ZipFile(bundle_path, "w") as zipf:
            zipf.write(real_model, arcname="model1.onnx")
            zipf.write(real_model, arcname="model2.onnx")

        with tempfile.TemporaryDirectory() as extract_dir:
            onnx_path, _, _ = extract_input_bundle(str(bundle_path), extract_dir, logs)

            # Should succeed but log a warning
            log_messages = [msg.message for msg in logs.messages]
            assert any("Multiple ONNX files" in msg or "Warning" in msg for msg in log_messages)
            assert onnx_path is not None


class TestBatchSize:
    """Tests for the batch_size conversion parameter"""

    BATCH_SIZE = 4

    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def mobilenet_model(self):
        models_dir = Path(__file__).parent / "test_models"
        models_dir.mkdir(exist_ok=True)
        model_path = download_model("mobilenetv2", str(models_dir))
        if not Path(model_path).exists():
            pytest.skip("MobileNetV2 not available")
        return model_path

    @pytest.fixture
    def yolov8n_model(self):
        models_dir = Path(__file__).parent / "test_models"
        model_path = models_dir / "yolov8n.onnx"
        if not model_path.exists():
            pytest.skip("yolov8n.onnx not available")
        return str(model_path)

    def test_config_batch_size_default(self):
        """Default batch_size is 1"""
        config = OptimizationConfig.from_default()
        assert config.get_batch_size() == 1

    def test_config_batch_size_custom(self):
        """batch_size is read from config dict"""
        config = OptimizationConfig({"advanced": {"batch_size": self.BATCH_SIZE}})
        assert config.get_batch_size() == self.BATCH_SIZE

    def test_config_batch_size_invalid(self):
        """Zero and negative batch_size raise ValueError"""
        for bad in (0, -1, -4):
            with pytest.raises(ValueError, match="batch_size"):
                OptimizationConfig({"advanced": {"batch_size": bad}})

    def test_batch4_input_shape(self, mobilenet_model, temp_dir):
        """Converted model has input batch dim == 4"""
        config = OptimizationConfig({"advanced": {"batch_size": self.BATCH_SIZE}})
        logs = Logs()
        zip_path = convert_to_ir(mobilenet_model, temp_dir, logs, config)

        extract_dir = Path(temp_dir) / "ir"
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)

        model = ov.Core().read_model(str(next(extract_dir.glob("*.xml"))))
        assert (
            model.inputs[0].shape[0] == self.BATCH_SIZE
        ), f"Expected batch dim {self.BATCH_SIZE}, got {model.inputs[0].shape[0]}"

    def test_batch4_output_shape(self, mobilenet_model, temp_dir):
        """Converted model output also has batch dim == 4"""
        config = OptimizationConfig({"advanced": {"batch_size": self.BATCH_SIZE}})
        logs = Logs()
        zip_path = convert_to_ir(mobilenet_model, temp_dir, logs, config)

        extract_dir = Path(temp_dir) / "ir"
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)

        model = ov.Core().read_model(str(next(extract_dir.glob("*.xml"))))
        assert model.outputs[0].shape[0] == self.BATCH_SIZE

    def test_batch4_inference(self, mobilenet_model, temp_dir):
        """Model converted with batch=4 runs inference on a batch of 4 images"""
        config = OptimizationConfig({"advanced": {"batch_size": self.BATCH_SIZE}})
        logs = Logs()
        zip_path = convert_to_ir(mobilenet_model, temp_dir, logs, config)

        extract_dir = Path(temp_dir) / "ir"
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)

        core = ov.Core()
        model = core.read_model(str(next(extract_dir.glob("*.xml"))))
        compiled = core.compile_model(model, "CPU")

        input_data = np.random.randn(self.BATCH_SIZE, 3, 224, 224).astype(np.float32)
        result = compiled(input_data)[compiled.output(0)]

        assert result.shape[0] == self.BATCH_SIZE
        assert result.shape[1] == 1000  # ImageNet classes

    def test_batch4_bundle_with_config(self, mobilenet_model, temp_dir):
        """batch_size specified via config.json inside the zip bundle"""
        bundle_path = Path(temp_dir) / "bundle.zip"
        with zipfile.ZipFile(bundle_path, "w") as z:
            z.write(mobilenet_model, arcname="model.onnx")
            z.writestr("config.json", json.dumps({"advanced": {"batch_size": self.BATCH_SIZE}}))

        logs = Logs()
        with tempfile.TemporaryDirectory() as extract_dir:
            onnx_path, config_path, _ = extract_input_bundle(str(bundle_path), extract_dir, logs)
            config = OptimizationConfig.from_file(config_path)
            assert config.get_batch_size() == self.BATCH_SIZE

            zip_path = convert_to_ir(onnx_path, temp_dir, logs, config)

        extract_dir2 = Path(temp_dir) / "ir"
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir2)

        model = ov.Core().read_model(str(next(extract_dir2.glob("*.xml"))))
        assert model.inputs[0].shape[0] == self.BATCH_SIZE

    def test_hardcoded_batch_error_message(self, yolov8n_model, temp_dir):
        """YOLO models with hardcoded batch=1 raise a clear error"""
        config = OptimizationConfig({"advanced": {"batch_size": self.BATCH_SIZE}})
        logs = Logs()
        with pytest.raises(RuntimeError, match="batch_size=4"):
            convert_to_ir(yolov8n_model, temp_dir, logs, config)


class TestPreprocessingConfig:
    """Unit tests for the preprocessing section of OptimizationConfig."""

    def test_defaults_are_none(self):
        config = OptimizationConfig.from_default()
        assert config.get_preprocessing_input_dtype() is None
        assert config.get_mean_values() is None
        assert config.get_scale_values() is None

    def test_has_preprocessing_false_by_default(self):
        assert OptimizationConfig.from_default().has_preprocessing() is False

    def test_has_preprocessing_true_when_any_field_set(self):
        assert OptimizationConfig({"preprocessing": {"input_dtype": "u8"}}).has_preprocessing()
        assert OptimizationConfig({"preprocessing": {"mean_values": [0.0, 0.0, 0.0]}}).has_preprocessing()
        assert OptimizationConfig({"preprocessing": {"scale_values": [255.0, 255.0, 255.0]}}).has_preprocessing()

    def test_valid_input_dtypes(self):
        for dtype in ("u8", "f16", "f32"):
            cfg = OptimizationConfig({"preprocessing": {"input_dtype": dtype}})
            assert cfg.get_preprocessing_input_dtype() == dtype

    def test_invalid_input_dtype_raises(self):
        with pytest.raises(ValueError, match="input_dtype"):
            OptimizationConfig({"preprocessing": {"input_dtype": "int32"}})

    def test_valid_mean_and_scale(self):
        cfg = OptimizationConfig(
            {"preprocessing": {"mean_values": [123.675, 116.28, 103.53], "scale_values": [255.0, 255.0, 255.0]}}
        )
        assert cfg.get_mean_values() == [123.675, 116.28, 103.53]
        assert cfg.get_scale_values() == [255.0, 255.0, 255.0]

    def test_invalid_mean_not_a_list_raises(self):
        with pytest.raises(ValueError, match="mean_values"):
            OptimizationConfig({"preprocessing": {"mean_values": 123.0}})

    def test_invalid_scale_non_numeric_raises(self):
        with pytest.raises(ValueError, match="scale_values"):
            OptimizationConfig({"preprocessing": {"scale_values": ["a", "b", "c"]}})

    def test_config_json_roundtrip(self, tmp_path):
        data = {"preprocessing": {"input_dtype": "u8", "scale_values": [255.0, 255.0, 255.0]}}
        cfg_file = tmp_path / "config.json"
        cfg_file.write_text(json.dumps(data))
        cfg = OptimizationConfig.from_file(str(cfg_file))
        assert cfg.get_preprocessing_input_dtype() == "u8"
        assert cfg.get_scale_values() == [255.0, 255.0, 255.0]


class TestPreprocessingConversion:
    """Functional tests verifying that preprocessing is correctly baked into the IR."""

    @pytest.fixture
    def temp_dir(self):
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    @pytest.fixture
    def mobilenet_path(self):
        models_dir = Path(__file__).parent / "test_models"
        models_dir.mkdir(exist_ok=True)
        model_path = download_model("mobilenetv2", str(models_dir))
        if not Path(model_path).exists():
            pytest.skip("MobileNetV2 not available")
        return model_path

    def _load_compiled(self, zip_path: str) -> ov.CompiledModel:
        extract_dir = Path(zip_path).parent / "ir"
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)
        model = ov.Core().read_model(str(next(extract_dir.glob("*.xml"))))
        return ov.Core().compile_model(model, "CPU"), model

    def test_u8_input_dtype_baked_into_ir(self, mobilenet_path, temp_dir):
        """After conversion with input_dtype=u8, the IR input boundary is u8."""
        config = OptimizationConfig({"preprocessing": {"input_dtype": "u8"}})
        logs = Logs()
        zip_path = convert_to_ir(mobilenet_path, temp_dir, logs, config)

        extract_dir = Path(temp_dir) / "ir_u8"
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)

        model = ov.Core().read_model(str(next(extract_dir.glob("*.xml"))))
        assert model.inputs[0].get_element_type() == ov.Type.u8, "Input type should be u8"

    def test_no_preprocessing_input_type_unchanged(self, mobilenet_path, temp_dir):
        """Without preprocessing config, the IR keeps its original f32 input type."""
        logs = Logs()
        zip_path = convert_to_ir(mobilenet_path, temp_dir, logs)

        extract_dir = Path(temp_dir) / "ir_f32"
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(extract_dir)

        model = ov.Core().read_model(str(next(extract_dir.glob("*.xml"))))
        assert model.inputs[0].get_element_type() == ov.Type.f32

    def test_scale_normalization_matches_manual(self, mobilenet_path, temp_dir):
        """Model with baked-in ÷255 and u8 input produces the same output as the
        baseline model fed manually-normalized f32 inputs."""
        np.random.seed(42)
        pixels_u8 = np.random.randint(0, 256, (1, 3, 224, 224), dtype=np.uint8)
        pixels_f32 = pixels_u8.astype(np.float32) / 255.0

        # Disable FP16 compression so precision differences don't mask PPP correctness
        no_fp16 = {"optimization": {"fp16_compression": False}}
        logs = Logs()

        baseline_zip = convert_to_ir(mobilenet_path, Path(temp_dir) / "baseline", logs, OptimizationConfig(no_fp16))
        extract_dir = Path(temp_dir) / "baseline_ir"
        with zipfile.ZipFile(baseline_zip) as z:
            z.extractall(extract_dir)
        baseline_model = ov.Core().compile_model(ov.Core().read_model(str(next(extract_dir.glob("*.xml")))), "CPU")
        ref_output = baseline_model(pixels_f32)[baseline_model.output(0)]

        config = OptimizationConfig(
            {**no_fp16, "preprocessing": {"input_dtype": "u8", "scale_values": [255.0, 255.0, 255.0]}}
        )
        pp_zip = convert_to_ir(mobilenet_path, Path(temp_dir) / "pp", logs, config)
        extract_dir2 = Path(temp_dir) / "pp_ir"
        with zipfile.ZipFile(pp_zip) as z:
            z.extractall(extract_dir2)
        pp_model = ov.Core().compile_model(ov.Core().read_model(str(next(extract_dir2.glob("*.xml")))), "CPU")
        pp_output = pp_model(pixels_u8)[pp_model.output(0)]

        np.testing.assert_allclose(
            ref_output, pp_output, rtol=1e-4, atol=1e-4, err_msg="Baked-in scale normalisation should match manual ÷255"
        )

    def test_mean_and_scale_normalization_matches_manual(self, mobilenet_path, temp_dir):
        """Model with baked-in mean + scale produces the same output as the baseline
        model fed inputs that were manually mean-subtracted and scaled."""
        np.random.seed(7)
        MEAN = [123.675, 116.28, 103.53]
        SCALE = [58.395, 57.12, 57.375]

        pixels_u8 = np.random.randint(0, 256, (1, 3, 224, 224), dtype=np.uint8)
        pixels_f32 = pixels_u8.astype(np.float32)
        for c in range(3):
            pixels_f32[0, c] = (pixels_f32[0, c] - MEAN[c]) / SCALE[c]

        no_fp16 = {"optimization": {"fp16_compression": False}}
        logs = Logs()

        baseline_zip = convert_to_ir(mobilenet_path, Path(temp_dir) / "baseline_ms", logs, OptimizationConfig(no_fp16))
        extract_dir = Path(temp_dir) / "baseline_ms_ir"
        with zipfile.ZipFile(baseline_zip) as z:
            z.extractall(extract_dir)
        baseline_model = ov.Core().compile_model(ov.Core().read_model(str(next(extract_dir.glob("*.xml")))), "CPU")
        ref_output = baseline_model(pixels_f32)[baseline_model.output(0)]

        config = OptimizationConfig(
            {**no_fp16, "preprocessing": {"input_dtype": "u8", "mean_values": MEAN, "scale_values": SCALE}}
        )
        pp_zip = convert_to_ir(mobilenet_path, Path(temp_dir) / "pp_ms", logs, config)
        extract_dir2 = Path(temp_dir) / "pp_ms_ir"
        with zipfile.ZipFile(pp_zip) as z:
            z.extractall(extract_dir2)
        pp_model = ov.Core().compile_model(ov.Core().read_model(str(next(extract_dir2.glob("*.xml")))), "CPU")
        pp_output = pp_model(pixels_u8)[pp_model.output(0)]

        np.testing.assert_allclose(
            ref_output, pp_output, rtol=1e-4, atol=1e-4, err_msg="Baked-in mean+scale should match manual normalization"
        )

    def test_log_records_preprocessing(self, mobilenet_path, temp_dir):
        """Conversion logs should mention the baked-in preprocessing."""
        config = OptimizationConfig({"preprocessing": {"input_dtype": "u8", "scale_values": [255.0, 255.0, 255.0]}})
        logs = Logs()
        convert_to_ir(mobilenet_path, temp_dir, logs, config)
        logs_str = str(logs).lower()
        assert "preprocessing" in logs_str


class TestInvalidConfigurations:
    """Test error handling for invalid configurations"""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs"""
        temp = tempfile.mkdtemp()
        yield temp
        shutil.rmtree(temp)

    def test_quantization_without_calibration_data(self, temp_dir):
        """Test that quantization fails without calibration data"""
        # Skip if NNCF is not available
        if not is_nncf_available():
            pytest.skip("NNCF not available, quantization will be skipped")

        logs = Logs()

        # Download a real model
        models_dir = Path(__file__).parent / "test_models"
        models_dir.mkdir(exist_ok=True)
        real_model = download_model("squeezenet", str(models_dir))

        if not Path(real_model).exists():
            pytest.skip("SqueezeNet not available")

        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir()

        # Create config with quantization enabled but no calibration data
        config = OptimizationConfig({"optimization": {"quantization": {"enabled": True}}})

        with pytest.raises(ValueError) as exc_info:
            convert_to_ir(real_model, str(output_dir), logs, config, calibration_dir=None)

        assert "calibration data" in str(exc_info.value).lower()

    def test_invalid_json_config(self, temp_dir):
        """Test error handling for malformed JSON config"""
        bundle_path = Path(temp_dir) / "bad_config.zip"

        # Download a real model
        models_dir = Path(__file__).parent / "test_models"
        models_dir.mkdir(exist_ok=True)
        real_model = download_model("squeezenet", str(models_dir))

        if not Path(real_model).exists():
            pytest.skip("SqueezeNet not available")

        # Create zip with invalid JSON config
        with zipfile.ZipFile(bundle_path, "w") as zipf:
            zipf.write(real_model, arcname="model.onnx")
            zipf.writestr("config.json", "{ invalid json }")

        logs = Logs()

        with tempfile.TemporaryDirectory() as extract_dir:
            # Should still extract
            onnx_path, config_path, _ = extract_input_bundle(str(bundle_path), extract_dir, logs)

            # But loading config should fail
            with pytest.raises(json.JSONDecodeError):
                OptimizationConfig.from_file(config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
