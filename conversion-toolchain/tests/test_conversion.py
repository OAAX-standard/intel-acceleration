"""
Phase 1 Unit Tests: Conversion Toolchain with Real ONNX Models
Tests using production models: ResNet-18, MobileNetV2, SqueezeNet
"""
import pytest
import tempfile
import shutil
import json
import zipfile
from pathlib import Path
import numpy as np
import openvino as ov

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from conversion_toolchain.utils import convert_to_ir, md5_hash, extract_input_bundle
from conversion_toolchain.config import OptimizationConfig
from conversion_toolchain.logger import Logs
from conversion_toolchain.quantization import is_nncf_available
from tests.download_test_models import download_model, get_model_info, TEST_MODELS


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

        return {
            "name": model_name,
            "path": str(model_path),
            "info": model_info
        }

    def test_real_model_conversion_creates_zip(self, real_model, temp_dir):
        """Test that real models convert and create zip files"""
        logs = Logs()
        output_dir = Path(temp_dir) / 'output'

        zip_path = convert_to_ir(real_model["path"], str(output_dir), logs)

        # Assert zip file exists
        assert Path(zip_path).exists(), f"Zip file not found: {zip_path}"
        assert zip_path.endswith('.zip'), "Output file should have .zip extension"
        print(f"✓ {real_model['name']}: Zip created successfully")

    def test_real_model_zip_contents(self, real_model, temp_dir):
        """Test that real model zip contains .xml and .bin files"""
        logs = Logs()
        output_dir = Path(temp_dir) / 'output'

        zip_path = convert_to_ir(real_model["path"], str(output_dir), logs)

        # Check zip contents
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            namelist = zipf.namelist()

        # Should contain exactly 2 files
        assert len(namelist) == 2, f"Zip should contain 2 files, found {len(namelist)}"

        # Should have one .xml and one .bin
        xml_files = [f for f in namelist if f.endswith('.xml')]
        bin_files = [f for f in namelist if f.endswith('.bin')]

        assert len(xml_files) == 1, "Zip should contain exactly 1 .xml file"
        assert len(bin_files) == 1, "Zip should contain exactly 1 .bin file"
        print(f"✓ {real_model['name']}: Zip contains correct files")

    def test_real_model_openvino_loading(self, real_model, temp_dir):
        """Test that converted real models load in OpenVINO"""
        logs = Logs()
        output_dir = Path(temp_dir) / 'output'

        zip_path = convert_to_ir(real_model["path"], str(output_dir), logs)

        # Extract zip
        extract_dir = Path(temp_dir) / 'extracted'
        extract_dir.mkdir()

        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_dir)

        # Find XML file
        xml_files = list(extract_dir.glob('*.xml'))
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
        output_dir = Path(temp_dir) / 'output'

        zip_path = convert_to_ir(real_model["path"], str(output_dir), logs)

        # Extract zip
        extract_dir = Path(temp_dir) / 'extracted'
        extract_dir.mkdir()

        with zipfile.ZipFile(zip_path, 'r') as zipf:
            zipf.extractall(extract_dir)

        xml_path = list(extract_dir.glob('*.xml'))[0]

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
        output_dir = Path(temp_dir) / 'output'

        # Create config with FP16 enabled
        config = OptimizationConfig({"optimization": {"fp16_compression": True}})

        zip_path = convert_to_ir(real_model["path"], str(output_dir), logs, config)

        # Check that conversion succeeded
        assert Path(zip_path).exists()

        # Verify logs mention FP16 (check entire log structure, not just messages)
        logs_str = str(logs).lower()
        assert 'fp16' in logs_str, "Logs should mention FP16 compression"

        print(f"✓ {real_model['name']}: FP16 compression applied")

    def test_fp32_no_compression(self, real_model, temp_dir):
        """Test FP32 (no compression) on real models"""
        logs = Logs()
        output_dir = Path(temp_dir) / 'output'

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

        with zipfile.ZipFile(bundle_path, 'w') as zipf:
            zipf.write(resnet18_model, arcname="model.onnx")

        assert bundle_path.exists(), "Bundle should be created"

        # Extract and convert
        output_dir = Path(temp_dir) / 'output'
        logs = Logs()

        with tempfile.TemporaryDirectory() as extract_dir:
            onnx_path, config_path, calibration_dir = extract_input_bundle(
                str(bundle_path),
                extract_dir,
                logs
            )

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
        config_data = {
            "optimization": {
                "fp16_compression": False
            }
        }

        config_file = Path(temp_dir) / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f)

        with zipfile.ZipFile(bundle_path, 'w') as zipf:
            zipf.write(resnet18_model, arcname="model.onnx")
            zipf.write(config_file, arcname="config.json")

        # Extract and verify config
        logs = Logs()
        with tempfile.TemporaryDirectory() as extract_dir:
            onnx_path, config_path, calibration_dir = extract_input_bundle(
                str(bundle_path),
                extract_dir,
                logs
            )

            assert config_path is not None, "Config should be found"

            config = OptimizationConfig.from_file(config_path)
            assert config.get_fp16_compression() == False, "Config should disable FP16"

        print("✓ Bundle with config test passed")


class TestUtilities:
    """Test utility functions"""

    def test_md5_hash_consistency(self):
        """Test MD5 hash function"""
        import tempfile

        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
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
        logs.add_message('Test message', {'key': 'value'})

        assert len(logs.messages) == 1
        assert logs.messages[0].message == 'Test message'
        assert logs.messages[0].data == {'key': 'value'}

    def test_config_defaults(self):
        """Test configuration defaults"""
        config = OptimizationConfig.from_default()

        assert config.get_fp16_compression() == True, "FP16 should be enabled by default"
        assert config.is_quantization_enabled() == False, "Quantization should be disabled by default"


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
        with open(invalid_zip, 'w') as f:
            f.write("This is not a valid zip file content")

        with pytest.raises(zipfile.BadZipFile):
            extract_input_bundle(str(invalid_zip), temp_dir, logs)

    def test_zip_without_onnx_model(self, temp_dir):
        """Test error handling when zip doesn't contain ONNX model"""
        logs = Logs()
        bundle_path = Path(temp_dir) / "no_model.zip"

        # Create zip with only a text file
        with zipfile.ZipFile(bundle_path, 'w') as zipf:
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
        with zipfile.ZipFile(bundle_path, 'w') as zipf:
            zipf.writestr("model.onnx", "")

        with pytest.raises(ValueError) as exc_info:
            extract_input_bundle(str(bundle_path), temp_dir, logs)

        assert "empty" in str(exc_info.value).lower()

    def test_corrupted_onnx_model(self, temp_dir):
        """Test error handling for corrupted/invalid ONNX model"""
        logs = Logs()
        bundle_path = Path(temp_dir) / "corrupted_model.zip"

        # Create zip with invalid ONNX file
        with zipfile.ZipFile(bundle_path, 'w') as zipf:
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
        with zipfile.ZipFile(bundle_path, 'w') as zipf:
            zipf.write(real_model, arcname="model1.onnx")
            zipf.write(real_model, arcname="model2.onnx")

        with tempfile.TemporaryDirectory() as extract_dir:
            onnx_path, _, _ = extract_input_bundle(str(bundle_path), extract_dir, logs)

            # Should succeed but log a warning
            log_messages = [msg.message for msg in logs.messages]
            assert any("Multiple ONNX files" in msg or "Warning" in msg for msg in log_messages)
            assert onnx_path is not None


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
        config = OptimizationConfig({
            "optimization": {
                "quantization": {
                    "enabled": True
                }
            }
        })

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
        with zipfile.ZipFile(bundle_path, 'w') as zipf:
            zipf.write(real_model, arcname="model.onnx")
            zipf.writestr("config.json", "{ invalid json }")

        logs = Logs()

        with tempfile.TemporaryDirectory() as extract_dir:
            # Should still extract
            onnx_path, config_path, _ = extract_input_bundle(str(bundle_path), extract_dir, logs)

            # But loading config should fail
            with pytest.raises(Exception):  # JSON parse error
                OptimizationConfig.from_file(config_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
