"""
Docker Integration Tests for OpenVINO Conversion Toolchain

These tests verify the Docker image works correctly end-to-end.
Run with: pytest tests/test_docker.py -v

Prerequisites:
- Docker installed and running
- Docker image built: docker build -t openvino-converter .
"""
import pytest
import subprocess
import tempfile
import shutil
import zipfile
import json
from pathlib import Path
import time

# Import model downloader
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from tests.download_test_models import download_model

# Docker image name
DOCKER_IMAGE = "openvino-converter:latest"


def docker_available():
    """Check if Docker is available and image is built"""
    try:
        # Check Docker is running
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5
        )
        if result.returncode != 0:
            return False

        # Check if image exists
        result = subprocess.run(
            ["docker", "images", "-q", DOCKER_IMAGE],
            capture_output=True,
            text=True,
            timeout=5
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


@pytest.fixture(scope="session")
def docker_check():
    """Check Docker availability at session start"""
    if not docker_available():
        pytest.skip(
            f"Docker not available or image '{DOCKER_IMAGE}' not built. "
            f"Build with: docker build -t {DOCKER_IMAGE} ."
        )


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for tests"""
    workspace = tempfile.mkdtemp()
    input_dir = Path(workspace) / "input"
    output_dir = Path(workspace) / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    yield {
        "workspace": Path(workspace),
        "input": input_dir,
        "output": output_dir
    }

    # Cleanup
    shutil.rmtree(workspace)


@pytest.fixture
def sample_model():
    """Download a sample model for testing"""
    models_dir = Path(__file__).parent / "test_models"
    models_dir.mkdir(exist_ok=True)

    model_path = download_model("squeezenet", str(models_dir))
    if not Path(model_path).exists():
        pytest.skip("Sample model not available")

    return model_path


class TestDockerBasics:
    """Test basic Docker image functionality"""

    def test_docker_image_exists(self, docker_check):
        """Test that Docker image exists"""
        result = subprocess.run(
            ["docker", "images", "-q", DOCKER_IMAGE],
            capture_output=True,
            text=True
        )
        assert result.stdout.strip(), f"Docker image {DOCKER_IMAGE} not found"

    def test_docker_help_command(self, docker_check):
        """Test that --help works"""
        result = subprocess.run(
            ["docker", "run", "--rm", DOCKER_IMAGE, "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0, "Help command failed"
        assert "INPUT_ZIP" in result.stdout or "input_zip" in result.stdout
        assert "OUTPUT_DIR" in result.stdout or "output_dir" in result.stdout

    def test_docker_version_info(self, docker_check):
        """Test Docker image labels"""
        result = subprocess.run(
            ["docker", "inspect", DOCKER_IMAGE],
            capture_output=True,
            text=True,
            timeout=10
        )
        assert result.returncode == 0

        # Check that we have labels
        output = result.stdout
        assert "openvino" in output.lower() or "conversion" in output.lower()


class TestDockerConversion:
    """Test actual model conversion in Docker"""

    def test_simple_conversion(self, docker_check, temp_workspace, sample_model):
        """Test basic model conversion through Docker"""
        # Create input bundle
        bundle_path = temp_workspace["input"] / "model.zip"
        with zipfile.ZipFile(bundle_path, 'w') as zipf:
            zipf.write(sample_model, arcname="model.onnx")

        # Run conversion
        result = subprocess.run([
            "docker", "run", "--rm",
            "-v", f"{temp_workspace['input']}:/input",
            "-v", f"{temp_workspace['output']}:/output",
            DOCKER_IMAGE,
            "/input/model.zip",
            "/output"
        ], capture_output=True, text=True, timeout=120)

        # Check success
        assert result.returncode == 0, f"Conversion failed: {result.stderr}"

        # Check output files exist
        output_files = list(temp_workspace["output"].glob("*.zip"))
        assert len(output_files) >= 1, "No output zip file created"

        # Check logs exist
        logs_file = temp_workspace["output"] / "logs.json"
        assert logs_file.exists(), "No logs.json created"

        # Verify logs
        with open(logs_file) as f:
            logs = json.load(f)
        assert isinstance(logs, list), "Logs should be a list"
        assert len(logs) > 0, "Logs should not be empty"

        # Check for success message
        messages = [msg.get("Message", "") for msg in logs]
        assert any("Successful" in msg for msg in messages), "No success message in logs"

    def test_conversion_with_fp16(self, docker_check, temp_workspace, sample_model):
        """Test conversion with FP16 compression"""
        # Create bundle with config
        bundle_path = temp_workspace["input"] / "model_fp16.zip"
        config = {
            "optimization": {
                "fp16_compression": True
            }
        }

        config_file = temp_workspace["input"] / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)

        with zipfile.ZipFile(bundle_path, 'w') as zipf:
            zipf.write(sample_model, arcname="model.onnx")
            zipf.write(config_file, arcname="config.json")

        # Run conversion
        result = subprocess.run([
            "docker", "run", "--rm",
            "-v", f"{temp_workspace['input']}:/input",
            "-v", f"{temp_workspace['output']}:/output",
            DOCKER_IMAGE,
            "/input/model_fp16.zip",
            "/output"
        ], capture_output=True, text=True, timeout=120)

        assert result.returncode == 0, f"Conversion failed: {result.stderr}"

        # Verify FP16 in logs
        logs_file = temp_workspace["output"] / "logs.json"
        with open(logs_file) as f:
            logs_content = f.read()
        assert "fp16" in logs_content.lower(), "FP16 not mentioned in logs"

    def test_conversion_with_fp32(self, docker_check, temp_workspace, sample_model):
        """Test conversion without compression (FP32)"""
        # Create bundle with config disabling FP16
        bundle_path = temp_workspace["input"] / "model_fp32.zip"
        config = {
            "optimization": {
                "fp16_compression": False
            }
        }

        config_file = temp_workspace["input"] / "config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f)

        with zipfile.ZipFile(bundle_path, 'w') as zipf:
            zipf.write(sample_model, arcname="model.onnx")
            zipf.write(config_file, arcname="config.json")

        # Run conversion
        result = subprocess.run([
            "docker", "run", "--rm",
            "-v", f"{temp_workspace['input']}:/input",
            "-v", f"{temp_workspace['output']}:/output",
            DOCKER_IMAGE,
            "/input/model_fp32.zip",
            "/output"
        ], capture_output=True, text=True, timeout=120)

        assert result.returncode == 0, f"Conversion failed: {result.stderr}"

        # Output file should exist
        output_files = list(temp_workspace["output"].glob("*.zip"))
        assert len(output_files) >= 1, "No output created"


class TestDockerErrorHandling:
    """Test error handling in Docker"""

    def test_nonexistent_input(self, docker_check, temp_workspace):
        """Test error when input file doesn't exist"""
        result = subprocess.run([
            "docker", "run", "--rm",
            "-v", f"{temp_workspace['input']}:/input",
            "-v", f"{temp_workspace['output']}:/output",
            DOCKER_IMAGE,
            "/input/nonexistent.zip",
            "/output"
        ], capture_output=True, text=True, timeout=30)

        # Should fail with exit code 1
        assert result.returncode == 1, "Should fail for nonexistent file"
        assert "does not exist" in result.stdout or "not found" in result.stdout.lower()

    def test_invalid_zip(self, docker_check, temp_workspace):
        """Test error with invalid zip file"""
        # Create invalid zip
        invalid_zip = temp_workspace["input"] / "invalid.zip"
        with open(invalid_zip, 'w') as f:
            f.write("Not a valid zip file")

        result = subprocess.run([
            "docker", "run", "--rm",
            "-v", f"{temp_workspace['input']}:/input",
            "-v", f"{temp_workspace['output']}:/output",
            DOCKER_IMAGE,
            "/input/invalid.zip",
            "/output"
        ], capture_output=True, text=True, timeout=30)

        # Should fail with exit code 2
        assert result.returncode == 2, "Should fail for invalid zip"
        assert "zip" in result.stdout.lower()

    def test_zip_without_model(self, docker_check, temp_workspace):
        """Test error when zip has no ONNX model"""
        # Create zip without model
        bundle_path = temp_workspace["input"] / "no_model.zip"
        with zipfile.ZipFile(bundle_path, 'w') as zipf:
            zipf.writestr("readme.txt", "No model here")

        result = subprocess.run([
            "docker", "run", "--rm",
            "-v", f"{temp_workspace['input']}:/input",
            "-v", f"{temp_workspace['output']}:/output",
            DOCKER_IMAGE,
            "/input/no_model.zip",
            "/output"
        ], capture_output=True, text=True, timeout=30)

        # Should fail with exit code 2
        assert result.returncode == 2, "Should fail when no model found"
        assert "ONNX" in result.stdout or "onnx" in result.stdout


class TestDockerPerformance:
    """Test Docker image performance characteristics"""

    def test_conversion_completes_in_reasonable_time(self, docker_check, temp_workspace, sample_model):
        """Test that conversion completes within reasonable time"""
        # Create input bundle
        bundle_path = temp_workspace["input"] / "model.zip"
        with zipfile.ZipFile(bundle_path, 'w') as zipf:
            zipf.write(sample_model, arcname="model.onnx")

        # Measure time
        start_time = time.time()

        result = subprocess.run([
            "docker", "run", "--rm",
            "-v", f"{temp_workspace['input']}:/input",
            "-v", f"{temp_workspace['output']}:/output",
            DOCKER_IMAGE,
            "/input/model.zip",
            "/output"
        ], capture_output=True, text=True, timeout=120)

        elapsed = time.time() - start_time

        assert result.returncode == 0, "Conversion failed"
        assert elapsed < 60, f"Conversion took too long: {elapsed:.2f}s"

        print(f"✓ Conversion completed in {elapsed:.2f}s")


class TestDockerVolumes:
    """Test Docker volume mounting"""

    def test_output_directory_creation(self, docker_check, temp_workspace, sample_model):
        """Test that output directory is used correctly"""
        # Create input bundle
        bundle_path = temp_workspace["input"] / "model.zip"
        with zipfile.ZipFile(bundle_path, 'w') as zipf:
            zipf.write(sample_model, arcname="model.onnx")

        # Use subdirectory for output
        sub_output = temp_workspace["output"] / "subdir"
        sub_output.mkdir()

        result = subprocess.run([
            "docker", "run", "--rm",
            "-v", f"{temp_workspace['input']}:/input",
            "-v", f"{sub_output}:/output",
            DOCKER_IMAGE,
            "/input/model.zip",
            "/output"
        ], capture_output=True, text=True, timeout=120)

        assert result.returncode == 0, f"Conversion failed: {result.stderr}"

        # Check output in subdirectory
        output_files = list(sub_output.glob("*.zip"))
        assert len(output_files) >= 1, "No output in subdirectory"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
