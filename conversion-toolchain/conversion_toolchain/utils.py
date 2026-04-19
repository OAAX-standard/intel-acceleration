import hashlib
import tempfile
import zipfile
from pathlib import Path

import openvino as ov

from .config import OptimizationConfig
from .quantization import is_nncf_available, quantize_model


def extract_input_bundle(input_zip: str, extract_dir: str, logs=None) -> tuple[str, str | None, str | None]:
    """
    Extract input zip bundle containing model, config, and calibration data.

    Args:
        input_zip: Path to input zip file
        extract_dir: Directory to extract files to
        logs: Optional logger instance

    Returns:
        Tuple of (onnx_path, config_path, calibration_dir)

    Raises:
        FileNotFoundError: If input zip doesn't exist
        zipfile.BadZipFile: If input file is not a valid zip archive
        ValueError: If no ONNX model found in archive
        PermissionError: If cannot read zip or write to extract directory
    """
    input_path = Path(input_zip)

    # Validate input file exists
    if not input_path.exists():
        error_msg = f"Input zip file does not exist: {input_zip}"
        if logs:
            logs.add_message("Error: Input validation failed", {"Error": error_msg})
        raise FileNotFoundError(error_msg)

    # Validate it's a file, not a directory
    if not input_path.is_file():
        error_msg = f"Input path is not a file: {input_zip}"
        if logs:
            logs.add_message("Error: Input validation failed", {"Error": error_msg})
        raise ValueError(error_msg)

    # Validate file size (not empty)
    if input_path.stat().st_size == 0:
        error_msg = f"Input zip file is empty: {input_zip}"
        if logs:
            logs.add_message("Error: Input validation failed", {"Error": error_msg})
        raise ValueError(error_msg)

    if logs:
        logs.add_message(
            "Validating input archive", {"Path": str(input_path), "Size": f"{input_path.stat().st_size} bytes"}
        )

    extract_path = Path(extract_dir)
    extract_path.mkdir(parents=True, exist_ok=True)

    # Extract zip with validation
    try:
        if logs:
            logs.add_message("Extracting input archive")

        with zipfile.ZipFile(input_zip, "r") as zipf:
            # Test the zip file integrity
            bad_file = zipf.testzip()
            if bad_file:
                error_msg = f"Corrupted file in archive: {bad_file}"
                if logs:
                    logs.add_message("Error: Archive validation failed", {"Error": error_msg})
                raise zipfile.BadZipFile(error_msg)

            # List contents for logging
            file_list = zipf.namelist()
            if logs:
                logs.add_message("Archive contents", {"Files": file_list, "Count": len(file_list)})

            # Extract all files
            zipf.extractall(extract_path)

            if logs:
                logs.add_message("Archive extracted successfully")

    except zipfile.BadZipFile as e:
        error_msg = f"Invalid zip archive: {str(e)}"
        if logs:
            logs.add_message("Error: Failed to extract archive", {"Error": error_msg})
        raise zipfile.BadZipFile(error_msg) from e

    except PermissionError as e:
        error_msg = f"Permission denied: {str(e)}"
        if logs:
            logs.add_message("Error: Permission denied", {"Error": error_msg})
        raise

    # Find ONNX model
    onnx_files = list(extract_path.glob("*.onnx"))
    if not onnx_files:
        error_msg = "No ONNX model (*.onnx) found in input archive"
        if logs:
            logs.add_message(
                "Error: Model validation failed",
                {"Error": error_msg, "Files found": [f.name for f in extract_path.iterdir()]},
            )
        raise ValueError(error_msg)

    if len(onnx_files) > 1:
        if logs:
            logs.add_message(
                "Warning: Multiple ONNX files found, using first one",
                {"Files": [str(f) for f in onnx_files], "Selected": str(onnx_files[0])},
            )

    onnx_path = str(onnx_files[0])

    # Validate ONNX file is not empty
    if Path(onnx_path).stat().st_size == 0:
        error_msg = f"ONNX model file is empty: {onnx_path}"
        if logs:
            logs.add_message("Error: Model validation failed", {"Error": error_msg})
        raise ValueError(error_msg)

    if logs:
        logs.add_message("Found ONNX model", {"Path": onnx_path, "Size": f"{Path(onnx_path).stat().st_size} bytes"})

    # Find config file
    config_path = extract_path / "config.json"
    if config_path.exists():
        if logs:
            logs.add_message("Found configuration file", {"Path": str(config_path)})
        config_path = str(config_path)
    else:
        if logs:
            logs.add_message("No configuration file found, using defaults")
        config_path = None

    # Find calibration directory
    calibration_dir = None
    for possible_dir in ["calibration", "images", "data"]:
        cal_path = extract_path / possible_dir
        if cal_path.exists() and cal_path.is_dir():
            calibration_dir = str(cal_path)
            if logs:
                image_count = len(list(cal_path.glob("*")))
                logs.add_message("Found calibration data", {"Directory": possible_dir, "Images": image_count})
            break

    if calibration_dir is None and logs:
        logs.add_message("No calibration data found (quantization will be unavailable)")

    return onnx_path, config_path, calibration_dir


def convert_to_ir(
    onnx_path: str, output_dir: str, logs, config: OptimizationConfig | None = None, calibration_dir: str | None = None
):
    """
    Convert ONNX model to OpenVINO IR format with optional optimization.

    Args:
        onnx_path: Path to the input ONNX model
        output_dir: Directory where the zipped IR will be saved
        logs: Logger instance for tracking conversion progress
        config: Optimization configuration (uses defaults if None)
        calibration_dir: Path to calibration images (for quantization)

    Returns:
        str: Path to the generated zip file containing .xml and .bin files

    Raises:
        FileNotFoundError: If ONNX model doesn't exist
        ValueError: If model is invalid or cannot be loaded
        RuntimeError: If conversion or quantization fails
        IOError: If output files cannot be written
    """
    # Validate ONNX model exists
    onnx_file = Path(onnx_path)
    if not onnx_file.exists():
        error_msg = f"ONNX model file does not exist: {onnx_path}"
        logs.add_message("Error: Model file not found", {"Error": error_msg})
        raise FileNotFoundError(error_msg)

    # Validate ONNX model is readable
    if not onnx_file.is_file():
        error_msg = f"ONNX path is not a file: {onnx_path}"
        logs.add_message("Error: Invalid model path", {"Error": error_msg})
        raise ValueError(error_msg)

    # Use default config if not provided
    if config is None:
        config = OptimizationConfig.from_default()

    try:
        logs.add_message(
            "Starting model conversion", {"Model": str(onnx_file.name), "Size": f"{onnx_file.stat().st_size} bytes"}
        )

        logs.add_message("Converting ONNX model to OpenVINO IR format")

        batch_size = config.get_batch_size()

        # Load and convert the ONNX model using OpenVINO.
        # For batch_size > 1 we probe the default conversion to get input names/shapes,
        # then re-convert with explicit batch-sized inputs so the batch dim propagates
        # through the entire graph (including internal Reshape constants that post-hoc
        # reshape/set_batch cannot update).
        try:
            if batch_size == 1:
                model = ov.convert_model(onnx_path)
            else:
                probe = ov.convert_model(onnx_path)
                input_specs = []
                for inp in probe.inputs:
                    ps = inp.partial_shape
                    new_shape = [batch_size] + [
                        ps[i].get_length() if ps[i].is_static else -1 for i in range(1, len(ps))
                    ]
                    input_specs.append((inp.any_name, new_shape))
                try:
                    model = ov.convert_model(onnx_path, input=input_specs)
                except Exception as e:
                    raise RuntimeError(
                        f"batch_size={batch_size} conversion failed. "
                        "The ONNX model likely has hardcoded batch=1 constants in its post-processing. "
                        "Re-export the model from PyTorch with the desired batch size and try again. "
                        f"Original error: {e}"
                    ) from e
                logs.add_message("Batch size set", {"batch_size": batch_size})
        except RuntimeError:
            raise
        except Exception as e:
            error_msg = f"Failed to convert ONNX model: {str(e)}"
            logs.add_message(
                "Error: Model conversion failed",
                {"Error": error_msg, "Model": onnx_path, "Exception type": type(e).__name__},
            )
            raise RuntimeError(error_msg) from e

        logs.add_message("Successfully converted to OpenVINO IR format")

        # Apply quantization if enabled
        if config.is_quantization_enabled():
            if not is_nncf_available():
                logs.add_message("Warning: NNCF not available, skipping quantization")
            else:
                # Determine calibration directory
                if calibration_dir is None:
                    raise ValueError(
                        "Quantization is enabled but no calibration data provided. "
                        "Please include calibration images in the input zip or disable quantization."
                    )

                logs.add_message("Applying INT8 quantization")

                model = quantize_model(
                    model=model,
                    calibration_dir=calibration_dir,
                    preset=config.get_quantization_preset(),
                    subset_size=config.get_quantization_subset_size(),
                    logs=logs,
                )

        # Generate output paths
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Use the base name of the input file for output
        model_name = Path(onnx_path).stem

        # Create a temporary directory for IR files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            xml_path = temp_path / f"{model_name}.xml"
            bin_path = temp_path / f"{model_name}.bin"

            # Save the model in IR format with FP16 compression setting
            compress_fp16 = config.get_fp16_compression()

            logs.add_message(
                "Saving model to IR format",
                {"FP16 compression": compress_fp16, "Quantized": config.is_quantization_enabled()},
            )

            try:
                ov.save_model(model, str(xml_path), compress_to_fp16=compress_fp16)
            except Exception as e:
                error_msg = f"Failed to save model to IR format: {str(e)}"
                logs.add_message(
                    "Error: Model save failed",
                    {"Error": error_msg, "Output path": str(xml_path), "Exception type": type(e).__name__},
                )
                raise OSError(error_msg) from e

            # Verify output files were created
            if not xml_path.exists() or not bin_path.exists():
                error_msg = "IR files were not created properly"
                logs.add_message(
                    "Error: Output validation failed",
                    {"XML exists": xml_path.exists(), "BIN exists": bin_path.exists()},
                )
                raise OSError(error_msg)

            logs.add_message(
                "Model saved successfully",
                {
                    "XML file": f"{model_name}.xml",
                    "XML size": f"{xml_path.stat().st_size} bytes",
                    "BIN file": f"{model_name}.bin",
                    "BIN size": f"{bin_path.stat().st_size} bytes",
                },
            )

            # Create zip file containing both IR files
            zip_path = output_path / f"{model_name}.zip"

            logs.add_message("Packaging IR files into zip archive")

            try:
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(xml_path, arcname=f"{model_name}.xml")
                    zipf.write(bin_path, arcname=f"{model_name}.bin")
            except Exception as e:
                error_msg = f"Failed to create output zip archive: {str(e)}"
                logs.add_message("Error: Zip creation failed", {"Error": error_msg, "Output path": str(zip_path)})
                raise OSError(error_msg) from e

            # Verify zip was created
            if not zip_path.exists():
                error_msg = "Output zip file was not created"
                logs.add_message("Error: Output validation failed", {"Error": error_msg})
                raise OSError(error_msg)

            logs.add_message(
                "Successfully packaged IR files",
                {
                    "Zip file": str(zip_path),
                    "Zip size": f"{zip_path.stat().st_size} bytes",
                    "Contents": [f"{model_name}.xml", f"{model_name}.bin"],
                    "Optimizations applied": {
                        "FP16 compression": compress_fp16,
                        "INT8 quantization": config.is_quantization_enabled(),
                    },
                },
            )

        return str(zip_path)

    except Exception as e:
        logs.add_message("Failed to convert ONNX model to IR", {"Error": str(e)})
        raise


def md5_hash(file_path):
    """
    Calculate MD5 hash of a file.

    Args:
        file_path: Path to the file

    Returns:
        str: MD5 hash as hexadecimal string
    """
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()
