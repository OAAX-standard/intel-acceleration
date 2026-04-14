import sys
import zipfile


def _fatal(logs, output_dir: str, label: str, error: Exception, hint: str, code: int) -> None:
    """Log a fatal error, save logs, print a user-friendly message, and exit."""
    logs.add_message(f"FATAL ERROR: {label}", {"Error": str(error)})
    logs.save_as_json(f"{output_dir}/logs.json")
    print(logs)
    print(f"\nERROR: {error}")
    print(f"Hint: {hint}")
    sys.exit(code)


def cli():
    import argparse
    import tempfile
    from os import makedirs
    from os.path import basename

    from .config import OptimizationConfig
    from .logger import Logs
    from .utils import convert_to_ir, extract_input_bundle, md5_hash

    parser = argparse.ArgumentParser(
        description="Convert ONNX model to OpenVINO IR format with optional quantization",
        usage="conversion_toolchain INPUT_ZIP OUTPUT_DIR",
    )

    # Positional arguments
    parser.add_argument(
        "input_zip",
        help="Path to input zip bundle (containing model.onnx, optional config.json, and optional calibration/)",
    )

    parser.add_argument("output_dir", help="Output directory for converted model")

    args = parser.parse_args()

    input_zip = args.input_zip
    output_dir = args.output_dir

    makedirs(output_dir, exist_ok=True)

    logs = Logs()

    try:
        logs.add_message("Starting OpenVINO conversion", {"Input bundle": input_zip, "Output directory": output_dir})

        # Calculate MD5 of input (with error handling)
        try:
            input_md5 = md5_hash(input_zip)
            logs.add_message("Input validation", {"MD5": input_md5})
        except Exception as e:
            logs.add_message("Warning: Could not calculate input MD5", {"Error": str(e)})

        # Extract bundle to temporary directory
        with tempfile.TemporaryDirectory() as temp_extract_dir:
            onnx_path, config_path, calibration_dir = extract_input_bundle(
                input_zip,
                temp_extract_dir,
                logs,
            )

            logs.add_message(
                "Bundle extraction complete",
                {
                    "ONNX model": onnx_path,
                    "Config found": config_path is not None,
                    "Calibration data found": calibration_dir is not None,
                },
            )

            # Load configuration
            if config_path:
                config = OptimizationConfig.from_file(config_path)
                logs.add_message("Configuration loaded", config.to_dict())
            else:
                config = OptimizationConfig.from_default()
                logs.add_message("Using default configuration (FP16 compression enabled)")

            # Convert ONNX to OpenVINO IR
            zip_path = convert_to_ir(onnx_path, output_dir, logs, config, calibration_dir)

        # Final output processing
        zip_filename = basename(zip_path)
        logs_path = f"{output_dir}/logs.json"

        # MIME type for zipped OpenVINO IR
        mime_type = "application/zip; content=openvino-ir; device=cpu"

        # Calculate hash for zip file
        zip_md5 = md5_hash(zip_path)

        # Save logs
        logs.save_as_json(logs_path)

        # Add final success message to logs
        logs.add_message(
            "Successful Conversion",
            {
                "Output Directory": output_dir,
                "Output file name": zip_filename,
                "MIME type": mime_type,
                "Output file MD5": zip_md5,
                "Logs file name": "logs.json",
                "Optimizations": {
                    "FP16 compression": config.get_fp16_compression(),
                    "INT8 quantization": config.is_quantization_enabled(),
                },
            },
        )

        print(logs)

    except FileNotFoundError as e:
        _fatal(
            logs,
            output_dir,
            "File not found",
            e,
            "Ensure the input zip file path is correct and the file exists.",
            1,
        )

    except zipfile.BadZipFile as e:
        _fatal(
            logs,
            output_dir,
            "Invalid zip archive",
            e,
            "The input must be a valid .zip file — not a raw .onnx or a corrupted archive.",
            2,
        )

    except ValueError as e:
        _fatal(
            logs,
            output_dir,
            "Invalid input",
            e,
            "The bundle may be missing model.onnx, or config.json is malformed.",
            2,
        )

    except RuntimeError as e:
        _fatal(
            logs,
            output_dir,
            "Conversion failed",
            e,
            "The model may contain unsupported ONNX operators. "
            "Check OpenVINO compatibility or try without quantization.",
            3,
        )

    except OSError as e:
        _fatal(
            logs,
            output_dir,
            "I/O operation failed",
            e,
            "Check available disk space and write permissions on the output directory.",
            4,
        )

    except Exception as e:
        _fatal(
            logs,
            output_dir,
            f"Unexpected error ({type(e).__name__})",
            e,
            "This is an unhandled error. Please report it at "
            "https://github.com/OAAX-standard/intel-acceleration/issues",
            255,
        )
