def cli():
    import argparse
    import tempfile
    import zipfile
    from os import makedirs
    from os.path import basename
    from .utils import convert_to_ir, md5_hash, extract_input_bundle
    from .config import OptimizationConfig
    from .logger import Logs

    parser = argparse.ArgumentParser(
        description='Convert ONNX model to OpenVINO IR format with optional quantization',
        usage='conversion_toolchain INPUT_ZIP OUTPUT_DIR'
    )

    # Positional arguments
    parser.add_argument(
        'input_zip',
        help='Path to input zip bundle (containing model.onnx, optional config.json, and optional calibration/)'
    )

    parser.add_argument(
        'output_dir',
        help='Output directory for converted model'
    )

    args = parser.parse_args()

    input_zip = args.input_zip
    output_dir = args.output_dir

    makedirs(output_dir, exist_ok=True)

    logs = Logs()

    try:
        logs.add_message('Starting OpenVINO conversion', {
            'Input bundle': input_zip,
            'Output directory': output_dir
        })

        # Calculate MD5 of input (with error handling)
        try:
            input_md5 = md5_hash(input_zip)
            logs.add_message('Input validation', {'MD5': input_md5})
        except Exception as e:
            logs.add_message('Warning: Could not calculate input MD5', {'Error': str(e)})

        # Extract bundle to temporary directory
        with tempfile.TemporaryDirectory() as temp_extract_dir:
            onnx_path, config_path, calibration_dir = extract_input_bundle(
                input_zip,
                temp_extract_dir,
                logs  # Pass logs for detailed error tracking
            )

            logs.add_message('Bundle extraction complete', {
                'ONNX model': onnx_path,
                'Config found': config_path is not None,
                'Calibration data found': calibration_dir is not None
            })

            # Load configuration
            if config_path:
                config = OptimizationConfig.from_file(config_path)
                logs.add_message('Configuration loaded', config.to_dict())
            else:
                config = OptimizationConfig.from_default()
                logs.add_message('Using default configuration (FP16 compression enabled)')

            # Convert ONNX to OpenVINO IR
            zip_path = convert_to_ir(onnx_path, output_dir, logs, config, calibration_dir)

        # Final output processing
        zip_filename = basename(zip_path)
        logs_path = f"{output_dir}/logs.json"

        # MIME type for zipped OpenVINO IR
        mime_type = 'application/zip; content=openvino-ir; device=cpu'

        # Calculate hash for zip file
        zip_md5 = md5_hash(zip_path)

        # Save logs
        logs.save_as_json(logs_path)

        # Add final success message to logs
        logs.add_message('Successful Conversion',
                        {
                            "Output Directory": output_dir,
                            "Output file name": zip_filename,
                            "MIME type": mime_type,
                            "Output file MD5": zip_md5,
                            "Logs file name": "logs.json",
                            "Optimizations": {
                                "FP16 compression": config.get_fp16_compression(),
                                "INT8 quantization": config.is_quantization_enabled()
                            }
                        }
                        )

        # Print logs to stdout
        print(logs)
        print('Exiting.')

    except FileNotFoundError as e:
        logs.add_message('FATAL ERROR: File not found', {'Error': str(e)})
        logs.save_as_json(f"{output_dir}/logs.json")
        print(logs)
        print(f"\nERROR: {e}")
        exit(1)

    except zipfile.BadZipFile as e:
        logs.add_message('FATAL ERROR: Invalid zip archive', {'Error': str(e)})
        logs.save_as_json(f"{output_dir}/logs.json")
        print(logs)
        print(f"\nERROR: {e}")
        exit(2)

    except ValueError as e:
        logs.add_message('FATAL ERROR: Invalid input', {'Error': str(e)})
        logs.save_as_json(f"{output_dir}/logs.json")
        print(logs)
        print(f"\nERROR: {e}")
        exit(2)

    except RuntimeError as e:
        logs.add_message('FATAL ERROR: Conversion failed', {'Error': str(e)})
        logs.save_as_json(f"{output_dir}/logs.json")
        print(logs)
        print(f"\nERROR: {e}")
        exit(3)

    except IOError as e:
        logs.add_message('FATAL ERROR: I/O operation failed', {'Error': str(e)})
        logs.save_as_json(f"{output_dir}/logs.json")
        print(logs)
        print(f"\nERROR: {e}")
        exit(4)

    except Exception as e:
        logs.add_message('FATAL ERROR: Unexpected error', {
            'Error': str(e),
            'Type': type(e).__name__
        })
        logs.save_as_json(f"{output_dir}/logs.json")
        print(logs)
        print(f"\nUNEXPECTED ERROR: {e}")
        exit(255)
