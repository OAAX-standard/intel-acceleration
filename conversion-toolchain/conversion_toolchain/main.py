def cli():
    import argparse
    parser = argparse.ArgumentParser(description='Simplify any ONNX file')

    parser.add_argument('--onnx-path', required=True, help='Path to the ONNX file')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    args = parser.parse_args()

    onnx_path = args.onnx_path
    output_dir = args.output_dir

    from os import makedirs
    from os.path import split, join
    from .utils import simplify_onnx, md5_hash
    from .logger import Logs

    logs = Logs()
    logs.add_message('Simplifying ONNX model',
                     {'ONNX Path': onnx_path,
                      'MD5': md5_hash(onnx_path)})

    makedirs(output_dir, exist_ok=True)

    optimized_onnx_path = simplify_onnx(onnx_path, logs)

    new_onnx_filename = split(optimized_onnx_path)[1]
    new_onnx_path = join(output_dir, new_onnx_filename)
    logs_path = join(output_dir, 'logs.json')

    mime_type = 'application/x-onnx; device=cpu'

    # Copy optimized model and logs to the output directory
    from shutil import copy
    logs.save_as_json(logs_path)
    if optimized_onnx_path.strip() != new_onnx_path.strip():
        copy(optimized_onnx_path, new_onnx_path)
        
    # Add message to logs
    logs.add_message('Successful Conversion',
                    {
                        "Output Directory": output_dir,
                        "Output file name": new_onnx_filename,
                        "MIME type": mime_type,
                        "Output file MD5": md5_hash(new_onnx_path),
                        "Logs file name": "logs.json"
                    }
                    )

    # Print logs to stdout
    print(logs)
    print('Exiting.')
