from onnx import load, save
from onnxsim import simplify
from os.path import splitext
import hashlib

def simplify_onnx(onnx_path: str, logs):
    # logs
    try:
        model, check = simplify(onnx_path, check_n=3)
        assert check, 'Failed to simplify ONNX model'
        logs.add_message('Simplified ONNX model successfully')
    except Exception as e:
        # logs
        if logs:
            logs.add_message('Failed to simplify ONNX model', {'Error': str(e), 'Fallback': 'Using the original ONNX file'})
        model = load(onnx_path)

    simp_onnx_path = splitext(onnx_path)[0] + '-simplified.onnx'
    save(model, simp_onnx_path)

    return simp_onnx_path

def md5_hash(file_path):
    with open(file_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()