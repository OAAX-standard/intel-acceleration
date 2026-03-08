# Intel Conversion Toolchain

Convert ONNX models to optimized OpenVINO IR format with FP16/INT8 compression.

**Deployment:** Production-ready Docker container
**Input:** ONNX model in zip bundle
**Output:** OpenVINO IR (.xml + .bin) in zip package

---

## Quick Start

### 1. Build Docker Image

```bash
bash build-toolchain.sh
```

### 2. Convert Your Model

```bash
# Prepare your model
mkdir -p input output
zip input/model.zip your_model.onnx config.json calibration_images/

# Run conversion
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  oaax-intel-toolchain:latest /input/model.zip /output
```

### 3. Use the Result

```bash
# Extract converted model
unzip output/model.zip -d converted/

# Files created:
# - converted/model.xml  (OpenVINO IR)
# - converted/model.bin  (weights)
# - output/logs.json     (conversion logs)
```

**Done!** Your model is now optimized for OpenVINO.

---

## Features

- ✅ **FP16 Compression** - Default 50% size reduction
- ✅ **INT8 Quantization** - Optional 75% reduction with NNCF
- ✅ **Simple Interface** - Just 2 arguments
- ✅ **Comprehensive Logging** - Detailed JSON logs
- ✅ **Error Handling** - Clear exit codes (0-4, 255)
- ✅ **Production Ready** - Docker-based deployment

---

## Configuration

### Default (FP16 Compression)

Just zip your ONNX model:
```bash
zip input/model.zip model.onnx
```

### Custom Settings

Create `config.json`:
```json
{
  "optimization": {
    "fp16_compression": false
  }
}
```

Add to bundle:
```bash
zip input/model.zip model.onnx config.json
```

### INT8 Quantization

Create `config.json` with quantization settings:
```json
{
  "optimization": {
    "fp16_compression": false,
    "quantization": {
      "enabled": true,
      "preset": "mixed",
      "subset_size": 300
    }
  }
}
```

Add calibration images:
```bash
zip -r input/model.zip model.onnx config.json calibration/
```

**Presets:**
- `performance` - Max speed, some accuracy loss
- `mixed` - Balanced (recommended)
- `accuracy` - Max accuracy preservation

See [CONFIG_SCHEMA.md](CONFIG_SCHEMA.md) for all options.

---

## Testing

### Quick Validation

```bash
./quick-test.sh
```

Validates installation without requiring Docker.

### Full Docker Test

```bash
./test_docker_image.sh
```

Downloads test model, builds image, and validates conversion.

### Unit Tests

```bash
# Install test dependencies
pip install pytest

# Run tests
pytest tests/test_conversion.py -v
```

---

## Exit Codes

| Code | Meaning | Action |
|------|---------|--------|
| 0 | Success | Model converted |
| 1 | File not found | Check input path |
| 2 | Invalid input | Verify zip/model |
| 3 | Conversion failed | Check logs.json |
| 4 | I/O error | Check permissions |
| 255 | Unexpected error | Check logs.json |

**Example:**
```bash
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  openvino-converter /input/model.zip /output

if [ $? -eq 0 ]; then
  echo "✓ Success"
else
  echo "✗ Failed - check output/logs.json"
fi
```

---

## Troubleshooting

### Build Fails

```bash
# Clean build
docker build --no-cache -t oaax-intel-toolchain .
```

### Conversion Fails

```bash
# Check detailed logs
cat output/logs.json | python -m json.tool
```

### Permission Errors

```bash
# Fix output permissions
chmod 777 output/

# Or run as current user
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  oaax-intel-toolchain /input/model.zip /output
```

### Debug Mode

```bash
# Enter container
docker run --rm -it \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  --entrypoint /bin/bash \
  oaax-intel-toolchain

# Run manually
conversion_toolchain /input/model.zip /output
```

---

## Local Development

### Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install package
uv pip install -e .
```

### Usage

```bash
conversion_toolchain input/model.zip output/
```

### Development

```bash
# Install with test dependencies
uv pip install -e ".[test]"

# Run tests
pytest tests/ -v

# Run validation
./quick-test.sh
```

---

## Requirements

**Runtime:**
- Python 3.10+
- OpenVINO 2024.0+
- Docker 20.10+ (for containerized deployment)

**Optional:**
- NNCF 2.11+ (for INT8 quantization)

---

## Architecture

```
Input Bundle (zip)
├── model.onnx          (required)
├── config.json         (optional)
└── calibration/        (optional, for INT8)
    └── *.jpg/png

         ↓

OpenVINO Converter
├── Validate input
├── Extract bundle
├── Convert ONNX → IR
├── Apply optimizations
└── Package output

         ↓

Output
├── model.zip           (model.xml + model.bin)
└── logs.json           (detailed logs)
```

---

## License

See repository license.

---

## Quick Commands

```bash
# Build
docker build -t oaax-intel-toolchain .

# Convert
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  oaax-intel-toolchain /input/model.zip /output

# Test
./quick-test.sh

# Help
docker run --rm oaax-intel-toolchain --help
```

---

**Ready to optimize your ONNX models!** 🚀
