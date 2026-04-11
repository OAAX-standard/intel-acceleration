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

See [CONFIG_SCHEMA.md](CONFIG_SCHEMA.md) for all options.

---

## Testing

Tests live in the project root `tests/` directory and are run with `uv` from there.

### Quick Validation

```bash
./quick-test.sh
```

### Unit Tests

```bash
# From project root
uv sync
pytest tests/test_conversion.py -v
```

### Docker Tests

```bash
# Requires image to be built first
IMAGE_NAME=oaax-intel-toolchain bash build-toolchain.sh
pytest tests/test_docker.py -v
```

### YOLO Integration Tests

```bash
# From project root
uv sync --extra integration
pytest tests/test_yolo_integration.py -v
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
# From project root — installs conversion_toolchain + test deps
uv venv && source .venv/bin/activate
uv sync
```

### Usage

```bash
conversion_toolchain input/model.zip output/
```

### Development

```bash
# Run all tests from project root
pytest tests/ -v

# Run validation script (no Docker required)
./quick-test.sh
```

---

## Requirements

**Runtime:**
- Python 3.13+
- OpenVINO 2026.1.0+
- Docker 20.10+ (for containerized deployment)

**Optional:**
- NNCF 2.19+ (for INT8 quantization)

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
