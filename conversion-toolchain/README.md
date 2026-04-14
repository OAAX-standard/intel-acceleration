# OAAX Conversion Toolchain

Docker container that converts an ONNX model to OpenVINO IR format with optional
FP16 compression or INT8 quantization via NNCF.

**Input:** ONNX model (bare `.onnx` file or a `.zip` bundle)
**Output:** OpenVINO IR (`.xml` + `.bin`) in a `.zip` archive, plus `logs.json`

## Quick start

### 1. Build the Docker image

```bash
bash build-toolchain.sh
```

### 2. Convert a model

The simplest case — just zip your ONNX model and run the container:

```bash
zip input/model.zip model.onnx
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  oaax-intel-toolchain:latest /input/model.zip /output
```

### 3. Use the output

```bash
unzip output/model.zip -d converted/
# converted/model.xml   — OpenVINO IR graph
# converted/model.bin   — weights
# output/logs.json      — conversion log
```

## Configuration

By default the toolchain applies **FP16 compression** (≈50% size reduction). Add a
`config.json` to the zip bundle to change behaviour.

### Disable FP16

```json
{
  "optimization": {
    "fp16_compression": false
  }
}
```

### INT8 quantization (requires calibration images)

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

Bundle with calibration images:

```bash
zip -r input/model.zip model.onnx config.json calibration/
```

`preset` options:
- `mixed` — recommended; quantizes most layers, preserves sensitive ones
- `performance` — quantizes all layers; fastest but may reduce accuracy

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Input file not found |
| 2 | Invalid input (bad zip, missing model.onnx) |
| 3 | Conversion or quantization failed — check `logs.json` |
| 4 | I/O error (permissions, disk full) |
| 255 | Unexpected error — check `logs.json` |

Example in a shell script:

```bash
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  oaax-intel-toolchain:latest /input/model.zip /output

if [ $? -ne 0 ]; then
  echo "Conversion failed — check output/logs.json"
  cat output/logs.json
fi
```

## Troubleshooting

**Permission errors on output directory**
```bash
# Either pre-create the directory with open permissions:
mkdir -p output && chmod 777 output

# Or run as your user:
docker run --rm --user $(id -u):$(id -g) \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  oaax-intel-toolchain:latest /input/model.zip /output
```

**Conversion fails (exit code 3)**
```bash
# Inspect the structured log
python3 -m json.tool output/logs.json
```

**Debug interactively**
```bash
docker run --rm -it \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  --entrypoint /bin/bash \
  oaax-intel-toolchain:latest
# then run: conversion_toolchain /input/model.zip /output
```

**Docker build fails (cache issue)**
```bash
docker build --no-cache -f Dockerfile -t oaax-intel-toolchain ..
```

## Local development

Install the package (and test dependencies) without Docker:

```bash
# From the repo root
uv venv && source .venv/bin/activate
uv sync --extra integration --extra quantization
```

Run the CLI directly:

```bash
conversion_toolchain input/model.zip output/
```

## Testing

```bash
# From the repo root (venv activated)
pytest tests/test_conversion.py -v       # unit tests
pytest tests/test_docker.py -v           # Docker image tests (image must be built)
pytest tests/test_yolo_integration.py -v # YOLO end-to-end (requires ultralytics)
```

Or use the quick test script (builds venv, checks structure, runs unit tests):

```bash
cd conversion-toolchain
bash quick-test.sh
```

## Requirements

- Docker 20.10+
- Python 3.13+ (for local development without Docker)
- OpenVINO 2026.1.0+ (installed automatically inside the Docker image)
- NNCF 2.19+ (installed automatically; required for INT8 quantization)

## Project structure

```
conversion-toolchain/
├── conversion_toolchain/
│   ├── main.py          # CLI entrypoint and exit codes
│   ├── utils.py         # ONNX → OpenVINO IR conversion
│   ├── quantization.py  # NNCF INT8 quantization
│   ├── config.py        # OptimizationConfig (FP16 / INT8 settings)
│   └── logger.py        # JSON structured logging
├── Dockerfile
├── build-toolchain.sh
├── pyproject.toml
└── VERSION
```

## License

See repository [LICENSE](../LICENSE).
