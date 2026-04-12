# intel-acceleration

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Build](https://github.com/OAAX-standard/intel-acceleration/actions/workflows/build.yml/badge.svg)](https://github.com/OAAX-standard/intel-acceleration/actions/workflows/build.yml)

OAAX implementation for Intel hardware (CPU, GPU, NPU) on x86_64. Two components:

- **Conversion Toolchain** — Docker container that converts an ONNX model to OpenVINO IR with optional FP16 compression or INT8 quantization.
- **Runtime Library** — C++ shared library that loads an OpenVINO IR model and runs inference via the OpenVINO native API.

> Full deployment guide: [https://docs.oaax.org/Intel/](https://docs.oaax.org/Intel/)

## Repository structure

```
conversion-toolchain/   ONNX → OpenVINO IR conversion (Docker)
runtime-library/        C++ shared library for inference (CPU/GPU/NPU)
tests/                  Conversion tests, YOLO integration tests, runtime benchmarks
scripts/                CI setup helpers
```

## Pre-built artifacts

Pre-built toolchain images and runtime libraries are available in the
[contributions](https://github.com/oaax-standard/contributions) repository.
Usage examples are in the [examples](https://github.com/oaax-standard/examples) repository.

## Building from source

```bash
# Conversion toolchain (produces a Docker image tarball)
cd conversion-toolchain
IMAGE_NAME=oaax-intel-toolchain bash build-toolchain.sh

# Runtime library (Linux x86_64, produces libRuntimeLibrary.so + OpenVINO .so)
cd runtime-library
bash build-runtimes.sh        # set OPENVINO_DIR if not at /opt/intel/openvino/runtime
```

See [`conversion-toolchain/README.md`](conversion-toolchain/README.md) and
[`runtime-library/README.md`](runtime-library/README.md) for detailed instructions.

## Testing

Tests run in two stages. Stage 1 compiles models; Stage 2 benchmarks them.

```bash
# One-time setup
uv venv && source .venv/bin/activate
uv sync --extra integration --extra quantization

# Stage 1 — compile YOLO models to FP32/FP16/INT8 IR and run conversion tests
python tests/stage1.py

# Stage 2 — benchmark with benchmark_app + yolo_test across all variants and devices
python tests/stage2.py [--devices CPU,GPU.0] [--duration 10] [--csv results.csv]
```

Individual test suites:

```bash
pytest tests/test_conversion.py -v             # toolchain unit tests
pytest tests/test_yolo_integration.py -v       # YOLO IR validation
pytest tests/test_quantization_accuracy.py -v  # INT8/FP16 accuracy vs FP32 baseline
pytest tests/test_docker.py -v                 # Docker image tests (image must be built)
pytest -m slow tests/test_memory.py           # memory leak tests (~30 min)
```

## Benchmark results (Intel Core i7-12700K, hint=throughput)

| Tool | Model | Precision | Device | Throughput |
|------|-------|-----------|--------|------------|
| benchmark_app | yolo11n | FP32 | CPU | ~100 FPS |
| benchmark_app | yolo11n | FP16 | CPU | ~97 FPS |
| benchmark_app | yolo11n | INT8 | CPU | **~245 FPS** |
| yolo_test (OAAX) | yolo11n | FP32 | CPU | ~96 FPS |
| yolo_test (OAAX) | yolo11n | FP16 | CPU | ~97 FPS |
| yolo_test (OAAX) | yolo11n | INT8 | CPU | ~236 FPS |
| benchmark_app | yolo11n | FP32 | GPU | ~82 FPS |
| benchmark_app | yolo11n | INT8 | GPU | ~113 FPS |
| yolo_test (OAAX) | yolo11n | FP32 | GPU | ~75 FPS |
| yolo_test (OAAX) | yolo11n | INT8 | GPU | ~101 FPS |

The OAAX runtime matches `benchmark_app` within ~4% on CPU across all precisions.
The GPU gap (~9–13%) is driven by dispatch overhead through the C API boundary becoming
significant at higher GPU throughput; on CPU this cost is negligible relative to inference time.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
