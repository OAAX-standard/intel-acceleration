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

### CPU benchmarks (`benchmark_app`, batch=1)

| Model | Precision | Throughput |
|-------|-----------|------------|
| yolo11n | FP32 | ~96 FPS |
| yolo11n | FP16 | ~94 FPS |
| yolo11n | INT8 | **~229 FPS** |
| yolov8n | FP32 | ~81 FPS |
| yolov8n | FP16 | ~80 FPS |
| yolo11s | FP32 | ~32 FPS |
| yolo11s | FP16 | ~32 FPS |

> FP32 and FP16 are identical on this CPU — the i7-12700K has no native FP16 ALU;
> OpenVINO computes FP16 models as FP32 internally. Only INT8 benefits from AVX2/VNNI (~2.5×).

### GPU benchmarks (iGPU UHD 770 + dGPU RTX A4000, `yolo_test` OAAX runtime)

| Model | Precision | GPU.0 (iGPU) | GPU.1 (dGPU) | GPU (MULTI) |
|-------|-----------|--------------|--------------|-------------|
| yolov8n | FP32 | ~72 FPS | ~55 FPS | **~117 FPS** |
| yolov8n | INT8 | ~101 FPS | ~60 FPS | **~152 FPS** |
| yolo11n | FP32 | ~75 FPS | ~67 FPS | **~130 FPS** |
| yolo11n | INT8 | ~103 FPS | ~71 FPS | **~156 FPS** |

The OAAX runtime matches `benchmark_app` within ~4% on CPU. The GPU throughput gap vs
`benchmark_app` (~9–13%) comes from H2D/D2H memory transfers through the C API boundary;
use the `MULTI` device with `perf_hint=cumulative_throughput` for best aggregate GPU FPS.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
