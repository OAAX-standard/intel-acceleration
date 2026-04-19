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

> **v2 API:** The runtime now implements the OAAX v2 interface (`oaax_runtime.h`).
> Key changes: multi-model loading (`runtime_load_models`), request correlation via `Tensors.id`,
> blocking `runtime_retrieve_output` with timeout, and `runtime_get_info` JSON diagnostics.
> See [`runtime-library/README.md`](runtime-library/README.md) for the full API reference.

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

## Benchmark results (Intel Core i7-12700K, CPU, batch=1)

### `benchmark_app` vs OAAX v2 runtime (300 runs, warmup=5)

| Model | Precision | benchmark_app | OAAX runtime | Delta |
|-------|-----------|--------------|--------------|-------|
| yolo11n | FP32 | ~99 FPS | ~93 FPS | -6% |
| yolo11n | FP16 | ~97 FPS | ~95 FPS | -2% |
| yolo11n | INT8 | **~238 FPS** | **~223 FPS** | -6% |
| yolov8n | FP32 | ~84 FPS | ~82 FPS | -2% |
| yolov8n | FP16 | ~83 FPS | ~83 FPS | 0% |
| yolov8n | INT8 | **~235 FPS** | **~223 FPS** | -5% |
| yolo11s | FP32 | ~33 FPS | ~33 FPS | 0% |
| yolo11s | INT8 | ~99 FPS | ~98 FPS | -1% |

> FP32 and FP16 are identical on this CPU — the i7-12700K has no native FP16 ALU;
> OpenVINO computes FP16 models as FP32 internally. Only INT8 benefits from AVX2/VNNI (~2.5×).
>
> The ~5% gap vs `benchmark_app` comes from producer/consumer thread overhead in the C API boundary.

### GPU benchmarks (iGPU UHD 770 + dGPU RTX A4000, `yolo_test` OAAX runtime)

| Model | Precision | GPU.0 (iGPU) | GPU.1 (dGPU) | GPU (MULTI) |
|-------|-----------|--------------|--------------|-------------|
| yolov8n | FP32 | ~72 FPS | ~55 FPS | **~117 FPS** |
| yolov8n | INT8 | ~101 FPS | ~60 FPS | **~152 FPS** |
| yolo11n | FP32 | ~75 FPS | ~67 FPS | **~130 FPS** |
| yolo11n | INT8 | ~103 FPS | ~71 FPS | **~156 FPS** |

Use `device_type=GPU` with `perf_hint=cumulative_throughput` for best multi-GPU aggregate FPS.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
