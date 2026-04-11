# intel-acceleration

This folder contains the source code of the shared library and the Docker image that can be used by AI application developers to benefit from the acceleration offered by Intel CPU, GPU and NPU on x86_64 machines.

> To learn how to deploy on Intel, please check out the technical docs at [https://docs.oaax.org/Intel/](https://docs.oaax.org/Intel/)

## Repository structure

- [conversion-toolchain/](conversion-toolchain): OAAX conversion toolchain — converts ONNX models to OpenVINO IR (FP32/FP16/INT8).
- [runtime-library/](runtime-library): OAAX runtime library — C++ shared library for OpenVINO inference on Intel CPU/GPU/NPU.
- [tests/](tests): Conversion unit tests, YOLO integration tests, Docker tests, and C++ runtime tests.
- [scripts/](scripts): Two-stage test runner (`stage1_compile.sh`, `stage2_run.sh`).

## Building

```bash
# Conversion toolchain (Docker image)
cd conversion-toolchain
IMAGE_NAME=oaax-intel-toolchain bash build-toolchain.sh

# Runtime library (C++ shared library, X86_64)
cd runtime-library
bash build-runtimes.sh        # OPENVINO_DIR defaults to the Python package location
```

Outputs: `conversion-toolchain/artifacts/` (Docker image tarball) and `runtime-library/artifacts/X86_64/` (shared library + headers).

## Testing

Tests run in two stages. Stage 1 compiles models and validates the conversion pipeline; Stage 2 benchmarks those compiled models using `benchmark_app` and the C++ runtime library.

### Setup

```bash
uv venv && source .venv/bin/activate
uv sync --extra integration --extra quantization
```

### Stage 1 — Compile models + run conversion tests

Converts YOLOv8n and YOLOv11n to FP32/FP16/INT8 OpenVINO IR (cached in `tests/compiled_models/`), then runs all conversion and IR validation tests.

```bash
bash scripts/stage1_compile.sh
```

### Stage 2 — Benchmark + runtime validation

Uses compiled models from Stage 1. Builds the C++ runtime automatically if needed.

```bash
bash scripts/stage2_run.sh [--devices CPU,GPU.0] [--duration 10] [--csv results.csv]
```

- **Step 1:** Runs `benchmark_app` (throughput hint, p95) across all models × precision variants × devices.
- **Step 2:** Runs the C++ `yolo_test` binary (throughput hint, warmup + 30 runs) across all variants × devices.
- `--csv results.csv` appends all results for cross-run comparison.

### Individual pytest suites

```bash
pytest tests/test_conversion.py -v            # toolchain unit tests
pytest tests/test_yolo_integration.py -v      # YOLO IR validation (uses cached models)
pytest tests/test_quantization_accuracy.py -v # INT8/FP16 accuracy vs FP32 (requires compiled models)
pytest tests/test_docker.py -v                # Docker tests (image must be built first)
```

### C++ runtime smoke tests

```bash
cd runtime-library/build
./simple_test
./yolo_test /path/to/model.xml [device] [--runs N] [--warmup N] [--perf-hint latency|throughput]
```

## Benchmark results (reference, Intel Core i7-13700K)

Measured with `benchmark_app -hint throughput` and `yolo_test --perf-hint throughput`.

| Tool | Model | Precision | Device | Throughput |
|------|-------|-----------|--------|------------|
| benchmark_app | yolo11n | FP32 | CPU | ~97 FPS |
| benchmark_app | yolo11n | FP16 | CPU | ~98 FPS |
| benchmark_app | yolo11n | INT8 | CPU | **~243 FPS** |
| yolo_test (OAAX) | yolo11n | FP32 | CPU | ~94 FPS |
| yolo_test (OAAX) | yolo11n | FP16 | CPU | ~96 FPS |
| yolo_test (OAAX) | yolo11n | INT8 | CPU | ~229 FPS |

The OAAX runtime matches `benchmark_app` within ~2% for FP32/FP16 and ~5% for INT8. The gap is the cost of the mandatory output `memcpy` in the OAAX abstraction layer (~2.8 MB per inference). `benchmark_app` reads tensors in-place with no copy.

## Pre-built OAAX artifacts

If you're interested in using the OAAX toolchain and runtime without building them, you can find them in the
[contributions](https://github.com/oaax-standard/contributions) repository.
Additionally, you can find a diverse set of examples and applications of using the OAAX runtime in the
[examples](https://github.com/oaax-standard/examples) repository.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
