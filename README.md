# intel-acceleration

This folder contains the source code of the shared library and the Docker image that can be used by AI application developers to benefit from the acceleration offered by Intel CPU, GPU and NPU on x86_64 machines.

> To learn how to deploy on Intel, please check out the technical docs at [https://docs.oaax.org/Intel/](https://docs.oaax.org/Intel/)

## Repository structure

- [conversion-toolchain/](conversion-toolchain): OAAX conversion toolchain — converts ONNX models to OpenVINO IR (FP32/FP16/INT8).
- [runtime-library/](runtime-library): OAAX runtime library — C++ shared library for OpenVINO inference on Intel CPU/GPU/NPU.
- [tests/](tests): All tests. Conversion unit tests, YOLO integration tests, Docker tests, and C++ runtime tests.
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

Converts YOLOv8n and YOLOv11n to FP32/FP16/INT8 OpenVINO IR (cached in `tests/compiled_models/`), then runs all conversion tests.

```bash
bash scripts/stage1_compile.sh
```

Runs: `tests/test_conversion.py` (toolchain unit tests) + `tests/test_yolo_integration.py` (IR validation).

### Stage 2 — Benchmark + runtime validation

Uses compiled models from Stage 1. Requires the C++ runtime to be built.

```bash
bash scripts/stage2_run.sh [--devices CPU,GPU.0] [--duration 10] [--csv results.csv]
```

- Runs `benchmark_app` (latency hint, p95) across all models × precision variants × devices.
- Runs the C++ `yolo_test` binary (OAAX runtime, warmup + 30 runs, avg/min/p95 reported).
- With `--csv results.csv`: appends all results to a CSV for cross-run comparison.

### Individual pytest suites

```bash
pytest tests/test_conversion.py -v       # toolchain unit tests (no GPU needed)
pytest tests/test_yolo_integration.py -v # YOLO IR validation (uses cached models)
pytest tests/test_docker.py -v           # Docker tests (image must be built first)
```

### C++ runtime smoke test

```bash
cd runtime-library/build
./simple_test                             # no model needed
./yolo_test /path/to/model.xml [device] [--runs N] [--warmup N]
```

## Benchmark results (reference, Intel Core i7-13700K + Intel UHD 770)

| Model | Precision | Device | Avg latency | p95 | Throughput |
|-------|-----------|--------|-------------|-----|------------|
| YOLOv8n | FP32 | CPU | 13.8 ms | 14.5 ms | 71.6 FPS |
| YOLOv8n | FP16 | CPU | 13.5 ms | 14.0 ms | 72.9 FPS |
| YOLOv8n | INT8 | CPU | **5.2 ms** | 5.6 ms | **187 FPS** |
| YOLOv8n | FP32 | GPU.0 (iGPU) | ~12 ms | — | ~80 FPS |
| YOLOv8n | INT8 | GPU.0 (iGPU) | ~8 ms | — | ~116 FPS |

Numbers measured with `benchmark_app -hint latency`. INT8 is ~2.5× faster than FP32 on CPU via AVX-512 VNNI.

> **Note on `yolo_test` latency:** The C++ runtime uses an async queue (`send_input` → background thread → `receive_output`). The ~100 ms measured by `yolo_test` includes queue round-trip overhead, not just inference time. Use `benchmark_app` for pure inference latency.

## Pre-built OAAX artifacts

If you're interested in using the OAAX toolchain and runtime without building them, you can find them in the
[contributions](https://github.com/oaax-standard/contributions) repository.
Additionally, you can find a diverse set of examples and applications of using the OAAX runtime in the
[examples](https://github.com/oaax-standard/examples) repository.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to get started.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
