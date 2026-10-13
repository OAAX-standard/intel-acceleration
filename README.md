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
tests/                  Conversion tests, YOLOX integration tests, runtime benchmarks
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

# Stage 1 — compile YOLOX models to FP32/FP16/INT8 IR and run conversion tests
python tests/stage1.py

# Stage 2 — benchmark with benchmark_app + detector_test across all variants and devices
python tests/stage2.py [--devices CPU,GPU.0] [--duration 10] [--csv results.csv]
```

Individual test suites:

```bash
pytest tests/test_conversion.py -v             # toolchain unit tests
pytest tests/test_detector_integration.py -v   # YOLOX IR validation
pytest tests/test_quantization_accuracy.py -v  # INT8/FP16 accuracy vs FP32 baseline
pytest tests/test_docker.py -v                 # Docker image tests (image must be built)
pytest -m slow tests/test_memory.py           # memory leak tests (~30 min)
```

## Benchmarks

Throughput numbers are produced by Stage 2 on the YOLOX test models (`yolox_nano`, `yolox_tiny`, `yolox_s`);
run it on an otherwise idle machine and compare `benchmark_app` with the OAAX runtime (`detector_test`):

```bash
python tests/stage2.py --devices CPU,GPU --duration 10 --csv results.csv
```

Earlier measurements on an Intel Core i7-12700K (CPU, iGPU UHD 770 + dGPU RTX A4000), taken with different test models and
not yet re-verified on YOLOX, showed:

- FP32 and FP16 perform the same on CPUs without a native FP16 ALU — OpenVINO computes FP16 as FP32 internally.
  Only INT8 benefits from AVX2/VNNI (~2.5×).
- The OAAX runtime lands within ~5% of `benchmark_app`; the gap is producer/consumer thread overhead at the C API boundary.
- On multi-GPU hosts, `device_type=GPU` with `perf_hint=cumulative_throughput` gives the best aggregate FPS.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
