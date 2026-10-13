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

Intel Core i7-12700K, CPU, batch=1, `throughput` hint, YOLOX test models. `benchmark_app` runs 10 s per
configuration; the OAAX runtime numbers come from `detector_test` (300 runs, in-flight depth = the runtime's
infer-request count). Median of 3 repetitions; spread between repetitions was ≤ 3%.

| Model | Precision | benchmark_app | OAAX runtime | Delta |
|-------|-----------|--------------|--------------|-------|
| yolox_nano | FP32 | ~422 FPS | ~395 FPS | -7% |
| yolox_nano | FP16 | ~429 FPS | ~405 FPS | -6% |
| yolox_nano | INT8 | **~664 FPS** | **~606 FPS** | -9% |
| yolox_tiny | FP32 | ~117 FPS | ~115 FPS | -2% |
| yolox_tiny | FP16 | ~120 FPS | ~119 FPS | -1% |
| yolox_tiny | INT8 | **~351 FPS** | **~337 FPS** | -4% |
| yolox_s | FP32 | ~30 FPS | ~29 FPS | -0% |
| yolox_s | FP16 | ~30 FPS | ~30 FPS | -0% |
| yolox_s | INT8 | **~92 FPS** | **~91 FPS** | -1% |

- FP32 and FP16 perform the same: this CPU has no native FP16 ALU, so OpenVINO computes FP16 as FP32 internally.
  INT8 is ~3× faster for tiny and s, and ~1.6× for nano.
- The runtime stays within ~1–9% of `benchmark_app`; the gap grows with model speed, which points to a per-inference
  cost at the C API boundary (input/output copies) rather than to the inference itself.
- Keep enough requests in flight: the runtime sizes its infer-request pool from the CPU (18 on this machine) and a
  caller that submits fewer starves fast models. With only 5 in flight, INT8 nano/tiny reached about half of the
  numbers above. `detector_test` now defaults to the runtime's request count.

Reproduce with `python tests/stage2.py --devices CPU --duration 10 --runs 300 --csv results.csv` on an otherwise
idle machine. Multi-GPU hosts should use `device_type=GPU` with `perf_hint=cumulative_throughput`
(not re-benchmarked with these models).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
