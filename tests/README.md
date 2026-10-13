# Tests

End-to-end and unit tests for the Intel acceleration runtime and conversion toolchain.

---

## Prerequisites

- Python 3.13+, [uv](https://github.com/astral-sh/uv)
- Docker (for toolchain tests) — build the image first:
  ```bash
  IMAGE_NAME=oaax-intel-toolchain bash conversion-toolchain/build-toolchain.sh
  ```
- OpenVINO runtime installed (defaults to `/opt/intel/openvino/runtime`)
- Cross-compilation toolchain at `/opt/x86_64-unknown-linux-gnu-gcc-9.5.0` (for C++ runtime tests)
- `patchelf` (for C++ runtime tests — RPATH won't be fixed without it)

---

## Setup (one time)

From the **repo root**:

```bash
uv venv
source .venv/bin/activate
uv sync --extra integration --extra quantization
```

---

## Python unit tests

```bash
# Conversion toolchain tests (no Docker required)
pytest tests/test_conversion.py -v

# Docker image tests
pytest tests/test_docker.py -v

# YOLOX integration tests (convert + infer via Python API)
pytest tests/test_detector_integration.py -v
```

---

## End-to-end two-stage benchmark

### Stage 1 — convert models to OpenVINO IR

Runs conversion unit tests and compiles YOLOX models (nano / tiny / s; native, 320x320 and batch=4 variants). Output lands in `tests/compiled_models/`.

```bash
python tests/stage1.py
```

### Stage 2 — benchmark and runtime validation

Requires `tests/compiled_models/` to be populated by Stage 1.

Stage 2 builds all C++ test binaries, benchmarks every compiled model with `detector_test`, then runs the runtime API
tests (`simple_test`, `lifecycle_test`, `output_validation_test`, `multi_model_test`) against `yolox_nano`/`yolox_tiny`
FP32. It exits non-zero if any of them fails.

**Quick run (defaults):**

```bash
python tests/stage2.py
```

**With options:**

```bash
python tests/stage2.py \
  --devices CPU,GPU.0 \
  --perf-hints latency,throughput \
  --runs 300 --warmup 5 \
  --duration 10 \
  --csv results.csv
```

**Using the shell wrapper** (env-var driven, convenient for CI):

```bash
# Defaults: CPU device, throughput hint, 2000 runs, INT8 precision
bash tests/run_stage2.sh

# Override via env vars
DEVICES="CPU GPU.0" PRECISIONS="FP32 FP16" RUNS=500 bash tests/run_stage2.sh
```

Key env vars for `run_stage2.sh`:

| Variable | Default | Description |
|---|---|---|
| `DEVICES` | `CPU` | Space or comma-separated device list |
| `PERF_HINTS` | `throughput` | `latency` / `throughput` / `cumulative_throughput` |
| `PRECISIONS` | `INT8` | `FP32`, `FP16`, `INT8` |
| `MODELS` | `yolox_nano_320` | Space-separated model name filter (empty = all) |
| `RUNS` | `2000` | Inference runs per config |
| `WARMUP` | `100` | Warm-up runs before measurement |
| `CSV` | _(none)_ | Path to write CSV results |
| `SKIP_BENCH` | `0` | Set `1` to skip `benchmark_app` section |
| `SKIP_RUNTIME` | `0` | Set `1` to skip `detector_test` section |

---

## C++ runtime tests

Build the runtime library first (`bash runtime-library/build-runtimes.sh`), then:

```bash
bash tests/runtime/build-tests.sh

# Smoke test (no model required)
./tests/runtime/build/simple_test

# Detector inference test (INT8 IRs take u8 input, FP16 take f16 — pass --input-dtype; --imgsz 416 for nano/tiny)
./tests/runtime/build/detector_test <model.zip> [device] [--runs N] [--warmup N] [--imgsz N] [--input-dtype f32|u8|f16] [--perf-hint latency|throughput]

# Multi-model test
./tests/runtime/build/multi_model_test <model1.zip> [model2.zip] [--imgsz N]
```

---

## Batch-size throughput sweep

Builds the official YOLOX ONNX at each batch size, converts to IR, and benchmarks.
Compiled IR models are cached in `tests/compiled_models/_batch_sweep/` — re-runs skip the build.

```bash
python tests/benchmark_batch_sweep.py \
  --model yolox_nano \
  --device CPU \
  --batches 1,2,4,8 \
  --runs 100 \
  --duration 15
```

`--model` is one of `yolox_nano`, `yolox_tiny`, `yolox_s`.

---

## Test models

Detector tests use the official [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) ONNX models (Apache-2.0), downloaded
and SHA-256-verified on first use, then adapted by `tests/models.py` to the batch size and resolution a test needs.
Calibration uses 128 COCO images listed in `tests/calibration_images.txt`. Nothing is committed; no AGPL-licensed
models or packages are used.
