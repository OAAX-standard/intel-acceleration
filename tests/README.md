# Tests

End-to-end and unit tests for the Intel acceleration runtime and conversion toolchain.

---

## Prerequisites

- Python 3.10+, [uv](https://github.com/astral-sh/uv)
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

# YOLO integration tests (convert + infer via Python API)
pytest tests/test_yolo_integration.py -v
```

---

## End-to-end two-stage benchmark

### Stage 1 — convert models to OpenVINO IR

Runs conversion unit tests and compiles YOLO models. Output lands in `tests/compiled_models/`.

```bash
python tests/stage1.py
```

### Stage 2 — benchmark and runtime validation

Requires `tests/compiled_models/` to be populated by Stage 1.

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
| `MODELS` | `yolo26s_320` | Space-separated model name filter (empty = all) |
| `RUNS` | `2000` | Inference runs per config |
| `WARMUP` | `100` | Warm-up runs before measurement |
| `CSV` | _(none)_ | Path to write CSV results |
| `SKIP_BENCH` | `0` | Set `1` to skip `benchmark_app` section |
| `SKIP_RUNTIME` | `0` | Set `1` to skip `yolo_test` section |

---

## C++ runtime tests

Build the runtime library first (`bash runtime-library/build-runtimes.sh`), then:

```bash
bash tests/runtime/build-tests.sh

# Smoke test (no model required)
./tests/runtime/build/simple_test

# YOLO inference test
./tests/runtime/build/yolo_test <model.zip> [device] [--runs N] [--warmup N] [--perf-hint latency|throughput]

# Multi-model test
./tests/runtime/build/multi_model_test <model1.zip> [model2.zip]
```

---

## Batch-size throughput sweep

Exports a `.pt` model → ONNX at each batch size, converts to IR, and benchmarks.
Compiled IR models are cached in `tests/compiled_models/_batch_sweep/` — re-runs skip re-export.

```bash
python tests/benchmark_batch_sweep.py \
  --model yolov8n \
  --device CPU \
  --batches 1,2,4,8 \
  --runs 100 \
  --duration 15
```

The `.pt` file must exist in the repo root (e.g. `yolov8n.pt`).
