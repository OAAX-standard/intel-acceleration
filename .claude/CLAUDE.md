# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OAAX implementation for Intel hardware (CPU, GPU, NPU) using OpenVINO. Two components:

1. **Conversion Toolchain** — Docker container converting ONNX models to OpenVINO IR
2. **Runtime Library** — C++ shared library for inference via OpenVINO native API

Model flow: `ONNX → [Toolchain] → OpenVINO IR (.xml + .bin) → [Runtime] → Inference`

- Repository: https://github.com/OAAX-standard/intel-acceleration
- Version: see `VERSION` file
- OAAX interface spec: `runtime-library/include/oaax_runtime.h` (v2)

---

## Build Commands

### Conversion Toolchain (Docker)

```bash
cd conversion-toolchain
IMAGE_NAME=oaax-intel-toolchain bash build-toolchain.sh
```

### Runtime Library (C++)

Requires cross-compilation toolchain at `/opt/x86_64-unknown-linux-gnu-gcc-9.5.0` and OpenVINO installed (defaults to `/opt/intel/openvino/runtime`).

```bash
cd runtime-library
OPENVINO_DIR=/path/to/openvino bash build-runtimes.sh
```

Output: `runtime-library/artifacts/X86_64/` and `runtime-library-X86_64.tar.gz`.

### Tests (all from project root)

```bash
# One-time setup
uv venv && source .venv/bin/activate
uv sync --extra integration --extra quantization

# Python tests
pytest tests/test_conversion.py -v
pytest tests/test_docker.py -v           # image must be built
pytest tests/test_yolo_integration.py -v

# Two-stage E2E
python tests/stage1.py
python tests/stage2.py [--devices CPU,GPU.0] [--duration 10] [--csv results.csv]

# Batch-size throughput sweep (exports .pt → ONNX with explicit batch, converts to IR, benchmarks)
python tests/benchmark_batch_sweep.py [--model yolov8n] [--device CPU] [--batches 1,2,4,8]
# IR models cached in tests/compiled_models/_batch_sweep/ — re-runs skip export

# C++ runtime tests (build after running build-runtimes.sh)
bash tests/runtime/build-tests.sh
cd tests/runtime/build && ./simple_test
cd tests/runtime/build && ./yolo_test <model.zip> [device] [--runs N] [--warmup N] [--perf-hint latency|throughput]
cd tests/runtime/build && ./multi_model_test <model.zip> [model2.zip]
```

---

## Architecture

### Toolchain (`conversion-toolchain/`)

- `conversion_toolchain/main.py` — CLI entrypoint, handles all exit codes
- `conversion_toolchain/utils.py` — OpenVINO `convert_model()`, FP16/INT8, zip I/O
- `conversion_toolchain/quantization.py` — NNCF INT8 quantization
- `conversion_toolchain/config.py` — `OptimizationConfig` (FP16 compression, quantization, preprocessing)
- `conversion_toolchain/logger.py` — JSON structured logging (`Logs` class)

Input: `.onnx` or `.zip` bundle (may contain `model.onnx`, `config.json`, `calibration/`). Output: `.zip` of OpenVINO IR + `logs.json`.

**config.json `preprocessing` keys** (all optional, baked into IR via PPP before quantization):

| Key | Default | Notes |
|-----|---------|-------|
| `input_dtype` | `null` | Model input boundary type: `"u8"`, `"f16"`, `"f32"`. Lets callers feed raw pixels without manual conversion. |
| `mean_values` | `null` | Per-channel means to subtract (after cast to f32), e.g. `[123.675, 116.28, 103.53]` |
| `scale_values` | `null` | Per-channel divisors, e.g. `[255.0, 255.0, 255.0]` to normalise u8→[0,1] |

Example — bake u8 input + ÷255 normalisation into the IR:
```json
{ "preprocessing": { "input_dtype": "u8", "scale_values": [255.0, 255.0, 255.0] } }
```
When quantization is also enabled, calibration images are fed in the configured format so NNCF statistics stay consistent.

**config.json `advanced` keys** (all optional):

| Key | Default | Notes |
|-----|---------|-------|
| `batch_size` | `1` | Fix the batch dimension to a specific integer at conversion time. Mutually exclusive with `dynamic_batch`. |
| `dynamic_batch` | `false` | Set batch dimension to `-1` (dynamic) in the IR so any batch size is accepted at inference. Mutually exclusive with `batch_size > 1`. Combined with quantization: NNCF still calibrates with batch=1. |

Example — keep batch dynamic:
```json
{ "advanced": { "dynamic_batch": true } }
```

**Exit codes:** 0=success, 1=file not found, 2=invalid input, 3=conversion failed, 4=I/O error, 255=unexpected.

### Runtime Library (`runtime-library/`)

- `src/runtime_core.cpp` — OpenVINO inference, async queue-based execution
- `src/runtime_utils.cpp` — type mapping between `TensorElementType` and `ov::element::Type`
- `include/oaax_runtime.h` — public C API (OAAX v2 interface)
- `include/runtime_utils.hpp` — internal utilities
- `deps/` — vendored: spdlog, concurrentqueue, c-utilities

**Inference architecture (v2):**
- Per-model manager thread dequeues inputs, acquires a free `ov::InferRequest` slot, calls `start_async()`
- Completion callbacks malloc output `Tensors`, copy data, post to a single global output queue
- Caller retrieves from global queue via `runtime_retrieve_output(int *model_id, Tensors**, timeout_ms)`
- `timeout_ms=0`: non-blocking; `<0`: block forever; `>0`: timed wait via `sem_timedwait`
- `Tensors.id` (set by caller on input) is echoed on output for request correlation
- FIFO ordering is NOT guaranteed across models

**Multi-GPU auto-detection:**
When `device_type="GPU"` and multiple GPU devices are present, the runtime automatically constructs a `MULTI:GPU.0,GPU.1,...` device string. Explicit device strings are passed through unchanged. Use `perf_hint=cumulative_throughput` with MULTI for best aggregate FPS.

**Runtime config** (`runtime_init` Config keys, overridable per-model in `ModelConfig.config`):

| Key | Default | Notes |
|-----|---------|-------|
| `device_type` | `"CPU"` | `"CPU"`, `"GPU"` (auto-MULTI if multiple GPUs), `"GPU.0"`, `"NPU"`, `"AUTO"` (starts on CPU, migrates to best accelerator), `"AUTO:GPU,CPU"` (priority order) |
| `perf_hint` | `"latency"` | `"latency"` / `"throughput"` / `"cumulative_throughput"`. Use `cumulative_throughput` with MULTI. Do **not** mix with `inference_num_threads`. |
| `log_level` | `"2"` (info) | spdlog level int as string |
| `log_file` | `"runtime.log"` | Log file path |
| `cache_dir` | `"."` (CWD) | OpenVINO compiled-model cache. Set to `""` to disable. |
| `log_stdout` | `"0"` | Set to `"1"` to also print logs to stdout (default: file only). |
| `max_queue_size` | `"100"` | Max pending items in the input and output queues. New inputs are rejected with a warning when either queue reaches this limit. Set to `"0"` to disable. |
| `num_streams` | `"0"` (auto) | Number of parallel inference streams. `"0"` lets OpenVINO decide. Recommended `"4"` for discrete Intel GPU with `throughput` hint. No effect with `latency` hint. |
| `auto_batch_size` | `"0"` (disabled) | Wraps the device in OpenVINO's `BATCH` pseudo-device to transparently aggregate concurrent requests into hardware batches. Set to the desired batch size (e.g. `"8"`). GPU only — ignored with a warning on CPU. |

**Input dtype contract:** the runtime reads the IR's input element type from the compiled model and validates every enqueue call against it. The toolchain auto-bakes the correct type — FP32 models expect `f32`, FP16 models expect `f16`, INT8 models expect `u8`. Mismatches are rejected immediately with `RUNTIME_STATUS_INVALID_ARGUMENT`.

**Note:** do not set `ov::inference_num_threads` or `INFERENCE_NUM_THREADS` alongside a `perf_hint` — the hint owns thread scheduling internally. `num_streams` is the exception and can be combined with `throughput` hint for GPU tuning.

**Output ownership:** caller owns `Tensors*` returned by `runtime_retrieve_output` and must free
`tensors[i].name`, `tensors[i].shape`, `tensors[i].data`, `tensors`, and the `Tensors` struct itself.

**Batch size behaviour (verified on i7-12700K):**
- **CPU**: Batch=1 is optimal with `throughput` hint.
- **iGPU (Intel)**: Minor plateau at batch 4-8; batch=1 is still near-optimal.
- **dGPU via OpenCL (NVIDIA)**: Degrades sharply beyond batch=2.
- For batch > 1, use `tests/benchmark_batch_sweep.py` which exports from `.pt` with explicit batch dims.

### CMake Build

CMake requires `-DPLATFORM=X86_64 -DRUNTIME_VERSION=<ver> -DOPENVINO_DIR=<path>`. The build uses cross-compiler from `/opt/x86_64-unknown-linux-gnu-gcc-9.5.0`. OpenVINO `.so` libs are copied to `build/` automatically post-build.

---

## Key Constraints

- **Never break the public C API** in `oaax_runtime.h` — dynamically loaded by callers
- **Preserve exit codes** — automation depends on them
- **Toolchain input must be a zip** containing `model.onnx` (`.onnx` bare file also accepted per OAAX spec)
- **Runtime loads `.xml` files** (OpenVINO IR); the `.bin` file must be co-located
- **Do not set `ov::inference_num_threads` alongside a performance hint** — it constrains OpenVINO's internal scheduler. `perf_hint` alone is sufficient.

---

## Tech Stack

| Component | Tools |
|-----------|-------|
| Toolchain | Python 3.10+, OpenVINO Python API, NNCF, UV, Docker |
| Runtime | C++14/17, OpenVINO C++ API, spdlog, moodycamel::ConcurrentQueue, CMake |
| Testing | pytest + uv (Python), C++ binaries (runtime) |

---

## Reference Files

- `.claude/STATUS.md` — current status, phase goals, decision log
- `.claude/SKILLS.md` — implementation patterns and code examples
