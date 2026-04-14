# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OAAX implementation for Intel hardware (CPU, GPU, NPU) using OpenVINO. Two components:

1. **Conversion Toolchain** — Docker container converting ONNX models to OpenVINO IR
2. **Runtime Library** — C++ shared library for inference via OpenVINO native API

Model flow: `ONNX → [Toolchain] → OpenVINO IR (.xml + .bin) → [Runtime] → Inference`

- Repository: https://github.com/OAAX-standard/intel-acceleration
- Version: see `VERSION` file
- OAAX interface spec: `runtime-library/include/runtime_core.hpp`

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

# C++ runtime tests (after building runtime)
cd runtime-library/build && ./simple_test
cd runtime-library/build && ./yolo_test <model.xml> [device] [--runs N] [--warmup N] [--perf-hint latency|throughput]
```

---

## Architecture

### Toolchain (`conversion-toolchain/`)

- `conversion_toolchain/main.py` — CLI entrypoint, handles all exit codes
- `conversion_toolchain/utils.py` — OpenVINO `convert_model()`, FP16/INT8, zip I/O
- `conversion_toolchain/quantization.py` — NNCF INT8 quantization
- `conversion_toolchain/config.py` — `OptimizationConfig` (FP16 compression, quantization)
- `conversion_toolchain/logger.py` — JSON structured logging (`Logs` class)

Input: `.onnx` or `.zip` bundle (may contain `model.onnx`, `config.json`, `calibration/`). Output: `.zip` of OpenVINO IR + `logs.json`.

**Exit codes:** 0=success, 1=file not found, 2=invalid input, 3=conversion failed, 4=I/O error, 255=unexpected.

### Runtime Library (`runtime-library/`)

- `src/runtime_core.cpp` — OpenVINO inference, async queue-based execution
- `src/runtime_utils.cpp` — type mapping between `tensor_data_type` and `ov::element::Type`
- `include/tensors_struct.h` — OAAX C tensor struct (names, ranks, shapes, data_types, data)
- `include/runtime_core.hpp` — public C API (OAAX interface)
- `deps/` — vendored: spdlog, concurrentqueue, c-utilities

**Inference architecture:**
- Single manager thread dequeues inputs, acquires a free `ov::InferRequest` slot, and calls `start_async()`
- Completion callbacks post results to `output_tensors_queue` and return the slot to the pool
- Workers write output into a pre-allocated buffer pool (no malloc on hot path)
- Results enqueue to `output_tensors_queue`; caller polls via `receive_output()`
- FIFO ordering is NOT guaranteed when N > 1

**Multi-GPU auto-detection:**
When `device_type="GPU"` and multiple GPU devices are present, the runtime automatically constructs a `MULTI:GPU.0,GPU.1,...` device string and compiles for all GPUs. OpenVINO's MULTI plugin then distributes inference requests across them. Explicit device strings (e.g. `"GPU.0"`, `"MULTI:GPU.0,GPU.1"`) are passed through unchanged.

Benchmark results on Intel Core i7-12700K (UHD 770 iGPU + RTX A4000 dGPU), YOLOv8n FP32:

| Config | Hint | Infer requests | Throughput |
|--------|------|---------------|------------|
| GPU.0 (iGPU only) | latency | 1 | ~68 FPS |
| GPU.1 (dGPU only) | latency | 1 | ~52 FPS |
| GPU (MULTI auto) | latency | 2 | ~68 FPS (routes to fastest) |
| GPU (MULTI auto) | cumulative_throughput | 8 | **~114 FPS** |
| GPU (MULTI auto) | throughput | 8 | ~112 FPS |

Note: iGPU outperforms the RTX A4000 here because OpenVINO's GPU plugin is optimised for Intel hardware; NVIDIA runs via a generic OpenCL path.

**Runtime initialization args** (passed via `runtime_initialization_with_args`):

| Key | Default | Notes |
|-----|---------|-------|
| `device_type` | `"CPU"` | `"CPU"`, `"GPU"` (auto-MULTI if multiple GPUs found), `"GPU.0"`, `"NPU"` |
| `perf_hint` | `"latency"` | `"latency"` / `"throughput"` / `"cumulative_throughput"` — passed as `ov::hint::performance_mode` at compile time; worker count is inferred automatically via `OPTIMAL_NUMBER_OF_INFER_REQUESTS`. Use `cumulative_throughput` with multi-GPU for best aggregate FPS. |
| `log_level` | `2` (info) | spdlog level int |
| `log_file` | `"runtime.log"` | Log file path |

**Key public API additions beyond the base OAAX spec:**
- `runtime_return_output(tensors_struct*)` — return a received output buffer to the pool instead of calling `deep_free_tensors_struct`. Callers SHOULD use this for correct pool reuse. Falls back to `deep_free_tensors_struct` when the pool is not active (dynamic shapes).

**Output buffer pool:**
After model loading, if all output shapes are static, the runtime pre-allocates `actual_requests × 4` `tensors_struct` objects. Workers memcpy into these — no malloc/free on the hot path. Eliminates mmap/munmap syscalls from large allocation calls, which is significant at INT8 speeds (~255 FPS).

**Batch size behaviour (verified on i7-12700K):**
- **CPU**: Batch=1 is optimal with `throughput` hint — OpenVINO already runs multiple concurrent InferRequests across all cores; batching serializes work without adding parallelism.
- **iGPU (Intel)**: Minor throughput plateau at batch 4-8; batch=1 is still near-optimal.
- **dGPU via OpenCL (NVIDIA)**: Degrades sharply beyond batch=2 — OpenCL kernel overhead on NVIDIA not optimized in OpenVINO. An Intel Xe dGPU would differ.
- For batch > 1, use `tests/benchmark_batch_sweep.py` which exports from `.pt` with explicit batch dims (simple IR reshape fails on YOLO due to hardcoded DFL reshape ops).

### CMake Build

CMake requires `-DPLATFORM=X86_64 -DRUNTIME_VERSION=<ver> -DOPENVINO_DIR=<path>`. The build uses cross-compiler from `/opt/x86_64-unknown-linux-gnu-gcc-9.5.0`. OpenVINO `.so` libs are copied to `build/` automatically post-build.

---

## Key Constraints

- **Never break the public C API** in `runtime_core.hpp` — dynamically loaded by callers
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
