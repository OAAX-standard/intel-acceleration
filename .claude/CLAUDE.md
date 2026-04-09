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

The script builds the Docker image and runs `--help` to verify. Default image name is `openvino-converter:latest` unless overridden via `IMAGE_NAME`/`IMAGE_TAG` env vars.

### Runtime Library (C++)

Requires cross-compilation toolchain at `/opt/x86_64-unknown-linux-gnu-gcc-9.5.0` and OpenVINO installed (defaults to `/usr/local/lib/python3.10/dist-packages/openvino`).

```bash
cd runtime-library
OPENVINO_DIR=/path/to/openvino bash build-runtimes.sh  # OPENVINO_DIR is optional if default is correct
```

Output: `runtime-library/artifacts/X86_64/` and `runtime-library-X86_64.tar.gz`.

### Tests (all from project root)

```bash
# One-time setup
uv venv && source .venv/bin/activate
uv sync                                 # unit + conversion tests
uv sync --extra integration            # + YOLO integration tests

# Run
pytest tests/test_conversion.py -v                         # conversion unit tests
pytest tests/test_docker.py -v                             # Docker tests (image must be built)
pytest tests/test_yolo_integration.py -v                   # YOLO pipeline tests
pytest tests/test_conversion.py::TestClass::test_name -v   # single test

# Full E2E (toolchain + C++ runtime)
bash scripts/run_integration_tests.sh
bash scripts/run_integration_tests.sh --skip-runtime       # Python only
bash scripts/run_integration_tests.sh --device GPU         # test on GPU

# C++ runtime tests (after building runtime)
cd runtime-library/build && ./simple_test
cd runtime-library/build && ./yolo_test <model.xml>
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
- `deps/` — vendored: spdlog, concurrentqueue, nlohmann/json, c-utilities

**Inference architecture:** Async — `send_input()` enqueues to `input_tensors_queue`; background thread dequeues, runs `infer_request->infer()`, and enqueues results to `output_tensors_queue`; `receive_output()` dequeues results. Both queues use `moodycamel::ConcurrentQueue`.

**Runtime initialization args** (passed via `runtime_initialization_with_args`):

| Key | Default | Notes |
|-----|---------|-------|
| `device_type` | `"CPU"` | `"CPU"`, `"GPU"`, `"NPU"` |
| `num_threads` | `8` | 1–8, CPU only |
| `precision` | `"FP32"` | Informational |
| `log_level` | `info` | spdlog level int |
| `log_file` | `"runtime.log"` | Log file path |

### CMake Build

CMake requires `-DPLATFORM=X86_64 -DRUNTIME_VERSION=<ver> -DOPENVINO_DIR=<path>`. The build uses cross-compiler from `/opt/x86_64-unknown-linux-gnu-gcc-9.5.0`. OpenVINO `.so` libs are copied to `build/` automatically post-build.

---

## Key Constraints

- **Never break the public C API** in `runtime_core.hpp` — dynamically loaded by callers
- **Preserve exit codes** — automation depends on them
- **Toolchain input must be a zip** containing `model.onnx` (`.onnx` bare file also accepted per OAAX spec)
- **Runtime loads `.xml` files** (OpenVINO IR); the `.bin` file must be co-located

---

## Tech Stack

| Component | Tools |
|-----------|-------|
| Toolchain | Python 3.10+, OpenVINO Python API, NNCF, UV, Docker |
| Runtime | C++14/17, OpenVINO C++ API, spdlog, moodycamel::ConcurrentQueue, CMake |
| Testing | pytest + uv (Python), C++ binaries (runtime) |

---

## Reference Files

- `.claude/PLAN.md` — current status, phase goals, decision log
- `.claude/SKILLS.md` — implementation patterns and code examples
- `.claude/FINAL_SUMMARY.md` — phase completion summary
