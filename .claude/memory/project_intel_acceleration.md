---
name: intel-acceleration project state
description: Current architecture, test framework, artifact packaging, and benchmark results for the OAAX Intel acceleration project
type: project
---
OAAX runtime + conversion toolchain for Intel hardware using OpenVINO 2026.1.0 + NNCF 2.19.0.
Repo: https://github.com/OAAX-standard/intel-acceleration  Branch: pure-openvino-implementation (open PR #17 — do NOT merge without Ayoub's approval)

**Why:** Production OAAX implementation for Intel CPU/GPU/NPU, replaces ONNX Runtime with native OpenVINO C++ API.
**How to apply:** Understand the two-component split (Python toolchain + C++ runtime) and the two-stage test workflow.

## Artifact packaging (CI, as of 2026-04-11)

**Linux** (S3: `runtimes/<branch>/INTEL/x86_64/Ubuntu/library.tar.gz`):
- `libRuntimeLibrary.so` + all OpenVINO `.so` libs (no unused frontends) + TBB `.so` libs
- All .so files have `$ORIGIN` RPATH set by `patchelf` in `build-runtimes.sh`
- ~86 MB total

**Windows** (S3: `runtimes/<branch>/INTEL/x86_64/Windows/library.tar.gz`):
- `RuntimeLibrary.dll` + `openvino.dll` + device plugins (CPU/GPU/NPU/auto/hetero) + `openvino_ir_frontend.dll` + TBB DLLs
- ~60 MB total
- Built on `windows-latest` runner via `build-runtime.bat` + MSVC

## Two-stage test workflow

**Stage 1** (`python tests/stage1.py`):
- Converts YOLO models (yolov8n, yolo11n) to FP32/FP16/INT8 IR → `tests/compiled_models/` (cached)
- Runs `tests/test_conversion.py` + `tests/test_yolo_integration.py`

**Stage 2** (`python tests/stage2.py [--devices CPU,GPU.0] [--csv results.csv]`):
- Step 1: `benchmark_app -hint throughput -latency_percentile 95` on all variants × devices
- Step 2: C++ `yolo_test` binary on all variants (FP32/FP16/INT8) × devices
- `--csv` appends rows: `timestamp,tool,model,variant,device,avg_ms,min_ms,p95_ms,throughput_fps`

## Runtime initialization args (key/value pairs via runtime_initialization_with_args)

| Key | Default | Notes |
|-----|---------|-------|
| `device_type` | `"CPU"` | `"CPU"`, `"GPU"`, `"NPU"` |
| `perf_hint` | `"latency"` | `"latency"` / `"throughput"` / `"cumulative_throughput"` |
| `num_requests` | (auto) | Always inferred from compiled model via `OPTIMAL_NUMBER_OF_INFER_REQUESTS`; not settable |
| `precision` | `"FP32"` | Informational only |
| `log_level` | `2` (info) | spdlog level int |
| `log_file` | `"runtime.log"` | Log file path |

## Benchmark results (yolo_test, throughput hint, CPU)

| Model | Precision | yolo_test FPS | benchmark_app FPS | Gap |
|-------|-----------|---------------|-------------------|-----|
| yolo11n | FP32 | ~100 | ~102 | <1% |
| yolo11n | FP16 | ~95 | ~102 | ~7% |
| yolo11n | INT8 | ~231 | ~255 | ~9% |

INT8 gap is dominated by memcpy (2.8 MB output × ~231/s ≈ 650 MB/s). Further reduction requires zero-copy API change.

## yolo_test CLI

```
./yolo_test <model.xml> [device] [--runs N] [--warmup N] [--perf-hint latency|throughput]
```

Uses `max_in_flight=10` and calls `runtime_return_output()` (not `deep_free_tensors_struct`) to return pool buffers.
