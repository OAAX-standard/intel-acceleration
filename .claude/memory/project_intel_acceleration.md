---
name: intel-acceleration project state
description: Current architecture, test framework, artifact packaging, and benchmark results for the OAAX Intel acceleration project
type: project
---
OAAX runtime + conversion toolchain for Intel hardware using OpenVINO 2026.1.0 + NNCF 2.19.0. Version: 1.2.0
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
| `device_type` | `"CPU"` | `"CPU"`, `"GPU"` (auto-MULTI if multiple GPUs detected), `"GPU.0"`, `"NPU"` |
| `perf_hint` | `"latency"` | `"latency"` / `"throughput"` / `"cumulative_throughput"`. Use `cumulative_throughput` with multi-GPU for best aggregate FPS. |
| `log_level` | `2` (info) | spdlog level int |
| `log_file` | `"runtime.log"` | Log file path |

## Multi-GPU behaviour

When `device_type="GPU"` and multiple GPUs are found, the runtime auto-constructs `MULTI:GPU.0,GPU.1,...`.
- `latency` hint: routes to fastest single device (no multi-GPU benefit)
- `cumulative_throughput` hint: saturates all GPUs; ~95% of theoretical sum

Benchmark on i7-12700K (UHD 770 iGPU + RTX A4000 dGPU), YOLOv8n FP32:

| Config | Hint | Throughput |
|--------|------|------------|
| GPU.0 only | latency | ~68 FPS |
| GPU.1 only | latency | ~52 FPS |
| MULTI auto | latency | ~68 FPS |
| MULTI auto | cumulative_throughput | ~114 FPS |

Note: iGPU outperforms RTX A4000 under OpenVINO because Intel's GPU plugin targets Intel hardware; NVIDIA runs via generic OpenCL.

## GPU vs benchmark_app throughput gap

`benchmark_app` keeps tensors in GPU memory — no H2D/D2H per inference.
Our runtime uses caller-provided CPU buffers, adding per-inference transfers:
- iGPU (shared memory): ~1 ms overhead (cache coherency) → ~7% gap
- dGPU (PCIe): ~0.4 ms overhead, but inference is ~37 ms → gap lost in noise

## Benchmark results (CPU, throughput hint, yolo_test)

| Model | Precision | yolo_test FPS | benchmark_app FPS | Gap |
|-------|-----------|---------------|-------------------|-----|
| yolov8n | FP32 | ~84.8 | ~84 | <1% |
| yolo11n | FP32 | ~97.7 | ~95.6 | <1% |
| yolo11n | INT8 | ~237 | ~244 | ~3% |

## Debug profiler

`runtime_profiler.hpp` — enabled with `OAAX_PROFILE=1` (auto-set for Debug builds).
Reports per-stage breakdown (input_wait, slot_wait, pool_wait, tensor_setup, inference, output_queue) in µs/inference on `runtime_destruction()`.
Zero overhead in Release builds.

## yolo_test CLI

```
./yolo_test <model.xml> [device] [--runs N] [--warmup N] [--perf-hint latency|throughput|cumulative_throughput]
```

Uses `max_in_flight=5` hardcoded. Calls `runtime_return_output()` (not `deep_free_tensors_struct`) to return pool buffers.
Average latency is skewed by queue depth with 1 infer slot — use **min latency** and **throughput** as the meaningful metrics.
