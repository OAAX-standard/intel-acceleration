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

## Comprehensive benchmark results (i7-12700K, UHD 770 iGPU + RTX A4000 dGPU)

Throughput hint, batch=1, 100 runs / 15s per config.

### benchmark_app (no H2D/D2H — GPU tensors stay on device)

| Model | Prec | CPU | GPU.0 (iGPU) | GPU.1 (dGPU) | GPU ("auto") |
|-------|------|-----|--------------|--------------|--------------|
| yolov8n | FP32 | ~85 FPS | ~79 FPS | ~56 FPS | ~79 FPS |
| yolov8n | INT8 | ~240 FPS | ~116 FPS | ~67 FPS | ~116 FPS |
| yolo11n | FP32 | ~98 FPS | ~82 FPS | ~70 FPS | ~82 FPS |
| yolo11n | INT8 | ~243 FPS | ~113 FPS | ~84 FPS | ~113 FPS |

Note: `benchmark_app -d GPU` does NOT use MULTI — it routes to fastest single GPU.

### yolo_test (with H2D/D2H per inference; GPU = our MULTI auto-detection)

| Model | Prec | CPU | GPU.0 (iGPU) | GPU.1 (dGPU) | GPU (MULTI) |
|-------|------|-----|--------------|--------------|-------------|
| yolov8n | FP32 | ~84 FPS | ~72 FPS | ~55 FPS | ~117 FPS |
| yolov8n | INT8 | ~226 FPS | ~101 FPS | ~60 FPS | ~152 FPS |
| yolo11n | FP32 | ~97 FPS | ~75 FPS | ~67 FPS | ~130 FPS |
| yolo11n | INT8 | ~235 FPS | ~103 FPS | ~71 FPS | ~156 FPS |

### Key findings
- **MULTI GPU delivers 1.6× throughput** vs single iGPU; benchmark_app "GPU" doesn't use MULTI
- **CPU INT8 beats MULTI GPU INT8** on this machine (~226 vs ~152 FPS) — optimal target is CPU
- **H2D/D2H gap** on single iGPU: ~9-13% (higher at INT8 speeds); dGPU gap is lost in noise
- **stage2.py regex bug** fixed (commit af3d9279): `Avg latency:` pattern updated

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

## CI gotchas on `oaax-v2` (2026-08-05)

- `.github/workflows/build.yml` and `lint.yml` trigger only on `push`/`pull_request` targeting
  **`main`**. A PR from `oaax-v2` into `oaax-v2` (or any non-`main` base) never runs them. In
  practice CI only actually runs because there's a long-lived `oaax-v2` → `main` PR ("feat: OAAX v2
  runtime interface") that re-triggers on every push to `oaax-v2` — check `gh run list --branch
  oaax-v2` against *that* PR's checks, not your own PR's checks, to see real build/lint status.
- `Lint Dockerfiles` (hadolint-docker) pulls `ghcr.io/hadolint/hadolint:latest` — an unpinned tag.
  It passed on 2026-07-17 and started failing by 2026-08-05 on the exact same `conversion-toolchain/
  Dockerfile:67` line (`DL3066`) with no local change to that file — the image drifted stricter
  upstream. Don't assume a red Dockerfile lint means your diff broke something; check whether the
  file you're touching is even the one it's complaining about first.
- `integration-tests` (Stage 1 model-conversion pytest suite) was hitting the 30-minute job timeout
  mid-run while still passing everything it had gotten through — not a hang, just insufficient
  margin. Bumped `integration-tests` and `integration-tests-windows` (which depends on it) to 60m.
- **`runtime_cleanup()` must stay safe to call repeatedly** (each followed by another
  `runtime_init()`, same loaded module, no unload in between) — see
  `.claude/memory/project_windows_runtime_fixes.md`'s 2026-08-05 entry for the `ov::shutdown()`
  regression this caused (AIMP-1477) and why `ov::shutdown()` can never go back in there.
