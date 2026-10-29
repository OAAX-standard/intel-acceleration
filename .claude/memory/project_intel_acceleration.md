---
name: intel-acceleration project state
description: Current architecture, test framework, artifact packaging, and benchmark results for the OAAX Intel acceleration project
type: project
---
OAAX runtime + conversion toolchain for Intel hardware using OpenVINO 2026.1.0 + NNCF 2.19.0. Version: see `VERSION` file
Repo: https://github.com/OAAX-standard/intel-acceleration  Branch: `oaax-v2` (OAAX v2 interface), merging into `main`

**Why:** Production OAAX implementation for Intel CPU/GPU/NPU, replaces ONNX Runtime with native OpenVINO C++ API.
**How to apply:** Understand the two-component split (Python toolchain + C++ runtime) and the two-stage test workflow.

## Artifact packaging (CI, as of 2026-04-11)

Since 2026-10 CI uploads to S3 only on a `vX.Y.Z` tag (`<VERSION>/` written once + `latest/`) or from a PR (`<branch>/`);
`main` pushes upload nothing, and the tests use the GitHub build artifacts. The S3 paths below are the keys of those uploads.

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
- Converts YOLOX models (yolox_nano/tiny/s; native, 320 and batch=4 variants) to FP32/FP16/INT8/INT8_NPU IR → `tests/compiled_models/` (cached). Models are the official Apache-2.0 YOLOX ONNX, adapted by `tests/models.py` — never use AGPL models/packages (ultralytics) in this repo
- Runs `tests/test_conversion.py` + `tests/test_detector_integration.py`

**Stage 2** (`python tests/stage2.py [--devices CPU,GPU.0] [--csv results.csv]`):
- Step 1: `benchmark_app -hint throughput -latency_percentile 95` on all variants × devices
- Step 2: C++ `detector_test` binary on all variants (FP32/FP16/INT8) × devices
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

Measured earlier on i7-12700K (UHD 770 iGPU + RTX A4000 dGPU) with a different model, not re-run:

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

## Benchmarks

Current numbers (YOLOX, median of 3, with method and caveats) live in the root `README.md`. Earlier tables here were measured on other models and have been removed.

## detector_test CLI

```
./detector_test <model.zip> [device] [--runs N] [--warmup N] [--batch N] [--imgsz N] [--in-flight N]
                [--input-dtype f32|u8|f16] [--perf-hint latency|throughput|cumulative_throughput]
```

INT8 IRs take u8 input and FP16 IRs take f16 (the toolchain bakes the boundary type) — pass `--input-dtype` or the runtime rejects the tensor. Output is validated as `[batch, anchors, 85]` (YOLOX).
Average latency is skewed by queue depth when several requests are in flight — use **min latency** and **throughput** as the meaningful metrics.

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
  regression this caused and why `ov::shutdown()` can never go back in there.
