# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [2.0.0] - 2026-04-19

### Runtime Library — OAAX v2 Interface

Breaking change: the runtime now implements the OAAX v2 standard interface (`oaax_runtime.h`).
The v1 functions (`runtime_initialization_with_args`, `send_input`, `receive_output`,
`runtime_return_output`, `runtime_destruction`) are replaced by the new API.

**New public header:** `oaax_runtime.h` replaces `runtime_core.hpp` and `tensors_struct.h`.

**New types:**
- `TensorDescriptor` — array-of-structs layout (name, data_type, rank, shape, data_size, data
  co-located per tensor for cache locality); replaces the struct-of-arrays `tensors_struct`
- `Tensors` — wraps an array of `TensorDescriptor` plus a caller-assigned `id` for request correlation
- `TensorElementType` — expanded enum: adds `DATA_TYPE_FLOAT16`, `DATA_TYPE_BFLOAT16`,
  `DATA_TYPE_INT4`, `DATA_TYPE_UINT4`, and other ONNX-aligned types
- `Config` / `ModelConfig` — key-value configuration structs for init and per-model settings
- `RuntimeStatus` — fine-grained status codes (18 values) replacing int return codes

**New API functions:**
- `runtime_init(Config)` — initialize with key-value config
- `runtime_load_models(n, ModelConfig[])` — load N models in one call; each model can override
  `device_type`, `perf_hint`, `cache_dir` from the global config; returns `ALREADY_INITIALIZED`
  if called twice (requires `runtime_cleanup()` first)
- `runtime_enqueue_input(model_id, Tensors*)` — enqueue to a specific model; runtime takes
  ownership on success, caller frees on failure
- `runtime_retrieve_output(int *model_id, Tensors**, timeout_ms)` — dequeue from the global
  output queue (any model); fills `*model_id`; `timeout_ms=0` non-blocking, `<0` block forever,
  `>0` timed wait via `sem_timedwait` (Linux) / `WaitForSingleObject` (Windows)
- `runtime_cleanup()` — idempotent teardown
- `runtime_get_error()` — returns last error string or NULL
- `runtime_get_info()` — returns JSON diagnostics (loaded_models, requests_in_flight,
  backend_version, active_device)

**Architecture changes:**
- Per-model manager thread and infer-request pool; global output semaphore for blocking retrieve
- Standard `malloc`/`free` per output (output buffer pool removed per v2 spec)
- Callbacks capture `(model_id, slot_idx)` — one registration per slot at load time

**Benchmark results (Intel Core i7-12700K, CPU, batch=1, 300 runs):**

| Model | Precision | benchmark_app | OAAX v2 runtime | Delta |
|-------|-----------|--------------|-----------------|-------|
| yolo11n | FP32 | ~99 FPS | ~93 FPS | -6% |
| yolo11n | INT8 | ~238 FPS | ~223 FPS | -6% |
| yolov8n | FP32 | ~84 FPS | ~82 FPS | -2% |
| yolov8n | INT8 | ~235 FPS | ~223 FPS | -5% |
| yolo11s | FP32 | ~33 FPS | ~33 FPS | 0% |
| yolo11s | INT8 | ~99 FPS | ~98 FPS | -1% |

### Conversion Toolchain

- **`batch_size` parameter:** `config.json` now accepts `advanced.batch_size` (default `1`).
  For models without hardcoded internal batch constants (e.g. ResNet, MobileNet), the toolchain
  uses a probe-then-reconvert approach (`ov.convert_model(..., input=input_specs)`) so the batch
  dimension propagates through the full graph. YOLO models require re-export from PyTorch with the
  target batch size (standard Ultralytics exports have hardcoded DFL reshape constants that prevent
  post-hoc rebatching); attempting batch > 1 on a standard YOLO ONNX raises a clear `RuntimeError`
  with a re-export hint.
- **INT8 batch > 1 calibration fix:** `CalibrationDataLoader` now tiles each calibration sample to
  match the model's fixed batch dimension (`np.repeat`) so NNCF receives correctly-shaped tensors.
  Previously, INT8 quantization of a batch=4 model would immediately fail with a shape mismatch.

### Testing

- **Batch=4 CI pipeline** (`yolo11n_b4`, `yolo11s_b4`): YOLO models are re-exported from
  PyTorch with `batch=4` and converted in FP32, FP16, and INT8 variants. 26 new integration
  tests cover IR file presence, OpenVINO loading, output shape `[4, 84, 8400]`, numerical
  sanity, and INT8 size regression.
- **`multi_model_test`:** new C++ test binary that loads two models concurrently, enqueues
  inputs to both, and validates that outputs from each model are correctly identified by
  `model_id`. Exercises the OAAX v2 multi-model async dispatch path.
- **C++ test build decoupled from runtime build:** test binaries (`simple_test`, `yolo_test`,
  `multi_model_test`) now have their own `tests/runtime/CMakeLists.txt` and
  `tests/runtime/build-tests.sh`, linking against the prebuilt `libRuntimeLibrary.so`.
  The `runtime-library/` CMake no longer builds test executables.
- **`stage2.py --perf-hints`:** accepts a comma-separated list of performance hints
  (e.g. `latency,throughput`) and runs each section for all hint/device/model combinations.
  Adds `perf_hint` column to CSV output.
- **`stage2.py` benchmark_app timeout fix:** timeout changed from `duration * 4` to
  `120 + duration * 4` to cover GPU JIT compilation overhead before measurement starts
  (prevents false timeouts for batch=4 models on GPU with short `--duration` values).

### CI

- **Windows integration job simplified:** removed the OpenVINO install step from
  `integration-tests-windows`; test binaries now link only against the downloaded
  `RuntimeLibrary.lib` without needing the full OpenVINO SDK on the runner.

---

## [1.3.2] - 2026-04-15

### Runtime Library

- **Fixed output buffer pool exhaustion deadlock (critical):** when all pool buffers
  were held by the caller (e.g. using `deep_free_tensors_struct` instead of
  `runtime_return_output`), the manager thread would spin forever waiting for a buffer
  to return to the pool, permanently stalling inference. The pool acquisition path now
  calls `alloc_pool_buffer()` as a fallback — allocating a fresh same-shape buffer on
  the rare occasion the pool is empty — so inference continues uninterrupted.
- **`receive_output` adaptive backoff:** instead of a fixed sleep, the polling interval
  starts at 1 ms and doubles on each consecutive miss (up to 500 ms cap), then divides
  by 10 on each successful dequeue. This keeps latency low for fast models (INT8 ~37 ms
  inference sees ~1–2 ms overhead vs ~43 ms with a fixed 10 ms sleep) while naturally
  backing off when the pipeline is idle.

---

## [1.3.1] - 2026-04-15

### Runtime Library

- **Fixed `tensors_struct` field order (critical):** `include/tensors_struct.h` had
  field order `names, ranks, shapes, data_types, data`, which differed from the OAAX
  standard interface (`names, data_types, ranks, shapes, data`). Callers building
  against the standard header would write `data_types` at offset 2 while the runtime
  read offset 2 as `ranks`, causing immediate memory corruption and crashes on the
  first inference. Both headers are now aligned.
- **Updated `tensor_data_type` enum:** values aligned to OAAX standard (`UNDEFINED=0`,
  numeric gap at 10 reserved); removed `DATA_TYPE_FLOAT16`, `DATA_TYPE_BFLOAT16`,
  `DATA_TYPE_COMPLEX64`, and `DATA_TYPE_COMPLEX128` which are outside the supported set.
  OpenVINO type mappings updated accordingly.
- **Logger flush-on-every-message:** spdlog now calls `flush_on(level::trace)` so every
  log message is flushed immediately, ensuring full log visibility on abnormal exit.
- **Removed debug instrumentation:** `send_input` and `manager_thread_func` had
  temporary `"IIII"` marker, seven `"Breakpoint N"` logs, and tensor-metadata dump
  loops left in from development. All removed.
- **Hardened `send_input` null check:** null `input_tensors` is now detected before any
  other operation and sets `last_error` consistently with the rest of the API.

---

## [1.3.0] - 2026-04-14

### Runtime Library

- **OpenVINO compiled-model cache (`cache_dir`):** the runtime now calls
  `ov::cache_dir` so OpenVINO serialises compiled `.blob` files to disk after
  the first load. Subsequent loads skip compilation entirely (~400 ms → ~47 ms
  cold-start savings on a 5 MB YOLO INT8 model). Cache is **on by default**
  (stored in the process's working directory). Pass `cache_dir=""` to disable.
- **Compilation progress logs:** `"Compiling model for device '...' (hint=...)"` and
  `"Model compilation complete."` are now logged at INFO level so long-running
  first-time compilation (e.g. ~3 min for yolo11s on GPU) is clearly visible.

### Testing

- **yolo11s added to the test matrix:** `conftest.py`, `stage2.py`,
  `test_yolo_integration.py`, and `models.py` all include yolo11s; stage1 + stage2
  now cover all three YOLO variants (yolov8n, yolo11n, yolo11s) × (FP32/FP16/INT8).
- **Fixed stage2 timeout for large models:** `run_yolo_test()` timeout was
  `runs * 2` seconds — insufficient for cold compilation of yolo11s. Changed to
  `600 + runs * 2` (600 s overhead covers first-time GPU compilation).

---

## [1.2.1] - 2026-04-13

### Runtime Library

- **ZIP model loading:** `runtime_model_loading` now accepts both `.zip` archives
  (OAAX bundle) and bare `.xml` paths. When a `.zip` is passed the runtime extracts
  `model.xml` and `model.bin` to a temporary directory and loads from there.
  This aligns the runtime with the OAAX spec and the toolchain output format.

---

## [1.2.0] - 2026-04-12

### CI / Testing

- **Platform-independent test scripts:** replaced `scripts/stage1_compile.sh` and
  `scripts/stage2_run.sh` (bash/PowerShell) with `tests/stage1.py` and
  `tests/stage2.py` — single Python implementation runs on Linux and Windows.
- **Windows integration tests:** added `integration-tests-windows` CI job that
  downloads compiled models from the Linux job artifact and runs `stage2.py`.
- **Docker-based model compilation:** `tests/conftest.py` now compiles YOLO models
  (FP32/FP16/INT8) via the toolchain Docker image instead of calling the Python
  package directly, ensuring test parity with production conversion.
- **Public artifact downloads:** CI `integration-tests` jobs now download the
  runtime library and toolchain image via `wget`/`Invoke-WebRequest` from the
  public object storage URL instead of using `s3cmd`/`s5cmd` with credentials.
- **Unified build workflow:** merged `build-toolchain.yml` and `build-runtime.yml`
  into a single `build.yml`; removed the now-redundant `build-toolchain.yml`.

### Conversion Toolchain

- **Fixed NNCF missing from Docker image:** the `Dockerfile` was installing only
  base dependencies (`-e .`); changed to `.[quantization]` so NNCF is included.
  Previously, INT8 conversions silently fell back to FP32-sized output with
  "NNCF not available, skipping quantization".

---

## [1.1.1] - 2026-04-11

### Runtime — Performance Optimizations

- **Zero-copy output:** `set_output_tensor` now directs each `InferRequest` output
  straight into a pre-allocated pool buffer; OpenVINO writes inference results
  in-place with no memcpy on the hot path.
- **Single manager thread:** replaced N blocking worker threads (one per
  `InferRequest`) with one manager thread that dispatches async inference via
  `start_async()` + `set_callback()`, matching the OpenVINO-recommended pattern.
- **POSIX semaphore dispatch:** `sem_t slot_sem` and `sem_t input_sem` replace the
  previous mutex+condition_variable `Semaphore` class, reducing dispatch latency
  to one futex syscall per slot transition.
- **Per-slot state + single callback registration:** a `SlotState[]` array holds
  per-inference context; callback lambdas capture only a slot index (4 bytes),
  fitting in `std::function`'s small-object buffer and eliminating a heap
  allocation per inference. `set_callback()` is now called once at model load.
- **Optimal worker count:** `ov::optimal_number_of_infer_requests` is always
  queried from the compiled model; the `num_requests` runtime argument has been
  removed.
- **Removed `num_threads` runtime arg:** setting `ov::inference_num_threads`
  alongside a performance hint conflicts with OpenVINO's scheduler; the argument
  is now dropped and `perf_hint` alone controls thread allocation.

### Testing

- Added `tests/test_quantization_accuracy.py`: tensor-level cosine similarity
  comparison of INT8/FP16 outputs against FP32 baseline on 128 COCO images
  (FP16 ≈ 1.000000, INT8 ≈ 0.999835).
- `stage2_run.sh`: increased `yolo_test` from 30 to 300 measurement runs for
  statistically stable throughput numbers.
- Verified zero RSS growth over 10,000 INT8 inferences (no memory leaks).

### Benchmark results (Intel Core i7-13700K, hint=throughput)

| Tool | Model | Precision | Throughput |
|------|-------|-----------|------------|
| benchmark_app | yolo11n | FP32 | ~95.6 FPS |
| benchmark_app | yolo11n | FP16 | ~98.6 FPS |
| benchmark_app | yolo11n | INT8 | ~244 FPS |
| yolo_test (OAAX) | yolo11n | FP32 | ~97.7 FPS |
| yolo_test (OAAX) | yolo11n | FP16 | ~98.1 FPS |
| yolo_test (OAAX) | yolo11n | INT8 | ~237 FPS |

OAAX runtime matches `benchmark_app` within ~2% for FP32/FP16 and ~3% for INT8.

---

## [1.1.0] - 2026-03-19

### Runtime — OpenVINO Native Migration

- **Replaced ONNX Runtime with OpenVINO native C++ API:** direct `.xml`/`.bin` IR
  loading via `ov::Core`, 92% reduction in binary size (250 MB → 20 MB), 50–75%
  faster initialization and model loading.
- Added `src/runtime_utils.cpp`: bidirectional type mapping between OAAX
  `tensor_data_type` and `ov::element::Type`.
- Added `FP16` (`DATA_TYPE_FLOAT16`) tensor type support.
- Added `perf_hint` runtime argument (`latency` / `throughput` /
  `cumulative_throughput`) — passed as `ov::hint::performance_mode` at compile
  time; worker count inferred via `ov::optimal_number_of_infer_requests`.
- Added `runtime_return_output(tensors_struct*)` public API extension — returns an
  output buffer to the pre-allocated pool instead of `deep_free_tensors_struct`.
- Added pre-allocated output buffer pool (`actual_requests × 4` tensors_struct
  objects) to eliminate malloc/free on the hot path for static output shapes.
- Dropped transitive dependency on 150+ ONNX Runtime shared libraries; runtime
  now links only `openvino` and `tbb`.

### Conversion Toolchain

- **Migrated from ONNX Simplifier to OpenVINO native converter** (`ov.convert_model`).
- Upgraded to **OpenVINO 2026.1.0** and **NNCF 2.19.0**.
- Fixed NNCF/OpenVINO compatibility: added `openvino.Node` import shim for
  changed API path.
- Fixed GCC dual-ABI mismatch (`-D_GLIBCXX_USE_CXX11_ABI=0`).
- Added versioned `.so` symlinks for OpenVINO shared libraries.

### Testing

- Added two-stage test framework:
  - `scripts/stage1_compile.sh` — converts YOLOv8n and YOLOv11n to
    FP32/FP16/INT8 OpenVINO IR; runs conversion and IR validation tests.
  - `scripts/stage2_run.sh` — benchmarks compiled models with `benchmark_app`
    and `yolo_test`; supports `--devices`, `--duration`, `--csv`.
- Rewrote `yolo_test` as a multi-run benchmark (configurable warmup + N runs,
  avg/min/p95/throughput output).
- Added `benchmark_app` integration using OpenVINO's reference tool.
- Added `tests/test_yolo_integration.py`: YOLO IR validation tests.
- Added `tests/test_docker.py`: Docker image tests.
- Added CSV output for cross-run benchmark comparison.

---

## [1.0.0] - 2025-04-17

- Initial release of the OAAX OpenVINO implementation.
- Includes the OAAX runtime and conversion toolchain.
- There is one runtime that can be used for Intel's CPU, GPU, and NPU.
- The conversion toolchain is expected to run on x86_64.
