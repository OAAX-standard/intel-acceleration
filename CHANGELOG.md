# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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
