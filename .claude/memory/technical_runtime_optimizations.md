---
name: Runtime optimization findings
description: Key design decisions and performance findings for the OAAX OpenVINO C++ runtime
type: project
---
## Multi-InferRequest parallel workers

Each worker thread owns one `ov::InferRequest` exclusively and competes on the shared input queue.
FIFO ordering is NOT guaranteed when `num_requests > 1`.

**Why:** OpenVINO's `THROUGHPUT` hint splits CPU cores into N streams; you need N concurrent
InferRequests to keep all streams fed. With `num_requests=1` under THROUGHPUT hint, only one stream
is ever active and FPS is worse than LATENCY hint.

**How to apply:** Worker count is always inferred via
`compiled_model->get_property(ov::optimal_number_of_infer_requests)` (returns 4 on i7-13700K).
Never set it manually.

## perf_hint replaces num_threads

`ov::hint::performance_mode` is passed at `compile_model` time and encodes the full resource
strategy — OpenVINO manages thread distribution internally. Explicitly setting
`ov::inference_num_threads` alongside a hint conflicts with OpenVINO's scheduler. `num_threads`
was removed from the runtime init args entirely.

## Output buffer pool

Pre-allocate `actual_requests × 4` `tensors_struct` objects with fully-sized data buffers after
model loading (shapes are static after compile). Workers do `memcpy` only — no `malloc` on the
hot path.

**Why:** At INT8 speeds (~255 FPS), per-inference `malloc(2.8 MB)` triggers mmap/munmap syscalls
(~5µs each × 510/s = ~2.5ms/s overhead = ~12% of throughput). Pool eliminates this.

**How to apply:**
- Pool is only active for static output shapes (`port.get_partial_shape().is_static()`). Falls back
  to malloc for dynamic shapes.
- Consumers MUST call `runtime_return_output(ts)` instead of `deep_free_tensors_struct(ts)`. Pool
  drains and pre-frees on model reload and destruction.
- Pool size = `actual_requests * 4` — sized for `max_in_flight=10` with headroom.

## DATA_TYPE_FLOAT16 was missing

`ov::element::f16` was unhandled in both `map_to_ov_type` and `map_to_tensors_struct_type` — would
throw at runtime on any FP16 model output. Added `DATA_TYPE_FLOAT16 = 11` to the enum (2 bytes).

## memcpy floor (cannot close without API change)

YOLO output is [1×84×8400×f32] = 2.8 MB. At 255 FPS that's 714 MB/s of sustained copies. The
remaining ~9% INT8 gap vs `benchmark_app` (which reads tensors in-place) is entirely this copy.
Closing it would require a "borrow" ownership model in tensors_struct (caller borrows pointer,
doesn't free it), which is a breaking API change.

## benchmark_app vs yolo_test latency numbers are not comparable

`benchmark_app -hint throughput` reports latency-at-max-throughput with multiple parallel requests
(high avg_ms). `yolo_test` with `max_in_flight=10` also queues requests — avg_ms reflects queue
depth, not single-request latency. Use `benchmark_app -hint latency` or
`yolo_test --num-requests 1 --perf-hint latency` for true per-request latency.

## POSIX sem_t + per-slot state

`sem_t` (futex-backed) is lighter than `std::mutex` + `std::condition_variable` — 1 syscall vs 2
in the contended path. Per-slot `SlotState` avoids capturing large data in the callback lambda,
keeping the capture under GCC's 16-byte std::function SOO buffer (no heap alloc per inference).
`set_callback()` is registered once per slot at model load, not once per inference.
