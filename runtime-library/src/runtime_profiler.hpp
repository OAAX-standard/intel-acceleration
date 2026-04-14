// Debug-only dispatch-pipeline profiler for the OAAX runtime.
// Enabled when OAAX_PROFILE=1 (set automatically for Debug builds by CMake).
//
// Usage:
//   PROF_TS(t);             // capture timestamp into local int64_t `t`
//   /* work */
//   PROF_ADD(field, t);     // add elapsed nanoseconds to g_profiler.field
//   PROF_INC(dispatches);   // increment a counter
//
// All macros are no-ops in Release builds — zero overhead, no code emitted.
#pragma once

#ifdef OAAX_PROFILE

#include <atomic>
#include <chrono>
#include <cstdint>

// Accumulator for one profiling run (reset on each runtime_model_loading()
// call). All fields are in nanoseconds; use dispatches to compute per-inference
// averages.
struct OaaxProfiler {
  std::atomic<uint64_t> input_wait_ns{
      0};  // manager: blocked on sem_wait(&input_sem)
  std::atomic<uint64_t> slot_wait_ns{
      0};  // manager: blocked on sem_wait(&slot_sem)
  std::atomic<uint64_t> pool_wait_ns{0};  // manager: spinning for a pool buffer
  std::atomic<uint64_t> tensor_setup_ns{
      0};  // manager: set_output_tensor + set_tensor
  std::atomic<uint64_t> inference_ns{
      0};  // start_async() to completion callback
  std::atomic<uint64_t> output_queue_ns{
      0};                               // callback enqueue to receive_output()
  std::atomic<uint64_t> dispatches{0};  // total inferences dispatched

  void reset() {
    input_wait_ns = slot_wait_ns = pool_wait_ns = tensor_setup_ns = 0;
    inference_ns = output_queue_ns = dispatches = 0;
  }
};

inline int64_t prof_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// Capture current time into a local variable.
#define PROF_TS(var) int64_t var = prof_now_ns()

// Add elapsed nanoseconds since `start` into the named accumulator.
#define PROF_ADD(field, start)                                               \
  g_profiler.field.fetch_add(static_cast<uint64_t>(prof_now_ns() - (start)), \
                             std::memory_order_relaxed)

// Increment a counter by 1.
#define PROF_INC(field) g_profiler.field.fetch_add(1, std::memory_order_relaxed)

#else  // !OAAX_PROFILE — all macros are complete no-ops

#define PROF_TS(var) ((void)0)
#define PROF_ADD(field, start) ((void)0)
#define PROF_INC(field) ((void)0)

#endif  // OAAX_PROFILE
