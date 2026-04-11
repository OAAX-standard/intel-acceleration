
#ifndef RUNTIME_CORE_HPP
#define RUNTIME_CORE_HPP

#include "tensors_struct.h"

#ifdef _WIN32
#define EXPOSE_FUNCTION __declspec(dllexport)
#else
#define EXPOSE_FUNCTION __attribute__((visibility("default")))
#endif

/**
 * @file runtime_core.hpp
 * @brief OAAX public C API for OpenVINO-based inference on Intel CPU / GPU / NPU.
 *
 * Typical call sequence:
 * @code
 *   runtime_initialization_with_args(n, keys, values);
 *   runtime_model_loading("/path/to/model.xml");
 *   send_input(input);
 *   tensors_struct *output;
 *   while (receive_output(&output) != 0) {}  // poll until result ready
 *   runtime_return_output(output);   // return buffer to pool
 *   runtime_destruction();
 * @endcode
 *
 * Thread safety: send_input() and receive_output() are thread-safe.
 * All other functions must be called from a single thread.
 */

/**
 * @brief Initialize the runtime with configuration key-value pairs.
 *
 * Supported keys:
 * | Key           | Type   | Default        | Description                                      |
 * |---------------|--------|----------------|--------------------------------------------------|
 * | `device_type` | string | `"CPU"`        | Target device: `"CPU"`, `"GPU"`, `"NPU"`         |
 * | `perf_hint`   | string | `"latency"`    | `"latency"`, `"throughput"`, `"cumulative_throughput"` |
 * | `precision`   | string | `"FP32"`       | Informational only; actual precision set at conversion |
 * | `log_level`   | int    | `2` (info)     | spdlog level: 0=trace … 6=off                    |
 * | `log_file`    | string | `"runtime.log"`| Path to the log file                             |
 *
 * @param length  Number of key-value pairs.
 * @param keys    Array of null-terminated key strings.
 * @param values  Array of value pointers (cast to char* for string/int values).
 * @return 0 on success, -1 on failure.
 */
extern "C" EXPOSE_FUNCTION int runtime_initialization_with_args(int length, char **keys, void **values);

/**
 * @brief Initialize the runtime with default configuration.
 *
 * Equivalent to calling runtime_initialization_with_args() with zero arguments.
 * Device defaults to CPU, perf_hint defaults to latency.
 *
 * @return 0 on success, -1 on failure.
 */
extern "C" EXPOSE_FUNCTION int runtime_initialization();

/**
 * @brief Load and compile an OpenVINO IR model.
 *
 * Reads the `.xml` file at @p model_path; the corresponding `.bin` weights file
 * must reside in the same directory with the same base name.
 *
 * Calling this function on a runtime that already has a model loaded will
 * stop the current inference pipeline, release all resources, and start fresh.
 *
 * After a successful call:
 * - An async inference pipeline is running (manager thread + N InferRequests).
 * - A pre-allocated output buffer pool is active for static output shapes.
 * - The optimal number of InferRequests is inferred from the compiled model
 *   according to the configured perf_hint.
 *
 * @param model_path  Absolute or relative path to the `.xml` IR file.
 * @return 0 on success, -1 on failure (check log file for details).
 */
extern "C" EXPOSE_FUNCTION int runtime_model_loading(const char *model_path);

/**
 * @brief Enqueue an input tensor set for asynchronous inference.
 *
 * The runtime takes ownership of @p input_tensors. Do NOT free it after this
 * call; the runtime will call deep_free_tensors_struct() internally once
 * inference completes.
 *
 * Returns immediately; use receive_output() to poll for the result.
 *
 * @param input_tensors  Pointer to the populated input tensor structure.
 * @return 0 on success, -1 if the input could not be enqueued.
 */
extern "C" EXPOSE_FUNCTION int send_input(tensors_struct *input_tensors);

/**
 * @brief Attempt to dequeue a completed inference result (non-blocking).
 *
 * Returns immediately. If no result is available yet, @p *output_tensors is
 * left unchanged and -1 is returned — the caller should retry.
 *
 * The caller is responsible for releasing the returned buffer by calling
 * runtime_return_output() (preferred, enables pool reuse) or
 * deep_free_tensors_struct() (fallback, disables pool reuse).
 *
 * Output ordering: FIFO order is NOT guaranteed when more than one
 * InferRequest is active (throughput hint).
 *
 * @param[out] output_tensors  Set to the dequeued result on success.
 * @return 0 if a result was dequeued, -1 if the queue is empty.
 */
extern "C" EXPOSE_FUNCTION int receive_output(tensors_struct **output_tensors);

/**
 * @brief Return an output buffer received via receive_output() back to the pool.
 *
 * Use this instead of deep_free_tensors_struct() to enable zero-copy buffer
 * reuse on the hot path. The buffer will be recycled into the pre-allocated
 * pool and reused by the next inference.
 *
 * Falls back to deep_free_tensors_struct() automatically when the pool is not
 * active (e.g. dynamic output shapes).
 *
 * Passing NULL is safe and has no effect.
 *
 * @param output  Buffer previously returned by receive_output().
 */
extern "C" EXPOSE_FUNCTION void runtime_return_output(tensors_struct *output);

/**
 * @brief Stop the inference pipeline and release all resources.
 *
 * Blocks until all in-flight inference requests complete and the manager
 * thread exits. Safe to call even if no model has been loaded.
 *
 * After this call the runtime is fully torn down; call runtime_initialization()
 * again to reuse the library.
 *
 * @return 0 on success, -1 on failure.
 */
extern "C" EXPOSE_FUNCTION int runtime_destruction();

/**
 * @brief Return a human-readable description of the last error.
 *
 * @return Null-terminated static string (do not free). Points the caller to
 *         stdout or the log file for detailed error information.
 */
extern "C" EXPOSE_FUNCTION const char *runtime_error_message();

/**
 * @brief Return the runtime library version string.
 *
 * @return Null-terminated static string in MAJOR.MINOR.PATCH format,
 *         matching the root VERSION file.
 */
extern "C" EXPOSE_FUNCTION const char *runtime_version();

/**
 * @brief Return the runtime library name.
 *
 * @return Null-terminated static string identifying this OAAX implementation.
 */
extern "C" EXPOSE_FUNCTION const char *runtime_name();


#endif // RUNTIME_CORE_HPP
