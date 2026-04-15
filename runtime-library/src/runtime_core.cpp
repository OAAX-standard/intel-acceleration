#include <atomic>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define NOMINMAX  // prevent windows.h from defining min/max macros
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
// Minimal POSIX semaphore shim — maps sem_t to a Windows HANDLE semaphore.
struct sem_t {
  HANDLE h;
};
static inline int sem_init(sem_t* s, int /*pshared*/, unsigned int value) {
  s->h = CreateSemaphoreW(nullptr, (LONG)value, 0x7FFFFFFF, nullptr);
  return s->h ? 0 : -1;
}
static inline int sem_wait(sem_t* s) {
  return WaitForSingleObject(s->h, INFINITE) == WAIT_OBJECT_0 ? 0 : -1;
}
static inline int sem_post(sem_t* s) {
  return ReleaseSemaphore(s->h, 1, nullptr) ? 0 : -1;
}
static inline int sem_destroy(sem_t* s) { return CloseHandle(s->h) ? 0 : -1; }
#else
#include <semaphore.h>
#endif

#include <spdlog/spdlog.h>

#include <openvino/openvino.hpp>

#include "concurrentqueue.h"
#include "runtime_core.hpp"
#include "runtime_profiler.hpp"
#include "runtime_utils.hpp"
#include "tensors_struct.h"
#include "zip_utils.hpp"

using namespace std;

// Helper functions
static void log_available_devices();
static void on_inference_complete(int idx, std::exception_ptr ex);
static void manager_thread_func();
static void drain_output_pool();
static void stop_and_join_manager();
static tensors_struct* alloc_pool_buffer();

// OpenVINO vars
static std::shared_ptr<ov::Core> core;
static std::shared_ptr<ov::CompiledModel> compiled_model;
static std::vector<ov::InferRequest> infer_requests;

// Queue variables
static moodycamel::ConcurrentQueue<tensors_struct*> input_tensors_queue;
static moodycamel::ConcurrentQueue<tensors_struct*> output_tensors_queue;

// Output buffer pool: pre-allocated tensors_struct whose data[] pointers are
// set as InferRequest output tensors before each start_async() call.
// OpenVINO writes directly into these buffers — zero copy.
// Consumers call runtime_return_output() to return buffers instead of
// deep_free.
static moodycamel::ConcurrentQueue<tensors_struct*> output_buffer_pool;
static std::atomic<bool> pool_active{false};

// Per-output metadata needed to call set_output_tensor before each dispatch.
struct OutInfo {
  ov::Shape shape;
  ov::element::Type ov_type;
  tensor_data_type dtype;
  size_t byte_size;
};
static std::vector<OutInfo> pool_out_infos;

// Free InferRequest slot pool: holds indices into infer_requests[] for idle
// slots. Manager dequeues a slot before dispatching; on_inference_complete
// re-enqueues it.
static moodycamel::ConcurrentQueue<int> free_requests;

// POSIX semaphores (futex-backed) — one sem_post/wait per slot transition,
// one sem_post/wait per queued input.  Lighter than mutex+CV on the hot path.
static sem_t slot_sem;   // counts available InferRequest slots
static sem_t input_sem;  // signals that send_input() enqueued a new item

// Per-slot state written by the manager before each start_async() and read by
// on_inference_complete.  Storing it here — rather than in the lambda capture —
// keeps the callback lambda down to a single int capture, which fits inside
// std::function's small-object buffer and avoids a heap allocation per
// inference.
struct SlotState {
  tensors_struct* input{nullptr};
  tensors_struct* output{nullptr};
  bool from_pool{false};
#ifdef OAAX_PROFILE
  int64_t start_async_ns{0};  // timestamp just before start_async(); used to
                              // measure inference time
#endif
};
static std::vector<SlotState> slot_states;

// Session variables
static vector<string> output_names;
static vector<string> input_names;
// Temp directory created when a .zip model path is provided; cleaned up on
// next model load or runtime_destruction().
static string model_temp_dir;

// Manager thread: one thread dispatches async inference and returns when
// stopped.
static std::thread manager_thread;
static std::atomic<bool> stop_manager{false};

// Logger
std::shared_ptr<spdlog::logger> logger;

// Runtime arguments
static int log_level = spdlog::level::info;
static string log_file = "runtime.log";
static string device_type = "CPU";
static string perf_hint = "latency";
// Model compilation cache directory.  Default "." = current working directory.
// Set to "" to disable caching entirely.
static string cache_dir = ".";

// Last error message returned by runtime_error_message().
static string last_error;

#ifdef OAAX_PROFILE
static OaaxProfiler g_profiler;
// Parallel queue of enqueue timestamps — one entry per output_tensors_queue
// push. Used by receive_output() to measure how long results sit in the output
// queue.
static moodycamel::ConcurrentQueue<int64_t> output_enqueue_times;
#endif

extern "C" int runtime_initialization_with_args(int length, char** keys,
                                                void** values) {
  std::vector<string> unknown_args;
  std::vector<string> bad_perf_hints;

  for (int i = 0; i < length; ++i) {
    string key = string(keys[i]);
    if (key == "log_level") {
      int value = std::stoi(static_cast<char*>(values[i]));
      if (value >= spdlog::level::trace && value <= spdlog::level::off)
        log_level = value;
      else
        log_level = spdlog::level::info;
    } else if (key == "log_file")
      log_file = string(static_cast<char*>(values[i]));
    else if (key == "device_type")
      device_type = string(static_cast<char*>(values[i]));
    else if (key == "perf_hint") {
      string val = string(static_cast<char*>(values[i]));
      if (val == "latency" || val == "throughput" ||
          val == "cumulative_throughput")
        perf_hint = val;
      else
        bad_perf_hints.push_back(val);
    } else if (key == "cache_dir") {
      cache_dir = string(static_cast<char*>(values[i]));
    } else {
      unknown_args.push_back(key);
    }
  }

  int ret = runtime_initialization();  // sets up logger

  for (const auto& val : bad_perf_hints)
    logger->warn("Unknown perf_hint value '{}' — keeping default '{}'", val,
                 perf_hint);
  for (const auto& k : unknown_args)
    logger->warn("Unknown runtime argument '{}' — ignored", k);

  return ret;
}

extern "C" int runtime_initialization() {
  try {
    logger = initialize_logger(log_file, log_level, log_level, runtime_name());
    logger->info("Initializing the runtime");

    core = std::make_shared<ov::Core>();
    logger->trace("OpenVINO Core initialized");

    logger->info("Runtime arguments:");
    logger->info("  log_level: {}", log_level);
    logger->info("  log_file: {}", log_file);
    logger->info("  device_type: {}", device_type);
    logger->info("  perf_hint: {}", perf_hint);

    // Model compilation cache — eliminates recompilation on subsequent loads
    // of the same model+device combination.
    if (!cache_dir.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(cache_dir, ec);
      if (!ec) {
        auto canon = std::filesystem::canonical(cache_dir, ec);
        string cache_path = ec ? cache_dir : canon.string();
        core->set_property(ov::cache_dir(cache_path));
        logger->info("  cache_dir: {} (model cache active)", cache_path);
      } else {
        logger->warn("  cache_dir: cannot create '{}' ({}) — caching disabled",
                     cache_dir, ec.message());
      }
    } else {
      logger->info("  cache_dir: (disabled)");
    }

    log_available_devices();
    return 0;
  } catch (const std::exception& e) {
    last_error = e.what();
    logger->error("Runtime initialization failed: {}", last_error);
    return -1;
  }
}

extern "C" int runtime_model_loading(const char* model_path) {
  logger->info("Loading model from: {}", model_path);
  try {
    // Extract the zip archive to a temp directory and resolve the .xml path.
    std::string new_temp_dir;
    std::string xml_path = extract_zip_model(model_path, new_temp_dir);
    if (xml_path.empty()) {
      last_error = "Failed to extract model archive (expected a valid zip): " +
                   std::string(model_path);
      logger->error("{}", last_error);
      cleanup_temp_dir(new_temp_dir);
      return -1;
    }
    logger->info("Extracted model XML: {}", xml_path);

    // Release the previous temp dir (if any) before recording the new one.
    cleanup_temp_dir(model_temp_dir);
    model_temp_dir = new_temp_dir;

    std::shared_ptr<ov::Model> model = core->read_model(xml_path);
    logger->debug("Model read successfully from: {}", model_path);

    input_names.clear();
    output_names.clear();

    for (const auto& input : model->inputs()) {
      input_names.push_back(input.get_any_name());
      logger->trace("Input: {}", input.get_any_name());
    }
    for (const auto& output : model->outputs()) {
      output_names.push_back(output.get_any_name());
      logger->trace("Output: {}", output.get_any_name());
    }

    ov::AnyMap config;
    if (perf_hint == "throughput")
      config[ov::hint::performance_mode.name()] =
          ov::hint::PerformanceMode::THROUGHPUT;
    else if (perf_hint == "cumulative_throughput")
      config[ov::hint::performance_mode.name()] =
          ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT;
    else
      config[ov::hint::performance_mode.name()] =
          ov::hint::PerformanceMode::LATENCY;

    // When the caller requests "GPU" and multiple GPU devices are present,
    // automatically switch to the MULTI plugin so all GPUs share the load.
    // Explicit device strings (e.g. "GPU.0", "MULTI:...") are left unchanged.
    string effective_device = device_type;
    if (device_type == "GPU") {
      vector<string> gpus;
      try {
        for (const auto& d : core->get_available_devices())
          if (d.rfind("GPU", 0) == 0) gpus.push_back(d);
      } catch (...) {
      }
      if (gpus.size() > 1) {
        effective_device = "MULTI:";
        for (size_t i = 0; i < gpus.size(); ++i) {
          if (i > 0) effective_device += ",";
          effective_device += gpus[i];
        }
        logger->info("Multiple GPUs detected — compiling for {}",
                     effective_device);
      }
    }

    logger->info("Compiling model for device '{}' (hint={})...",
                 effective_device, perf_hint);
    try {
      compiled_model = std::make_shared<ov::CompiledModel>(
          core->compile_model(model, effective_device, config));
      logger->info("Model compilation complete.");
    } catch (const std::exception& e) {
      if (device_type != "CPU") {
        logger->warn("Failed to compile model on {} ({}). Falling back to CPU.",
                     effective_device, e.what());
        device_type = "CPU";
        effective_device = "CPU";
        logger->info("Compiling model for device 'CPU' (hint={})...",
                     perf_hint);
        compiled_model = std::make_shared<ov::CompiledModel>(
            core->compile_model(model, "CPU", config));
        logger->info("Model compilation complete.");
      } else {
        throw;
      }
    }

    int actual_requests =
        compiled_model->get_property(ov::optimal_number_of_infer_requests);
    logger->info("Optimal infer requests for hint={}: {}", perf_hint,
                 actual_requests);

    // Stop any existing manager and wait for all in-flight completions.
    stop_and_join_manager();
    drain_output_pool();

    // Reset profiling counters now that all old-model callbacks have completed.
#ifdef OAAX_PROFILE
    g_profiler.reset();
    {
      int64_t ts;
      while (output_enqueue_times.try_dequeue(ts)) {
      }
    }
#endif

    // Reset all per-session state.
    {
      int idx;
      while (free_requests.try_dequeue(idx)) {
      }
    }
    {
      tensors_struct* p;
      while (input_tensors_queue.try_dequeue(p)) deep_free_tensors_struct(p);
    }
    sem_destroy(&slot_sem);
    sem_init(&slot_sem, 0, 0);
    sem_destroy(&input_sem);
    sem_init(&input_sem, 0, 0);
    pool_out_infos.clear();
    slot_states.clear();

    // Create InferRequests.
    infer_requests.clear();
    for (int i = 0; i < actual_requests; ++i)
      infer_requests.emplace_back(compiled_model->create_infer_request());
    logger->debug("Created {} infer request(s)", actual_requests);

    // Build output-buffer pool if all shapes are static.
    int pool_size = actual_requests * 4;
    bool can_pool = true;

    for (size_t i = 0; i < output_names.size(); ++i) {
      auto port = compiled_model->output(output_names[i]);
      if (!port.get_partial_shape().is_static()) {
        can_pool = false;
        break;
      }
      ov::Shape shape = port.get_partial_shape().to_shape();
      ov::element::Type ov_type = port.get_element_type();
      tensor_data_type dtype = map_to_tensors_struct_type(ov_type);
      size_t byte_size = ov_type.size() * ov::shape_size(shape);
      pool_out_infos.push_back({shape, ov_type, dtype, byte_size});
    }

    if (can_pool) {
      for (int i = 0; i < pool_size; ++i) {
        tensors_struct* ts = allocate_tensors_struct(output_names.size());
        for (size_t j = 0; j < output_names.size(); ++j) {
          ts->names[j] = strdup(output_names[j].c_str());
          ts->ranks[j] = pool_out_infos[j].shape.size();
          ts->shapes[j] =
              (size_t*)malloc(sizeof(size_t) * pool_out_infos[j].shape.size());
          for (size_t k = 0; k < pool_out_infos[j].shape.size(); ++k)
            ts->shapes[j][k] = pool_out_infos[j].shape[k];
          ts->data_types[j] = pool_out_infos[j].dtype;
          ts->data[j] = malloc(pool_out_infos[j].byte_size);
        }
        output_buffer_pool.enqueue(ts);
      }
      pool_active = true;
      logger->info("Pre-allocated {} output buffers (zero-copy pool)",
                   pool_size);
    } else {
      logger->info(
          "Dynamic output shapes — buffer pool disabled, using malloc per "
          "inference");
    }

    // Populate free-slot queue and semaphore.
    slot_states.resize(actual_requests);
    for (int i = 0; i < actual_requests; ++i) free_requests.enqueue(i);
    sem_post(&slot_sem);  // will be called actual_requests times via loop below
    // Reset the single post above and do it properly:
    sem_destroy(&slot_sem);
    sem_init(&slot_sem, 0, actual_requests);

    // Set the completion callback ONCE per slot — lambda captures only the
    // slot index (4 bytes), fitting inside std::function's small-object buffer
    // and avoiding a per-inference heap allocation.
    for (int i = 0; i < actual_requests; ++i) {
      infer_requests[i].set_callback(
          [i](std::exception_ptr ex) { on_inference_complete(i, ex); });
    }

    // Start the single manager thread.
    stop_manager = false;
    manager_thread = std::thread(manager_thread_func);
    logger->info("Started async manager thread ({} infer slot(s))",
                 actual_requests);

    return 0;
  } catch (const std::exception& e) {
    last_error = e.what();
    logger->error("Model loading failed: {}", last_error);
    return -1;
  }
}

extern "C" int send_input(tensors_struct* input_tensors) {
  if (!input_tensors) {
    last_error = "send_input called with null tensors";
    logger->error("{}", last_error);
    return -1;
  }
  if (!compiled_model) {
    last_error =
        "send_input called before a model was loaded — call "
        "runtime_model_loading() first";
    logger->error("{}", last_error);
    return -1;
  }
  if (!input_tensors_queue.try_enqueue(input_tensors)) {
    last_error =
        "Input queue is full — inference is not keeping up with the producer";
    logger->warn("{}", last_error);
    return -1;
  }
  sem_post(&input_sem);
  logger->trace("Input tensors enqueued.");
  return 0;
}

extern "C" int receive_output(tensors_struct** output_tensors) {
  thread_local int sleep_ms = 1;
  if (!output_tensors_queue.try_dequeue(*output_tensors)) {
    logger->trace("No output tensors available.");
    std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
    sleep_ms = std::min(sleep_ms * 2, 100);
    return -1;
  }
  sleep_ms = std::max(sleep_ms / 10, 1);  // decay backoff on success
#ifdef OAAX_PROFILE
  int64_t enqueue_ns;
  if (output_enqueue_times.try_dequeue(enqueue_ns))
    g_profiler.output_queue_ns.fetch_add(
        static_cast<uint64_t>(prof_now_ns() - enqueue_ns),
        std::memory_order_relaxed);
#endif
  logger->debug("Output tensors received.");
  return 0;
}

// Completion callback — invoked on OpenVINO's internal thread when slot `idx`
// finishes inference.  Reads per-slot state set by the manager before dispatch.
static void on_inference_complete(int idx, std::exception_ptr ex) {
  SlotState& s = slot_states[idx];

#ifdef OAAX_PROFILE
  g_profiler.inference_ns.fetch_add(
      static_cast<uint64_t>(prof_now_ns() - s.start_async_ns),
      std::memory_order_relaxed);
#endif

  if (ex) {
    try {
      std::rethrow_exception(ex);
    } catch (const std::exception& e) {
      logger->error("[slot {}] Async inference error: {}", idx, e.what());
    }
    deep_free_tensors_struct(s.input);
    runtime_return_output(s.output);
    free_requests.enqueue(idx);
    sem_post(&slot_sem);
    return;
  }

  tensors_struct* result = s.output;

  if (!s.from_pool) {
    // Dynamic shapes: allocate and copy from the request's output tensors.
    ov::InferRequest& r = infer_requests[idx];
    result = allocate_tensors_struct(output_names.size());
    for (size_t i = 0; i < output_names.size(); ++i) {
      ov::Tensor out = r.get_tensor(output_names[i]);
      result->names[i] = strdup(output_names[i].c_str());
      ov::Shape shape = out.get_shape();
      result->ranks[i] = shape.size();
      result->shapes[i] = (size_t*)malloc(sizeof(size_t) * shape.size());
      for (size_t j = 0; j < shape.size(); ++j) result->shapes[i][j] = shape[j];
      result->data_types[i] =
          map_to_tensors_struct_type(out.get_element_type());
      size_t sz = out.get_byte_size();
      result->data[i] = malloc(sz);
      memcpy(result->data[i], out.data(), sz);
    }
  }

  deep_free_tensors_struct(s.input);

  // Push timestamp BEFORE result so receive_job() always finds it available.
#ifdef OAAX_PROFILE
  output_enqueue_times.enqueue(prof_now_ns());
#endif
  if (!output_tensors_queue.try_enqueue(result)) {
#ifdef OAAX_PROFILE
    int64_t discard;
    output_enqueue_times.try_dequeue(discard);
#endif
    logger->error("[slot {}] Failed to enqueue output.", idx);
    runtime_return_output(result);
  } else {
    logger->debug("[slot {}] Output enqueued.", idx);
  }

  free_requests.enqueue(idx);
  sem_post(&slot_sem);
}

// Allocate a fresh output buffer with the same shape/type layout as a pool
// buffer.  Used when the pool is empty (e.g. caller freed the previous buffer
// via deep_free_tensors_struct instead of runtime_return_output).
static tensors_struct* alloc_pool_buffer() {
  tensors_struct* ts = allocate_tensors_struct(output_names.size());
  for (size_t i = 0; i < output_names.size(); ++i) {
    ts->names[i] = strdup(output_names[i].c_str());
    ts->ranks[i] = pool_out_infos[i].shape.size();
    ts->shapes[i] =
        (size_t*)malloc(sizeof(size_t) * pool_out_infos[i].shape.size());
    for (size_t k = 0; k < pool_out_infos[i].shape.size(); ++k)
      ts->shapes[i][k] = pool_out_infos[i].shape[k];
    ts->data_types[i] = pool_out_infos[i].dtype;
    ts->data[i] = malloc(pool_out_infos[i].byte_size);
  }
  return ts;
}

// Manager thread: dequeues inputs, acquires a free slot, sets up tensors,
// then fires start_async().  Wakes immediately via semaphores — no polling
// sleeps.
static void manager_thread_func() {
  while (true) {
    // Wait for a queued input.
    PROF_TS(t_input);
    while (sem_wait(&input_sem) == -1 && errno == EINTR) continue;
    PROF_ADD(input_wait_ns, t_input);
    if (stop_manager) return;

    tensors_struct* input = nullptr;
    input_tensors_queue.try_dequeue(input);  // guaranteed to succeed

    // Wait for a free InferRequest slot.
    PROF_TS(t_slot);
    while (sem_wait(&slot_sem) == -1 && errno == EINTR) continue;
    PROF_ADD(slot_wait_ns, t_slot);
    if (stop_manager) {
      deep_free_tensors_struct(input);
      return;
    }
    int idx = -1;
    free_requests.try_dequeue(idx);  // guaranteed to succeed

    logger->debug("[manager] Dispatching to slot {}.", idx);

    ov::InferRequest& req = infer_requests[idx];

    tensors_struct* output = nullptr;
    bool from_pool = pool_active.load(std::memory_order_relaxed);

    // Acquire a pool buffer (static shapes only).
    // If the pool is empty (caller freed the buffer instead of returning it
    // via runtime_return_output), allocate a fresh one so inference continues
    // uninterrupted. Zero-copy is preserved; the new buffer is freed normally
    // by the caller.
    if (from_pool) {
      PROF_TS(t_pool);
      if (!output_buffer_pool.try_dequeue(output)) output = alloc_pool_buffer();
      PROF_ADD(pool_wait_ns, t_pool);
    }

    // Tensor setup: redirect outputs to pool buffers (if pooled), then set
    // inputs.
    PROF_TS(t_setup);

    if (from_pool) {
      for (size_t i = 0; i < output_names.size(); ++i)
        req.set_output_tensor(
            i, ov::Tensor(pool_out_infos[i].ov_type, pool_out_infos[i].shape,
                          output->data[i]));
    }

    // Set input tensors (zero-copy: wraps caller's data pointer).
    size_t failed_tensor = 0;
    try {
      for (size_t i = 0; i < input->num_tensors; ++i) {
        failed_tensor = i;
        ov::Shape shape;

        for (size_t j = 0; j < input->ranks[i]; ++j)
          shape.push_back(input->shapes[i][j]);
        req.set_tensor(input->names[i],
                       ov::Tensor(map_to_ov_type(input->data_types[i]), shape,
                                  input->data[i]));
      }
    } catch (const std::exception& e) {
      PROF_ADD(tensor_setup_ns, t_setup);
      logger->error(
          "[manager] Failed to set tensor '{}' on slot {} (shape/type "
          "mismatch?): {}",
          input->names[failed_tensor], idx, e.what());
      deep_free_tensors_struct(input);
      runtime_return_output(output);
      free_requests.enqueue(idx);
      sem_post(&slot_sem);
      continue;
    }

    PROF_ADD(tensor_setup_ns, t_setup);

    // Write per-slot state — read by on_inference_complete when the callback
    // fires.
#ifdef OAAX_PROFILE
    slot_states[idx] = {input, output, from_pool, prof_now_ns()};
#else
    slot_states[idx] = {input, output, from_pool};
#endif

    logger->debug("[manager] starting inference on slot {} with input {}.", idx,
                  input->names[0]);

    req.start_async();
    PROF_INC(dispatches);
    logger->debug("[manager] start_async fired on slot {}.", idx);
  }
}

static void stop_and_join_manager() {
  stop_manager = true;
  sem_post(&slot_sem);   // unblock manager waiting for a slot
  sem_post(&input_sem);  // unblock manager waiting for input
  if (manager_thread.joinable()) manager_thread.join();

  // Wait for all callbacks still in-flight on OpenVINO's internal threads.
  for (auto& req : infer_requests) req.wait();
}

extern "C" void runtime_return_output(tensors_struct* output) {
  if (!output) return;
  if (pool_active.load(std::memory_order_relaxed))
    output_buffer_pool.enqueue(output);
  else
    deep_free_tensors_struct(output);
}

static void drain_output_pool() {
  pool_active = false;
  tensors_struct* ts;
  while (output_buffer_pool.try_dequeue(ts)) deep_free_tensors_struct(ts);
}

extern "C" int runtime_destruction() {
  logger->info("Destroying runtime...");

  stop_and_join_manager();

  // Clean up extracted zip temp directory if one is active.
  cleanup_temp_dir(model_temp_dir);
  model_temp_dir.clear();
  logger->trace("Manager stopped, all in-flight requests complete.");

  free_queue(input_tensors_queue);
  logger->trace("Freed input tensor queue.");

  free_queue(output_tensors_queue);
  logger->trace("Freed output tensor queue.");

  drain_output_pool();
  logger->trace("Freed output buffer pool.");

  {
    int idx;
    while (free_requests.try_dequeue(idx)) {
    }
  }
  sem_destroy(&slot_sem);
  sem_destroy(&input_sem);
  pool_out_infos.clear();
  slot_states.clear();

  infer_requests.clear();
  logger->trace("Freed OpenVINO infer requests.");

  compiled_model.reset();
  logger->trace("Freed OpenVINO compiled model.");

  core.reset();
  logger->debug("Runtime destroyed.");

#ifdef OAAX_PROFILE
  uint64_t n = g_profiler.dispatches.load();
  if (n > 0) {
    auto us = [](uint64_t ns_total, uint64_t count) -> double {
      return static_cast<double>(ns_total) / static_cast<double>(count) /
             1000.0;
    };
    logger->info("=== OAAX profiling report ({} inferences) ===", n);
    logger->info("  input_wait:   {:8.2f} us/inf",
                 us(g_profiler.input_wait_ns, n));
    logger->info("  slot_wait:    {:8.2f} us/inf",
                 us(g_profiler.slot_wait_ns, n));
    logger->info("  pool_wait:    {:8.2f} us/inf",
                 us(g_profiler.pool_wait_ns, n));
    logger->info("  tensor_setup: {:8.2f} us/inf",
                 us(g_profiler.tensor_setup_ns, n));
    logger->info("  inference:    {:8.2f} us/inf",
                 us(g_profiler.inference_ns, n));
    logger->info("  output_queue: {:8.2f} us/inf",
                 us(g_profiler.output_queue_ns, n));
    uint64_t tracked = g_profiler.input_wait_ns + g_profiler.slot_wait_ns +
                       g_profiler.pool_wait_ns + g_profiler.tensor_setup_ns +
                       g_profiler.inference_ns + g_profiler.output_queue_ns;
    logger->info("  total tracked:{:8.2f} us/inf", us(tracked, n));
  }
#endif

  destroy_logger(logger);
  return 0;
}

extern "C" const char* runtime_error_message() {
  return last_error.empty() ? "No error recorded." : last_error.c_str();
}

extern "C" const char* runtime_version() { return RUNTIME_VERSION; }

extern "C" const char* runtime_name() {
  return "OAAX Intel Runtime (OpenVINO Native)";
}

// Scan all OpenVINO-visible devices and log their supported precisions.
// Called once at the end of runtime_initialization() so the operator can see
// at a glance what hardware is available and what model formats will run on it.
static void log_available_devices() {
  static const std::vector<std::string> kPrecisions = {"FP32", "FP16", "BF16",
                                                       "INT8", "INT16"};
  try {
    auto devices = core->get_available_devices();
    logger->info("Available devices ({} found):", devices.size());

    for (const auto& dev : devices) {
      std::string full_name = dev;
      try {
        full_name = core->get_property(dev, ov::device::full_name);
      } catch (...) {
      }

      std::string precisions_str;
      try {
        auto caps = core->get_property(dev, ov::device::capabilities);
        for (const auto& cap : caps)
          for (const auto& p : kPrecisions)
            if (cap == p) {
              if (!precisions_str.empty()) precisions_str += ", ";
              precisions_str += cap;
              break;
            }
      } catch (...) {
      }

      if (precisions_str.empty()) precisions_str = "(unknown)";
      logger->info("  {} — {} — precisions: {}", dev, full_name,
                   precisions_str);
    }
  } catch (const std::exception& e) {
    logger->warn("Device enumeration failed: {}", e.what());
  }
}
