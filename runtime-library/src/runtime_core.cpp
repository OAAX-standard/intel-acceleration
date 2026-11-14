#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
struct sem_t {
  HANDLE h;
};
static inline int sem_init(sem_t* s, int, unsigned int value) {
  s->h = CreateSemaphoreW(nullptr, (LONG)value, 0x7FFFFFFF, nullptr);
  return s->h ? 0 : -1;
}
static inline int sem_wait(sem_t* s) {
  return WaitForSingleObject(s->h, INFINITE) == WAIT_OBJECT_0 ? 0 : -1;
}
static inline int sem_trywait(sem_t* s) {
  return WaitForSingleObject(s->h, 0) == WAIT_OBJECT_0 ? 0 : -1;
}
static inline int sem_timedwait_ms(sem_t* s, int ms) {
  return WaitForSingleObject(s->h, (DWORD)ms) == WAIT_OBJECT_0 ? 0 : -1;
}
static inline int sem_post(sem_t* s) {
  return ReleaseSemaphore(s->h, 1, nullptr) ? 0 : -1;
}
static inline int sem_destroy(sem_t* s) { return CloseHandle(s->h) ? 0 : -1; }
#else
#include <errno.h>
#include <semaphore.h>
#include <time.h>
static inline int sem_timedwait_ms(sem_t* s, int ms) {
  struct timespec deadline;
  clock_gettime(CLOCK_REALTIME, &deadline);
  deadline.tv_sec += ms / 1000;
  deadline.tv_nsec += (long)(ms % 1000) * 1000000L;
  if (deadline.tv_nsec >= 1000000000L) {
    deadline.tv_sec++;
    deadline.tv_nsec -= 1000000000L;
  }
  int r;
  do {
    r = sem_timedwait(s, &deadline);
  } while (r == -1 && errno == EINTR);
  return r;
}
#endif

#include <spdlog/spdlog.h>

#include <openvino/openvino.hpp>

#include "concurrentqueue.h"
#include "oaax_runtime.h"
#include "runtime_utils.hpp"
#include "zip_utils.hpp"

// ─── CPU capability check ────────────────────────────────────────────────────
// OpenVINO 2026.0+ requires AVX2 as the minimum x86-64 instruction set; on
// older CPUs libopenvino crashes with an illegal-instruction fault. Detect it
// up front so runtime_init can fail with a clear error instead.

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || \
    defined(_M_IX86)
#ifdef _MSC_VER
#include <intrin.h>
static bool cpu_supports_avx2() {
  int info[4];
  __cpuid(info, 1);
  // AVX2 needs OS-enabled YMM state: OSXSAVE + XCR0 bits 1-2 (XMM|YMM).
  if (!(info[2] & (1 << 27))) return false;
  if ((_xgetbv(0) & 0x6) != 0x6) return false;
  __cpuid(info, 0);
  if (info[0] < 7) return false;
  __cpuidex(info, 7, 0);
  return (info[1] & (1 << 5)) != 0;  // EBX bit 5 = AVX2
}
#else
static bool cpu_supports_avx2() { return __builtin_cpu_supports("avx2"); }
#endif
#else
// Non-x86 (e.g. ARM): the AVX2 requirement does not apply.
static bool cpu_supports_avx2() { return true; }
#endif

// ─── Internal helpers ────────────────────────────────────────────────────────

static void deep_free_tensors(Tensors* t) {
  if (!t) return;
  if (t->tensors) {
    for (int i = 0; i < t->num_tensors; ++i) {
      free(t->tensors[i].name);
      free(t->tensors[i].shape);
      free(t->tensors[i].data);
    }
    free(t->tensors);
  }
  free(t);
}

// ─── Per-model state ─────────────────────────────────────────────────────────

struct SlotState {
  Tensors* input{nullptr};
  // Set once the slot's infer request has been started. A request that never
  // ran has nothing to wait for at cleanup (and some devices report it as
  // "not finished" forever).
  std::atomic<bool> started{false};
};

struct ModelState {
  int id{0};
  ov::CompiledModel compiled_model;
  std::vector<ov::InferRequest> infer_requests;
  std::vector<SlotState> slot_states;
  moodycamel::ConcurrentQueue<Tensors*> input_queue;
  moodycamel::ConcurrentQueue<int> free_slots;
  sem_t input_sem;
  std::thread manager_thread;
  std::atomic<bool> stop{false};
  // I/O contract as declared in the IR. Tensors are addressed by index, so
  // these names are only what callers send and receive, never what the device
  // sees (see shorten_long_tensor_names()).
  std::vector<std::string> input_names;
  std::vector<ov::element::Type> input_types;
  std::vector<ov::PartialShape> input_shapes;
  std::unordered_map<std::string, size_t> input_index_by_name;
  std::vector<std::string> output_names;
  std::vector<ov::PartialShape> output_shapes;
  std::atomic<uint64_t> failed_requests{0};
  // Stand-ins for inputs a request leaves out (see zero_input()).
  std::mutex zero_inputs_mutex;
  std::vector<ov::Tensor> zero_inputs;
  std::atomic<bool> warned_missing_input{false};
  std::string temp_dir;
  std::string effective_device;
};

struct OutputItem {
  int model_id;
  Tensors* tensors;
};

// ─── Global state
// ─────────────────────────────────────────────────────────────

static std::shared_ptr<ov::Core> g_core;
static std::vector<ModelState*> g_models;
static std::atomic<bool> g_initialized{false};
static std::atomic<bool> g_models_loaded{false};

static moodycamel::ConcurrentQueue<OutputItem> g_output_queue;
static sem_t g_output_sem;

static std::shared_ptr<spdlog::logger> g_logger;
// g_last_error is written by inference threads too, hence the mutex.
// runtime_get_error() returns g_error_buf, a fixed buffer that is never
// reallocated, so a pointer handed to the caller cannot dangle.
static std::mutex g_error_mutex;
static std::string g_last_error;
static char g_error_buf[2048];
static std::string g_info_json;

// Failed requests are logged for the first few and then every 100th: a model
// that fails every inference would otherwise fill the log within seconds.
static const uint64_t kLoggedFailures = 5;
// Tensor names longer than this are replaced before compiling.
static const size_t kMaxDeviceTensorName = 128;
// How long cleanup waits for one in-flight request before cancelling it.
static const std::chrono::milliseconds kStopTimeout{30000};

// Global config defaults (can be overridden per-model)
static int g_log_level = spdlog::level::info;
static std::string g_log_file = "runtime.log";
static bool g_log_stdout = false;
static std::string g_device_type = "CPU";
static std::string g_perf_hint = "latency";
static std::string g_cache_dir = ".";
static int g_num_requests = 0;  // 0 = use ov::optimal_number_of_infer_requests
static int g_max_queue_size = 100;
static int g_num_streams = 0;  // 0 = let OpenVINO decide; >0 sets NUM_STREAMS
static int g_auto_batch_size =
    0;  // 0 = disabled; >0 wraps device in BATCH:<dev>(N)
static bool g_npu_turbo = false;  // inject NPU_TURBO=YES into compile config

// ─── Config helpers
// ───────────────────────────────────────────────────────────

static std::string config_get(const Config& cfg, const char* key,
                              const std::string& fallback) {
  for (int i = 0; i < cfg.length; ++i)
    if (cfg.keys[i] && strcmp(cfg.keys[i], key) == 0)
      return cfg.values[i] ? cfg.values[i] : fallback;
  return fallback;
}

static void config_warn_unknown(const Config& cfg,
                                std::initializer_list<const char*> known) {
  for (int i = 0; i < cfg.length; ++i) {
    if (!cfg.keys[i]) continue;
    bool found = false;
    for (const char* k : known)
      if (strcmp(cfg.keys[i], k) == 0) {
        found = true;
        break;
      }
    if (!found)
      g_logger->warn("Unknown config key '{}' — ignored", cfg.keys[i]);
  }
}

static void store_error(const std::string& msg) {
  std::lock_guard<std::mutex> lock(g_error_mutex);
  g_last_error = msg;
}

static void set_error(const std::string& msg) {
  store_error(msg);
  if (g_logger) g_logger->error("{}", msg);
}

// Runs the body of an exported function. No C++ exception may cross the C
// API: the host is C (or another compiler's C++) and would abort.
template <typename F>
static RuntimeStatus guarded(const char* function, F&& body) {
  try {
    return body();
  } catch (const std::exception& e) {
    set_error(std::string(function) + ": " + e.what());
  } catch (...) {
    set_error(std::string(function) + ": unknown exception");
  }
  return RUNTIME_STATUS_ERROR;
}

static std::string json_escape(const std::string& in) {
  std::string out;
  out.reserve(in.size());
  for (unsigned char c : in) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += (char)c;
    } else if (c < 0x20) {
      char buf[8];
      snprintf(buf, sizeof(buf), "\\u%04x", c);
      out += buf;
    } else {
      out += (char)c;
    }
  }
  return out;
}

// A request that was accepted but produced no output: count it, keep the
// reason for runtime_get_error() / runtime_get_info(), and log it (rate
// limited). The caller gets no output for this request id.
static void note_failed_request(ModelState& m, int slot_idx, int request_id,
                                const char* stage, const std::string& what) {
  uint64_t n = ++m.failed_requests;
  std::string msg = "[model " + std::to_string(m.id) + " slot " +
                    std::to_string(slot_idx) + "] " + stage +
                    " failed for request id=" + std::to_string(request_id) +
                    " (" + std::to_string(n) + " failed so far): " + what;
  store_error(msg);
  if (g_logger && (n <= kLoggedFailures || n % 100 == 0))
    g_logger->error("{}", msg);
}

// ─── Wait-for-output helper
// ───────────────────────────────────────────────────

static bool wait_for_output(int timeout_ms) {
  if (timeout_ms == 0) return sem_trywait(&g_output_sem) == 0;
  if (timeout_ms < 0) {
#ifndef _WIN32
    while (sem_wait(&g_output_sem) == -1 && errno == EINTR) continue;
    return true;
#else
    return WaitForSingleObject(g_output_sem.h, INFINITE) == WAIT_OBJECT_0;
#endif
  }
  return sem_timedwait_ms(&g_output_sem, timeout_ms) == 0;
}

// ─── Output builder ──────────────────────────────────────────────────────────
// Returns nullptr when memory runs out; may throw what OpenVINO throws.

static Tensors* build_output(int model_id, int slot_idx, int request_id) {
  ModelState& m = *g_models[model_id];
  ov::InferRequest& req = m.infer_requests[slot_idx];
  int n = (int)m.output_names.size();

  Tensors* out = (Tensors*)calloc(1, sizeof(Tensors));
  if (!out) return nullptr;
  out->id = request_id;
  out->num_tensors = n;
  // Zero-filled so deep_free_tensors() is safe on a half-built result.
  out->tensors = (TensorDescriptor*)calloc((size_t)(n > 0 ? n : 1),
                                           sizeof(TensorDescriptor));
  if (!out->tensors) {
    free(out);
    return nullptr;
  }

  try {
    for (int i = 0; i < n; ++i) {
      ov::Tensor t = req.get_output_tensor((size_t)i);
      ov::Shape shape = t.get_shape();
      size_t bytes = t.get_byte_size();
      TensorDescriptor& td = out->tensors[i];

      td.data_type = map_from_ov_type(t.get_element_type());
      td.rank = (int)shape.size();
      td.data_size = bytes;
      td.name = strdup(m.output_names[i].c_str());
      td.shape = (int*)malloc((shape.empty() ? 1 : shape.size()) * sizeof(int));
      td.data = malloc(bytes ? bytes : 1);
      if (!td.name || !td.shape || !td.data) {
        deep_free_tensors(out);
        return nullptr;
      }
      for (size_t j = 0; j < shape.size(); ++j) td.shape[j] = (int)shape[j];
      if (bytes) memcpy(td.data, t.data(), bytes);
    }
  } catch (...) {
    deep_free_tensors(out);
    throw;
  }
  return out;
}

// An input the request did not send is fed zeros. It must be set on every
// request: the slot's infer request otherwise still points at the buffer of
// the previous request, which has been freed since. Hosts do leave inputs out
// (a model input they have no value for), so this is not an error.
static ov::Tensor zero_input(ModelState& m, size_t idx) {
  std::lock_guard<std::mutex> lock(m.zero_inputs_mutex);
  if (m.zero_inputs.empty()) m.zero_inputs.resize(m.input_names.size());
  if (!m.zero_inputs[idx]) {
    ov::Tensor t(m.input_types[idx], m.input_shapes[idx].to_shape());
    memset(t.data(), 0, t.get_byte_size());
    m.zero_inputs[idx] = t;
  }
  if (!m.warned_missing_input.exchange(true))
    g_logger->warn(
        "[model {}] Input '{}' was not sent with the request; "
        "zeros are used for inputs that are left out",
        m.id, m.input_names[idx]);
  return m.zero_inputs[idx];
}

// ─── Slot dispatcher ─────────────────────────────────────────────────────────
// Called from any thread (OV callback, manager, or enqueue caller). Starts the
// request and returns true; from then on the slot owns `input`. Returns false
// with the reason in `error` when the request could not be started: the caller
// still owns `input` and the slot.

static bool dispatch_to_slot(int model_id, int slot_idx, Tensors* input,
                             std::string& error) {
  ModelState& m = *g_models[model_id];
  SlotState& s = m.slot_states[slot_idx];
  ov::InferRequest& req = m.infer_requests[slot_idx];
  s.input = input;
  try {
    std::vector<bool> given(m.input_names.size(), false);
    for (int i = 0; i < input->num_tensors; ++i) {
      TensorDescriptor& td = input->tensors[i];
      ov::Shape shape;
      for (int j = 0; j < td.rank; ++j) shape.push_back((size_t)td.shape[j]);
      size_t idx = m.input_index_by_name.at(td.name);
      given[idx] = true;
      // Zero-copy: the request reads the caller's buffer, which the slot keeps
      // alive until the inference completes.
      req.set_input_tensor(
          idx, ov::Tensor(map_to_ov_type(td.data_type), shape, td.data));
    }
    for (size_t idx = 0; idx < given.size(); ++idx)
      if (!given[idx]) req.set_input_tensor(idx, zero_input(m, idx));
    g_logger->debug("[model {} slot {}] start_async()", model_id, slot_idx);
    s.started = true;
    req.start_async();
    return true;
  } catch (const std::exception& e) {
    error = e.what();
  } catch (...) {
    error = "unknown exception";
  }
  s.input = nullptr;
  return false;
}

// Gives a slot that just finished its next queued input, or marks it idle.
// Inputs that cannot be started are dropped (and counted) so one bad request
// does not stall the ones queued behind it.
static void reload_or_release_slot(int model_id, int slot_idx) {
  ModelState& m = *g_models[model_id];
  Tensors* next = nullptr;
  while (!m.stop && m.input_queue.try_dequeue(next)) {
    g_logger->debug("[model {} slot {}] Input dequeued (input_queue={})",
                    model_id, slot_idx, m.input_queue.size_approx());
    std::string error;
    if (dispatch_to_slot(model_id, slot_idx, next, error)) return;
    note_failed_request(m, slot_idx, next->id, "Starting the inference", error);
    deep_free_tensors(next);
  }
  m.free_slots.enqueue(slot_idx);
}

// ─── Inference completion callback ───────────────────────────────────────────
// Runs on an OpenVINO thread: nothing may be thrown from here, and the slot
// must always be handed on, whatever happened to the request.

static void on_inference_complete(int model_id, int slot_idx,
                                  std::exception_ptr ex) noexcept {
  ModelState& m = *g_models[model_id];
  SlotState& s = m.slot_states[slot_idx];
  int request_id = s.input ? s.input->id : 0;
  Tensors* output = nullptr;

  try {
    if (ex) std::rethrow_exception(ex);
    output = build_output(model_id, slot_idx, request_id);
    if (!output)
      note_failed_request(m, slot_idx, request_id, "Copying the output",
                          "out of memory");
  } catch (const std::exception& e) {
    note_failed_request(m, slot_idx, request_id, "Inference", e.what());
  } catch (...) {
    note_failed_request(m, slot_idx, request_id, "Inference",
                        "unknown exception");
  }

  try {
    deep_free_tensors(s.input);
    s.input = nullptr;

    // Hot path: immediately reload this slot if input is waiting — zero GPU
    // idle. Cold path: slot goes idle; manager will pair it with the next
    // enqueue call. Also after a failed request, or the inputs queued behind
    // it would wait for a callback that never comes.
    reload_or_release_slot(model_id, slot_idx);

    if (!output) return;
    if (g_output_queue.try_enqueue(OutputItem{model_id, output})) {
      sem_post(&g_output_sem);
      g_logger->debug("[model {} slot {}] Output enqueued (output_queue={})",
                      model_id, slot_idx, g_output_queue.size_approx());
    } else {
      deep_free_tensors(output);
      note_failed_request(m, slot_idx, request_id, "Queueing the output",
                          "output queue rejected the item");
    }
  } catch (...) {
    // Only an allocation failure in a queue or the logger can end up here.
  }
}

// ─── Per-model manager thread (cold-start handler)
// ────────────────────────────────────────────────
// Hot path: on_inference_complete() reloads slots directly — no thread hop.
// This thread only runs when all slots went idle and new input arrives later.

static void manager_loop(int model_id) {
  ModelState& m = *g_models[model_id];

  while (true) {
#ifndef _WIN32
    while (sem_wait(&m.input_sem) == -1 && errno == EINTR) continue;
#else
    sem_wait(&m.input_sem);
#endif
    if (m.stop) return;

    // Try to pair an idle slot with a queued input.
    // If no idle slot: all slots are busy — the next callback will drain the
    // queue. If no input: spurious wakeup (callback or fast-path already took
    // it). Either way, just loop back.
    int slot_idx = -1;
    if (!m.free_slots.try_dequeue(slot_idx)) continue;

    Tensors* input = nullptr;
    if (!m.input_queue.try_dequeue(input)) {
      m.free_slots.enqueue(slot_idx);  // put slot back
      continue;
    }
    g_logger->debug("[model {} slot {}] Input dequeued (input_queue={})",
                    model_id, slot_idx, m.input_queue.size_approx());

    // Nothing may escape: an exception here would end the process.
    try {
      std::string error;
      if (!dispatch_to_slot(model_id, slot_idx, input, error)) {
        note_failed_request(m, slot_idx, input->id, "Starting the inference",
                            error);
        deep_free_tensors(input);
        reload_or_release_slot(model_id, slot_idx);
      }
    } catch (...) {
    }
  }
}

// ─── Device resolution
// ────────────────────────────────────────────────────────

static std::string resolve_device(const std::string& requested) {
  if (requested != "GPU") return requested;
  std::vector<std::string> gpus;
  try {
    for (const auto& d : g_core->get_available_devices())
      if (d.rfind("GPU", 0) == 0) gpus.push_back(d);
  } catch (...) {
  }
  if (gpus.size() <= 1) return requested;
  std::string multi = "MULTI:";
  for (size_t i = 0; i < gpus.size(); ++i) {
    if (i > 0) multi += ",";
    multi += gpus[i];
  }
  g_logger->info("Multiple GPUs detected — using {}", multi);
  return multi;
}

static ov::AnyMap build_perf_config(const std::string& hint,
                                    int num_streams = 0) {
  ov::AnyMap cfg;
  if (hint == "throughput")
    cfg[ov::hint::performance_mode.name()] =
        ov::hint::PerformanceMode::THROUGHPUT;
  else if (hint == "cumulative_throughput")
    cfg[ov::hint::performance_mode.name()] =
        ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT;
  else
    cfg[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
  if (num_streams > 0)
    cfg[ov::num_streams.name()] = ov::streams::Num(num_streams);
  return cfg;
}

static void log_available_devices() {
  static const std::vector<std::string> kPrec = {"FP32", "FP16", "BF16",
                                                 "INT8"};
  try {
    auto devices = g_core->get_available_devices();
    g_logger->info("Available devices ({}):", devices.size());
    for (const auto& dev : devices) {
      std::string full = dev;
      try {
        full = g_core->get_property(dev, ov::device::full_name);
      } catch (...) {
      }
      std::string prec;
      try {
        for (const auto& cap :
             g_core->get_property(dev, ov::device::capabilities))
          for (const auto& p : kPrec)
            if (cap == p) {
              if (!prec.empty()) prec += ", ";
              prec += cap;
            }
      } catch (...) {
      }
      if (prec.empty()) prec = "(unknown)";
      g_logger->info("  {} — {} — {}", dev, full, prec);
    }
  } catch (const std::exception& e) {
    g_logger->warn("Device enumeration failed: {}", e.what());
  }
}

// ─── Public API
// ───────────────────────────────────────────────────────────────

static RuntimeStatus init_impl(const Config& config) {
  if (g_initialized) {
    // Allow re-init only after cleanup
    set_error(
        "runtime_init called while already initialized — call "
        "runtime_cleanup() first");
    return RUNTIME_STATUS_ALREADY_INITIALIZED;
  }

  // Parse global config
  std::string log_level_str = config_get(config, "log_level", "2");
  g_log_file = config_get(config, "log_file", "runtime.log");
  g_log_stdout = config_get(config, "log_stdout", "0") == "1";
  g_device_type = config_get(config, "device_type", "CPU");
  g_cache_dir = config_get(config, "cache_dir", ".");
  try {
    g_num_requests = std::stoi(config_get(config, "num_requests", "0"));
    if (g_num_requests < 0) g_num_requests = 0;
  } catch (...) {
    g_num_requests = 0;
  }
  try {
    g_max_queue_size = std::stoi(config_get(config, "max_queue_size", "100"));
    if (g_max_queue_size < 0) g_max_queue_size = 100;
  } catch (...) {
    g_max_queue_size = 100;
  }
  std::string hint = config_get(config, "perf_hint", "latency");
  if (hint == "latency" || hint == "throughput" ||
      hint == "cumulative_throughput")
    g_perf_hint = hint;
  else
    g_perf_hint = "latency";

  try {
    g_num_streams = std::stoi(config_get(config, "num_streams", "0"));
    if (g_num_streams < 0) g_num_streams = 0;
  } catch (...) {
    g_num_streams = 0;
  }
  try {
    g_auto_batch_size = std::stoi(config_get(config, "auto_batch_size", "0"));
    if (g_auto_batch_size < 0) g_auto_batch_size = 0;
  } catch (...) {
    g_auto_batch_size = 0;
  }
  g_npu_turbo = config_get(config, "npu_turbo", "0") == "1";

  try {
    g_log_level = std::stoi(log_level_str);
  } catch (...) {
    g_log_level = spdlog::level::info;
  }
  if (g_log_level < 0 || g_log_level > 6) g_log_level = spdlog::level::info;

  try {
    g_logger = initialize_logger(g_log_file, g_log_level, g_log_stdout,
                                 g_log_level, runtime_get_name());
    g_logger->info("Initializing runtime v{} (OpenVINO {})", RUNTIME_VERSION,
                   ov::get_openvino_version().buildNumber);
    g_logger->info("Configuration:");
    g_logger->info("  device_type:     {}", g_device_type);
    g_logger->info("  perf_hint:       {}", g_perf_hint);
    g_logger->info("  log_level:       {}", g_log_level);
    g_logger->info("  log_file:        {}", g_log_file);
    g_logger->info("  log_stdout:      {}", g_log_stdout ? "true" : "false");
    g_logger->info(
        "  max_queue_size:  {}",
        g_max_queue_size > 0 ? std::to_string(g_max_queue_size) : "disabled");
    g_logger->info("  num_requests:    {}", g_num_requests > 0
                                                ? std::to_string(g_num_requests)
                                                : "auto");

    if (!cpu_supports_avx2()) {
      set_error(
          "CPU does not support AVX2 — OpenVINO 2026.0+ requires AVX2 as the "
          "minimum instruction set on x86-64; this runtime cannot run on this "
          "machine");
      return RUNTIME_STATUS_DEVICE_ERROR;
    }

    g_core = std::make_shared<ov::Core>();

    if (!g_cache_dir.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(g_cache_dir, ec);
      if (!ec) {
        auto canon = std::filesystem::canonical(g_cache_dir, ec);
        std::string cp = ec ? g_cache_dir : canon.string();
        g_core->set_property(ov::cache_dir(cp));
        g_logger->info("  cache_dir:       {} (active)", cp);
      } else {
        g_logger->warn(
            "  cache_dir:       cannot create '{}' — caching disabled",
            g_cache_dir);
      }
    } else {
      g_logger->info("  cache_dir:       (disabled)");
    }

    g_logger->info("  num_streams:     {}",
                   g_num_streams > 0 ? std::to_string(g_num_streams) : "auto");
    g_logger->info(
        "  auto_batch_size: {}",
        g_auto_batch_size > 0 ? std::to_string(g_auto_batch_size) : "disabled");
    g_logger->info("  npu_turbo:       {}",
                   g_npu_turbo ? "enabled" : "disabled");

    // Warn if the caller mixes a performance hint with low-level thread
    // controls. num_streams is intentionally excluded — it is a valid tuning
    // knob alongside throughput hint (especially on discrete GPU).
    for (const char* thread_key :
         {"inference_num_threads", "num_threads", "INFERENCE_NUM_THREADS"}) {
      if (config_get(config, thread_key, "").empty()) continue;
      g_logger->warn(
          "Config key '{}' is set alongside perf_hint='{}'. Do not mix "
          "performance hints with low-level thread controls — the hint manages "
          "threading internally and the manual value will be ignored or "
          "produce "
          "suboptimal results.",
          thread_key, g_perf_hint);
    }

    config_warn_unknown(
        config, {"log_level", "log_file", "log_stdout", "device_type",
                 "perf_hint", "cache_dir", "num_requests", "max_queue_size",
                 "num_streams", "auto_batch_size", "npu_turbo"});
    log_available_devices();
    // Initialize the output semaphore here so it is always ready before any
    // thread can call runtime_retrieve_output (which only checks
    // g_initialized). Initializing it inside runtime_load_models is too late: a
    // caller that starts a consumer thread right after runtime_init can reach
    // sem_timedwait before sem_init runs, and the subsequent sem_init memset
    // wipes the nwaiters field, causing sem_post to never issue futex_wake.
    sem_init(&g_output_sem, 0, 0);

#ifndef _WIN32
    // Safety net for hosts that dlclose() or exit without calling
    // runtime_cleanup(): glibc's per-DSO atexit runs this at unload, before
    // static destructors and before the code is unmapped, so runtime threads
    // are joined instead of crashing on unmapped code. Registered here (not
    // as a static destructor) so it runs before all static teardown, and
    // only once per process. Not used on Windows: DLL_PROCESS_DETACH runs
    // under the loader lock, where joining threads deadlocks — Windows hosts
    // must call runtime_cleanup() before FreeLibrary.
    static const bool cleanup_hook_registered = [] {
      std::atexit([] { runtime_cleanup(); });
      return true;
    }();
    (void)cleanup_hook_registered;
#endif

    g_initialized = true;
    g_logger->info("Runtime initialized successfully");
    return RUNTIME_STATUS_SUCCESS;
  } catch (const std::exception& e) {
    set_error(std::string("runtime_init failed: ") + e.what());
    g_core.reset();
    return RUNTIME_STATUS_ERROR;
  }
}

RuntimeStatus runtime_init(Config config) {
  return guarded("runtime_init", [&] { return init_impl(config); });
}

// ─── Model teardown ──────────────────────────────────────────────────────────
// Shared by runtime_cleanup() and a failed runtime_load_models(). Must get to
// the end whatever state the requests are in: a request that failed keeps its
// exception and rethrows it from wait(), which used to abort the cleanup and
// leave the runtime half torn down.

static void destroy_models() {
  for (auto* ms : g_models) {
    ms->stop = true;
    sem_post(&ms->input_sem);
    if (ms->manager_thread.joinable()) ms->manager_thread.join();
  }

  for (auto* ms : g_models) {
    for (size_t i = 0; i < ms->infer_requests.size(); ++i) {
      ov::InferRequest& req = ms->infer_requests[i];
      if (!ms->slot_states[i].started) continue;
      bool finished = false;
      try {
        finished = req.wait_for(kStopTimeout);
        if (!finished) {
          g_logger->warn(
              "[model {} slot {}] Still running after {} ms — "
              "cancelling",
              ms->id, i, kStopTimeout.count());
          req.cancel();
          finished = req.wait_for(kStopTimeout);
        }
      } catch (const std::exception& e) {
        // The request already ended, with an error that was reported when it
        // happened.
        finished = true;
        g_logger->debug("[model {} slot {}] Last request had failed: {}",
                        ms->id, i, e.what());
      } catch (...) {
        finished = true;
      }
      if (finished) {
        deep_free_tensors(ms->slot_states[i].input);
      } else {
        // The device still reads this input: leaking it is the safe choice.
        g_logger->error(
            "[model {} slot {}] Request did not stop; its input "
            "is leaked",
            ms->id, i);
      }
      ms->slot_states[i].input = nullptr;
    }
  }

  for (auto* ms : g_models) {
    Tensors* t;
    while (ms->input_queue.try_dequeue(t)) deep_free_tensors(t);
    sem_destroy(&ms->input_sem);
    try {
      ms->infer_requests.clear();
    } catch (...) {
    }
    cleanup_temp_dir(ms->temp_dir);
    delete ms;
  }
  g_models.clear();

  OutputItem item;
  while (g_output_queue.try_dequeue(item)) deep_free_tensors(item.tensors);
}

// Removes the model's temp dir unless release() was called.
struct TempDirGuard {
  std::string dir;
  ~TempDirGuard() { cleanup_temp_dir(dir); }
  std::string release() {
    std::string d;
    d.swap(dir);
    return d;
  }
};

// Device plugins do not all cope with long tensor names: the NPU plugin loses
// the link between a dynamic output and its shape tensor when the name exceeds
// the driver's 256-byte limit, and every inference then fails. Models with the
// class list in the output name (`bboxes-format:xyxysc;0:person;...`) hit that.
// The runtime addresses tensors by index, so the device does not need the real
// names: long ones are replaced here, and callers keep using the IR's names.
static void shorten_long_tensor_names(const std::shared_ptr<ov::Model>& model,
                                      int m_idx) {
  // Devices also derive tensor names from the nodes' own names (the node that
  // produces an output usually carries the output's name), so every long name
  // in the graph goes, not only those of the model's inputs and outputs.
  size_t renamed = 0;
  for (const auto& node : model->get_ordered_ops()) {
    if (node->get_friendly_name().size() > kMaxDeviceTensorName) {
      node->set_friendly_name("oaax_node_" + std::to_string(renamed));
      ++renamed;
    }
    for (auto& port : node->outputs()) {
      bool too_long = false;
      for (const auto& name : port.get_names())
        too_long = too_long || name.size() > kMaxDeviceTensorName;
      if (!too_long) continue;
      port.set_names({"oaax_tensor_" + std::to_string(renamed)});
      ++renamed;
    }
  }
  if (renamed)
    g_logger->info(
        "[model {}] {} name(s) longer than {} characters were "
        "shortened for the device; callers keep the model's names",
        m_idx, renamed, kMaxDeviceTensorName);
}

// Loads one model and appends it to g_models. Throws on any failure, leaving
// nothing behind.
static void load_one_model(int m_idx, const ModelConfig& mc) {
  if (!mc.file_path)
    throw std::runtime_error("model_configs[" + std::to_string(m_idx) +
                             "].file_path is null");

  // Per-model config (inherits global defaults)
  std::string device = config_get(mc.config, "device_type", g_device_type);
  std::string hint = config_get(mc.config, "perf_hint", g_perf_hint);
  std::string cdir = config_get(mc.config, "cache_dir", g_cache_dir);
  config_warn_unknown(
      mc.config, {"device_type", "perf_hint", "cache_dir", "num_requests"});

  g_logger->info("[model {}] Loading from: {}", m_idx, mc.file_path);

  TempDirGuard temp;
  std::string xml_path = extract_zip_model(mc.file_path, temp.dir);
  if (xml_path.empty())
    throw std::runtime_error(std::string("Failed to extract model archive: ") +
                             mc.file_path);
  g_logger->info("[model {}] Extracted XML: {}", m_idx, xml_path);

  auto model = g_core->read_model(xml_path);

  std::unique_ptr<ModelState> ms(new ModelState());
  ms->id = m_idx;

  // Record the model I/O contract as declared in the IR. The toolchain bakes
  // the correct input type into the model (f32 / f16 / u8), so no runtime
  // PPP conversion is needed — callers must match the IR's input type.
  for (size_t i = 0; i < model->inputs().size(); ++i) {
    const auto& inp = model->input(i);
    const auto& names = inp.get_names();
    std::string name =
        names.empty() ? "input_" + std::to_string(i) : inp.get_any_name();
    ms->input_names.push_back(name);
    ms->input_types.push_back(inp.get_element_type());
    ms->input_shapes.push_back(inp.get_partial_shape());
    ms->input_index_by_name.emplace(name, i);
    for (const auto& alias : names) ms->input_index_by_name.emplace(alias, i);
    g_logger->info("[model {}] Input  '{}': {} {}", m_idx, name,
                   inp.get_element_type().to_string(),
                   inp.get_partial_shape().to_string());
  }
  for (size_t i = 0; i < model->outputs().size(); ++i) {
    const auto& out = model->output(i);
    std::string name = out.get_names().empty() ? "output_" + std::to_string(i)
                                               : out.get_any_name();
    ms->output_names.push_back(name);
    ms->output_shapes.push_back(out.get_partial_shape());
    g_logger->info("[model {}] Output '{}': {} {}", m_idx, name,
                   out.get_element_type().to_string(),
                   out.get_partial_shape().to_string());
  }
  shorten_long_tensor_names(model, m_idx);

  std::string eff_device = resolve_device(device);

  // AUTO device: OpenVINO starts inference on CPU immediately and
  // transparently migrates to the best available accelerator in the
  // background once it finishes loading.
  if (eff_device.rfind("AUTO", 0) == 0)
    g_logger->info(
        "[model {}] AUTO device selected — inference starts on CPU and "
        "migrates to accelerator in background when ready.",
        m_idx);

  // For MULTI-device execution the recommended hint is
  // cumulative_throughput, which distributes requests across all devices
  // simultaneously. Any other hint applies per-device and leaves aggregate
  // throughput on the table.
  if (eff_device.rfind("MULTI:", 0) == 0 && hint != "cumulative_throughput") {
    g_logger->warn(
        "[model {}] Device is '{}' but perf_hint is '{}'. Use "
        "perf_hint=cumulative_throughput with MULTI to maximise aggregate "
        "throughput across all devices.",
        m_idx, eff_device, hint);
  }

  // BATCH pseudo-device: wraps the underlying device and transparently
  // aggregates concurrent single requests into hardware batches.
  // Most effective on GPU with batch sizes 4–64; not beneficial on CPU.
  if (g_auto_batch_size > 0) {
    if (eff_device == "CPU") {
      g_logger->warn(
          "[model {}] auto_batch_size={} ignored — BATCH device is not "
          "beneficial on CPU.",
          m_idx, g_auto_batch_size);
    } else {
      eff_device =
          "BATCH:" + eff_device + "(" + std::to_string(g_auto_batch_size) + ")";
      g_logger->info("[model {}] BATCH device: {}", m_idx, eff_device);
    }
  }

  // num_streams: refines GPU parallelism alongside throughput hint.
  // Recommended value for discrete Intel GPU: 4.
  int effective_streams = g_num_streams;
  if (effective_streams > 0 && hint == "latency") {
    g_logger->warn(
        "[model {}] num_streams={} has no effect with perf_hint=latency "
        "(latency hint uses a single stream).",
        m_idx, effective_streams);
    effective_streams = 0;
  }

  auto perf_cfg = build_perf_config(hint, effective_streams);

  // NPU_TURBO raises the NPU clock to its maximum sustained frequency.
  // Driver-permitting; silently ignored on non-NPU devices or older
  // drivers.
  if (g_npu_turbo) {
    bool is_npu = eff_device == "NPU" || eff_device.rfind("NPU.", 0) == 0;
    if (is_npu) {
      perf_cfg["NPU_TURBO"] = std::string("YES");
      g_logger->info("[model {}] NPU_TURBO=YES injected", m_idx);
    } else {
      g_logger->warn("[model {}] npu_turbo=1 ignored — device is '{}', not NPU",
                     m_idx, eff_device);
    }
  }

  g_logger->info("[model {}] Compiling for '{}' (hint={})...", m_idx,
                 eff_device, hint);

  ov::CompiledModel compiled;
  try {
    compiled = g_core->compile_model(model, eff_device, perf_cfg);
  } catch (const std::exception& e) {
    std::string err_msg = e.what();
    bool no_pclmulqdq = err_msg.find("pclmulqdq") != std::string::npos ||
                        err_msg.find("CRC algorithm") != std::string::npos;
    if (no_pclmulqdq) {
      // CPU lacks pclmulqdq (common in VMs); disable cache globally and
      // retry
      g_logger->warn("[model {}] CPU lacks pclmulqdq — model cache disabled",
                     m_idx);
      g_core->set_property(ov::cache_dir(""));
      compiled = g_core->compile_model(model, eff_device, perf_cfg);
    } else if (device != "CPU") {
      g_logger->warn(
          "[model {}] Compile on {} failed ({}), falling back to CPU", m_idx,
          eff_device, e.what());
      eff_device = "CPU";
      compiled = g_core->compile_model(model, "CPU", perf_cfg);
    } else {
      throw;
    }
  }

  // Tensors are addressed by index, which only works if the device kept the
  // IR's inputs and outputs as they are.
  if (compiled.inputs().size() != ms->input_names.size() ||
      compiled.outputs().size() != ms->output_names.size())
    throw std::runtime_error(
        "the compiled model does not have the inputs/outputs of the IR");

  int n_req = 1;
  try {
    n_req = (int)compiled.get_property(ov::optimal_number_of_infer_requests);
  } catch (const std::exception& e) {
    g_logger->warn(
        "[model {}] Device does not report its optimal number of "
        "infer requests ({}); using 1",
        m_idx, e.what());
  }
  try {
    int actual_streams = (int)compiled.get_property(ov::num_streams);
    g_logger->info("[model {}] Streams: {} ({}), infer requests: {}", m_idx,
                   actual_streams, effective_streams > 0 ? "user-set" : "auto",
                   n_req);
  } catch (...) {
    // not all devices expose num_streams (e.g. NPU)
    g_logger->info("[model {}] Optimal infer requests: {}", m_idx, n_req);
  }
  int forced_nreq = 0;
  try {
    forced_nreq = std::stoi(
        config_get(mc.config, "num_requests",
                   g_num_requests > 0 ? std::to_string(g_num_requests) : "0"));
  } catch (...) {
  }
  if (forced_nreq > 0) {
    g_logger->info("[model {}] num_requests overridden: {} → {}", m_idx, n_req,
                   forced_nreq);
    n_req = forced_nreq;
  }
  if (n_req < 1) n_req = 1;

  ms->compiled_model = std::move(compiled);
  ms->effective_device = eff_device;

  for (int i = 0; i < n_req; ++i)
    ms->infer_requests.emplace_back(ms->compiled_model.create_infer_request());
  ms->slot_states = std::vector<SlotState>(n_req);
  for (int i = 0; i < n_req; ++i) ms->free_slots.enqueue(i);

  // Set callbacks with captured model id and slot index
  for (int i = 0; i < n_req; ++i) {
    ms->infer_requests[i].set_callback([m_idx, i](std::exception_ptr ex) {
      on_inference_complete(m_idx, i, ex);
    });
  }

  // From here on destroy_models() owns the model, its temp dir included.
  g_models.push_back(ms.get());
  ModelState* loaded = ms.release();
  loaded->temp_dir = temp.release();
  sem_init(&loaded->input_sem, 0, 0);
  loaded->stop = false;
  loaded->manager_thread = std::thread(manager_loop, m_idx);
  g_logger->info("[model {}] Ready ({} slots, device={})", m_idx, n_req,
                 eff_device);
}

static RuntimeStatus load_models_impl(int num_models,
                                      const ModelConfig* model_configs) {
  if (!g_initialized) {
    store_error("runtime_load_models: runtime not initialized");
    return RUNTIME_STATUS_NOT_INITIALIZED;
  }
  if (g_models_loaded) {
    set_error(
        "runtime_load_models called twice — call runtime_cleanup() first");
    return RUNTIME_STATUS_ALREADY_INITIALIZED;
  }
  if (num_models <= 0 || !model_configs) {
    set_error("runtime_load_models: invalid arguments");
    return RUNTIME_STATUS_INVALID_ARGUMENT;
  }

  // On any failure the models loaded so far are removed again: it is all or
  // nothing, and the runtime stays usable for another runtime_load_models().
  try {
    for (int m_idx = 0; m_idx < num_models; ++m_idx) {
      try {
        load_one_model(m_idx, model_configs[m_idx]);
      } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load model " +
                                 std::to_string(m_idx) + ": " + e.what());
      }
    }
  } catch (const std::exception& e) {
    std::string reason = e.what();
    destroy_models();
    set_error(reason);
    return RUNTIME_STATUS_ERROR;
  } catch (...) {
    destroy_models();
    set_error("Failed to load the models: unknown exception");
    return RUNTIME_STATUS_ERROR;
  }

  g_models_loaded = true;
  g_logger->info("{} model(s) loaded and ready", num_models);
  return RUNTIME_STATUS_SUCCESS;
}

RuntimeStatus runtime_load_models(int num_models,
                                  const ModelConfig* model_configs) {
  return guarded("runtime_load_models",
                 [&] { return load_models_impl(num_models, model_configs); });
}

// Checks a request against the model before it is accepted, so the caller gets
// the reason right away. Without this a bad request was only found out when its
// turn came: it was dropped with a log line and the caller waited for an output
// that never arrived. Also keeps a too-small buffer from being read past its
// end. Returns an empty string when the request is valid.
static std::string validate_input(const ModelState& m, const Tensors* in) {
  if (in->num_tensors < 0 || (in->num_tensors > 0 && !in->tensors))
    return "invalid tensor list";
  std::vector<bool> seen(m.input_names.size(), false);
  for (int i = 0; i < in->num_tensors; ++i) {
    const TensorDescriptor& td = in->tensors[i];
    if (!td.name) return "input tensor " + std::to_string(i) + " has no name";
    std::string name = td.name;
    auto it = m.input_index_by_name.find(name);
    if (it == m.input_index_by_name.end())
      return "the model has no input named '" + name + "'";
    size_t idx = it->second;
    if (seen[idx]) return "input '" + name + "' is given twice";
    seen[idx] = true;

    ov::element::Type actual;
    try {
      actual = map_to_ov_type(td.data_type);
    } catch (...) {
      return "unsupported tensor dtype for '" + name + "'";
    }
    if (actual != m.input_types[idx])
      return "dtype mismatch for input '" + name + "': model expects " +
             m.input_types[idx].to_string() + " but caller sent " +
             actual.to_string();

    if (td.rank < 0 || (td.rank > 0 && !td.shape))
      return "input '" + name + "' has an invalid shape";
    ov::Shape shape;
    for (int j = 0; j < td.rank; ++j) {
      if (td.shape[j] < 0)
        return "input '" + name + "' has a negative dimension";
      shape.push_back((size_t)td.shape[j]);
    }
    if (!m.input_shapes[idx].compatible(ov::PartialShape(shape)))
      return "shape mismatch for input '" + name + "': model expects " +
             m.input_shapes[idx].to_string() + " but caller sent " +
             ov::PartialShape(shape).to_string();

    size_t expected = ov::shape_size(shape) * actual.size();
    if (td.data_size != expected)
      return "input '" + name + "' has " + std::to_string(td.data_size) +
             " bytes but its shape and dtype need " + std::to_string(expected);
    if (expected > 0 && !td.data) return "input '" + name + "' has no data";
  }
  // Inputs that are left out get zeros (see zero_input()), which needs a
  // known size.
  for (size_t idx = 0; idx < seen.size(); ++idx)
    if (!seen[idx] && m.input_shapes[idx].is_dynamic())
      return "input '" + m.input_names[idx] +
             "' is missing and has no fixed shape to fill in";
  return "";
}

static RuntimeStatus enqueue_impl(int model_id, Tensors* input_tensors) {
  if (!g_initialized) return RUNTIME_STATUS_NOT_INITIALIZED;
  if (!g_models_loaded) return RUNTIME_STATUS_MODEL_NOT_LOADED;
  if (model_id < 0 || model_id >= (int)g_models.size())
    return RUNTIME_STATUS_INVALID_MODEL_ID;
  if (!input_tensors) {
    set_error("runtime_enqueue_input: null input_tensors");
    return RUNTIME_STATUS_INVALID_ARGUMENT;
  }

  ModelState& m = *g_models[model_id];

  std::string problem = validate_input(m, input_tensors);
  if (!problem.empty()) {
    set_error("runtime_enqueue_input: " + problem);
    return RUNTIME_STATUS_INVALID_ARGUMENT;
  }

  if (g_max_queue_size > 0) {
    size_t in_pending = m.input_queue.size_approx();
    if ((int)in_pending >= g_max_queue_size) {
      g_logger->warn(
          "[model {}] Input queue at capacity ({}/{}), rejecting input id={}",
          model_id, in_pending, g_max_queue_size, input_tensors->id);
      set_error("runtime_enqueue_input: input queue at capacity");
      return RUNTIME_STATUS_ERROR;
    }
    size_t out_pending = g_output_queue.size_approx();
    if ((int)out_pending >= g_max_queue_size) {
      g_logger->warn(
          "[model {}] Output queue at capacity ({}/{}), rejecting input id={}",
          model_id, out_pending, g_max_queue_size, input_tensors->id);
      set_error("runtime_enqueue_input: output queue at capacity");
      return RUNTIME_STATUS_ERROR;
    }
  }

  // Fast path: an idle slot is available — dispatch directly, no thread hop.
  int idle_slot = -1;
  if (m.free_slots.try_dequeue(idle_slot)) {
    std::string error;
    if (!dispatch_to_slot(model_id, idle_slot, input_tensors, error)) {
      // Nothing was started: the caller keeps the tensors, like for any other
      // rejected input.
      m.free_slots.enqueue(idle_slot);
      set_error("runtime_enqueue_input: " + error);
      return RUNTIME_STATUS_ERROR;
    }
    g_logger->trace("[enqueue model={} id={}] fast path (direct dispatch)",
                    model_id, input_tensors->id);
    return RUNTIME_STATUS_SUCCESS;
  }

  // Slow path: all slots busy — queue for the next callback to pick up.
  // Also post input_sem as a safety net for the race where a callback goes idle
  // between our free_slots check and the input_queue.enqueue below.
  if (!m.input_queue.try_enqueue(input_tensors)) {
    set_error("runtime_enqueue_input: input queue full");
    return RUNTIME_STATUS_ERROR;
  }
  sem_post(&m.input_sem);

  g_logger->debug("[model {}] Input enqueued id={} (input_queue={})", model_id,
                  input_tensors->id, m.input_queue.size_approx());
  return RUNTIME_STATUS_SUCCESS;
}

RuntimeStatus runtime_enqueue_input(int model_id, Tensors* input_tensors) {
  return guarded("runtime_enqueue_input",
                 [&] { return enqueue_impl(model_id, input_tensors); });
}

static RuntimeStatus retrieve_impl(int* model_id, Tensors** output_tensors,
                                   int timeout_ms) {
  if (!g_initialized) return RUNTIME_STATUS_NOT_INITIALIZED;
  if (!g_models_loaded) return RUNTIME_STATUS_MODEL_NOT_LOADED;
  if (!model_id || !output_tensors) {
    set_error("runtime_retrieve_output: null output parameter");
    return RUNTIME_STATUS_INVALID_ARGUMENT;
  }

  if (!wait_for_output(timeout_ms)) {
    g_logger->trace("[retrieve] no output available (timeout={}ms)",
                    timeout_ms);
    return RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;
  }

  // sem_wait succeeded so the enqueue side already ran try_enqueue + sem_post.
  // The item is in the queue but may not be visible yet due to lock-free
  // ordering. Spin briefly before giving up; restore the semaphore count if we
  // ultimately miss so the deficit doesn't accumulate.
  OutputItem item;
  {
    int retries = 0;
    while (!g_output_queue.try_dequeue(item)) {
      if (++retries > 200) {
        sem_post(&g_output_sem);
        g_logger->warn(
            "[retrieve] dequeue miss after {} retries — semaphore restored",
            retries);
        return RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;
      }
      std::this_thread::yield();
    }
  }

  *model_id = item.model_id;
  *output_tensors = item.tensors;
  g_logger->debug("[model {}] Output retrieved id={} (output_queue={})",
                  item.model_id, item.tensors->id,
                  g_output_queue.size_approx());
  return RUNTIME_STATUS_SUCCESS;
}

RuntimeStatus runtime_retrieve_output(int* model_id, Tensors** output_tensors,
                                      int timeout_ms) {
  return guarded("runtime_retrieve_output", [&] {
    return retrieve_impl(model_id, output_tensors, timeout_ms);
  });
}

RuntimeStatus runtime_cleanup(void) {
  if (!g_initialized) return RUNTIME_STATUS_SUCCESS;  // idempotent

  // Every step runs whatever the previous one did: a cleanup that stops half
  // way leaves a runtime that can neither be used nor initialized again.
  if (g_logger) g_logger->info("Cleaning up runtime...");
  try {
    destroy_models();
  } catch (const std::exception& e) {
    if (g_logger) g_logger->error("Cleanup of the models failed: {}", e.what());
  } catch (...) {
    if (g_logger) g_logger->error("Cleanup of the models failed");
  }
  g_models.clear();
  sem_destroy(&g_output_sem);

  // Deliberately NOT calling ov::shutdown() here: per its own doc comment it
  // "delete[s] all static-duration objects allocated by [OpenVINO]" and is
  // meant for final teardown right before this library itself is unloaded —
  // not for a cleanup that a subsequent runtime_init() can follow. The OAAX
  // contract requires repeated init/load/cleanup cycles to work within the
  // same loaded module (see tests/runtime/lifecycle_test.cpp), which is
  // exactly what a pipeline/model reselect in the host does without ever
  // unloading this library. Calling ov::shutdown() here raced OpenVINO's
  // fresh-Core creation on the next runtime_init() against TBB's own
  // internal NUMA/topology teardown, crashing inside tbbbind.
  try {
    g_core.reset();
  } catch (...) {
  }
  g_initialized = false;
  g_models_loaded = false;
  store_error("");

  try {
    if (g_logger) {
      g_logger->info("Runtime cleanup complete.");
      destroy_logger(g_logger);
    }
  } catch (...) {
  }
  g_logger = nullptr;
  // Join the async logging thread last — after this, no thread whose code
  // lives in this library is left running, so the host can FreeLibrary()
  // and delete/replace the file on Windows.
  shutdown_logging();

  return RUNTIME_STATUS_SUCCESS;
}

const char* runtime_get_error(void) {
  std::lock_guard<std::mutex> lock(g_error_mutex);
  if (g_last_error.empty()) return nullptr;
  snprintf(g_error_buf, sizeof(g_error_buf), "%s", g_last_error.c_str());
  return g_error_buf;
}

const char* runtime_get_version(void) { return RUNTIME_VERSION; }

const char* runtime_get_name(void) { return "OAAX Intel"; }

static const char* get_info_impl() {
  if (!g_initialized) return nullptr;

  int in_flight = 0;
  uint64_t failed = 0;
  for (const auto* ms : g_models) {
    in_flight += (int)ms->input_queue.size_approx();
    failed += ms->failed_requests.load();
  }

  std::string bv;
  try {
    bv = ov::get_openvino_version().buildNumber;
  } catch (...) {
    bv = "unknown";
  }

  // Available hardware devices visible to OpenVINO
  std::string devices_json = "[";
  try {
    auto devs = g_core->get_available_devices();
    for (size_t i = 0; i < devs.size(); ++i) {
      if (i) devices_json += ",";
      devices_json += "\"" + json_escape(devs[i]) + "\"";
    }
  } catch (...) {
  }
  devices_json += "]";

  // Serialize a partial shape as a JSON array; dynamic dims become -1
  auto shape_json = [](const ov::PartialShape& ps) {
    std::string out = "[";
    for (size_t d = 0; d < ps.size(); ++d) {
      if (d) out += ",";
      out += ps[d].is_static() ? std::to_string(ps[d].get_length()) : "-1";
    }
    return out + "]";
  };

  // Per-model details
  std::string models_json = "[";
  for (size_t i = 0; i < g_models.size(); ++i) {
    const auto* ms = g_models[i];
    if (i) models_json += ",";
    models_json += "{";
    models_json += "\"id\":" + std::to_string(ms->id) + ",";
    models_json +=
        "\"effective_device\":\"" + json_escape(ms->effective_device) + "\",";
    models_json +=
        "\"failed_requests\":" + std::to_string(ms->failed_requests.load()) +
        ",";
    models_json +=
        "\"num_infer_requests\":" + std::to_string(ms->infer_requests.size()) +
        ",";
    models_json += "\"input_queue_depth\":" +
                   std::to_string(ms->input_queue.size_approx()) + ",";

    // Inputs: name, dtype, shape
    std::string inputs_json = "[";
    for (size_t j = 0; j < ms->input_names.size(); ++j) {
      if (j) inputs_json += ",";
      const std::string& name = ms->input_names[j];
      std::string dtype = ms->input_types[j].get_type_name();
      std::string shape = shape_json(ms->input_shapes[j]);
      inputs_json += "{\"name\":\"" + json_escape(name) + "\",\"dtype\":\"" +
                     dtype + "\",\"shape\":" + shape + "}";
    }
    inputs_json += "]";
    models_json += "\"inputs\":" + inputs_json + ",";

    // Outputs: name and shape only (dtype less relevant for callers)
    std::string outputs_json = "[";
    for (size_t j = 0; j < ms->output_names.size(); ++j) {
      if (j) outputs_json += ",";
      std::string shape = shape_json(ms->output_shapes[j]);
      outputs_json += "{\"name\":\"" + json_escape(ms->output_names[j]) +
                      "\",\"shape\":" + shape + "}";
    }
    outputs_json += "]";
    models_json += "\"outputs\":" + outputs_json;

    models_json += "}";
  }
  models_json += "]";

  g_info_json = "{";
  g_info_json += "\"loaded_models\":" + std::to_string(g_models.size()) + ",";
  g_info_json += "\"requests_in_flight\":" + std::to_string(in_flight) + ",";
  g_info_json += "\"failed_requests\":" + std::to_string(failed) + ",";
  g_info_json +=
      "\"output_queue_depth\":" + std::to_string(g_output_queue.size_approx()) +
      ",";
  g_info_json += "\"backend_version\":\"" + json_escape(bv) + "\",";
  g_info_json += "\"perf_hint\":\"" + g_perf_hint + "\",";
  g_info_json += "\"max_queue_size\":" + std::to_string(g_max_queue_size) + ",";
  g_info_json += "\"available_devices\":" + devices_json + ",";
  g_info_json += "\"models\":" + models_json;
  g_info_json += "}";

  return g_info_json.c_str();
}

const char* runtime_get_info(void) {
  try {
    return get_info_impl();
  } catch (...) {
    return nullptr;
  }
}
