#include <atomic>
#include <cstdlib>
#include <cstring>
#include <filesystem>
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
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
  std::unordered_map<std::string, ov::element::Type> input_type_by_name;
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
static std::string g_last_error;
static std::string g_info_json;

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

static void set_error(const std::string& msg) {
  g_last_error = msg;
  if (g_logger) g_logger->error("{}", msg);
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

static Tensors* build_output(int model_id, int slot_idx, int request_id) {
  ModelState& m = *g_models[model_id];
  ov::InferRequest& req = m.infer_requests[slot_idx];
  int n = (int)m.output_names.size();

  Tensors* out = (Tensors*)malloc(sizeof(Tensors));
  if (!out) return nullptr;
  out->id = request_id;
  out->num_tensors = n;
  out->tensors =
      (TensorDescriptor*)malloc((size_t)n * sizeof(TensorDescriptor));
  if (!out->tensors) {
    free(out);
    return nullptr;
  }

  for (int i = 0; i < n; ++i) {
    ov::Tensor t = req.get_tensor(m.output_names[i]);
    ov::Shape shape = t.get_shape();

    out->tensors[i].name = strdup(m.output_names[i].c_str());
    out->tensors[i].data_type = map_from_ov_type(t.get_element_type());
    out->tensors[i].rank = (int)shape.size();
    out->tensors[i].shape = (int*)malloc(shape.size() * sizeof(int));
    for (size_t j = 0; j < shape.size(); ++j)
      out->tensors[i].shape[j] = (int)shape[j];
    out->tensors[i].data_size = t.get_byte_size();
    out->tensors[i].data = malloc(t.get_byte_size());
    memcpy(out->tensors[i].data, t.data(), t.get_byte_size());
  }
  return out;
}

// ─── Slot dispatcher ─────────────────────────────────────────────────────────
// Called from any thread (OV callback, manager, or enqueue caller).
// Takes ownership of `input`; on tensor-setup failure frees it and returns the
// slot to free_slots so the pipeline stays live.

static void dispatch_to_slot(int model_id, int slot_idx, Tensors* input) {
  ModelState& m = *g_models[model_id];
  SlotState& s = m.slot_states[slot_idx];
  s.input = input;

  ov::InferRequest& req = m.infer_requests[slot_idx];
  try {
    for (int i = 0; i < input->num_tensors; ++i) {
      TensorDescriptor& td = input->tensors[i];
      ov::Shape shape;
      for (int j = 0; j < td.rank; ++j) shape.push_back((size_t)td.shape[j]);
      req.set_tensor(td.name,
                     ov::Tensor(map_to_ov_type(td.data_type), shape, td.data));
    }
  } catch (const std::exception& e) {
    g_logger->error("[model {} slot {}] Tensor setup failed: {}", model_id,
                    slot_idx, e.what());
    deep_free_tensors(input);
    s.input = nullptr;
    m.free_slots.enqueue(slot_idx);
    return;
  }

  g_logger->debug("[model {} slot {}] start_async()", model_id, slot_idx);
  req.start_async();
}

// ─── Inference completion callback ───────────────────────────────────────────

static void on_inference_complete(int model_id, int slot_idx,
                                  std::exception_ptr ex) {
  ModelState& m = *g_models[model_id];
  SlotState& s = m.slot_states[slot_idx];
  int request_id = s.input ? s.input->id : 0;

  if (ex) {
    try {
      std::rethrow_exception(ex);
    } catch (const std::exception& e) {
      g_logger->error("[model {} slot {}] Inference error: {}", model_id,
                      slot_idx, e.what());
    }
    deep_free_tensors(s.input);
    s.input = nullptr;
    m.free_slots.enqueue(slot_idx);
    return;
  }

  Tensors* output = build_output(model_id, slot_idx, request_id);
  deep_free_tensors(s.input);
  s.input = nullptr;

  // Hot path: immediately reload this slot if input is waiting — zero GPU idle.
  // Cold path: slot goes idle; manager will pair it with the next enqueue call.
  Tensors* next = nullptr;
  if (!m.stop && m.input_queue.try_dequeue(next)) {
    g_logger->debug("[model {} slot {}] Input dequeued (input_queue={})",
                    model_id, slot_idx, m.input_queue.size_approx());
    dispatch_to_slot(model_id, slot_idx, next);
  } else {
    m.free_slots.enqueue(slot_idx);
  }

  OutputItem out_item{model_id, output};

  if (output && g_output_queue.try_enqueue(std::move(out_item))) {
    sem_post(&g_output_sem);
    g_logger->debug("[model {} slot {}] Output enqueued (output_queue={})",
                    model_id, slot_idx, g_output_queue.size_approx());
  } else {
    g_logger->error("[model {} slot {}] Failed to enqueue output — dropping.",
                    model_id, slot_idx);
    deep_free_tensors(output);
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

    dispatch_to_slot(model_id, slot_idx, input);
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

RuntimeStatus runtime_init(Config config) {
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
    g_last_error = e.what();
    if (g_logger) g_logger->error("runtime_init failed: {}", g_last_error);
    return RUNTIME_STATUS_ERROR;
  }
}

RuntimeStatus runtime_load_models(int num_models,
                                  const ModelConfig* model_configs) {
  if (!g_initialized) {
    g_last_error = "runtime_load_models: runtime not initialized";
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

  // Load each model; on any failure clean up already-created models
  for (int m_idx = 0; m_idx < num_models; ++m_idx) {
    const ModelConfig& mc = model_configs[m_idx];
    if (!mc.file_path) {
      set_error("model_configs[" + std::to_string(m_idx) +
                "].file_path is null");
      goto fail;
    }

    try {
      // Per-model config (inherits global defaults)
      std::string device = config_get(mc.config, "device_type", g_device_type);
      std::string hint = config_get(mc.config, "perf_hint", g_perf_hint);
      std::string cdir = config_get(mc.config, "cache_dir", g_cache_dir);
      config_warn_unknown(
          mc.config, {"device_type", "perf_hint", "cache_dir", "num_requests"});

      g_logger->info("[model {}] Loading from: {}", m_idx, mc.file_path);

      std::string temp_dir;
      std::string xml_path = extract_zip_model(mc.file_path, temp_dir);
      if (xml_path.empty()) {
        set_error(std::string("Failed to extract model archive: ") +
                  mc.file_path);
        cleanup_temp_dir(temp_dir);
        goto fail;
      }
      g_logger->info("[model {}] Extracted XML: {}", m_idx, xml_path);

      auto model = g_core->read_model(xml_path);

      // Log the model I/O contract as declared in the IR. The toolchain bakes
      // the correct input type into the model (f32 / f16 / u8), so no runtime
      // PPP conversion is needed — callers must match the IR's input type.
      for (size_t i = 0; i < model->inputs().size(); ++i) {
        const auto& inp = model->input(i);
        g_logger->info("[model {}] Input  '{}': {} {}", m_idx,
                       inp.get_any_name(), inp.get_element_type().to_string(),
                       inp.get_partial_shape().to_string());
      }
      for (size_t i = 0; i < model->outputs().size(); ++i) {
        const auto& out = model->output(i);
        std::string name;
        try {
          name = out.get_any_name();
        } catch (...) {
          name = "<unnamed>";
        }
        g_logger->info("[model {}] Output '{}': {} {}", m_idx, name,
                       out.get_element_type().to_string(),
                       out.get_partial_shape().to_string());
      }

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
      if (eff_device.rfind("MULTI:", 0) == 0 &&
          hint != "cumulative_throughput") {
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
          eff_device = "BATCH:" + eff_device + "(" +
                       std::to_string(g_auto_batch_size) + ")";
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
          g_logger->warn(
              "[model {}] npu_turbo=1 ignored — device is '{}', not NPU", m_idx,
              eff_device);
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
          g_logger->warn(
              "[model {}] CPU lacks pclmulqdq — model cache disabled", m_idx);
          g_core->set_property(ov::cache_dir(""));
          compiled = g_core->compile_model(model, eff_device, perf_cfg);
        } else if (device != "CPU") {
          g_logger->warn(
              "[model {}] Compile on {} failed ({}), falling back to CPU",
              m_idx, eff_device, e.what());
          eff_device = "CPU";
          compiled = g_core->compile_model(model, "CPU", perf_cfg);
        } else {
          throw;
        }
      }

      int n_req = compiled.get_property(ov::optimal_number_of_infer_requests);
      try {
        int actual_streams = (int)compiled.get_property(ov::num_streams);
        g_logger->info("[model {}] Streams: {} ({}), infer requests: {}", m_idx,
                       actual_streams,
                       effective_streams > 0 ? "user-set" : "auto", n_req);
      } catch (...) {
        // not all devices expose num_streams (e.g. NPU)
        g_logger->info("[model {}] Optimal infer requests: {}", m_idx, n_req);
      }
      int forced_nreq = 0;
      try {
        forced_nreq = std::stoi(config_get(
            mc.config, "num_requests",
            g_num_requests > 0 ? std::to_string(g_num_requests) : "0"));
      } catch (...) {
      }
      if (forced_nreq > 0) {
        g_logger->info("[model {}] num_requests overridden: {} → {}", m_idx,
                       n_req, forced_nreq);
        n_req = forced_nreq;
      }

      ModelState* ms = new ModelState();
      ms->id = m_idx;
      ms->compiled_model = std::move(compiled);
      ms->effective_device = eff_device;
      ms->temp_dir = temp_dir;

      for (const auto& inp : ms->compiled_model.inputs()) {
        ms->input_names.push_back(inp.get_any_name());
        ms->input_type_by_name[inp.get_any_name()] = inp.get_element_type();
      }
      for (const auto& out : ms->compiled_model.outputs())
        ms->output_names.push_back(out.get_any_name());

      for (int i = 0; i < n_req; ++i)
        ms->infer_requests.emplace_back(
            ms->compiled_model.create_infer_request());

      ms->slot_states.resize(n_req);
      sem_init(&ms->input_sem, 0, 0);

      for (int i = 0; i < n_req; ++i) ms->free_slots.enqueue(i);

      // Set callbacks with captured model id and slot index
      for (int i = 0; i < n_req; ++i) {
        ms->infer_requests[i].set_callback([m_idx, i](std::exception_ptr ex) {
          on_inference_complete(m_idx, i, ex);
        });
      }

      g_models.push_back(ms);
      ms->stop = false;
      ms->manager_thread = std::thread(manager_loop, m_idx);
      g_logger->info("[model {}] Ready ({} slots, device={})", m_idx, n_req,
                     eff_device);

    } catch (const std::exception& e) {
      set_error(std::string("Failed to load model ") + std::to_string(m_idx) +
                ": " + e.what());
      goto fail;
    }
  }

  g_models_loaded = true;
  g_logger->info("{} model(s) loaded and ready", num_models);
  return RUNTIME_STATUS_SUCCESS;

fail:
  // Stop and clean up all models created so far
  for (auto* ms : g_models) {
    ms->stop = true;
    sem_post(&ms->input_sem);
    if (ms->manager_thread.joinable()) ms->manager_thread.join();
    for (auto& req : ms->infer_requests) req.wait();
    Tensors* t;
    while (ms->input_queue.try_dequeue(t)) deep_free_tensors(t);
    sem_destroy(&ms->input_sem);
    cleanup_temp_dir(ms->temp_dir);
    delete ms;
  }
  g_models.clear();
  // Drain any stray output items (sem_destroy is handled by runtime_cleanup)
  OutputItem item;
  while (g_output_queue.try_dequeue(item)) deep_free_tensors(item.tensors);
  return RUNTIME_STATUS_ERROR;
}

RuntimeStatus runtime_enqueue_input(int model_id, Tensors* input_tensors) {
  if (!g_initialized) return RUNTIME_STATUS_NOT_INITIALIZED;
  if (!g_models_loaded) return RUNTIME_STATUS_MODEL_NOT_LOADED;
  if (model_id < 0 || model_id >= (int)g_models.size())
    return RUNTIME_STATUS_INVALID_MODEL_ID;
  if (!input_tensors) {
    set_error("runtime_enqueue_input: null input_tensors");
    return RUNTIME_STATUS_INVALID_ARGUMENT;
  }

  ModelState& m = *g_models[model_id];

  // Validate input dtypes against the compiled model's expected types.
  for (int i = 0; i < input_tensors->num_tensors; ++i) {
    const TensorDescriptor& td = input_tensors->tensors[i];
    if (!td.name) continue;
    auto it = m.input_type_by_name.find(std::string(td.name));
    if (it == m.input_type_by_name.end()) continue;
    try {
      ov::element::Type actual = map_to_ov_type(td.data_type);
      if (actual != it->second) {
        set_error("runtime_enqueue_input: dtype mismatch for input '" +
                  std::string(td.name) + "': model expects " +
                  it->second.to_string() + " but caller sent " +
                  actual.to_string());
        return RUNTIME_STATUS_INVALID_ARGUMENT;
      }
    } catch (...) {
      set_error("runtime_enqueue_input: unsupported tensor dtype for '" +
                std::string(td.name) + "'");
      return RUNTIME_STATUS_INVALID_ARGUMENT;
    }
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
    dispatch_to_slot(model_id, idle_slot, input_tensors);
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

RuntimeStatus runtime_retrieve_output(int* model_id, Tensors** output_tensors,
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

RuntimeStatus runtime_cleanup(void) {
  if (!g_initialized) return RUNTIME_STATUS_SUCCESS;  // idempotent

  g_logger->info("Cleaning up runtime...");

  // Stop all manager threads and wait for in-flight requests
  for (auto* ms : g_models) {
    ms->stop = true;
    sem_post(&ms->input_sem);
    if (ms->manager_thread.joinable()) ms->manager_thread.join();
    for (auto& req : ms->infer_requests) req.wait();
  }

  // Drain input queues and destroy per-model resources
  for (auto* ms : g_models) {
    Tensors* t;
    while (ms->input_queue.try_dequeue(t)) deep_free_tensors(t);
    sem_destroy(&ms->input_sem);
    ms->infer_requests.clear();
    cleanup_temp_dir(ms->temp_dir);
    delete ms;
  }
  g_models.clear();

  // Drain global output queue
  OutputItem item;
  while (g_output_queue.try_dequeue(item)) deep_free_tensors(item.tensors);
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
  // internal NUMA/topology teardown, crashing inside tbbbind (AIMP-1477).
  g_core.reset();
  g_initialized = false;
  g_models_loaded = false;
  g_last_error.clear();

  g_logger->info("Runtime cleanup complete.");
  destroy_logger(g_logger);
  g_logger = nullptr;
  // Join the async logging thread last — after this, no thread whose code
  // lives in this library is left running, so the host can FreeLibrary()
  // and delete/replace the file on Windows.
  shutdown_logging();

  return RUNTIME_STATUS_SUCCESS;
}

const char* runtime_get_error(void) {
  return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

const char* runtime_get_version(void) { return RUNTIME_VERSION; }

const char* runtime_get_name(void) { return "OAAX Intel"; }

const char* runtime_get_info(void) {
  if (!g_initialized) return nullptr;

  int in_flight = 0;
  for (const auto* ms : g_models)
    in_flight += (int)ms->input_queue.size_approx();

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
      devices_json += "\"" + devs[i] + "\"";
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
    models_json += "\"effective_device\":\"" + ms->effective_device + "\",";
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
      auto dtype_it = ms->input_type_by_name.find(name);
      std::string dtype = dtype_it != ms->input_type_by_name.end()
                              ? dtype_it->second.get_type_name()
                              : "unknown";
      std::string shape = "[]";
      try {
        shape = shape_json(ms->compiled_model.input(j).get_partial_shape());
      } catch (...) {
      }
      inputs_json += "{\"name\":\"" + name + "\",\"dtype\":\"" + dtype +
                     "\",\"shape\":" + shape + "}";
    }
    inputs_json += "]";
    models_json += "\"inputs\":" + inputs_json + ",";

    // Outputs: name and shape only (dtype less relevant for callers)
    std::string outputs_json = "[";
    for (size_t j = 0; j < ms->output_names.size(); ++j) {
      if (j) outputs_json += ",";
      std::string shape = "[]";
      try {
        shape = shape_json(ms->compiled_model.output(j).get_partial_shape());
      } catch (...) {
      }
      outputs_json +=
          "{\"name\":\"" + ms->output_names[j] + "\",\"shape\":" + shape + "}";
    }
    outputs_json += "]";
    models_json += "\"outputs\":" + outputs_json;

    models_json += "}";
  }
  models_json += "]";

  g_info_json = "{";
  g_info_json += "\"loaded_models\":" + std::to_string(g_models.size()) + ",";
  g_info_json += "\"requests_in_flight\":" + std::to_string(in_flight) + ",";
  g_info_json +=
      "\"output_queue_depth\":" + std::to_string(g_output_queue.size_approx()) +
      ",";
  g_info_json += "\"backend_version\":\"" + bv + "\",";
  g_info_json += "\"perf_hint\":\"" + g_perf_hint + "\",";
  g_info_json += "\"max_queue_size\":" + std::to_string(g_max_queue_size) + ",";
  g_info_json += "\"available_devices\":" + devices_json + ",";
  g_info_json += "\"models\":" + models_json;
  g_info_json += "}";

  return g_info_json.c_str();
}
