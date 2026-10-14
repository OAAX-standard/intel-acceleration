#include "runtime_utils.hpp"

#include <spdlog/async.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <openvino/openvino.hpp>
#include <vector>

ov::element::Type map_to_ov_type(TensorElementType t) {
  switch (t) {
    case DATA_TYPE_FLOAT:
      return ov::element::f32;
    case DATA_TYPE_FLOAT16:
      return ov::element::f16;
    case DATA_TYPE_BFLOAT16:
      return ov::element::bf16;
    case DATA_TYPE_DOUBLE:
      return ov::element::f64;
    case DATA_TYPE_INT8:
      return ov::element::i8;
    case DATA_TYPE_INT16:
      return ov::element::i16;
    case DATA_TYPE_INT32:
      return ov::element::i32;
    case DATA_TYPE_INT64:
      return ov::element::i64;
    case DATA_TYPE_UINT8:
      return ov::element::u8;
    case DATA_TYPE_UINT16:
      return ov::element::u16;
    case DATA_TYPE_UINT32:
      return ov::element::u32;
    case DATA_TYPE_UINT64:
      return ov::element::u64;
    case DATA_TYPE_BOOL:
      return ov::element::boolean;
    case DATA_TYPE_STRING:
      return ov::element::string;
    case DATA_TYPE_INT4:
      return ov::element::i4;
    case DATA_TYPE_UINT4:
      return ov::element::u4;
    default:
      throw std::runtime_error("Unsupported or unmapped TensorElementType: " +
                               std::to_string(static_cast<int>(t)));
  }
}

TensorElementType map_from_ov_type(ov::element::Type type) {
  if (type == ov::element::f32) return DATA_TYPE_FLOAT;
  if (type == ov::element::f16) return DATA_TYPE_FLOAT16;
  if (type == ov::element::bf16) return DATA_TYPE_BFLOAT16;
  if (type == ov::element::f64) return DATA_TYPE_DOUBLE;
  if (type == ov::element::i8) return DATA_TYPE_INT8;
  if (type == ov::element::i16) return DATA_TYPE_INT16;
  if (type == ov::element::i32) return DATA_TYPE_INT32;
  if (type == ov::element::i64) return DATA_TYPE_INT64;
  if (type == ov::element::u8) return DATA_TYPE_UINT8;
  if (type == ov::element::u16) return DATA_TYPE_UINT16;
  if (type == ov::element::u32) return DATA_TYPE_UINT32;
  if (type == ov::element::u64) return DATA_TYPE_UINT64;
  if (type == ov::element::boolean) return DATA_TYPE_BOOL;
  if (type == ov::element::string) return DATA_TYPE_STRING;
  if (type == ov::element::i4) return DATA_TYPE_INT4;
  if (type == ov::element::u4) return DATA_TYPE_UINT4;
  throw std::runtime_error("Unsupported OpenVINO element type: " +
                           type.get_type_name());
}

// Owned here (not a function-local static) so shutdown_logging() can join the
// pool's worker thread during runtime_cleanup. A static would only be
// destroyed at library unload — under the loader lock on Windows, where
// joining a thread deadlocks, and the still-running worker pins the DLL.
static std::shared_ptr<spdlog::details::thread_pool> g_logging_pool;

std::shared_ptr<spdlog::logger> initialize_logger(const std::string &log_file,
                                                  int file_level,
                                                  bool log_to_stdout,
                                                  int console_level,
                                                  const std::string prefix) {
  if (!g_logging_pool)
    g_logging_pool = std::make_shared<spdlog::details::thread_pool>(8192, 1);
  auto thread_pool = g_logging_pool;

  std::vector<spdlog::sink_ptr> sinks;
  int effective_level = file_level;
  std::string file_sink_error;

  try {
    // rotate_on_open=true: each launch starts a fresh runtime.log; the previous
    // three sessions are kept as runtime.1.log / .2.log / .3.log.
    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_st>(
        log_file, 1024 * 1024 * 5, 4, true);
    file_sink->set_level(static_cast<spdlog::level::level_enum>(file_level));
    sinks.push_back(file_sink);
  } catch (const spdlog::spdlog_ex &ex) {
    // The log file may be unwritable (e.g. the host's CWD is a system
    // directory on Windows). A shared library must never terminate its host
    // over logging — fall back to console-only output and keep going.
    file_sink_error = ex.what();
  }

  if (log_to_stdout || !file_sink_error.empty()) {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    int level = log_to_stdout ? console_level : file_level;
    console_sink->set_level(static_cast<spdlog::level::level_enum>(level));
    effective_level = std::min(effective_level, level);
    sinks.push_back(console_sink);
  }

  auto logger = std::make_shared<spdlog::async_logger>(
      prefix, sinks.begin(), sinks.end(), thread_pool,
      spdlog::async_overflow_policy::overrun_oldest);

  spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [" + prefix + "] [%^%l%$] %v");
  logger->set_level(static_cast<spdlog::level::level_enum>(effective_level));
  logger->flush_on(spdlog::level::trace);

  if (!file_sink_error.empty()) {
    logger->warn("Could not open log file '{}': {} — logging to console only",
                 log_file, file_sink_error);
  }

  return logger;
}

void destroy_logger(std::shared_ptr<spdlog::logger> logger) {
  if (logger) {
    logger->flush();
    spdlog::drop(logger->name());
    logger = nullptr;
  }
}

void shutdown_logging() {
  // The pool destructor posts a terminate message and joins the worker
  // thread. Loggers hold a reference to the pool, so this must run after
  // every logger created by initialize_logger has been destroyed.
  g_logging_pool.reset();
}
