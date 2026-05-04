#include "runtime_utils.hpp"

#include <spdlog/async.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <iostream>
#include <openvino/openvino.hpp>

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

std::shared_ptr<spdlog::logger> initialize_logger(const std::string &log_file,
                                                  int file_level,
                                                  bool log_to_stdout,
                                                  int console_level,
                                                  const std::string prefix) {
  try {
    // rotate_on_open=true: each launch starts a fresh runtime.log; the previous
    // three sessions are kept as runtime.1.log / .2.log / .3.log.
    auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_st>(
        log_file, 1024 * 1024 * 5, 4, true);
    file_sink->set_level(static_cast<spdlog::level::level_enum>(file_level));

    spdlog::sinks_init_list sinks;
    int effective_level = file_level;

    std::shared_ptr<spdlog::async_logger> logger;
    static auto thread_pool =
        std::make_shared<spdlog::details::thread_pool>(8192, 1);

    if (log_to_stdout) {
      auto console_sink =
          std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
      console_sink->set_level(
          static_cast<spdlog::level::level_enum>(console_level));
      effective_level = std::min(file_level, console_level);
      logger = std::make_shared<spdlog::async_logger>(
          prefix, spdlog::sinks_init_list{console_sink, file_sink}, thread_pool,
          spdlog::async_overflow_policy::overrun_oldest);
    } else {
      logger = std::make_shared<spdlog::async_logger>(
          prefix, spdlog::sinks_init_list{file_sink}, thread_pool,
          spdlog::async_overflow_policy::overrun_oldest);
    }

    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [" + prefix + "] [%^%l%$] %v");
    logger->set_level(static_cast<spdlog::level::level_enum>(effective_level));
    logger->flush_on(spdlog::level::trace);

    return logger;
  } catch (const spdlog::spdlog_ex &ex) {
    std::cerr << "Logger initialization failed: " << ex.what() << "\n";
    exit(EXIT_FAILURE);
  }
}

void destroy_logger(std::shared_ptr<spdlog::logger> logger) {
  if (logger) {
    logger->flush();
    spdlog::drop(logger->name());
    logger = nullptr;
  }
}
