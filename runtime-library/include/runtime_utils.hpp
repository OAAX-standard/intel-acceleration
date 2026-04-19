#ifndef RUNTIME_UTILS_HPP
#define RUNTIME_UTILS_HPP

#include <spdlog/spdlog.h>

#include <openvino/openvino.hpp>
#include <string>

#include "oaax_runtime.h"

ov::element::Type map_to_ov_type(TensorElementType t);
TensorElementType map_from_ov_type(ov::element::Type type);

std::shared_ptr<spdlog::logger> initialize_logger(
    const std::string &log_file, int file_level = spdlog::level::info,
    int console_level = spdlog::level::info, const std::string prefix = "OAAX");

void destroy_logger(std::shared_ptr<spdlog::logger> logger);

#endif  // RUNTIME_UTILS_HPP
