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
    bool log_to_stdout = false, int console_level = spdlog::level::info,
    const std::string prefix = "OAAX");

void destroy_logger(std::shared_ptr<spdlog::logger> logger);

// Joins and destroys the async logging thread pool. Call after the last
// logger is destroyed (runtime_cleanup) so no thread whose code lives in
// this library outlives cleanup — on Windows a surviving thread pins the
// DLL in the host process, and joining it later from DllMain (static
// destruction under the loader lock) can deadlock.
void shutdown_logging();

#endif  // RUNTIME_UTILS_HPP
