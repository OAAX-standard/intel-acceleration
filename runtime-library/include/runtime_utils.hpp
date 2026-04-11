
#ifndef RUNTIME_UTILS_HPP
#define RUNTIME_UTILS_HPP

#include <spdlog/spdlog.h>

#include <openvino/openvino.hpp>
#include <vector>

#include "concurrentqueue.h"
#include "runtime_core.hpp"

using namespace std;

// Map tensor_data_type to OpenVINO element type
ov::element::Type map_to_ov_type(tensor_data_type t);

// Map OpenVINO element type to tensor_data_type
tensor_data_type map_to_tensors_struct_type(ov::element::Type type);

shared_ptr<spdlog::logger> initialize_logger(
    const string &log_file, int file_level = spdlog::level::info,
    int console_level = spdlog::level::info, const string prefix = "OAAX");

void destroy_logger(std::shared_ptr<spdlog::logger> logger);

void free_queue(moodycamel::ConcurrentQueue<tensors_struct *> &queue);

#endif  // RUNTIME_UTILS_HPP
