
#ifndef RUNTIME_UTILS_HPP
#define RUNTIME_UTILS_HPP

#include "runtime_core.hpp"
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <spdlog/spdlog.h>
#include "concurrentqueue.h"

using namespace std;

// Get output names from the ONNX Runtime session
vector<char *> get_output_names(Ort::Session &session);

// Map tensor_data_type to ONNX Runtime ONNXTensorElementDataType
ONNXTensorElementDataType map_to_ort_type(tensor_data_type t);

// Map ONNX Runtime ONNXTensorElementDataType to tensor_data_type
tensor_data_type map_to_tensors_struct_type(ONNXTensorElementDataType type);

shared_ptr<spdlog::logger> initialize_logger(const string &log_file,
                                             int file_level = spdlog::level::info,
                                             int console_level = spdlog::level::info,
                                             const string prefix = "OAAX");

void destroy_logger(std::shared_ptr<spdlog::logger> logger);

void free_queue(moodycamel::ConcurrentQueue<tensors_struct *> &queue);

#endif // RUNTIME_UTILS_HPP