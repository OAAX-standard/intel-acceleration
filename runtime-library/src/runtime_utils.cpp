#include <iostream>
#include <vector>

#include "runtime_utils.hpp"
#include <onnxruntime_cxx_api.h>
#include "concurrentqueue.h"

#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

using namespace std;

vector<char *> get_output_names(Ort::Session &session)
{
    // Define the allocator
    Ort::AllocatorWithDefaultOptions allocator;
    // Read number of outputs
    size_t output_count = session.GetOutputCount();
    // Create names holder
    std::vector<char *> output_names;
    output_names.reserve(output_count);
    // Read output names
    for (size_t i = 0; i < output_count; ++i)
    {
        Ort::AllocatedStringPtr nameAllocated = session.GetOutputNameAllocated(i, allocator);
        output_names.emplace_back(nameAllocated.release()); // transfer ownership
    }
    // Return
    return output_names;
}

// Map tensor_data_type to ONNX Runtime ONNXTensorElementDataType
ONNXTensorElementDataType map_to_ort_type(tensor_data_type t)
{
    switch (t)
    {
    case DATA_TYPE_FLOAT:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case DATA_TYPE_UINT8:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case DATA_TYPE_INT8:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case DATA_TYPE_UINT16:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16;
    case DATA_TYPE_INT16:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case DATA_TYPE_INT32:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case DATA_TYPE_INT64:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    case DATA_TYPE_BOOL:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    case DATA_TYPE_DOUBLE:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE;
    case DATA_TYPE_UINT32:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32;
    case DATA_TYPE_UINT64:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64;
    default:
        throw std::runtime_error("Unsupported data type!");
    }
}

// Map ONNX Runtime ONNXTensorElementDataType to tensor_data_type
tensor_data_type map_to_tensors_struct_type(ONNXTensorElementDataType type)
{
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return DATA_TYPE_FLOAT;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return DATA_TYPE_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return DATA_TYPE_INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        return DATA_TYPE_UINT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return DATA_TYPE_INT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return DATA_TYPE_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return DATA_TYPE_INT64;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        return DATA_TYPE_BOOL;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return DATA_TYPE_DOUBLE;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
        return DATA_TYPE_UINT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        return DATA_TYPE_UINT64;
    default:
        throw std::runtime_error("Unsupported data type!");
    }
}

void free_queue(moodycamel::ConcurrentQueue<tensors_struct *> &queue)
{
    tensors_struct *tensor;
    while (queue.try_dequeue(tensor))
    {
        deep_free_tensors_struct(tensor);
    }
}

std::shared_ptr<spdlog::logger> initialize_logger(const string &log_file,
                                                  int file_level,
                                                  int console_level,
                                                  const string prefix)
{
    try
    {
        // Create a console logger
        auto console_sink = make_shared<spdlog::sinks::stdout_color_sink_mt>();
        console_sink->set_level(
            static_cast<spdlog::level::level_enum>(console_level));

        // Create a rotating file logger
        auto file_sink = make_shared<spdlog::sinks::rotating_file_sink_st>(
            log_file, 1024 * 1024 * 5, 3); // 5MB max size, 3 rotated files
        file_sink->set_level(
            static_cast<spdlog::level::level_enum>(file_level));

        // Configure the thread pool for async logging
        static auto thread_pool =
            make_shared<spdlog::details::thread_pool>(8192, 1);

        // Create the async logger with both sinks using the thread pool
        auto logger = make_shared<spdlog::async_logger>(
            prefix, spdlog::sinks_init_list{console_sink, file_sink},
            thread_pool, spdlog::async_overflow_policy::overrun_oldest);

        // Set the logging pattern
        spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [" + prefix +
                            "] [%^%l%$] %v");

        // Set the logger level to the lowest level among the sinks
        logger->set_level(static_cast<spdlog::level::level_enum>(
            std::min(file_level, console_level)));

        return logger; // Return the logger instance
    }
    catch (const spdlog::spdlog_ex &ex)
    {
        cerr << "Logger initialization failed: " << ex.what() << "\n";
        exit(EXIT_FAILURE);
    }
}

void destroy_logger(std::shared_ptr<spdlog::logger> logger)
{
    if (logger)
    {
        logger->flush();
        spdlog::drop(logger->name());
        logger = nullptr; // This destroys the logger and frees resources
    }
}