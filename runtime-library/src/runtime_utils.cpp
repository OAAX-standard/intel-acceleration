#include <iostream>
#include <vector>

#include "runtime_utils.hpp"
#include <openvino/openvino.hpp>
#include "concurrentqueue.h"

#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

using namespace std;

// Map tensor_data_type to OpenVINO element type
ov::element::Type map_to_ov_type(tensor_data_type t)
{
    switch (t)
    {
    case DATA_TYPE_FLOAT:
        return ov::element::f32;
    case DATA_TYPE_UINT8:
        return ov::element::u8;
    case DATA_TYPE_INT8:
        return ov::element::i8;
    case DATA_TYPE_UINT16:
        return ov::element::u16;
    case DATA_TYPE_INT16:
        return ov::element::i16;
    case DATA_TYPE_INT32:
        return ov::element::i32;
    case DATA_TYPE_INT64:
        return ov::element::i64;
    case DATA_TYPE_BOOL:
        return ov::element::boolean;
    case DATA_TYPE_DOUBLE:
        return ov::element::f64;
    case DATA_TYPE_UINT32:
        return ov::element::u32;
    case DATA_TYPE_UINT64:
        return ov::element::u64;
    case DATA_TYPE_FLOAT16:
        return ov::element::f16;
    default:
        throw std::runtime_error("Unsupported data type!");
    }
}

// Map OpenVINO element type to tensor_data_type
tensor_data_type map_to_tensors_struct_type(ov::element::Type type)
{
    if (type == ov::element::f32)
        return DATA_TYPE_FLOAT;
    else if (type == ov::element::u8)
        return DATA_TYPE_UINT8;
    else if (type == ov::element::i8)
        return DATA_TYPE_INT8;
    else if (type == ov::element::u16)
        return DATA_TYPE_UINT16;
    else if (type == ov::element::i16)
        return DATA_TYPE_INT16;
    else if (type == ov::element::i32)
        return DATA_TYPE_INT32;
    else if (type == ov::element::i64)
        return DATA_TYPE_INT64;
    else if (type == ov::element::boolean)
        return DATA_TYPE_BOOL;
    else if (type == ov::element::f64)
        return DATA_TYPE_DOUBLE;
    else if (type == ov::element::u32)
        return DATA_TYPE_UINT32;
    else if (type == ov::element::u64)
        return DATA_TYPE_UINT64;
    else if (type == ov::element::f16)
        return DATA_TYPE_FLOAT16;
    else
        throw std::runtime_error("Unsupported OpenVINO element type!");
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
