#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <numeric>
#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <thread>
#include <atomic>

#ifdef _WIN32
#include <windows.h>
#endif

#include <onnxruntime_cxx_api.h>
#include "tensors_struct.h"
#include "concurrentqueue.h"
#include <spdlog/spdlog.h>

#include "runtime_core.hpp"
#include "runtime_utils.hpp"

using namespace std;

// Helper function
static void inference_thread_func();

// ONNX Runtime vars
static std::unique_ptr<Ort::SessionOptions> session_options;
static std::unique_ptr<Ort::Env> env;
static std::unique_ptr<Ort::Session> session;
// Queue variables
static moodycamel::ConcurrentQueue<tensors_struct *> input_tensors_queue;
static moodycamel::ConcurrentQueue<tensors_struct *> output_tensors_queue;
// Session variables
static vector<char *> output_names;
// Threads variables
static std::thread inference_thread;
static std::atomic<bool> stop_inference_thread{false};
// Logger
std::shared_ptr<spdlog::logger> logger;
// Runtime arguments
static int log_level = spdlog::level::info; // Possible values: spdlog::level::trace, debug, info, warn, err, critical, off
static string log_file = "runtime.log";
static int num_threads = 8;
static string device_type = "CPU";
static string precision = "FP32";

extern "C" int runtime_initialization_with_args(int length, char **keys, void **values)
{
    for (int i = 0; i < length; ++i)
    {
        string key = string(keys[i]);
        if (key == "log_level")
        {
            int value = std::stoi(static_cast<char *>(values[i]));
            if (value >= spdlog::level::trace && value <= spdlog::level::off)
                log_level = value;
            else
                log_level = spdlog::level::info;
        }
        else if (key == "log_file")
        {
            log_file = string(static_cast<char *>(values[i]));
        }
        else if (key == "num_threads")
        {
            num_threads = std::stoi(static_cast<char *>(values[i]));
            if (num_threads < 1)
                num_threads = 1;
            else if (num_threads > 8)
                num_threads = 8;
        }
        else if (key == "device_type")
        {
            device_type = string(static_cast<char *>(values[i]));
        }
        else if (key == "precision")
        {
            precision = string(static_cast<char *>(values[i]));
        }
        else
        {
            // Unknown key, ignore
        }
    }

    return runtime_initialization();
}

extern "C" int runtime_initialization()
{
    try
    {
        // Init logger
        logger = initialize_logger(log_file, log_level, log_level, runtime_name());
        logger->info("Initializing the runtime");
        env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_ERROR, runtime_name());
        logger->trace("ORT logging initialized");
        session_options = std::make_unique<Ort::SessionOptions>();
        logger->trace("Session options initialized");
        // Add OpenVINO execution provider
        std::unordered_map<std::string, std::string> provider_options;
        provider_options["device_type"] = device_type;
        provider_options["precision"] = precision;
        provider_options["num_of_threads"] = std::to_string(num_threads);
        provider_options["num_streams"] = "1";
        provider_options["enable_opencl_throttling"] = "false";

        session_options->AppendExecutionProvider_OpenVINO_V2(provider_options);

        stop_inference_thread = false;
        inference_thread = std::thread(inference_thread_func);
        logger->info("Inference thread started");
        logger->info("Runtime arguments:");
        logger->info("  log_level: {}", log_level);
        logger->info("  log_file: {}", log_file);
        logger->info("  num_threads: {}", num_threads);
        logger->info("  device_type: {}", device_type);
        logger->info("  precision: {}", precision);
        return 0;
    }
    catch (const std::exception &e)
    {
        logger->error("Error during runtime initialization: {}", e.what());
        return -1;
    }
}

extern "C" int runtime_model_loading(const char *model_path)
{
    logger->info("Loading model from: {}", model_path);
    try
    {
#ifdef _WIN32
        // On Windows the Ort::Session constructor expects a wide (UTF-16) string
        int size_needed = MultiByteToWideChar(CP_UTF8, 0, model_path, -1, NULL, 0);
        if (size_needed <= 0)
        {
            logger->error("Failed to convert model path to wide string.");
            return -1;
        }
        std::wstring wpath(size_needed, 0);
        MultiByteToWideChar(CP_UTF8, 0, model_path, -1, &wpath[0], size_needed);
        session = std::make_unique<Ort::Session>(*env, wpath.c_str(), *session_options);
#else
        session = std::make_unique<Ort::Session>(*env, model_path, *session_options);
#endif
        logger->debug("Model loaded successfully from: {}", model_path);
        output_names = get_output_names(*session);
        logger->trace("Output names retrieved successfully");
        return 0;
    }
    catch (const std::exception &e)
    {
        logger->error("Error during model loading: {}", e.what());
        return -1;
    }
}

extern "C" int send_input(tensors_struct *input_tensors)
{
    // Push the input tensors onto the queue
    logger->debug("Enqueuing input tensors.");
    logger->debug("Input queue contains {} tensors.", input_tensors_queue.size_approx());
    bool success = input_tensors_queue.try_enqueue(input_tensors);
    if (!success)
    {
        logger->warn("Failed to enqueue input tensors.");
        return -1;
    }
    logger->trace("Input tensors enqueued successfully.");
    return 0;
}

// Inference thread function: polls for input tensors and runs inference
static void inference_thread_func()
{
    while (!stop_inference_thread)
    {
        tensors_struct *input_tensors = nullptr;
        if (!input_tensors_queue.try_dequeue(input_tensors))
        {
            logger->trace("No input tensors available, sleeping for 10ms...");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        logger->debug("Input tensors dequeued successfully.");
        Ort::AllocatorWithDefaultOptions allocator_;
        std::vector<const char *> input_names;
        std::vector<Ort::Value> ort_inputs;
        logger->trace("Preparing input tensors for inference...");
        for (size_t i = 0; i < input_tensors->num_tensors; ++i)
        {
            logger->trace("Preparing input tensor {}...", i);
            input_names.push_back(input_tensors->names[i]);
            std::vector<int64_t> shape;
            for (size_t j = 0; j < input_tensors->ranks[i]; ++j)
                shape.push_back(static_cast<int64_t>(input_tensors->shapes[i][j]));
            ONNXTensorElementDataType dtype = map_to_ort_type(input_tensors->data_types[i]);
            int dtype_size = get_data_type_byte_size(input_tensors->data_types[i]);
            int tensor_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>()) * dtype_size;
            ort_inputs.push_back(Ort::Value::CreateTensor(
                Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault),
                input_tensors->data[i],
                tensor_size,
                shape.data(),
                shape.size(),
                dtype));
        }
        logger->debug("Performing inference...");
        std::vector<Ort::Value> ort_outputs = session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            ort_inputs.data(),
            ort_inputs.size(),
            output_names.data(),
            output_names.size());
        logger->debug("Inference completed.");
        deep_free_tensors_struct(input_tensors);
        logger->trace("Freed input tensors.");
        tensors_struct *output_tensors = allocate_tensors_struct(ort_outputs.size());
        logger->trace("Building output tensors.");
        for (size_t i = 0; i < ort_outputs.size(); ++i)
        {
            logger->trace("Building output tensor {}...", i);
            size_t name_len = strlen(output_names[i]);
            output_tensors->names[i] = (char *)malloc(name_len + 1);
            strcpy(output_tensors->names[i], output_names[i]);

            std::vector<int64_t> shape = ort_outputs[i].GetTensorTypeAndShapeInfo().GetShape();
            output_tensors->ranks[i] = shape.size();
            output_tensors->shapes[i] = (size_t *)malloc(sizeof(size_t) * shape.size());
            for (size_t j = 0; j < shape.size(); ++j)
                output_tensors->shapes[i][j] = static_cast<size_t>(shape[j]);

            output_tensors->data_types[i] = map_to_tensors_struct_type(ort_outputs[i].GetTensorTypeAndShapeInfo().GetElementType());

            size_t data_size = ort_outputs[i].GetTensorTypeAndShapeInfo().GetElementCount() *
                               get_data_type_byte_size(output_tensors->data_types[i]);
            output_tensors->data[i] = (void *)malloc(data_size);
            if (!output_tensors->data[i])
            {
                throw std::runtime_error("Failed to allocate memory for output tensor data.");
            }
            memcpy(output_tensors->data[i], ort_outputs[i].GetTensorMutableData<void>(), data_size);
        }
        logger->trace("Output tensors built successfully.");
        int success = output_tensors_queue.try_enqueue(output_tensors);
        if (!success)
        {
            logger->error("Failed to enqueue output tensors.");
        }
        logger->debug("Output tensors enqueued successfully.");
    }
}

extern "C" int receive_output(tensors_struct **output_tensors)
{
    logger->debug("Waiting for output tensors...");
    logger->debug("Output queue contains {} tensors.", output_tensors_queue.size_approx());
    // Check if there are any output tensors in the queue
    if (output_tensors_queue.size_approx() == 0)
    {
        // Sleep for 100ms
        logger->debug("No output tensors available, sleeping for 100ms...");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        logger->trace("Woke up from sleep.");
        return -1;
    }
    // Get the next output tensor from the queue
    if (!output_tensors_queue.try_dequeue(*output_tensors))
    {
        logger->error("Failed to dequeue output tensors.");
        return -1;
    }
    logger->debug("Output tensors received successfully.");
    return 0;
}

extern "C" int runtime_destruction()
{
    logger->info("Destroying runtime...");
    stop_inference_thread = true;
    logger->trace("Waiting for inference thread to stop...");
    if (inference_thread.joinable())
        inference_thread.join();
    logger->trace("Inference thread stopped.");
    for (auto name : output_names)
        free(name);
    logger->trace("Freed output tensor names.");
    free_queue(input_tensors_queue);
    logger->trace("Freed input tensor queue.");
    free_queue(output_tensors_queue);
    logger->trace("Freed output tensor queue.");
    session.reset();
    logger->trace("Freed ONNX runtime session.");
    session_options.reset();
    logger->trace("Freed ONNX runtime session options.");
    env.reset();
    logger->debug("Runtime destroyed.");
    destroy_logger(logger);
    return 0;
}

extern "C" const char *runtime_error_message()
{
    return "Check the stdout and/or log files for any error message.";
}

extern "C" const char *runtime_version()
{
    return RUNTIME_VERSION;
}

extern "C" const char *runtime_name()
{
    return "OAAX Intel Runtime";
}