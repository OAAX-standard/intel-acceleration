#include <iostream>
#include <vector>
#include <string>
#include <queue>
#include <numeric>
#include <thread>
#include <atomic>

#ifdef _WIN32
#include <windows.h>
#endif

#include <openvino/openvino.hpp>
#include "tensors_struct.h"
#include "concurrentqueue.h"
#include <spdlog/spdlog.h>

#include "runtime_core.hpp"
#include "runtime_utils.hpp"

using namespace std;

// Helper function
static void inference_thread_func(int thread_idx);

// OpenVINO vars
static std::shared_ptr<ov::Core> core;
static std::shared_ptr<ov::CompiledModel> compiled_model;
static std::vector<ov::InferRequest> infer_requests;

// Queue variables
static moodycamel::ConcurrentQueue<tensors_struct *> input_tensors_queue;
static moodycamel::ConcurrentQueue<tensors_struct *> output_tensors_queue;

// Session variables
static vector<string> output_names;
static vector<string> input_names;

// Threads variables
static std::vector<std::thread> inference_threads;
static std::atomic<bool> stop_inference_thread{false};

// Logger
std::shared_ptr<spdlog::logger> logger;

// Runtime arguments
static int log_level = spdlog::level::info;
static string log_file = "runtime.log";
static int num_threads = 8;
static int num_requests = 1;
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
        else if (key == "num_requests")
        {
            num_requests = std::stoi(static_cast<char *>(values[i]));
            if (num_requests < 1)
                num_requests = 1;
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

        // Initialize OpenVINO Core
        core = std::make_shared<ov::Core>();
        logger->trace("OpenVINO Core initialized");

        // Configure device properties
        if (device_type == "CPU")
        {
            core->set_property(device_type, ov::inference_num_threads(num_threads));
            logger->trace("CPU inference threads set to {}", num_threads);
        }

        logger->info("Runtime arguments:");
        logger->info("  log_level: {}", log_level);
        logger->info("  log_file: {}", log_file);
        logger->info("  num_threads: {}", num_threads);
        logger->info("  num_requests: {}", num_requests);
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
        // Read the model (OpenVINO IR format: .xml + .bin or .xml only)
        std::shared_ptr<ov::Model> model = core->read_model(model_path);
        logger->debug("Model read successfully from: {}", model_path);

        // Get input and output names
        input_names.clear();
        output_names.clear();

        for (const auto& input : model->inputs())
        {
            input_names.push_back(input.get_any_name());
            logger->trace("Input: {}", input.get_any_name());
        }

        for (const auto& output : model->outputs())
        {
            output_names.push_back(output.get_any_name());
            logger->trace("Output: {}", output.get_any_name());
        }

        // Compile the model for the target device
        compiled_model = std::make_shared<ov::CompiledModel>(
            core->compile_model(model, device_type));
        logger->debug("Model compiled successfully for device: {}", device_type);

        // Stop any existing inference threads before replacing the request pool
        if (!inference_threads.empty())
        {
            stop_inference_thread = true;
            for (auto& t : inference_threads)
                if (t.joinable()) t.join();
            inference_threads.clear();
        }

        // Create one InferRequest per worker thread
        infer_requests.clear();
        for (int i = 0; i < num_requests; ++i)
            infer_requests.emplace_back(compiled_model->create_infer_request());
        logger->debug("Created {} infer request(s)", num_requests);

        // Start inference worker threads
        stop_inference_thread = false;
        for (int i = 0; i < num_requests; ++i)
            inference_threads.emplace_back(inference_thread_func, i);
        logger->info("Started {} inference thread(s)", num_requests);

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

// Inference thread function: each thread owns infer_requests[thread_idx] exclusively.
// All threads compete on the shared input queue and post to the shared output queue.
// FIFO ordering is NOT guaranteed when num_requests > 1.
static void inference_thread_func(int thread_idx)
{
    ov::InferRequest& req = infer_requests[thread_idx];

    while (!stop_inference_thread)
    {
        tensors_struct *input_tensors = nullptr;
        if (!input_tensors_queue.try_dequeue(input_tensors))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        logger->debug("[thread {}] Input tensors dequeued.", thread_idx);

        try
        {
            // Set input tensors
            for (size_t i = 0; i < input_tensors->num_tensors; ++i)
            {
                std::string tensor_name = input_tensors->names[i];

                ov::Shape shape;
                for (size_t j = 0; j < input_tensors->ranks[i]; ++j)
                    shape.push_back(input_tensors->shapes[i][j]);

                ov::element::Type ov_type = map_to_ov_type(input_tensors->data_types[i]);
                ov::Tensor input_tensor(ov_type, shape, input_tensors->data[i]);
                req.set_tensor(tensor_name, input_tensor);
            }

            logger->debug("[thread {}] Performing inference...", thread_idx);
            req.infer();
            logger->debug("[thread {}] Inference completed.", thread_idx);

            deep_free_tensors_struct(input_tensors);

            // Build output tensors
            tensors_struct *output_tensors = allocate_tensors_struct(output_names.size());

            for (size_t i = 0; i < output_names.size(); ++i)
            {
                // Get output tensor
                ov::Tensor output_tensor = req.get_tensor(output_names[i]);

                // Set name
                size_t name_len = output_names[i].length();
                output_tensors->names[i] = (char *)malloc(name_len + 1);
                strcpy(output_tensors->names[i], output_names[i].c_str());

                // Set shape
                ov::Shape shape = output_tensor.get_shape();
                output_tensors->ranks[i] = shape.size();
                output_tensors->shapes[i] = (size_t *)malloc(sizeof(size_t) * shape.size());
                for (size_t j = 0; j < shape.size(); ++j)
                {
                    output_tensors->shapes[i][j] = shape[j];
                }

                // Set data type
                output_tensors->data_types[i] = map_to_tensors_struct_type(output_tensor.get_element_type());

                // Copy data
                size_t data_size = output_tensor.get_byte_size();
                output_tensors->data[i] = (void *)malloc(data_size);
                if (!output_tensors->data[i])
                {
                    throw std::runtime_error("Failed to allocate memory for output tensor data.");
                }
                memcpy(output_tensors->data[i], output_tensor.data(), data_size);
            }

            if (!output_tensors_queue.try_enqueue(output_tensors))
            {
                logger->error("[thread {}] Failed to enqueue output tensors.", thread_idx);
                deep_free_tensors_struct(output_tensors);
            }
            else
            {
                logger->debug("[thread {}] Output tensors enqueued.", thread_idx);
            }
        }
        catch (const std::exception &e)
        {
            logger->error("[thread {}] Inference error: {}", thread_idx, e.what());
            deep_free_tensors_struct(input_tensors);
        }
    }
}

extern "C" int receive_output(tensors_struct **output_tensors)
{
    // Non-blocking: returns 0 with output on success, -1 when nothing is ready.
    // Callers are responsible for their own retry/sleep policy.
    if (!output_tensors_queue.try_dequeue(*output_tensors))
    {
        logger->trace("No output tensors available.");
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

    for (auto& t : inference_threads)
        if (t.joinable()) t.join();
    inference_threads.clear();
    logger->trace("Inference threads stopped.");

    free_queue(input_tensors_queue);
    logger->trace("Freed input tensor queue.");

    free_queue(output_tensors_queue);
    logger->trace("Freed output tensor queue.");

    infer_requests.clear();
    logger->trace("Freed OpenVINO infer requests.");

    compiled_model.reset();
    logger->trace("Freed OpenVINO compiled model.");

    core.reset();
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
    return "OAAX Intel Runtime (OpenVINO Native)";
}
