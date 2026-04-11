#include <iostream>
#include <vector>
#include <string>
#include <numeric>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>

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

// Helper functions
static void manager_thread_func();
static void drain_output_pool();
static void stop_and_join_manager();

// OpenVINO vars
static std::shared_ptr<ov::Core> core;
static std::shared_ptr<ov::CompiledModel> compiled_model;
static std::vector<ov::InferRequest> infer_requests;

// Queue variables
static moodycamel::ConcurrentQueue<tensors_struct *> input_tensors_queue;
static moodycamel::ConcurrentQueue<tensors_struct *> output_tensors_queue;

// Output buffer pool: pre-allocated tensors_struct whose data[] pointers are
// set as the InferRequest output tensors before each start_async() call.
// OpenVINO writes inference results directly into these buffers — zero copy.
// Consumers call runtime_return_output() to return buffers instead of deep_free.
static moodycamel::ConcurrentQueue<tensors_struct *> output_buffer_pool;
static std::atomic<bool> pool_active{false};

// Per-output metadata needed to set_output_tensor before each dispatch.
// Populated at pool-allocation time; valid while pool_active is true.
struct OutInfo {
    ov::Shape          shape;
    ov::element::Type  ov_type;
    tensor_data_type   dtype;
    size_t             byte_size;
};
static std::vector<OutInfo> pool_out_infos;

// Free InferRequest slot pool: holds indices into infer_requests[] for idle slots.
// Manager dequeues a slot before dispatching; callback re-enqueues it on completion.
static moodycamel::ConcurrentQueue<int> free_requests;

// Semaphore that tracks the number of available slots.
// Manager blocks on acquire() instead of busy-sleeping; callback calls release().
// C++17-compatible (std::counting_semaphore requires C++20).
class Semaphore {
    std::mutex mtx;
    std::condition_variable cv;
    int count{0};
public:
    void release() {
        { std::lock_guard<std::mutex> lk(mtx); ++count; }
        cv.notify_one();
    }
    void release_n(int n) {
        { std::lock_guard<std::mutex> lk(mtx); count += n; }
        cv.notify_all();
    }
    void acquire() {
        std::unique_lock<std::mutex> lk(mtx);
        cv.wait(lk, [this] { return count > 0; });
        --count;
    }
    void reset() {
        std::lock_guard<std::mutex> lk(mtx);
        count = 0;
    }
};
static Semaphore slot_semaphore;

// Session variables
static vector<string> output_names;
static vector<string> input_names;

// Manager thread: one thread dispatches async inference and returns when stopped.
static std::thread manager_thread;
static std::atomic<bool> stop_manager{false};

// Logger
std::shared_ptr<spdlog::logger> logger;

// Runtime arguments
static int log_level = spdlog::level::info;
static string log_file = "runtime.log";
static string device_type = "CPU";
static string precision = "FP32";
static string perf_hint = "latency";   // "latency" | "throughput" | "cumulative_throughput"

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
        else if (key == "device_type")
        {
            device_type = string(static_cast<char *>(values[i]));
        }
        else if (key == "precision")
        {
            precision = string(static_cast<char *>(values[i]));
        }
        else if (key == "perf_hint")
        {
            string val = string(static_cast<char *>(values[i]));
            if (val == "latency" || val == "throughput" || val == "cumulative_throughput")
                perf_hint = val;
        }
    }

    return runtime_initialization();
}

extern "C" int runtime_initialization()
{
    try
    {
        logger = initialize_logger(log_file, log_level, log_level, runtime_name());
        logger->info("Initializing the runtime");

        core = std::make_shared<ov::Core>();
        logger->trace("OpenVINO Core initialized");

        logger->info("Runtime arguments:");
        logger->info("  log_level: {}", log_level);
        logger->info("  log_file: {}", log_file);
        logger->info("  device_type: {}", device_type);
        logger->info("  precision: {}", precision);
        logger->info("  perf_hint: {}", perf_hint);
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
        std::shared_ptr<ov::Model> model = core->read_model(model_path);
        logger->debug("Model read successfully from: {}", model_path);

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

        // Build compile config: performance hint (OpenVINO manages threads internally)
        ov::AnyMap config;
        if (perf_hint == "throughput")
            config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
        else if (perf_hint == "cumulative_throughput")
            config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT;
        else
            config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;

        compiled_model = std::make_shared<ov::CompiledModel>(
            core->compile_model(model, device_type, config));
        logger->debug("Model compiled for device: {} (hint={})", device_type, perf_hint);

        // OpenVINO returns 1 for LATENCY and N streams for THROUGHPUT/CUMULATIVE_THROUGHPUT.
        int actual_requests = compiled_model->get_property(ov::optimal_number_of_infer_requests);
        logger->info("Optimal infer requests for hint={}: {}", perf_hint, actual_requests);

        // Stop any existing manager and wait for in-flight async requests to complete.
        stop_and_join_manager();
        drain_output_pool();

        // Drain stale state from the previous model session.
        { int idx; while (free_requests.try_dequeue(idx)) {} }
        slot_semaphore.reset();
        pool_out_infos.clear();

        // Create one InferRequest per slot.
        infer_requests.clear();
        for (int i = 0; i < actual_requests; ++i)
            infer_requests.emplace_back(compiled_model->create_infer_request());
        logger->debug("Created {} infer request(s)", actual_requests);

        // Pre-allocate output buffer pool if all output shapes are static.
        // Pool size: 4× slot count gives headroom for max_in_flight bursts.
        int pool_size = actual_requests * 4;
        bool can_pool = true;

        for (size_t i = 0; i < output_names.size(); ++i)
        {
            auto port = compiled_model->output(output_names[i]);
            if (!port.get_partial_shape().is_static()) { can_pool = false; break; }
            ov::Shape shape = port.get_partial_shape().to_shape();
            ov::element::Type ov_type = port.get_element_type();
            tensor_data_type dtype = map_to_tensors_struct_type(ov_type);
            size_t byte_size = ov_type.size() * ov::shape_size(shape);
            pool_out_infos.push_back({shape, ov_type, dtype, byte_size});
        }

        if (can_pool)
        {
            for (int i = 0; i < pool_size; ++i)
            {
                tensors_struct *ts = allocate_tensors_struct(output_names.size());
                for (size_t j = 0; j < output_names.size(); ++j)
                {
                    ts->names[j] = strdup(output_names[j].c_str());
                    ts->ranks[j] = pool_out_infos[j].shape.size();
                    ts->shapes[j] = (size_t *)malloc(sizeof(size_t) * pool_out_infos[j].shape.size());
                    for (size_t k = 0; k < pool_out_infos[j].shape.size(); ++k)
                        ts->shapes[j][k] = pool_out_infos[j].shape[k];
                    ts->data_types[j] = pool_out_infos[j].dtype;
                    ts->data[j] = malloc(pool_out_infos[j].byte_size);
                }
                output_buffer_pool.enqueue(ts);
            }
            pool_active = true;
            logger->info("Pre-allocated {} output buffers (zero-copy pool)", pool_size);
        }
        else
        {
            logger->info("Dynamic output shapes — buffer pool disabled, using malloc per inference");
        }

        // Populate the free-slot queue and semaphore with all InferRequest indices.
        for (int i = 0; i < actual_requests; ++i)
            free_requests.enqueue(i);
        slot_semaphore.release_n(actual_requests);

        // Start the single manager thread.
        stop_manager = false;
        manager_thread = std::thread(manager_thread_func);
        logger->info("Started async manager thread ({} infer slot(s))", actual_requests);

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
    logger->debug("Enqueuing input tensors.");
    if (!input_tensors_queue.try_enqueue(input_tensors))
    {
        logger->warn("Failed to enqueue input tensors.");
        return -1;
    }
    logger->trace("Input tensors enqueued successfully.");
    return 0;
}

// Manager thread: dequeues inputs, acquires a free slot, sets input + output tensors,
// then fires start_async(). For static-shape models the output tensor is redirected to
// a pool buffer so OpenVINO writes directly into the consumer buffer — zero copy.
// FIFO ordering is NOT guaranteed when actual_requests > 1.
static void manager_thread_func()
{
    while (!stop_manager)
    {
        tensors_struct *input = nullptr;
        if (!input_tensors_queue.try_dequeue(input))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        logger->debug("[manager] Input dequeued, waiting for free slot.");

        // Block until a slot is available (woken immediately by the callback's release()).
        slot_semaphore.acquire();
        if (stop_manager)
        {
            deep_free_tensors_struct(input);
            return;
        }
        int idx = -1;
        free_requests.try_dequeue(idx); // guaranteed to succeed after acquire()

        logger->debug("[manager] Dispatching to slot {}.", idx);

        ov::InferRequest &req = infer_requests[idx];

        // Grab an output buffer and wire it up as the request's output tensor so
        // OpenVINO writes inference results directly into it (zero copy).
        // For dynamic shapes fall back to allocating in the callback.
        tensors_struct *output = nullptr;
        bool from_pool = pool_active.load(std::memory_order_relaxed);

        if (from_pool)
        {
            while (!output_buffer_pool.try_dequeue(output))
            {
                if (stop_manager)
                {
                    deep_free_tensors_struct(input);
                    free_requests.enqueue(idx);
                    slot_semaphore.release();
                    return;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            for (size_t i = 0; i < output_names.size(); ++i)
                req.set_output_tensor(i, ov::Tensor(pool_out_infos[i].ov_type,
                                                     pool_out_infos[i].shape,
                                                     output->data[i]));
        }

        // Set input tensors.
        try
        {
            for (size_t i = 0; i < input->num_tensors; ++i)
            {
                ov::Shape shape;
                for (size_t j = 0; j < input->ranks[i]; ++j)
                    shape.push_back(input->shapes[i][j]);
                ov::element::Type ov_type = map_to_ov_type(input->data_types[i]);
                req.set_tensor(input->names[i], ov::Tensor(ov_type, shape, input->data[i]));
            }
        }
        catch (const std::exception &e)
        {
            logger->error("[manager] set_tensor error on slot {}: {}", idx, e.what());
            deep_free_tensors_struct(input);
            runtime_return_output(output); // returns to pool or frees (handles nullptr)
            free_requests.enqueue(idx);
            slot_semaphore.release();
            continue;
        }

        // Completion callback: runs on OpenVINO's internal thread when inference is done.
        // Pool path: output buffer already filled by OV — just enqueue it.
        // Dynamic path: allocate output and copy from the request's output tensors.
        req.set_callback([idx, input, output, from_pool](std::exception_ptr ex)
        {
            if (ex)
            {
                try { std::rethrow_exception(ex); }
                catch (const std::exception &e) {
                    logger->error("[slot {}] Async inference error: {}", idx, e.what());
                }
                deep_free_tensors_struct(input);
                runtime_return_output(output);
                free_requests.enqueue(idx);
                slot_semaphore.release();
                return;
            }

            tensors_struct *result = output;

            if (!from_pool)
            {
                // Dynamic shapes: allocate output and copy from the request's tensors.
                ov::InferRequest &r = infer_requests[idx];
                result = allocate_tensors_struct(output_names.size());
                for (size_t i = 0; i < output_names.size(); ++i)
                {
                    ov::Tensor out = r.get_tensor(output_names[i]);
                    result->names[i] = strdup(output_names[i].c_str());
                    ov::Shape shape = out.get_shape();
                    result->ranks[i] = shape.size();
                    result->shapes[i] = (size_t *)malloc(sizeof(size_t) * shape.size());
                    for (size_t j = 0; j < shape.size(); ++j)
                        result->shapes[i][j] = shape[j];
                    result->data_types[i] = map_to_tensors_struct_type(out.get_element_type());
                    size_t sz = out.get_byte_size();
                    result->data[i] = malloc(sz);
                    memcpy(result->data[i], out.data(), sz);
                }
            }

            deep_free_tensors_struct(input);

            if (!output_tensors_queue.try_enqueue(result))
            {
                logger->error("[slot {}] Failed to enqueue output.", idx);
                runtime_return_output(result);
            }
            else
            {
                logger->debug("[slot {}] Output enqueued.", idx);
            }

            free_requests.enqueue(idx);
            slot_semaphore.release();
        });

        req.start_async();
        logger->debug("[manager] start_async fired on slot {}.", idx);
    }
}

// Stop the manager thread and wait for all in-flight async requests to complete.
static void stop_and_join_manager()
{
    stop_manager = true;
    slot_semaphore.release(); // unblock manager if it's waiting for a slot
    if (manager_thread.joinable())
        manager_thread.join();

    // Wait for callbacks that are still in-flight on OpenVINO's internal threads.
    for (auto &req : infer_requests)
        req.wait();
}

extern "C" int receive_output(tensors_struct **output_tensors)
{
    // Non-blocking: returns 0 on success, -1 when nothing is ready.
    if (!output_tensors_queue.try_dequeue(*output_tensors))
    {
        logger->trace("No output tensors available.");
        return -1;
    }
    logger->debug("Output tensors received successfully.");
    return 0;
}

extern "C" void runtime_return_output(tensors_struct *output)
{
    if (!output) return;
    if (pool_active.load(std::memory_order_relaxed))
        output_buffer_pool.enqueue(output);
    else
        deep_free_tensors_struct(output);
}

static void drain_output_pool()
{
    pool_active = false;
    tensors_struct *ts;
    while (output_buffer_pool.try_dequeue(ts))
        deep_free_tensors_struct(ts);
}

extern "C" int runtime_destruction()
{
    logger->info("Destroying runtime...");

    // Stop manager and wait for all in-flight async completions.
    stop_and_join_manager();
    logger->trace("Manager thread stopped, all in-flight requests complete.");

    free_queue(input_tensors_queue);
    logger->trace("Freed input tensor queue.");

    free_queue(output_tensors_queue);
    logger->trace("Freed output tensor queue.");

    drain_output_pool();
    logger->trace("Freed output buffer pool.");

    // Drain stale free_requests indices before clearing the vector.
    { int idx; while (free_requests.try_dequeue(idx)) {} }
    slot_semaphore.reset();
    pool_out_infos.clear();

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
