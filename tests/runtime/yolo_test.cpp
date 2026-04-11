/**
 * Integration test + benchmark: OAAX runtime with YOLOv8n / YOLOv11n OpenVINO IR models.
 *
 * Exercises the async queue architecture using dedicated producer and consumer threads:
 *   - Producer thread: calls send_input() for each request and records send timestamps.
 *   - Consumer thread: calls receive_output() as results arrive and computes per-request
 *     latency from the matching send timestamp (FIFO ordering is guaranteed by the runtime).
 *
 * Usage:
 *   ./yolo_test <model.xml> [device] [--runs N] [--warmup N]
 * Defaults: device=CPU, runs=30, warmup=5
 */

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <atomic>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>

#include "../include/runtime_core.hpp"
#include "../include/tensors_struct.h"

static const size_t YOLO_BATCH    = 1;
static const size_t YOLO_CHANNELS = 3;
static const size_t YOLO_HEIGHT   = 640;
static const size_t YOLO_WIDTH    = 640;
static const size_t YOLO_OUT_CH   = 84;
static const size_t YOLO_ANCHORS  = 8400;

using Clock = std::chrono::steady_clock;
using Ms    = std::chrono::duration<double, std::milli>;

#define CHECK(expr, msg)                                \
    do {                                                \
        if (!(expr)) {                                  \
            std::cerr << "FAIL: " << msg << std::endl; \
            runtime_destruction();                      \
            return 1;                                   \
        }                                               \
    } while (0)

static tensors_struct *make_yolo_input()
{
    tensors_struct *ts = allocate_tensors_struct(1);
    if (!ts) return nullptr;

    ts->names[0]    = strdup("images");
    ts->ranks[0]    = 4;
    ts->shapes[0]   = (size_t *)malloc(4 * sizeof(size_t));
    ts->shapes[0][0] = YOLO_BATCH;
    ts->shapes[0][1] = YOLO_CHANNELS;
    ts->shapes[0][2] = YOLO_HEIGHT;
    ts->shapes[0][3] = YOLO_WIDTH;
    ts->data_types[0] = DATA_TYPE_FLOAT;

    size_t n_bytes = YOLO_BATCH * YOLO_CHANNELS * YOLO_HEIGHT * YOLO_WIDTH * sizeof(float);
    ts->data[0] = malloc(n_bytes);
    if (!ts->data[0]) { deep_free_tensors_struct(ts); return nullptr; }
    memset(ts->data[0], 0, n_bytes);
    return ts;
}

static bool validate_output(const tensors_struct *out)
{
    if (!out || out->num_tensors != 1)           return false;
    if (out->ranks[0] != 3)                      return false;
    if (out->shapes[0][0] != YOLO_BATCH)         return false;
    if (out->shapes[0][1] != YOLO_OUT_CH)        return false;
    if (out->shapes[0][2] != YOLO_ANCHORS)       return false;
    if (out->data_types[0] != DATA_TYPE_FLOAT)   return false;

    const float *data = static_cast<const float *>(out->data[0]);
    size_t n = YOLO_BATCH * YOLO_OUT_CH * YOLO_ANCHORS;
    for (size_t i = 0; i < n; ++i)
        if (data[i] != 0.0f) return true;
    return false;  // all zeros — suspicious
}

static double percentile(std::vector<double> v, double p)
{
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)(p / 100.0 * (double)(v.size() - 1) + 0.5);
    return v[std::min(idx, v.size() - 1)];
}

/**
 * Run n inferences using separate producer and consumer threads.
 *
 * The producer is paced by an in_flight counter so it never queues more than
 * max_in_flight requests ahead of the consumer.  With max_in_flight=1 (the
 * default), the producer sends the next input only after the consumer has
 * received the previous output, giving true per-request latency without queue
 * build-up.  Increasing max_in_flight exercises the async queue depth.
 *
 * FIFO ordering in the runtime queue ensures output i matches input i.
 * Returns per-request latencies (ms), or empty on failure.
 */
static std::vector<double> run_batch(int n,
                                     std::vector<Clock::time_point> &send_times,
                                     bool validate_first,
                                     int max_in_flight = 1)
{
    std::vector<double> latencies(n);
    std::atomic<bool> ok{true};
    std::atomic<int>  in_flight{0};

    std::thread producer([&]() {
        for (int i = 0; i < n; ++i) {
            // Wait until there is room in the pipeline.
            while (in_flight.load() >= max_in_flight) {
                if (!ok) return;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            tensors_struct *input = make_yolo_input();
            if (!input) { ok = false; return; }
            send_times[i] = Clock::now();
            in_flight++;
            if (send_input(input) != 0) { ok = false; return; }
        }
    });

    std::thread consumer([&]() {
        for (int i = 0; i < n; ++i) {
            tensors_struct *output = nullptr;
            while (receive_output(&output) != 0) {
                if (!ok) return;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            latencies[i] = Ms(Clock::now() - send_times[i]).count();
            in_flight--;

            if (validate_first && i == 0 && !validate_output(output)) {
                std::cerr << "FAIL: unexpected output shape, type, or all-zero data"
                          << std::endl;
                ok = false;
            }
            deep_free_tensors_struct(output);
        }
    });

    producer.join();
    consumer.join();

    return ok ? latencies : std::vector<double>{};
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.xml> [device] [--runs N] [--warmup N] [--num-requests N]" << std::endl;
        return 1;
    }

    const char *model_path = argv[1];
    const char *device     = "CPU";
    int runs        = 30;
    int warmup      = 5;
    int num_requests = 1;

    for (int i = 2; i < argc; ++i) {
        if      (strcmp(argv[i], "--runs")         == 0 && i+1 < argc) runs         = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup")       == 0 && i+1 < argc) warmup       = atoi(argv[++i]);
        else if (strcmp(argv[i], "--num-requests") == 0 && i+1 < argc) num_requests = atoi(argv[++i]);
        else device = argv[i];
    }

    std::cout << "=== OAAX YOLO Benchmark ===" << std::endl;
    std::cout << "Model    : " << model_path  << std::endl;
    std::cout << "Device   : " << device      << std::endl;
    std::cout << "Requests : " << num_requests << std::endl;
    std::cout << "Warmup   : " << warmup      << " runs" << std::endl;
    std::cout << "Runs     : " << runs        << std::endl << std::endl;

    // ── 1. Initialize runtime ─────────────────────────────────────────────────
    std::cout << "[1] Initializing runtime..." << std::endl;
    char device_key[]   = "device_type";
    char log_key[]      = "log_level";
    char log_val[]      = "2";
    char req_key[]      = "num_requests";
    char req_val[16];   snprintf(req_val, sizeof(req_val), "%d", num_requests);
    char *keys[]        = {device_key, log_key, req_key};
    void *values[]      = {const_cast<char *>(device), log_val, req_val};

    int rc = runtime_initialization_with_args(3, keys, values);
    CHECK(rc == 0, "runtime_initialization_with_args failed (rc=" + std::to_string(rc) + ")");
    std::cout << "  ✓ " << runtime_name() << " v" << runtime_version() << std::endl;

    // ── 2. Load model ─────────────────────────────────────────────────────────
    std::cout << "[2] Loading model..." << std::endl;
    auto tload = Clock::now();
    rc = runtime_model_loading(model_path);
    CHECK(rc == 0, "runtime_model_loading failed");
    double load_ms = Ms(Clock::now() - tload).count();
    std::cout << "  ✓ Loaded in " << load_ms << " ms" << std::endl;

    // ── 3. Warmup ─────────────────────────────────────────────────────────────
    std::cout << "[3] Warming up (" << warmup << " runs)..." << std::endl;
    {
        std::vector<Clock::time_point> ts(warmup);
        CHECK(!run_batch(warmup, ts, false, num_requests).empty(), "warmup failed");
    }
    std::cout << "  ✓ Done" << std::endl;

    // ── 4. Benchmark ──────────────────────────────────────────────────────────
    // Use max_in_flight=num_requests to keep all workers fed simultaneously.
    // When num_requests=1 this gives true per-request latency; higher values
    // measure throughput at the cost of individual request latency accuracy.
    std::cout << "[4] Benchmarking (" << runs << " runs, in-flight=" << num_requests << ")..." << std::endl;
    std::vector<Clock::time_point> send_times(runs);

    auto bench_start = Clock::now();
    auto latencies = run_batch(runs, send_times, true, num_requests);
    double bench_ms = Ms(Clock::now() - bench_start).count();

    CHECK(!latencies.empty(), "benchmark failed");

    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    double mn  = *std::min_element(latencies.begin(), latencies.end());
    double mx  = *std::max_element(latencies.begin(), latencies.end());
    double p95 = percentile(latencies, 95.0);
    double fps = runs * 1000.0 / bench_ms;

    std::cout << std::endl;
    std::cout << "=== Results ===" << std::endl;
    std::cout << "  Load time  : " << load_ms << " ms" << std::endl;
    std::cout << "  Avg        : " << avg     << " ms" << std::endl;
    std::cout << "  Min        : " << mn      << " ms" << std::endl;
    std::cout << "  Max        : " << mx      << " ms" << std::endl;
    std::cout << "  p95        : " << p95     << " ms" << std::endl;
    std::cout << "  Throughput : " << fps     << " FPS" << std::endl;

    // ── 5. Destroy runtime ────────────────────────────────────────────────────
    std::cout << std::endl << "[5] Destroying runtime..." << std::endl;
    rc = runtime_destruction();
    CHECK(rc == 0, "runtime_destruction failed");
    std::cout << "  ✓ Done" << std::endl;

    return 0;
}
