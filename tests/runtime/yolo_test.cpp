/**
 * Integration benchmark: OAAX v2 runtime with YOLOv8n / YOLOv11n models.
 *
 * Uses dedicated producer/consumer threads to exercise the async queue.
 *
 * Usage:
 *   ./yolo_test <model.zip> [device] [--runs N] [--warmup N] [--batch N]
 *               [--perf-hint latency|throughput]
 * Defaults: device=CPU, runs=30, warmup=5, batch=1
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "oaax_runtime.h"

static const int YOLO_CHANNELS = 3;
static const int YOLO_HEIGHT = 640;
static const int YOLO_WIDTH = 640;
static const int YOLO_OUT_CH = 84;
static const int YOLO_ANCHORS = 8400;

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

#define CHECK(expr, msg)                                 \
    do {                                                 \
        if (!(expr)) {                                   \
            std::cerr << "FAIL: " << (msg) << std::endl; \
            runtime_cleanup();                           \
            return 1;                                    \
        }                                                \
    } while (0)

// ─── Helpers ─────────────────────────────────────────────────────────────────

static void free_tensors(Tensors *t) {
    if (!t) return;
    for (int i = 0; i < t->num_tensors; ++i) {
        free(t->tensors[i].name);
        free(t->tensors[i].shape);
        free(t->tensors[i].data);
    }
    free(t->tensors);
    free(t);
}

static Tensors *make_yolo_input(int batch, int request_id) {
    Tensors *ts = (Tensors *)malloc(sizeof(Tensors));
    if (!ts) return nullptr;
    ts->id = request_id;
    ts->num_tensors = 1;
    ts->tensors = (TensorDescriptor *)malloc(sizeof(TensorDescriptor));
    if (!ts->tensors) {
        free(ts);
        return nullptr;
    }

    TensorDescriptor &td = ts->tensors[0];
    td.name = strdup("images");
    td.data_type = DATA_TYPE_FLOAT;
    td.rank = 4;
    td.shape = (int *)malloc(4 * sizeof(int));
    td.shape[0] = batch;
    td.shape[1] = YOLO_CHANNELS;
    td.shape[2] = YOLO_HEIGHT;
    td.shape[3] = YOLO_WIDTH;
    size_t n_bytes = (size_t)batch * YOLO_CHANNELS * YOLO_HEIGHT * YOLO_WIDTH * sizeof(float);
    td.data_size = n_bytes;
    td.data = calloc(1, n_bytes);
    if (!td.data) {
        free(td.shape);
        free(td.name);
        free(ts->tensors);
        free(ts);
        return nullptr;
    }
    return ts;
}

static bool validate_output(const Tensors *out, int batch) {
    if (!out || out->num_tensors != 1) return false;
    const TensorDescriptor &td = out->tensors[0];
    if (td.rank != 3) return false;
    if (td.shape[0] != batch) return false;
    if (td.shape[1] != YOLO_OUT_CH) return false;
    if (td.shape[2] != YOLO_ANCHORS) return false;
    if (td.data_type != DATA_TYPE_FLOAT) return false;
    return true;
}

static double percentile(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)(p / 100.0 * (double)(v.size() - 1) + 0.5);
    return v[std::min(idx, v.size() - 1)];
}

/**
 * Run n inferences using separate producer/consumer threads.
 * max_in_flight controls pipeline depth.
 * Returns per-request latencies (ms), or empty on failure.
 */
static std::vector<double> run_batch(int n, std::vector<Clock::time_point> &send_times, bool validate_first, int batch,
                                     int max_in_flight = 1) {
    std::vector<double> latencies(n);
    std::atomic<bool> ok{true};
    std::atomic<int> in_flight{0};

    std::thread producer([&]() {
        for (int i = 0; i < n; ++i) {
            while (in_flight.load() >= max_in_flight) {
                if (!ok) return;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            Tensors *input = make_yolo_input(batch, i);
            if (!input) {
                ok = false;
                return;
            }
            send_times[i] = Clock::now();
            in_flight++;
            if (runtime_enqueue_input(0, input) != RUNTIME_STATUS_SUCCESS) {
                // On failure caller must free (runtime didn't take ownership)
                free_tensors(input);
                ok = false;
                return;
            }
        }
    });

    std::thread consumer([&]() {
        for (int i = 0; i < n; ++i) {
            int model_id = -1;
            Tensors *output = nullptr;
            RuntimeStatus st;
            do {
                if (!ok) return;
                st = runtime_retrieve_output(&model_id, &output, 1);
            } while (st == RUNTIME_STATUS_NO_OUTPUT_AVAILABLE);

            if (st != RUNTIME_STATUS_SUCCESS) {
                ok = false;
                return;
            }

            latencies[output->id] = Ms(Clock::now() - send_times[output->id]).count();
            in_flight--;

            if (validate_first && i == 0 && !validate_output(output, batch)) {
                std::cerr << "FAIL: unexpected output shape or type" << std::endl;
                ok = false;
            }
            free_tensors(output);
        }
    });

    producer.join();
    consumer.join();
    return ok ? latencies : std::vector<double>{};
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.zip> [device] [--runs N] [--warmup N] [--batch N]"
                     " [--perf-hint latency|throughput]"
                  << std::endl;
        return 1;
    }

    const char *model_path = argv[1];
    const char *device = "CPU";
    const char *perf_hint = "latency";
    int runs = 30;
    int warmup = 5;
    int batch = 1;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc)
            runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc)
            batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--perf-hint") == 0 && i + 1 < argc)
            perf_hint = argv[++i];
        else
            device = argv[i];
    }

    std::cout << "=== OAAX YOLO Benchmark (v2) ===" << std::endl;
    std::cout << "Model     : " << model_path << std::endl;
    std::cout << "Device    : " << device << std::endl;
    std::cout << "Perf hint : " << perf_hint << std::endl;
    std::cout << "Batch     : " << batch << std::endl;
    std::cout << "Warmup    : " << warmup << " runs" << std::endl;
    std::cout << "Runs      : " << runs << std::endl << std::endl;

    // ── 1. Init ───────────────────────────────────────────────────────────────
    std::cout << "[1] Initializing runtime..." << std::endl;
    const char *init_keys[] = {"device_type", "perf_hint", "log_level"};
    const char *init_vals[] = {device, perf_hint, "2"};
    Config init_cfg = {3, init_keys, init_vals};
    CHECK(runtime_init(init_cfg) == RUNTIME_STATUS_SUCCESS, "runtime_init failed");
    std::cout << "  " << runtime_get_name() << " v" << runtime_get_version() << std::endl;

    // ── 2. Load model ─────────────────────────────────────────────────────────
    std::cout << "[2] Loading model..." << std::endl;
    auto tload = Clock::now();
    ModelConfig mc{};
    mc.file_path = model_path;
    CHECK(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS,
          std::string("runtime_load_models failed: ") + (runtime_get_error() ? runtime_get_error() : ""));
    double load_ms = Ms(Clock::now() - tload).count();
    std::cout << "  Loaded in " << load_ms << " ms" << std::endl;

    // ── 3. Warmup ─────────────────────────────────────────────────────────────
    std::cout << "[3] Warming up (" << warmup << " runs)..." << std::endl;
    {
        std::vector<Clock::time_point> ts(warmup);
        CHECK(!run_batch(warmup, ts, false, batch, 5).empty(), "warmup failed");
    }
    std::cout << "  Done" << std::endl;

    // ── 4. Benchmark ──────────────────────────────────────────────────────────
    std::cout << "[4] Benchmarking (" << runs << " runs, batch=" << batch << ", in-flight=5)..." << std::endl;
    std::vector<Clock::time_point> send_times(runs);
    auto bench_start = Clock::now();
    auto latencies = run_batch(runs, send_times, true, batch, 5);
    double bench_ms = Ms(Clock::now() - bench_start).count();
    CHECK(!latencies.empty(), "benchmark failed");

    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    double mn = *std::min_element(latencies.begin(), latencies.end());
    double mx = *std::max_element(latencies.begin(), latencies.end());
    double p95 = percentile(latencies, 95.0);
    double fps = runs * (double)batch * 1000.0 / bench_ms;

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "  Load time  : " << load_ms << " ms" << std::endl;
    std::cout << "  Avg latency: " << avg << " ms" << std::endl;
    std::cout << "  Min latency: " << mn << " ms" << std::endl;
    std::cout << "  Max latency: " << mx << " ms" << std::endl;
    std::cout << "  p95 latency: " << p95 << " ms" << std::endl;
    std::cout << "  Throughput : " << fps << " img/s" << std::endl;

    // ── 5. Cleanup ────────────────────────────────────────────────────────────
    std::cout << "\n[5] Cleaning up..." << std::endl;
    CHECK(runtime_cleanup() == RUNTIME_STATUS_SUCCESS, "runtime_cleanup failed");
    std::cout << "  Done" << std::endl;

    return 0;
}
