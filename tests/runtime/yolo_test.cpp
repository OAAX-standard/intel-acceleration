/**
 * Integration test + benchmark: OAAX runtime with YOLOv8n / YOLOv11n OpenVINO IR models.
 *
 * Validates the full runtime pipeline and measures inference latency over multiple runs.
 *
 * Usage:
 *   ./yolo_test <model.xml> [device] [--runs N] [--warmup N]
 *
 * Defaults: device=CPU, runs=30, warmup=5
 */

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <string>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cassert>

#include "../include/runtime_core.hpp"
#include "../include/tensors_struct.h"

// YOLO model constants (YOLOv8n / YOLOv11n with COCO 80 classes)
static const size_t YOLO_BATCH    = 1;
static const size_t YOLO_CHANNELS = 3;
static const size_t YOLO_HEIGHT   = 640;
static const size_t YOLO_WIDTH    = 640;
static const size_t YOLO_OUT_CH   = 84;    // 4 bbox + 80 classes
static const size_t YOLO_ANCHORS  = 8400;  // 20x20 + 40x40 + 80x80

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

    ts->names[0] = strdup("images");
    ts->ranks[0] = 4;
    ts->shapes[0] = (size_t *)malloc(4 * sizeof(size_t));
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

// Send one input, wait for output, return inference latency in ms.
// Returns -1.0 on failure.
static double run_one(bool validate)
{
    tensors_struct *input = make_yolo_input();
    if (!input) return -1.0;

    auto t0 = std::chrono::steady_clock::now();

    if (send_input(input) != 0) return -1.0;  // runtime owns input now

    tensors_struct *output = nullptr;
    for (int i = 0; i < 100; ++i) {
        if (receive_output(&output) == 0) break;
        output = nullptr;
    }
    if (!output) return -1.0;

    auto t1   = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (validate) {
        bool ok = output->num_tensors == 1
               && output->ranks[0]    == 3
               && output->shapes[0][0] == YOLO_BATCH
               && output->shapes[0][1] == YOLO_OUT_CH
               && output->shapes[0][2] == YOLO_ANCHORS
               && output->data_types[0] == DATA_TYPE_FLOAT;

        if (!ok) {
            std::cerr << "FAIL: unexpected output shape or type" << std::endl;
            deep_free_tensors_struct(output);
            return -1.0;
        }

        // Sanity-check: not all zeros
        float *data = static_cast<float *>(output->data[0]);
        size_t n = YOLO_BATCH * YOLO_OUT_CH * YOLO_ANCHORS;
        bool all_zero = true;
        for (size_t i = 0; i < n; ++i) {
            if (data[i] != 0.0f) { all_zero = false; break; }
        }
        if (all_zero) {
            std::cerr << "FAIL: output is all zeros" << std::endl;
            deep_free_tensors_struct(output);
            return -1.0;
        }
    }

    deep_free_tensors_struct(output);
    return ms;
}

static double percentile(std::vector<double> v, double p)
{
    std::sort(v.begin(), v.end());
    size_t idx = (size_t)(p / 100.0 * (v.size() - 1) + 0.5);
    return v[std::min(idx, v.size() - 1)];
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.xml> [device] [--runs N] [--warmup N]"
                  << std::endl;
        return 1;
    }

    const char *model_path = argv[1];
    const char *device     = "CPU";
    int runs   = 30;
    int warmup = 5;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--runs")   == 0 && i + 1 < argc) { runs   = atoi(argv[++i]); }
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) { warmup = atoi(argv[++i]); }
        else { device = argv[i]; }
    }

    std::cout << "=== OAAX YOLO Benchmark ===" << std::endl;
    std::cout << "Model  : " << model_path << std::endl;
    std::cout << "Device : " << device     << std::endl;
    std::cout << "Warmup : " << warmup     << " runs" << std::endl;
    std::cout << "Runs   : " << runs       << std::endl;
    std::cout << std::endl;

    // ── 1. Initialize runtime ─────────────────────────────────────────────────
    std::cout << "[1] Initializing runtime..." << std::endl;

    char device_key[] = "device_type";
    char log_key[]    = "log_level";
    char log_val[]    = "2";
    char *keys[]      = {device_key, log_key};
    void *values[]    = {const_cast<char *>(device), log_val};

    int rc = runtime_initialization_with_args(2, keys, values);
    CHECK(rc == 0, "runtime_initialization_with_args failed (rc=" + std::to_string(rc) + ")");
    std::cout << "  ✓ " << runtime_name() << " v" << runtime_version() << std::endl;

    // ── 2. Load model ─────────────────────────────────────────────────────────
    std::cout << "[2] Loading model..." << std::endl;
    auto tload0 = std::chrono::steady_clock::now();

    rc = runtime_model_loading(model_path);
    CHECK(rc == 0, "runtime_model_loading failed");

    double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - tload0).count();
    std::cout << "  ✓ Loaded in " << load_ms << " ms" << std::endl;

    // ── 3. Warmup ─────────────────────────────────────────────────────────────
    std::cout << "[3] Warming up (" << warmup << " runs)..." << std::endl;
    for (int i = 0; i < warmup; ++i) {
        double ms = run_one(false);
        CHECK(ms >= 0, "warmup run " + std::to_string(i) + " failed");
    }
    std::cout << "  ✓ Done" << std::endl;

    // ── 4. Benchmark ──────────────────────────────────────────────────────────
    std::cout << "[4] Benchmarking (" << runs << " runs)..." << std::endl;
    std::vector<double> times;
    times.reserve(runs);

    for (int i = 0; i < runs; ++i) {
        double ms = run_one(i == 0);  // validate output on first run only
        CHECK(ms >= 0, "benchmark run " + std::to_string(i) + " failed");
        times.push_back(ms);
    }

    double avg = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    double mn  = *std::min_element(times.begin(), times.end());
    double mx  = *std::max_element(times.begin(), times.end());
    double p95 = percentile(times, 95.0);

    std::cout << std::endl;
    std::cout << "=== Results ===" << std::endl;
    std::cout << "  Load time : " << load_ms << " ms"  << std::endl;
    std::cout << "  Avg       : " << avg     << " ms"  << std::endl;
    std::cout << "  Min       : " << mn      << " ms"  << std::endl;
    std::cout << "  Max       : " << mx      << " ms"  << std::endl;
    std::cout << "  p95       : " << p95     << " ms"  << std::endl;

    // ── 5. Destroy runtime ────────────────────────────────────────────────────
    std::cout << std::endl << "[5] Destroying runtime..." << std::endl;
    rc = runtime_destruction();
    CHECK(rc == 0, "runtime_destruction failed");
    std::cout << "  ✓ Done" << std::endl;

    return 0;
}
