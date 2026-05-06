/**
 * Integration benchmark: OAAX v2 runtime with YOLOv8n / YOLOv11n models.
 *
 * Uses dedicated producer/consumer threads to exercise the async queue.
 *
 * Usage:
 *   ./yolo_test <model.zip> [device] [--runs N] [--warmup N] [--batch N]
 *               [--perf-hint latency|throughput] [--input-dtype f32|u8|f16]
 *               [--in-flight N] [--imgsz N] [--queue-limit-test]
 * Defaults: device=CPU, runs=30, warmup=5, batch=1, input-dtype=f32, in-flight=5, imgsz=640
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
static const int YOLO_OUT_CH = 84;  // 4 bbox + 80 COCO classes, resolution-independent

static size_t dtype_byte_size(TensorElementType dtype) {
    switch (dtype) {
        case DATA_TYPE_UINT8:
            return 1;
        case DATA_TYPE_FLOAT16:
            return 2;
        default:
            return 4;  // f32
    }
}

static TensorElementType parse_input_dtype(const char *s) {
    if (strcmp(s, "u8") == 0 || strcmp(s, "uint8") == 0) return DATA_TYPE_UINT8;
    if (strcmp(s, "f16") == 0 || strcmp(s, "float16") == 0) return DATA_TYPE_FLOAT16;
    return DATA_TYPE_FLOAT;
}

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

static Tensors *make_yolo_input(int batch, int request_id, TensorElementType dtype, int imgsz,
                                const char *input_name = "images") {
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
    td.name = strdup(input_name);
    td.data_type = dtype;
    td.rank = 4;
    td.shape = (int *)malloc(4 * sizeof(int));
    td.shape[0] = batch;
    td.shape[1] = YOLO_CHANNELS;
    td.shape[2] = imgsz;
    td.shape[3] = imgsz;
    size_t n_bytes = (size_t)batch * YOLO_CHANNELS * imgsz * imgsz * dtype_byte_size(dtype);
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
    if (td.shape[2] <= 0) return false;  // anchors vary by input resolution
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
                                     TensorElementType dtype, int imgsz, int max_in_flight = 1,
                                     const char *input_name = "images") {
    std::vector<double> latencies(n);
    std::atomic<bool> ok{true};
    std::atomic<int> in_flight{0};

    std::thread producer([&]() {
        for (int i = 0; i < n; ++i) {
            while (in_flight.load() >= max_in_flight) {
                if (!ok) return;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            Tensors *input = make_yolo_input(batch, i, dtype, imgsz, input_name);
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
                st = runtime_retrieve_output(&model_id, &output, 1000);
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
                     " [--perf-hint latency|throughput] [--nireq N]"
                  << std::endl;
        return 1;
    }

    const char *model_path = argv[1];
    const char *device = "CPU";
    const char *perf_hint = "latency";
    const char *input_dtype_str = "f32";
    const char *input_name = "images";
    int runs = 30;
    int warmup = 5;
    int batch = 1;
    int nireq = 0;  // 0 = use runtime default (optimal)
    int in_flight = 5;
    int imgsz = 640;
    int log_level = 2;  // spdlog level: 0=trace 1=debug 2=info 3=warn 4=err
    bool test_queue_limit = false;
    bool validate_output_shape = true;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc)
            runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc)
            warmup = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc)
            batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--perf-hint") == 0 && i + 1 < argc)
            perf_hint = argv[++i];
        else if (strcmp(argv[i], "--nireq") == 0 && i + 1 < argc)
            nireq = atoi(argv[++i]);
        else if (strcmp(argv[i], "--input-dtype") == 0 && i + 1 < argc)
            input_dtype_str = argv[++i];
        else if (strcmp(argv[i], "--in-flight") == 0 && i + 1 < argc)
            in_flight = atoi(argv[++i]);
        else if (strcmp(argv[i], "--imgsz") == 0 && i + 1 < argc)
            imgsz = atoi(argv[++i]);
        else if (strcmp(argv[i], "--input-name") == 0 && i + 1 < argc)
            input_name = argv[++i];
        else if (strcmp(argv[i], "--log-level") == 0 && i + 1 < argc)
            log_level = atoi(argv[++i]);
        else if (strcmp(argv[i], "--no-validate") == 0)
            validate_output_shape = false;
        else if (strcmp(argv[i], "--queue-limit-test") == 0)
            test_queue_limit = true;
        else
            device = argv[i];
    }
    TensorElementType input_dtype = parse_input_dtype(input_dtype_str);

    std::cout << "=== OAAX YOLO Benchmark (v2) ===" << std::endl;
    std::cout << "Model     : " << model_path << std::endl;
    std::cout << "Device    : " << device << std::endl;
    std::cout << "Perf hint : " << perf_hint << std::endl;
    std::cout << "Batch     : " << batch << std::endl;
    std::cout << "nireq     : " << (nireq > 0 ? std::to_string(nireq) : "auto") << std::endl;
    std::cout << "Input dtype: " << input_dtype_str << std::endl;
    std::cout << "Input name : " << input_name << std::endl;
    std::cout << "In-flight  : " << in_flight << std::endl;
    std::cout << "Image size : " << imgsz << "x" << imgsz << std::endl;
    std::cout << "Warmup    : " << warmup << " runs" << std::endl;
    std::cout << "Runs      : " << runs << std::endl << std::endl;

    // ── 1. Init ───────────────────────────────────────────────────────────────
    std::cout << "[1] Initializing runtime..." << std::endl;
    std::string nireq_str = std::to_string(nireq);
    std::string log_level_str = std::to_string(log_level);
    const char *init_keys[] = {"device_type", "perf_hint", "log_level", "num_requests"};
    const char *init_vals[] = {device, perf_hint, log_level_str.c_str(), nireq_str.c_str()};
    Config init_cfg = {4, init_keys, init_vals};
    CHECK(runtime_init(init_cfg) == RUNTIME_STATUS_SUCCESS, "runtime_init failed");
    std::cout << "  " << runtime_get_name() << " v" << runtime_get_version() << std::endl;

    // ── 2. Load model ─────────────────────────────────────────────────────────
    std::cout << "[2] Loading model..." << std::endl;
    auto tload = Clock::now();
    ModelConfig mc{};
    mc.file_path = model_path;
    mc.config = {0, nullptr, nullptr};
    CHECK(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS,
          std::string("runtime_load_models failed: ") + (runtime_get_error() ? runtime_get_error() : ""));
    double load_ms = Ms(Clock::now() - tload).count();
    std::cout << "  Loaded in " << load_ms << " ms" << std::endl;

    // ── 3. Warmup ─────────────────────────────────────────────────────────────
    std::cout << "[3] Warming up (" << warmup << " runs)..." << std::endl;
    {
        std::vector<Clock::time_point> ts(warmup);
        CHECK(!run_batch(warmup, ts, false, batch, input_dtype, imgsz, in_flight, input_name).empty(), "warmup failed");
    }
    std::cout << "  Done" << std::endl;

    // ── 4. Benchmark ──────────────────────────────────────────────────────────
    std::cout << "[4] Benchmarking (" << runs << " runs, batch=" << batch << ", in-flight=" << in_flight << ")..."
              << std::endl;
    std::vector<Clock::time_point> send_times(runs);
    auto bench_start = Clock::now();
    auto latencies =
        run_batch(runs, send_times, validate_output_shape, batch, input_dtype, imgsz, in_flight, input_name);
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

    // ── 6. Queue limit test (optional) ────────────────────────────────────────
    // Verifies that max_queue_size rejects inputs when queues are at capacity.
    // Uses 1 infer slot and a limit of 3: after 4 accepted enqueues (1 in-slot +
    // 3 in input_queue) every further enqueue must be rejected.
    if (test_queue_limit) {
        std::cout << "\n[6] Queue limit test (max_queue_size=3, num_requests=1)..." << std::endl;

        const int QUEUE_LIMIT = 3;
        const int FLOOD = 20;
        std::string ql_str = std::to_string(QUEUE_LIMIT);
        const char *ql_keys[] = {"device_type", "log_level", "num_requests", "max_queue_size", "input_dtype"};
        const char *ql_vals[] = {device, "2", "1", ql_str.c_str(), input_dtype_str};
        Config ql_cfg = {5, ql_keys, ql_vals};
        CHECK(runtime_init(ql_cfg) == RUNTIME_STATUS_SUCCESS, "queue-limit re-init failed");

        ModelConfig ql_mc{};
        ql_mc.file_path = model_path;
        ql_mc.config = {0, nullptr, nullptr};
        CHECK(runtime_load_models(1, &ql_mc) == RUNTIME_STATUS_SUCCESS, "queue-limit model load failed");

        int rejected = 0;
        for (int i = 0; i < FLOOD; ++i) {
            Tensors *inp = make_yolo_input(1, i, input_dtype, imgsz);
            CHECK(inp != nullptr, "make_yolo_input returned null");
            RuntimeStatus st = runtime_enqueue_input(0, inp);
            if (st != RUNTIME_STATUS_SUCCESS) {
                rejected++;
                free_tensors(inp);  // runtime didn't take ownership on failure
            }
        }

        CHECK(rejected > 0, "expected at least one rejection with max_queue_size=3, got none");
        std::cout << "  Rejected " << rejected << "/" << FLOOD << " inputs as expected" << std::endl;

        // Drain remaining in-flight outputs
        int mid = -1;
        Tensors *out = nullptr;
        while (runtime_retrieve_output(&mid, &out, 100) == RUNTIME_STATUS_SUCCESS) free_tensors(out);

        CHECK(runtime_cleanup() == RUNTIME_STATUS_SUCCESS, "queue-limit cleanup failed");
        std::cout << "  PASS: max_queue_size enforced correctly" << std::endl;
    }

    return 0;
}
