/**
 * Multi-model async inference test.
 *
 * Loads two models simultaneously (same or different zips), then uses
 * concurrent producer threads to enqueue inputs to both models and a single
 * consumer thread to drain the global output queue. Verifies:
 *  - All outputs are received (no dropped results)
 *  - model_id on each output matches the model that produced it
 *  - Tensors.id echoed on output matches the id set on input (request correlation)
 *  - Output shapes are valid YOLO output (1 x 84 x 8400)
 *
 * Usage:
 *   ./multi_model_test <model.zip> [model2.zip]
 *
 * If only one path is given the same zip is loaded as both model 0 and model 1.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "oaax_runtime.h"

static const int YOLO_CHANNELS = 3;
static const int YOLO_HEIGHT = 640;
static const int YOLO_WIDTH = 640;
static const int YOLO_OUT_CH = 84;
static const int YOLO_ANCHORS = 8400;

static const int N_REQUESTS = 20;  // requests per model

using Clock = std::chrono::steady_clock;
using Ms = std::chrono::duration<double, std::milli>;

#define PASS(msg) std::cout << "  PASS: " << msg << std::endl
#define FAIL(msg)                                    \
    do {                                             \
        std::cerr << "  FAIL: " << msg << std::endl; \
        runtime_cleanup();                           \
        return 1;                                    \
    } while (0)
#define ASSERT(cond, msg)       \
    do {                        \
        if (!(cond)) FAIL(msg); \
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

static Tensors *make_yolo_input(int request_id) {
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
    td.shape[0] = 1;
    td.shape[1] = YOLO_CHANNELS;
    td.shape[2] = YOLO_HEIGHT;
    td.shape[3] = YOLO_WIDTH;
    td.data_size = YOLO_CHANNELS * YOLO_HEIGHT * YOLO_WIDTH * sizeof(float);
    td.data = calloc(1, td.data_size);
    if (!td.data) {
        free(td.shape);
        free(td.name);
        free(ts->tensors);
        free(ts);
        return nullptr;
    }
    return ts;
}

static bool valid_yolo_output(const Tensors *out) {
    if (!out || out->num_tensors != 1) return false;
    const TensorDescriptor &td = out->tensors[0];
    if (td.rank != 3) return false;
    if (td.shape[0] != 1) return false;
    if (td.shape[1] != YOLO_OUT_CH) return false;
    if (td.shape[2] != YOLO_ANCHORS) return false;
    if (td.data_type != DATA_TYPE_FLOAT) return false;
    return true;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.zip> [model2.zip]" << std::endl;
        return 1;
    }

    const char *path0 = argv[1];
    const char *path1 = argc > 2 ? argv[2] : argv[1];

    std::cout << "=== Multi-model async inference test ===" << std::endl;
    std::cout << "  Model 0: " << path0 << std::endl;
    std::cout << "  Model 1: " << path1 << std::endl;
    std::cout << "  Requests per model: " << N_REQUESTS << std::endl << std::endl;

    // ── 1. Init ───────────────────────────────────────────────────────────────
    std::cout << "[1] runtime_init()" << std::endl;
    const char *keys[] = {"device_type", "log_level"};
    const char *values[] = {"CPU", "2"};
    Config cfg = {2, keys, values};
    ASSERT(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS, "runtime_init failed");
    PASS("runtime initialized");

    // ── 2. Load two models ────────────────────────────────────────────────────
    std::cout << "\n[2] runtime_load_models(2, ...)" << std::endl;
    ModelConfig models[2] = {};
    models[0].file_path = path0;
    models[1].file_path = path1;
    ASSERT(runtime_load_models(2, models) == RUNTIME_STATUS_SUCCESS,
           std::string("load failed: ") + (runtime_get_error() ? runtime_get_error() : ""));
    PASS("both models loaded");

    // ── 3. Concurrent producers + single consumer ─────────────────────────────
    std::cout << "\n[3] Async inference: 2 producers + 1 consumer" << std::endl;

    // Model 0 uses ids [0 .. N-1], model 1 uses ids [1000 .. 1000+N-1].
    // This lets us verify model_id <-> id correlation on the consumer side.
    const int ID_OFFSET_M1 = 1000;
    const int total = N_REQUESTS * 2;

    std::atomic<bool> ok{true};
    std::atomic<int> in_flight{0};

    // Per-output tracking: id -> (model_id, receive_time)
    std::mutex result_mu;
    std::vector<int> received_ids;
    std::vector<int> received_model_ids;
    std::atomic<int> shape_errors{0};
    std::atomic<int> id_errors{0};

    std::vector<Clock::time_point> send_times(ID_OFFSET_M1 + N_REQUESTS);

    // Producer for model 0
    std::thread prod0([&]() {
        for (int i = 0; i < N_REQUESTS; ++i) {
            Tensors *input = make_yolo_input(i);
            if (!input) {
                ok = false;
                return;
            }
            send_times[i] = Clock::now();
            in_flight++;
            if (runtime_enqueue_input(0, input) != RUNTIME_STATUS_SUCCESS) {
                free_tensors(input);
                ok = false;
                return;
            }
        }
    });

    // Producer for model 1
    std::thread prod1([&]() {
        for (int i = 0; i < N_REQUESTS; ++i) {
            int id = ID_OFFSET_M1 + i;
            Tensors *input = make_yolo_input(id);
            if (!input) {
                ok = false;
                return;
            }
            send_times[id] = Clock::now();
            in_flight++;
            if (runtime_enqueue_input(1, input) != RUNTIME_STATUS_SUCCESS) {
                free_tensors(input);
                ok = false;
                return;
            }
        }
    });

    // Single consumer drains the global output queue
    std::thread consumer([&]() {
        for (int i = 0; i < total; ++i) {
            int model_id = -1;
            Tensors *output = nullptr;
            RuntimeStatus st;
            do {
                if (!ok) return;
                st = runtime_retrieve_output(&model_id, &output, 10);
            } while (st == RUNTIME_STATUS_NO_OUTPUT_AVAILABLE);

            if (st != RUNTIME_STATUS_SUCCESS) {
                ok = false;
                return;
            }
            in_flight--;

            // Verify output shape
            if (!valid_yolo_output(output)) shape_errors++;

            // Verify model_id <-> id correlation
            int id = output->id;
            bool id_belongs_to_model = (model_id == 0 && id >= 0 && id < N_REQUESTS) ||
                                       (model_id == 1 && id >= ID_OFFSET_M1 && id < ID_OFFSET_M1 + N_REQUESTS);
            if (!id_belongs_to_model) id_errors++;

            {
                std::lock_guard<std::mutex> lk(result_mu);
                received_ids.push_back(id);
                received_model_ids.push_back(model_id);
            }
            free_tensors(output);
        }
    });

    prod0.join();
    prod1.join();
    consumer.join();

    ASSERT(ok, "inference loop reported failure");
    ASSERT((int)received_ids.size() == total,
           "expected " + std::to_string(total) + " outputs, got " + std::to_string(received_ids.size()));
    ASSERT(shape_errors == 0, std::to_string(shape_errors.load()) + " output(s) had wrong shape");
    ASSERT(id_errors == 0, std::to_string(id_errors.load()) + " output(s) had mismatched model_id/id");

    // Check all expected ids were received
    std::sort(received_ids.begin(), received_ids.end());
    for (int i = 0; i < N_REQUESTS; ++i)
        ASSERT(std::find(received_ids.begin(), received_ids.end(), i) != received_ids.end(),
               "missing output id " + std::to_string(i) + " (model 0)");
    for (int i = 0; i < N_REQUESTS; ++i)
        ASSERT(std::find(received_ids.begin(), received_ids.end(), ID_OFFSET_M1 + i) != received_ids.end(),
               "missing output id " + std::to_string(ID_OFFSET_M1 + i) + " (model 1)");

    // Count outputs per model
    int from_m0 = (int)std::count(received_model_ids.begin(), received_model_ids.end(), 0);
    int from_m1 = (int)std::count(received_model_ids.begin(), received_model_ids.end(), 1);
    std::cout << "  Outputs from model 0: " << from_m0 << " / " << N_REQUESTS << std::endl;
    std::cout << "  Outputs from model 1: " << from_m1 << " / " << N_REQUESTS << std::endl;
    ASSERT(from_m0 == N_REQUESTS, "model 0 output count mismatch");
    ASSERT(from_m1 == N_REQUESTS, "model 1 output count mismatch");
    PASS("all " + std::to_string(total) + " outputs received with correct model_id and request id");

    // ── 4. Cleanup ────────────────────────────────────────────────────────────
    std::cout << "\n[4] runtime_cleanup()" << std::endl;
    ASSERT(runtime_cleanup() == RUNTIME_STATUS_SUCCESS, "cleanup failed");
    PASS("cleanup OK");

    std::cout << "\n=== All tests passed ===" << std::endl;
    return 0;
}
