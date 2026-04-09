/**
 * Integration test: OAAX runtime with YOLOv8n / YOLOv11n OpenVINO IR models.
 *
 * Validates the full runtime pipeline:
 *   runtime_initialization → runtime_model_loading → send_input → receive_output → runtime_destruction
 *
 * Usage:
 *   ./yolo_test <model.xml> [device]
 *
 * device defaults to "CPU". Pass "GPU" or "NPU" to test other devices.
 */

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <string>
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

#define CHECK(expr, msg)                              \
    do {                                              \
        if (!(expr)) {                                \
            std::cerr << "FAIL: " << msg << std::endl; \
            runtime_destruction();                    \
            return 1;                                 \
        }                                             \
    } while (0)

static tensors_struct *make_yolo_input()
{
    tensors_struct *ts = allocate_tensors_struct(1);
    if (!ts) return nullptr;

    // Name: "images" (standard ultralytics ONNX export)
    ts->names[0] = strdup("images");

    // Shape: [1, 3, 640, 640]
    ts->ranks[0] = 4;
    ts->shapes[0] = (size_t *)malloc(4 * sizeof(size_t));
    ts->shapes[0][0] = YOLO_BATCH;
    ts->shapes[0][1] = YOLO_CHANNELS;
    ts->shapes[0][2] = YOLO_HEIGHT;
    ts->shapes[0][3] = YOLO_WIDTH;

    ts->data_types[0] = DATA_TYPE_FLOAT;

    size_t n_bytes = YOLO_BATCH * YOLO_CHANNELS * YOLO_HEIGHT * YOLO_WIDTH * sizeof(float);
    ts->data[0] = malloc(n_bytes);
    if (!ts->data[0]) {
        deep_free_tensors_struct(ts);
        return nullptr;
    }
    memset(ts->data[0], 0, n_bytes);

    return ts;
}

int main(int argc, char **argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.xml> [device]" << std::endl;
        return 1;
    }

    const char *model_path = argv[1];
    const char *device     = (argc >= 3) ? argv[2] : "CPU";

    std::cout << "=== OAAX YOLO Integration Test ===" << std::endl;
    std::cout << "Model : " << model_path << std::endl;
    std::cout << "Device: " << device << std::endl;
    std::cout << std::endl;

    // ── 1. Initialize runtime ─────────────────────────────────────────────────
    std::cout << "[1] Initializing runtime..." << std::endl;

    // Set device via runtime args
    char device_key[]  = "device_type";
    char log_key[]     = "log_level";
    char log_val[]     = "2";  // warn level
    char *keys[]       = {device_key, log_key};
    void *values[]     = {const_cast<char *>(device), log_val};

    int rc = runtime_initialization_with_args(2, keys, values);
    CHECK(rc == 0, "runtime_initialization_with_args failed (rc=" + std::to_string(rc) + ")");
    std::cout << "  ✓ Runtime initialized (" << runtime_name() << " v" << runtime_version() << ")" << std::endl;

    // ── 2. Load model ─────────────────────────────────────────────────────────
    std::cout << "[2] Loading model..." << std::endl;
    auto t0 = std::chrono::steady_clock::now();

    rc = runtime_model_loading(model_path);
    CHECK(rc == 0, "runtime_model_loading failed");

    auto t1 = std::chrono::steady_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  ✓ Model loaded in " << load_ms << " ms" << std::endl;

    // ── 3. Prepare & send input ───────────────────────────────────────────────
    std::cout << "[3] Sending input [1, 3, 640, 640]..." << std::endl;
    tensors_struct *input = make_yolo_input();
    CHECK(input != nullptr, "Failed to allocate input tensors");

    auto t2 = std::chrono::steady_clock::now();
    rc = send_input(input);
    // input is now owned by the runtime
    CHECK(rc == 0, "send_input failed");
    std::cout << "  ✓ Input enqueued" << std::endl;

    // ── 4. Receive output ─────────────────────────────────────────────────────
    std::cout << "[4] Waiting for output..." << std::endl;
    tensors_struct *output = nullptr;
    const int max_attempts = 100;  // 100 × 100 ms = 10 s max

    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        if (receive_output(&output) == 0) break;
        output = nullptr;
    }

    CHECK(output != nullptr, "receive_output timed out after 10 s");

    auto t3 = std::chrono::steady_clock::now();
    double infer_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::cout << "  ✓ Output received in " << infer_ms << " ms" << std::endl;

    // ── 5. Validate output ────────────────────────────────────────────────────
    std::cout << "[5] Validating output..." << std::endl;

    CHECK(output->num_tensors == 1,
          "Expected 1 output tensor, got " + std::to_string(output->num_tensors));

    CHECK(output->ranks[0] == 3,
          "Expected 3D output, got rank " + std::to_string(output->ranks[0]));

    size_t *shape = output->shapes[0];
    CHECK(shape[0] == YOLO_BATCH,
          "Expected batch=1, got " + std::to_string(shape[0]));

    CHECK(shape[1] == YOLO_OUT_CH,
          "Expected output channels=" + std::to_string(YOLO_OUT_CH) +
          ", got " + std::to_string(shape[1]));

    CHECK(shape[2] == YOLO_ANCHORS,
          "Expected anchors=" + std::to_string(YOLO_ANCHORS) +
          ", got " + std::to_string(shape[2]));

    CHECK(output->data_types[0] == DATA_TYPE_FLOAT, "Expected float32 output");

    // Sanity-check: output data should not be all zeros (inference ran)
    float *data = static_cast<float *>(output->data[0]);
    size_t n_elements = YOLO_BATCH * YOLO_OUT_CH * YOLO_ANCHORS;
    bool all_zero = true;
    for (size_t i = 0; i < n_elements; ++i) {
        if (data[i] != 0.0f) { all_zero = false; break; }
    }
    CHECK(!all_zero, "Output is all zeros — inference may not have run");

    std::cout << "  ✓ Output shape: [" << shape[0] << ", " << shape[1] << ", " << shape[2] << "]" << std::endl;
    std::cout << "  ✓ Output dtype: float32" << std::endl;

    deep_free_tensors_struct(output);

    // ── 6. Destroy runtime ────────────────────────────────────────────────────
    std::cout << "[6] Destroying runtime..." << std::endl;
    rc = runtime_destruction();
    CHECK(rc == 0, "runtime_destruction failed");
    std::cout << "  ✓ Runtime destroyed" << std::endl;

    std::cout << std::endl;
    std::cout << "=== All tests passed! ===" << std::endl;
    std::cout << "  Load time  : " << load_ms << " ms" << std::endl;
    std::cout << "  Infer time : " << infer_ms << " ms" << std::endl;
    return 0;
}
