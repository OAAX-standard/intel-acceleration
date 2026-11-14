/**
 * Robustness tests for the OAAX v2 interface: what the runtime does when
 * things go wrong.
 *
 * Regression guard: an inference that fails must not break the runtime.
 * Before the fix, runtime_cleanup() rethrew the failed request's exception
 * (aborting the host), and requests queued behind a failed one were never run.
 *
 * Uses the model built by make_robustness_model.py: Reshape(x, shape) with a
 * dynamic output whose name is far longer than device drivers accept. A
 * `shape` that does not hold 4 elements makes the inference fail.
 *
 * Usage:
 *   ./robustness_test <robustness.zip> [device]
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "oaax_runtime.h"

#define PASS(msg) std::cout << "  PASS: " << msg << std::endl
#define FAIL(msg)                                      \
    do {                                               \
        std::cerr << "  FAIL: " << (msg) << std::endl; \
        runtime_cleanup();                             \
        return 1;                                      \
    } while (0)
#define ASSERT(cond, msg)       \
    do {                        \
        if (!(cond)) FAIL(msg); \
    } while (0)

static const char* g_device = "CPU";

static Config make_cfg() {
    static const char* keys[] = {"log_level", "device_type", "max_queue_size"};
    static const char* vals[] = {"2", nullptr, "1000"};
    vals[1] = g_device;
    return Config{3, keys, vals};
}

static std::string last_error() { return runtime_get_error() ? runtime_get_error() : "(no error text)"; }

static void free_tensors(Tensors* t) {
    if (!t) return;
    for (int i = 0; i < t->num_tensors; ++i) {
        free(t->tensors[i].name);
        free(t->tensors[i].shape);
        free(t->tensors[i].data);
    }
    free(t->tensors);
    free(t);
}

static void fill(TensorDescriptor& td, const char* name, TensorElementType type, std::vector<int> shape,
                 const void* data, size_t bytes) {
    td.name = strdup(name);
    td.data_type = type;
    td.rank = (int)shape.size();
    td.shape = (int*)malloc((shape.empty() ? 1 : shape.size()) * sizeof(int));
    for (size_t i = 0; i < shape.size(); ++i) td.shape[i] = shape[i];
    td.data_size = bytes;
    td.data = malloc(bytes ? bytes : 1);
    if (bytes) memcpy(td.data, data, bytes);
}

// A request for the Reshape model: x = {1,2,3,4}, reshaped to rows x cols.
// 2x2 is valid; anything that does not hold 4 elements fails in the device.
// Failing requests must each ask for a different shape: the CPU plugin only
// re-checks the shape when it changed since the slot's previous request.
static Tensors* make_request(int id, int64_t rows, int64_t cols) {
    Tensors* t = (Tensors*)calloc(1, sizeof(Tensors));
    t->id = id;
    t->num_tensors = 2;
    t->tensors = (TensorDescriptor*)calloc(2, sizeof(TensorDescriptor));
    const float x[4] = {1, 2, 3, 4};
    const int64_t shape[2] = {rows, cols};
    fill(t->tensors[0], "x", DATA_TYPE_FLOAT, {1, 4}, x, sizeof(x));
    fill(t->tensors[1], "shape", DATA_TYPE_INT64, {2}, shape, sizeof(shape));
    return t;
}

// Enqueue, retrying while the queue is full.
static RuntimeStatus enqueue(Tensors* t) {
    RuntimeStatus st = RUNTIME_STATUS_ERROR;
    for (int tries = 0; tries < 200; ++tries) {
        st = runtime_enqueue_input(0, t);
        if (st != RUNTIME_STATUS_ERROR) break;
    }
    return st;
}

static Tensors* retrieve(int timeout_ms = 10000) {
    int mid = -1;
    Tensors* out = nullptr;
    return runtime_retrieve_output(&mid, &out, timeout_ms) == RUNTIME_STATUS_SUCCESS ? out : nullptr;
}

static bool is_2x2_of_1234(const Tensors* out) {
    if (!out || out->num_tensors != 1) return false;
    const TensorDescriptor& td = out->tensors[0];
    if (td.rank != 2 || td.shape[0] != 2 || td.shape[1] != 2 || td.data_size != 4 * sizeof(float)) return false;
    const float* v = (const float*)td.data;
    return v[0] == 1 && v[1] == 2 && v[2] == 3 && v[3] == 4;
}

// A request the runtime must refuse: checks the status and that the caller still owns the tensors.
#define EXPECT_REJECTED(tensors, what)                                                                          \
    do {                                                                                                        \
        Tensors* rejected = (tensors);                                                                          \
        RuntimeStatus rst = runtime_enqueue_input(0, rejected);                                                 \
        free_tensors(rejected);                                                                                 \
        ASSERT(rst == RUNTIME_STATUS_INVALID_ARGUMENT,                                                          \
               std::string(what) + ": expected INVALID_ARGUMENT, " + "got status " + std::to_string((int)rst)); \
        ASSERT(runtime_get_error() != nullptr, std::string(what) + ": no error text");                          \
    } while (0)

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <robustness.zip> [device]" << std::endl;
        return 1;
    }
    const char* model_path = argv[1];
    if (argc > 2) g_device = argv[2];
    Config cfg = make_cfg();
    ModelConfig mc{};
    mc.file_path = model_path;

    std::cout << "=== OAAX Intel Runtime Robustness Tests (" << g_device << ") ===" << std::endl;

    // ── 1. Long output name ───────────────────────────────────────────────────
    std::cout << "\n[1] Output with a name longer than device drivers accept" << std::endl;
    ASSERT(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS, "init failed");
    ASSERT(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS, "load failed: " + last_error());
    {
        ASSERT(enqueue(make_request(1, 2, 2)) == RUNTIME_STATUS_SUCCESS, "enqueue failed: " + last_error());
        Tensors* out = retrieve();
        ASSERT(out != nullptr, "no output: " + last_error());
        ASSERT(out->id == 1, "output has the wrong request id");
        ASSERT(is_2x2_of_1234(out), "output is not the input reshaped to 2x2");
        std::string name = out->tensors[0].name;
        free_tensors(out);
        ASSERT(name.size() > 256 && name.rfind("bboxes-format:xyxysc;0:", 0) == 0,
               "the output does not carry the model's own (long) name");
        const char* info = runtime_get_info();
        ASSERT(info && strstr(info, name.c_str()), "runtime_get_info does not list the model's own output name");
    }
    PASS("the caller gets the model's own long output name and the right data");

    // ── 2. Rejected inputs ────────────────────────────────────────────────────
    std::cout << "\n[2] Invalid inputs are refused when enqueued" << std::endl;
    {
        Tensors* t = make_request(2, 2, 2);
        free(t->tensors[0].name);
        t->tensors[0].name = strdup("not-an-input");
        EXPECT_REJECTED(t, "unknown input name");

        t = make_request(2, 2, 2);
        free(t->tensors[1].name);
        t->tensors[1].name = strdup("x");
        EXPECT_REJECTED(t, "input given twice");

        t = make_request(2, 2, 2);
        t->tensors[0].data_type = DATA_TYPE_UINT8;
        EXPECT_REJECTED(t, "wrong dtype");

        t = make_request(2, 2, 2);
        t->tensors[0].data_size = 2 * sizeof(float);  // buffer smaller than the shape says
        EXPECT_REJECTED(t, "buffer smaller than the shape");

        t = make_request(2, 2, 2);
        t->tensors[0].shape[1] = 8;  // model wants [1,4]
        t->tensors[0].data_size = 8 * sizeof(float);
        t->tensors[0].data = realloc(t->tensors[0].data, 8 * sizeof(float));
        EXPECT_REJECTED(t, "wrong shape");

        t = make_request(2, 2, 2);
        t->tensors[0].shape[0] = -1;
        EXPECT_REJECTED(t, "negative dimension");

        t = make_request(2, 2, 2);
        free(t->tensors[0].data);
        t->tensors[0].data = nullptr;
        EXPECT_REJECTED(t, "no data");

        ASSERT(runtime_enqueue_input(0, nullptr) == RUNTIME_STATUS_INVALID_ARGUMENT, "null tensors accepted");
        ASSERT(runtime_enqueue_input(7, nullptr) == RUNTIME_STATUS_INVALID_MODEL_ID, "invalid model id accepted");

        // An input that is left out is not an error: it is fed zeros, and never
        // the buffer of an earlier request. Sent twice so every slot is reused.
        for (int round = 0; round < 16; ++round) {
            Tensors* only_shape = (Tensors*)calloc(1, sizeof(Tensors));
            only_shape->id = 50 + round;
            only_shape->num_tensors = 1;
            only_shape->tensors = (TensorDescriptor*)calloc(1, sizeof(TensorDescriptor));
            const int64_t shape[2] = {2, 2};
            fill(only_shape->tensors[0], "shape", DATA_TYPE_INT64, {2}, shape, sizeof(shape));
            ASSERT(enqueue(only_shape) == RUNTIME_STATUS_SUCCESS, "request without 'x' refused: " + last_error());
            Tensors* zeros = retrieve();
            ASSERT(zeros && zeros->tensors[0].data_size == 4 * sizeof(float), "no output for a request without 'x'");
            const float* v = (const float*)zeros->tensors[0].data;
            ASSERT(v[0] == 0 && v[1] == 0 && v[2] == 0 && v[3] == 0, "a left-out input was not fed zeros");
            free_tensors(zeros);
            // a full request in between, so the slot has a caller buffer to (wrongly) keep
            ASSERT(enqueue(make_request(90, 2, 2)) == RUNTIME_STATUS_SUCCESS, "valid request refused");
            Tensors* full = retrieve();
            ASSERT(is_2x2_of_1234(full), "wrong output between requests without 'x'");
            free_tensors(full);
        }

        // The runtime still works after all of that.
        ASSERT(enqueue(make_request(3, 2, 2)) == RUNTIME_STATUS_SUCCESS, "valid request refused: " + last_error());
        Tensors* out = retrieve();
        ASSERT(is_2x2_of_1234(out), "no valid output after the rejected inputs");
        free_tensors(out);
    }
    PASS("7 kinds of invalid input refused with INVALID_ARGUMENT; a left-out input gets zeros");

    // ── 3. Failed inferences do not stall or poison the runtime ──────────────
    std::cout << "\n[3] Inferences that fail in the device" << std::endl;
    {
        // One failing request: no output comes back for it.
        ASSERT(enqueue(make_request(100, 3, 3)) == RUNTIME_STATUS_SUCCESS, "enqueue failed: " + last_error());
        Tensors* none = retrieve(1500);
        ASSERT(none == nullptr, "a failed inference produced an output");
        ASSERT(runtime_get_error() != nullptr, "a failed inference left no error text");
        const char* info = runtime_get_info();
        ASSERT(info && strstr(info, "\"failed_requests\":1"), "runtime_get_info does not count the failed request");

        // A burst that mixes failing and valid requests, more than there are
        // infer slots, so most of them wait in the queue behind failing ones.
        const int kTotal = 200;
        int valid = 0;
        for (int i = 0; i < kTotal; ++i) {
            bool ok = (i % 4 == 3);
            valid += ok;
            ASSERT(enqueue(make_request(1000 + i, ok ? 2 : 5 + i, ok ? 2 : 7)) == RUNTIME_STATUS_SUCCESS,
                   "enqueue failed in the burst: " + last_error());
        }
        std::vector<bool> got(kTotal, false);
        for (int n = 0; n < valid; ++n) {
            Tensors* out = retrieve();
            ASSERT(out != nullptr, "valid request " + std::to_string(n + 1) + " of " + std::to_string(valid) +
                                       " never came back (queue stalled behind failed requests)");
            int idx = out->id - 1000;
            ASSERT(idx >= 0 && idx < kTotal && idx % 4 == 3 && !got[idx],
                   "unexpected or duplicate output id " + std::to_string(out->id));
            ASSERT(is_2x2_of_1234(out), "wrong output data in the burst");
            got[idx] = true;
            free_tensors(out);
        }
        ASSERT(retrieve(500) == nullptr, "more outputs than valid requests");
        info = runtime_get_info();
        std::string want = "\"failed_requests\":" + std::to_string(1 + kTotal - valid);
        ASSERT(info && strstr(info, want.c_str()), "failed request count is wrong, expected " + want);
    }
    PASS("50 valid requests answered among 151 failing ones; failures are counted");

    // ── 4. Cleanup after failed inferences, then a fresh session ─────────────
    std::cout << "\n[4] Cleanup after failed inferences" << std::endl;
    {
        // Leave failed requests as the last thing each slot did.
        for (int i = 0; i < 32; ++i)
            ASSERT(enqueue(make_request(5000 + i, 5 + i, 9)) == RUNTIME_STATUS_SUCCESS,
                   "enqueue failed: " + last_error());
        ASSERT(retrieve(1500) == nullptr, "a failed inference produced an output");
        ASSERT(runtime_cleanup() == RUNTIME_STATUS_SUCCESS, "cleanup failed after failed inferences");

        ASSERT(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS, "re-init failed: " + last_error());
        ASSERT(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS, "re-load failed: " + last_error());
        ASSERT(enqueue(make_request(1, 2, 2)) == RUNTIME_STATUS_SUCCESS, "enqueue failed after re-init");
        Tensors* out = retrieve();
        ASSERT(is_2x2_of_1234(out), "no valid output after cleanup + re-init");
        free_tensors(out);
        ASSERT(runtime_cleanup() == RUNTIME_STATUS_SUCCESS, "second cleanup failed");
    }
    PASS("cleanup completes and the next session works");

    // ── 5. A model that cannot be loaded leaves the runtime usable ───────────
    std::cout << "\n[5] Models that cannot be loaded" << std::endl;
    {
        const char* garbage = "robustness_garbage.zip";
        std::ofstream(garbage, std::ios::binary) << "this is not a zip archive";

        ASSERT(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS, "init failed");
        ModelConfig bad[2] = {};
        bad[0].file_path = model_path;  // loads
        bad[1].file_path = garbage;     // does not: the whole call must fail
        ASSERT(runtime_load_models(2, bad) == RUNTIME_STATUS_ERROR, "a garbage archive was loaded");
        ASSERT(runtime_get_error() != nullptr, "no error text for the garbage archive");
        bad[1].file_path = "does-not-exist.zip";
        ASSERT(runtime_load_models(2, bad) == RUNTIME_STATUS_ERROR, "a missing file was loaded");
        bad[1].file_path = nullptr;
        ASSERT(runtime_load_models(2, bad) == RUNTIME_STATUS_ERROR, "a null path was loaded");
        ASSERT(runtime_enqueue_input(0, nullptr) == RUNTIME_STATUS_MODEL_NOT_LOADED,
               "models of a failed load are still there");

        // Same session, no cleanup in between: a good model must load and run.
        ASSERT(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS, "load after failed loads: " + last_error());
        ASSERT(enqueue(make_request(1, 2, 2)) == RUNTIME_STATUS_SUCCESS, "enqueue failed: " + last_error());
        Tensors* out = retrieve();
        ASSERT(is_2x2_of_1234(out), "no valid output after failed loads");
        free_tensors(out);
        ASSERT(runtime_cleanup() == RUNTIME_STATUS_SUCCESS, "cleanup failed");
        std::remove(garbage);
    }
    PASS("3 failed loads leave nothing behind; a good model then loads and runs");

    std::cout << "\n=== All tests passed ===" << std::endl;
    return 0;
}
