/**
 * Runtime lifecycle tests for the OAAX v2 interface.
 *
 * Regression guard for AIMP-1477: runtime_cleanup() must be safe to follow
 * with another runtime_init() in the same loaded module, with a real model
 * compiled and run in between (not just an empty Core, which simple_test's
 * double-init/cleanup checks already cover). This is exactly the pipeline/
 * model-reselect pattern the host uses without ever unloading this library.
 *
 * Usage:
 *   ./lifecycle_test <model.zip> [--input-name NAME] [--imgsz N]
 */

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

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

static Config make_cfg() {
    static const char* keys[] = {"log_level"};
    static const char* vals[] = {"2"};
    return Config{1, keys, vals};
}

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

// Allocate a zero-filled float32 input (1 x 3 x imgsz x imgsz).
static Tensors* make_input(int imgsz, const char* input_name) {
    Tensors* t = (Tensors*)calloc(1, sizeof(Tensors));
    if (!t) return nullptr;
    t->id = 1;
    t->num_tensors = 1;
    t->tensors = (TensorDescriptor*)calloc(1, sizeof(TensorDescriptor));
    if (!t->tensors) {
        free(t);
        return nullptr;
    }

    TensorDescriptor& td = t->tensors[0];
    td.name = strdup(input_name);
    td.data_type = DATA_TYPE_FLOAT;
    td.rank = 4;
    td.shape = (int*)malloc(4 * sizeof(int));
    if (!td.shape) {
        free_tensors(t);
        return nullptr;
    }
    td.shape[0] = 1;
    td.shape[1] = 3;
    td.shape[2] = imgsz;
    td.shape[3] = imgsz;

    size_t n = (size_t)3 * imgsz * imgsz;
    td.data_size = n * sizeof(float);
    td.data = calloc(n, sizeof(float));
    if (!td.data) {
        free_tensors(t);
        return nullptr;
    }
    return t;
}

int main(int argc, char** argv) {
    const char* model_path = nullptr;
    const char* input_name = "images";
    int imgsz = 640;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--input-name") == 0 && i + 1 < argc)
            input_name = argv[++i];
        else if (strcmp(argv[i], "--imgsz") == 0 && i + 1 < argc)
            imgsz = atoi(argv[++i]);
        else if (argv[i][0] != '-' && !model_path)
            model_path = argv[i];
    }

    if (!model_path) {
        std::cerr << "Usage: " << argv[0] << " <model.zip> [--input-name NAME] [--imgsz N]" << std::endl;
        return 1;
    }

    Config cfg = make_cfg();

    std::cout << "=== OAAX Intel Runtime Lifecycle Tests ===" << std::endl;

    // ── 1. Full init/load/cleanup cycle repeated 3 times ─────────────────────
    std::cout << "\n[1] Full init/load/cleanup cycle x3: " << model_path << std::endl;
    for (int i = 1; i <= 3; ++i) {
        ASSERT(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS,
               std::string("cycle ") + std::to_string(i) + ": init failed");
        ModelConfig mc{};
        mc.file_path = model_path;
        ASSERT(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS,
               std::string("cycle ") + std::to_string(i) +
                   ": load failed: " + (runtime_get_error() ? runtime_get_error() : ""));
        ASSERT(runtime_cleanup() == RUNTIME_STATUS_SUCCESS,
               std::string("cycle ") + std::to_string(i) + ": cleanup failed");
    }
    PASS("3 full init/load/cleanup cycles completed");

    // ── 2. Inference round-trip works on cycle 2 (after a full teardown) ─────
    // This is the exact pattern a pipeline/model reselect exercises on the
    // live module: a real inference session torn down, then immediately
    // reinitialized and driven again, without ever unloading this library.
    std::cout << "\n[2] Inference round-trip works on cycle 2 (after full teardown/reinit)" << std::endl;
    {
        ModelConfig mc{};
        mc.file_path = model_path;

        // Cycle 1 — load, run one inference, then clean up.
        ASSERT(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS, "cycle 1: init failed");
        ASSERT(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS,
               std::string("cycle 1: load failed: ") + (runtime_get_error() ? runtime_get_error() : ""));

        Tensors* input1 = make_input(imgsz, input_name);
        ASSERT(input1 != nullptr, "cycle 1: failed to allocate input tensor");
        ASSERT(runtime_enqueue_input(0, input1) == RUNTIME_STATUS_SUCCESS, "cycle 1: enqueue failed");

        int mid = -1;
        Tensors* out = nullptr;
        RuntimeStatus st = RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;
        for (int tries = 0; tries < 50 && st != RUNTIME_STATUS_SUCCESS; ++tries)
            st = runtime_retrieve_output(&mid, &out, 200);
        ASSERT(st == RUNTIME_STATUS_SUCCESS, "cycle 1: retrieve failed");
        free_tensors(out);

        ASSERT(runtime_cleanup() == RUNTIME_STATUS_SUCCESS, "cycle 1: cleanup failed");

        // Cycle 2 — immediately reinit and run another full round-trip. Before
        // the AIMP-1477 fix, runtime_cleanup() called ov::shutdown() here,
        // racing this cycle's fresh ov::Core() against TBB's own teardown of
        // the previous cycle's NUMA/topology state (crash in tbbbind).
        ASSERT(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS, "cycle 2: init failed");
        ASSERT(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS,
               std::string("cycle 2: load failed: ") + (runtime_get_error() ? runtime_get_error() : ""));

        Tensors* input2 = make_input(imgsz, input_name);
        ASSERT(input2 != nullptr, "cycle 2: failed to allocate input tensor");
        ASSERT(runtime_enqueue_input(0, input2) == RUNTIME_STATUS_SUCCESS, "cycle 2: enqueue failed");

        mid = -1;
        out = nullptr;
        st = RUNTIME_STATUS_NO_OUTPUT_AVAILABLE;
        for (int tries = 0; tries < 50 && st != RUNTIME_STATUS_SUCCESS; ++tries)
            st = runtime_retrieve_output(&mid, &out, 200);
        ASSERT(st == RUNTIME_STATUS_SUCCESS, "cycle 2: retrieve failed");
        free_tensors(out);

        ASSERT(runtime_cleanup() == RUNTIME_STATUS_SUCCESS, "cycle 2: cleanup failed");
    }
    PASS("inference round-trip survives a full teardown/reinit cycle");

    std::cout << "\n=== All tests passed ===" << std::endl;
    return 0;
}
