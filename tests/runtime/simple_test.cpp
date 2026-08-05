/**
 * Simple test for the OAAX v2 runtime interface.
 *
 * Verifies:
 * 1. runtime_init() succeeds
 * 2. runtime_get_version() / runtime_get_name() return expected values
 * 3. runtime_enqueue_input() returns NOT_INITIALIZED / MODEL_NOT_LOADED before loading
 * 4. runtime_load_models() fails gracefully for a non-existent path
 * 5. runtime_get_error() returns a message after failure
 * 6. Model loading succeeds (optional, pass zip path as argument)
 * 7. runtime_cleanup() succeeds
 */

#include <cstring>
#include <iostream>

#include "oaax_runtime.h"

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

static void free_tensors(Tensors *t) __attribute__((unused));
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

int main(int argc, char **argv) {
    std::cout << "=== OAAX v2 Runtime Test ===" << std::endl;

    // ── 1. Init ───────────────────────────────────────────────────────────────
    std::cout << "\n[1] runtime_init()" << std::endl;
    const char *keys[] = {"log_level"};
    const char *values[] = {"2"};
    Config cfg = {1, keys, values};
    ASSERT(runtime_init(cfg) == RUNTIME_STATUS_SUCCESS, "runtime_init failed");
    PASS("runtime initialized");

    // ── 2. Version / name ─────────────────────────────────────────────────────
    std::cout << "\n[2] runtime_get_version() / runtime_get_name()" << std::endl;
    const char *ver = runtime_get_version();
    const char *name = runtime_get_name();
    ASSERT(ver && strlen(ver) > 0, "version is empty");
    ASSERT(name && strlen(name) > 0, "name is empty");
    ASSERT(strcmp(name, "OAAX Intel") == 0, std::string("unexpected name: ") + name);
    std::cout << "  name: " << name << "  version: " << ver << std::endl;
    PASS("version and name correct");

    // ── 3. Double init must fail ──────────────────────────────────────────────
    std::cout << "\n[3] double runtime_init() must return ALREADY_INITIALIZED" << std::endl;
    ASSERT(runtime_init(cfg) == RUNTIME_STATUS_ALREADY_INITIALIZED, "second runtime_init should fail");
    PASS("double-init rejected");

    // ── 4. enqueue_input before models loaded ─────────────────────────────────
    std::cout << "\n[4] runtime_enqueue_input() before load" << std::endl;
    {
        Tensors dummy{};
        dummy.num_tensors = 0;
        dummy.tensors = nullptr;
        RuntimeStatus st = runtime_enqueue_input(0, &dummy);
        ASSERT(st == RUNTIME_STATUS_MODEL_NOT_LOADED, "expected MODEL_NOT_LOADED");
    }
    PASS("enqueue rejected before load");

    // ── 5. load non-existent model ────────────────────────────────────────────
    std::cout << "\n[5] runtime_load_models() with bad path" << std::endl;
    {
        ModelConfig mc{};
        mc.file_path = "/nonexistent/path/model.zip";
        ASSERT(runtime_load_models(1, &mc) != RUNTIME_STATUS_SUCCESS, "load should fail for bad path");
        const char *err = runtime_get_error();
        ASSERT(err && strlen(err) > 0, "runtime_get_error() should be set after failure");
        std::cout << "  error: " << err << std::endl;
    }
    PASS("bad-path load failed with error message");

    // ── 6. load real model (optional) ─────────────────────────────────────────
    if (argc > 1) {
        std::cout << "\n[6] runtime_load_models(): " << argv[1] << std::endl;
        ModelConfig mc{};
        mc.file_path = argv[1];
        ASSERT(runtime_load_models(1, &mc) == RUNTIME_STATUS_SUCCESS,
               std::string("model load failed: ") + (runtime_get_error() ? runtime_get_error() : ""));
        PASS("model loaded");

        // Sanity: retrieve_output with timeout=0 must return NO_OUTPUT_AVAILABLE
        int mid = -1;
        Tensors *out = nullptr;
        ASSERT(runtime_retrieve_output(&mid, &out, 0) == RUNTIME_STATUS_NO_OUTPUT_AVAILABLE,
               "expected NO_OUTPUT_AVAILABLE with no inputs sent");
        PASS("retrieve on empty queue returns NO_OUTPUT_AVAILABLE");

        // runtime_get_info must return non-null JSON
        const char *info = runtime_get_info();
        ASSERT(info && info[0] == '{', "runtime_get_info should return JSON object");
        std::cout << "  info: " << info << std::endl;
        PASS("runtime_get_info OK");
    } else {
        std::cout << "\n[6] Skipping model load (no path provided)" << std::endl;
        std::cout << "    Usage: " << argv[0] << " <model.zip>" << std::endl;
    }

    // ── 7. Cleanup ────────────────────────────────────────────────────────────
    std::cout << "\n[7] runtime_cleanup()" << std::endl;
    ASSERT(runtime_cleanup() == RUNTIME_STATUS_SUCCESS, "cleanup failed");
    // Idempotent
    ASSERT(runtime_cleanup() == RUNTIME_STATUS_SUCCESS, "second cleanup failed");
    PASS("cleanup OK (idempotent)");

    // ── 8. log_stdout=1 accepted ──────────────────────────────────────────────
    std::cout << "\n[8] runtime_init() with log_stdout=1" << std::endl;
    {
        const char *k[] = {"log_level", "log_stdout"};
        const char *v[] = {"2", "1"};
        Config c = {2, k, v};
        ASSERT(runtime_init(c) == RUNTIME_STATUS_SUCCESS, "init with log_stdout=1 failed");
        ASSERT(runtime_cleanup() == RUNTIME_STATUS_SUCCESS, "cleanup after log_stdout=1 failed");
    }
    PASS("log_stdout=1 accepted");

    // ── 9. max_queue_size=0 (disabled) accepted ───────────────────────────────
    std::cout << "\n[9] runtime_init() with max_queue_size=0" << std::endl;
    {
        const char *k[] = {"log_level", "max_queue_size"};
        const char *v[] = {"2", "0"};
        Config c = {2, k, v};
        ASSERT(runtime_init(c) == RUNTIME_STATUS_SUCCESS, "init with max_queue_size=0 failed");
        ASSERT(runtime_cleanup() == RUNTIME_STATUS_SUCCESS, "cleanup after max_queue_size=0 failed");
    }
    PASS("max_queue_size=0 (disabled) accepted");

    std::cout << "\n=== All tests passed ===" << std::endl;
    return 0;
}
