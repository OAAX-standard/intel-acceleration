/**
 * Output validation test: runs inference and checks every output tensor for
 * NaN and Inf values across all floating-point types (f32, f64, f16, bf16).
 *
 * Usage:
 *   ./output_validation_test <model.zip> [device] [--runs N]
 * Defaults: device=CPU, runs=10
 *
 * Exit code: 0 = all outputs clean, 1 = garbage detected or error.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "oaax_runtime.h"

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

static Tensors *make_zero_input(int rank, const int *shape, TensorElementType dtype,
                                const char *input_name = "images") {
    Tensors *ts = (Tensors *)malloc(sizeof(Tensors));
    if (!ts) return nullptr;
    ts->id = 0;
    ts->num_tensors = 1;
    ts->tensors = (TensorDescriptor *)malloc(sizeof(TensorDescriptor));
    if (!ts->tensors) {
        free(ts);
        return nullptr;
    }

    TensorDescriptor &td = ts->tensors[0];
    td.name = strdup(input_name);
    td.data_type = dtype;
    td.rank = rank;
    td.shape = (int *)malloc((size_t)rank * sizeof(int));
    size_t n_elems = 1;
    for (int i = 0; i < rank; ++i) {
        td.shape[i] = shape[i];
        n_elems *= (size_t)shape[i];
    }
    // Use 4 bytes per element as a safe default (covers f32/i32/u32)
    size_t elem_bytes = 4;
    if (dtype == DATA_TYPE_UINT8 || dtype == DATA_TYPE_INT8) elem_bytes = 1;
    if (dtype == DATA_TYPE_FLOAT16 || dtype == DATA_TYPE_BFLOAT16 || dtype == DATA_TYPE_INT16 ||
        dtype == DATA_TYPE_UINT16)
        elem_bytes = 2;
    if (dtype == DATA_TYPE_DOUBLE || dtype == DATA_TYPE_INT64 || dtype == DATA_TYPE_UINT64) elem_bytes = 8;

    td.data_size = n_elems * elem_bytes;
    td.data = calloc(1, td.data_size);
    if (!td.data) {
        free(td.shape);
        free((void *)td.name);
        free(ts->tensors);
        free(ts);
        return nullptr;
    }
    return ts;
}

// ─── Per-tensor validation ────────────────────────────────────────────────────

static constexpr double RANGE_MIN = -1000.0;
static constexpr double RANGE_MAX = 1000.0;

struct ValidationResult {
    size_t nan_count{0};
    size_t inf_count{0};
    size_t out_of_range_count{0};
    size_t total_elements{0};
    bool checked{false};
};

// Decode an f16 bit pattern to double for range checking.
static double f16_to_double(uint16_t v) {
    int sign = (v >> 15) & 1;
    int exp = (v >> 10) & 0x1F;
    int mantissa = v & 0x3FF;
    if (exp == 0) return (sign ? -1.0 : 1.0) * std::ldexp((double)mantissa, -24);  // subnormal
    if (exp == 0x1F)
        return mantissa ? std::numeric_limits<double>::quiet_NaN()
                        : (sign ? -1.0 : 1.0) * std::numeric_limits<double>::infinity();
    return (sign ? -1.0 : 1.0) * std::ldexp(1.0 + mantissa / 1024.0, exp - 15);
}

// Decode a bf16 bit pattern to double for range checking.
static double bf16_to_double(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return (double)f;
}

static ValidationResult check_tensor(const TensorDescriptor &td) {
    ValidationResult r;
    if (!td.data || td.data_size == 0) return r;
    r.checked = true;

    if (td.data_type == DATA_TYPE_FLOAT) {
        const float *d = static_cast<const float *>(td.data);
        r.total_elements = td.data_size / sizeof(float);
        for (size_t i = 0; i < r.total_elements; ++i) {
            if (std::isnan(d[i])) {
                ++r.nan_count;
                continue;
            }
            if (std::isinf(d[i])) {
                ++r.inf_count;
                continue;
            }
            if (d[i] < RANGE_MIN || d[i] > RANGE_MAX) ++r.out_of_range_count;
        }
    } else if (td.data_type == DATA_TYPE_DOUBLE) {
        const double *d = static_cast<const double *>(td.data);
        r.total_elements = td.data_size / sizeof(double);
        for (size_t i = 0; i < r.total_elements; ++i) {
            if (std::isnan(d[i])) {
                ++r.nan_count;
                continue;
            }
            if (std::isinf(d[i])) {
                ++r.inf_count;
                continue;
            }
            if (d[i] < RANGE_MIN || d[i] > RANGE_MAX) ++r.out_of_range_count;
        }
    } else if (td.data_type == DATA_TYPE_FLOAT16) {
        // f16: sign(1) | exp(5) | mantissa(10). Exp all-1s => NaN or Inf.
        const uint16_t *d = static_cast<const uint16_t *>(td.data);
        r.total_elements = td.data_size / sizeof(uint16_t);
        for (size_t i = 0; i < r.total_elements; ++i) {
            uint16_t v = d[i];
            if ((v & 0x7C00u) == 0x7C00u) {
                if (v & 0x03FFu)
                    ++r.nan_count;
                else
                    ++r.inf_count;
                continue;
            }
            double val = f16_to_double(v);
            if (val < RANGE_MIN || val > RANGE_MAX) ++r.out_of_range_count;
        }
    } else if (td.data_type == DATA_TYPE_BFLOAT16) {
        // bf16: sign(1) | exp(8) | mantissa(7). Exp all-1s => NaN or Inf.
        const uint16_t *d = static_cast<const uint16_t *>(td.data);
        r.total_elements = td.data_size / sizeof(uint16_t);
        for (size_t i = 0; i < r.total_elements; ++i) {
            uint16_t v = d[i];
            if ((v & 0x7F80u) == 0x7F80u) {
                if (v & 0x007Fu)
                    ++r.nan_count;
                else
                    ++r.inf_count;
                continue;
            }
            double val = bf16_to_double(v);
            if (val < RANGE_MIN || val > RANGE_MAX) ++r.out_of_range_count;
        }
    } else if (td.data_type == DATA_TYPE_INT8) {
        const int8_t *d = static_cast<const int8_t *>(td.data);
        r.total_elements = td.data_size;
        for (size_t i = 0; i < r.total_elements; ++i)
            if (d[i] < (int8_t)RANGE_MIN || d[i] > (int8_t)RANGE_MAX) ++r.out_of_range_count;
    } else if (td.data_type == DATA_TYPE_INT16) {
        const int16_t *d = static_cast<const int16_t *>(td.data);
        r.total_elements = td.data_size / sizeof(int16_t);
        for (size_t i = 0; i < r.total_elements; ++i)
            if (d[i] < (int16_t)RANGE_MIN || d[i] > (int16_t)RANGE_MAX) ++r.out_of_range_count;
    } else if (td.data_type == DATA_TYPE_INT32) {
        const int32_t *d = static_cast<const int32_t *>(td.data);
        r.total_elements = td.data_size / sizeof(int32_t);
        for (size_t i = 0; i < r.total_elements; ++i)
            if (d[i] < (int32_t)RANGE_MIN || d[i] > (int32_t)RANGE_MAX) ++r.out_of_range_count;
    } else if (td.data_type == DATA_TYPE_INT64) {
        const int64_t *d = static_cast<const int64_t *>(td.data);
        r.total_elements = td.data_size / sizeof(int64_t);
        for (size_t i = 0; i < r.total_elements; ++i)
            if (d[i] < (int64_t)RANGE_MIN || d[i] > (int64_t)RANGE_MAX) ++r.out_of_range_count;
    } else if (td.data_type == DATA_TYPE_UINT8) {
        // uint8 is always >= 0; only check upper bound
        const uint8_t *d = static_cast<const uint8_t *>(td.data);
        r.total_elements = td.data_size;
        for (size_t i = 0; i < r.total_elements; ++i)
            if (d[i] > (uint8_t)RANGE_MAX) ++r.out_of_range_count;
    } else if (td.data_type == DATA_TYPE_UINT16) {
        const uint16_t *d = static_cast<const uint16_t *>(td.data);
        r.total_elements = td.data_size / sizeof(uint16_t);
        for (size_t i = 0; i < r.total_elements; ++i)
            if (d[i] > (uint16_t)RANGE_MAX) ++r.out_of_range_count;
    } else if (td.data_type == DATA_TYPE_UINT32) {
        const uint32_t *d = static_cast<const uint32_t *>(td.data);
        r.total_elements = td.data_size / sizeof(uint32_t);
        for (size_t i = 0; i < r.total_elements; ++i)
            if (d[i] > (uint32_t)RANGE_MAX) ++r.out_of_range_count;
    } else if (td.data_type == DATA_TYPE_UINT64) {
        const uint64_t *d = static_cast<const uint64_t *>(td.data);
        r.total_elements = td.data_size / sizeof(uint64_t);
        for (size_t i = 0; i < r.total_elements; ++i)
            if (d[i] > (uint64_t)RANGE_MAX) ++r.out_of_range_count;
    } else {
        // bool / string / exotic types: skip
        r.checked = false;
    }

    return r;
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.zip> [device] [--runs N]\n";
        return 1;
    }

    const char *model_path = argv[1];
    const char *device = "CPU";
    const char *input_name = "images";
    int runs = 10;
    int batch = 1;
    int imgsz = 640;

    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc)
            runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc)
            batch = atoi(argv[++i]);
        else if (strcmp(argv[i], "--imgsz") == 0 && i + 1 < argc)
            imgsz = atoi(argv[++i]);
        else if (strcmp(argv[i], "--input-name") == 0 && i + 1 < argc)
            input_name = argv[++i];
        else
            device = argv[i];
    }

    std::cout << "=== Output Validation Test ===\n";
    std::cout << "Model : " << model_path << "\n";
    std::cout << "Device: " << device << "\n";
    std::cout << "Runs  : " << runs << "\n";
    std::cout << "Input : " << batch << "x3x" << imgsz << "x" << imgsz << "\n\n";

    // Init
    const char *init_keys[] = {"device_type", "log_stdout"};
    const char *init_vals[] = {device, "0"};
    Config init_cfg{2, init_keys, init_vals};
    if (runtime_init(init_cfg) != RUNTIME_STATUS_SUCCESS) {
        std::cerr << "FAIL: runtime_init — " << (runtime_get_error() ? runtime_get_error() : "?") << "\n";
        return 1;
    }

    // Load model
    Config empty_cfg{0, nullptr, nullptr};
    ModelConfig mc{model_path, nullptr, 0, empty_cfg};
    if (runtime_load_models(1, &mc) != RUNTIME_STATUS_SUCCESS) {
        std::cerr << "FAIL: runtime_load_models — " << (runtime_get_error() ? runtime_get_error() : "?") << "\n";
        runtime_cleanup();
        return 1;
    }

    int shape[] = {batch, 3, imgsz, imgsz};
    int total_garbage = 0;
    int total_checked = 0;

    for (int run = 0; run < runs; ++run) {
        Tensors *input = make_zero_input(4, shape, DATA_TYPE_FLOAT, input_name);
        if (!input) {
            std::cerr << "FAIL: OOM allocating input\n";
            runtime_cleanup();
            return 1;
        }
        input->id = run;

        if (runtime_enqueue_input(0, input) != RUNTIME_STATUS_SUCCESS) {
            std::cerr << "FAIL: runtime_enqueue_input run=" << run << "\n";
            free_tensors(input);
            runtime_cleanup();
            return 1;
        }

        int model_id = -1;
        Tensors *output = nullptr;
        RuntimeStatus st;
        do {
            st = runtime_retrieve_output(&model_id, &output, 1000);
        } while (st == RUNTIME_STATUS_NO_OUTPUT_AVAILABLE);

        if (st != RUNTIME_STATUS_SUCCESS || !output) {
            std::cerr << "FAIL: runtime_retrieve_output run=" << run << "\n";
            runtime_cleanup();
            return 1;
        }

        bool run_clean = true;
        for (int t = 0; t < output->num_tensors; ++t) {
            const TensorDescriptor &td = output->tensors[t];
            ValidationResult r = check_tensor(td);
            if (!r.checked) continue;
            ++total_checked;

            const char *name = td.name ? td.name : "(unnamed)";
            bool tensor_bad = r.nan_count > 0 || r.inf_count > 0 || r.out_of_range_count > 0;
            if (tensor_bad) {
                ++total_garbage;
                run_clean = false;
                std::cerr << "  [run " << run << "] tensor '" << name << "': "
                          << "NaN=" << r.nan_count << " Inf=" << r.inf_count
                          << " OutOfRange[-1000,1000]=" << r.out_of_range_count << " / " << r.total_elements
                          << " elements\n";
            }
        }

        std::cout << "  run " << run << ": " << (run_clean ? "OK" : "GARBAGE DETECTED") << "\n";
        free_tensors(output);
    }

    runtime_cleanup();

    std::cout << "\n=== Summary ===\n";
    std::cout << "  Tensors checked : " << total_checked << "\n";
    std::cout << "  Garbage tensors : " << total_garbage << "\n";

    if (total_garbage > 0) {
        std::cout << "  Result: FAIL\n";
        return 1;
    }
    std::cout << "  Result: PASS\n";
    return 0;
}
