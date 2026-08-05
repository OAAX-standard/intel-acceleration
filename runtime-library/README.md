# OAAX Runtime Library (OpenVINO)

C++ shared library implementing the OAAX v2 inference interface on Intel hardware using the OpenVINO native C++ API. Supports CPU, GPU, and NPU on x86_64 Linux and Windows.

## Prerequisites

- **OpenVINO 2026.1.0** archive from [storage.openvinotoolkit.org](https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.1/)
- **GCC 9.5+** and cross-compilation toolchain at `/opt/x86_64-unknown-linux-gnu-gcc-9.5.0` (Linux)
- **CMake 3.10.2+**

The CI setup script installs all build dependencies:

```bash
sudo bash scripts/setup-env.sh   # from the repo root
```

## Build

```bash
cd runtime-library
OPENVINO_DIR=/opt/intel/openvino/runtime bash build-runtimes.sh
```

Output in `runtime-library/artifacts/X86_64/`:

```
libRuntimeLibrary.so       your runtime
libopenvino.so.2610        OpenVINO core
libopenvino_intel_cpu_plugin.so
libopenvino_intel_gpu_plugin.so
libopenvino_intel_npu_plugin.so
libtbb.so.12               Intel TBB
...
```

All `.so` files have `$ORIGIN` RPATH set so they find each other when deployed together.

### Windows

```bat
cd runtime-library
build-runtime.bat
```

Output in `runtime-library/artifacts/Windows/`: `RuntimeLibrary.dll` + OpenVINO + TBB DLLs.

## API usage

Include `oaax_runtime.h` and link against `RuntimeLibrary`.

```c
#include "oaax_runtime.h"

// 1. Initialize runtime
const char *keys[]   = {"device_type", "perf_hint"};
const char *values[] = {"CPU", "latency"};
Config cfg = {2, keys, values};
runtime_init(cfg);

// 2. Load one or more models (zip bundle from the toolchain)
ModelConfig mc = {};
mc.file_path = "/path/to/model.zip";
runtime_load_models(1, &mc);

// 3. Enqueue input (runtime takes ownership on success)
Tensors *input = /* allocate and fill */;
input->id = 42;           // caller-assigned ID echoed on output
runtime_enqueue_input(0, input);   // 0 = model index

// 4. Retrieve output (blocks up to timeout_ms; -1 = wait forever)
int model_id;
Tensors *output = NULL;
while (runtime_retrieve_output(&model_id, &output, 5) ==
       RUNTIME_STATUS_NO_OUTPUT_AVAILABLE) {}

// use output->tensors[i].data ...
// output->id matches the input->id you set above

// 5. Free output (caller owns it after retrieve)
for (int i = 0; i < output->num_tensors; ++i) {
    free(output->tensors[i].name);
    free(output->tensors[i].shape);
    free(output->tensors[i].data);
}
free(output->tensors);
free(output);

// 6. Tear down
runtime_cleanup();
```

## Configuration

### Global (`runtime_init`)

| Key | Default | Values | Description |
|-----|---------|--------|-------------|
| `device_type` | `"CPU"` | `"CPU"` `"GPU"` `"NPU"` | Target inference device. `"GPU"` auto-activates MULTI if multiple GPUs found. |
| `perf_hint` | `"latency"` | `"latency"` `"throughput"` `"cumulative_throughput"` | OpenVINO performance mode |
| `cache_dir` | `"."` | any path or `""` | OpenVINO compiled-model cache. Eliminates recompilation on restart. `""` to disable. |
| `log_level` | `"2"` | `"0"`–`"6"` | spdlog level: 0=trace, 2=info, 4=warn, 6=off |
| `log_file` | `"runtime.log"` | any path | Log output file |

### Per-model (`ModelConfig.config`)

Each model inherits the global config and can override any key:

| Key | Inherits from | Description |
|-----|--------------|-------------|
| `device_type` | global | Run this model on a specific device |
| `perf_hint` | global | Override performance mode for this model |
| `cache_dir` | global | Override cache directory for this model |

Example — run two models on different devices:

```c
ModelConfig models[2] = {};
models[0].file_path = "/path/to/detector.zip";
// detector uses global device (CPU)

const char *gpu_keys[]   = {"device_type"};
const char *gpu_values[] = {"GPU"};
models[1].file_path = "/path/to/classifier.zip";
models[1].config    = (Config){1, gpu_keys, gpu_values};

runtime_load_models(2, models);
```

## Working with tensors

`TensorDescriptor` and `Tensors` are defined in `include/oaax_runtime.h`.

```c
// Allocate input
Tensors *ts = (Tensors *)malloc(sizeof(Tensors));
ts->id          = request_id;   // echoed on output for correlation
ts->num_tensors = 1;
ts->tensors     = (TensorDescriptor *)malloc(sizeof(TensorDescriptor));

TensorDescriptor *td = &ts->tensors[0];
td->name      = strdup("images");
td->data_type = DATA_TYPE_FLOAT;
td->rank      = 4;
td->shape     = (int *)malloc(4 * sizeof(int));
td->shape[0]  = 1; td->shape[1] = 3;
td->shape[2]  = 640; td->shape[3] = 640;
td->data_size = 1 * 3 * 640 * 640 * sizeof(float);
td->data      = malloc(td->data_size);
```

Supported data types: `DATA_TYPE_FLOAT` (FP32), `DATA_TYPE_FLOAT16`, `DATA_TYPE_BFLOAT16`,
`DATA_TYPE_INT8`, `DATA_TYPE_UINT8`, `DATA_TYPE_INT32`, `DATA_TYPE_INT64`, `DATA_TYPE_INT4`,
`DATA_TYPE_UINT4`, and more — see `oaax_runtime.h`.

## Multi-model inference

When multiple models are loaded, all output to a single global queue.
Use `output->id` (echoed from input) to correlate results with requests.

```c
// Enqueue to model 0
input0->id = 100;
runtime_enqueue_input(0, input0);

// Enqueue to model 1
input1->id = 200;
runtime_enqueue_input(1, input1);

// Retrieve — can come from any model
int model_id;
Tensors *output;
runtime_retrieve_output(&model_id, &output, -1);  // -1 = wait forever
// model_id tells you which model produced this output
// output->id tells you which request it was
```

## Error handling

```c
RuntimeStatus st = runtime_init(cfg);
if (st != RUNTIME_STATUS_SUCCESS) {
    const char *msg = runtime_get_error();  // human-readable description
    fprintf(stderr, "Init failed (%d): %s\n", st, msg);
}
```

`runtime_get_error()` returns NULL if no error has occurred.

## Diagnostics

```c
const char *info = runtime_get_info();
// Returns a JSON object, e.g.:
// {"loaded_models":2,"requests_in_flight":3,"backend_version":"2026.1.0-...","active_device":"CPU"}
```

## Troubleshooting

**OpenVINO not found at configure time**
```
CMake Error: OpenVINO not found at /opt/intel/openvino/runtime
```
Set `OPENVINO_DIR` to the `runtime/` subdirectory of your OpenVINO archive:
```bash
OPENVINO_DIR=/path/to/openvino_toolkit_.../runtime bash build-runtimes.sh
```

**Missing shared libraries at runtime**
```
error while loading shared libraries: libopenvino.so.2610: cannot open shared object file
```
Deploy all `.so` files from `artifacts/X86_64/` alongside `libRuntimeLibrary.so`:
```bash
export LD_LIBRARY_PATH=/path/to/artifacts/X86_64:$LD_LIBRARY_PATH
```

**Model loading fails**
Pass a `.zip` bundle as produced by the conversion toolchain:
```c
mc.file_path = "/path/to/model.zip";
runtime_load_models(1, &mc);
```

## Project structure

```
runtime-library/
├── src/
│   ├── runtime_core.cpp    # inference pipeline (OpenVINO native API)
│   ├── runtime_utils.cpp   # OAAX ↔ OpenVINO type mapping
│   ├── zip_utils.cpp       # ZIP extraction for .zip model bundles
│   └── zip_utils.hpp
├── include/
│   ├── oaax_runtime.h      # public C API (OAAX v2 interface)
│   └── runtime_utils.hpp   # internal utilities
├── deps/
│   ├── spdlog/             # logging
│   └── concurrentqueue/    # lock-free queue
├── cmake/
│   └── copy_windows_dlls.cmake
├── CMakeLists.txt
├── build-runtimes.sh       # Linux build script
└── build-runtime.bat       # Windows build script
```

Test sources: `../tests/runtime/simple_test.cpp`, `../tests/runtime/yolo_test.cpp`

## License

See repository [LICENSE](../LICENSE).
