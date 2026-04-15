# OAAX Runtime Library (OpenVINO)

C++ shared library implementing the OAAX inference interface on Intel hardware using the OpenVINO native C++ API. Supports CPU, GPU, and NPU on x86_64 Linux and Windows.

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

Output in `runtime-library/artifacts/Windows/`:  `RuntimeLibrary.dll` + OpenVINO + TBB DLLs.

## API usage

Include `runtime_core.hpp` and link against `RuntimeLibrary`.

```c
#include "runtime_core.hpp"

// 1. Initialize (device and performance hint are optional)
char *keys[]   = {"device_type", "perf_hint"};
void *values[] = {"CPU", "throughput"};
runtime_initialization_with_args(2, keys, values);

// 2. Load model (.zip bundle from the toolchain, or bare .xml with .bin alongside)
runtime_model_loading("/path/to/model.zip");

// 3. Send input (runtime takes ownership; do not free after this call)
send_input(input_tensors);

// 4. Poll for result
tensors_struct *output = NULL;
while (receive_output(&output) != 0) {}   // non-blocking; retry until ready

// 5. Use output, then return buffer to pool
runtime_return_output(output);            // preferred over deep_free_tensors_struct

// 6. Tear down
runtime_destruction();
```

> **Important:** Use `runtime_return_output()` to release output buffers, not
> `deep_free_tensors_struct()`. The runtime pre-allocates an output buffer pool for
> zero-copy inference; `runtime_return_output()` recycles the buffer back into the pool.
> `deep_free_tensors_struct()` frees the memory outright and should only be used if
> you received the output after calling `runtime_destruction()`.

## Configuration

All parameters are passed as strings to `runtime_initialization_with_args`.

| Key | Default | Values | Description |
|-----|---------|--------|-------------|
| `device_type` | `"CPU"` | `"CPU"` `"GPU"` `"NPU"` | Target inference device |
| `perf_hint` | `"latency"` | `"latency"` `"throughput"` `"cumulative_throughput"` | OpenVINO performance mode |
| `cache_dir` | `"."` (CWD) | any path or `""` | Directory for OpenVINO compiled-model cache. Eliminates recompilation on restart (~400 ms → ~47 ms). Set to `""` to disable. |
| `log_level` | `"2"` (info) | `"0"`–`"6"` | spdlog level: 0=trace, 2=info, 4=warn, 6=off |
| `log_file` | `"runtime.log"` | any path | Log output file |

If `device_type` compilation fails (e.g. no GPU present), the runtime automatically falls
back to CPU and logs a warning.

## Working with tensors

`tensors_struct` is defined in `include/tensors_struct.h`. Use the provided helpers:

```c
// Allocate
tensors_struct *ts = allocate_tensors_struct(1);
ts->names[0]      = strdup("images");
ts->ranks[0]      = 4;
ts->shapes[0]     = malloc(4 * sizeof(size_t));
ts->shapes[0][0]  = 1; ts->shapes[0][1] = 3;
ts->shapes[0][2]  = 640; ts->shapes[0][3] = 640;
ts->data_types[0] = DATA_TYPE_FLOAT;
ts->data[0]       = malloc(1 * 3 * 640 * 640 * sizeof(float));

// Free (only when NOT using runtime_return_output)
deep_free_tensors_struct(ts);
```

Supported data types: `DATA_TYPE_FLOAT` (FP32), `DATA_TYPE_INT8`, `DATA_TYPE_UINT8`,
`DATA_TYPE_INT32`, `DATA_TYPE_INT64`, and others — see `tensors_struct.h`.

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
All required `.so` files are in the `artifacts/X86_64/` directory. Deploy them alongside
`libRuntimeLibrary.so` and set `LD_LIBRARY_PATH` if not using RPATH:
```bash
export LD_LIBRARY_PATH=/path/to/artifacts/X86_64:$LD_LIBRARY_PATH
```

**Model loading fails**
Pass either a `.zip` bundle (output from the toolchain) or a bare `.xml` path with the `.bin` file in the same directory:
```bash
# Preferred — toolchain output bundle
runtime_model_loading("/path/to/model.zip");

# Also accepted — bare IR files
runtime_model_loading("/path/to/model.xml");
ls model.xml model.bin   # .bin must be alongside .xml
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
│   ├── runtime_core.hpp    # public C API (OAAX interface)
│   └── tensors_struct.h    # OAAX tensor structure and helpers
├── deps/
│   ├── spdlog/             # logging
│   ├── concurrentqueue/    # lock-free queue
│   └── tools/c-utilities/  # OAAX C utilities
├── cmake/
│   └── copy_windows_dlls.cmake
├── CMakeLists.txt
├── build-runtimes.sh       # Linux build script
└── build-runtime.bat       # Windows build script
```

Test sources: `../tests/runtime/simple_test.cpp`, `../tests/runtime/yolo_test.cpp`

## License

See repository [LICENSE](../LICENSE).
