# Skills and Implementation Patterns

This document contains concrete examples and patterns for working on the Intel Acceleration Project. Use this as a reference when implementing features or fixing issues.

---

## Table of Contents

1. [OpenVINO Conversion Patterns](#openvino-conversion-patterns)
2. [OpenVINO Runtime Patterns](#openvino-runtime-patterns)
3. [Error Handling Patterns](#error-handling-patterns)
4. [Testing Patterns](#testing-patterns)
5. [Docker Patterns](#docker-patterns)
6. [CMake Patterns](#cmake-patterns)
7. [Type Mapping Patterns](#type-mapping-patterns)
8. [Memory Management Patterns](#memory-management-patterns)
9. [Secret & Confidentiality Scanning](#secret--confidentiality-scanning)

---

## OpenVINO Conversion Patterns

### Pattern: Convert ONNX to OpenVINO IR

**Location:** `conversion-toolchain/conversion_toolchain/utils.py`

```python
import openvino as ov
from pathlib import Path

def convert_to_ir(onnx_path: str, output_dir: str, compress_fp16: bool = True):
    """Convert ONNX model to OpenVINO IR format"""

    # Step 1: Load ONNX model using OpenVINO
    model = ov.convert_model(onnx_path)

    # Step 2: Apply FP16 compression if requested
    if compress_fp16:
        from openvino.tools.mo import compress_model_transformation
        model = compress_model_transformation(model)

    # Step 3: Save as IR format (generates .xml and .bin)
    output_path = Path(output_dir) / "model.xml"
    ov.save_model(model, output_path, compress_to_fp16=compress_fp16)

    return output_path
```

**Key Points:**
- `ov.convert_model()` replaces old ONNX simplifier
- Returns `ov.Model` object (can be manipulated before saving)
- `ov.save_model()` generates both .xml and .bin files
- FP16 compression happens during save

### Pattern: INT8 Quantization with NNCF

**Location:** `conversion-toolchain/conversion_toolchain/quantization.py`

```python
import nncf
import openvino as ov

def quantize_model_int8(model: ov.Model, calibration_data, preset: str = "mixed"):
    """Quantize model to INT8 using NNCF"""

    # Step 1: Create calibration dataset
    def transform_fn(data_item):
        # Transform your data to model input format
        return {input_name: data_item}

    calibration_dataset = nncf.Dataset(
        calibration_data,
        transform_fn
    )

    # Step 2: Quantize with specified preset
    quantized_model = nncf.quantize(
        model,
        calibration_dataset,
        preset=nncf.QuantizationPreset(preset),  # "performance", "mixed", "accuracy"
        subset_size=300  # Number of calibration samples
    )

    return quantized_model
```

**Presets:**
- `performance` - Max speed, some accuracy loss
- `mixed` - Balanced (recommended)
- `accuracy` - Max accuracy preservation

**Important:** INT8 quantization requires calibration images

### Pattern: Bundle Handling (Zip Input/Output)

**Location:** `conversion-toolchain/conversion_toolchain/utils.py`

```python
import zipfile
from pathlib import Path
import tempfile

def extract_input_bundle(input_zip: str, extract_dir: str, logs=None):
    """Extract and validate input bundle"""

    input_path = Path(input_zip)

    # Validation
    if not input_path.exists():
        raise FileNotFoundError(f"Input zip file does not exist: {input_zip}")

    if input_path.stat().st_size == 0:
        raise ValueError(f"Input zip file is empty: {input_zip}")

    # Test integrity
    with zipfile.ZipFile(input_zip, 'r') as zipf:
        bad_file = zipf.testzip()
        if bad_file:
            raise zipfile.BadZipFile(f"Corrupted file in archive: {bad_file}")

        # Extract
        zipf.extractall(extract_dir)

    # Find ONNX model
    onnx_files = list(Path(extract_dir).glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError("No ONNX model found in bundle")

    model_path = str(onnx_files[0])

    # Find optional config and calibration data
    config_path = Path(extract_dir) / "config.json"
    config_path = str(config_path) if config_path.exists() else None

    calib_dirs = [d for d in Path(extract_dir).iterdir() if d.is_dir()]
    calib_path = str(calib_dirs[0]) if calib_dirs else None

    return model_path, config_path, calib_path

def create_output_bundle(ir_xml_path: str, output_zip: str):
    """Bundle IR files (.xml + .bin) into zip"""

    xml_path = Path(ir_xml_path)
    bin_path = xml_path.with_suffix('.bin')

    # Validate both files exist
    if not xml_path.exists():
        raise FileNotFoundError(f"IR XML file not found: {xml_path}")
    if not bin_path.exists():
        raise FileNotFoundError(f"IR BIN file not found: {bin_path}")

    # Create zip with both files
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(xml_path, xml_path.name)
        zipf.write(bin_path, bin_path.name)

    return output_zip
```

**Bundle Structure:**

**Input Bundle:**
```
model.zip
├── model.onnx          # Required
├── config.json         # Optional
└── calibration/        # Optional (for INT8)
    ├── img1.jpg
    └── img2.jpg
```

**Output Bundle:**
```
model.zip
├── model.xml           # OpenVINO IR (graph)
└── model.bin           # OpenVINO IR (weights)
```

---

## OpenVINO Runtime Patterns

### Pattern: Initialize OpenVINO Runtime

**Location:** `runtime-library/src/runtime_core.cpp`

```cpp
#include <openvino/openvino.hpp>

// Global variables
static std::shared_ptr<ov::Core> core;
static std::shared_ptr<ov::CompiledModel> compiled_model;

extern "C" int runtime_initialization() {
    try {
        // Create OpenVINO Core
        core = std::make_shared<ov::Core>();
        return 0;  // Success
    }
    catch (const std::exception &e) {
        logger->error("Initialization failed: {}", e.what());
        return -1;  // Failure
    }
}
```

**Key Points:**
- `ov::Core` is the main entry point
- **DO NOT** call `core->set_property("CPU", ov::inference_num_threads(N))` when using a performance hint — it conflicts with OpenVINO's internal scheduler and hurts throughput
- Let `perf_hint` alone control thread allocation

### Pattern: Load and Compile Model

**Location:** `runtime-library/src/runtime_core.cpp`

```cpp
extern "C" int runtime_model_loading(const char *model_path) {
    try {
        // Step 1: Read IR model (.xml file, .bin must be in same directory)
        std::shared_ptr<ov::Model> model = core->read_model(model_path);

        // Step 2: Get input/output information
        for (const auto& input : model->inputs()) {
            std::string name = input.get_any_name();
            ov::Shape shape = input.get_shape();
            ov::element::Type type = input.get_element_type();
            logger->trace("Input: {} - Shape: {} - Type: {}",
                         name, shape, type);
        }

        for (const auto& output : model->outputs()) {
            std::string name = output.get_any_name();
            output_names.push_back(name);
        }

        // Step 3: Compile model for target device
        compiled_model = std::make_shared<ov::CompiledModel>(
            core->compile_model(model, "CPU")  // or "GPU", "NPU"
        );

        // Step 4: Create inference request
        infer_request = std::make_shared<ov::InferRequest>(
            compiled_model->create_infer_request()
        );

        return 0;
    }
    catch (const std::exception &e) {
        logger->error("Model loading failed: {}", e.what());
        return -1;
    }
}
```

**Model Loading Notes:**
- Pass `.xml` file path (not `.bin`)
- `.bin` file must be in same directory
- File names must match (model.xml + model.bin)
- `compile_model()` optimizes for target device

### Pattern: Run Inference

**Location:** `runtime-library/src/runtime_core.cpp`

```cpp
void perform_inference(tensors_struct *input_tensors) {
    // Step 1: Set input tensors
    for (size_t i = 0; i < input_tensors->num_tensors; ++i) {
        std::string tensor_name = input_tensors->names[i];

        // Create shape
        ov::Shape shape;
        for (size_t j = 0; j < input_tensors->ranks[i]; ++j) {
            shape.push_back(input_tensors->shapes[i][j]);
        }

        // Map data type
        ov::element::Type ov_type = map_to_ov_type(input_tensors->data_types[i]);

        // Create tensor (wraps existing data, no copy)
        ov::Tensor input_tensor(ov_type, shape, input_tensors->data[i]);

        // Set to infer request
        infer_request->set_tensor(tensor_name, input_tensor);
    }

    // Step 2: Run inference (synchronous)
    infer_request->infer();

    // Step 3: Get output tensors
    for (const auto& output_name : output_names) {
        ov::Tensor output_tensor = infer_request->get_tensor(output_name);

        // Access output data
        ov::Shape shape = output_tensor.get_shape();
        ov::element::Type type = output_tensor.get_element_type();
        void* data = output_tensor.data();
        size_t byte_size = output_tensor.get_byte_size();

        // Copy to output structure
        // ... (see runtime_core.cpp for full implementation)
    }
}
```

**Inference Notes:**
- `set_tensor()` wraps existing memory (efficient)
- `infer()` is synchronous (blocks until done)
- `get_tensor()` returns output without copy
- Must copy output data if ownership needed

### Pattern: Async Inference — Manager Thread + Semaphore Dispatch

This is the pattern used in production (`runtime_core.cpp`). One manager thread dispatches work; OpenVINO's own threads execute inference and fire the completion callback.

```cpp
#include <semaphore.h>

static sem_t slot_sem;   // counts free InferRequest slots
static sem_t input_sem;  // signals that send_input() enqueued a new item

struct SlotState {
    tensors_struct *input{nullptr};
    tensors_struct *output{nullptr};
    bool from_pool{false};
};
static std::vector<SlotState> slot_states;

// Register callback ONCE per slot at model load (not per inference).
// Lambda captures only int i (4 bytes) — fits std::function's SOO buffer,
// avoiding a heap allocation on every inference.
for (int i = 0; i < actual_requests; ++i) {
    infer_requests[i].set_callback([i](std::exception_ptr ex) {
        on_inference_complete(i, ex);
    });
}

// Manager thread: blocks on semaphores, never polls/sleeps.
static void manager_thread_func() {
    while (true) {
        while (sem_wait(&input_sem) == -1 && errno == EINTR) continue;
        if (stop_manager) return;
        tensors_struct *input = nullptr;
        input_tensors_queue.try_dequeue(input);

        while (sem_wait(&slot_sem) == -1 && errno == EINTR) continue;
        if (stop_manager) { deep_free_tensors_struct(input); return; }
        int idx = -1;
        free_requests.try_dequeue(idx);

        // Set outputs (zero-copy pool path) and inputs, then dispatch.
        slot_states[idx] = {input, output, from_pool};
        infer_requests[idx].start_async();
    }
}

// Completion callback: reads per-slot state, enqueues result, frees slot.
static void on_inference_complete(int idx, std::exception_ptr ex) {
    SlotState &s = slot_states[idx];
    // use s.input, s.output, s.from_pool
    output_tensors_queue.try_enqueue(s.output);
    free_requests.enqueue(idx);
    sem_post(&slot_sem);  // return slot to the pool
}

// send_input wakes the manager immediately — no 1ms sleep.
extern "C" int send_input(tensors_struct *input) {
    input_tensors_queue.try_enqueue(input);
    sem_post(&input_sem);
    return 0;
}
```

**Key Points:**
- `sem_t` (POSIX, futex-backed) is lighter than `std::mutex` + `std::condition_variable` — 1 syscall vs 2 in the contended path
- Per-slot `SlotState` avoids capturing large data in the lambda — keeping it inside std::function's small-object buffer
- `set_callback()` once at model load, not once per inference — eliminates repeated std::function construction
- FIFO ordering is NOT guaranteed when `actual_requests > 1`

### Pattern: Zero-Copy Output with set_output_tensor

Before calling `start_async()`, redirect the InferRequest's output directly into a pre-allocated pool buffer. OpenVINO writes inference results in-place — no memcpy in the callback.

```cpp
// Pre-allocate pool buffers (once at model load).
struct OutInfo { ov::Shape shape; ov::element::Type ov_type; size_t byte_size; };
static std::vector<OutInfo> pool_out_infos;
// ... fill pool_out_infos from compiled_model->output(name) ...

// Before each start_async():
tensors_struct *output = /* dequeue from output_buffer_pool */;
for (size_t i = 0; i < output_names.size(); ++i)
    req.set_output_tensor(i, ov::Tensor(pool_out_infos[i].ov_type,
                                        pool_out_infos[i].shape,
                                        output->data[i]));

// In the completion callback: output is already filled — just enqueue it.
output_tensors_queue.try_enqueue(s.output);

// Caller must return buffers to the pool instead of deep_free:
runtime_return_output(output);  // puts it back in output_buffer_pool
```

**Key Points:**
- Only works for static output shapes (dynamic shapes fall back to memcpy)
- Pool size: `actual_requests × 4` gives enough slack at high FPS without wasting memory
- Callers MUST use `runtime_return_output()` instead of `deep_free_tensors_struct()`

---

## Error Handling Patterns

### Pattern: Exit Codes

**Location:** `conversion-toolchain/conversion_toolchain/main.py`

```python
import sys
import zipfile

def main():
    try:
        # Normal execution
        convert_model(input_zip, output_dir)
        logs.save_as_json(f"{output_dir}/logs.json")
        sys.exit(0)  # Success

    except FileNotFoundError as e:
        logs.add_message('FATAL ERROR: File not found', {'Error': str(e)})
        logs.save_as_json(f"{output_dir}/logs.json")
        sys.exit(1)  # File not found

    except zipfile.BadZipFile as e:
        logs.add_message('FATAL ERROR: Invalid zip archive', {'Error': str(e)})
        logs.save_as_json(f"{output_dir}/logs.json")
        sys.exit(2)  # Invalid input

    except RuntimeError as e:
        logs.add_message('FATAL ERROR: Conversion failed', {'Error': str(e)})
        logs.save_as_json(f"{output_dir}/logs.json")
        sys.exit(3)  # Conversion failed

    except IOError as e:
        logs.add_message('FATAL ERROR: I/O operation failed', {'Error': str(e)})
        logs.save_as_json(f"{output_dir}/logs.json")
        sys.exit(4)  # I/O error

    except Exception as e:
        logs.add_message('FATAL ERROR: Unexpected error', {'Error': str(e)})
        logs.save_as_json(f"{output_dir}/logs.json")
        sys.exit(255)  # Unexpected error
```

**Exit Code Convention:**
| Code | Meaning | When to Use |
|------|---------|-------------|
| 0 | Success | Operation completed successfully |
| 1 | File not found | Input file doesn't exist |
| 2 | Invalid input | Bad zip, corrupt model, invalid format |
| 3 | Conversion/Inference failed | OpenVINO operation failed |
| 4 | I/O error | Cannot read/write files |
| 255 | Unexpected | Unknown error, shouldn't happen |

### Pattern: Logging with Context

**Location:** `conversion-toolchain/conversion_toolchain/logging_utils.py`

```python
class Logs:
    def add_message(self, message: str, attributes: dict = None):
        """Add structured log message"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': message,
            'attributes': attributes or {}
        }
        self.messages.append(log_entry)

    def save_as_json(self, filepath: str):
        """Save logs as JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.messages, f, indent=2)

# Usage
logs = Logs()
logs.add_message('Converting model', {
    'input': 'model.onnx',
    'output': 'model.xml',
    'fp16': True,
    'device': 'CPU'
})
logs.add_message('Conversion complete', {
    'xml_size': 1024,
    'bin_size': 4096
})
logs.save_as_json('output/logs.json')
```

**C++ Logging:**
```cpp
#include <spdlog/spdlog.h>

// Different log levels
logger->trace("Detailed debug info: {}", value);
logger->debug("Debug information: {}", status);
logger->info("Important milestone: {}", event);
logger->warn("Warning condition: {}", issue);
logger->error("Error occurred: {}", error);
```

---

## Testing Patterns

### Pattern: Unit Tests (Python)

**Location:** `conversion-toolchain/tests/test_conversion.py`

```python
import pytest
from pathlib import Path
import zipfile

class TestConversion:
    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Provide temporary directory"""
        return str(tmp_path)

    @pytest.fixture
    def sample_model(self, temp_dir):
        """Create sample ONNX model"""
        # Create simple model
        import onnx
        # ... model creation code ...
        return model_path

    def test_convert_to_ir(self, sample_model, temp_dir):
        """Test ONNX to IR conversion"""
        logs = Logs()

        # Perform conversion
        output_path = convert_to_ir(sample_model, temp_dir, logs)

        # Verify outputs exist
        assert Path(output_path).exists()
        assert Path(output_path).with_suffix('.bin').exists()

        # Verify loadable by OpenVINO
        import openvino as ov
        model = ov.Core().read_model(output_path)
        assert model is not None

    def test_invalid_input(self, temp_dir):
        """Test error handling for invalid input"""
        logs = Logs()

        with pytest.raises(FileNotFoundError):
            extract_input_bundle("/nonexistent/file.zip", temp_dir, logs)

# Run tests
# pytest tests/test_conversion.py -v
```

### Pattern: Docker Tests

**Location:** `conversion-toolchain/tests/test_docker.py`

```python
import pytest
import subprocess
import docker

class TestDockerImage:
    @pytest.fixture(scope="class")
    def docker_client(self):
        """Docker client fixture"""
        return docker.from_env()

    def test_docker_image_exists(self, docker_client):
        """Verify Docker image is built"""
        images = docker_client.images.list("oaax-intel-toolchain")
        assert len(images) > 0

    def test_docker_help(self, docker_client):
        """Test help command"""
        result = docker_client.containers.run(
            "oaax-intel-toolchain",
            command="--help",
            remove=True
        )
        assert b"usage:" in result.lower()

    def test_docker_conversion(self, docker_client, temp_dir):
        """Test full conversion in Docker"""
        # Prepare input
        input_dir = Path(temp_dir) / "input"
        output_dir = Path(temp_dir) / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create bundle
        # ... bundle creation ...

        # Run container
        container = docker_client.containers.run(
            "oaax-intel-toolchain",
            command="/input/model.zip /output",
            volumes={
                str(input_dir): {'bind': '/input', 'mode': 'ro'},
                str(output_dir): {'bind': '/output', 'mode': 'rw'}
            },
            remove=True,
            detach=False
        )

        # Verify output
        output_files = list(output_dir.glob("*.zip"))
        assert len(output_files) > 0

# Run Docker tests
# pytest tests/test_docker.py -v
```

### Pattern: C++ Tests

**Location:** `runtime-library/tests/simple_test.cpp`

```cpp
#include <iostream>
#include <cassert>
#include "../include/runtime_core.hpp"

int test_initialization() {
    std::cout << "[Test] Runtime initialization..." << std::endl;

    int result = runtime_initialization();
    assert(result == 0 && "Initialization should succeed");

    std::cout << "✓ Passed" << std::endl;
    return 0;
}

int test_version() {
    std::cout << "[Test] Runtime version..." << std::endl;

    const char* version = runtime_version();
    assert(version != nullptr && "Version should not be null");
    assert(strlen(version) > 0 && "Version should not be empty");

    std::cout << "✓ Version: " << version << std::endl;
    return 0;
}

int main() {
    int failed = 0;

    failed += test_initialization();
    failed += test_version();
    // ... more tests ...

    if (failed == 0) {
        std::cout << "All tests passed!" << std::endl;
    } else {
        std::cout << failed << " tests failed!" << std::endl;
    }

    return failed;
}
```

---

## Docker Patterns

### Pattern: Multi-Stage Dockerfile

**Location:** `conversion-toolchain/Dockerfile`

```dockerfile
# Stage 1: Builder (with build tools)
FROM python:3.10-slim as builder

# Install UV for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    /root/.local/bin/uv --version
ENV PATH="/root/.local/bin:${PATH}"

# Copy and install dependencies
WORKDIR /app
COPY pyproject.toml .
RUN uv pip install --system -e .

# Stage 2: Runtime (minimal)
FROM python:3.10-slim

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY conversion_toolchain /app/conversion_toolchain
WORKDIR /app

# Health check
RUN conversion_toolchain --help || exit 1

# Standard mount points
VOLUME ["/input", "/output"]

# Entrypoint
ENTRYPOINT ["conversion_toolchain"]
CMD ["--help"]
```

**Key Points:**
- Multi-stage for smaller final image
- UV for faster package installation
- Health check ensures working build
- Standard volume mount points
- Direct entrypoint (no shell wrapper)

### Pattern: Docker Build Script

**Location:** `conversion-toolchain/build-docker.sh`

```bash
#!/bin/bash
set -e

IMAGE_NAME="${IMAGE_NAME:-oaax-intel-toolchain}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"

# Build with progress
docker build -t "${IMAGE_NAME}:${IMAGE_TAG}" .

# Verify build
echo "Verifying image..."
docker run --rm "${IMAGE_NAME}:${IMAGE_TAG}" --help

echo "✓ Build successful!"
echo "  Image: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  Size: $(docker images ${IMAGE_NAME} --format '{{.Size}}')"
```

---

## CMake Patterns

### Pattern: OpenVINO Integration

**Location:** `runtime-library/CMakeLists.txt`

```cmake
# Find OpenVINO (archive layout: runtime/ subdirectory)
if(NOT DEFINED OPENVINO_DIR)
    set(OPENVINO_DIR "/opt/intel/openvino/runtime")
endif()

set(OPENVINO_INCLUDE_DIR "${OPENVINO_DIR}/include")

# CRITICAL: use WIN32, not MSVC — MSVC is only set after project() is called.
# Any if(MSVC) check before project() always evaluates to false.
if(WIN32)
    set(OPENVINO_LIB_DIR "${OPENVINO_DIR}/lib/intel64/Release")
    set(OPENVINO_BIN_DIR "${OPENVINO_DIR}/bin/intel64/Release")
else()
    set(OPENVINO_LIB_DIR "${OPENVINO_DIR}/lib/intel64")
endif()

# TBB is bundled at runtime/3rdparty/tbb/
set(OPENVINO_TBB_LIB_DIR "${OPENVINO_DIR}/3rdparty/tbb/lib")

# Verify installation
if(NOT EXISTS "${OPENVINO_INCLUDE_DIR}/openvino/openvino.hpp")
    message(FATAL_ERROR "OpenVINO not found at ${OPENVINO_DIR}")
endif()

target_include_directories(RuntimeLibrary PUBLIC ${OPENVINO_INCLUDE_DIR})
target_link_directories(RuntimeLibrary PUBLIC ${OPENVINO_LIB_DIR} ${OPENVINO_TBB_LIB_DIR})
target_link_libraries(RuntimeLibrary PUBLIC openvino spdlog::spdlog pthread dl c_utilities stdc++)
```

### Pattern: Copying OpenVINO libs post-build (Linux)

Use `find -exec {} +` (not `";"`) with VERBATIM to avoid shell glob expansion and the `missing argument to -exec` error.

```cmake
add_custom_command(TARGET RuntimeLibrary POST_BUILD
  COMMAND find ${OPENVINO_LIB_DIR} -maxdepth 1 -name "*.so*"
    "!" -name "libopenvino_onnx_frontend*"
    "!" -name "libopenvino_pytorch_frontend*"
    "!" -name "libopenvino_tensorflow_frontend*"
    "!" -name "libopenvino_tensorflow_lite_frontend*"
    "!" -name "libopenvino_paddle_frontend*"
    "!" -name "libopenvino_jax_frontend*"
    -exec cp -P --target-directory=${CMAKE_CURRENT_BINARY_DIR} {} +
  COMMAND find ${OPENVINO_TBB_LIB_DIR} -maxdepth 1 -name "*.so*"
    -exec cp -P --target-directory=${CMAKE_CURRENT_BINARY_DIR} {} +
  VERBATIM
)
```

**Pitfalls:**
- Without `VERBATIM`, CMake shell-expands `*.so*` in the generated Makefile — find matches nothing
- `";"` in a CMake COMMAND list is treated as a CMake list separator and silently dropped — use `+` terminator
- With `+` terminator, `{}` must be the second-to-last token before `+` (i.e. `--target-directory=DEST {} +`)

### Pattern: Copying OpenVINO DLLs post-build (Windows)

Windows DLLs may live in `bin/intel64/Release/` (OpenVINO 2026.x archive). Use a `cmake -P` script so globbing happens at **build time** (not configure time — the archive may not be extracted yet during `cmake ..`).

**`cmake/copy_windows_dlls.cmake`:**
```cmake
# Probe both candidate dirs in case archive layout varies across OV versions
foreach(_candidate "${SRC_BIN_DIR}" "${SRC_LIB_DIR}")
    file(GLOB _found "${_candidate}/*.dll")
    if(_found)
        set(_openvino_dlls ${_found})
        message(STATUS "copy_windows_dlls: found OpenVINO DLLs in ${_candidate}")
        break()
    endif()
endforeach()

foreach(dll ${_openvino_dlls})
    get_filename_component(name "${dll}" NAME)
    if(NOT name MATCHES "openvino_onnx_frontend|openvino_pytorch_frontend|...")
        file(COPY "${dll}" DESTINATION "${DST_DIR}")
    endif()
endforeach()

file(GLOB tbb_dlls "${SRC_TBB_DIR}/*.dll")
foreach(dll ${tbb_dlls})
    get_filename_component(name "${dll}" NAME)
    if(NOT name MATCHES "_debug\\.dll$")
        file(COPY "${dll}" DESTINATION "${DST_DIR}")
    endif()
endforeach()
```

**`CMakeLists.txt` invocation (inside `if(MSVC)` post-link block):**
```cmake
set(OPENVINO_TBB_BIN_DIR "${OPENVINO_DIR}/3rdparty/tbb/bin")
add_custom_command(TARGET RuntimeLibrary POST_BUILD
  COMMAND ${CMAKE_COMMAND}
    "-DSRC_BIN_DIR=${OPENVINO_BIN_DIR}"
    "-DSRC_LIB_DIR=${OPENVINO_LIB_DIR}"
    "-DSRC_TBB_DIR=${OPENVINO_TBB_BIN_DIR}"
    "-DDST_DIR=$<TARGET_FILE_DIR:RuntimeLibrary>"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/copy_windows_dlls.cmake"
  VERBATIM
)
```

**Why `cmake -P` instead of `file(GLOB)` + `cmake -E copy_if_different`:**
- `file(GLOB)` at configure time returns empty if the OpenVINO archive isn't extracted yet
- `cmake -P` scripts run at build time when files exist
- `cmd.exe`/MSBuild parse `|` and `{}` in PowerShell strings — avoid any shell command containing these characters in a post-build step

### Pattern: RPATH for bundled Linux .so files

Cross-linker bakes in build-time paths (e.g. `/opt/intel/openvino/...`) as DT_RPATH. Fix with `patchelf` on **all** `.so` files — not just `libRuntimeLibrary.so` — so transitive dependencies (e.g. `libopenvino.so → libtbb.so.12`) also find their neighbours at runtime.

```bash
rpath_origin='$ORIGIN'  # shellcheck disable=SC2016
find . -maxdepth 1 -name "*.so*" ! -type l | while read -r lib; do
    patchelf --set-rpath "$rpath_origin" "$lib" 2>/dev/null && echo "  RPATH \$ORIGIN: $lib"
done
```

`patchelf --set-rpath` converts DT_RPATH → DT_RUNPATH. DT_RUNPATH is NOT inherited by transitive deps; patching all libs avoids this.

### Pattern: Cross-Compilation Setup

```cmake
if(PLATFORM STREQUAL "X86_64")
    set(CMAKE_SYSTEM_PROCESSOR x86_64)
    set(CMAKE_SYSTEM_NAME Linux)
    set(CROSS_ROOT "/opt/x86_64-unknown-linux-gnu-gcc-9.5.0")
    set(COMPILER_PREFIX "x86_64-unknown-linux-gnu-")

    # Set toolchain
    set(CMAKE_C_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
    set(CMAKE_CXX_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++)

    # Optimization flags
    set(CMAKE_BUILD_FLAGS "-fno-math-errno -fopenmp -march=haswell")
endif()
```

---

## Type Mapping Patterns

### Pattern: OAAX ↔ OpenVINO Type Mapping

**Location:** `runtime-library/src/runtime_utils.cpp`

```cpp
#include <openvino/openvino.hpp>
#include "tensors_struct.h"

// OAAX → OpenVINO
ov::element::Type map_to_ov_type(tensor_data_type t) {
    switch (t) {
        case DATA_TYPE_FLOAT:   return ov::element::f32;
        case DATA_TYPE_UINT8:   return ov::element::u8;
        case DATA_TYPE_INT8:    return ov::element::i8;
        case DATA_TYPE_UINT16:  return ov::element::u16;
        case DATA_TYPE_INT16:   return ov::element::i16;
        case DATA_TYPE_INT32:   return ov::element::i32;
        case DATA_TYPE_INT64:   return ov::element::i64;
        case DATA_TYPE_BOOL:    return ov::element::boolean;
        case DATA_TYPE_DOUBLE:  return ov::element::f64;
        case DATA_TYPE_UINT32:  return ov::element::u32;
        case DATA_TYPE_UINT64:  return ov::element::u64;
        default:
            throw std::runtime_error("Unsupported data type!");
    }
}

// OpenVINO → OAAX
tensor_data_type map_to_tensors_struct_type(ov::element::Type type) {
    if (type == ov::element::f32)      return DATA_TYPE_FLOAT;
    if (type == ov::element::u8)       return DATA_TYPE_UINT8;
    if (type == ov::element::i8)       return DATA_TYPE_INT8;
    if (type == ov::element::u16)      return DATA_TYPE_UINT16;
    if (type == ov::element::i16)      return DATA_TYPE_INT16;
    if (type == ov::element::i32)      return DATA_TYPE_INT32;
    if (type == ov::element::i64)      return DATA_TYPE_INT64;
    if (type == ov::element::boolean)  return DATA_TYPE_BOOL;
    if (type == ov::element::f64)      return DATA_TYPE_DOUBLE;
    if (type == ov::element::u32)      return DATA_TYPE_UINT32;
    if (type == ov::element::u64)      return DATA_TYPE_UINT64;

    throw std::runtime_error("Unsupported OpenVINO element type!");
}

// Get byte size
int get_data_type_byte_size(tensor_data_type type) {
    switch (type) {
        case DATA_TYPE_FLOAT:   return 4;
        case DATA_TYPE_UINT8:   return 1;
        case DATA_TYPE_INT8:    return 1;
        case DATA_TYPE_UINT16:  return 2;
        case DATA_TYPE_INT16:   return 2;
        case DATA_TYPE_INT32:   return 4;
        case DATA_TYPE_INT64:   return 8;
        case DATA_TYPE_BOOL:    return 1;
        case DATA_TYPE_DOUBLE:  return 8;
        case DATA_TYPE_UINT32:  return 4;
        case DATA_TYPE_UINT64:  return 8;
        default: return 0;
    }
}
```

**Type Mapping Table:**
| OAAX Type | OpenVINO Type | Size (bytes) |
|-----------|---------------|--------------|
| DATA_TYPE_FLOAT | ov::element::f32 | 4 |
| DATA_TYPE_DOUBLE | ov::element::f64 | 8 |
| DATA_TYPE_INT8 | ov::element::i8 | 1 |
| DATA_TYPE_UINT8 | ov::element::u8 | 1 |
| DATA_TYPE_INT16 | ov::element::i16 | 2 |
| DATA_TYPE_UINT16 | ov::element::u16 | 2 |
| DATA_TYPE_INT32 | ov::element::i32 | 4 |
| DATA_TYPE_UINT32 | ov::element::u32 | 4 |
| DATA_TYPE_INT64 | ov::element::i64 | 8 |
| DATA_TYPE_UINT64 | ov::element::u64 | 8 |
| DATA_TYPE_BOOL | ov::element::boolean | 1 |

---

## Memory Management Patterns

### Pattern: OAAX Tensor Structure

**Location:** `runtime-library/include/tensors_struct.h`

```c
typedef struct {
    size_t num_tensors;
    char **names;
    size_t *ranks;
    size_t **shapes;
    tensor_data_type *data_types;
    void **data;
} tensors_struct;

// Allocate tensor structure
tensors_struct* allocate_tensors_struct(size_t num_tensors) {
    tensors_struct* ts = (tensors_struct*)malloc(sizeof(tensors_struct));
    if (!ts) return NULL;

    ts->num_tensors = num_tensors;
    ts->names = (char**)calloc(num_tensors, sizeof(char*));
    ts->ranks = (size_t*)calloc(num_tensors, sizeof(size_t));
    ts->shapes = (size_t**)calloc(num_tensors, sizeof(size_t*));
    ts->data_types = (tensor_data_type*)calloc(num_tensors, sizeof(tensor_data_type));
    ts->data = (void**)calloc(num_tensors, sizeof(void*));

    return ts;
}

// Free tensor structure
void deep_free_tensors_struct(tensors_struct* ts) {
    if (!ts) return;

    for (size_t i = 0; i < ts->num_tensors; i++) {
        if (ts->names && ts->names[i]) free(ts->names[i]);
        if (ts->shapes && ts->shapes[i]) free(ts->shapes[i]);
        if (ts->data && ts->data[i]) free(ts->data[i]);
    }

    if (ts->names) free(ts->names);
    if (ts->ranks) free(ts->ranks);
    if (ts->shapes) free(ts->shapes);
    if (ts->data_types) free(ts->data_types);
    if (ts->data) free(ts->data);

    free(ts);
}
```

### Pattern: Smart Pointers in C++

**Location:** `runtime-library/src/runtime_core.cpp`

```cpp
#include <memory>

// Use shared_ptr for shared ownership
static std::shared_ptr<ov::Core> core;
static std::shared_ptr<ov::CompiledModel> compiled_model;
static std::shared_ptr<ov::InferRequest> infer_request;

void initialize() {
    // Automatic memory management
    core = std::make_shared<ov::Core>();
}

void cleanup() {
    // Reset releases memory
    infer_request.reset();
    compiled_model.reset();
    core.reset();
}

// Use unique_ptr for exclusive ownership
void process_temporary_data() {
    auto temp_data = std::make_unique<std::vector<float>>(1000);
    // ... use temp_data ...
    // Automatically freed when out of scope
}
```

### Pattern: Zero-Copy Tensor Wrapping

```cpp
// Wrap existing memory without copying
ov::Tensor create_tensor_wrapper(void* data, ov::Shape shape, ov::element::Type type) {
    // This wraps the data pointer without copying
    ov::Tensor tensor(type, shape, data);
    return tensor;
}

// Usage
float* my_data = /* ... existing data ... */;
ov::Shape shape = {1, 3, 224, 224};
ov::Tensor tensor = create_tensor_wrapper(my_data, shape, ov::element::f32);

// tensor now points to my_data, no copy occurred
// WARNING: my_data must remain valid while tensor is used!
```

---

## Common Recipes

### Recipe: Add New Configuration Option

1. **Update config schema** (if using JSON config):
```python
# conversion_toolchain/utils.py
DEFAULT_CONFIG = {
    "optimization": {
        "fp16_compression": True,
        "new_option": "default_value"  # Add here
    }
}
```

2. **Read the option**:
```python
def apply_optimization(model, config):
    new_option = config.get("optimization", {}).get("new_option", "default")
    if new_option == "some_value":
        # Apply optimization
        pass
```

3. **Document it**:
```markdown
# README.md
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| new_option | string | default_value | What it does |
```

### Recipe: Add New Model Format Support

1. **Add detection**:
```python
def detect_model_format(model_path: Path) -> str:
    if model_path.suffix == '.onnx':
        return 'onnx'
    elif model_path.suffix == '.xml':
        return 'openvino_ir'
    elif model_path.suffix == '.pb':
        return 'tensorflow'  # New format
    else:
        raise ValueError(f"Unknown model format: {model_path.suffix}")
```

2. **Add conversion logic**:
```python
def convert_model(model_path: str, format: str):
    if format == 'onnx':
        return ov.convert_model(model_path)
    elif format == 'tensorflow':
        return ov.convert_model(model_path, input_model_is_text=False)
```

3. **Add tests**:
```python
def test_tensorflow_conversion(sample_tf_model, temp_dir):
    result = convert_model(sample_tf_model, 'tensorflow')
    assert result is not None
```

### Recipe: Add Device-Specific Optimization

```cpp
// runtime_core.cpp — pass performance hint at compile_model() time.
// DO NOT set ov::inference_num_threads alongside a perf_hint — they conflict.
ov::AnyMap config;
if (perf_hint == "throughput")
    config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
else if (perf_hint == "cumulative_throughput")
    config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT;
else
    config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;

compiled_model = std::make_shared<ov::CompiledModel>(
    core->compile_model(model, device_type, config));

// Query OpenVINO for the optimal number of InferRequests given the hint:
int actual_requests = compiled_model->get_property(ov::optimal_number_of_infer_requests);
```

---

## Quick Reference

### Essential OpenVINO Functions

**Python:**
- `ov.convert_model(path)` - Convert model to IR
- `ov.save_model(model, path)` - Save IR to disk
- `ov.compile_model(model, device)` - Compile for device
- `nncf.quantize(model, dataset)` - INT8 quantization

**C++:**
- `ov::Core()` — Initialize OpenVINO
- `core->read_model(path)` — Load IR model
- `core->compile_model(model, device, config)` — Compile with perf hint
- `compiled->get_property(ov::optimal_number_of_infer_requests)` — Query optimal N
- `compiled->create_infer_request()` — Create inference slot
- `request->set_tensor(name, ov::Tensor(type, shape, ptr))` — Zero-copy input
- `request->set_output_tensor(i, ov::Tensor(type, shape, ptr))` — Zero-copy output (static shapes only)
- `request->set_callback(fn)` — Register async completion callback (do this once at load, not per inference)
- `request->start_async()` — Fire async inference
- `request->wait()` — Block until complete (use in cleanup, not hot path)

### Common Issues & Solutions

**Issue:** Model conversion fails
```python
# Solution: Check model validity first
import onnx
model = onnx.load("model.onnx")
onnx.checker.check_model(model)
```

**Issue:** Runtime can't find .bin file
```cpp
// Solution: Ensure both files in same directory
// Pass .xml path, not .bin
core->read_model("/path/to/model.xml");  // Correct
// Not: core->read_model("/path/to/model.bin");  // Wrong
```

**Issue:** Type mismatch errors
```cpp
// Solution: Verify tensor types match
auto input_type = model->inputs()[0].get_element_type();
logger->info("Expected type: {}", input_type);
// Then map correctly from OAAX type
```

---

## Secret & Confidentiality Scanning

Run this check before every commit, PR review, and before sharing any file externally. The script is at `scripts/check-secrets.sh`.

### What to scan for

| Category | Examples |
|----------|---------|
| **Credentials** | AWS/S3 keys, passwords, tokens, API keys |
| **Endpoints** | Internal storage URLs, private hostnames |
| **Proprietary data** | Model weights committed by accident (`.pt`, `.onnx`, `.bin` >1 MB) |
| **Private config** | `.env` files, `~/.s3cfg`, private certificates |
| **PII** | Email addresses, names in test data |

### Pattern: How to run the scan

```bash
# Scan the whole working tree (respects .gitignore)
bash scripts/check-secrets.sh

# Scan only staged files (pre-commit use)
bash scripts/check-secrets.sh --staged

# Scan a specific file or directory
bash scripts/check-secrets.sh path/to/file
```

### What the script checks

1. **Hardcoded secrets** — regex patterns for common secret shapes (AWS keys, generic tokens, passwords in assignments)
2. **Sensitive filenames** — `.env`, `*.pem`, `*.key`, `credentials*`, `secrets*`
3. **Large binary files** — model weights, `.bin` files over 1 MB that should never be committed
4. **S3/storage endpoints** — literal hostnames from `~/.s3cfg` or CI secrets that must stay in GitHub Secrets only
5. **Internal URLs** — `nbg1.your-objectstorage.com` and similar private storage endpoints hardcoded in source

### Project-specific confidential items

The following must NEVER appear in source files or logs:

- `S3_ACCESS_KEY` / `S3_SECRET_KEY` values — must only be in GitHub repository secrets
- `S3_ENDPOINT_URL` value (`nbg1.your-objectstorage.com`) — acceptable as a hostname pattern but not combined with credentials
- Bucket name `oaax` — public knowledge, but don't log alongside credentials
- Any NNCF calibration dataset contents — may contain proprietary images
- Model weights (`.pt`, `.onnx` files) — large, may be proprietary; confirmed excluded in `.gitignore`

### False positive handling

Add inline suppression with a comment if a match is a known false positive:

```python
# nosec: test fixture, not a real credential
DUMMY_KEY = "AKIAIOSFODNN7EXAMPLE"
```

```bash
# check-secrets-ignore: internal test URL, not a real endpoint
TEST_URL="https://fake.nbg1.your-objectstorage.com"
```

### When to escalate

If the scan finds a real secret that was already committed:
1. Do NOT just delete it in a new commit — the secret is in git history
2. Rotate the credential immediately (invalidate the old one)
3. Use `git filter-repo` or BFG to purge the secret from history
4. Force-push only after the credential is rotated

This guide provides concrete patterns for common tasks. Refer to CLAUDE.md for project context and workflow guidance.
