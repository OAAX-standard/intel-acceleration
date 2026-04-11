# Contributing to OAAX OpenVINO implementation

Thank you for your interest in contributing to OAAX's OpenVINO implementation! We welcome contributions from the community to improve the current implementation of the OAAX runtime and conversion toolchain.

Before you start contributing, please take a moment to read through [OAAX general CONTRIBUTING document](https://github.com/OAAX-standard/OAAX/blob/main/CONTRIBUTING.md). It sets out the guidelines for contributing to OAAX projects, including code style, commit message format, and other important information.

## Table of contents
- [How to contribute](#how-to-contribute)
- [Development environment](#development-environment)
- [Development Setup](#development-setup)  
- [Code Style & Standards](#code-style--standards)  
- [Pre-commit Hooks & Automation](#pre-commit-hooks--automation)  
- [Testing Changes Locally](#testing-changes-locally)  
- [Project Structure & Architecture Overview](#project-structure--architecture-overview)  
- [Development Dependencies](#development-dependencies)  
- [First Contribution Guide](#first-contribution-guide)  
- [Submitting Pull Requests](#submitting-pull-requests)  

## How to contribute

The OAAX OpenVINO implementation is built to be able to make use of Intel's CPU, GPU, and NPU for model inference. If you'd like to suggest improvements to the documentation or implementation, report a bug, or contribute a new feature, please follow these steps:
1. **Open an issue** in the repository to discuss your idea or improvement.
2. **Fork the repository** and create a new branch for your changes.
3. **Make your changes** and commit them with a clear message.
4. **Push your changes** to your forked repository.
5. **Submit a pull request** to the main repository for review.


## Development environment

The OAAX OpenVINO implementation has been tested for building on x86_64 architecture machines running Ubuntu 22.04 or later.

### Runtime

The runtime library is built using CMake and several other dependencies. You can easily install them on a host machine by running this command (assuming you have cloned the repository):

```bash
sudo bash scripts/setup-env.sh
```

### Conversion toolchain

The conversion toolchain is built using Docker and requires Docker to be installed on your machine. You can find instructions for installing Docker [here](https://docs.docker.com/get-docker/).

## Development Setup

Follow these steps to get a local development environment up and running:

### 1. Clone the repository  

```
git clone https://github.com/OAAX-standard/intel-acceleration.git
cd intel-acceleration
```

### 2. Choose your target platform

This repository contains two major parts:
- **conversion-toolchain/:** the source for building the OAAX conversion toolchain
- **runtime-library/:** the source for building the OAAX runtime

Each part has its own README with more specific build instructions.

### 3. Install required tools & versions

Ensure you have the following minimum versions installed:
- Git (>= 2.30)
- CMake (>= 3.15)
- A C++ compiler supporting C++17 (e.g., g++ 9+, clang 10+)
- Docker (for building the toolchain images)
- Python 3.8+ (if you work with Python scripts)
- Any OS prerequisites (Ubuntu 22.04)

### 4. Build locally

For example, to build the runtime library:

```bash
cd runtime-library
bash build-runtimes.sh
```

Artifacts will be placed in `runtime-library/artifacts/X86_64/`.

Similarly for the conversion toolchain (Docker):

```bash
cd conversion-toolchain
IMAGE_NAME=oaax-intel-toolchain bash build-toolchain.sh
```

### 5. IDE / Editor recommendation

- Use Visual Studio Code or CLion.
- Enable “format on save” with the style guide below.
- Configure the include paths to the **runtime-library/include** folder and link to **artifacts**.
- Use the built-in debugger for stepping through runtime code.

### 6. Troubleshooting & common issues

- If Docker build fails, verify you have sufficient permissions and available disk space.
- If CMake cannot locate OpenVINO, set the `OPENVINO_DIR` environment variable to the
  OpenVINO Python package path (e.g., `/usr/local/lib/python3.10/dist-packages/openvino`).
- Clean old build artifacts when switching branches: `git clean -fdx`.

## Code Style & Standards

To keep the codebase clean and maintainable, we follow these conventions:
- Formatting: Use clang-format (version 12+) with the style file at .clang-format.
- Naming:
	- Classes: **PascalCase**
	- Methods / functions: **camelCase**
	- Variables: **snake_case**
	- Constants: **kConstantName**
- File organization:
	- **include/** for public headers
	- **src/** for implementation files
	- **tests/** (if any) for test code
- Commit messages: Follow the Conventional Commits format:

```makefile
feat(module): add support for X
fix(runtime): correct buffer overflow
docs: update development setup
```
- Code comments: Use Doxygen-style comments for public APIs.

## Project Structure & Architecture Overview

### High-level overview:

```structure
intel-acceleration/
├── conversion-toolchain/  # Convert ONNX models into XPU-specific binaries  
├── runtime-library/       # Shared library + APIs to load and run binaries  
├── scripts/               # Utility scripts (model download, benchmarking)  
├── .github/workflows/     # CI pipelines  
├── CHANGELOG.md  
├── VERSION  
└── README.md
```

### Key modules

- **Toolchain:** Takes an ONNX model (or zip bundle) and converts it to OpenVINO IR
  (`.xml` + `.bin`) with optional FP16 compression or INT8 quantization via NNCF.
- **Runtime:** A C++ shared library that loads OpenVINO IR and executes inference on
  Intel CPU, GPU, or NPU using the OpenVINO native C++ API.
- **Artifacts:** Build outputs — Docker image (toolchain) and `libRuntimeLibrary.so`
  with headers (runtime) — for deployment.

### Data flow

1. Developer provides an ONNX model.
2. Conversion toolchain processes it → produces XPU‐optimized model.
3. Runtime loads optimized model and executes inference calls.
4. Application uses its API to send inputs / receive outputs.

This architecture allows plug-ins for different accelerators while keeping the high-level API stable.

## Testing Changes Locally

Tests run in two stages. Run Stage 1 first to compile models; Stage 2 benchmarks them.

```bash
# One-time Python setup
uv venv && source .venv/bin/activate
uv sync --extra integration --extra quantization

# Stage 1: convert models + run Python tests
bash scripts/stage1_compile.sh

# Stage 2: benchmark compiled models with benchmark_app + yolo_test
bash scripts/stage2_run.sh [--devices CPU,GPU.0] [--duration 10] [--csv results.csv]

# Individual pytest suites
pytest tests/test_conversion.py -v
pytest tests/test_yolo_integration.py -v
pytest tests/test_quantization_accuracy.py -v

# C++ runtime smoke tests (after building)
cd runtime-library/build
./simple_test
./yolo_test /path/to/model.xml [device] [--runs 300] [--warmup 5] [--perf-hint throughput]
```

## Development Dependencies

Python dependencies are managed with [uv](https://github.com/astral-sh/uv) via
`pyproject.toml`. To set up the test environment:

```bash
uv venv && source .venv/bin/activate
uv sync --extra integration --extra quantization
```

The optional extras are:
- `integration` — adds `ultralytics` for YOLO integration tests
- `quantization` — adds `nncf` for INT8 accuracy tests

C++ tooling requirements:
- `clang-format` (version 12+) — enforced by the `.clang-format` config at the repo root
- `cmake` (>= 3.15)
- `g++` 9+ or `clang` 10+ with C++17 support

If you add new dependencies, update `pyproject.toml` (Python) or the relevant
`CMakeLists.txt` (C++) and note it in your PR description.

## First Contribution Guide

If this is your first time contributing:
1. Fork the repository (top-right corner of GitHub).
2. Clone your fork locally (```git clone …```).
3. Create a branch (e.g., ```feat-new-accelerator``` or ```fix-cleanup```).
4. Ensure you pull the latest changes from **main** and rebase if needed.
5. Make small, focused changes and verify build/tests succeed.
6. Push your branch and open a pull request.

Focus on delivering one change per PR to simplify review.

## Submitting Pull Requests

When you’re ready to submit:
- Ensure your branch is rebased on the latest **main**.
- Run all pre-commit hooks, run builds/tests locally.
- Fill out the PR template (if provided), and include:
	- A clear title and description.
	- Reference to any issue (if applicable).
	- Screenshots or logs if the change affects output.
- Be responsive to review feedback — our reviewers may ask for adjustments.
