# Contributing to OAAX Intel Acceleration

Thank you for your interest in contributing! Before you start, please read the
[OAAX general contributing guide](https://github.com/OAAX-standard/OAAX/blob/main/CONTRIBUTING.md)
for project-wide conventions (commit message format, code style, etc.).

## How to contribute

1. **Open an issue** to discuss the bug or feature before writing code.
2. **Fork the repository** and create a branch from `main`.
3. **Make your changes**, keeping each PR focused on one thing.
4. **Run tests** locally (see below) to confirm nothing is broken.
5. **Push your branch** and open a pull request.

## Development environment

Tested on x86_64 Ubuntu 22.04+. The setup script installs all build dependencies:

```bash
sudo bash scripts/setup-env.sh
```

Requirements:
- Git 2.30+
- CMake 3.10.2+
- GCC 9.5+ / Clang 10+ with C++17
- Docker 20.10+ (for toolchain builds)
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (Python package manager)

## First-time setup

```bash
# Clone
git clone https://github.com/OAAX-standard/intel-acceleration.git
cd intel-acceleration

# Python environment + test dependencies
uv venv && source .venv/bin/activate
uv sync --extra integration --extra quantization

# Pre-commit hooks (ruff, clang-format, shellcheck, hadolint)
pip install pre-commit
pre-commit install
```

After `pre-commit install`, every `git commit` automatically runs linters and
formatters. To run them manually across the whole repo:

```bash
pre-commit run --all-files
```

## Building

```bash
# Conversion toolchain (Docker image)
cd conversion-toolchain
IMAGE_NAME=oaax-intel-toolchain bash build-toolchain.sh

# Runtime library (Linux x86_64)
cd runtime-library
OPENVINO_DIR=/opt/intel/openvino/runtime bash build-runtimes.sh
```

See each component's README for more options.

## Testing locally

```bash
# Stage 1: compile YOLO models to FP32/FP16/INT8 IR, run conversion tests
python tests/stage1.py

# Stage 2: benchmark with benchmark_app + yolo_test
python tests/stage2.py [--devices CPU,GPU.0] [--duration 10] [--csv results.csv]

# Individual suites
pytest tests/test_conversion.py -v
pytest tests/test_yolo_integration.py -v
pytest tests/test_quantization_accuracy.py -v

# C++ smoke tests
cd runtime-library/build
./simple_test
./yolo_test /path/to/model.xml [device] [--runs 300] [--warmup 5] [--perf-hint throughput]
```

## Code style

| Language | Tool | Config |
|----------|------|--------|
| Python | ruff (lint + format) | `pyproject.toml` |
| C++ | clang-format | `.clang-format` (Google style, 120-char limit) |
| Shell | shellcheck | default |
| Dockerfile | hadolint | default |

Naming conventions for C++:
- Classes: `PascalCase`
- Functions / methods: `snake_case` (matching the public C API style)
- Variables: `snake_case`
- Constants: `kConstantName`

Use Doxygen-style comments for any public API additions.

## Submitting a pull request

- Rebase on the latest `main` before opening the PR.
- Fill out the PR template: include a clear description and link to the related issue.
- All CI checks must pass.
- If your change affects the public C API (`runtime_core.hpp` or `tensors_struct.h`),
  note it explicitly — the API must remain backwards-compatible.
- Update `CHANGELOG.md` under an `[Unreleased]` section.

## Adding dependencies

- **Python:** add to `pyproject.toml` and document the reason in your PR.
- **C++:** add to `CMakeLists.txt`; prefer vendoring small libraries under `deps/`.
