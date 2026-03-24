# Claude Agent Guide for Intel Acceleration Project

## Project Overview

This is the **OAAX implementation for Intel hardware** (CPU, GPU, and NPU).

- **Project name:** intel-acceleration
- **Repository:** https://github.com/OAAX-standard/intel-acceleration
- **Purpose:** Provides an implementation of the OAAX standard for Intel hardware by leveraging OpenVINO toolkit to implement the conversion toolchain and runtime library
- **License:** Apache 2.0
- **Version:** See `VERSION` file (currently 1.1.1)

---

## OAAX Standard Overview

**OAAX** is a vendor-neutral specification for AI **inference** acceleration that provides a standardized way to accelerate AI models on different hardware platforms.

### Core Components

| Component | Description | Role |
|-----------|-------------|------|
| **OAAX Toolchain** | Docker-based conversion pipeline | Converts ONNX models to hardware-specific formats |
| **OAAX Runtime** | Shared library (.so or .dll) | Executes converted models on target hardware |

### OAAX Toolchain Specification

Docker container that converts ONNX models into hardware-optimized formats.

**Requirements:**
- **Container:** Runs from `/app`, no network/root privileges
- **Entrypoint:** Two arguments: input path and output directory
- **Input:** `.onnx` file OR `.zip` (model + config/calibration data)
- **Output:** Converted model + `logs.json`
- **Error Handling:** Exit 0=success, non-zero=failure (logs.json required)
- **Naming:** `oaax-intel-toolchain:tag`

**Example:**
```bash
docker run --rm -v "$PWD/data:/app/run" \
  oaax-intel-toolchain:latest "/app/run/model.zip" "/app/run/output"
```

### OAAX Runtime Specification

**Pure C API** shared library providing:
- Runtime initialization/destruction
- Model loading
- Inference input/output
- Error and version reporting

**Interface:** [interface.h](https://github.com/OAAX-standard/OAAX/blob/main/Illustrative%20example/Runtime/include/interface.h)

Dynamic loading via `dlmopen`/`dlsym` (POSIX) or `LoadLibrary`/`GetProcAddress` (Windows).

---

## Project Structure

```
intel-acceleration/
├── conversion-toolchain/      # OAAX Toolchain (ONNX → OpenVINO IR)
├── runtime-library/           # OAAX Runtime (OpenVINO inference)
├── scripts/                   # Utility scripts
├── .github/workflows/         # CI/CD pipelines
├── VERSION                    # Version: 1.1.1
├── PLAN.md                    # Roadmap and current status
├── CHANGELOG.md               # Version history
├── CONTRIBUTING.md            # Contribution guidelines
├── CLAUDE.md                  # This file - Agent guide
├── SKILLS.md                  # Implementation patterns
└── README.md                  # User documentation
```

### Conversion Toolchain (`conversion-toolchain/`)
```
├── conversion_toolchain/
│   ├── main.py              # CLI entrypoint
│   ├── utils.py             # OpenVINO conversion logic
│   ├── quantization.py      # INT8 quantization (NNCF)
│   └── logging_utils.py     # JSON logging
├── tests/
│   ├── test_conversion.py   # 45+ unit tests
│   └── test_docker.py       # Docker integration tests
├── Dockerfile               # Production container
├── pyproject.toml           # UV-based package config
├── build-toolchain.sh       # Build script
└── README.md                # User documentation
```

### Runtime Library (`runtime-library/`)
```
├── src/
│   ├── runtime_core.cpp     # OpenVINO native implementation
│   └── runtime_utils.cpp    # Type mappings
├── include/
│   ├── runtime_core.hpp     # Public OAAX API
│   ├── runtime_utils.hpp    # Internal utilities
│   └── tensors_struct.h     # OAAX tensor structure
├── tests/
│   └── simple_test.cpp      # Validation tests
├── CMakeLists.txt           # Build configuration
├── build-runtimes.sh        # Build script
└── README.md                # User documentation
```

---

## Getting Started

**Read in this order:**
1. **CLAUDE.md** (this file) - Project overview
2. **PLAN.md** - Current status and roadmap
3. **SKILLS.md** - Code patterns and examples
4. **CONTRIBUTING.md** - Contribution guidelines
5. **README.md** - User documentation

---

## Common Tasks

### Build Toolchain
```bash
cd conversion-toolchain
bash build-toolchain.sh
```

### Convert Model
```bash
docker run --rm \
  -v $(pwd)/input:/input \
  -v $(pwd)/output:/output \
  oaax-intel-toolchain /input/model.zip /output
```

### Build Runtime
```bash
cd runtime-library
bash build-runtimes.sh
```

### Run Tests
```bash
# Toolchain tests
cd conversion-toolchain && pytest tests/ -v

# Runtime tests
cd runtime-library/build && ./simple_test
```

---

## Making Changes

### Before Starting
1. **Identify component** - Toolchain (Python/Docker) or Runtime (C++/CMake)
2. **Check tests** - Will changes break existing tests?
3. **Review PLAN.md** - Understand priorities and design decisions

### During Development
1. **Use TodoWrite:**
   - Break into specific steps
   - Mark `in_progress` (only ONE at a time)
   - Mark `completed` immediately

2. **Test thoroughly:**
   - Run existing tests after changes
   - Add tests for new functionality
   - Verify Docker builds (toolchain)

3. **Maintain compatibility:**
   - Never break public APIs
   - Preserve exit codes (0, 1-4, 255)
   - Keep log formats consistent

4. **Update docs:**
   - README.md for user-facing changes
   - PLAN.md for architectural updates
   - Code comments

### After Completion
1. **Validate builds:**
   ```bash
   # Toolchain
   docker build -t oaax-intel-toolchain .
   ./quick-test.sh

   # Runtime
   bash build-runtimes.sh
   ```

2. **Run tests:**
   ```bash
   pytest tests/ -v              # Toolchain
   cd build && ./simple_test     # Runtime
   ```

3. **Update project files:**
   - VERSION (releases)
   - CHANGELOG.md (changes)
   - PLAN.md (milestones)

---

## Design Principles

1. **Docker-First (Toolchain)** - Container is primary distribution
2. **API Stability (Runtime)** - Never break public APIs
3. **Performance Matters** - Optimize hot paths
4. **Testing Required** - No untested code
5. **Clear Documentation** - Users understand without reading code

## Code Standards

**Python (Toolchain):**
- Type hints, clear names, docstrings
- pytest, structured exit codes

**C++ (Runtime):**
- Modern C++14/17, smart pointers, RAII
- spdlog logging, clear error messages

**Build Systems:**
- CMake (C++), pyproject.toml (Python)
- Shell scripts with error handling
- Multi-stage Docker builds

---

## Tech Stack

| Component | Tools |
|-----------|-------|
| Package Management | `uv` (Python), `cmake` (C++) |
| Containerization | `docker` |
| Testing | `pytest` (Python), custom (C++) |
| Logging | JSON (Python), `spdlog` (C++) |
| AI Framework | OpenVINO 2026.1+ |

---

## Quick Reference

### Essential Commands
```bash
# Build all
cd conversion-toolchain && bash build-toolchain.sh
cd ../runtime-library && bash build-runtimes.sh

# Test all
cd conversion-toolchain && ./quick-test.sh
cd ../runtime-library/build && ./simple_test

# Check status
cat VERSION && cat PLAN.md
```

### File Quick Access

| Purpose | Path |
|---------|------|
| Toolchain CLI | `conversion-toolchain/conversion_toolchain/main.py` |
| Conversion logic | `conversion-toolchain/conversion_toolchain/utils.py` |
| Runtime core | `runtime-library/src/runtime_core.cpp` |
| Runtime API | `runtime-library/include/runtime_core.hpp` |
| Docker config | `conversion-toolchain/Dockerfile` |
| CMake config | `runtime-library/CMakeLists.txt` |
| Project plan | `PLAN.md` |
| Code patterns | `SKILLS.md` |

---

## Important Notes

### Exit Codes (Toolchain)

| Code | Meaning | When |
|------|---------|------|
| 0 | Success | Completed |
| 1 | File not found | Missing input |
| 2 | Invalid input | Bad zip/model |
| 3 | Conversion failed | OpenVINO error |
| 4 | I/O error | Read/write failed |
| 255 | Unexpected | Unknown error |

### OpenVINO Integration

**Toolchain:**
- `openvino.convert_model()` - ONNX → IR
- `openvino.save_model()` - Save IR (.xml + .bin)
- NNCF for INT8 quantization

**Runtime:**
- `ov::Core` - Initialization
- `ov::Core::read_model()` - Load IR
- `ov::Core::compile_model()` - Compile for device
- `ov::InferRequest::infer()` - Run inference

### Model Flow

```
ONNX model → [Toolchain] → OpenVINO IR (.xml + .bin) → [Runtime] → Inference
```

---

## Success Criteria

✅ Tests pass
✅ Documentation updated
✅ TodoWrite tracks progress
✅ Exit codes preserved
✅ API compatibility maintained
✅ Docker builds successfully
✅ Follows existing patterns (SKILLS.md)
✅ Aligns with PLAN.md

---

## When You Need Help

1. **PLAN.md** - Status, priorities, decisions
2. **SKILLS.md** - Code patterns, examples
3. **Completion reports** - PHASE1/2_COMPLETION_REPORT.md
4. **Tests** - Expected behavior
5. **README files** - User intent

---

**Remember:** Production-quality OAAX implementation. Quality, testing, and docs are critical. Understand before changing. When in doubt, consult PLAN.md and SKILLS.md.
