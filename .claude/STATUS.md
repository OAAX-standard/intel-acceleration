# Intel Acceleration Project Status

**Project:** OAAX Implementation for Intel Hardware
**Version:** 1.1.1
**Last Updated:** 2026-03-19

---

## Project Mission

Provide a production-ready implementation of the OAAX standard for Intel hardware (CPU, GPU, NPU) using OpenVINO toolkit, consisting of:
1. **Conversion Toolchain** - Docker container converting ONNX models to OpenVINO IR format
2. **Runtime Library** - C++ shared library for efficient AI inference on Intel hardware

---

## Current Status

**Last Updated:** 2026-04-11 (runtime hot-path optimizations complete)

### ✅ Completed Phases

#### Phase 1: Conversion Toolchain (COMPLETED)
**Status:** Production-ready, fully tested
**Completion Date:** March 8, 2026

**Achievements:**
- ✅ Migrated from ONNX Simplifier to OpenVINO native converter
- ✅ Converts ONNX models to OpenVINO IR format (.xml + .bin)
- ✅ FP16 compression (50% size reduction)
- ✅ INT8 quantization with NNCF (75% size reduction)
- ✅ Docker-first deployment with multi-stage builds
- ✅ Comprehensive error handling with exit codes (0, 1-4, 255)
- ✅ 45+ unit tests + Docker integration tests
- ✅ JSON structured logging
- ✅ Bundle-based I/O (zip input/output)

**Deliverables:**
- Docker image: `oaax-intel-toolchain:latest`
- Test coverage: 98%+ (45 tests passing)
- Documentation: Complete README, test plans

**Details:** See [PHASE1_COMPLETION_REPORT.md](PHASE1_COMPLETION_REPORT.md)

#### Phase 2: Runtime Library Migration (COMPLETED)
**Status:** Production-ready, OpenVINO native API
**Completion Date:** March 19, 2026

**Achievements:**
- ✅ Migrated from ONNX Runtime to OpenVINO native C++ API
- ✅ 92% reduction in binary size (250MB → 20MB)
- ✅ 75% faster initialization, 50% faster model loading
- ✅ 100% API compatibility maintained
- ✅ Direct OpenVINO IR loading (.xml + .bin)
- ✅ Comprehensive type mapping (OAAX ↔ OpenVINO)
- ✅ Thread-safe inference queue
- ✅ Device configuration (CPU, GPU, NPU)
- ✅ Smart pointer memory management

**Deliverables:**
- Shared library: `libRuntimeLibrary.so`
- Reduced dependencies: 150+ libs → 2 libs (openvino + tbb)
- Basic test suite: `simple_test.cpp`
- Documentation: Complete README, API reference

**Details:** See [PHASE2_COMPLETION_REPORT.md](PHASE2_COMPLETION_REPORT.md)

---

## Architecture Overview

### System Flow

```
┌─────────────┐
│ ONNX Model  │
│  (.onnx)    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│   Conversion Toolchain (Docker) │
│   - OpenVINO convert_model()    │
│   - FP16/INT8 optimization      │
│   - Bundle I/O                  │
└──────────┬──────────────────────┘
           │
           ▼
    ┌──────────────┐
    │  OpenVINO IR │
    │ .xml + .bin  │
    └──────┬───────┘
           │
           ▼
┌─────────────────────────────────┐
│    Runtime Library (C++)        │
│   - ov::Core                    │
│   - ov::CompiledModel           │
│   - ov::InferRequest            │
└──────────┬──────────────────────┘
           │
           ▼
    ┌──────────────┐
    │   Inference  │
    │ (CPU/GPU/NPU)│
    └──────────────┘
```

### Technology Stack

| Layer | Technology |
|-------|------------|
| Model Format | ONNX → OpenVINO IR |
| Conversion | OpenVINO Python API + NNCF |
| Runtime | OpenVINO C++ Native API |
| Deployment | Docker (toolchain), Shared Library (runtime) |
| Testing | pytest (Python), Custom runner (C++) |
| Build | UV (Python), CMake (C++) |

---

## Next Steps (Phase 3+)

### Phase 3: Production Hardening (IN PROGRESS)
**Target:** Q2 2026
**Priority:** High
**Status:** Partially complete

**Completed in Phase 3:**
- ✅ Windows and Linux CI builds fully working with OpenVINO 2026.1.0 archive
- ✅ Linux artifact: libRuntimeLibrary.so + all OpenVINO .so + TBB, all patched to $ORIGIN RPATH
- ✅ Windows artifact: RuntimeLibrary.dll + openvino.dll + device plugins + TBB DLLs (~60 MB)
- ✅ Upgraded to OpenVINO 2026.1.0 + NNCF 2.19.0
- ✅ Two-stage test framework (tests/stage1.py + tests/stage2.py)
- ✅ GPU/NPU validation on real hardware (Intel UHD 770 + NVIDIA RTX A4000)
- ✅ Benchmark suite: benchmark_app with FP32/FP16/INT8 × all devices
- ✅ CSV output for cross-run benchmark comparison
- ✅ yolo_test rewritten as multi-run benchmark (warmup + N runs, avg/min/max/p95)
- ✅ GCC dual-ABI fix (-D_GLIBCXX_USE_CXX11_ABI=0) for C++ runtime
- ✅ NNCF/OpenVINO compatibility shim for openvino.Node import path changes
- ✅ Async queue-based inference with single manager thread + N InferRequest workers
- ✅ Pre-allocated output buffer pool (`actual_requests × 4` tensors_struct); zero malloc on hot path
- ✅ Zero-copy output via `set_output_tensor` — OpenVINO writes directly into pool buffers
- ✅ POSIX `sem_t` semaphores for slot and input signalling (replaces polling/mutex+CV)
- ✅ Per-slot state struct + single `set_callback()` registration per slot (no std::function heap alloc per inference)
- ✅ Memory leak verification: zero RSS growth over 10,000 INT8 inferences
- ✅ INT8/FP16 quantization accuracy test (`test_quantization_accuracy.py`; FP16 cosine ≈1.000, INT8 cosine ≈0.9998)

**Goals:**
1. **Comprehensive Testing**
   - ✅ End-to-end integration tests (two-stage framework)
   - ✅ Performance benchmarking suite (benchmark_app + yolo_test + CSV)
   - ✅ GPU/NPU device testing (Intel UHD 770, NVIDIA RTX A4000)
   - ✅ Memory leak verification (RSS monitoring over 10k inferences)
   - ✅ Quantization accuracy tests (tensor-level cosine similarity vs FP32)
   - ☐ Automated long-running stability test (24h)

2. **Performance Optimization**
   - ✅ Async inference queue with optimal InferRequest count (perf_hint)
   - ✅ Pre-allocated output buffer pool (zero malloc on hot path)
   - ✅ Zero-copy output (`set_output_tensor`)
   - ✅ Semaphore-driven dispatch (no polling, no sleep)
   - ✅ Single callback registration per slot (no per-inference std::function alloc)
   - ☐ Model caching across `runtime_model_loading` calls

4. **Packaging**
   - ✅ Linux artifact complete (libRuntimeLibrary.so + OpenVINO .so + TBB, $ORIGIN RPATH)
   - ✅ Windows artifact complete (RuntimeLibrary.dll + OpenVINO .dll + TBB)

3. **Enhanced Error Handling**
   - Detailed error messages with recovery suggestions
   - Graceful degradation (fallback to CPU)
   - Better validation of model compatibility

4. **Developer Experience**
   - Setup automation scripts
   - Troubleshooting guide
   - Example applications

**Success Criteria:**
- [x] Performance benchmarks documented
- [x] Zero memory leaks (verified over 10k inferences)
- [x] GPU/NPU validation on real hardware
- [ ] Automated 24h stability test

### Phase 4: Advanced Features (PLANNED)
**Target:** Q3 2026
**Priority:** Medium
**Status:** Not started

**Goals:**
1. **Multi-Device Support**
   - Heterogeneous execution (CPU+GPU)
   - Automatic device selection
   - Load balancing across devices
   - Device-specific tuning

2. **Extended Format Support**
   - TensorFlow model conversion
   - PyTorch model conversion
   - Direct IR input (skip ONNX)

3. **Advanced Optimizations**
   - Model pruning integration
   - Automatic mixed precision
   - Custom operation support
   - Profile-guided optimization

4. **Platform Support**
   - Windows build validation
   - ARM64 support
   - Docker containers for runtime
   - Pre-built binary releases

**Success Criteria:**
- [ ] Multi-device inference working
- [ ] TensorFlow/PyTorch conversion supported
- [ ] Windows builds automated
- [ ] ARM64 cross-compilation working

### Phase 5: Enterprise Features (FUTURE)
**Target:** Q4 2026
**Priority:** Low
**Status:** Not started

**Goals:**
1. **Monitoring & Telemetry**
   - Performance metrics collection
   - Model profiling tools
   - Resource usage tracking
   - Inference latency analysis

2. **Security Hardening**
   - Model encryption support
   - Secure model loading
   - Input validation hardening
   - Sandboxed execution

3. **Scalability**
   - Model server deployment
   - Load balancing
   - Model versioning
   - A/B testing support

4. **CI/CD Integration**
   - Automated benchmarking
   - Regression detection
   - Release automation
   - Documentation generation

---

## Known Limitations

### Conversion Toolchain
- ❌ No TensorFlow/PyTorch direct support (requires ONNX export first)
- ⚠️ INT8 quantization requires calibration data
- ⚠️ Large models (>2GB) require significant memory
- ⚠️ Some ONNX operators may not be supported by OpenVINO

### Runtime Library
- ❌ No ARM64 support yet
- ⚠️ GPU/NPU features implemented but not extensively tested
- ⚠️ Dynamic shape support limited
- ⚠️ Batch inference not optimized

### Testing
- ⚠️ GPU/NPU tests require specific hardware
- ⚠️ Performance benchmarks need diverse model set
- ⚠️ Long-running stability tests not automated

---

## Performance Benchmarks (Phase 3, Intel Core i7-13700K)

Measured with `benchmark_app -hint throughput` and `yolo_test --perf-hint throughput` (300 runs, 5 warmup).

| Tool | Model | Precision | Device | Throughput |
|------|-------|-----------|--------|------------|
| benchmark_app | yolo11n | FP32 | CPU | ~95.6 FPS |
| benchmark_app | yolo11n | FP16 | CPU | ~98.6 FPS |
| benchmark_app | yolo11n | INT8 | CPU | **~244 FPS** |
| yolo_test (OAAX) | yolo11n | FP32 | CPU | ~97.7 FPS |
| yolo_test (OAAX) | yolo11n | FP16 | CPU | ~98.1 FPS |
| yolo_test (OAAX) | yolo11n | INT8 | CPU | ~237 FPS |
| benchmark_app | yolov8n | FP32 | CPU | ~84 FPS |
| benchmark_app | yolov8n | INT8 | CPU | ~242 FPS |
| yolo_test (OAAX) | yolov8n | FP32 | CPU | ~84.8 FPS |
| yolo_test (OAAX) | yolov8n | INT8 | CPU | ~235 FPS |

OAAX runtime matches `benchmark_app` within ~2% for FP32/FP16 and ~3% for INT8. The remaining INT8 gap is manager dispatch overhead at 235+ FPS.

---

## Performance Metrics

### Conversion Toolchain (Phase 1)
| Metric | Value |
|--------|-------|
| Docker image size | ~1.2 GB |
| Build time | ~3 min |
| Average conversion time | 5-30s (model dependent) |
| FP16 size reduction | ~50% |
| INT8 size reduction | ~75% |

### Runtime Library (Phase 2)
| Metric | Before (ONNX RT) | After (Native) | Improvement |
|--------|------------------|----------------|-------------|
| Binary size | 250 MB | 20 MB | **92%** |
| Initialization time | ~800ms | ~200ms | **75%** |
| Model loading | ~1.2s | ~600ms | **50%** |
| Inference (first) | ~150ms | ~100ms | **33%** |
| Memory footprint | 180 MB | 80 MB | **56%** |

---

## Risk Assessment

### Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| OpenVINO API changes | Medium | High | Pin version, test upgrades |
| Model compatibility issues | Medium | Medium | Comprehensive test suite |
| Performance regressions | Low | High | Automated benchmarking |
| Memory leaks | Low | High | Valgrind, stress testing |

### Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| GPU/NPU hardware unavailable | High | Medium | CPU fallback tested |
| Breaking API changes | Low | High | Semantic versioning, tests |
| Documentation outdated | Medium | Low | Review process |

---

## Success Metrics

### Quality Metrics
- **Test Coverage:** Target 95%+ for critical paths
- **Performance:** Meet or exceed ONNX Runtime baseline
- **Stability:** Zero crashes in 24h stress test
- **Documentation:** All public APIs documented

### User Satisfaction
- **Build Success:** >98% successful builds
- **Conversion Success:** >95% ONNX model compatibility
- **Runtime Stability:** >99.9% inference success rate

---

## Dependencies

### Build-Time Dependencies
| Dependency | Version | Purpose |
|------------|---------|---------|
| Python | 3.10+ | Toolchain implementation |
| OpenVINO | 2024.0+ | AI framework |
| UV | Latest | Python package manager |
| Docker | 20.10+ | Container deployment |
| CMake | 3.10.2+ | C++ build system |
| GCC | 9.5+ | C++ compiler |

### Runtime Dependencies
| Dependency | Version | Size | Purpose |
|------------|---------|------|---------|
| OpenVINO | 2024.6.0 | 16 MB | Core inference |
| TBB | 12.x | 320 KB | Threading |
| CPU Plugin | 2024.6.0 | 55 MB | CPU acceleration |
| GPU Plugin | 2024.6.0 | 28 MB | GPU acceleration (optional) |
| NPU Plugin | 2024.6.0 | 3 MB | NPU acceleration (optional) |

---

## Decision Log

### Major Technical Decisions

**2026-03-08: Adopt OpenVINO Native Converter (Phase 1)**
- **Decision:** Replace ONNX Simplifier with OpenVINO's native converter
- **Rationale:** Better optimization, native IR support, maintained by Intel
- **Impact:** Cleaner pipeline, better performance
- **Status:** Implemented, validated

**2026-03-19: Migrate to OpenVINO Native C++ API (Phase 2)**
- **Decision:** Remove ONNX Runtime, use OpenVINO API directly
- **Rationale:** 92% size reduction, 50-75% performance improvement
- **Impact:** Major architecture change, complete rewrite
- **Trade-offs:** Lose ONNX model support (must use IR)
- **Status:** Implemented, validated

**2026-03-19: Docker-First Deployment (Toolchain)**
- **Decision:** Prioritize Docker over local installation
- **Rationale:** Dependency isolation, reproducibility, OAAX spec compliance
- **Impact:** All examples use Docker, local dev secondary
- **Status:** Implemented

**2026-03-19: Exit Code Standardization**
- **Decision:** Use specific exit codes (0, 1-4, 255) for automation
- **Rationale:** Enable automated pipelines, clear error categories
- **Impact:** All error handling preserves exit codes
- **Status:** Implemented, enforced by tests

**2026-04-11: Single Manager Thread + Async InferRequests**
- **Decision:** Replace N inference threads (one per InferRequest) with one manager thread that dispatches async inference via `start_async()` + `set_callback()`
- **Rationale:** N blocking threads compete on OS scheduler; one manager + OpenVINO's own threads eliminates thread-per-slot overhead and let's OpenVINO manage parallelism
- **Impact:** Cleaner architecture, fewer threads, matches how OpenVINO is designed to be used
- **Status:** Implemented, benchmarked

**2026-04-11: Zero-Copy Output via `set_output_tensor`**
- **Decision:** Call `req.set_output_tensor(i, ov::Tensor(type, shape, pool_buf->data[i]))` before each `start_async()` so OpenVINO writes inference results directly into pre-allocated pool buffers
- **Rationale:** Previous approach memcpy'd 2.8MB of output per FP32 inference inside the callback, capping throughput. Zero-copy eliminates that bottleneck at INT8 speeds
- **Impact:** FP32/FP16 now within ~2% of benchmark_app (was ~5% gap)
- **Trade-off:** Pool path only works for static output shapes; dynamic shapes still use memcpy fallback
- **Status:** Implemented, validated, benchmarked

**2026-04-11: POSIX sem_t + Per-Slot State**
- **Decision:** Replace custom mutex+CV `Semaphore` class with POSIX `sem_t`; move per-inference state into a `SlotState[]` array instead of lambda captures
- **Rationale:** (1) `sem_t` is futex-backed (1 syscall contended vs 2 for mutex+CV); (2) capturing `idx+input+output+from_pool` = 21 bytes overflows GCC's 16-byte std::function SOO buffer → heap alloc per inference. Capturing only `int i` (4 bytes) fits SOO
- **Impact:** Eliminates per-inference heap allocation in callback; lower dispatch latency at high FPS
- **Constraint:** `set_callback()` is now registered once per slot at model load, not per inference
- **Status:** Implemented, benchmarked

**2026-04-11: Do Not Set ov::inference_num_threads with perf_hint**
- **Decision:** Never call `core->set_property("CPU", ov::inference_num_threads(N))` when also using a performance hint
- **Rationale:** The two settings conflict — `inference_num_threads` constrains OpenVINO's internal scheduler, preventing it from using the optimal thread count implied by the hint
- **Impact:** Removing this call improved throughput; `perf_hint` alone is sufficient
- **Status:** Enforced (removed from codebase, documented as forbidden in CLAUDE.md)

**2026-04-11: Switch from pip-installed OpenVINO to official archive for CI builds**
- **Decision:** Download `openvino_toolkit_ubuntu22_*.tgz` / `openvino_toolkit_windows_*.zip` from storage.openvinotoolkit.org instead of `pip install openvino`
- **Rationale:** pip package puts .so files in `dist-packages/openvino/libs/` with versioned filenames and missing unversioned symlinks needed by the cross-linker; the archive provides the standard `runtime/lib/intel64/` layout and includes bundled TBB
- **Archive layout (Linux):** `runtime/lib/intel64/*.so*`, `runtime/3rdparty/tbb/lib/*.so*`
- **Archive layout (Windows):** `runtime/lib/intel64/Release/*.lib`, `runtime/bin/intel64/Release/*.dll`, `runtime/3rdparty/tbb/bin/*.dll`
- **Status:** Implemented in CI (`build.yml`), version pinned to OpenVINO 2026.1.0

**2026-04-11: CMake if(WIN32) before project(), if(MSVC) after**
- **Decision:** Use `if(WIN32)` for path configuration before `project()`, `if(MSVC)` only for post-`project()` checks
- **Rationale:** `MSVC` is only set after CMake detects the compiler during `project()`. Any `if(MSVC)` block before `project()` silently evaluates false, leaving variables like `OPENVINO_BIN_DIR` unset. `WIN32` is available at CMake startup.
- **Impact:** Fixed Windows DLL bundling — `OPENVINO_BIN_DIR` was empty, causing the cmake -P script to copy nothing
- **Status:** Fixed in CMakeLists.txt (line 30)

**2026-04-11: Use cmake -P script for Windows DLL copy, not file(GLOB) at configure time**
- **Decision:** Copy OpenVINO DLLs via `cmake -P copy_windows_dlls.cmake` invoked from `add_custom_command POST_BUILD`
- **Rationale:** `file(GLOB)` at configure time returns empty if the OpenVINO archive hasn't been extracted yet. `cmake -P` runs at build time when all files are present. Also avoids cmd.exe/MSBuild parsing issues with `|` and `{}` in PowerShell commands.
- **Status:** Implemented in `runtime-library/cmake/copy_windows_dlls.cmake`

---

## Resources

### Documentation
- [OAAX Standard](https://github.com/OAAX-standard/OAAX)
- [OpenVINO Documentation](https://docs.openvino.ai/)
- [Project README](README.md)
- [CLAUDE.md](CLAUDE.md) - Agent guide
- [SKILLS.md](SKILLS.md) - Implementation patterns

### Reports
- [Phase 1 Completion Report](PHASE1_COMPLETION_REPORT.md)
- [Phase 2 Completion Report](PHASE2_COMPLETION_REPORT.md)

### Repository
- **GitHub:** https://github.com/OAAX-standard/intel-acceleration
- **License:** Apache 2.0

---

## Roadmap Timeline

```
2026 Q1: ✅ Phase 1 Complete - Conversion Toolchain
2026 Q1: ✅ Phase 2 Complete - Runtime Migration
2026 Q2: 🔄 Phase 3 Planned - Production Hardening
2026 Q3: 📋 Phase 4 Planned - Advanced Features
2026 Q4: 📋 Phase 5 Planned - Enterprise Features
```

---

## Contact & Support

**Maintainers:** See [CONTRIBUTING.md](CONTRIBUTING.md)
**Issues:** GitHub Issues
**License:** Apache 2.0

---

**Last Review:** 2026-03-19
**Next Review:** When Phase 3 starts or major changes occur
