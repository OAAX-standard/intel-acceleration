# Memory Index

- [Use uv sync instead of uv pip](feedback_uv_sync.md) — Prefer `uv sync --extra <name>` over `uv pip install` for dependency management in this project
- [intel-acceleration project state](project_intel_acceleration.md) — Architecture, CI artifact packaging, two-stage test workflow, benchmark results
- [OpenVINO/NNCF compatibility notes](technical_openvino_compatibility.md) — GCC ABI, CMake WIN32 vs MSVC pitfall, Windows DLL copy, RPATH, archive layout
- [Runtime optimization findings](technical_runtime_optimizations.md) — perf_hint design, buffer pool, memcpy floor, DATA_TYPE_FLOAT16, sem_t dispatch
- [Windows runtime loading fixes (AIMP-1457)](project_windows_runtime_fixes.md) — Root causes of silent Windows failure, fixes in nx plugin + runtime logger, diagnostic tool, handoff for other vendors
