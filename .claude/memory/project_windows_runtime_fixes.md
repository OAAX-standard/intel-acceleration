---
name: Windows runtime loading fixes (AIMP-1457)
description: Root causes and fixes for the Intel runtime silently failing on Windows; handoff written for other vendor runtimes
type: project
---
Investigated 2026-07-16: the Intel runtime "didn't work on Windows with no errors". Package was fine; three real causes:

1. **Host bug (nx plugin):** plain `LoadLibraryA` can't resolve the DLL's co-located dependencies (no RPATH on Windows). Fixed on nx branch `x/AIMP-1457_fix_intel_runtime_on_windows` (origin aassis/nx, base `upstream/vms_6.1_patch`, commit `95c2cc855e6`) using `LoadLibraryExA` + `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR` (needs absolute path) + GetLastError/dlerror logging in `nxai_runtime.cpp`.
2. **Runtime bug:** `initialize_logger` called `exit(1)` when `runtime.log` (CWD-relative; unwritable for Windows services) couldn't be created — killing the host. Fixed in `runtime-library/src/runtime_utils.cpp`: console-only fallback + warning. Verified with unwritable log_file → init SUCCESS.
3. Silent-failure ergonomics: failures now log root cause instead of generic "Runtime not initialized."

Also renamed `runtime_get_name()` → "OAAX Intel" (`runtime_core.cpp`, asserted in `tests/runtime/simple_test.cpp`).

Diagnostic tool: `tests/runtime/windows_load_test.c` (PE bitness check, per-dep probes, plugin-DLL presence, CWD writability, `--init`). Must be built from the **x64** Native Tools prompt.

Handoff for auditing other vendor runtimes: `/home/ayoub/nvidia-acceleration/.claude/HANDOFF_windows_runtime_audit.md`. Same audit applies to [[deepx/hailo/memryx/sima repos|other vendor repos]].

**2026-07-17 — DLL-pinning / unloadability fix** (from the "OAAX CPU Runtime" bug report about FreeLibrary not releasing the DLL; same class confirmed in the Intel runtime):
- spdlog async pool was a `static` in `initialize_logger` — its worker thread (code inside our lib) survived `runtime_cleanup`, pinning the DLL on Windows and risking a loader-lock deadlock at unload. Fixed: pool owned in a global `shared_ptr`, joined via `shutdown_logging()` called at the end of `runtime_cleanup` (after `destroy_logger`). Also added `ov::shutdown()` in cleanup.
- Linux unloadability required three build changes in `runtime-library/CMakeLists.txt`: `-fno-gnu-unique` (GNU-unique symbols make glibc refuse to unload), `-fvisibility=hidden -fvisibility-inlines-hidden`, and a linker version script `cmake/runtime.map` exporting only the 9 OAAX symbols (otherwise NODELETE libs like libstdc++/libopenvino bind our vague-linkage/RTTI copies — "relocation dependency" — pinning us forever).
- Genuine unload exposed a latent crash: init + dlclose WITHOUT runtime_cleanup segfaults (destructors/unmap under live threads; previously masked because dlclose was a NODELETE no-op). Fixed with a per-DSO `atexit(runtime_cleanup)` hook registered inside `runtime_init` — Linux only; on Windows joining threads in DLL_PROCESS_DETACH deadlocks (loader lock), hosts must call cleanup before FreeLibrary.
- Verified: proper cycle now truly unmaps on dlclose; no-cleanup paths don't crash; double init/cleanup OK; simple_test suite passes; real yolo11s inference OK (~108 img/s CPU). Only surviving thread after cleanup is an Intel OpenCL driver thread (libigdrcl — driver-owned, not our code, lives outside the runtime folder).
- Gotcha discovered while testing: `printf("%d %d", init(cfg), cleanup())` evaluates args right-to-left in GCC — cleanup ran BEFORE init in an early test, faking a "still pinned" result. Watch for this in repro programs.

As of 2026-07-17 all runtime fixes are uncommitted on `oaax-v2`; a Windows rebuild of the DLL is still needed to ship them. The reference-implementation ("OAAX CPU Runtime") repo has the reported worker-thread-not-joined bug and likely the same spdlog/static issues — not yet fixed there.
