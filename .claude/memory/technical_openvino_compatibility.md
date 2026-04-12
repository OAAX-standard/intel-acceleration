---
name: OpenVINO/NNCF compatibility notes
description: Version compatibility fixes, CMake pitfalls, and Windows packaging gotchas for OpenVINO C++ builds
type: project
---
Hard-won compatibility fixes — apply these before debugging strange import or link errors.

**Why:** These broke silently across version upgrades and took significant debugging time.
**How to apply:** When touching OpenVINO version, NNCF version, or C++ build flags, check these.

## Python: NNCF requires openvino.Node at import time

NNCF (all versions) does `from openvino import Node` at import. The path changed across OV versions:
- OV 2024.6: `Node` moved to `openvino.runtime.Node`, `openvino.op` removed
- OV 2026.1.0: `Node` restored to top-level `openvino.Node`, `openvino.op` restored

Compatibility shim in `conversion_toolchain/quantization.py` (already applied):
```python
if not hasattr(ov, 'Node'):
    try:
        import openvino.runtime as _ov_runtime
        ov.Node = _ov_runtime.Node
    except ModuleNotFoundError:
        pass
```

Current pins: `openvino>=2026.1.0`, `nncf>=2.19.0,<3.0.0`

## C++: GCC dual ABI — archive install uses C++11 ABI

The OpenVINO **archive** install (`openvino_toolkit_ubuntu22_*.tgz`) is built with the default C++11 ABI
(`_GLIBCXX_USE_CXX11_ABI=1`). Do NOT force `-D_GLIBCXX_USE_CXX11_ABI=0` when linking against
the archive — it would mangle `std::string` differently and cause `ov::` symbol mismatches.

(The pip package used the old ABI; the archive does not. If you see ABI mismatch link errors,
verify which OpenVINO source you're using before flipping this flag.)

## C++: versioned .so files need unversioned symlinks for the linker

OpenVINO ships only versioned files (`libopenvino.so.2450`, `libtbb.so.12`).
The cross-linker needs unversioned names (`libopenvino.so`).

When using the archive, set `OPENVINO_LINK_DIR` to a directory of unversioned symlinks.
`tests/stage2.py` creates symlinks automatically in `build/openvino_links/` and passes
`-DOPENVINO_LINK_DIR=<path>` to cmake. Don't skip this step.

## benchmark_app: use latency hint for per-request latency

`-hint throughput` uses multiple parallel inference requests — reported latency is
"latency at max throughput" (higher than serial latency).
`-hint latency` uses a single request — reported latency is the true per-request time.
Add `-latency_percentile 95` to get the p95 line in output.

## CMake: use if(WIN32) before project(), not if(MSVC)

`MSVC` is only set after the compiler is detected during `project()`. Any `if(MSVC)` block before
`project()` silently evaluates false. Use `if(WIN32)` for path/layout configuration that must
happen early (before `project()` is called).

Concretely: `OPENVINO_BIN_DIR` was never set because it was inside an `if(MSVC)` block at line 30,
before `project()` at line 105. The cmake -P copy script received an empty `SRC_BIN_DIR` and found
no DLLs. Fixed by changing that block to `if(WIN32)`.

## CMake: Windows DLL copy — cmake -P at build time, not file(GLOB) at configure time

`file(GLOB "${OPENVINO_BIN_DIR}/*.dll")` at configure time returns empty if the OpenVINO archive
hasn't been extracted yet (common in CI where extraction happens in a prior step). Solution: use
`add_custom_command POST_BUILD ... -P cmake/copy_windows_dlls.cmake` — `file(GLOB)` inside a
`cmake -P` script runs at build time when files exist.

Also: never put `|` or `{}` in a CMake Windows post-build shell command — cmd.exe/MSBuild interpret
them as pipe operators and property-function delimiters respectively, breaking the command silently.

## CMake Linux: find -exec pitfalls in add_custom_command

Three ways the `find -exec` post-build command can silently break:
1. Without `VERBATIM`, CMake shell-expands `*.so*` in the generated Makefile — find matches nothing
2. `";"` in a CMake COMMAND list is treated as a list separator and silently dropped — use `+` terminator
3. With `+` terminator, `{}` must be the second-to-last token: `--target-directory=DEST {} +`

Always use `VERBATIM` and the `+` form:
```cmake
COMMAND find ${DIR} -name "*.so*" -exec cp -P --target-directory=${DEST} {} +
VERBATIM
```

## RPATH: patchelf all bundled .so files, not just the main library

Cross-linker bakes build-time paths (e.g. `/opt/intel/openvino/...`) as DT_RPATH.
`patchelf --set-rpath '$ORIGIN'` converts to DT_RUNPATH=$ORIGIN. DT_RUNPATH is NOT inherited
by transitive dependencies — so patch every bundled `.so`, not just `libRuntimeLibrary.so`,
so that e.g. `libopenvino.so → libtbb.so.12` resolution also works at runtime.

## OpenVINO archive layout (2026.1.0)

**Linux** (`openvino_toolkit_ubuntu22_*.tgz`):
- .so libs: `runtime/lib/intel64/*.so*`
- TBB: `runtime/3rdparty/tbb/lib/*.so*`

**Windows** (`openvino_toolkit_windows_*.zip`):
- Import libs: `runtime/lib/intel64/Release/*.lib`
- DLLs: `runtime/bin/intel64/Release/*.dll`
- TBB: `runtime/3rdparty/tbb/bin/*.dll`

Use the archive install (not `pip install openvino`) for C++ builds — the pip package lacks
unversioned `.so` symlinks and uses a non-standard directory layout.
