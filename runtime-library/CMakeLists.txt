cmake_minimum_required(VERSION 3.10.2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable verbose makefile output (set to OFF for less output)
set(CMAKE_VERBOSE_MAKEFILE OFF)

# === Directory Structure ===
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(DEPS_DIR ${CMAKE_CURRENT_LIST_DIR}/deps)
set(SPDLOG_DIR ${DEPS_DIR}/spdlog)
set(CONCURRENT_QUEUE_DIR ${DEPS_DIR}/concurrentqueue)
set(TOOLS_C_UTILITIES_DIR ${DEPS_DIR}/tools/c-utilities)
set(TOOLS_C_UTILITIES_INCLUDE_DIR ${DEPS_DIR}/tools/c-utilities/include)

# === OpenVINO Configuration ===
# Default assumes the archive installation extracted to /opt/intel/openvino,
# i.e. OPENVINO_DIR points to the `runtime/` subdirectory of the archive.
# Override with -DOPENVINO_DIR=<path> or the OPENVINO_DIR env variable.
if(NOT DEFINED OPENVINO_DIR)
    set(OPENVINO_DIR "/opt/intel/openvino/runtime")
endif()

set(OPENVINO_INCLUDE_DIR "${OPENVINO_DIR}/include")

# Archive layout: runtime/lib/intel64/ (Linux) or runtime/lib/intel64/Release/ (Windows)
if(MSVC)
    set(OPENVINO_LIB_DIR "${OPENVINO_DIR}/lib/intel64/Release")
    set(OPENVINO_BIN_DIR "${OPENVINO_DIR}/bin/intel64/Release")
else()
    set(OPENVINO_LIB_DIR "${OPENVINO_DIR}/lib/intel64")
endif()

# Verify OpenVINO installation
if(NOT EXISTS "${OPENVINO_INCLUDE_DIR}/openvino/openvino.hpp")
    message(FATAL_ERROR "OpenVINO not found at ${OPENVINO_DIR}. Please specify -DOPENVINO_DIR=<path>")
endif()

message(STATUS "OpenVINO include directory: ${OPENVINO_INCLUDE_DIR}")
message(STATUS "OpenVINO library directory: ${OPENVINO_LIB_DIR}")

# === Platform Selection and Toolchain Setup ===
if(WIN32)
  set(PLATFORM WINDOWS)
endif()

if(NOT DEFINED PLATFORM)
  message(FATAL_ERROR "Please specify the target platform: -DPLATFORM=X86_64 or use -DPLATFORM=WINDOWS.")
endif()

# === Runtime Version ===
if(NOT DEFINED RUNTIME_VERSION)
  message(FATAL_ERROR "Please specify the runtime version: -DRUNTIME_VERSION=<version>")
endif()

add_definitions(-DRUNTIME_VERSION="${RUNTIME_VERSION}")
message(STATUS "Runtime version: ${RUNTIME_VERSION}")

if(PLATFORM STREQUAL "X86_64")
  set(CMAKE_SYSTEM_PROCESSOR x86_64)
  set(CMAKE_SYSTEM_NAME Linux)
  set(CROSS_ROOT "/opt/x86_64-unknown-linux-gnu-gcc-9.5.0")
  set(COMPILER_PREFIX "x86_64-unknown-linux-gnu-")
  set(SYSROOT "/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/x86_64-unknown-linux-gnu/sysroot")
  set(CMAKE_BUILD_FLAGS "-fno-math-errno -fopenmp -march=haswell")
elseif(PLATFORM STREQUAL "WINDOWS")
  set(CMAKE_SYSTEM_PROCESSOR x86_64)
  set(CMAKE_SYSTEM_NAME Windows)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3 /MD /D_CRT_SECURE_NO_WARNINGS /DNOMINMAX")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MD /EHsc /D_CRT_SECURE_NO_WARNINGS /DNOMINMAX")
else()
  message(FATAL_ERROR "Please specify the target platform: X86_64, or WINDOWS. Found ${PLATFORM}")
endif()

# Print toolchain info
message(STATUS "CROSS_ROOT=${CROSS_ROOT}")

# Set the root path for finding libraries and headers
set(CMAKE_FIND_ROOT_PATH ${SDKTARGETSYSROOT})

# Set CMake toolchain variables for cross-compilation
if(NOT WIN32)
  set(CMAKE_AR ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ar)
  set(CMAKE_AS ${CROSS_ROOT}/bin/${COMPILER_PREFIX}as)
  set(CMAKE_LD ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld)
  set(CMAKE_RANLIB ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ranlib)
  set(CMAKE_C_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
  set(CMAKE_CXX_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++)
  set(CMAKE_ASM_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
  set(CMAKE_NM ${CROSS_ROOT}/bin/${COMPILER_PREFIX}NM)
endif()

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Reset default linker flags
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS " ")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS " ")

project(RuntimeLibrary)

# === Compile Flags ===
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT MSVC)
  set(CMAKE_BUILD_FLAGS "${CMAKE_BUILD_FLAGS} -fno-common -Wall -Wextra -Wformat -Wformat-security")
  set(CMAKE_BUILD_FLAGS "${CMAKE_BUILD_FLAGS} -Wno-missing-field-initializers -Wno-attribute-warning")
  set(CMAKE_BUILD_FLAGS "${CMAKE_BUILD_FLAGS} -fdata-sections -ffunction-sections -Wl,--gc-sections -fno-exceptions")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wformat -Wformat-security -fdata-sections -ffunction-sections -Wl,--gc-sections")
  # The OpenVINO archive is built for Ubuntu 22.04 and uses the new C++11 ABI
  # (_GLIBCXX_USE_CXX11_ABI=1, the default). Do NOT force the old ABI here —
  # doing so would mangle std::string differently and cause ov:: symbol mismatches.
  set(CMAKE_C_FLAGS " ${CMAKE_C_FLAGS} ${CMAKE_BUILD_FLAGS} -D_GNU_SOURCE -I${CROSS_ROOT}/include")
  set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} ${CMAKE_BUILD_FLAGS} -D_GNU_SOURCE -I${CROSS_ROOT}/include -fexceptions")
  set(CMAKE_CPP_FLAGS " ${CMAKE_CPP_FLAGS} ${CMAKE_BUILD_FLAGS} -D_GNU_SOURCE -I${CROSS_ROOT}/include -fexceptions")
endif()

# === Generic Build Settings ===
set(LINKER_FLAGS "${LINKER_FLAGS} ${CMAKE_BUILD_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_BUILD_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_BUILD_FLAGS}")

# Add spdlog for logging
add_subdirectory(${SPDLOG_DIR} ${CMAKE_CURRENT_BINARY_DIR}/spdlog)

# Add c-utilities
add_subdirectory(${TOOLS_C_UTILITIES_DIR} ${CMAKE_CURRENT_BINARY_DIR}/c_utilities)

# === Source Files ===
file(GLOB_RECURSE SRC
  ${SRC_DIR}/runtime_utils.cpp
  ${SRC_DIR}/runtime_core.cpp
)

# === Library Target ===
add_library(RuntimeLibrary SHARED ${SRC})

if(NOT MSVC)
  target_link_options(RuntimeLibrary PRIVATE "-Wl,-z,noexecstack")
  # Cross-compilation: the target system (Ubuntu 22.04) has GLIBC 2.32-2.34 and
  # TBB, but the cross-compiler sysroot does not. Allow unresolved shared-library
  # symbols so the linker doesn't fail — they are satisfied at runtime.
  target_link_options(RuntimeLibrary PRIVATE "-Wl,--allow-shlib-undefined")
endif()

# === Optimization Flags ===
if(MSVC)
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    target_compile_options(RuntimeLibrary PUBLIC /O2 /DNDEBUG)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    target_compile_options(RuntimeLibrary PUBLIC /Od /Zi /DDEBUG)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    target_compile_options(RuntimeLibrary PUBLIC /O2 /Zi)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel")
    target_compile_options(RuntimeLibrary PUBLIC /O1)
  endif()
else()
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    target_compile_options(RuntimeLibrary PUBLIC -Ofast -s -g)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    target_compile_options(RuntimeLibrary PUBLIC -O0 -g3)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    target_compile_options(RuntimeLibrary PUBLIC -O2 -g)
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel")
    target_compile_options(RuntimeLibrary PUBLIC -Os -g)
  endif()
endif()

# === Include Directories ===
target_include_directories(RuntimeLibrary PUBLIC
  ${INCLUDE_DIR}
  ${TOOLS_C_UTILITIES_INCLUDE_DIR}
  ${CONCURRENT_QUEUE_DIR}
  ${OPENVINO_INCLUDE_DIR}
)

# === Link Directories ===
# OPENVINO_LINK_DIR can point to a directory of unversioned symlinks when the
# OpenVINO package only ships versioned .so files (e.g. libopenvino.so.2450).
if(NOT DEFINED OPENVINO_LINK_DIR)
  set(OPENVINO_LINK_DIR ${OPENVINO_LIB_DIR})
endif()
# TBB is bundled in the archive at runtime/3rdparty/tbb/lib/
set(OPENVINO_TBB_LIB_DIR "${OPENVINO_DIR}/3rdparty/tbb/lib")
target_link_directories(RuntimeLibrary PUBLIC
  ${OPENVINO_LINK_DIR}
  ${OPENVINO_LIB_DIR}
  ${OPENVINO_TBB_LIB_DIR}
)

# === Link Libraries ===
if(MSVC)
  target_link_libraries(RuntimeLibrary PUBLIC
    spdlog::spdlog
    c_utilities
    openvino
    ws2_32 bcrypt
  )

  # Copy OpenVINO DLLs if found (archive layout: bin/intel64/Release/)
  file(GLOB OPENVINO_DLLS "${OPENVINO_BIN_DIR}/*.dll")
  if(OPENVINO_DLLS)
    add_custom_command(TARGET RuntimeLibrary POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${OPENVINO_DLLS}
      "$<TARGET_FILE_DIR:RuntimeLibrary>/"
    )
  endif()
else()
  target_link_libraries(RuntimeLibrary PUBLIC
    -Wl,--start-group
    openvino
    spdlog::spdlog
    pthread dl c_utilities stdc++
    -Wl,--end-group
  )

  # Copy only the OpenVINO libs needed at runtime: core, device plugins,
  # IR frontend, and threading libs. Unused model-format frontends are excluded
  # to keep the package lean — the runtime only loads OpenVINO IR (.xml) files.
  add_custom_command(TARGET RuntimeLibrary POST_BUILD
    COMMAND find ${OPENVINO_LIB_DIR} -maxdepth 1 -name "*.so*"
      "!" -name "libopenvino_onnx_frontend*"
      "!" -name "libopenvino_pytorch_frontend*"
      "!" -name "libopenvino_tensorflow_frontend*"
      "!" -name "libopenvino_tensorflow_lite_frontend*"
      "!" -name "libopenvino_paddle_frontend*"
      "!" -name "libopenvino_jax_frontend*"
      -exec cp -P --target-directory=${CMAKE_CURRENT_BINARY_DIR} {} +
    # Copy TBB and hwloc — libopenvino.so links against them at runtime.
    COMMAND find ${OPENVINO_TBB_LIB_DIR} -maxdepth 1 -name "*.so*"
      -exec cp -P --target-directory=${CMAKE_CURRENT_BINARY_DIR} {} +
    VERBATIM
  )
endif()

# === Test Executables ===
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../tests/runtime)

foreach(TEST_NAME simple_test yolo_test)
  add_executable(${TEST_NAME} ${TEST_DIR}/${TEST_NAME}.cpp)
  target_include_directories(${TEST_NAME} PRIVATE ${INCLUDE_DIR})
  target_link_directories(${TEST_NAME} PRIVATE ${OPENVINO_LINK_DIR} ${OPENVINO_LIB_DIR} ${OPENVINO_TBB_LIB_DIR})
  target_link_libraries(${TEST_NAME} PRIVATE RuntimeLibrary)
  if(NOT MSVC)
    target_link_options(${TEST_NAME} PRIVATE "-Wl,--allow-shlib-undefined")
  endif()
  # Ensure test binary finds the shared libraries in the same directory at runtime
  set_target_properties(${TEST_NAME} PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
  )
endforeach()
