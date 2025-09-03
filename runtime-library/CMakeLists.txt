cmake_minimum_required(VERSION 3.10.2)

# Enable verbose makefile output (set to OFF for less output)
set(CMAKE_VERBOSE_MAKEFILE OFF)

# === Directory Structure ===
# Define key source, include, and dependency directories
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(DEPS_DIR ${CMAKE_CURRENT_LIST_DIR}/deps)
set(SPDLOG_DIR ${DEPS_DIR}/spdlog)
set(CONCURRENT_QUEUE_DIR ${DEPS_DIR}/concurrentqueue)
set(TOOLS_C_UTILITIES_DIR ${DEPS_DIR}/tools/c-utilities)
set(TOOLS_C_UTILITIES_INCLUDE_DIR ${DEPS_DIR}/tools/c-utilities/include)
set(ONNX_RUNTIME_DIR "${DEPS_DIR}/onnxruntime")

# === Compile spdlog ===
# Enable position-independent code for shared library compatibility
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# === Platform Selection and Toolchain Setup ===
# Select platform and set cross-compilation toolchain variables
if(WIN32)
  set(PLATFORM WINDOWS)
endif()

if(NOT DEFINED PLATFORM)
  message(FATAL_ERROR "Please specify the target platform: -DPLATFORM=X86_64 or use -DPLATFORM=WINDOWS.")
endif()

# === Runtime Version ===
# Request the runtime version from the user
if(NOT DEFINED RUNTIME_VERSION)
  message(FATAL_ERROR "Please specify the runtime version: -DRUNTIME_VERSION=<version>")
endif()

# Set version as macro definition
add_definitions(-DRUNTIME_VERSION="${RUNTIME_VERSION}")

message(STATUS "Runtime version: ${RUNTIME_VERSION}")

if(PLATFORM STREQUAL "X86_64")
  set(GENERIC_BUILD_TARGET X86_64-1.21.0)
  set(CMAKE_SYSTEM_PROCESSOR x86_64)
  set(CMAKE_SYSTEM_NAME Linux)
  set(CROSS_ROOT "/opt/x86_64-unknown-linux-gnu-gcc-9.5.0")
  set(COMPILER_PREFIX "x86_64-unknown-linux-gnu-")
  set(SYSROOT "/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/x86_64-unknown-linux-gnu/sysroot")

  # -fno-math-errno: Avoid setting errno after math functions (improves performance).
  # -fopenmp: Enable OpenMP for parallel programming.
  # -std=c99: Use the C99 standard for C code.
  # -march=haswell: Optimize code for Intel Haswell CPUs.
  set(CMAKE_BUILD_FLAGS "-fno-math-errno -fopenmp -march=haswell")
elseif(PLATFORM STREQUAL "WINDOWS")
  set(GENERIC_BUILD_TARGET X86_64_WINDOWS-1.21.0)
  set(CMAKE_SYSTEM_PROCESSOR x86_64)
  set(CMAKE_SYSTEM_NAME Windows)

  # Use default MSVC toolchain, do not set cross-compilation variables
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3 /MD /D_CRT_SECURE_NO_WARNINGS")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MD /D_CRT_SECURE_NO_WARNINGS")

# Do not add compiler flags to CMAKE_BUILD_FLAGS to avoid linker warnings
else()
  message(FATAL_ERROR "Please specify the target platform: X86_64, AARCH64, or WINDOWS. Found ${PLATFORM}")
endif()

# Print toolchain and sysroot info for debugging
message(STATUS "SDKTARGETSYSROOT=${SDKTARGETSYSROOT}")
message(STATUS "CROSS_ROOT=${CROSS_ROOT}")

# Set the root path for finding libraries and headers on the target
set(CMAKE_FIND_ROOT_PATH ${SDKTARGETSYSROOT})

# Set CMake toolchain variables for cross-compilation
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_AR ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ar)
set(CMAKE_AS ${CROSS_ROOT}/bin/${COMPILER_PREFIX}as)
set(CMAKE_LD ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld)
set(CMAKE_RANLIB ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ranlib)
set(CMAKE_C_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++)
set(CMAKE_ASM_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
set(CMAKE_NM ${CROSS_ROOT}/bin/${COMPILER_PREFIX}NM)

# Do not use the root path when searching for programs (executables)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# Only search the root path when searching for libraries
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)

# Only search the root path when searching for header files
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Only search the root path when searching for packages (e.g., via find_package)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Reset default linker flags for shared libraries
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS " ")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS " ")

project(RuntimeLibrary)

# Add common warning and optimization flags for C and C++
if(NOT MSVC)
  set(CMAKE_BUILD_FLAGS "${CMAKE_BUILD_FLAGS} -fno-common -Wall -Wextra -Wformat -Wformat-security")
  set(CMAKE_BUILD_FLAGS "${CMAKE_BUILD_FLAGS} -Wno-missing-field-initializers -Wno-attribute-warning")
  set(CMAKE_BUILD_FLAGS "${CMAKE_BUILD_FLAGS} -fdata-sections -ffunction-sections -Wl,--gc-sections -fno-exceptions")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wformat -Wformat-security -fdata-sections -ffunction-sections -Wl,--gc-sections")
  set(CMAKE_C_FLAGS " ${CMAKE_C_FLAGS} ${CMAKE_BUILD_FLAGS}  -D_GNU_SOURCE -I${CROSS_ROOT}/include")
  set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} ${CMAKE_BUILD_FLAGS}  -D_GNU_SOURCE -I${CROSS_ROOT}/include -fexceptions")
  set(CMAKE_CPP_FLAGS " ${CMAKE_CPP_FLAGS} ${CMAKE_BUILD_FLAGS}  -D_GNU_SOURCE -I${CROSS_ROOT}/include -fexceptions")
endif()

# === Generic Build Settings ===
# Set basic C, C++, and linker flags
set(LINKER_FLAGS "${LINKER_FLAGS} ${CMAKE_BUILD_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_BUILD_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_BUILD_FLAGS}")

# Linker flag to error on undefined symbols when building shared libraries
# set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")

# Add spdlog for logging
add_subdirectory(${SPDLOG_DIR} ${CMAKE_CURRENT_BINARY_DIR}/spdlog)

# === Link to c-utilities ===
# Add the c-utilities subdirectory for building and linking
add_subdirectory(${TOOLS_C_UTILITIES_DIR} ${CMAKE_CURRENT_BINARY_DIR}/c_utilities)

# === Source Files ===
# Add all source files for the runtime library
file(GLOB_RECURSE SRC
  ${SRC_DIR}/runtime_utils.cpp
  ${SRC_DIR}/runtime_core.cpp
)

# ####################### main bin build settings #######################################################################

# === Library Target ===
# Create the shared library target for the runtime
add_library(RuntimeLibrary SHARED ${SRC})

# Add optimization and debug flags for Release builds
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
    target_compile_options(RuntimeLibrary PUBLIC
      -Ofast # Enable aggressive optimizations for speed (may break strict standards compliance)
      -s # Strip all symbol information from the output binary (reduces size)
      -g # Generate debug information (minimal, for backtraces)
    )
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    target_compile_options(RuntimeLibrary PUBLIC
      -O0 # No optimization (easier debugging)
      -g3 # Generate maximum debug information
    )
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "RelWithDebInfo")
    target_compile_options(RuntimeLibrary PUBLIC
      -O2 # Optimize for speed, but less aggressively than -Ofast
      -g # Generate debug information
    )
  elseif("${CMAKE_BUILD_TYPE}" STREQUAL "MinSizeRel")
    target_compile_options(RuntimeLibrary PUBLIC
      -Os # Optimize for minimum binary size
      -g # Generate debug information
    )
  endif()
endif()

# === Include Directories ===
# Specify public include paths for consumers
target_include_directories(RuntimeLibrary PUBLIC
  ${INCLUDE_DIR}
  ${TOOLS_C_UTILITIES_INCLUDE_DIR}
  ${CONCURRENT_QUEUE_DIR}
  ${ONNX_RUNTIME_DIR}/${GENERIC_BUILD_TARGET}/include
  ${ONNX_RUNTIME_DIR}/${GENERIC_BUILD_TARGET}/include/onnxruntime/core/session
)

# === Link Directories ===
# Specify library search paths for dependencies
target_link_directories(RuntimeLibrary PUBLIC
  "${ONNX_RUNTIME_DIR}/${GENERIC_BUILD_TARGET}"
)

# === Link Libraries ===
# Link all required libraries and group them for the linker
if(MSVC)
  target_link_libraries(RuntimeLibrary PUBLIC
    absl_bad_any_cast_impl
    absl_bad_optional_access
    absl_bad_variant_access
    absl_base
    absl_city
    absl_civil_time
    absl_cord
    absl_cordz_functions
    absl_cordz_handle
    absl_cordz_info
    absl_cord_internal
    absl_crc32c
    absl_crc_cord_state
    absl_crc_cpu_detect
    absl_crc_internal
    absl_debugging_internal
    absl_decode_rust_punycode
    absl_demangle_internal
    absl_demangle_rust
    absl_examine_stack
    absl_exponential_biased
    absl_failure_signal_handler
    absl_flags_commandlineflag
    absl_flags_commandlineflag_internal
    absl_flags_config
    absl_flags_internal
    absl_flags_marshalling
    absl_flags_parse
    absl_flags_private_handle_accessor
    absl_flags_program_name
    absl_flags_reflection
    absl_flags_usage
    absl_flags_usage_internal
    absl_graphcycles_internal
    absl_hash
    absl_hashtablez_sampler
    absl_int128
    absl_kernel_timeout_internal
    absl_log_entry
    absl_log_globals
    absl_log_internal_check_op
    absl_log_internal_conditions
    absl_log_internal_fnmatch
    absl_log_internal_format
    absl_log_internal_globals
    absl_log_internal_log_sink_set
    absl_log_internal_message
    absl_log_internal_nullguard
    absl_log_internal_proto
    absl_log_severity
    absl_log_sink
    absl_low_level_hash
    absl_malloc_internal
    absl_raw_hash_set
    absl_raw_logging_internal
    absl_spinlock_wait
    absl_stacktrace
    absl_strerror
    absl_strings
    absl_strings_internal
    absl_string_view
    absl_str_format_internal
    absl_symbolize
    absl_synchronization
    absl_throw_delegate
    absl_time
    absl_time_zone
    absl_utf8_for_code_point
    absl_vlog_config_internal
    clog
    cpuinfo
    spdlog::spdlog

    # custom_op_get_const_input_test_library.dll
    custom_op_get_const_input_test_library

    # custom_op_invalid_library.dll
    custom_op_invalid_library

    # custom_op_library.dll
    custom_op_library

    # custom_op_local_function.dll
    custom_op_local_function

    # custom_op_openvino_wrapper_library.dll
    custom_op_openvino_wrapper_library
    flatbuffers
    gmock
    gmock_main
    gtest
    gtest_main
    libprotobuf
    libprotoc
    onnx

    # onnxruntime.dll
    onnxruntime
    onnxruntime_common
    onnxruntime_flatbuffers
    onnxruntime_framework
    onnxruntime_graph
    onnxruntime_lora
    onnxruntime_mlas
    onnxruntime_mocked_allocator
    onnxruntime_optimizer
    onnxruntime_providers

    # onnxruntime_providers_openvino.dll
    onnxruntime_providers_openvino

    # onnxruntime_providers_shared.dll
    onnxruntime_providers_shared
    onnxruntime_session
    onnxruntime_test_utils
    onnxruntime_util
    onnx_proto
    onnx_test_data_proto
    onnx_test_runner_common
    re2

    # test_execution_provider.dll
    test_execution_provider
    win_getopt_wide

    c_utilities
    ws2_32 bcrypt
  )

  # copy shared libraries to binary directory
  file(GLOB ONNXRUNTIME_DLLS "${ONNX_RUNTIME_DIR}/${GENERIC_BUILD_TARGET}/*.dll")
  add_custom_command(TARGET RuntimeLibrary POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${ONNXRUNTIME_DLLS}
    "$<TARGET_FILE_DIR:RuntimeLibrary>/"
  )
else()
  target_link_libraries(RuntimeLibrary PUBLIC
    -Wl,--start-group
    absl_bad_optional_access
    absl_bad_variant_access
    absl_base
    absl_city
    absl_civil_time
    absl_cord
    absl_cord_internal
    absl_cordz_functions
    absl_cordz_handle
    absl_cordz_info
    absl_crc32c
    absl_crc_cord_state
    absl_crc_cpu_detect
    absl_crc_internal
    absl_debugging_internal
    absl_decode_rust_punycode
    absl_demangle_internal
    absl_demangle_rust
    absl_examine_stack
    absl_exponential_biased
    absl_flags_commandlineflag
    absl_flags_commandlineflag_internal
    absl_flags_config
    absl_flags_internal
    absl_flags_marshalling
    absl_flags_private_handle_accessor
    absl_flags_program_name
    absl_flags_reflection
    absl_graphcycles_internal
    absl_hash
    absl_hashtablez_sampler
    absl_int128
    absl_kernel_timeout_internal
    absl_log_entry
    absl_log_globals
    absl_log_internal_check_op
    absl_log_internal_conditions
    absl_log_internal_fnmatch
    absl_log_internal_format
    absl_log_internal_globals
    absl_log_internal_log_sink_set
    absl_log_internal_message
    absl_log_internal_nullguard
    absl_log_internal_proto
    absl_log_severity
    absl_log_sink
    absl_low_level_hash
    absl_malloc_internal
    absl_raw_hash_set
    absl_raw_logging_internal
    absl_spinlock_wait
    absl_stacktrace
    absl_strerror
    absl_str_format_internal
    absl_strings
    absl_strings_internal
    absl_string_view
    absl_symbolize
    absl_synchronization
    absl_throw_delegate
    absl_time
    absl_time_zone
    absl_utf8_for_code_point
    absl_vlog_config_internal
    clog
    cpuinfo
    flatbuffers
    onnx
    onnx_proto
    onnxruntime_common
    onnxruntime_flatbuffers
    onnxruntime_framework
    onnxruntime_graph
    onnxruntime_lora
    onnxruntime_mlas
    onnxruntime_optimizer
    onnxruntime_providers
    onnxruntime_session
    onnxruntime_util
    protobuf
    protoc
    re2
    spdlog::spdlog
    pthread dl c_utilities stdc++
    -Wl,--end-group
  )

  # copy shared libraries to binary directory
  add_custom_command(TARGET RuntimeLibrary POST_BUILD
    COMMAND cp ${ONNX_RUNTIME_DIR}/${GENERIC_BUILD_TARGET}/*.so* ${CMAKE_CURRENT_BINARY_DIR}/
  )
endif()