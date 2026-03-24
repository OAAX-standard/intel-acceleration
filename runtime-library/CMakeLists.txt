cmake_minimum_required(VERSION 3.10.2)

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
# Define OpenVINO paths (can be overridden via -DOPENVINO_DIR)
if(NOT DEFINED OPENVINO_DIR)
    # Try to find OpenVINO in common locations
    set(OPENVINO_DIR "/usr/local/lib/python3.10/dist-packages/openvino")
endif()

set(OPENVINO_INCLUDE_DIR "${OPENVINO_DIR}/include")
set(OPENVINO_LIB_DIR "${OPENVINO_DIR}/libs")

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
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W3 /MD /D_CRT_SECURE_NO_WARNINGS")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3 /MD /D_CRT_SECURE_NO_WARNINGS")
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
target_link_directories(RuntimeLibrary PUBLIC
  ${OPENVINO_LIB_DIR}
)

# === Link Libraries ===
if(MSVC)
  target_link_libraries(RuntimeLibrary PUBLIC
    spdlog::spdlog
    c_utilities
    openvino
    tbb
    ws2_32 bcrypt
  )

  # Copy OpenVINO DLLs
  file(GLOB OPENVINO_DLLS "${OPENVINO_LIB_DIR}/*.dll")
  add_custom_command(TARGET RuntimeLibrary POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${OPENVINO_DLLS}
    "$<TARGET_FILE_DIR:RuntimeLibrary>/"
  )
else()
  target_link_libraries(RuntimeLibrary PUBLIC
    -Wl,--start-group
    openvino
    tbb
    spdlog::spdlog
    pthread dl c_utilities stdc++
    -Wl,--end-group
  )

  # Copy OpenVINO shared libraries
  add_custom_command(TARGET RuntimeLibrary POST_BUILD
    COMMAND cp -P ${OPENVINO_LIB_DIR}/*.so* ${CMAKE_CURRENT_BINARY_DIR}/ || true
  )
endif()
