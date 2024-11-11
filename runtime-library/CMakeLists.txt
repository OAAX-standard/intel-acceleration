cmake_minimum_required(VERSION 3.10.2)

project(RuntimeLibrary)

# where is the nx-c source code located?
set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(DEPS_DIR ${CMAKE_CURRENT_LIST_DIR}/deps)

# Specify
set(OPENVINO_VERSION "NONE" CACHE STRING "The OpenVINO version: 2023")
# Make sure the user has set the correct OpenVINO version
if(NOT OPENVINO_VERSION STREQUAL "2023")
        message(FATAL_ERROR "Please set the OpenVINO version using -DOPENVINO_VERSION=2023")
endif()
message("OPENVINO_VERSION: ${OPENVINO_VERSION}")
######################### customize when cross-compiling ###############################################################

# set COMPILER_PREFIX, for example, "" for default compiler, arm-linux- , or aarch64-linux- etc for cross compilers
# set generic build target
set(GENERIC_BUILD_TARGET X86_64)
set(CROSS_ROOT "/opt/x86_64-unknown-linux-gnu-gcc-9.5.0")
set(COMPILER_PREFIX "x86_64-unknown-linux-gnu-")
set(SYSROOT "/opt/x86_64-unknown-linux-gnu-gcc-9.5.0/x86_64-unknown-linux-gnu/sysroot")
set(CMAKE_BUILD_FLAGS " -fno-math-errno -fopenmp -std=c99 -march=haswell")

set(CMAKE_C_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++)
set(CMAKE_LINKER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

#--build: the machine you are building on
#--host: the machine you are building for
#--target: the machine that GCC will produce binary for
set(BUILD X86_64-linux-gnu)
set(HOST ${GENERIC_BUILD_TARGET}-linux)
set(TARGET ${GENERIC_BUILD_TARGET})

######################### webassembly wamr settings ####################################################################

# set make system
set(CMAKE_SYSTEM_NAME Linux)

######################## webassembly wamr library build ################################################################

# Reset default linker flags
set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS " ")
set(CMAKE_SHARED_LIBRARY_LINK_CXX_FLAGS " ")

# compiling and linking flags
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wformat -Wformat-security -fdata-sections -ffunction-sections -Wl,--gc-sections")

####################### toolchain arguments ############################################################################

INCLUDE(CMakeForceCompiler)

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_BUILD_FLAGS " ${CMAKE_BUILD_FLAGS} -std=c99 -fno-common -Wall -Wextra")
set(CMAKE_BUILD_FLAGS " ${CMAKE_BUILD_FLAGS} -Wno-missing-field-initializers -fno-exceptions")
set(CMAKE_BUILD_FLAGS " ${CMAKE_BUILD_FLAGS} -Wno-attribute-warning -fdata-sections -ffunction-sections -Wl,--gc-sections ")

set(CMAKE_C_FLAGS " ${CMAKE_C_FLAGS} ${CMAKE_BUILD_FLAGS}  -D_GNU_SOURCE -I${CROSS_ROOT}/include")
set(CMAKE_CXX_FLAGS " ${CMAKE_CXX_FLAGS} ${CMAKE_BUILD_FLAGS}  -D_GNU_SOURCE -I${CROSS_ROOT}/include")
set(CMAKE_CPP_FLAGS " ${CMAKE_CPP_FLAGS} ${CMAKE_BUILD_FLAGS}  -D_GNU_SOURCE -I${CROSS_ROOT}/include")

message(STATUS "SDKTARGETSYSROOT=${SDKTARGETSYSROOT}")
message(STATUS "CROSS_ROOT=${CROSS_ROOT}")

set(CMAKE_AR ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ar)
set(CMAKE_AS ${CROSS_ROOT}/bin/${COMPILER_PREFIX}as)
set(CMAKE_LD ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ld)
set(CMAKE_RANLIB ${CROSS_ROOT}/bin/${COMPILER_PREFIX}ranlib)
set(CMAKE_C_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}g++)
set(CMAKE_ASM_COMPILER ${CROSS_ROOT}/bin/${COMPILER_PREFIX}gcc)
set(CMAKE_NM ${CROSS_ROOT}/bin/${COMPILER_PREFIX}NM)

# this is the file system root of the target
set(CMAKE_FIND_ROOT_PATH ${SDKTARGETSYSROOT})

# search for programs in the build host directories
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)

# for libraries and headers in the target directories
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

######################## generic build settings ########################################################################

# set some basic c, c++ and linker flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${CMAKE_BUILD_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CMAKE_BUILD_FLAGS}")
set(LINKER_FLAGS "${LINKER_FLAGS} ${CMAKE_BUILD_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_BUILD_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_BUILD_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_BUILD_FLAGS}")

add_definitions(-DOPENVINO_VERSION=${OPENVINO_VERSION})

# add all source files
file(GLOB_RECURSE SRC
        ${SRC_DIR}/runtime_utils.c
        ${SRC_DIR}/runtime_core.c
        ${DEPS_DIR}/utils/queue.c
        ${DEPS_DIR}/utils/logger.c
)
######################## main bin build settings #######################################################################

# binary target
add_library(RuntimeLibrary SHARED ${SRC})

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
        target_compile_options(RuntimeLibrary PUBLIC
                -Ofast -s
        )
endif()

# debugging on/off RuntimeLibrary
if(NX_DEBUG)
        target_compile_definitions(RuntimeLibrary PRIVATE NX_DEBUG_PRINT)
endif()

# target include / link dirs
target_include_directories(RuntimeLibrary PUBLIC ${INCLUDE_DIR})
target_include_directories(RuntimeLibrary PUBLIC ${DEPS_DIR}/utils)

set(RE2_DIR "${DEPS_DIR}/re2")
set(ONNX_RUNTIME_DIR "${DEPS_DIR}/onnxruntime")
set(CPUINFO_DIR "${DEPS_DIR}/cpuinfo")

target_include_directories(RuntimeLibrary PUBLIC ${ONNX_RUNTIME_DIR}/${GENERIC_BUILD_TARGET}-${OPENVINO_VERSION}/include)
target_include_directories(RuntimeLibrary PUBLIC ${ONNX_RUNTIME_DIR}/${GENERIC_BUILD_TARGET}-${OPENVINO_VERSION}/include/onnxruntime/core/session)
target_link_directories(RuntimeLibrary PUBLIC ${ONNX_RUNTIME_DIR}/${GENERIC_BUILD_TARGET}-${OPENVINO_VERSION})
# check if openvino version is 2023
if(OPENVINO_VERSION STREQUAL "2023")
        target_include_directories(RuntimeLibrary PUBLIC ${RE2_DIR}/include)
        target_link_directories(RuntimeLibrary PUBLIC "${RE2_DIR}/${GENERIC_BUILD_TARGET}")
        target_link_libraries(RuntimeLibrary PUBLIC re2)

        target_include_directories(RuntimeLibrary PUBLIC ${CPUINFO_DIR}/include)
        target_link_directories(RuntimeLibrary PUBLIC "${CPUINFO_DIR}/${GENERIC_BUILD_TARGET}")
        target_link_libraries(RuntimeLibrary PUBLIC cpuinfo)

        target_link_libraries(
                RuntimeLibrary
                PUBLIC
                onnxruntime_session
                onnxruntime_optimizer
                onnxruntime_providers
                onnxruntime_util
                onnxruntime_framework
                onnxruntime_graph
                onnxruntime_mlas
                onnxruntime_common
                onnxruntime_flatbuffers
                absl_base
                absl_log_severity
                absl_raw_logging_internal
                absl_malloc_internal
                absl_spinlock_wait
                absl_throw_delegate
                absl_hashtablez_sampler
                absl_raw_hash_set
                absl_hash
                absl_city
                absl_low_level_hash
                absl_int128
                absl_exponential_biased
                absl_cord
                absl_cord_internal
                absl_cordz_functions
                absl_cordz_handle
                absl_cordz_info
                absl_strings
                absl_strings_internal
                absl_graphcycles_internal
                absl_synchronization
                absl_civil_time
                absl_time
                absl_time_zone
                absl_bad_optional_access
                absl_bad_variant_access
                flatbuffers
                clog
                nsync_cpp
                onnx
                onnx_proto
                cpuinfo
                protobuf
                re2

                stdc++
        )

elseif(OPENVINO_VERSION STREQUAL "20241")
        target_link_libraries(
                RuntimeLibrary
                PUBLIC
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
                absl_demangle_internal
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
                absl_log_severity
                absl_low_level_hash
                absl_malloc_internal
                absl_raw_hash_set
                absl_raw_logging_internal
                absl_spinlock_wait
                absl_stacktrace
                absl_str_format_internal
                absl_strings
                absl_strings_internal
                absl_string_view
                absl_symbolize
                absl_synchronization
                absl_throw_delegate
                absl_time
                absl_time_zone
                clog
                cpuinfo
                flatbuffers
                nsync_cpp
                onnx
                onnx_proto
                onnxruntime_common
                onnxruntime_flatbuffers
                onnxruntime_framework
                onnxruntime_graph
                onnxruntime_mlas
                onnxruntime_optimizer
                onnxruntime_providers
                onnxruntime_session
                onnxruntime_util
                protobuf
                protoc
                re2

                # Finally, generic imports
                dl
                m
                pthread
                stdc++
                -Wl,--end-group

        )
elseif(OPENVINO_VERSION STREQUAL "20244")
        target_link_libraries(
                RuntimeLibrary
                PUBLIC
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
                cpuinfo
                flatbuffers
                nsync_cpp
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

                # Finally, generic imports
                dl
                m
                pthread
                stdc++
                -Wl,--end-group

        )
endif()

# copy shared libraries to binary directory
add_custom_command(TARGET RuntimeLibrary POST_BUILD
        COMMAND cp ${ONNX_RUNTIME_DIR}/${GENERIC_BUILD_TARGET}-${OPENVINO_VERSION}/*.so* ${CMAKE_CURRENT_BINARY_DIR}/
)
