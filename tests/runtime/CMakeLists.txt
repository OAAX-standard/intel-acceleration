cmake_minimum_required(VERSION 3.10.2)
project(OAAXRuntimeTests)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Path to the prebuilt runtime library (libRuntimeLibrary.so / RuntimeLibrary.lib)
if(NOT DEFINED RUNTIME_LIB_DIR)
    set(RUNTIME_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../runtime-library/build")
endif()

# Path to the runtime public headers
if(NOT DEFINED RUNTIME_INCLUDE_DIR)
    set(RUNTIME_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../runtime-library/include")
endif()

message(STATUS "Runtime include dir: ${RUNTIME_INCLUDE_DIR}")
message(STATUS "Runtime lib dir:     ${RUNTIME_LIB_DIR}")

foreach(TEST_NAME simple_test yolo_test multi_model_test output_validation_test lifecycle_test)
    add_executable(${TEST_NAME} ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_NAME}.cpp)
    target_include_directories(${TEST_NAME} PRIVATE ${RUNTIME_INCLUDE_DIR})
    target_link_directories(${TEST_NAME} PRIVATE ${RUNTIME_LIB_DIR})
    target_link_libraries(${TEST_NAME} PRIVATE RuntimeLibrary)
    if(NOT MSVC)
        target_link_options(${TEST_NAME} PRIVATE "-Wl,--allow-shlib-undefined")
    endif()
    set_target_properties(${TEST_NAME} PROPERTIES
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
    )
endforeach()
