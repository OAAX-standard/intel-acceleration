# Copies OpenVINO DLLs (minus unused frontends) and TBB DLLs to DST_DIR.
# Invoked at build time via cmake -P from CMakeLists.txt post-build step.
# SRC_BIN_DIR, SRC_TBB_DIR, DST_DIR are passed as -D arguments.
#
# SRC_BIN_DIR may not be the right location across all OpenVINO versions, so
# we probe a few candidate directories and use the first one that has DLLs.

set(_openvino_dlls "")

foreach(_candidate "${SRC_BIN_DIR}" "${SRC_LIB_DIR}")
    if(_candidate STREQUAL "")
        continue()
    endif()
    file(GLOB _found "${_candidate}/*.dll")
    if(_found)
        set(_openvino_dlls ${_found})
        message(STATUS "copy_windows_dlls: found OpenVINO DLLs in ${_candidate}")
        break()
    endif()
endforeach()

if(NOT _openvino_dlls)
    message(WARNING "copy_windows_dlls: no OpenVINO DLLs found in SRC_BIN_DIR='${SRC_BIN_DIR}' or SRC_LIB_DIR='${SRC_LIB_DIR}'")
endif()

foreach(dll ${_openvino_dlls})
    get_filename_component(name "${dll}" NAME)
    if(NOT name MATCHES "openvino_onnx_frontend|openvino_pytorch_frontend|openvino_tensorflow_frontend|openvino_tensorflow_lite_frontend|openvino_paddle_frontend")
        file(COPY "${dll}" DESTINATION "${DST_DIR}")
    endif()
endforeach()

file(GLOB tbb_dlls "${SRC_TBB_DIR}/*.dll")
foreach(dll ${tbb_dlls})
    get_filename_component(name "${dll}" NAME)
    if(NOT name MATCHES "_debug\\.dll$")
        file(COPY "${dll}" DESTINATION "${DST_DIR}")
    endif()
endforeach()

# Copy Visual C++ runtime DLLs required on machines without the redistributable.
get_filename_component(_runtime_lib_dir "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
file(GLOB _vcrt_dlls "${_runtime_lib_dir}/deps/windows-runtime/*.dll")
foreach(dll ${_vcrt_dlls})
    file(COPY "${dll}" DESTINATION "${DST_DIR}")
endforeach()
