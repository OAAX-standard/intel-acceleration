# Copies OpenVINO DLLs (minus unused frontends) and TBB DLLs to DST_DIR.
# Invoked at build time via cmake -P from CMakeLists.txt post-build step.
# SRC_BIN_DIR, SRC_TBB_DIR, DST_DIR are passed as -D arguments.

file(GLOB openvino_dlls "${SRC_BIN_DIR}/*.dll")
foreach(dll ${openvino_dlls})
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
