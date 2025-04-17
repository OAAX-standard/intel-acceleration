# OAAX OpenVINO runtime library

This OpenVINO implementation of an OAAX runtime is a wrapper around the [ONNX Runtime](https://github.com/microsoft/onnxruntime) with the OpenVINO execution provider.

## Pre-requisites

Before you start, make sure you have set up your environment using the following script:

```bash
bash scripts/setup-env.sh
```

This will install the required dependencies and set up the environment for building the OAAX runtime.
The script will also set up the cross-compilation toolchain for the target architecture (X86_64).

## Getting started

The OAAX runtime is leveraging the ONNX Runtime library to load and run the model. ORT requires
the [CPU INFOrmation library](https://github.com/pytorch/cpuinfo), and the [RE2 library](https://github.com/google/re2).

All of these dependencies are included in the `deps` directory, and are already cross-compiled for the target
architecture using the compiler toolchain setup usnig the shell script above.
However, you can recompile them separately by running the Shell scripts inside each directory.

To build the OAAX runtime, run the following command:

```bash
bash build-runtimes.sh
```

This will create an `artifacts/` directory containing the compiled shared library: `libRuntimeLibrary.so` along with other openvino-related libraries.

## Running Inference using the OAAX runtime

To run the inference process, you need to have the optimized model file, the shared library built in the previous step, along with a simple C/C++ code that loads the shared library and the model and runs it. Ensure that you have all necessary OpenVINO-related libraries available reachable to the AI application by setting the `LD_LIBRARY_PATH` environment variable to include the path to the OpenVINO libraries.

You can find diverse examples and applications of using the OAAX runtime in the
[examples](https://github.com/oaax-standard/examples) repository.
