# intel-acceleration

This folder contains the source code of the shared library and the Docker image that can be used by AI application developers to benefit from the acceleration offered by Intel CPU, GPU and NPU on x86_64 machines.

> To learn how to deploy on Intel, please check out the technical docs at [https://docs.oaax.org/Intel/](https://docs.oaax.org/Intel/)

## Repository structure

- [Conversion toolchain](conversion-toolchain): Source for the OAAX conversion toolchain (ONNX → OpenVINO IR).
- [Runtime library](runtime-library): Source for the OAAX runtime library (OpenVINO inference).
- [tests/](tests): All tests — conversion, Docker, YOLO integration, and C++ runtime tests.
- [scripts/](scripts): Utility and CI scripts, including `run_integration_tests.sh`.

Each component folder contains a README with further details.

## Building the implementation

You can build the conversion toolchain and the runtime separately by calling the (Shell) build scripts in each folder.
That will create an `artifacts/` directory in each folder containing the compiled binaries: a compressed Docker image and shared libraries (for X86_64 target machines) respectively.

## Pre-built OAAX artifacts

If you're interested in using the OAAX toolchain and runtime without building them, you can find them in the
[contributions](https://github.com/oaax-standard/contributions) repository.   
Additionally, you can find a diverse set of examples and applications of using the OAAX runtime in the 
[examples](https://github.com/oaax-standard/examples) repository.

## Testing

All tests are in [`tests/`](tests) and run from the project root with `uv` and `pytest`.

### Setup

```bash
uv sync                                           # base deps (conversion unit tests)
uv sync --extra integration                       # + ultralytics (YOLO tests)
uv sync --extra quantization                      # + nncf (INT8 tests)
uv sync --extra integration --extra quantization  # both
uv sync --extra all                               # everything
```

### Run

```bash
# Conversion unit tests (no Docker, no GPU required)
uv run pytest tests/test_conversion.py -v

# YOLO pipeline tests — conversion + OpenVINO inference (YOLOv8n, YOLOv11n)
uv run pytest tests/test_yolo_integration.py -v

# INT8 quantization tests (requires --extra quantization)
uv run pytest tests/test_yolo_integration.py -v -k int8

# Docker tests (requires image: IMAGE_NAME=oaax-intel-toolchain bash conversion-toolchain/build-toolchain.sh)
uv run pytest tests/test_docker.py -v

# Full end-to-end including C++ runtime (requires runtime to be built)
bash scripts/run_integration_tests.sh
```

### C++ runtime tests

```bash
cd runtime-library && bash build-runtimes.sh   # builds simple_test and yolo_test too
cd build
./simple_test                   # smoke test, no model needed
./yolo_test /path/to/model.xml  # full inference test
```

## Contributing

If you're interested in contributing to the OAAX reference implementation, please check out the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information on how to get started.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.
