# intel-acceleration

This folder contains the source code of the shared library and the Docker image that can be used by AI application developers to benefit from the acceleration offered by Intel CPU, GPU and NPU on x86_64 machines.

## Repository structure

The repository is structured as follows:

- [Conversion toolchain](conversion-toolchain): Contains the source code for building the OAAX conversion toolchain.
- [Runtime library](runtime-library): Contains the source code for building the OAAX runtime.

Each folder contains a README file that provides more details about the different parts of the implementation.

## Building the implementation

You can build the conversion toolchain and the runtime separately by calling the (Shell) build scripts in each folder.
That will create an `artifacts/` directory in each folder containing the compiled binaries: a compressed Docker image and shared libraries (for X86_64 target machines) respectively.

## Pre-built OAAX artifacts

If you're interested in using the OAAX toolchain and runtime without building them, you can find them in the
[contributions](https://github.com/oaax-standard/contributions) repository.   
Additionally, you can find a diverse set of examples and applications of using the OAAX runtime in the 
[examples](https://github.com/oaax-standard/examples) repository.

## Contributing

If you're interested in contributing to the OAAX reference implementation, please check out the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information on how to get started.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.