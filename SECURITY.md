# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| Latest (`main`) | ✅ |
| Older releases | ❌ |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Send a report to **aassis@networkoptix.com** with:
- A description of the vulnerability and its potential impact
- Steps to reproduce or a proof-of-concept
- The component affected (runtime library, conversion toolchain, or both)
- Any suggested fix if you have one

You can expect an acknowledgement within **48 hours** and a status update within **7 days**.

## Scope

This policy covers:
- `runtime-library/` — the C++ shared library and its public C API
- `conversion-toolchain/` — the Docker-based ONNX-to-OpenVINO conversion toolchain

Third-party dependencies (OpenVINO, NNCF, spdlog, etc.) should be reported to their respective maintainers.
