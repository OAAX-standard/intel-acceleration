## Summary

<!-- Briefly describe what this PR does and why. -->

## Related issue

<!-- Link to the issue this PR addresses, e.g. Closes #123 -->

Closes #

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Refactor (no functional change)
- [ ] Documentation update
- [ ] CI / build change

## Checklist

- [ ] I have read [CONTRIBUTING.md](../CONTRIBUTING.md)
- [ ] My branch is rebased on the latest `main`
- [ ] The build passes locally (`bash runtime-library/build-runtimes.sh` / Docker toolchain build)
- [ ] Relevant tests pass (`python tests/stage1.py` and/or `pytest tests/ -v`)
- [ ] I have added or updated tests for any new behaviour
- [ ] I have updated `CHANGELOG.md` under the appropriate version section
- [ ] I have updated documentation (README, API headers, CLAUDE.md) if needed
- [ ] This PR does **not** break the public C API in `runtime_core.hpp`

## Breaking changes

<!-- List any breaking changes to the public API, runtime args, exit codes, or file formats. Write "None" if there are none. -->

None

## Testing notes

<!-- Describe how you tested this change. Include device, precision, and model if relevant. -->
