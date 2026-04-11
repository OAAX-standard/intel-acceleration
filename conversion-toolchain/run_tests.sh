#!/bin/bash
# Tests have moved to the project root. Run from there.
set -e

ROOT_DIR="$(dirname "$0")/.."
cd "$ROOT_DIR"

if [ ! -d ".venv" ]; then
    uv venv
fi
# shellcheck source=/dev/null
source .venv/bin/activate

uv sync

echo ""
pytest tests/test_conversion.py tests/test_docker.py -v --tb=short
echo ""
