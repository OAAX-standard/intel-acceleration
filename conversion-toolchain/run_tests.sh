#!/bin/bash
set -e

echo "========================================="
echo "Phase 1 Unit Tests - Conversion Toolchain"
echo "========================================="
echo ""

# Navigate to the conversion-toolchain directory
cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

echo "Installing dependencies with uv..."
uv pip install -e ".[test]"

echo ""
echo "Running pytest..."
echo ""

pytest tests/ -v --tb=short

echo ""
echo "========================================="
echo "All Phase 1 tests completed!"
echo "========================================="
