#!/usr/bin/env bash
# Stage 1: Compile models and run conversion tests.
#
# Converts all YOLO models (FP32/FP16/INT8) to tests/compiled_models/ and
# runs the full conversion test suite.  Compiled models are consumed by Stage 2.
#
# Usage:
#   bash scripts/stage1_compile.sh
#
# Output:
#   tests/compiled_models/<model>/<variant>/<model>.xml + .bin

set -e
cd "$(dirname "$0")/.."

GREEN='\033[0;32m'; RED='\033[0;31m'; BLUE='\033[0;34m'; NC='\033[0m'
pass()   { echo -e "${GREEN}✓${NC} $1"; }
fail()   { echo -e "${RED}✗${NC} $1"; exit 1; }
header() { echo -e "\n${BLUE}=== $1 ===${NC}"; }

# ── 1. Environment ─────────────────────────────────────────────────────────────

header "Step 1: Setting up environment"

[[ ! -d .venv ]] && uv venv
# shellcheck source=/dev/null
source .venv/bin/activate
uv sync --extra integration --extra quantization -q

pass "Environment ready"

# ── 2. Conversion unit tests (ResNet/MobileNet/SqueezeNet, toolchain error handling) ──

header "Step 2: Conversion toolchain unit tests"

pytest tests/test_conversion.py -v --tb=short
pass "Conversion unit tests passed"

# ── 3. YOLO integration tests (compiles + caches to tests/compiled_models/) ───

header "Step 3: YOLO integration tests (FP32 / FP16 / INT8)"

pytest tests/test_yolo_integration.py -v --tb=short
pass "YOLO integration tests passed"

# ── Summary ───────────────────────────────────────────────────────────────────

header "Stage 1 complete"
echo "  Compiled models saved to: tests/compiled_models/"
echo "  Run Stage 2:  bash scripts/stage2_run.sh"
