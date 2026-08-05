#!/usr/bin/env bash
# Creates a self-contained stage2 package for running on a remote machine.
# Output: stage2_package.tar.gz
#
# Package layout mirrors the paths stage2.py expects relative to ROOT:
#   tests/stage2.py
#   tests/compiled_models/<model>/<variant>/<model>.[xml,bin,zip]
#   tests/runtime/build/yolo_test  (+ all .so files for LD_LIBRARY_PATH)
#
# On the remote machine:
#   tar xzf stage2_package.tar.gz
#   cd stage2_package
#   python3 tests/stage2.py --devices CPU [--csv results.csv]

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPILED_DIR="$ROOT/tests/compiled_models"
TEST_BUILD_DIR="$ROOT/tests/runtime/build"
OUT_DIR="$ROOT/stage2_package"
ARCHIVE="$ROOT/stage2_package.tar.gz"

# ── Sanity checks ──────────────────────────────────────────────────────────────

if [ ! -f "$TEST_BUILD_DIR/yolo_test" ]; then
    echo "ERROR: yolo_test binary not found at $TEST_BUILD_DIR/yolo_test"
    echo "       Run: bash tests/runtime/build-tests.sh"
    exit 1
fi

if [ ! -d "$COMPILED_DIR" ] || [ -z "$(find "$COMPILED_DIR" -name '*.xml' 2>/dev/null)" ]; then
    echo "ERROR: No compiled models found in $COMPILED_DIR"
    echo "       Run: python tests/stage1.py"
    exit 1
fi

# ── Build package directory ────────────────────────────────────────────────────

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/tests/runtime/build"
mkdir -p "$OUT_DIR/runtime-library/build"  # placeholder so ROOT paths resolve

echo "Copying stage2.py..."
cp "$ROOT/tests/stage2.py" "$OUT_DIR/tests/stage2.py"

echo "Copying compiled models..."
cp -r "$COMPILED_DIR" "$OUT_DIR/tests/compiled_models"

echo "Copying yolo_test binary..."
cp "$TEST_BUILD_DIR/yolo_test" "$OUT_DIR/tests/runtime/build/yolo_test"
chmod +x "$OUT_DIR/tests/runtime/build/yolo_test"

echo "Copying shared libraries..."
# Resolve symlinks so the package is self-contained
find "$TEST_BUILD_DIR" -maxdepth 1 \( -name "*.so" -o -name "*.so.*" \) | while read -r lib; do
    real="$(realpath "$lib")"
    dest="$OUT_DIR/tests/runtime/build/$(basename "$lib")"
    cp "$real" "$dest"
done

# ── Archive ────────────────────────────────────────────────────────────────────

echo "Creating archive $ARCHIVE..."
rm "$ARCHIVE" || true
tar czf "$ARCHIVE" -C "$ROOT" stage2_package

SIZE=$(du -sh "$ARCHIVE" | cut -f1)
echo ""
echo "Done: stage2_package.tar.gz ($SIZE)"
echo ""
echo "To run on the remote machine:"
echo "  scp stage2_package.tar.gz user@remote:~/"
echo "  ssh user@remote"
echo "  tar xzf stage2_package.tar.gz && cd stage2_package"
echo "  python3 tests/stage2.py --devices CPU [--csv results.csv]"
