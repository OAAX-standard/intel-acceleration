#!/bin/bash
# check-secrets.sh — scan for secrets, proprietary data, and confidential info leaks.
# Usage:
#   bash scripts/check-secrets.sh              # scan working tree
#   bash scripts/check-secrets.sh --staged     # scan only staged files (pre-commit)
#   bash scripts/check-secrets.sh <path>       # scan specific file or directory

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

STAGED_ONLY=false
SCAN_TARGET="${REPO_ROOT}"
FINDINGS=0

# ── Argument parsing ──────────────────────────────────────────────────────────
if [[ "${1:-}" == "--staged" ]]; then
    STAGED_ONLY=true
elif [[ -n "${1:-}" ]]; then
    SCAN_TARGET="$1"
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; FINDINGS=$((FINDINGS + 1)); }
fail() { echo -e "${RED}[FAIL]${NC} $*"; FINDINGS=$((FINDINGS + 1)); }
info() { echo -e "  $*"; }

# Directories excluded from all pattern scans.
# deps/ contains vendored third-party code with its own history.
# .claude/ contains internal agent documentation.
EXCLUDE_DIRS=("runtime-library/deps" ".claude")

# Get the list of files to scan (excludes EXCLUDE_DIRS).
get_files() {
    local all
    if $STAGED_ONLY; then
        all=$(git -C "${REPO_ROOT}" diff --cached --name-only --diff-filter=ACM)
    else
        all=$(git -C "${REPO_ROOT}" ls-files -- "${SCAN_TARGET}" 2>/dev/null \
            || find "${SCAN_TARGET}" -type f -not -path "*/.git/*")
    fi

    local filtered="$all"
    for dir in "${EXCLUDE_DIRS[@]}"; do
        filtered=$(echo "$filtered" | grep -v "^${dir}/" || true)
    done
    echo "$filtered"
}

# Run grep on collected files; print matching lines with file:line context.
scan_pattern() {
    local label="$1"
    local pattern="$2"
    local files_list
    files_list=$(get_files)

    if [[ -z "$files_list" ]]; then return; fi

    local matches
    matches=$(echo "$files_list" | \
        xargs grep -rn --with-filename -iE "$pattern" 2>/dev/null \
        | grep -v "check-secrets-ignore" \
        | grep -v "nosec" \
        | grep -v "# nosec" \
        | grep -v "EXAMPLE\|example_\|dummy\|fake_\|test_fixture\|placeholder" \
        || true)

    if [[ -n "$matches" ]]; then
        fail "$label"
        echo "$matches" | while IFS= read -r line; do info "$line"; done
    fi
}

echo "=== Secret & Confidentiality Scan ==="
if $STAGED_ONLY; then
    echo "Mode: staged files only"
else
    echo "Target: ${SCAN_TARGET}"
fi
echo ""

# ── 1. Hardcoded credentials ──────────────────────────────────────────────────
echo "[1/5] Checking for hardcoded credentials..."

# AWS-style access key IDs (20 uppercase alphanumeric chars starting with AKIA)
scan_pattern "AWS access key ID" \
    'AKIA[0-9A-Z]{16}'

# Generic secret/password/token assignments with a non-placeholder value
scan_pattern "Secret/password/token assignment" \
    '(secret|password|passwd|api_key|apikey|auth_token|access_token)\s*[=:]\s*["\x27][^"\x27$\{][^"\x27]{7,}'

# Private key headers
scan_pattern "Private key material" \
    'BEGIN (RSA |EC |OPENSSH |DSA )?PRIVATE KEY'

# ── 2. Project-specific sensitive values ──────────────────────────────────────
echo "[2/5] Checking for project-specific sensitive values..."

# S3 endpoint hostname — acceptable as a pattern reference but flag if next to credentials
scan_pattern "S3 endpoint URL hardcoded" \
    'nbg1\.your-objectstorage\.com'

# Literal s3cfg access_key or secret_key values (not the GitHub secret variable references)
scan_pattern "s3cfg credential assignment" \
    '^(access_key|secret_key)\s*=\s*[A-Za-z0-9+/]{20,}'

# ── 3. Sensitive filenames tracked by git ─────────────────────────────────────
echo "[3/5] Checking for sensitive files tracked in git..."

sensitive_file_patterns=(
    "\.env$"
    "\.env\."
    "^secrets\."            # file literally named secrets.something (not check-secrets.sh)
    "^credentials\."
    "\.pem$"
    "\.key$"
    "\.p12$"
    "\.pfx$"
    "s3cfg$"
    "\.s3cfg"
    "htpasswd"
    "id_rsa"
    "id_ed25519"
    "id_ecdsa"
)

tracked_files=$(git -C "${REPO_ROOT}" ls-files 2>/dev/null || true)
for pat in "${sensitive_file_patterns[@]}"; do
    matches=$(echo "$tracked_files" | grep -iE "$pat" || true)
    if [[ -n "$matches" ]]; then
        fail "Sensitive filename tracked by git (pattern: $pat)"
        echo "$matches" | while IFS= read -r f; do info "  $f"; done
    fi
done

# ── 4. Large binary / model weight files ──────────────────────────────────────
echo "[4/5] Checking for large binary files..."

# Files over 1 MB tracked in git that look like model weights
large_threshold=1048576  # 1 MB in bytes
model_extensions="\.pt$|\.pth$|\.onnx$|\.pb$|\.h5$|\.weights$|\.bin$"

while IFS= read -r tracked_file; do
    # Skip vendored deps — they contain third-party test data, not our model weights
    _skip=false
    for dir in "${EXCLUDE_DIRS[@]}"; do
        if [[ "$tracked_file" == ${dir}/* ]]; then _skip=true; break; fi
    done
    $_skip && continue

    abs_path="${REPO_ROOT}/${tracked_file}"
    if [[ -f "$abs_path" ]]; then
        size=$(stat -c%s "$abs_path" 2>/dev/null || stat -f%z "$abs_path" 2>/dev/null || echo 0)
        if echo "$tracked_file" | grep -qiE "$model_extensions" && \
           [[ "$size" -gt "$large_threshold" ]]; then
            warn "Large model/binary file tracked in git: ${tracked_file} ($(( size / 1024 ))KB)"
        fi
    fi
done < <(git -C "${REPO_ROOT}" ls-files 2>/dev/null || true)

# ── 5. GitHub Actions secret references used correctly ───────────────────────
echo "[5/5] Checking GitHub Actions workflows..."

# Flag any workflow that assigns a secret to an env var with a literal value
# (i.e. value is not the ${{ secrets.* }} syntax)
workflow_dir="${REPO_ROOT}/.github/workflows"
if [[ -d "$workflow_dir" ]]; then
    matches=$(grep -rn --with-filename -E \
        '(ACCESS_KEY|SECRET_KEY|API_KEY|PASSWORD|TOKEN)\s*:\s*["\x27]?[A-Za-z0-9+/]{16,}' \
        "$workflow_dir" 2>/dev/null \
        | grep -v "\${{" \
        | grep -v "check-secrets-ignore" \
        || true)
    if [[ -n "$matches" ]]; then
        fail "Hardcoded credential in GitHub Actions workflow"
        echo "$matches" | while IFS= read -r line; do info "$line"; done
    fi
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
if [[ "$FINDINGS" -eq 0 ]]; then
    echo -e "${GREEN}✓ No issues found.${NC}"
    exit 0
else
    echo -e "${RED}✗ ${FINDINGS} issue(s) found. Review the output above.${NC}"
    echo ""
    echo "If a match is a known false positive, suppress it with:"
    echo "  # nosec: <reason>              (same line)"
    echo "  # check-secrets-ignore: <reason>"
    echo ""
    echo "If a real secret was committed, rotate it immediately before cleaning history."
    exit 1
fi
