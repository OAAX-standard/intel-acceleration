#!/usr/bin/env bash
# Upload the Intel runtimes and the conversion toolchain to the OAAX bucket.
#
#   scripts/publish-artifacts.sh release <version> <artifacts-dir>   # on a v* tag
#   scripts/publish-artifacts.sh branch  <branch>  <artifacts-dir>   # on a pull request, for testing
#   scripts/publish-artifacts.sh dry     <name>    <artifacts-dir>   # only check the files and print the keys
#
# <artifacts-dir> holds the CI artifacts, one directory per artifact name (what actions/download-artifact creates).
# S3_ACCESS_KEY, S3_SECRET_KEY and S3_ENDPOINT_URL must be set for release and branch.
#
# release: keys under <version>/ are written once (if any exists nothing is uploaded: a published version is
#          immutable, oaax-manager pins its sha256); latest/ follows the newest release.
# branch:  keys under <branch>/ are overwritten on every run, latest/ is never touched, and a name that could be a
#          release (latest, or starting with a digit or v<digit>) is refused so a branch can never replace one.
set -euo pipefail
shopt -u patsub_replacement 2>/dev/null || true  # bash >= 5.2: '&' in a branch name must not expand to '@'

mode=${1:?usage: publish-artifacts.sh release|branch|dry <name> <artifacts-dir>}
name=${2:?usage: publish-artifacts.sh release|branch|dry <name> <artifacts-dir>}
dir=${3:?usage: publish-artifacts.sh release|branch|dry <name> <artifacts-dir>}

case $mode in
  release | dry) ;;
  branch)
    if [[ $name =~ ^(latest|v?[0-9]) ]]; then
      echo "branch name '$name' looks like a release, rename the branch to upload its artifacts" >&2
      exit 1
    fi
    ;;
  *) echo "unknown mode: $mode" >&2; exit 1 ;;
esac

# local file -> key suffix after "<runtimes|conversion-toolchain>/<version>/"
declare -A files=(
  ["$dir/build-runtime-linux-x86_64/runtime-library-X86_64.tar.gz"]="runtimes/@/INTEL/x86_64/Ubuntu/library.tar.gz"
  ["$dir/build-runtime-windows-x86_64/runtime-library-X86_64-Windows.tar.gz"]="runtimes/@/INTEL/x86_64/Windows/library.tar.gz"
  ["$dir/build-toolchain/oaax-intel-toolchain.tar"]="conversion-toolchain/@/INTEL/oaax-intel-toolchain.tar"
)

for f in "${!files[@]}"; do
  [ -s "$f" ] || { echo "missing or empty artifact: $f" >&2; exit 1; }
done

if [ "$mode" = dry ]; then
  for f in "${!files[@]}"; do
    echo "would upload $f -> s3://oaax/${files[$f]//@/$name}"
  done
  exit 0
fi

s3() {
  s3cmd --access_key="$S3_ACCESS_KEY" --secret_key="$S3_SECRET_KEY" \
    --host="$S3_ENDPOINT_URL" --host-bucket="%(bucket)s.$S3_ENDPOINT_URL" --bucket-location=us-east-1 --ssl "$@"
}

if [ "$mode" = release ]; then
  # s3cmd info: 0 = exists, 12 = not found, anything else = a real error (credentials, network)
  for f in "${!files[@]}"; do
    url="s3://oaax/${files[$f]//@/$name}"
    rc=0
    s3 info "$url" >/dev/null 2>&1 || rc=$?
    case $rc in
      0) echo "$url already exists: versions are immutable, bump VERSION for a new release" >&2; exit 1 ;;
      12) ;;
      *) echo "could not check $url (s3cmd exit $rc)" >&2; exit 1 ;;
    esac
  done
fi

for f in "${!files[@]}"; do
  s3 put "$f" "s3://oaax/${files[$f]//@/$name}"
  if [ "$mode" = release ]; then s3 put "$f" "s3://oaax/${files[$f]//@/latest}"; fi
done
