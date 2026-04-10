#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${PROJECT_ROOT}/third_party/physics-IQ-benchmark"

if [[ -d "${TARGET_DIR}/.git" ]]; then
  echo "physics-IQ-benchmark already exists: ${TARGET_DIR}"
  exit 0
fi

git clone --depth 1 https://github.com/google-deepmind/physics-IQ-benchmark "${TARGET_DIR}"
echo "Cloned physics-IQ-benchmark -> ${TARGET_DIR}"
