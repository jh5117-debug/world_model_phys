#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${PROJECT_ROOT}/third_party/VideoREPA"

if [[ -d "${TARGET_DIR}/.git" ]]; then
  echo "VideoREPA already exists: ${TARGET_DIR}"
  exit 0
fi

git clone --depth 1 https://github.com/aHapBean/VideoREPA.git "${TARGET_DIR}"
echo "Cloned VideoREPA -> ${TARGET_DIR}"
