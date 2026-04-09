#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${PROJECT_ROOT}/third_party/videophy"

if [[ -d "${TARGET_DIR}/.git" ]]; then
  echo "videophy already exists: ${TARGET_DIR}"
  exit 0
fi

git clone --depth 1 https://github.com/Hritikbansal/videophy.git "${TARGET_DIR}"
echo "Cloned videophy -> ${TARGET_DIR}"
