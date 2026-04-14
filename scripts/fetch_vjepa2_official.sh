#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="${CODE_DIR:-${PROJECT_ROOT}/third_party/vjepa2_official}"
WEIGHT_DIR="${WEIGHT_DIR:-${PROJECT_ROOT}/../weight/vjepa2_1}"
SOURCE_REF="${SOURCE_REF:-main}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-vjepa2_1_vitb_dist_vitG_384.pt}"
SOURCE_URL="${SOURCE_URL:-https://codeload.github.com/facebookresearch/vjepa2/tar.gz/refs/heads/${SOURCE_REF}}"
CHECKPOINT_URL="${CHECKPOINT_URL:-https://dl.fbaipublicfiles.com/vjepa2/${CHECKPOINT_NAME}}"

if [[ "${ENABLE_H20_PROXY:-0}" == "1" ]]; then
  if [[ -f /home/nvme01/clash-for-linux/start.sh ]]; then
    bash /home/nvme01/clash-for-linux/start.sh
  fi
  if [[ -f /home/nvme01/clash-for-linux/clash.sh ]]; then
    # shellcheck disable=SC1091
    source /home/nvme01/clash-for-linux/clash.sh
    proxy_on
  fi
fi

TMP_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "${TMP_DIR}"
}
trap cleanup EXIT

mkdir -p "${WEIGHT_DIR}"

ARCHIVE_PATH="${TMP_DIR}/vjepa2.tar.gz"
EXTRACT_DIR="${TMP_DIR}/vjepa2-main"

download_with_curl() {
  local url="$1"
  local output="$2"
  curl \
    --fail \
    --location \
    --http1.1 \
    --retry 5 \
    --retry-delay 2 \
    --retry-all-errors \
    --output "${output}" \
    "${url}"
}

download_with_wget() {
  local url="$1"
  local output="$2"
  wget -O "${output}" --tries=5 --waitretry=2 "${url}"
}

download_source_archive() {
  local output="$1"
  if command -v curl >/dev/null 2>&1; then
    if download_with_curl "${SOURCE_URL}" "${output}"; then
      return 0
    fi
    echo "[WARN] curl archive download failed, trying wget" >&2
  fi
  if command -v wget >/dev/null 2>&1; then
    if download_with_wget "${SOURCE_URL}" "${output}"; then
      return 0
    fi
    echo "[WARN] wget archive download failed, trying shallow git clone fallback" >&2
  fi
  if command -v git >/dev/null 2>&1; then
    git clone --depth 1 --branch "${SOURCE_REF}" https://github.com/facebookresearch/vjepa2.git "${TMP_DIR}/vjepa2-main"
    return 0
  fi
  return 1
}

download_checkpoint() {
  local output="$1"
  if command -v curl >/dev/null 2>&1; then
    if download_with_curl "${CHECKPOINT_URL}" "${output}"; then
      return 0
    fi
    echo "[WARN] curl checkpoint download failed, trying wget" >&2
  fi
  if command -v wget >/dev/null 2>&1; then
    download_with_wget "${CHECKPOINT_URL}" "${output}"
    return 0
  fi
  return 1
}

echo "[FETCH] Downloading official V-JEPA 2.1 source archive from ${SOURCE_URL}"
if [[ ! -d "${TMP_DIR}/vjepa2-main" ]]; then
  if ! download_source_archive "${ARCHIVE_PATH}"; then
    echo "[ERROR] Failed to download the official V-JEPA 2.1 source tree" >&2
    exit 1
  fi
fi

if [[ -f "${ARCHIVE_PATH}" ]]; then
  tar -xzf "${ARCHIVE_PATH}" -C "${TMP_DIR}"
fi

if [[ ! -f "${EXTRACT_DIR}/hubconf.py" ]]; then
  echo "[ERROR] Downloaded archive does not look like the official facebookresearch/vjepa2 source tree" >&2
  exit 1
fi

rm -rf "${CODE_DIR}"
mkdir -p "$(dirname "${CODE_DIR}")"
mv "${EXTRACT_DIR}" "${CODE_DIR}"

echo "[FETCH] Downloading official V-JEPA 2.1 checkpoint ${CHECKPOINT_NAME}"
if ! download_checkpoint "${WEIGHT_DIR}/${CHECKPOINT_NAME}"; then
  echo "[ERROR] Failed to download the official V-JEPA 2.1 checkpoint" >&2
  exit 1
fi

echo "[DONE] Official source: ${CODE_DIR}"
echo "[DONE] Official checkpoint: ${WEIGHT_DIR}/${CHECKPOINT_NAME}"
