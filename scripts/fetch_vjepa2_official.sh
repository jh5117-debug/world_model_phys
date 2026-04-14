#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_DIR="${CODE_DIR:-${PROJECT_ROOT}/third_party/vjepa2_official}"
WEIGHT_DIR="${WEIGHT_DIR:-${PROJECT_ROOT}/../weight/vjepa2_1}"
SOURCE_REF="${SOURCE_REF:-main}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-vjepa2_1_vitb_dist_vitG_384.pt}"
SOURCE_URL="${SOURCE_URL:-https://github.com/facebookresearch/vjepa2/archive/refs/heads/${SOURCE_REF}.tar.gz}"
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

echo "[FETCH] Downloading official V-JEPA 2.1 source archive from ${SOURCE_URL}"
curl --fail --location --retry 3 --output "${ARCHIVE_PATH}" "${SOURCE_URL}"
tar -xzf "${ARCHIVE_PATH}" -C "${TMP_DIR}"

if [[ ! -f "${EXTRACT_DIR}/hubconf.py" ]]; then
  echo "[ERROR] Downloaded archive does not look like the official facebookresearch/vjepa2 source tree" >&2
  exit 1
fi

rm -rf "${CODE_DIR}"
mkdir -p "$(dirname "${CODE_DIR}")"
mv "${EXTRACT_DIR}" "${CODE_DIR}"

echo "[FETCH] Downloading official V-JEPA 2.1 checkpoint ${CHECKPOINT_NAME}"
curl --fail --location --retry 3 --output "${WEIGHT_DIR}/${CHECKPOINT_NAME}" "${CHECKPOINT_URL}"

echo "[DONE] Official source: ${CODE_DIR}"
echo "[DONE] Official checkpoint: ${WEIGHT_DIR}/${CHECKPOINT_NAME}"
