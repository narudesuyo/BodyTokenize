#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/motion_vqvae_assembly101.yaml}"

TRAIN_CACHE_PT="${TRAIN_CACHE_PT:-data/assembly101_train_cache.pt}"
VAL_CACHE_PT="${VAL_CACHE_PT:-data/assembly101_val_cache.pt}"

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  bash tools/prepare_assembly101_cache.sh
fi

if [[ ! -f "${TRAIN_CACHE_PT}" ]]; then
  echo "[ERROR] train cache not found: ${TRAIN_CACHE_PT}" >&2
  exit 1
fi
if [[ ! -f "${VAL_CACHE_PT}" ]]; then
  echo "[ERROR] val cache not found: ${VAL_CACHE_PT}" >&2
  exit 1
fi

echo "[TRAIN] Assembly101 only"
echo "  config: ${TRAIN_CONFIG}"
"${PYTHON_BIN}" src/train/train.py --config "${TRAIN_CONFIG}" "$@"
