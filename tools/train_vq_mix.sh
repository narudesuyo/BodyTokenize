#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SKIP_PREPARE="${SKIP_PREPARE:-0}"
TRAIN_CONFIG="${TRAIN_CONFIG:-config/motion_vqvae_mix.yaml}"

EE4D_TRAIN_CACHE="${EE4D_TRAIN_CACHE:-data/preprocessed_motion_train.pt}"
EE4D_VAL_CACHE="${EE4D_VAL_CACHE:-data/preprocessed_motion_val.pt}"
ASSEMBLY_TRAIN_CACHE="${ASSEMBLY_TRAIN_CACHE:-data/assembly101_train_cache.pt}"
ASSEMBLY_VAL_CACHE="${ASSEMBLY_VAL_CACHE:-data/assembly101_val_cache.pt}"

MIX_TRAIN_CACHE="${MIX_TRAIN_CACHE:-data/preprocessed_motion_train_mix.pt}"
MIX_VAL_CACHE="${MIX_VAL_CACHE:-data/preprocessed_motion_val_mix.pt}"

EE4D_PREFIX="${EE4D_PREFIX:-ee4d}"
ASSEMBLY_PREFIX="${ASSEMBLY_PREFIX:-a101}"

if [[ "${SKIP_PREPARE}" != "1" ]]; then
  bash tools/prepare_assembly101_cache.sh
fi

for f in "${EE4D_TRAIN_CACHE}" "${EE4D_VAL_CACHE}" "${ASSEMBLY_TRAIN_CACHE}" "${ASSEMBLY_VAL_CACHE}"; do
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] cache not found: ${f}" >&2
    exit 1
  fi
done

echo "[1/3] Merge train caches"
"${PYTHON_BIN}" tools/merge_motion_caches.py \
  --inputs "${EE4D_TRAIN_CACHE}" "${ASSEMBLY_TRAIN_CACHE}" \
  --prefixes "${EE4D_PREFIX}" "${ASSEMBLY_PREFIX}" \
  --output "${MIX_TRAIN_CACHE}"

echo "[2/3] Merge validation caches"
"${PYTHON_BIN}" tools/merge_motion_caches.py \
  --inputs "${EE4D_VAL_CACHE}" "${ASSEMBLY_VAL_CACHE}" \
  --prefixes "${EE4D_PREFIX}" "${ASSEMBLY_PREFIX}" \
  --output "${MIX_VAL_CACHE}"

echo "[3/3] Train mixed model"
echo "  config: ${TRAIN_CONFIG}"
"${PYTHON_BIN}" src/train/train.py --config "${TRAIN_CONFIG}" "$@"
