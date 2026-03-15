#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
ASSEMBLY_ROOT="${ASSEMBLY_ROOT:-/work/narus/data/Assembly101}"
MOTION_ROOT="${MOTION_ROOT:-${ASSEMBLY_ROOT}/motion/v1}"
TEXT_ROOT="${TEXT_ROOT:-${ASSEMBLY_ROOT}/text/v2}"
SMPLH_MODEL_DIR="${SMPLH_MODEL_DIR:-/home/narus/2026/EgoHand/BodyTokenize/models/smplx/smplh}"
GENDER="${GENDER:-male}"
DEVICE="${DEVICE:-cuda:0}"
CLIP_LEN="${CLIP_LEN:-41}"
OVERLAP="${OVERLAP:-20}"
MAX_TAKES="${MAX_TAKES:--1}"
MAX_CLIPS="${MAX_CLIPS:--1}"
DOWNSAMPLE="${DOWNSAMPLE:-3}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"
BASE_IDX="${BASE_IDX:-15}"

TRAIN_RAW_PT="${TRAIN_RAW_PT:-data/assembly101_train_raw.pt}"
VAL_RAW_PT="${VAL_RAW_PT:-data/assembly101_val_raw.pt}"
TRAIN_CACHE_PT="${TRAIN_CACHE_PT:-data/assembly101_train_cache.pt}"
VAL_CACHE_PT="${VAL_CACHE_PT:-data/assembly101_val_cache.pt}"

if [[ ! -d "${MOTION_ROOT}" ]]; then
  echo "[ERROR] motion root not found: ${MOTION_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${TEXT_ROOT}" ]]; then
  echo "[ERROR] text root not found: ${TEXT_ROOT}" >&2
  exit 1
fi

if [[ "${SKIP_IF_EXISTS}" == "1" && -f "${TRAIN_CACHE_PT}" ]]; then
  echo "[SKIP] train cache exists: ${TRAIN_CACHE_PT}"
else
  if [[ "${SKIP_IF_EXISTS}" == "1" && -f "${TRAIN_RAW_PT}" ]]; then
    echo "[SKIP] train raw exists: ${TRAIN_RAW_PT}"
  else
    echo "[1/4] Build Assembly101 train raw pt"
    "${PYTHON_BIN}" preprocess/assembly101_motion_to_raw_pt.py \
      --motion-root "${MOTION_ROOT}" \
      --text-root "${TEXT_ROOT}" \
      --split train \
      --clip-len "${CLIP_LEN}" \
      --overlap "${OVERLAP}" \
      --model-type smplh \
      --model-dir "${SMPLH_MODEL_DIR}" \
      --gender "${GENDER}" \
      --device "${DEVICE}" \
      --max-takes "${MAX_TAKES}" \
      --max-clips "${MAX_CLIPS}" \
      --downsample "${DOWNSAMPLE}" \
      --out-pt "${TRAIN_RAW_PT}"
  fi

  echo "[3/4] Precompute train cache"
  "${PYTHON_BIN}" precompute.py \
    --raw-pt "${TRAIN_RAW_PT}" \
    --output-pt "${TRAIN_CACHE_PT}" \
    --include-fingertips \
    --input-up-axis y \
    --base-idx "${BASE_IDX}" \
    --hand-local
fi

if [[ "${SKIP_IF_EXISTS}" == "1" && -f "${VAL_CACHE_PT}" ]]; then
  echo "[SKIP] validation cache exists: ${VAL_CACHE_PT}"
else
  if [[ "${SKIP_IF_EXISTS}" == "1" && -f "${VAL_RAW_PT}" ]]; then
    echo "[SKIP] validation raw exists: ${VAL_RAW_PT}"
  else
    echo "[2/4] Build Assembly101 validation raw pt"
    "${PYTHON_BIN}" preprocess/assembly101_motion_to_raw_pt.py \
      --motion-root "${MOTION_ROOT}" \
      --text-root "${TEXT_ROOT}" \
      --split validation \
      --clip-len "${CLIP_LEN}" \
      --overlap "${OVERLAP}" \
      --model-type smplh \
      --model-dir "${SMPLH_MODEL_DIR}" \
      --gender "${GENDER}" \
      --device "${DEVICE}" \
      --max-takes "${MAX_TAKES}" \
      --max-clips "${MAX_CLIPS}" \
      --downsample "${DOWNSAMPLE}" \
      --out-pt "${VAL_RAW_PT}"
  fi

  echo "[4/4] Precompute validation cache"
  "${PYTHON_BIN}" precompute.py \
    --raw-pt "${VAL_RAW_PT}" \
    --output-pt "${VAL_CACHE_PT}" \
    --include-fingertips \
    --input-up-axis y \
    --base-idx "${BASE_IDX}" \
    --hand-local
fi

echo "[DONE] Assembly101 caches:"
echo "  train: ${TRAIN_CACHE_PT}"
echo "  val  : ${VAL_CACHE_PT}"
