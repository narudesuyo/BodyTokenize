#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"

EE4D_ROOT="${EE4D_ROOT:-/work/narus/data/ee4d/ee4d_motion_uniegomotion/uniegomotion}"
TRAIN_SRC_PT="${TRAIN_SRC_PT:-${EE4D_ROOT}/ee_train_joints_tips.pt}"
VAL_SRC_PT="${VAL_SRC_PT:-${EE4D_ROOT}/ee_val_joints_tips.pt}"

CLIP_LEN="${CLIP_LEN:-41}"
OVERLAP="${OVERLAP:-20}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"
NUM_WORKERS="${NUM_WORKERS:-6}"

INPUT_UP_AXIS="${INPUT_UP_AXIS:-z}"
BASE_IDX="${BASE_IDX:-15}"
INCLUDE_FINGERTIPS="${INCLUDE_FINGERTIPS:-1}"
HAND_LOCAL="${HAND_LOCAL:-1}"

SUFFIX="clip${CLIP_LEN}_ov${OVERLAP}"
TRAIN_RAW_PT="${TRAIN_RAW_PT:-data/ee4d_train_raw_${SUFFIX}.pt}"
VAL_RAW_PT="${VAL_RAW_PT:-data/ee4d_val_raw_${SUFFIX}.pt}"
TRAIN_CACHE_PT="${TRAIN_CACHE_PT:-data/preprocessed_motion_train_${SUFFIX}.pt}"
VAL_CACHE_PT="${VAL_CACHE_PT:-data/preprocessed_motion_val_${SUFFIX}.pt}"

if [[ ! -f "${TRAIN_SRC_PT}" ]]; then
  echo "[ERROR] train source not found: ${TRAIN_SRC_PT}" >&2
  exit 1
fi
if [[ ! -f "${VAL_SRC_PT}" ]]; then
  echo "[ERROR] validation source not found: ${VAL_SRC_PT}" >&2
  exit 1
fi

build_raw() {
  local split="$1"
  local src_pt="$2"
  local out_pt="$3"

  if [[ "${SKIP_IF_EXISTS}" == "1" && -f "${out_pt}" ]]; then
    echo "[SKIP] ${split} raw exists: ${out_pt}"
    return 0
  fi

  echo "[RAW] build ${split}: ${out_pt}"
  "${PYTHON_BIN}" preprocess/ee4d_window_to_raw_pt.py \
    --in-pt "${src_pt}" \
    --out-pt "${out_pt}" \
    --clip-len "${CLIP_LEN}" \
    --overlap "${OVERLAP}" \
    --key-prefix ee4d
}

build_cache() {
  local split="$1"
  local raw_pt="$2"
  local cache_pt="$3"

  if [[ "${SKIP_IF_EXISTS}" == "1" && -f "${cache_pt}" ]]; then
    echo "[SKIP] ${split} cache exists: ${cache_pt}"
    return 0
  fi

  echo "[CACHE] build ${split}: ${cache_pt}"
  cmd=(
    "${PYTHON_BIN}" precompute.py
    --raw-pt "${raw_pt}"
    --output-pt "${cache_pt}"
    --input-up-axis "${INPUT_UP_AXIS}"
    --base-idx "${BASE_IDX}"
    --num-workers "${NUM_WORKERS}"
  )
  if [[ "${INCLUDE_FINGERTIPS}" == "1" ]]; then
    cmd+=(--include-fingertips)
  fi
  if [[ "${HAND_LOCAL}" == "1" ]]; then
    cmd+=(--hand-local)
  fi
  "${cmd[@]}"
}

echo "[1/4] EE4D train raw windows"
build_raw "train" "${TRAIN_SRC_PT}" "${TRAIN_RAW_PT}"

echo "[2/4] EE4D train cache"
build_cache "train" "${TRAIN_RAW_PT}" "${TRAIN_CACHE_PT}"

echo "[3/4] EE4D validation raw windows"
build_raw "validation" "${VAL_SRC_PT}" "${VAL_RAW_PT}"

echo "[4/4] EE4D validation cache"
build_cache "validation" "${VAL_RAW_PT}" "${VAL_CACHE_PT}"

echo "[DONE] EE4D windowed caches:"
echo "  train_raw  : ${TRAIN_RAW_PT}"
echo "  val_raw    : ${VAL_RAW_PT}"
echo "  train_cache: ${TRAIN_CACHE_PT}"
echo "  val_cache  : ${VAL_CACHE_PT}"

