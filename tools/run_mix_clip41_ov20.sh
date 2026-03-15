#!/usr/bin/env bash
set -euo pipefail

SKIP_PREPARE=1 \
EE4D_TRAIN_CACHE=data/preprocessed_motion_train_clip41_ov20.pt \
EE4D_VAL_CACHE=data/preprocessed_motion_val_clip41_ov20.pt \
MIX_TRAIN_CACHE=data/preprocessed_motion_train_mix_clip41_ov20.pt \
MIX_VAL_CACHE=data/preprocessed_motion_val_mix_clip41_ov20.pt \
TRAIN_CONFIG=config/motion_vqvae_mix_clip41_ov20.yaml \
bash tools/train_vq_mix.sh \
| tee runs/train_mix_clip41_ov20_$(date +%Y%m%d_%H%M%S).log
