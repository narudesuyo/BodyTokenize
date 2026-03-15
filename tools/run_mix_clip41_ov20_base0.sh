#!/usr/bin/env bash
set -euo pipefail

SKIP_PREPARE=1 \
EE4D_TRAIN_CACHE=data/preprocessed_motion_train_clip41_ov20_base0.pt \
EE4D_VAL_CACHE=data/preprocessed_motion_val_clip41_ov20_base0.pt \
ASSEMBLY_TRAIN_CACHE=data/assembly101_train_cache_base0.pt \
ASSEMBLY_VAL_CACHE=data/assembly101_val_cache_base0.pt \
MIX_TRAIN_CACHE=data/preprocessed_motion_train_mix_clip41_ov20_base0.pt \
MIX_VAL_CACHE=data/preprocessed_motion_val_mix_clip41_ov20_base0.pt \
TRAIN_CONFIG=config/motion_vqvae_mix_clip41_ov20_base0.yaml \
bash tools/train_vq_mix.sh \
| tee runs/train_mix_clip41_ov20_base0_$(date +%Y%m%d_%H%M%S).log
