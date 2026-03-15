#!/usr/bin/env python3
"""Extract target skeleton offsets from the first valid sample in a raw .pt file.

Usage:
    python tools/extract_tgt_offsets.py \
        --raw-pt data/ee4d_train_raw_clip21_ov10.pt \
        --output preprocess/statistics/tips/tgt_offsets.pt \
        --include-fingertips
"""
import argparse
import sys
import os
import numpy as np
import torch

sys.path.append(".")
from preprocess.paramUtil import t2m_raw_offsets, t2m_body_hand_kinematic_chain
from preprocess.paramUtil_add_tips import (
    t2m_raw_offsets_with_tips,
    t2m_body_hand_kinematic_chain_with_tips,
)
from common.skeleton import Skeleton


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-pt", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--include-fingertips", action="store_true")
    ap.add_argument("--kp-field", default="kp3d")
    args = ap.parse_args()

    if args.include_fingertips:
        n_raw_offsets = torch.from_numpy(t2m_raw_offsets_with_tips).float()
        kinematic_chain = t2m_body_hand_kinematic_chain_with_tips
    else:
        n_raw_offsets = torch.from_numpy(t2m_raw_offsets).float()
        kinematic_chain = t2m_body_hand_kinematic_chain

    print(f"Loading {args.raw_pt} ...")
    db = torch.load(args.raw_pt, map_location="cpu", weights_only=False)
    print(f"  {len(db)} samples")

    for key, item in db.items():
        if not isinstance(item, dict) or args.kp_field not in item:
            continue
        kp0 = item[args.kp_field]
        if torch.is_tensor(kp0):
            kp0 = kp0.detach().cpu().numpy()
        if kp0.ndim != 3 or kp0.shape[1] < 55 or kp0.shape[2] != 3:
            continue

        if args.include_fingertips:
            pos0 = np.concatenate(
                [kp0[0, :22, :], kp0[0, 25:55, :], kp0[0, -10:, :]], axis=0
            )
        else:
            pos0 = np.concatenate([kp0[0, :22, :], kp0[0, 25:55, :]], axis=0)

        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        tgt_offsets = skel.get_offsets_joints(torch.from_numpy(pos0).float())

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        torch.save(tgt_offsets, args.output)
        print(f"  Extracted from key={key}")
        print(f"  Shape: {tgt_offsets.shape}")
        print(f"  Saved to {args.output}")
        return

    print("[ERROR] No valid sample found")
    sys.exit(1)


if __name__ == "__main__":
    main()
