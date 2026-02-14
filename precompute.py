#!/usr/bin/env python3
"""
Precompute Motion Representation from raw kp3d data
Converts kp3d → body(263) + hand(480) representation once
Saves to new pt file for fast loading
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path
import argparse
from tqdm import tqdm

# Adjust path as needed
sys.path.append(".")

from preprocess.paramUtil import t2m_raw_offsets, t2m_body_hand_kinematic_chain
from preprocess.paramUtil_add_tips import t2m_raw_offsets_with_tips, t2m_body_hand_kinematic_chain_with_tips
from common.skeleton import Skeleton
from src.dataset.kp3d2motion_rep import kp3d_to_motion_rep


class MotionPrecomputer:
    def __init__(
        self,
        include_fingertips: bool = False,
        base_idx: int = 0,
        hand_local: bool = False,
        feet_thre: float = 0.002,
        kp_field: str = "kp3d",
        verbose: bool = True,
    ):
        self.include_fingertips = include_fingertips
        self.base_idx = base_idx
        self.hand_local = hand_local
        self.feet_thre = feet_thre
        self.kp_field = kp_field
        self.verbose = verbose

        # ---- 623 block boundaries ----
        if self.include_fingertips:
            self.NO_ROOT_J = 61
        else:
            self.NO_ROOT_J = 51

        self.I_ROOT0 = 0
        self.I_ROOT1 = 4
        self.I_RIC0 = self.I_ROOT1
        self.I_RIC1 = self.I_RIC0 + self.NO_ROOT_J * 3
        self.I_ROT0 = self.I_RIC1
        self.I_ROT1 = self.I_ROT0 + self.NO_ROOT_J * 6
        self.I_VEL0 = self.I_ROT1
        self.I_VEL1 = self.I_VEL0 + (self.NO_ROOT_J + 1) * 3
        self.I_FEET0 = self.I_VEL1
        self.I_FEET1 = self.I_FEET0 + 4

        # ---- skeleton ----
        self.n_raw_offsets = (
            torch.from_numpy(t2m_raw_offsets_with_tips).float()
            if include_fingertips
            else torch.from_numpy(t2m_raw_offsets).float()
        )
        self.kinematic_chain = (
            t2m_body_hand_kinematic_chain_with_tips
            if include_fingertips
            else t2m_body_hand_kinematic_chain
        )

        # ---- target offsets: will be set on first valid data ----
        self.tgt_offsets = None

    def find_target_offsets(self, db, kp_field="kp3d"):
        """Find first valid clip to extract target offsets"""
        for key, item in db.items():
            if not isinstance(item, dict) or kp_field not in item:
                continue

            kp0 = item[kp_field]
            if torch.is_tensor(kp0):
                kp0 = kp0.detach().cpu().numpy()

            if kp0.ndim != 3 or kp0.shape[1] < 52 or kp0.shape[2] != 3:
                continue

            # Extract positions
            if self.include_fingertips:
                pos0 = np.concatenate(
                    [kp0[0, :22, :], kp0[0, 25:55, :], kp0[0, -10:, :]], axis=0
                )
            else:
                pos0 = np.concatenate([kp0[0, :22, :], kp0[0, 25:55, :]], axis=0)

            tgt_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, "cpu")
            self.tgt_offsets = tgt_skel.get_offsets_joints(
                torch.from_numpy(pos0).float()
            )
            return True

        return False

    def process_single(self, kp3d_sequence):
        """
        Process single kp3d sequence
        Args:
            kp3d_sequence: [T, 52, 3] or similar
        Returns:
            body: [T, 263], hand: [T, 480] (or appropriate dims)
        """
        if torch.is_tensor(kp3d_sequence):
            kp3d_sequence = kp3d_sequence.numpy()

        # Validate shape
        if kp3d_sequence.ndim != 3 or kp3d_sequence.shape[2] != 3:
            return None, None

        # Extract 52 joints
        if self.include_fingertips:
            if kp3d_sequence.shape[1] < 55 or kp3d_sequence.shape[1] < 65:
                return None, None
            kp52 = np.concatenate(
                [
                    kp3d_sequence[:, :22, :],
                    kp3d_sequence[:, 25:55, :],
                    kp3d_sequence[:, -10:, :],
                ],
                axis=1,
            )
        else:
            if kp3d_sequence.shape[1] < 55:
                return None, None
            kp52 = np.concatenate(
                [kp3d_sequence[:, :22, :], kp3d_sequence[:, 25:55, :]], axis=1
            )

        # Check NaN
        if np.isnan(kp52).any() or np.isinf(kp52).any():
            return None, None

        # Convert to motion representation
        try:
            arr = kp3d_to_motion_rep(
                kp3d_52_yup=kp52,
                feet_thre=self.feet_thre,
                tgt_offsets=self.tgt_offsets,
                n_raw_offsets=self.n_raw_offsets,
                kinematic_chain=self.kinematic_chain,
                base_idx=self.base_idx,
                hand_local=self.hand_local,
            )
        except Exception as e:
            if self.verbose:
                print(f"  [ERROR] kp3d_to_motion_rep failed: {e}")
            return None, None

        # Check NaN in output
        if np.isnan(arr).any() or np.isinf(arr).any():
            if self.verbose:
                print(f"  [ERROR] Output contains NaN/Inf")
            return None, None

        # Split into body/hand
        Tm1 = arr.shape[0]

        # Extract components
        root = arr[:, self.I_ROOT0 : self.I_ROOT1]  # [T, 4]
        ric = arr[:, self.I_RIC0 : self.I_RIC1].reshape(Tm1, self.NO_ROOT_J, 3)
        rot = arr[:, self.I_ROT0 : self.I_ROT1].reshape(Tm1, self.NO_ROOT_J, 6)
        vel = arr[:, self.I_VEL0 : self.I_VEL1].reshape(
            Tm1, self.NO_ROOT_J + 1, 3
        )
        feet = arr[:, self.I_FEET0 : self.I_FEET1]  # [T, 4]

        # Split body/hand
        if self.include_fingertips:
            ric_body, ric_hand = ric[:, :21], ric[:, 21:61]
            rot_body, rot_hand = rot[:, :21], rot[:, 21:61]
            vel_body, vel_hand = vel[:, :22], vel[:, 22:62]
        else:
            ric_body, ric_hand = ric[:, :21], ric[:, 21:51]
            rot_body, rot_hand = rot[:, :21], rot[:, 21:51]
            vel_body, vel_hand = vel[:, :22], vel[:, 22:52]

        # Concatenate
        body = np.concatenate(
            [
                root,
                ric_body.reshape(Tm1, -1),
                rot_body.reshape(Tm1, -1),
                vel_body.reshape(Tm1, -1),
                feet,
            ],
            axis=1,
        )  # [T, 263]

        hand = np.concatenate(
            [
                ric_hand.reshape(Tm1, -1),
                rot_hand.reshape(Tm1, -1),
                vel_hand.reshape(Tm1, -1),
            ],
            axis=1,
        )  # [T, 480]

        # Final NaN check
        if np.isnan(body).any() or np.isinf(body).any():
            if self.verbose:
                print(f"  [ERROR] Body contains NaN/Inf")
            return None, None

        if np.isnan(hand).any() or np.isinf(hand).any():
            if self.verbose:
                print(f"  [ERROR] Hand contains NaN/Inf")
            return None, None

        return body, hand

    def precompute(
        self,
        raw_pt_path: str,
        output_pt_path: str,
        skip_errors: bool = True,
    ):
        """
        Precompute motion representation for all samples
        Args:
            raw_pt_path: Path to raw pt file with kp3d
            output_pt_path: Path to save precomputed pt file
            skip_errors: If True, skip samples with errors; if False, raise
        """
        print(f"Loading raw data from {raw_pt_path}...")
        db_raw = torch.load(raw_pt_path, map_location="cpu", weights_only=False)
        print(f"  Loaded {len(db_raw)} samples")

        # Find target offsets
        print("Finding target offsets...")
        if not self.find_target_offsets(db_raw, self.kp_field):
            raise RuntimeError(
                f"Could not find valid {self.kp_field} in database"
            )
        print(f"  Found target offsets")

        # Process each sample
        db_processed = {}
        n_success = 0
        n_fail = 0
        n_skip = 0

        print(f"Processing {len(db_raw)} samples...")
        for key in tqdm(db_raw.keys(), desc="Precomputing"):
            item = db_raw[key]

            # Validate structure
            if not isinstance(item, dict):
                n_skip += 1
                continue

            if self.kp_field not in item:
                n_skip += 1
                continue

            kp3d = item[self.kp_field]

            # Process
            try:
                body, hand = self.process_single(kp3d)

                if body is None or hand is None:
                    n_fail += 1
                    if not skip_errors:
                        raise ValueError(f"Processing returned None for key={key}")
                    continue

                # Save
                db_processed[key] = {
                    "body": torch.from_numpy(body).float(),
                    "hand": torch.from_numpy(hand).float(),
                    "T": int(body.shape[0]),
                    "key": key,
                }
                n_success += 1

            except Exception as e:
                n_fail += 1
                if not skip_errors:
                    print(f"[ERROR] key={key}: {e}")
                    raise
                continue

        # Save
        print(f"\nProcessing complete:")
        print(f"  Success: {n_success}")
        print(f"  Failed:  {n_fail}")
        print(f"  Skipped: {n_skip}")

        print(f"\nSaving to {output_pt_path}...")
        os.makedirs(os.path.dirname(output_pt_path), exist_ok=True)
        torch.save(db_processed, output_pt_path)
        print(f"  Saved {len(db_processed)} samples")

        # Stats
        if db_processed:
            sample_body = db_processed[list(db_processed.keys())[0]]["body"]
            sample_hand = db_processed[list(db_processed.keys())[0]]["hand"]
            print(f"\nOutput shapes:")
            print(f"  Body: {sample_body.shape}")
            print(f"  Hand: {sample_hand.shape}")

        return output_pt_path


def main():
    parser = argparse.ArgumentParser(
        description="Precompute motion representation from raw kp3d"
    )
    parser.add_argument(
        "--raw-pt",
        type=str,
        required=True,
        help="Path to raw pt file with kp3d",
    )
    parser.add_argument(
        "--output-pt",
        type=str,
        required=True,
        help="Path to save precomputed pt file",
    )
    parser.add_argument(
        "--include-fingertips",
        action="store_true",
        help="Include fingertip joints",
    )
    parser.add_argument(
        "--base-idx",
        type=int,
        default=15,
        help="Base index for motion representation",
    )
    parser.add_argument(
        "--hand-local",
        action="store_true",
        help="Use hand-local representation",
    )
    parser.add_argument(
        "--feet-thre",
        type=float,
        default=0.002,
        help="Feet threshold",
    )
    parser.add_argument(
        "--kp-field",
        type=str,
        default="kp3d",
        help="Field name for keypoints in database",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        default=True,
        help="Skip samples with errors",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )

    args = parser.parse_args()

    # Validate
    if not os.path.exists(args.raw_pt):
        print(f"Error: {args.raw_pt} does not exist")
        return

    # Precompute
    precomputer = MotionPrecomputer(
        include_fingertips=args.include_fingertips,
        base_idx=args.base_idx,
        hand_local=args.hand_local,
        feet_thre=args.feet_thre,
        kp_field=args.kp_field,
        verbose=args.verbose,
    )

    output_path = precomputer.precompute(
        raw_pt_path=args.raw_pt,
        output_pt_path=args.output_pt,
        skip_errors=args.skip_errors,
    )

    print(f"\n✓ Done! Precomputed data saved to {output_path}")


if __name__ == "__main__":
    main()