#!/usr/bin/env python3
"""
Precompute Motion Representation from raw kp3d data
Converts kp3d → body(263) + hand(480) representation once
Saves to new pt file for fast loading
"""

import sys
import os
import subprocess
import time
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
        input_up_axis: str = "z",
        feet_thre: float = 0.002,
        kp_field: str = "kp3d",
        verbose: bool = True,
        hand_root: bool = False,
    ):
        self.include_fingertips = include_fingertips
        self.base_idx = base_idx
        self.hand_local = hand_local
        self.input_up_axis = input_up_axis
        self.feet_thre = feet_thre
        self.kp_field = kp_field
        self.verbose = verbose
        self.hand_root = hand_root

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

        # ---- target offsets: will be set on first valid data or from file ----
        self.tgt_offsets = None
        self._tgt_offsets_fixed = False

    def set_fixed_target_offsets(self, path):
        """Load target offsets from a saved .pt file (shared across datasets)."""
        self.tgt_offsets = torch.load(path, map_location="cpu", weights_only=False)
        self._tgt_offsets_fixed = True
        print(f"  Using fixed target offsets from {path}")

    def find_target_offsets(self, db, kp_field="kp3d"):
        """Find first valid clip to extract target offsets"""
        if self._tgt_offsets_fixed:
            return True
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
            _result = kp3d_to_motion_rep(
                kp3d_52_yup=kp52,
                feet_thre=self.feet_thre,
                tgt_offsets=self.tgt_offsets,
                n_raw_offsets=self.n_raw_offsets,
                kinematic_chain=self.kinematic_chain,
                base_idx=self.base_idx,
                hand_local=self.hand_local,
                input_up_axis=self.input_up_axis,
                compute_hand_root=self.hand_root,
            )
            if self.hand_root:
                arr, lh_root, rh_root = _result
            else:
                arr = _result
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

        hand_parts = []
        if self.hand_root:
            hand_parts.extend([lh_root, rh_root])  # (Tm1, 9) each
        hand_parts.extend([
            ric_hand.reshape(Tm1, -1),
            rot_hand.reshape(Tm1, -1),
            vel_hand.reshape(Tm1, -1),
        ])
        hand = np.concatenate(hand_parts, axis=1)  # [T, 360/480 or 378/498]

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
        num_shards: int = 1,
        shard_index: int = 0,
        sort_keys_for_shard: bool = False,
    ):
        """
        Precompute motion representation for all samples
        Args:
            raw_pt_path: Path to raw pt file with kp3d
            output_pt_path: Path to save precomputed pt file
            skip_errors: If True, skip samples with errors; if False, raise
            num_shards: Total number of shards for parallel runs
            shard_index: Index of this shard (0-based)
            sort_keys_for_shard: If True, sort keys before sharding for stability
        """
        if num_shards < 1:
            raise ValueError(f"num_shards must be >= 1, got {num_shards}")
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError(
                f"shard_index must be in [0, {num_shards - 1}], got {shard_index}"
            )

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

        all_keys = list(db_raw.keys())
        if sort_keys_for_shard:
            all_keys = sorted(all_keys)
        process_keys = all_keys[shard_index::num_shards]

        print(
            f"Processing {len(process_keys)} samples "
            f"(shard {shard_index + 1}/{num_shards})..."
        )
        for key in tqdm(process_keys, desc="Precomputing"):
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


def _build_child_cmd(args, shard_index: int, shard_out: Path):
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--raw-pt",
        args.raw_pt,
        "--output-pt",
        str(shard_out),
        "--base-idx",
        str(args.base_idx),
        "--input-up-axis",
        args.input_up_axis,
        "--feet-thre",
        str(args.feet_thre),
        "--kp-field",
        args.kp_field,
        "--num-shards",
        str(args.num_workers),
        "--shard-index",
        str(shard_index),
        "--num-workers",
        "1",
    ]
    if args.include_fingertips:
        cmd.append("--include-fingertips")
    if args.hand_local:
        cmd.append("--hand-local")
    if args.hand_root:
        cmd.append("--hand-root")
    if args.skip_errors:
        cmd.append("--skip-errors")
    if args.verbose:
        cmd.append("--verbose")
    if args.sort_keys_for_shard:
        cmd.append("--sort-keys-for-shard")
    if args.tgt_offsets:
        cmd.extend(["--tgt-offsets", args.tgt_offsets])
    return cmd


def _run_parallel_shards(args):
    if args.num_workers <= 1:
        return None
    if args.num_shards != 1 or args.shard_index != 0:
        raise ValueError(
            "Use either --num-workers (>1) OR manual --num-shards/--shard-index."
        )

    ts = int(time.time())
    out_path = Path(args.output_pt)
    shard_dir = (
        out_path.parent
        / f".precompute_shards_{out_path.stem}_{ts}_{os.getpid()}"
    )
    shard_dir.mkdir(parents=True, exist_ok=False)

    print(f"[PARALLEL] num_workers={args.num_workers}")
    print(f"[PARALLEL] shard_dir={shard_dir}")

    children = []
    for i in range(args.num_workers):
        shard_pt = shard_dir / f"shard_{i:03d}.pt"
        log_pt = shard_dir / f"shard_{i:03d}.log"
        cmd = _build_child_cmd(args, shard_index=i, shard_out=shard_pt)
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        f = open(log_pt, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
        )
        children.append((i, proc, f, shard_pt, log_pt))
        print(f"[PARALLEL] worker={i} pid={proc.pid} log={log_pt.name}")

    failed = []
    for i, proc, f, _, log_pt in children:
        rc = proc.wait()
        f.close()
        print(f"[PARALLEL] worker={i} exit_code={rc}")
        if rc != 0:
            failed.append((i, log_pt))

    if failed:
        print("[ERROR] Some workers failed:")
        for i, log_pt in failed:
            print(f"  worker={i} log={log_pt}")
        raise RuntimeError("Parallel shard workers failed")

    merged = {}
    total_in = 0
    dup = 0
    for i, _, _, shard_pt, _ in children:
        db = torch.load(shard_pt, map_location="cpu", weights_only=False)
        if not isinstance(db, dict):
            raise TypeError(f"Expected dict in shard output: {shard_pt}")
        total_in += len(db)
        for k, v in db.items():
            if k in merged:
                dup += 1
                continue
            merged[k] = v
        print(f"[MERGE] shard={i} samples={len(db)}")

    os.makedirs(os.path.dirname(args.output_pt), exist_ok=True)
    torch.save(merged, args.output_pt)
    print(f"[DONE] input_total={total_in}")
    print(f"[DONE] merged={len(merged)}")
    print(f"[DONE] duplicates_skipped={dup}")
    print(f"[DONE] saved={args.output_pt}")
    print(f"[DONE] shard_artifacts={shard_dir}")
    return args.output_pt


class HandOnlyPrecomputer:
    """Precompute hand-only motion representation from MANO-derived hand joints.

    Input pt format (from hot3d_mano_to_raw_pt.py):
        {key: {"kp3d_hands": (T, 32, 3), "head_pos": (T, 3), "head_rot": (T, 4)}}

    Output pt format (compatible with MotionDataset cache mode):
        {key: {"body": zeros(T-1, 263), "hand": (T-1, hand_dim)}}
    """

    def __init__(
        self,
        body_in_dim: int = 263,
        hand_root: bool = True,
        hand_root_dim: int = 9,
        hand_local: bool = True,
        include_fingertips: bool = False,
        verbose: bool = True,
    ):
        self.body_in_dim = body_in_dim
        self.hand_root = hand_root
        self.hand_root_dim = hand_root_dim
        self.hand_local = hand_local
        self.include_fingertips = include_fingertips
        self.verbose = verbose

        # Per hand (excluding wrist which is root):
        # include_fingertips=False: 15 finger joints
        # include_fingertips=True:  15 finger + 5 tips = 20 joints
        self.joints_per_hand = 20 if include_fingertips else 15

    def _compute_hand_rep(self, joints_hand, head_pos, head_rot, wrist_rot=None):
        """Compute hand motion representation for one hand.

        Args:
            joints_hand: (T, 16/21, 3) - wrist + 15 finger [+ 5 tips] in world frame
            head_pos: (T, 3) - headset position
            head_rot: (T, 4) - headset rotation (wxyz)
            wrist_rot: (T, 4) - wrist rotation quaternion (wxyz), optional

        Returns:
            hand_parts: dict with keys 'ric_finger', 'rot6d_finger', 'vel_finger',
                        and optionally 'ric_tip', 'rot6d_tip', 'vel_tip', 'hand_root'
            T_out: output time length (T-1)
        """
        from common.quaternion import qrot_np, qinv_np, quaternion_to_cont6d_np, qmul_np

        T = joints_hand.shape[0]
        wrist_pos = joints_hand[:, 0]       # (T, 3) wrist position
        finger_pos = joints_hand[:, 1:16]   # (T, 15, 3) finger joints (always 15)

        if self.include_fingertips:
            tip_pos = joints_hand[:, 16:21]  # (T, 5, 3) fingertip joints

        # --- RIC: joint positions relative to wrist (hand_local) ---
        if self.hand_local:
            ric_finger = finger_pos - wrist_pos[:, None, :]
        else:
            ric_finger = finger_pos - head_pos[:, None, :]
        ric_finger_flat = ric_finger[:-1].reshape(T - 1, 15 * 3)

        # --- rot6d: use identity rotations as placeholder ---
        rot6d_finger = np.zeros((T - 1, 15, 6), dtype=np.float32)
        rot6d_finger[..., 0] = 1.0
        rot6d_finger[..., 4] = 1.0
        rot6d_finger_flat = rot6d_finger.reshape(T - 1, 15 * 6)

        # --- vel: joint velocity ---
        if self.hand_local:
            finger_rel = finger_pos - wrist_pos[:, None, :]
            vel_finger = finger_rel[1:] - finger_rel[:-1]
        else:
            vel_finger = finger_pos[1:] - finger_pos[:-1]
        vel_finger_flat = vel_finger.reshape(T - 1, 15 * 3)

        parts = {
            'ric_finger': ric_finger_flat,
            'rot6d_finger': rot6d_finger_flat,
            'vel_finger': vel_finger_flat,
        }

        # --- Fingertip features (same structure, 5 joints) ---
        if self.include_fingertips:
            if self.hand_local:
                ric_tip = tip_pos - wrist_pos[:, None, :]
            else:
                ric_tip = tip_pos - head_pos[:, None, :]
            parts['ric_tip'] = ric_tip[:-1].reshape(T - 1, 5 * 3)

            rot6d_tip = np.zeros((T - 1, 5, 6), dtype=np.float32)
            rot6d_tip[..., 0] = 1.0
            rot6d_tip[..., 4] = 1.0
            parts['rot6d_tip'] = rot6d_tip.reshape(T - 1, 5 * 6)

            if self.hand_local:
                tip_rel = tip_pos - wrist_pos[:, None, :]
                vel_tip = tip_rel[1:] - tip_rel[:-1]
            else:
                vel_tip = tip_pos[1:] - tip_pos[:-1]
            parts['vel_tip'] = vel_tip.reshape(T - 1, 5 * 3)

        # --- hand_root: wrist vel(3) + rot6d(6) relative to head ---
        if self.hand_root:
            head_rot_inv = qinv_np(head_rot)  # (T, 4)
            wrist_rel = wrist_pos - head_pos
            wrist_vel = wrist_rel[1:] - wrist_rel[:-1]
            wrist_vel = qrot_np(head_rot_inv[1:], wrist_vel)

            if wrist_rot is not None:
                # Wrist rotation relative to head: q_rel = q_head_inv * q_wrist
                wrist_rot_rel = qmul_np(head_rot_inv[:-1], wrist_rot[:-1])  # (T-1, 4)
                wrist_rot6d = quaternion_to_cont6d_np(wrist_rot_rel)  # (T-1, 6)
            else:
                wrist_rot6d = np.zeros((T - 1, 6), dtype=np.float32)
                wrist_rot6d[:, 0] = 1.0
                wrist_rot6d[:, 4] = 1.0

            parts['hand_root'] = np.concatenate([wrist_vel, wrist_rot6d], axis=-1)

        return parts, T - 1

    def process_single(self, item):
        """Process a single clip.

        Args:
            item: dict with kp3d_hands (T, 32, 3), head_pos (T, 3), head_rot (T, 4)

        Returns:
            body: (T-1, 263) zeros
            hand: (T-1, hand_dim) computed
            meta: dict with head_pos (T-1, 3), head_rot (T-1, 4),
                  lh_wrist_world (T-1, 3), rh_wrist_world (T-1, 3)
        """
        kp3d_hands = item["kp3d_hands"]
        head_pos = item["head_pos"]
        head_rot = item["head_rot"]
        wrist_rot_lh = item.get("wrist_rot_lh", None)
        wrist_rot_rh = item.get("wrist_rot_rh", None)

        if torch.is_tensor(kp3d_hands):
            kp3d_hands = kp3d_hands.numpy()
        if torch.is_tensor(head_pos):
            head_pos = head_pos.numpy()
        if torch.is_tensor(head_rot):
            head_rot = head_rot.numpy()
        if wrist_rot_lh is not None and torch.is_tensor(wrist_rot_lh):
            wrist_rot_lh = wrist_rot_lh.numpy()
        if wrist_rot_rh is not None and torch.is_tensor(wrist_rot_rh):
            wrist_rot_rh = wrist_rot_rh.numpy()

        T = kp3d_hands.shape[0]
        if T < 2:
            return None, None, None

        # Split into LH and RH
        # Without tips: (T, 32, 3) = [16 LH, 16 RH]
        # With tips:    (T, 42, 3) = [21 LH, 21 RH]
        n_per_hand = kp3d_hands.shape[1] // 2
        lh_joints = kp3d_hands[:, :n_per_hand]
        rh_joints = kp3d_hands[:, n_per_hand:]

        # Compute hand representation for each hand
        lh_parts, T_out = self._compute_hand_rep(lh_joints, head_pos, head_rot, wrist_rot=wrist_rot_lh)
        rh_parts, _ = self._compute_hand_rep(rh_joints, head_pos, head_rot, wrist_rot=wrist_rot_rh)

        # Interleave LH/RH matching full-body pipeline order:
        # [LH_fingers, RH_fingers, LH_tips, RH_tips] per feature type
        # Full order: hand_root_lh, hand_root_rh,
        #   ric_lh_finger, ric_rh_finger, ric_lh_tip, ric_rh_tip,
        #   rot6d_lh_finger, rot6d_rh_finger, rot6d_lh_tip, rot6d_rh_tip,
        #   vel_lh_finger, vel_rh_finger, vel_lh_tip, vel_rh_tip
        hand_parts = []
        if self.hand_root:
            hand_parts.extend([lh_parts['hand_root'], rh_parts['hand_root']])

        for feat in ['ric', 'rot6d', 'vel']:
            hand_parts.extend([lh_parts[f'{feat}_finger'], rh_parts[f'{feat}_finger']])
            if self.include_fingertips:
                hand_parts.extend([lh_parts[f'{feat}_tip'], rh_parts[f'{feat}_tip']])

        hand = np.concatenate(hand_parts, axis=-1)  # (T-1, hand_dim)

        # Body: zeros
        body = np.zeros((T_out, self.body_in_dim), dtype=np.float32)

        # NaN check
        if np.isnan(hand).any() or np.isinf(hand).any():
            return None, None, None

        # World-frame metadata for visualization
        meta = {
            "head_pos": head_pos[:-1].astype(np.float32),       # (T-1, 3)
            "head_rot": head_rot[:-1].astype(np.float32),       # (T-1, 4) wxyz
            "lh_wrist_world": lh_joints[:-1, 0].astype(np.float32),  # (T-1, 3)
            "rh_wrist_world": rh_joints[:-1, 0].astype(np.float32),  # (T-1, 3)
        }

        return body, hand, meta

    def precompute(self, raw_pt_path: str, output_pt_path: str, skip_errors: bool = True):
        """Precompute hand-only representation for all clips."""
        print(f"Loading raw data from {raw_pt_path}...")
        db_raw = torch.load(raw_pt_path, map_location="cpu", weights_only=False)
        print(f"  Loaded {len(db_raw)} clips")

        db_processed = {}
        n_success = 0
        n_fail = 0

        for key in tqdm(db_raw.keys(), desc="Precomputing (hand-only)"):
            item = db_raw[key]
            try:
                body, hand, meta = self.process_single(item)
                if body is None:
                    n_fail += 1
                    continue

                entry = {
                    "body": torch.from_numpy(body).float(),
                    "hand": torch.from_numpy(hand).float(),
                    "T": int(body.shape[0]),
                    "key": key,
                }
                if meta is not None:
                    for mk, mv in meta.items():
                        entry[mk] = torch.from_numpy(mv).float()
                db_processed[key] = entry
                n_success += 1
            except Exception as e:
                n_fail += 1
                if not skip_errors:
                    raise
                if self.verbose:
                    print(f"  [ERROR] key={key}: {e}")

        print(f"\nProcessing complete: success={n_success}, fail={n_fail}")
        os.makedirs(os.path.dirname(output_pt_path) or ".", exist_ok=True)
        torch.save(db_processed, output_pt_path)
        print(f"Saved {len(db_processed)} clips to {output_pt_path}")

        if db_processed:
            sample = db_processed[list(db_processed.keys())[0]]
            print(f"Output shapes: body={sample['body'].shape}, hand={sample['hand'].shape}")

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
        "--hand-root",
        action="store_true",
        help="Compute hand root (9D per hand: wrist vel + rot6d)",
    )
    parser.add_argument(
        "--input-up-axis",
        type=str,
        default="z",
        choices=["z", "y", "auto"],
        help="Input kp3d up-axis before conversion to motion representation",
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
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards for parallel runs (>=1)",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Shard index to process (0-based)",
    )
    parser.add_argument(
        "--sort-keys-for-shard",
        action="store_true",
        help="Sort sample keys before sharding for stable assignment",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Run shard workers in parallel automatically (>=1)",
    )
    parser.add_argument(
        "--hand-only",
        action="store_true",
        help="Hand-only precompute mode (for HOT3D: kp3d_hands → body=zeros + hand)",
    )
    parser.add_argument(
        "--body-in-dim",
        type=int,
        default=263,
        help="Body input dimension (for hand-only zero padding)",
    )
    parser.add_argument(
        "--tgt-offsets",
        type=str,
        default=None,
        help="Path to shared target skeleton offsets .pt file (overrides auto-detect from first sample)",
    )

    args = parser.parse_args()

    # Validate
    if not os.path.exists(args.raw_pt):
        print(f"Error: {args.raw_pt} does not exist")
        return

    if args.hand_only:
        # Hand-only mode: HOT3D MANO-derived data
        precomputer = HandOnlyPrecomputer(
            body_in_dim=args.body_in_dim,
            hand_root=args.hand_root,
            hand_root_dim=9,
            hand_local=args.hand_local,
            include_fingertips=args.include_fingertips,
            verbose=args.verbose,
        )
        output_path = precomputer.precompute(
            raw_pt_path=args.raw_pt,
            output_pt_path=args.output_pt,
            skip_errors=args.skip_errors,
        )
        print(f"\n Done! Hand-only precomputed data saved to {output_path}")
        return

    output_path = _run_parallel_shards(args)
    if output_path is None:
        # Precompute
        precomputer = MotionPrecomputer(
            include_fingertips=args.include_fingertips,
            base_idx=args.base_idx,
            hand_local=args.hand_local,
            input_up_axis=args.input_up_axis,
            feet_thre=args.feet_thre,
            kp_field=args.kp_field,
            verbose=args.verbose,
            hand_root=args.hand_root,
        )
        if args.tgt_offsets:
            precomputer.set_fixed_target_offsets(args.tgt_offsets)

        output_path = precomputer.precompute(
            raw_pt_path=args.raw_pt,
            output_pt_path=args.output_pt,
            skip_errors=args.skip_errors,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
            sort_keys_for_shard=args.sort_keys_for_shard,
        )

    print(f"\n✓ Done! Precomputed data saved to {output_path}")


if __name__ == "__main__":
    main()
