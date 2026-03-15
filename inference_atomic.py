"""BodyTokenize inference for atomic-clip motion data (.npy kp3d files).

Reads individual kp3d .npy files (from prepare_atomic_clips.py), runs the
VQ-VAE model to produce tokenized pose indices, and saves as .npz files.

All .npy files are loaded into a single batch dataset for efficient
parallel processing (batch_size=128, num_workers=8 by default).

With --recon, also decodes token IDs back to body pose and saves
GT-vs-Reconstructed skeleton visualizations as .mp4 files.

Usage:
    DATA_ROOT=/large/naru/EgoHand/data python inference_atomic.py \
        --split train

    # With reconstruction visualization:
    DATA_ROOT=/large/naru/EgoHand/data python inference_atomic.py \
        --split train --recon --recon-max 20
"""

import sys
sys.path.append(".")

import argparse
import glob
import os

_HERE = os.path.dirname(os.path.abspath(__file__))  # BodyTokenize/
_DATA_ROOT = os.environ.get("DATA_ROOT", "/large/naru/EgoHand/data")
_INTERNVIDEO_SCRIPT_DIR = os.path.join(
    _HERE, "../InternVideo/InternVideo2/multi_modality/scripts/pretraining/stage2/1B_motion"
)

import matplotlib
matplotlib.use("Agg")

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        for i, item in enumerate(iterable):
            if i % 50 == 0:
                print(f"\r{desc}: {i}", end="", flush=True)
            yield item
        print()

from src.train.utils import build_model_from_args
from src.dataset.kp3d2motion_rep import kp3d_to_motion_rep
from src.evaluate.utils import reconstruct_623_from_body_hand, recover_from_ric
from src.evaluate.vis import visualize_two_motions

from preprocess.paramUtil import t2m_raw_offsets, t2m_body_hand_kinematic_chain
from preprocess.paramUtil_add_tips import (
    t2m_raw_offsets_with_tips,
    t2m_body_hand_kinematic_chain_with_tips,
)
from common.skeleton import Skeleton


class NpyMotionAtomicBatchDataset(Dataset):
    """Load all atomic kp3d .npy files for batch inference.

    Each .npy file contains exactly 41 frames (fixed by prepare_atomic_clips.py).
    One file = one sample, no sliding window needed.
    """

    def __init__(
        self,
        npy_files,  # list of (npy_path, take_name, sample_id)
        clip_len=41,
        feet_thre=0.002,
        include_fingertips=False,
        hand_local=False,
        base_idx=0,
    ):
        super().__init__()
        self.npy_files = npy_files
        self.clip_len = clip_len
        self.feet_thre = feet_thre
        self.include_fingertips = include_fingertips
        self.hand_local = hand_local
        self.base_idx = base_idx

        if include_fingertips:
            self.NO_ROOT_J = 61
            self.n_raw_offsets = torch.from_numpy(t2m_raw_offsets_with_tips).float()
            self.kinematic_chain = t2m_body_hand_kinematic_chain_with_tips
        else:
            self.NO_ROOT_J = 51
            self.n_raw_offsets = torch.from_numpy(t2m_raw_offsets).float()
            self.kinematic_chain = t2m_body_hand_kinematic_chain

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

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        npy_path, take_name, sample_id = self.npy_files[idx]

        kp = np.load(npy_path).astype(np.float32)  # [41, J, 3]
        J = kp.shape[1]

        # Handle different joint formats:
        #   154 joints: EgoExo4D format (body[0:22] + jaw/eyes[22:25] + hands[25:55] + ...)
        #    62 joints: SMPL-H + fingertips (body[0:22] + hands[22:52] + tips[52:62])
        #    52 joints: SMPL-H (body[0:22] + hands[22:52])
        if J == 154:
            if self.include_fingertips:
                kp52 = np.concatenate([kp[:, :22, :], kp[:, 25:55, :], kp[:, -10:, :]], axis=1)
            else:
                kp52 = np.concatenate([kp[:, :22, :], kp[:, 25:55, :]], axis=1)
        elif J == 62:
            kp52 = kp[:, :62, :] if self.include_fingertips else kp[:, :52, :]
        elif J == 52:
            kp52 = kp
        else:
            raise ValueError(f"Unexpected joint count {J} in {npy_path}")

        # Pad or truncate to clip_len (should be exactly 41 from prepare_atomic_clips.py)
        T = kp52.shape[0]
        if T < self.clip_len:
            pad = np.repeat(kp52[-1:, :, :], self.clip_len - T, axis=0)
            kp52 = np.concatenate([kp52, pad], axis=0)
        elif T > self.clip_len:
            kp52 = kp52[:self.clip_len]

        # Target offsets from first frame
        tgt_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, "cpu")
        tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(kp52[0]).float())

        # kp3d -> motion representation (623D)
        arr = kp3d_to_motion_rep(
            kp3d_52_yup=kp52,
            feet_thre=self.feet_thre,
            tgt_offsets=tgt_offsets,
            n_raw_offsets=self.n_raw_offsets,
            kinematic_chain=self.kinematic_chain,
            base_idx=self.base_idx,
            hand_local=self.hand_local,
        )

        # Split body/hand
        Tm1 = arr.shape[0]
        root = arr[:, self.I_ROOT0:self.I_ROOT1]
        ric = arr[:, self.I_RIC0:self.I_RIC1].reshape(Tm1, self.NO_ROOT_J, 3)
        rot = arr[:, self.I_ROT0:self.I_ROT1].reshape(Tm1, self.NO_ROOT_J, 6)
        vel = arr[:, self.I_VEL0:self.I_VEL1].reshape(Tm1, self.NO_ROOT_J + 1, 3)
        feet = arr[:, self.I_FEET0:self.I_FEET1]

        if self.include_fingertips:
            ric_body, ric_hand = ric[:, :21], ric[:, 21:61]
            rot_body, rot_hand = rot[:, :21], rot[:, 21:61]
            vel_body, vel_hand = vel[:, :22], vel[:, 22:62]
        else:
            ric_body, ric_hand = ric[:, :21], ric[:, 21:51]
            rot_body, rot_hand = rot[:, :21], rot[:, 21:51]
            vel_body, vel_hand = vel[:, :22], vel[:, 22:52]

        body = np.concatenate([
            root,
            ric_body.reshape(Tm1, -1),
            rot_body.reshape(Tm1, -1),
            vel_body.reshape(Tm1, -1),
            feet,
        ], axis=1)
        hand = np.concatenate([
            ric_hand.reshape(Tm1, -1),
            rot_hand.reshape(Tm1, -1),
            vel_hand.reshape(Tm1, -1),
        ], axis=1)

        return {
            "mB": torch.from_numpy(body).float(),
            "mH": torch.from_numpy(hand).float(),
            "take_name": take_name,
            "sample_id": sample_id,
            "npy_path": npy_path,
        }


def collate_atomic(batch):
    """Custom collate: stack tensors, keep strings as lists."""
    return {
        "mB": torch.stack([b["mB"] for b in batch], dim=0),
        "mH": torch.stack([b["mH"] for b in batch], dim=0),
        "take_name": [b["take_name"] for b in batch],
        "sample_id": [b["sample_id"] for b in batch],
        "npy_path": [b["npy_path"] for b in batch],
    }


def load_valid_sample_ids(annotation_json):
    """Load sample IDs that have at least one video from the intermediate annotation."""
    import json as _json
    with open(annotation_json, "r") as f:
        entries = _json.load(f)
    valid = set()
    for entry in entries:
        if ("video_ego" in entry or "video_exo" in entry or "video" in entry
                or entry.get("_motion_ok") or "motion_kp3d" in entry):
            valid.add(entry["sample_id"])
    return valid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["train", "val"], default="train",
                    help="Dataset split (default: train)")
    ap.add_argument("--data-root", type=str, default=None,
                    help="Base egoexo directory "
                         "(default: {DATA_ROOT}/{split}/takes_clipped/egoexo)")
    ap.add_argument("--config", type=str,
                    default=os.path.join(_HERE, "ckpt_vq/config.yaml"))
    ap.add_argument("--ckpt", type=str,
                    default=os.path.join(_HERE, "ckpt_vq/ckpt_best.pt"))
    ap.add_argument("--motion-dir", type=str, default=None,
                    help="Directory of kp3d .npy files "
                         "(default: {data-root}/motion_atomic)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Output directory for .npz token files "
                         "(default: {data-root}/tok_pose_atomic_40)")
    ap.add_argument("--annotation-json", type=str, default=None,
                    help="Intermediate annotation JSON from prepare_atomic_clips.py. "
                         "Only samples with ego or exo video will be processed.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--clip-len", type=int, default=41,
                    help="Clip length in frames (41 = fixed output from prepare_atomic_clips.py)")
    ap.add_argument("--batch-size", type=int, default=128,
                    help="Batch size for inference (default: 128)")
    ap.add_argument("--num-workers", type=int, default=8,
                    help="Number of DataLoader workers (default: 8)")
    ap.add_argument("--recon", action="store_true",
                    help="Decode token IDs back to body pose and save GT-vs-Recon mp4.")
    ap.add_argument("--recon-dir", type=str,
                    default=os.path.join(_HERE, "recon"),
                    help="Directory to save reconstruction mp4 files.")
    ap.add_argument("--recon-max", type=int, default=0,
                    help="Max number of samples to reconstruct (0 = all).")
    ap.add_argument("--recon-fps", type=int, default=10)
    ap.add_argument("--recon-view", type=str, default="all",
                    choices=["all", "body", "hands", "lh", "rh"])
    args = ap.parse_args()

    # Resolve split-dependent defaults
    if args.data_root is None:
        args.data_root = os.path.join(_DATA_ROOT, args.split, "takes_clipped", "egoexo")
    if args.motion_dir is None:
        args.motion_dir = os.path.join(args.data_root, "motion_atomic")
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_root, "tok_pose_atomic_40")
    if args.annotation_json is None:
        suffix = "" if args.split == "train" else f"_{args.split}"
        args.annotation_json = os.path.join(
            _INTERNVIDEO_SCRIPT_DIR, f"annotation_atomic_intermediate{suffix}.json"
        )

    print(f"data_root:       {args.data_root}")
    print(f"motion_dir:      {args.motion_dir}")
    print(f"output_dir:      {args.output_dir}")
    print(f"annotation_json: {args.annotation_json}")

    # Load valid sample IDs (ego or exo exist)
    valid_ids = load_valid_sample_ids(args.annotation_json)
    print(f"Valid samples (ego+exo): {len(valid_ids)}")

    # Load model
    cfg = OmegaConf.load(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = build_model_from_args(cfg, device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # Handle DDP state_dict (module. prefix)
    state_dict = ckpt["model"]
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model.eval()

    include_fingertips = cfg.get("include_fingertips", False)
    use_root_loss = cfg.get("use_root_loss", False)
    base_idx = cfg.get("base_idx", 0)
    hand_local = cfg.get("hand_local", False)
    joints_num = 62 if include_fingertips else 52
    normalize = cfg.get("normalize", False)

    # Load normalization statistics (split into body / hand)
    if normalize:
        mean_path = cfg.get("mean_path", "./preprocess/statistics/tips/mean.npy")
        std_path = cfg.get("std_path", "./preprocess/statistics/tips/std.npy")
        mean_full = torch.from_numpy(np.load(mean_path)).float().to(device)
        std_full = torch.from_numpy(np.load(std_path)).float().to(device)
        body_dim = cfg.get("body_in_dim", 263)
        mean_B = mean_full[:body_dim]
        std_B = std_full[:body_dim]
        mean_H = mean_full[body_dim:]
        std_H = std_full[body_dim:]
        print(f"Normalization enabled: mean/std shape={mean_full.shape}")

    # Scan .npy files, filter by valid IDs and already-processed
    all_npy = sorted(glob.glob(os.path.join(args.motion_dir, "**", "*_kp3d.npy"), recursive=True))
    npy_files = []
    skipped = 0
    for p in all_npy:
        rel_path = os.path.relpath(p, args.motion_dir)
        take_name = os.path.dirname(rel_path)
        sample_id = os.path.basename(rel_path).replace("_kp3d.npy", "")

        if sample_id not in valid_ids:
            continue

        if not args.overwrite:
            save_dir = os.path.join(args.output_dir, take_name)
            if glob.glob(os.path.join(save_dir, f"{sample_id}_*.npz")):
                skipped += 1
                continue

        npy_files.append((p, take_name, sample_id))

    print(f"Found {len(all_npy)} motion files, {len(npy_files)} to process, {skipped} already done")

    if not npy_files:
        print("Nothing to process.")
        return

    # Create dataset and dataloader
    ds = NpyMotionAtomicBatchDataset(
        npy_files=npy_files,
        clip_len=args.clip_len,
        include_fingertips=include_fingertips,
        hand_local=hand_local,
        base_idx=base_idx,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_atomic,
    )

    processed = 0
    recon_count = 0

    with torch.no_grad():
        for batch in tqdm(dl, desc="Tokenizing motion"):
            mB_raw = batch["mB"].to(device)
            mH_raw = batch["mH"].to(device)

            # Normalize before feeding to model
            if normalize:
                mB = (mB_raw - mean_B) / (std_B + 1e-8)
                mH = (mH_raw - mean_H) / (std_H + 1e-8)
            else:
                mB, mH = mB_raw, mH_raw

            _, _, idx = model(mB, mH)

            idxB = idx["idxB"].detach().cpu().numpy()
            idxH = idx["idxH"].detach().cpu().numpy()
            combined = np.concatenate([idxB, idxH], axis=-1)  # (B, T, 8)

            B = combined.shape[0]
            for i in range(B):
                take_name = batch["take_name"][i]
                sample_id = batch["sample_id"][i]

                save_dir = os.path.join(args.output_dir, take_name)
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"{sample_id}_0000.npz")

                np.savez_compressed(save_path, idx=combined[i:i+1])  # keep (1, T, 8) shape
                processed += 1

                # --recon: decode IDs -> joints -> mp4
                if args.recon and (args.recon_max <= 0 or recon_count < args.recon_max):
                    pr_n = model.decode_from_ids(
                        idxH=torch.from_numpy(idxH[i:i+1]).to(device).long(),
                        idxB=torch.from_numpy(idxB[i:i+1]).to(device).long(),
                    )
                    # Denormalize decoded output
                    if normalize:
                        bd = mB_raw.shape[-1]
                        pr_body = pr_n[..., :bd] * (std_B + 1e-8) + mean_B
                        pr_hand = pr_n[..., bd:] * (std_H + 1e-8) + mean_H
                    else:
                        bd = mB_raw.shape[-1]
                        pr_body = pr_n[..., :bd]
                        pr_hand = pr_n[..., bd:]
                    pr_full = reconstruct_623_from_body_hand(pr_body, pr_hand, include_fingertips)

                    # GT: use raw (unnormalized) data
                    gt_full = reconstruct_623_from_body_hand(
                        mB_raw[i:i+1], mH_raw[i:i+1], include_fingertips
                    )

                    # Recover 3D joint positions
                    j_pr = recover_from_ric(
                        pr_full, joints_num,
                        use_root_loss=use_root_loss,
                        base_idx=base_idx,
                        hand_local=hand_local,
                    )[0]
                    j_gt = recover_from_ric(
                        gt_full, joints_num,
                        use_root_loss=use_root_loss,
                        base_idx=base_idx,
                        hand_local=hand_local,
                    )[0]

                    # Save mp4
                    recon_sub = os.path.join(args.recon_dir, take_name)
                    mp4_path = os.path.join(recon_sub, f"{sample_id}_0000.mp4")
                    visualize_two_motions(
                        j_gt, j_pr,
                        save_path=mp4_path,
                        fps=args.recon_fps,
                        view=args.recon_view,
                        rotate=False,
                        include_fingertips=include_fingertips,
                        origin_align=True,
                        base_idx=base_idx,
                        only_gt=False,
                    )
                    recon_count += 1

    print(f"\nProcessed: {processed}, Skipped (already done): {skipped}")
    if args.recon:
        print(f"Reconstruction mp4 saved: {recon_count} (dir: {args.recon_dir})")


if __name__ == "__main__":
    main()
