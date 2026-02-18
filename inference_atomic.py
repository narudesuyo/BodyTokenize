"""BodyTokenize inference for atomic-clip motion data (.npy kp3d files).

Reads individual kp3d .npy files (from prepare_atomic_clips.py), runs the
VQ-VAE model to produce tokenized pose indices, and saves as .npz files.

With --recon, also decodes token IDs back to body pose and saves
GT-vs-Reconstructed skeleton visualizations as .mp4 files.

Usage:
    DATA_ROOT=/large/naru/EgoHand/data python inference_atomic.py \
        --motion-dir /large/naru/EgoHand/data/train/takes_clipped/egoexo/motion_atomic \
        --output-dir /large/naru/EgoHand/data/train/takes_clipped/egoexo/tok_pose_atomic

    # With reconstruction visualization:
    DATA_ROOT=/large/naru/EgoHand/data python inference_atomic.py \
        --motion-dir ... --output-dir ... --recon --recon-max 20
"""

import sys
sys.path.append(".")

import argparse
import glob
import math
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
from src.dataset.collate import collate_stack
from src.dataset.kp3d2motion_rep import kp3d_to_motion_rep
from src.evaluate.utils import reconstruct_623_from_body_hand, recover_from_ric
from src.evaluate.vis import visualize_two_motions

from preprocess.paramUtil import t2m_raw_offsets, t2m_body_hand_kinematic_chain
from preprocess.paramUtil_add_tips import (
    t2m_raw_offsets_with_tips,
    t2m_body_hand_kinematic_chain_with_tips,
)
from common.skeleton import Skeleton


class NpyMotionInferenceDataset(Dataset):
    """Load a single kp3d .npy file and yield sliding-window clips.

    Mirrors MotionInferenceDataset but reads from a .npy file instead of a .pt dict.
    """

    def __init__(
        self,
        npy_path: str,
        clip_len: int = 80,
        overlap: int = 0,
        feet_thre: float = 0.002,
        include_fingertips: bool = False,
        hand_local: bool = False,
        base_idx: int = 0,
    ):
        super().__init__()

        self.npy_path = npy_path
        self.clip_len = int(clip_len)
        self.overlap = int(overlap)
        self.stride = self.clip_len - self.overlap
        self.feet_thre = feet_thre
        self.include_fingertips = include_fingertips
        self.hand_local = hand_local
        self.base_idx = base_idx

        # Load kp3d [T, 154, 3]
        kp = np.load(npy_path).astype(np.float32)
        if kp.ndim != 3 or kp.shape[2] != 3:
            raise ValueError(f"Bad kp3d shape: {kp.shape}, expected (T, J, 3)")

        # Block boundaries (same as MotionInferenceDataset)
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

        # Skeleton
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

        # kp -> kp52
        if self.include_fingertips:
            kp52_full = np.concatenate([kp[:, :22, :], kp[:, 25:55, :], kp[:, -10:, :]], axis=1)
        else:
            kp52_full = np.concatenate([kp[:, :22, :], kp[:, 25:55, :]], axis=1)

        self.kp = kp
        self.kp52_full = kp52_full
        self.Tfull = int(kp52_full.shape[0])

        # Target offsets from first frame
        tgt_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, "cpu")
        self.tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(kp52_full[0]).float())

        # Number of clips
        if self.Tfull <= self.clip_len:
            self.n_clips = 1
        else:
            self.n_clips = math.ceil((self.Tfull - self.clip_len) / self.stride) + 1

    def __len__(self):
        return self.n_clips

    def __getitem__(self, i: int):
        L = self.clip_len
        start = int(i) * self.stride
        end = start + L

        kp52 = self.kp52_full[start:end]
        t = kp52.shape[0]
        if t < L:
            pad = np.repeat(kp52[-1:, :, :], L - t, axis=0)
            kp52 = np.concatenate([kp52, pad], axis=0)

        arr = kp3d_to_motion_rep(
            kp3d_52_yup=kp52,
            feet_thre=self.feet_thre,
            tgt_offsets=self.tgt_offsets,
            n_raw_offsets=self.n_raw_offsets,
            kinematic_chain=self.kinematic_chain,
            base_idx=self.base_idx,
            hand_local=self.hand_local,
        )

        # Split body/hand (same as MotionInferenceDataset)
        Tm1 = arr.shape[0]
        if self.include_fingertips:
            ric = arr[:, self.I_RIC0:self.I_RIC1].reshape(Tm1, self.NO_ROOT_J, 3)
            rot = arr[:, self.I_ROT0:self.I_ROT1].reshape(Tm1, self.NO_ROOT_J, 6)
            vel = arr[:, self.I_VEL0:self.I_VEL1].reshape(Tm1, self.NO_ROOT_J + 1, 3)
            ric_body, ric_hand = ric[:, :21], ric[:, 21:61]
            rot_body, rot_hand = rot[:, :21], rot[:, 21:61]
            vel_body, vel_hand = vel[:, :22], vel[:, 22:62]
        else:
            ric = arr[:, self.I_RIC0:self.I_RIC1].reshape(Tm1, self.NO_ROOT_J, 3)
            rot = arr[:, self.I_ROT0:self.I_ROT1].reshape(Tm1, self.NO_ROOT_J, 6)
            vel = arr[:, self.I_VEL0:self.I_VEL1].reshape(Tm1, self.NO_ROOT_J + 1, 3)
            ric_body, ric_hand = ric[:, :21], ric[:, 21:51]
            rot_body, rot_hand = rot[:, :21], rot[:, 21:51]
            vel_body, vel_hand = vel[:, :22], vel[:, 22:52]

        root = arr[:, self.I_ROOT0:self.I_ROOT1]
        feet = arr[:, self.I_FEET0:self.I_FEET1]

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
            "clip_index": int(i),
            "start": int(start),
            "end": int(min(end, self.Tfull)),
            "Tfull": int(self.Tfull),
            "T": int(Tm1),
            "body": torch.from_numpy(body).float(),
            "hand": torch.from_numpy(hand).float(),
        }


def load_valid_sample_ids(annotation_json):
    """Load sample IDs that have at least one video (ego or exo) from the intermediate annotation."""
    import json as _json
    with open(annotation_json, "r") as f:
        entries = _json.load(f)
    valid = set()
    for entry in entries:
        if "video_ego" in entry or "video_exo" in entry:
            valid.add(entry["sample_id"])
    return valid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", choices=["train", "val"], default="train",
                    help="Dataset split (default: train)")
    ap.add_argument("--config", type=str,
                    default=os.path.join(_HERE, "ckpt_vq/config.yaml"))
    ap.add_argument("--ckpt", type=str,
                    default=os.path.join(_HERE, "ckpt_vq/ckpt_best.pt"))
    ap.add_argument("--motion-dir", type=str, default=None,
                    help="Directory of kp3d .npy files (default: {DATA_ROOT}/{split}/takes_clipped/egoexo/motion_atomic)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="Output directory for .npz token files (default: {DATA_ROOT}/{split}/takes_clipped/egoexo/tok_pose_atomic_40)")
    ap.add_argument("--annotation-json", type=str, default=None,
                    help="Intermediate annotation JSON from prepare_atomic_clips.py. "
                         "Only samples with both ego and exo video will be processed.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--clip-len", type=int, default=41)
    ap.add_argument("--overlap", type=int, default=1)
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
    egoexo_dir = os.path.join(_DATA_ROOT, args.split, "takes_clipped", "egoexo")
    if args.motion_dir is None:
        args.motion_dir = os.path.join(egoexo_dir, "motion_atomic")
    if args.output_dir is None:
        args.output_dir = os.path.join(egoexo_dir, "tok_pose_atomic_40")
    if args.annotation_json is None:
        suffix = "" if args.split == "train" else f"_{args.split}"
        args.annotation_json = os.path.join(
            _INTERNVIDEO_SCRIPT_DIR, f"annotation_atomic_intermediate{suffix}.json"
        )

    # Load valid sample IDs (ego + exo both exist)
    valid_ids = load_valid_sample_ids(args.annotation_json)
    print(f"Valid samples (ego+exo): {len(valid_ids)}")

    # Load model
    cfg = OmegaConf.load(args.config)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = build_model_from_args(cfg, device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"])
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
        mean_full = torch.from_numpy(np.load(mean_path)).float().to(device)  # (743,) or (623,)
        std_full = torch.from_numpy(np.load(std_path)).float().to(device)
        body_dim = cfg.get("body_in_dim", 263)
        mean_B = mean_full[:body_dim]
        std_B = std_full[:body_dim]
        mean_H = mean_full[body_dim:]
        std_H = std_full[body_dim:]
        print(f"Normalization enabled: mean/std shape={mean_full.shape}")

    recon_count = 0

    # Find all kp3d .npy files, filter by valid IDs
    all_npy = sorted(glob.glob(os.path.join(args.motion_dir, "**", "*_kp3d.npy"), recursive=True))
    npy_files = []
    for p in all_npy:
        sample_id = os.path.basename(p).replace("_kp3d.npy", "")
        if sample_id in valid_ids:
            npy_files.append(p)
    print(f"Found {len(all_npy)} motion files, {len(npy_files)} with ego+exo video")

    skipped = 0
    processed = 0

    for npy_path in tqdm(npy_files, desc="Tokenizing motion"):
        # Derive output path: motion_atomic/{take}/{id}_kp3d.npy -> tok_pose_atomic/{take}/{id}_{chunk}.npz
        rel_path = os.path.relpath(npy_path, args.motion_dir)
        take_name = os.path.dirname(rel_path)
        sample_id = os.path.basename(rel_path).replace("_kp3d.npy", "")

        save_dir = os.path.join(args.output_dir, take_name)
        os.makedirs(save_dir, exist_ok=True)

        do_recon = args.recon and (args.recon_max <= 0 or recon_count < args.recon_max)

        # Check if already processed (skip tokenization but still allow recon)
        already_tokenized = False
        if not args.overwrite:
            existing = glob.glob(os.path.join(save_dir, f"{sample_id}_*.npz"))
            if existing:
                if not do_recon:
                    skipped += 1
                    continue
                already_tokenized = True

        try:
            ds = NpyMotionInferenceDataset(
                npy_path=npy_path,
                clip_len=args.clip_len,
                overlap=args.overlap,
                include_fingertips=include_fingertips,
                hand_local=hand_local,
                base_idx=base_idx,
            )
        except (ValueError, KeyError) as e:
            print(f"[skip] {npy_path}: {e}")
            skipped += 1
            continue

        dl = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_stack,
        )

        with torch.no_grad():
            for batch in dl:
                chunk_idx = batch["clip_index"].item()
                save_path = os.path.join(save_dir, f"{sample_id}_{chunk_idx:04d}.npz")

                mB_raw = batch["mB"].to(device)
                mH_raw = batch["mH"].to(device)

                # Normalize before feeding to model
                if normalize:
                    mB = (mB_raw - mean_B) / (std_B + 1e-8)
                    mH = (mH_raw - mean_H) / (std_H + 1e-8)
                else:
                    mB, mH = mB_raw, mH_raw

                recon, losses, idx = model(mB, mH)

                idxH = idx["idxH"].detach().cpu().numpy()
                idxB = idx["idxB"].detach().cpu().numpy()
                combined = np.concatenate([idxB, idxH], axis=-1)

                if not already_tokenized:
                    np.savez_compressed(save_path, idx=combined)

                # --recon: decode IDs -> joints -> mp4
                if do_recon:
                    pr_n = model.decode_from_ids(
                        idxH=torch.from_numpy(idxH).to(device).long(),
                        idxB=torch.from_numpy(idxB).to(device).long(),
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
                    gt_full = reconstruct_623_from_body_hand(mB_raw, mH_raw, include_fingertips)

                    # recover 3D joint positions
                    j_pr = recover_from_ric(
                        pr_full, joints_num,
                        use_root_loss=use_root_loss,
                        base_idx=base_idx,
                        hand_local=hand_local,
                    )[0]  # (T, J, 3)
                    j_gt = recover_from_ric(
                        gt_full, joints_num,
                        use_root_loss=use_root_loss,
                        base_idx=base_idx,
                        hand_local=hand_local,
                    )[0]  # (T, J, 3)

                    # save mp4
                    recon_sub = os.path.join(args.recon_dir, take_name)
                    mp4_path = os.path.join(recon_sub, f"{sample_id}_{chunk_idx:04d}.mp4")
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
                    if args.recon_max > 0 and recon_count >= args.recon_max:
                        do_recon = False

        processed += 1

    print(f"\nProcessed: {processed}, Skipped: {skipped}")
    if args.recon:
        print(f"Reconstruction mp4 saved: {recon_count} (dir: {args.recon_dir})")


if __name__ == "__main__":
    main()
