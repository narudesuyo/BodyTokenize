# eval_vqvae_623_to_joints.py
import os, glob
import numpy as np
import torch
import argparse
import sys

sys.path.append(".")
from src.model.vqvae import H2VQ
from src.dataset.dataloader import MotionDataset

from common.quaternion import qrot, qinv
import matplotlib
matplotlib.use("Agg")   # ★ GUIなし環境用（重要）

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import torch

import torch

def reconstruct_623_from_body_hand(mb: torch.Tensor, mh: torch.Tensor,
                                    include_fingertips: bool = False,
                                    hand_root_dim: int = 0):
    """
    mb: (B,T,263)
    mh: (B,T,360/480 or 378/498 when hand_root)
    hand_root_dim: 0 (no hand root) or 18 (9 per hand, prepended to mh)
    return: x623 (B,T,623/743)
    """

    B, T, _ = mb.shape
    device = mb.device

    # ---------- strip hand root if present ----------
    if hand_root_dim > 0:
        mh_local = mh[..., hand_root_dim:]
    else:
        mh_local = mh

    # ---------- split body ----------
    i = 0
    root = mb[..., i:i+4]; i += 4                 # (B,T,4)

    ric_body = mb[..., i:i+63].view(B,T,21,3); i += 63
    rot_body = mb[..., i:i+126].view(B,T,21,6); i += 126
    vel_body = mb[..., i:i+66].view(B,T,22,3); i += 66
    feet     = mb[..., i:i+4];                    # (B,T,4)

    # ---------- split hand ----------
    if include_fingertips:
        j = 0
        ric_hand = mh_local[..., j:j+120].view(B,T,40,3); j += 120
        rot_hand = mh_local[..., j:j+240].view(B,T,40,6); j += 240
        vel_hand = mh_local[..., j:j+120].view(B,T,40,3); j += 120
        NO_ROOT_J = 61
    else:
        NO_ROOT_J = 51
        j = 0
        ric_hand = mh_local[..., j:j+90].view(B,T,30,3); j += 90
        rot_hand = mh_local[..., j:j+180].view(B,T,30,6); j += 180
        vel_hand = mh_local[..., j:j+90].view(B,T,30,3); j += 90

    # ---------- reassemble original order ----------
    ric = torch.cat([ric_body, ric_hand], dim=2).reshape(B,T,NO_ROOT_J*3)
    rot = torch.cat([rot_body, rot_hand], dim=2).reshape(B,T,NO_ROOT_J*6)
    vel = torch.cat([vel_body, vel_hand], dim=2).reshape(B,T,(NO_ROOT_J+1)*3)

    # ---------- final concat ----------
    x623 = torch.cat([root, ric, rot, vel, feet], dim=-1)

    if include_fingertips:
        assert x623.shape[-1] == 743
    else:
        assert x623.shape[-1] == 623

    return x623





# -----------------------------
# same as train
# -----------------------------
def collate_crop_pad(batch, T0: int):
    """
    batch[i] is dict from Motion623SplitDataset:
      body: [Ti,263], hand: [Ti,360]
    -> return mB [B,T0,263], mH [B,T0,360], mask [B,T0]
    """
    B = len(batch)
    body_dim = batch[0]["body"].shape[-1]
    hand_dim = batch[0]["hand"].shape[-1]

    mB = torch.zeros(B, T0, body_dim, dtype=torch.float32)
    mH = torch.zeros(B, T0, hand_dim, dtype=torch.float32)
    mask = torch.zeros(B, T0, dtype=torch.bool)

    paths = []
    for i, item in enumerate(batch):
        b = item["body"]
        h = item["hand"]
        Ti = b.shape[0]
        paths.append(item.get("path", ""))

        if Ti >= T0:
            s = torch.randint(0, Ti - T0 + 1, (1,)).item()
            mB[i] = b[s:s+T0]
            mH[i] = h[s:s+T0]
            mask[i] = True
        else:
            mB[i, :Ti] = b
            mH[i, :Ti] = h
            mask[i, :Ti] = True

    return {"mB": mB, "mH": mH, "mask": mask, "paths": paths}

def recover_root_rot_pos(data_t: torch.Tensor):
    rot_vel = data_t[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel)
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data_t.shape[:-1] + (4,), device=data_t.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data_t.shape[:-1] + (3,), device=data_t.device)
    r_pos[..., 1:, [0, 2]] = data_t[..., :-1, 1:3]
    r_pos = qrot(r_rot_quat, r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data_t[..., 3]
    return r_rot_quat, r_pos


def recover_from_ric(data_t: torch.Tensor, joints_num: int, use_root_loss: bool = True, base_idx: int = 0,
                     hand_local: bool = False, lh_wrist_idx: int = 20, rh_wrist_idx: int = 21):
    r_rot_quat, r_pos = recover_root_rot_pos(data_t)
    if not use_root_loss:
        # translation off
        r_pos = r_pos.clone()
        r_pos[..., 0] = 0.0
        r_pos[..., 2] = 0.0
        # rotation off (identity quaternion, w=1)
        r_rot_quat = torch.zeros_like(r_rot_quat)
        r_rot_quat[..., 0] = 1.0
    positions = data_t[..., 4:(joints_num - 1) * 3 + 4].view(data_t.shape[:-1] + (-1, 3))
    positions = qrot(
        r_rot_quat[..., None, :].expand(positions.shape[:-1] + (4,)),
        positions
    )
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = torch.cat([positions[:,:, :base_idx], r_pos.unsqueeze(-2), positions[:,:, base_idx:]], dim=-2)

    # ---- hand local: undo base XZ and add wrist position ----
    if hand_local:
        if joints_num > 52:  # with fingertips (62 joints)
            lh_joints = list(range(22, 37)) + list(range(52, 57))
            rh_joints = list(range(37, 52)) + list(range(57, 62))
        else:
            lh_joints = list(range(22, 37))
            rh_joints = list(range(37, 52))
        # Hand RIC was stored wrist-relative: R_inv*(hand - wrist).
        # After un-rotation + base XZ add, hand = (hand - wrist) + base_XZ.
        # Need to subtract the wrongly-added base_XZ and add wrist instead.
        base_xz = torch.zeros_like(r_pos.unsqueeze(-2))  # (B, T, 1, 3)
        base_xz[..., 0] = r_pos[..., 0:1]
        base_xz[..., 2] = r_pos[..., 2:3]
        positions[:, :, lh_joints] += positions[:, :, lh_wrist_idx:lh_wrist_idx+1] - base_xz
        positions[:, :, rh_joints] += positions[:, :, rh_wrist_idx:rh_wrist_idx+1] - base_xz

    return positions


def recover_hand_only_joints(
    mh: torch.Tensor,
    *,
    include_fingertips: bool = False,
    hand_root_dim: int = 0,
    joints_num: int = 52,
    hand_local: bool = True,
    lh_wrist_idx: int = 20,
    rh_wrist_idx: int = 21,
    lh_wrist_world: torch.Tensor = None,
    rh_wrist_world: torch.Tensor = None,
):
    """
    Reconstruct HOT3D hand-only joints from cached hand features.

    If lh_wrist_world / rh_wrist_world are provided (T,3), use them directly
    as wrist positions in world frame. Otherwise fall back to velocity
    integration from hand_root (anchored at origin).
    """
    squeeze = False
    if mh.ndim == 2:
        mh = mh.unsqueeze(0)
        squeeze = True
        if lh_wrist_world is not None and lh_wrist_world.ndim == 2:
            lh_wrist_world = lh_wrist_world.unsqueeze(0)
        if rh_wrist_world is not None and rh_wrist_world.ndim == 2:
            rh_wrist_world = rh_wrist_world.unsqueeze(0)
    if mh.ndim != 3:
        raise ValueError(f"Expected mh with shape (B,T,D) or (T,D), got {tuple(mh.shape)}")

    B, T, _ = mh.shape
    out = mh.new_zeros((B, T, joints_num, 3))

    if hand_root_dim < 0 or hand_root_dim % 2 != 0:
        raise ValueError(f"hand_root_dim must be a non-negative even integer, got {hand_root_dim}")

    # Wrist positions
    if lh_wrist_world is not None and rh_wrist_world is not None:
        lh_wrist = lh_wrist_world.to(mh.device)
        rh_wrist = rh_wrist_world.to(mh.device)
    else:
        per_hand_root_dim = hand_root_dim // 2
        if per_hand_root_dim > 0:
            lh_root = mh[..., :per_hand_root_dim]
            rh_root = mh[..., per_hand_root_dim:hand_root_dim]
            lh_wrist = mh.new_zeros((B, T, 3))
            rh_wrist = mh.new_zeros((B, T, 3))
            if per_hand_root_dim >= 3:
                lh_wrist[..., 1:, :] = torch.cumsum(lh_root[..., :-1, :3], dim=-2)
                rh_wrist[..., 1:, :] = torch.cumsum(rh_root[..., :-1, :3], dim=-2)
        else:
            lh_wrist = mh.new_zeros((B, T, 3))
            rh_wrist = mh.new_zeros((B, T, 3))

    out[..., lh_wrist_idx, :] = lh_wrist
    out[..., rh_wrist_idx, :] = rh_wrist

    mh_local = mh[..., hand_root_dim:]
    ric_dim = 120 if include_fingertips else 90
    hand_joint_count = 40 if include_fingertips else 30
    ric = mh_local[..., :ric_dim].view(B, T, hand_joint_count, 3)

    lh_fingers = ric[..., :15, :]
    rh_fingers = ric[..., 15:30, :]
    if hand_local:
        lh_fingers = lh_fingers + lh_wrist.unsqueeze(-2)
        rh_fingers = rh_fingers + rh_wrist.unsqueeze(-2)
    out[..., 22:37, :] = lh_fingers
    out[..., 37:52, :] = rh_fingers

    if include_fingertips:
        lh_tips = ric[..., 30:35, :]
        rh_tips = ric[..., 35:40, :]
        if hand_local:
            lh_tips = lh_tips + lh_wrist.unsqueeze(-2)
            rh_tips = rh_tips + rh_wrist.unsqueeze(-2)
        out[..., 52:57, :] = lh_tips
        out[..., 57:62, :] = rh_tips

    if squeeze:
        out = out.squeeze(0)
    return out


def recover_joints_from_body_hand(
    mb: torch.Tensor,
    mh: torch.Tensor,
    *,
    include_fingertips: bool = False,
    hand_root_dim: int = 0,
    joints_num: int = None,
    use_root_loss: bool = True,
    base_idx: int = 0,
    hand_local: bool = False,
    hand_only: bool = False,
):
    """Unified body/hand feature-to-joints reconstruction helper."""
    if joints_num is None:
        joints_num = 62 if include_fingertips else 52

    if hand_only:
        return recover_hand_only_joints(
            mh,
            include_fingertips=include_fingertips,
            hand_root_dim=hand_root_dim,
            joints_num=joints_num,
            hand_local=hand_local,
        )

    x623 = reconstruct_623_from_body_hand(
        mb, mh,
        include_fingertips=include_fingertips,
        hand_root_dim=hand_root_dim,
    )
    return recover_from_ric(
        x623,
        joints_num=joints_num,
        use_root_loss=use_root_loss,
        base_idx=base_idx,
        hand_local=hand_local,
    )


def get_bone_pairs(kinematic_chain):
    """Extract bone (parent, child) pairs from kinematic chain."""
    pairs = []
    for chain in kinematic_chain:
        for i in range(len(chain) - 1):
            pairs.append((chain[i], chain[i + 1]))
    return pairs


def compute_bone_lengths(positions, bone_pairs):
    """
    positions: (B, T, J, 3)
    bone_pairs: list of (parent_idx, child_idx)
    returns: (B, T, num_bones) bone lengths
    """
    parent = torch.stack([positions[..., p, :] for p, c in bone_pairs], dim=-2)
    child = torch.stack([positions[..., c, :] for p, c in bone_pairs], dim=-2)
    return torch.norm(child - parent, dim=-1)


def _recon_to_623(recon, mB, mH):
    """
    H2VQ_CNNTransformer の実装差異吸収：
      - recon が Tensor (B,T,623) の場合
      - recon が dict で body/hand 別の場合
    """
    if torch.is_tensor(recon):
        return recon
    if isinstance(recon, dict):
        # よくあるキー候補を吸収
        for kb, kh in [("mB_hat", "mH_hat"), ("reconB", "reconH"), ("body", "hand"), ("B", "H")]:
            if kb in recon and kh in recon:
                return torch.cat([recon[kb], recon[kh]], dim=-1)
        # もし dict だけど見つからないなら落とす
        raise KeyError(f"Unknown recon dict keys: {list(recon.keys())}")
    raise TypeError(f"Unknown recon type: {type(recon)}")


def build_model_from_args(args, device):
    model = H2VQ_CNNTransformer(
        T=args.T,
        body_in_dim=263,
        hand_in_dim=360,
        code_dim=args.code_dim,
        K=args.K,
        ema_decay=args.ema_decay,
        alpha_commit=args.alpha_commit,
        body_tokens_per_t=args.body_tokens_per_t,
        hand_tokens_per_t=args.hand_tokens_per_t,
        body_down=args.body_down,
        hand_down=args.hand_down,
        enc_depth=args.enc_depth,
        enc_heads=args.enc_heads,
        mlp_ratio=args.mlp_ratio,
    ).to(device)
    return model


def load_ckpt_to_model(model, ckpt_path: str, strict: bool = True):
    sd = torch.load(ckpt_path, map_location="cpu")
    state = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    missing, unexpected = model.load_state_dict(state, strict=strict)
    return sd, missing, unexpected


def count_params(model):
    n_all = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_all, n_train


