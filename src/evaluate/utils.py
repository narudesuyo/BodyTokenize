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

def reconstruct_623_from_body_hand(mb: torch.Tensor, mh: torch.Tensor, include_fingertips: bool = False):
    """
    mb: (B,T,263)
    mh: (B,T,360)
    return: x623 (B,T,623)
    """

    B, T, _ = mb.shape
    device = mb.device

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
        ric_hand = mh[..., j:j+120].view(B,T,40,3); j += 120
        rot_hand = mh[..., j:j+240].view(B,T,40,6); j += 240
        vel_hand = mh[..., j:j+120].view(B,T,40,3); j += 120
        NO_ROOT_J = 61
    else:
        NO_ROOT_J = 51
        j = 0
        ric_hand = mh[..., j:j+90].view(B,T,30,3); j += 90
        rot_hand = mh[..., j:j+180].view(B,T,30,6); j += 180
        vel_hand = mh[..., j:j+90].view(B,T,30,3); j += 90    

    # ---------- reassemble original order ----------
    # ric: 51 = body(21) + hand(30)
    ric = torch.cat([ric_body, ric_hand], dim=2).reshape(B,T,NO_ROOT_J*3)

    # rot: 51 = body(21) + hand(30)
    rot = torch.cat([rot_body, rot_hand], dim=2).reshape(B,T,NO_ROOT_J*6)

    # vel: 52 = body(root+21) + hand(30)
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
    r_pos = qrot(qinv(r_rot_quat), r_pos)
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
        # r_rot_quat = torch.zeros_like(r_rot_quat)
        # r_rot_quat[..., 0] = 1.0
    positions = data_t[..., 4:(joints_num - 1) * 3 + 4].view(data_t.shape[:-1] + (-1, 3))
    positions = qrot(
        qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)),
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



