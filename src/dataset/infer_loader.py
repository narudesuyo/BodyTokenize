import sys
sys.path.append(".")

import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocess.paramUtil import t2m_raw_offsets, t2m_body_hand_kinematic_chain
from preprocess.paramUtil_add_tips import (
    t2m_raw_offsets_with_tips,
    t2m_body_hand_kinematic_chain_with_tips,
)
from common.skeleton import Skeleton
from src.dataset.kp3d2motion_rep import kp3d_to_motion_rep


class MotionInferenceDataset(Dataset):
    """
    - pt_path の中の特定 key を取り出す
    - clip_len でスライド分割 (stride = clip_len - overlap)
    - 最後が足りなければ最後フレームで pad して必ず clip_len にする
    - MotionDataset と同じ 623 split (body/hand) を返す
    """

    def __init__(
        self,
        pt_path: str,
        key: str,
        clip_len: int = 80,
        overlap: int = 0,
        to_torch: bool = True,
        feet_thre: float = 0.002,
        kp_field: str = "kp3d",
        assume_y_up: bool = True,
        include_fingertips: bool = False,
        tgt_offsets_from: str = "first_frame",  # "first_frame" or "from_key"
    ):
        super().__init__()

        if overlap < 0 or overlap >= clip_len:
            raise ValueError(f"overlap must satisfy 0 <= overlap < clip_len, got {overlap} vs {clip_len}")

        self.pt_path = pt_path
        self.key = key
        self.clip_len = int(clip_len)
        self.overlap = int(overlap)
        self.stride = self.clip_len - self.overlap

        self.to_torch = to_torch
        self.feet_thre = feet_thre
        self.kp_field = kp_field
        self.assume_y_up = assume_y_up
        self.include_fingertips = include_fingertips

        # ---- load db ----
        self.db = torch.load(pt_path, map_location="cpu", weights_only=False)
        if key not in self.db:
            raise KeyError(f"key not found: {key}")

        item = self.db[key]
        if not isinstance(item, dict) or (kp_field not in item):
            raise KeyError(f"'{kp_field}' not found in db[{key}]")

        kp = item[kp_field]
        if torch.is_tensor(kp):
            kp = kp.detach().cpu().numpy()

        if kp.ndim != 3 or kp.shape[2] != 3 or kp.shape[1] < 55:
            raise ValueError(f"bad kp shape: {kp.shape} (need (T,J,3) with J>=55)")

        # ---- 623 block boundaries (MotionDataset と同じ) ----
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

        # ---- skeleton stuff (MotionDataset と同じ) ----
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

        # ---- kp -> kp52_full (MotionDataset と同じ定義) ----
        if self.include_fingertips:
            kp52_full = np.concatenate([kp[:, :22, :], kp[:, 25:55, :], kp[:, -10:, :]], axis=1)
        else:
            kp52_full = np.concatenate([kp[:, :22, :], kp[:, 25:55, :]], axis=1)

        self.kp = kp
        self.kp52_full = kp52_full
        self.Tfull = int(kp52_full.shape[0])

        # ---- tgt_offsets ----
        if tgt_offsets_from == "from_key":
            pos0 = kp52_full[0]
        else:
            # "first_frame" と同じ（この dataset は key 固定なので同義）
            pos0 = kp52_full[0]

        tgt_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, "cpu")
        self.tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(pos0).float())

        # ---- number of clips (pad 前提で ceil) ----
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

        kp52 = self.kp52_full[start:end]  # (t, J, 3)
        t = kp52.shape[0]
        if t < L:
            pad = np.repeat(kp52[-1:, :, :], L - t, axis=0)
            kp52 = np.concatenate([kp52, pad], axis=0)

        # ---- kp3d -> motion rep ----
        arr = kp3d_to_motion_rep(
            kp3d_52_yup=kp52,
            feet_thre=self.feet_thre,
            tgt_offsets=self.tgt_offsets,
            n_raw_offsets=self.n_raw_offsets,
            kinematic_chain=self.kinematic_chain,
        )

        # ---- split (MotionDataset と同じ) ----
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

        body = np.concatenate(
            [
                root,
                ric_body.reshape(Tm1, -1),
                rot_body.reshape(Tm1, -1),
                vel_body.reshape(Tm1, -1),
                feet,
            ],
            axis=1,
        )
        hand = np.concatenate(
            [
                ric_hand.reshape(Tm1, -1),
                rot_hand.reshape(Tm1, -1),
                vel_hand.reshape(Tm1, -1),
            ],
            axis=1,
        )

        out = {
            "key": self.key,
            "clip_index": int(i),
            "start": int(start),
            "end": int(min(end, self.Tfull)),
            "Tfull": int(self.Tfull),
            "T": int(Tm1),
            "body": torch.from_numpy(body).float() if self.to_torch else body,
            "hand": torch.from_numpy(hand).float() if self.to_torch else hand,
            "kp": torch.from_numpy(self.kp).float() if self.to_torch else self.kp,
            "kp52": torch.from_numpy(kp52).float() if self.to_torch else kp52,
        }
        return out