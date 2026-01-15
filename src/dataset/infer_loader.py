import torch
from torch.utils.data import Dataset
import numpy as np
from preprocess.paramUtil import t2m_raw_offsets, t2m_body_hand_kinematic_chain
from common.skeleton import Skeleton
from src.dataset.kp3d2motion_rep import kp3d_to_motion_rep
import math

class MotionInferenceDataset(Dataset):
    def __init__(
        self,
        pt_path: str,
        key: str,
        clip_len: int = 80,
        overlap: int = 0,          # ★追加
        to_torch: bool = True,
        feet_thre: float = 0.002,
        kp_field: str = "kp3d",
        assume_y_up: bool = True,
    ):
        super().__init__()

        if overlap < 0 or overlap >= clip_len:
            raise ValueError(
                f"overlap must satisfy 0 <= overlap < clip_len, "
                f"got overlap={overlap}, clip_len={clip_len}"
            )

        self.clip_len = int(clip_len)
        self.overlap = int(overlap)
        self.stride = self.clip_len - self.overlap   # ★核心
        self.db = torch.load(pt_path, map_location="cpu", weights_only=False)
        if key not in self.db:
            raise KeyError(f"key not found: {key}")

        self.key = key
        self.clip_len = int(clip_len)
        self.to_torch = to_torch
        self.feet_thre = feet_thre
        self.kp_field = kp_field
        self.assume_y_up = assume_y_up

        # ---- 623 block boundaries ----
        self.NO_ROOT_J = 51
        self.I_ROOT0 = 0
        self.I_ROOT1 = 4
        self.I_RIC0  = self.I_ROOT1
        self.I_RIC1  = self.I_RIC0 + self.NO_ROOT_J * 3
        self.I_ROT0  = self.I_RIC1
        self.I_ROT1  = self.I_ROT0 + self.NO_ROOT_J * 6
        self.I_VEL0  = self.I_ROT1
        self.I_VEL1  = self.I_VEL0 + 52 * 3
        self.I_FEET0 = self.I_VEL1
        self.I_FEET1 = self.I_FEET0 + 4

        # ---- skeleton ----
        self.n_raw_offsets = torch.from_numpy(t2m_raw_offsets).float()
        self.kinematic_chain = t2m_body_hand_kinematic_chain

        item = self.db[self.key]
        kp = item[self.kp_field]
        if torch.is_tensor(kp):
            kp = kp.detach().cpu().numpy()

        # ここはあなたの include_fingertips に合わせて固定（必要なら引数化）
        # fingertips無し版:
        kp52_full = np.concatenate([kp[:, :22, :], kp[:, 25:55, :]], axis=1)  # (T,52,3)

        self.kp = kp
        self.kp52_full = kp52_full
        self.Tfull = kp52_full.shape[0]

        # clip数（最後はpadして必ず作る）
        L = self.clip_len
        self.n_clips = (self.Tfull + L - 1) // L  # ceil

        # tgt_offsets: 最初のフレームから
        pos0 = kp52_full[0]
        tgt_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, "cpu")
        self.tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(pos0).float())

    def __len__(self):
        self.Tfull = self.kp52_full.shape[0]
        if self.Tfull <= self.clip_len:
            self.n_clips = 1
        else:
            self.n_clips = math.ceil(
                (self.Tfull - self.clip_len) / self.stride
            ) + 1
        return self.n_clips

    def __getitem__(self, i: int):
        L = self.clip_len
        start = int(i) * self.stride
        end = start + L

        kp52 = self.kp52_full[start:end]  # (t,52,3)
        t = kp52.shape[0]
        if t < L:
            pad = np.repeat(kp52[-1:, :, :], L - t, axis=0)
            kp52 = np.concatenate([kp52, pad], axis=0)

        # kp52 -> motion rep
        arr = kp3d_to_motion_rep(
            kp3d_52_yup=kp52,
            feet_thre=self.feet_thre,
            tgt_offsets=self.tgt_offsets,
            n_raw_offsets=self.n_raw_offsets,
            kinematic_chain=self.kinematic_chain,
        )

        # split to body/hand (same as yours)
        Tm1 = arr.shape[0]
        root = arr[:, self.I_ROOT0:self.I_ROOT1]
        ric  = arr[:, self.I_RIC0:self.I_RIC1].reshape(Tm1, self.NO_ROOT_J, 3)
        rot  = arr[:, self.I_ROT0:self.I_ROT1].reshape(Tm1, self.NO_ROOT_J, 6)
        vel  = arr[:, self.I_VEL0:self.I_VEL1].reshape(Tm1, 52, 3)
        feet = arr[:, self.I_FEET0:self.I_FEET1]

        ric_body, ric_hand = ric[:, :21], ric[:, 21:51]
        rot_body, rot_hand = rot[:, :21], rot[:, 21:51]
        vel_body, vel_hand = vel[:, :22], vel[:, 22:52]

        body = np.concatenate(
            [root,
             ric_body.reshape(Tm1, -1),
             rot_body.reshape(Tm1, -1),
             vel_body.reshape(Tm1, -1),
             feet],
            axis=1
        )
        hand = np.concatenate(
            [ric_hand.reshape(Tm1, -1),
             rot_hand.reshape(Tm1, -1),
             vel_hand.reshape(Tm1, -1)],
            axis=1
        )

        out = {
            "key": self.key,
            "clip_index": int(i),
            "start": int(start),
            "end": int(min(end, self.Tfull)),
            "Tfull": int(self.Tfull),
            "body": torch.from_numpy(body).float() if self.to_torch else body,
            "hand": torch.from_numpy(hand).float() if self.to_torch else hand,
            "kp52": torch.from_numpy(kp52).float() if self.to_torch else kp52,
        }
        return out