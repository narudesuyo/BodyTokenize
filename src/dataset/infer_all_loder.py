import torch
import numpy as np
import math
from torch.utils.data import Dataset
from tqdm import tqdm

# Adjust imports based on your file structure
from preprocess.paramUtil import t2m_raw_offsets, t2m_body_hand_kinematic_chain
from preprocess.paramUtil_add_tips import (
    t2m_raw_offsets_with_tips,
    t2m_body_hand_kinematic_chain_with_tips,
)
from common.skeleton import Skeleton
from src.dataset.kp3d2motion_rep import kp3d_to_motion_rep


class MotionAllInferenceDataset(Dataset):
    """
    Loads a .pt file ONCE and iterates over ALL clips from ALL valid keys.
    """

    def __init__(
        self,
        pt_path: str,
        clip_len: int = 20,
        overlap: int = 1, # stride = clip_len - overlap
        to_torch: bool = True,
        feet_thre: float = 0.002,
        kp_field: str = "kp3d",
        assume_y_up: bool = True,
        include_fingertips: bool = False,
    ):
        super().__init__()
        
        self.clip_len = int(clip_len)
        self.overlap = int(overlap)
        self.stride = self.clip_len - self.overlap
        if self.stride <= 0:
            raise ValueError(f"stride must be > 0. clip_len={clip_len}, overlap={overlap}")

        self.to_torch = to_torch
        self.feet_thre = feet_thre
        self.kp_field = kp_field
        self.assume_y_up = assume_y_up
        self.include_fingertips = include_fingertips

        # ---- 1. Load DB Once ----
        print(f"Loading PT file: {pt_path} ...")
        self.db = torch.load(pt_path, map_location="cpu",weights_only=False) # weights_only=False might be needed
        
        # ---- 2. Build Index (Flatten all clips) ----
        self.samples = [] 
        
        valid_keys = 0
        keys = sorted(list(self.db.keys())) # Sort for deterministic order
        
        print("Indexing all clips...")
        for key in tqdm(keys, desc="Indexing"):
            item = self.db[key]
            
            if not isinstance(item, dict) or (kp_field not in item):
                continue
            
            kp = item[kp_field]
            if torch.is_tensor(kp):
                kp = kp.detach().cpu().numpy()
            
            # Shape Check
            # Assuming minimum joints 51 or 55 based on your logic
            min_joints = 55 if include_fingertips else 51
            if kp.ndim != 3 or kp.shape[2] != 3 or kp.shape[1] < min_joints:
                continue
                
            Tfull = kp.shape[0]
            
            # Calculate number of clips
            if Tfull <= self.clip_len:
                n_clips = 1
            else:
                n_clips = math.ceil((Tfull - self.clip_len) / self.stride) + 1
            
            # Register every clip
            for i in range(n_clips):
                start = i * self.stride
                self.samples.append({
                    "key": key,
                    "clip_index": i,
                    "start": start
                })
            valid_keys += 1

        print(f"Indexed {len(self.samples)} clips from {valid_keys} valid keys.")

        # ---- Skeleton Settings (Same as before) ----
        if self.include_fingertips:
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
        return len(self.samples)

    def __getitem__(self, idx):
        # Retrieve metadata
        meta = self.samples[idx]
        key = meta['key']
        clip_idx = meta['clip_index']
        start = meta['start']
        
        # Get raw data (Pointer access is fast)
        item = self.db[key]
        kp = item[self.kp_field]
        if torch.is_tensor(kp):
            kp = kp.detach().cpu().numpy()
            
        # Construct kp52_full
        if self.include_fingertips:
            kp52_full = np.concatenate([kp[:, :22, :], kp[:, 25:55, :], kp[:, -10:, :]], axis=1)
        else:
            kp52_full = np.concatenate([kp[:, :22, :], kp[:, 25:55, :]], axis=1)
            
        Tfull = kp52_full.shape[0]

        # Crop
        L = self.clip_len
        end = start + L
        kp52 = kp52_full[start:end]
        
        # Padding
        t_current = kp52.shape[0]
        if t_current < L:
            pad = np.repeat(kp52[-1:, :, :], L - t_current, axis=0)
            kp52 = np.concatenate([kp52, pad], axis=0)

        # Target Offsets (Calc per sample)
        pos0 = kp52_full[0]
        tgt_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, "cpu")
        tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(pos0).float())

        # Conversion
        arr = kp3d_to_motion_rep(
            kp3d_52_yup=kp52,
            feet_thre=self.feet_thre,
            tgt_offsets=tgt_offsets,
            n_raw_offsets=self.n_raw_offsets,
            kinematic_chain=self.kinematic_chain,
        )

        # Split
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

        body = np.concatenate([root, ric_body.reshape(Tm1, -1), rot_body.reshape(Tm1, -1), vel_body.reshape(Tm1, -1), feet], axis=1)
        hand = np.concatenate([ric_hand.reshape(Tm1, -1), rot_hand.reshape(Tm1, -1), vel_hand.reshape(Tm1, -1)], axis=1)

        out = {
            "mB": torch.from_numpy(body).float() if self.to_torch else body,
            "mH": torch.from_numpy(hand).float() if self.to_torch else hand,
            # Metadata for saving files correctly
            "key": key,
            "save_idx": int(clip_idx), 
        }
        return out