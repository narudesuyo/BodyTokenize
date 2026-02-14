import sys
sys.path.append(".")

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocess.paramUtil import t2m_raw_offsets, t2m_body_hand_kinematic_chain
from preprocess.paramUtil_add_tips import t2m_raw_offsets_with_tips, t2m_body_hand_kinematic_chain_with_tips
from common.skeleton import Skeleton
from src.dataset.kp3d2motion_rep import kp3d_to_motion_rep


class MotionDataset(Dataset):
    def __init__(
        self,
        pt_path: str,
        to_torch: bool = True,
        feet_thre: float = 0.002,
        keys=None,
        kp_field: str = "kp3d",
        assume_y_up: bool = True,
        clip_len: int = 80,          # ★追加
        random_crop: bool = True,    # ★追加（Falseなら先頭から）
        pad_if_short: bool = True,  # ★追加（短いclipをどうするか）
        include_fingertips: bool = False,
        tgt_offsets_path: str = "tgt_offsets.npy",
        base_idx: int = 0,
        hand_local: bool = False,
        use_cache: bool = False,
    ):
        self.base_idx = base_idx
        self.hand_local = hand_local
        self.use_cache = use_cache
        super().__init__()
        self.pt_path = pt_path
        self.to_torch = to_torch
        self.feet_thre = feet_thre
        self.kp_field = kp_field
        self.assume_y_up = assume_y_up
        self.clip_len = clip_len
        self.random_crop = random_crop
        self.pad_if_short = pad_if_short
        self.include_fingertips = include_fingertips
        self.tgt_offsets_path = tgt_offsets_path
        # ---- load db once ----
        self.db = torch.load(pt_path, map_location="cpu", weights_only=False)
        assert isinstance(self.db, dict), f"Expected dict in pt, got {type(self.db)}"

        if keys is None:
            self.keys = list(self.db.keys())
        else:
            self.keys = list(keys)

        # ---- cache mode: precomputed body/hand, skip skeleton setup ----
        if self.use_cache:
            return

        # ---- 623 block boundaries (same as your Motion623SplitDataset) ----
        if self.include_fingertips:
            self.NO_ROOT_J = 61
        else:
            self.NO_ROOT_J = 51
        self.I_ROOT0 = 0
        self.I_ROOT1 = 4
        self.I_RIC0  = self.I_ROOT1
        self.I_RIC1  = self.I_RIC0 + self.NO_ROOT_J * 3
        self.I_ROT0  = self.I_RIC1
        self.I_ROT1  = self.I_ROT0 + self.NO_ROOT_J * 6
        self.I_VEL0  = self.I_ROT1
        self.I_VEL1  = self.I_VEL0 + (self.NO_ROOT_J+1) * 3
        self.I_FEET0 = self.I_VEL1
        self.I_FEET1 = self.I_FEET0 + 4

        # ---- skeleton stuff ----
        self.n_raw_offsets = torch.from_numpy(t2m_raw_offsets_with_tips).float() if include_fingertips else torch.from_numpy(t2m_raw_offsets).float()
        self.kinematic_chain = t2m_body_hand_kinematic_chain_with_tips if include_fingertips else t2m_body_hand_kinematic_chain


        # ---- target offsets: first valid clip's first frame ----
        self.tgt_offsets = None
        for k in self.keys:
            item = self.db.get(k, None)
            if not isinstance(item, dict) or (self.kp_field not in item):
                continue
            kp0 = item[self.kp_field]
            if torch.is_tensor(kp0):
                kp0 = kp0.detach().cpu().numpy()
            if kp0.ndim == 3 and kp0.shape[1] >= 52 and kp0.shape[2] == 3:
                tgt_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, "cpu")
                # self.tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(kp0[0, :52]).float())
                if self.include_fingertips:
                    pos0 = np.concatenate([kp0[0, :22, :], kp0[0, 25:55, :], kp0[0, -10:, :]], axis=0)  # positions
                else:
                    pos0 = np.concatenate([kp0[0, :22, :], kp0[0, 25:55, :]], axis=0)  # positions
                tgt_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, "cpu")
                self.tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(pos0).float())
                break

        if self.tgt_offsets is None:
            raise RuntimeError(f"No valid kp3d found in {pt_path} under field '{kp_field}'")

    def __len__(self):
        return len(self.keys)
    import warnings
    import numpy as np
    import torch

    def _getitem_cache(self, idx):
        """Cache mode: load precomputed body/hand directly."""
        L = self.clip_len
        idx = int(idx) % len(self.keys)

        max_resample_key = 20
        for key_try in range(max_resample_key):
            key = self.keys[idx]
            item = self.db.get(key, None)

            if not isinstance(item, dict) or "body" not in item or "hand" not in item:
                idx = np.random.randint(0, len(self.keys))
                continue

            body = item["body"]  # [T, 263]
            hand = item["hand"]  # [T, 480]
            if torch.is_tensor(body):
                body = body.numpy()
            if torch.is_tensor(hand):
                hand = hand.numpy()

            Tfull = body.shape[0]

            # ---------- crop / pad ----------
            if L is not None and L > 0:
                # clip_len is T+1 frames for raw kp3d, but precomputed is already T-1
                # body/hand have shape [Tfull, D] where Tfull = original_frames - 1
                target_len = L - 1  # match the original __getitem__ output length
                if Tfull >= target_len:
                    s = np.random.randint(0, Tfull - target_len + 1) if self.random_crop else 0
                    body = body[s:s + target_len]
                    hand = hand[s:s + target_len]
                else:
                    if self.pad_if_short:
                        s = 0
                        pad_body = np.repeat(body[-1:], target_len - Tfull, axis=0)
                        pad_hand = np.repeat(hand[-1:], target_len - Tfull, axis=0)
                        body = np.concatenate([body, pad_body], axis=0)
                        hand = np.concatenate([hand, pad_hand], axis=0)
                    else:
                        idx = np.random.randint(0, len(self.keys))
                        continue
            else:
                s = 0
                target_len = Tfull

            out = {
                "key": key,
                "T": int(body.shape[0]),
                "body": torch.from_numpy(body).float() if self.to_torch else body,
                "hand": torch.from_numpy(hand).float() if self.to_torch else hand,
                "start": int(s),
                "Tfull": int(Tfull),
            }
            return out

        raise RuntimeError("Too many invalid samples in cache.")

    def __getitem__(self, idx):
        if self.use_cache:
            return self._getitem_cache(idx)

        max_try_per_key = 10
        max_resample_key = 20
        verbose_nan = True   # ★ Falseにすれば黙る

        L = self.clip_len
        idx = int(idx) % len(self.keys)

        for key_try in range(max_resample_key):
            key = self.keys[idx]
            item = self.db.get(key, None)

            if not isinstance(item, dict) or (self.kp_field not in item):
                if verbose_nan:
                    print(f"[MotionDataset] skip key={key}: invalid structure")
                idx = np.random.randint(0, len(self.keys))
                continue

            kp = item[self.kp_field]
            if torch.is_tensor(kp):
                kp = kp.detach().cpu().numpy()

            if kp.ndim != 3 or kp.shape[2] != 3 or kp.shape[1] < 52:
                if verbose_nan:
                    print(f"[MotionDataset] skip key={key}: bad kp shape {kp.shape}")
                idx = np.random.randint(0, len(self.keys))
                continue

            if self.include_fingertips:
                kp52_full = np.concatenate([kp[:, :22, :], kp[:, 25:55, :], kp[:, -10:, :]], axis=1)
            else:
                kp52_full = np.concatenate([kp[:, :22, :], kp[:, 25:55, :]], axis=1)
            Tfull = kp52_full.shape[0]
            can_retry_s = (L is not None and L > 0 and Tfull >= L)

            for s_try in range(max_try_per_key if can_retry_s else 1):
                # ---------- crop ----------
                if L is not None and L > 0:
                    if Tfull >= L:
                        s = np.random.randint(0, Tfull - L + 1) if self.random_crop else 0
                        kp52 = kp52_full[s:s + L]
                    else:
                        if self.pad_if_short:
                            s = 0
                            pad = np.repeat(kp52_full[-1:, :, :], L - Tfull, axis=0)
                            kp52 = np.concatenate([kp52_full, pad], axis=0)
                        else:
                            break
                else:
                    s = 0
                    kp52 = kp52_full

                # ---------- NaN check (kp) ----------
                if np.isnan(kp52).any() or np.isinf(kp52).any():
                    nan_mask = ~np.isfinite(kp52)
                    idx = np.argwhere(nan_mask)
                    print(f"[NaN@kp52] key={key} start={s} try={s_try}")
                    print(f"kp52 shape = {kp52.shape}")
                    print("first 20 NaN indices:", idx[:20])
                    exit()

                # ---------- kp3d -> motion rep ----------
                # try:
                #     arr = kp3d_to_motion_rep(
                #         kp3d_52_yup=kp52,
                #         feet_thre=self.feet_thre,
                #         tgt_offsets=self.tgt_offsets,
                #         n_raw_offsets=self.n_raw_offsets,
                #         kinematic_chain=self.kinematic_chain,
                #     )
                # except Exception as e:
                #     if verbose_nan:
                #         print(
                #             f"[ERROR@kp3d_to_motion_rep] key={key} "
                #             f"start={s} try={s_try}\n  {repr(e)}"
                #         )
                #     continue
                arr = kp3d_to_motion_rep(
                    kp3d_52_yup=kp52,
                    feet_thre=self.feet_thre,
                    tgt_offsets=self.tgt_offsets,
                    n_raw_offsets=self.n_raw_offsets,
                    kinematic_chain=self.kinematic_chain,
                    base_idx=self.base_idx,
                    hand_local=self.hand_local,
                )

                # ---------- NaN check (arr) ----------
                if np.isnan(arr).any() or np.isinf(arr).any():
                    nan_mask = ~np.isfinite(arr)
                    idx = np.argwhere(nan_mask)
                    print(f"[NaN@arr] key={key} start={s} try={s_try}")
                    print(f"arr shape = {arr.shape}")
                    print("first 20 NaN indices:", idx)
                    exit()

                # ---------- split ----------
                Tm1 = arr.shape[0]
                root = arr[:, self.I_ROOT0:self.I_ROOT1]
                ric  = arr[:, self.I_RIC0:self.I_RIC1].reshape(Tm1, self.NO_ROOT_J, 3)
                rot  = arr[:, self.I_ROT0:self.I_ROT1].reshape(Tm1, self.NO_ROOT_J, 6)
                vel  = arr[:, self.I_VEL0:self.I_VEL1].reshape(Tm1, self.NO_ROOT_J+1, 3)
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
                    [root, ric_body.reshape(Tm1, -1),
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

                # ---------- NaN check (final) ----------
                if (
                    np.isnan(body).any() or np.isinf(body).any() or
                    np.isnan(hand).any() or np.isinf(hand).any()
                ):
                    if verbose_nan:
                        print(
                            f"[NaN@output] key={key} "
                            f"start={s} try={s_try}"
                        )
                    continue

                out = {
                    "key": key,
                    "T": int(Tm1),
                    "body": torch.from_numpy(body).float() if self.to_torch else body,
                    "hand": torch.from_numpy(hand).float() if self.to_torch else hand,
                    "start": int(s),
                    "Tfull": int(Tfull),
                    "kp" : torch.from_numpy(kp).float() if self.to_torch else kp,
                    "kp52" : torch.from_numpy(kp52).float() if self.to_torch else kp52,
                }
                return out

            # このkeyはダメ
            if verbose_nan:
                print(f"[MotionDataset] abandon key={key}")
            idx = np.random.randint(0, len(self.keys))

        raise RuntimeError("Too many NaN / invalid samples encountered.")



if __name__ == "__main__":
    import argparse
    from torch.utils.data import DataLoader
    import numpy as np
    import torch
    from tqdm import tqdm

    def collate_stack(batch):
        body = torch.stack([b["body"] for b in batch], dim=0)  # (B,T,263)
        hand = torch.stack([b["hand"] for b in batch], dim=0)  # (B,T,360)
        out = {"body": body, "hand": hand}
        return out

    @torch.no_grad()
    def compute_mean_var_623_split(dl, device="cpu", max_batches=-1, eps=1e-12, include_fingertips=False):
        # --- root: yaw(1), vel_xz(2), y(1) を別々に ---
        NO_ROOT_J = 61 if include_fingertips else 51
        I_RIC0 = 4
        I_RIC1 = I_RIC0 + NO_ROOT_J * 3
        I_ROT0 = I_RIC1
        I_ROT1 = I_ROT0 + NO_ROOT_J * 6
        I_VEL0 = I_ROT1
        I_VEL1 = I_VEL0 + (NO_ROOT_J+1) * 3
        I_FEET0 = I_VEL1
        I_FEET1 = I_FEET0 + 4


        n_yaw = 0
        mean_yaw = torch.zeros(1, device=device)
        M2_yaw   = torch.zeros(1, device=device)

        n_vxz = 0
        mean_vxz = torch.zeros(2, device=device)
        M2_vxz   = torch.zeros(2, device=device)

        n_ry = 0
        mean_ry = torch.zeros(1, device=device)
        M2_ry   = torch.zeros(1, device=device)

        # --- ric ---
        n_ric = 0
        mean_ric = torch.zeros(NO_ROOT_J*3, device=device)
        M2_ric   = torch.zeros(NO_ROOT_J*3, device=device)

        # --- vel ---
        n_vel = 0
        mean_vel = torch.zeros((NO_ROOT_J+1)*3, device=device)
        M2_vel   = torch.zeros((NO_ROOT_J+1)*3, device=device)

        for bi, batch in tqdm(enumerate(dl), desc="Computing mean/var (split+root)"):
            if max_batches > 0 and bi >= max_batches:
                break

            body = batch["body"].to(device, non_blocking=True)
            hand = batch["hand"].to(device, non_blocking=True)
            x = torch.cat([body, hand], dim=-1).float()  # (B,T,623)
            x = x.reshape(-1, I_FEET1)                       # (N,623)

            ok = torch.isfinite(x).all(dim=1)
            x = x[ok]
            if x.numel() == 0:
                continue

            # root parts
            xyaw = x[:, 0:1]     # (N,1)
            xvxz = x[:, 1:3]     # (N,2)
            xry  = x[:, 3:4]     # (N,1)

            # other blocks
            xric = x[:, I_RIC0:I_RIC1]   # (N,153)
            xvel = x[:, I_VEL0:I_VEL1] # (N,156)

            # --- update yaw ---
            for row in xyaw:
                n_yaw += 1
                delta = row - mean_yaw
                mean_yaw += delta / n_yaw
                delta2 = row - mean_yaw
                M2_yaw += delta * delta2

            # --- update vel_xz ---
            for row in xvxz:
                n_vxz += 1
                delta = row - mean_vxz
                mean_vxz += delta / n_vxz
                delta2 = row - mean_vxz
                M2_vxz += delta * delta2

            # --- update root_y ---
            for row in xry:
                n_ry += 1
                delta = row - mean_ry
                mean_ry += delta / n_ry
                delta2 = row - mean_ry
                M2_ry += delta * delta2

            # --- update ric ---
            for row in xric:
                n_ric += 1
                delta = row - mean_ric
                mean_ric += delta / n_ric
                delta2 = row - mean_ric
                M2_ric += delta * delta2

            # --- update vel ---
            for row in xvel:
                n_vel += 1
                delta = row - mean_vel
                mean_vel += delta / n_vel
                delta2 = row - mean_vel
                M2_vel += delta * delta2

            if (bi + 1) % 50 == 0:
                print(
                    f"[stats] batches={bi+1} frames="
                    f"yaw/vxz/ry/ric/vel={n_yaw}/{n_vxz}/{n_ry}/{n_ric}/{n_vel}"
                )

        # sanity
        if min(n_yaw, n_vxz, n_ry, n_ric, n_vel) < 2:
            raise RuntimeError(
                f"Not enough samples: yaw={n_yaw}, vxz={n_vxz}, ry={n_ry}, ric={n_ric}, vel={n_vel}"
            )

        var_yaw = M2_yaw / (n_yaw - 1)
        std_yaw = torch.sqrt(var_yaw.clamp_min(eps))

        var_vxz = M2_vxz / (n_vxz - 1)
        std_vxz = torch.sqrt(var_vxz.clamp_min(eps))

        var_ry = M2_ry / (n_ry - 1)
        std_ry = torch.sqrt(var_ry.clamp_min(eps))

        var_ric = M2_ric / (n_ric - 1)
        std_ric = torch.sqrt(var_ric.clamp_min(eps))

        var_vel = M2_vel / (n_vel - 1)
        std_vel = torch.sqrt(var_vel.clamp_min(eps))

        # ---- 623に組み立て（rot6d/footは mean=0, std=1）----
        mean_623 = torch.zeros(I_FEET1, device=device)
        var_623  = torch.ones(I_FEET1, device=device)
        std_623  = torch.ones(I_FEET1, device=device)

        # root
        mean_623[0:1] = mean_yaw
        std_623[0:1]  = std_yaw
        var_623[0:1]  = var_yaw

        mean_623[1:3] = mean_vxz
        std_623[1:3]  = std_vxz
        var_623[1:3]  = var_vxz

        mean_623[3:4] = mean_ry
        std_623[3:4]  = std_ry
        var_623[3:4]  = var_ry

        # ric
        mean_623[I_RIC0:I_RIC1] = mean_ric
        std_623[I_RIC0:I_RIC1]  = std_ric
        var_623[I_RIC0:I_RIC1]  = var_ric

        # vel
        mean_623[I_VEL0:I_VEL1] = mean_vel
        std_623[I_VEL0:I_VEL1]  = std_vel
        var_623[I_VEL0:I_VEL1]  = var_vel

        nframes_used = {"yaw": n_yaw, "vxz": n_vxz, "ry": n_ry, "ric": n_ric, "vel": n_vel}
        return mean_623, var_623, std_623, nframes_used

    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_path", default=f"{os.getenv('DATA_DIR')}/ee4d/ee4d_motion_uniegomotion/uniegomotion/ee_train_joints.pt")
    ap.add_argument("--clip_len", type=int, default=81)   # -> output T = clip_len-1
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_batches", type=int, default=-1)  # デバッグ用: 例えば 100 とか
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--prefix", default="motion623_fingertips")        # 保存名のprefix
    ap.add_argument("--include_fingertips", action="store_true")
    args = ap.parse_args()

    ds = MotionDataset(
        pt_path=args.pt_path,
        feet_thre=0.002,
        clip_len=args.clip_len,
        random_crop=True,     # train統計ならTrue推奨
        pad_if_short=True,
        to_torch=True,
        include_fingertips=args.include_fingertips,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_stack,

    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mean, var, std, nframes = compute_mean_var_623_split(dl, device=device, max_batches=args.max_batches, include_fingertips=args.include_fingertips)

    os.makedirs(args.out_dir, exist_ok=True)
    mean_np = mean.detach().cpu().numpy().astype(np.float32)
    var_np  = var.detach().cpu().numpy().astype(np.float32)
    std_np  = std.detach().cpu().numpy().astype(np.float32)

    mean_path = os.path.join(args.out_dir, f"{args.prefix}_mean.npy")
    var_path  = os.path.join(args.out_dir, f"{args.prefix}_var.npy")
    std_path  = os.path.join(args.out_dir, f"{args.prefix}_std.npy")

    np.save(mean_path, mean_np)
    np.save(var_path, var_np)
    np.save(std_path, std_np)

    print("saved:")
    print(" ", mean_path)
    print(" ", var_path)
    print(" ", std_path)
    print(f"frames used = {nframes}")