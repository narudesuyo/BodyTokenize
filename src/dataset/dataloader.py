# import os, glob
# import numpy as np
# import torch
# from torch.utils.data import Dataset

# class Motion623SplitDataset(Dataset):
#     """
#     directory 下の *.npy (shape=(T,623)) を読み込み、
#     body (263) / hand (360) に分割して返す。
#     """
#     def __init__(self, root_dir, pattern="*.npy", mmap=False, to_torch=True):
#         self.paths = sorted(glob.glob(os.path.join(root_dir, pattern)))
#         assert len(self.paths) > 0, f"No npy found in {root_dir}"
#         self.mmap = mmap
#         self.to_torch = to_torch

#         # joint 構成（52 joints）
#         self.BODY_J = 22
#         self.HAND_J = 30
#         self.NO_ROOT_J = 51  # 52 - 1

#         # 623 の各ブロック境界
#         self.I_ROOT0 = 0
#         self.I_ROOT1 = 4

#         self.I_RIC0  = self.I_ROOT1
#         self.I_RIC1  = self.I_RIC0 + self.NO_ROOT_J * 3      # 4..157

#         self.I_ROT0  = self.I_RIC1
#         self.I_ROT1  = self.I_ROT0 + self.NO_ROOT_J * 6      # 157..463

#         self.I_VEL0  = self.I_ROT1
#         self.I_VEL1  = self.I_VEL0 + 52 * 3                  # 463..619

#         self.I_FEET0 = self.I_VEL1
#         self.I_FEET1 = self.I_FEET0 + 4                      # 619..623

#     def __len__(self):
#         return len(self.paths)

#     def __getitem__(self, idx):
#         p = self.paths[idx]
#         arr = np.load(p, mmap_mode="r" if self.mmap else None)  # (T,623)
#         if arr.ndim != 2 or arr.shape[1] != 623:
#             raise ValueError(f"{p}: expected (T,623), got {arr.shape}")

#         T = arr.shape[0]

#         root = arr[:, self.I_ROOT0:self.I_ROOT1]                             # (T,4)

#         ric  = arr[:, self.I_RIC0:self.I_RIC1].reshape(T, self.NO_ROOT_J, 3) # (T,51,3)
#         rot  = arr[:, self.I_ROT0:self.I_ROT1].reshape(T, self.NO_ROOT_J, 6) # (T,51,6)
#         vel  = arr[:, self.I_VEL0:self.I_VEL1].reshape(T, 52, 3)             # (T,52,3)
#         feet = arr[:, self.I_FEET0:self.I_FEET1]                             # (T,4)

#         # 51 joints の並びは joint[1:]（root除外）なので：
#         # body(22) のうち rootを除いた 21 が先頭、hand 30 が後ろ
#         ric_body  = ric[:, :21]        # (T,21,3)
#         ric_hand  = ric[:, 21:51]      # (T,30,3)

#         rot_body  = rot[:, :21]        # (T,21,6)
#         rot_hand  = rot[:, 21:51]      # (T,30,6)

#         vel_body  = vel[:, :22]        # (T,22,3)  (root含む)
#         vel_hand  = vel[:, 22:52]      # (T,30,3)

#         # フラットにして body/hand ベクトルに戻す（必要なら）
#         body = np.concatenate([
#             root,                                  # 4
#             ric_body.reshape(T, -1),               # 63
#             rot_body.reshape(T, -1),               # 126
#             vel_body.reshape(T, -1),               # 66
#             feet                                  # 4
#         ], axis=1)                                  # => (T,263)

#         hand = np.concatenate([
#             ric_hand.reshape(T, -1),               # 90
#             rot_hand.reshape(T, -1),               # 180
#             vel_hand.reshape(T, -1),               # 90
#         ], axis=1)                                  # => (T,360)

#         out = {
#             "path": p,
#             "T": T,
#             "body": body,
#             "hand": hand,
#             # 必要なら構造化も返す
#             "root": root,
#             "feet": feet,
#             "ric_body": ric_body, "ric_hand": ric_hand,
#             "rot_body": rot_body, "rot_hand": rot_hand,
#             "vel_body": vel_body, "vel_hand": vel_hand,
#         }

#         if self.to_torch:
#             for k, v in list(out.items()):
#                 if isinstance(v, np.ndarray):
#                     out[k] = torch.from_numpy(np.asarray(v).copy()).float()
#         return out

# if __name__ == "__main__":
#     ds = Motion623SplitDataset(
#     f"{os.getenv('DATA_DIR')}/ee4d/ee4d_motion_uniegomotion/uniegomotion/ee_train_joints_motion_representation/new_joint_vecs",
#     mmap=True
#     )

#     batch0 = ds[0]
#     print(batch0["body"].shape, batch0["hand"].shape)  # (T,263) (T,360)
import sys
sys.path.append(".")

import os
import numpy as np
import torch
from torch.utils.data import Dataset

from preprocess.paramUtil import t2m_raw_offsets, t2m_body_hand_kinematic_chain
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
    ):
        super().__init__()
        self.pt_path = pt_path
        self.to_torch = to_torch
        self.feet_thre = feet_thre
        self.kp_field = kp_field
        self.assume_y_up = assume_y_up
        self.clip_len = clip_len
        self.random_crop = random_crop
        self.pad_if_short = pad_if_short
        # ---- load db once ----
        self.db = torch.load(pt_path, map_location="cpu", weights_only=False)
        assert isinstance(self.db, dict), f"Expected dict in pt, got {type(self.db)}"

        if keys is None:
            self.keys = list(self.db.keys())
        else:
            self.keys = list(keys)

        # ---- 623 block boundaries (same as your Motion623SplitDataset) ----
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

        # ---- skeleton stuff ----
        self.n_raw_offsets = torch.from_numpy(t2m_raw_offsets).float()
        self.kinematic_chain = t2m_body_hand_kinematic_chain


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
                pos0 = np.concatenate([kp0[0, :22, :], kp0[0, 25:55, :]], axis=0)  # positions
                tgt_skel = Skeleton(self.n_raw_offsets, self.kinematic_chain, "cpu")
                self.tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(pos0).float())
                break

        if self.tgt_offsets is None:
            raise RuntimeError(f"No valid kp3d found in {pt_path} under field '{kp_field}'")

    def __len__(self):
        return len(self.keys)


    # def __getitem__(self, idx):
    #     key = self.keys[idx]
    #     item = self.db[key]
    #     if not isinstance(item, dict) or (self.kp_field not in item):
    #         raise KeyError(f"{key}: no '{self.kp_field}'")

    #     kp = item[self.kp_field]  # (T,52,3) expected
    #     if torch.is_tensor(kp):
    #         kp = kp.detach().cpu().numpy()

    #     if kp.ndim != 3 or kp.shape[2] != 3 or kp.shape[1] < 52:
    #         raise ValueError(f"{key}: expected (T,>=52,3), got {kp.shape}")

    #     kp52 = kp[:, :52, :].astype(np.float32, copy=False)
    #     Tfull = kp52.shape[0]

    #     # -------------------------
    #     # ★ 80-frame sampling (crop)
    #     # -------------------------
    #     L = self.clip_len
    #     if L is not None and L > 0:
    #         if Tfull >= L:
    #             if self.random_crop:
    #                 s = np.random.randint(0, Tfull - L + 1)
    #             else:
    #                 s = 0
    #             kp52 = kp52[s:s+L]
    #         else:
    #             if self.pad_if_short:
    #                 # 後ろを最後のフレームでパディング（簡単版）
    #                 pad = np.repeat(kp52[-1:, :, :], L - Tfull, axis=0)
    #                 kp52 = np.concatenate([kp52, pad], axis=0)
    #             else:
    #                 raise ValueError(f"{key}: too short T={Tfull} < clip_len={L}")

    #     # ---- make (L-1,623) ----
    #     arr = kp3d_to_motion_rep(
    #         kp3d_52_yup=kp52,
    #         feet_thre=self.feet_thre,
    #         tgt_offsets=self.tgt_offsets,
    #         n_raw_offsets=self.n_raw_offsets,
    #         kinematic_chain=self.kinematic_chain,
    #     )  # (L-1,623)

    #     Tm1 = arr.shape[0]  # = clip_len-1

    #     root = arr[:, self.I_ROOT0:self.I_ROOT1]
    #     ric  = arr[:, self.I_RIC0:self.I_RIC1].reshape(Tm1, self.NO_ROOT_J, 3)
    #     rot  = arr[:, self.I_ROT0:self.I_ROT1].reshape(Tm1, self.NO_ROOT_J, 6)
    #     vel  = arr[:, self.I_VEL0:self.I_VEL1].reshape(Tm1, 52, 3)
    #     feet = arr[:, self.I_FEET0:self.I_FEET1]

    #     ric_body, ric_hand = ric[:, :21], ric[:, 21:51]
    #     rot_body, rot_hand = rot[:, :21], rot[:, 21:51]
    #     vel_body, vel_hand = vel[:, :22], vel[:, 22:52]

    #     body = np.concatenate(
    #         [root, ric_body.reshape(Tm1,-1), rot_body.reshape(Tm1,-1), vel_body.reshape(Tm1,-1), feet],
    #         axis=1
    #     )  # (T-1,263)

    #     hand = np.concatenate(
    #         [ric_hand.reshape(Tm1,-1), rot_hand.reshape(Tm1,-1), vel_hand.reshape(Tm1,-1)],
    #         axis=1
    #     )  # (T-1,360)

    #     out = {
    #         "key": key,
    #         "T": int(Tm1),
    #         "body": body,
    #         "hand": hand,
    #         # 便利：どこを切ったか返す
    #         "start": int(s) if (L is not None and Tfull >= L) else 0,
    #         "Tfull": int(Tfull),
    #     }

    #     if self.to_torch:
    #         out["body"] = torch.from_numpy(body.copy()).float()
    #         out["hand"] = torch.from_numpy(hand.copy()).float()

    #     return out
    import warnings
    import numpy as np
    import torch

    def __getitem__(self, idx):
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
                    if verbose_nan:
                        print(
                            f"[NaN@kp52] key={key} "
                            f"Tfull={Tfull} start={s} try={s_try}"
                        )
                    continue

                # ---------- kp3d -> motion rep ----------
                try:
                    arr = kp3d_to_motion_rep(
                        kp3d_52_yup=kp52,
                        feet_thre=self.feet_thre,
                        tgt_offsets=self.tgt_offsets,
                        n_raw_offsets=self.n_raw_offsets,
                        kinematic_chain=self.kinematic_chain,
                    )
                except Exception as e:
                    if verbose_nan:
                        print(
                            f"[ERROR@kp3d_to_motion_rep] key={key} "
                            f"start={s} try={s_try}\n  {repr(e)}"
                        )
                    continue

                # ---------- NaN check (arr) ----------
                if np.isnan(arr).any() or np.isinf(arr).any():
                    if verbose_nan:
                        print(
                            f"[NaN@arr] key={key} "
                            f"start={s} try={s_try} "
                            f"arr_shape={arr.shape}"
                        )
                    continue

                # ---------- split ----------
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



# if __name__ == "__main__":
#     ds = MotionDataset(
#         f"{os.getenv('DATA_DIR')}/ee4d/ee4d_motion_uniegomotion/uniegomotion/ee_train_joints.pt",
#         feet_thre=0.002,
#         clip_len=81,
#     )
#     x = ds[0]
#     print(x["key"], x["body"].shape, x["hand"].shape)  # (T-1,263) (T-1,360)
#     print(x["body"][0])
#     print(x["hand"][0])

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
    def compute_mean_var_623_split(dl, device="cpu", max_batches=-1, eps=1e-12):
        # --- root: yaw(1), vel_xz(2), y(1) を別々に ---
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
        mean_ric = torch.zeros(153, device=device)
        M2_ric   = torch.zeros(153, device=device)

        # --- vel ---
        n_vel = 0
        mean_vel = torch.zeros(156, device=device)
        M2_vel   = torch.zeros(156, device=device)

        for bi, batch in tqdm(enumerate(dl), desc="Computing mean/var (split+root)"):
            if max_batches > 0 and bi >= max_batches:
                break

            body = batch["body"].to(device, non_blocking=True)
            hand = batch["hand"].to(device, non_blocking=True)
            x = torch.cat([body, hand], dim=-1).float()  # (B,T,623)
            x = x.reshape(-1, 623)                       # (N,623)

            ok = torch.isfinite(x).all(dim=1)
            x = x[ok]
            if x.numel() == 0:
                continue

            # root parts
            xyaw = x[:, 0:1]     # (N,1)
            xvxz = x[:, 1:3]     # (N,2)
            xry  = x[:, 3:4]     # (N,1)

            # other blocks
            xric = x[:, 4:157]   # (N,153)
            xvel = x[:, 463:619] # (N,156)

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
        mean_623 = torch.zeros(623, device=device)
        var_623  = torch.ones(623, device=device)
        std_623  = torch.ones(623, device=device)

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
        mean_623[4:157] = mean_ric
        std_623[4:157]  = std_ric
        var_623[4:157]  = var_ric

        # vel
        mean_623[463:619] = mean_vel
        std_623[463:619]  = std_vel
        var_623[463:619]  = var_vel

        nframes_used = {"yaw": n_yaw, "vxz": n_vxz, "ry": n_ry, "ric": n_ric, "vel": n_vel}
        return mean_623, var_623, std_623, nframes_used

    ap = argparse.ArgumentParser()
    ap.add_argument("--pt_path", default=f"{os.getenv('DATA_DIR')}/ee4d/ee4d_motion_uniegomotion/uniegomotion/ee_train_joints.pt")
    ap.add_argument("--clip_len", type=int, default=81)   # -> output T = clip_len-1
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_batches", type=int, default=-1)  # デバッグ用: 例えば 100 とか
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--prefix", default="motion623")        # 保存名のprefix
    args = ap.parse_args()

    ds = MotionDataset(
        pt_path=args.pt_path,
        feet_thre=0.002,
        clip_len=args.clip_len,
        random_crop=True,     # train統計ならTrue推奨
        pad_if_short=True,
        to_torch=True,
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
    mean, var, std, nframes = compute_mean_var_623_split(dl, device=device, max_batches=args.max_batches)

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