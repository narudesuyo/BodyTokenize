# import sys
# sys.path.append(".")

# from src.train.utils import build_model_from_args
# from src.dataset.infer_loader import MotionInferenceDataset
# from src.dataset.collate import collate_stack
# from torch.utils.data import DataLoader

# import torch
# import argparse
# import os
# import numpy as np
# from omegaconf import OmegaConf
# from tqdm import tqdm
# import glob

# from src.evaluate.utils import recover_from_ric
# from src.evaluate.vis import visualize_two_motions

# # quaternion utils
# from common.quaternion import qrot


# def _to_int(x):
#     if torch.is_tensor(x):
#         return int(x.item()) if x.numel() == 1 else int(x.view(-1)[0].item())
#     if isinstance(x, (list, tuple)):
#         return int(x[0])
#     return int(x)


# # =========================================================
# # Forward-based yaw-only stitch (use curr[0] only, then drop it)
# # overlap = 1 fixed usage:
# #   - curr clip frame0 is used ONLY for yaw alignment
# #   - curr clip frame0 is DROPPED after alignment
# #   - no translation alignment (yaw-only)
# # =========================================================
# def estimate_forward_yaw(
#     joints_xyz: torch.Tensor,
#     r_hip: int = 2,
#     l_hip: int = 1,
#     sdr_r: int = 17,
#     sdr_l: int = 16,
#     eps: float = 1e-8,
# ):
#     """
#     joints_xyz: (J,3) world coords
#     return yaw (scalar, rad), where yaw=0 means facing +Z
#     """
#     across1 = joints_xyz[r_hip] - joints_xyz[l_hip]
#     across2 = joints_xyz[sdr_r] - joints_xyz[sdr_l]
#     across = across1 + across2  # (3,)

#     across_xz = across[[0, 2]]
#     across_xz = across_xz / (across_xz.norm() + eps)

#     # forward = up x across  => (-az, ax) in xz
#     fwd_xz = torch.stack([-across_xz[1], across_xz[0]], dim=0)
#     fwd_xz = fwd_xz / (fwd_xz.norm() + eps)

#     yaw = torch.atan2(fwd_xz[0], fwd_xz[1])  # atan2(x,z)
#     return yaw


# def yaw_quat(dyaw: torch.Tensor):
#     """
#     dyaw: scalar full yaw [rad]
#     quaternion about +Y:
#       q = [cos(dyaw/2), 0, sin(dyaw/2), 0]
#     """
#     half = 0.5 * dyaw
#     q = torch.zeros((4,), device=dyaw.device, dtype=dyaw.dtype)
#     q[0] = torch.cos(half)
#     q[2] = torch.sin(half)
#     return q


# def rotate_y_all(joints: torch.Tensor, dyaw: torch.Tensor):
#     """
#     joints: (T,J,3)
#     dyaw: scalar
#     """
#     q = yaw_quat(dyaw)  # (4,)
#     q = q.view(1, 1, 4).expand(joints.shape[0], joints.shape[1], 4)
#     return qrot(q, joints)


# def stitch_yaw_only_and_drop1(
#     curr_j: torch.Tensor,      # (T,J,3) unstitched clip
#     prev_last_j: torch.Tensor, # (J,3) stitched last frame (after drops)
#     face_joint_indx=(2, 1, 17, 16),
# ):
#     """
#     - use curr_j[0] ONLY to compute yaw alignment
#     - rotate whole curr clip by dyaw
#     - drop the first frame (curr_j[0]) AFTER rotation
#     return: curr_keep (T-1,J,3)
#     """
#     if curr_j.shape[0] <= 1:
#         return curr_j[:0]

#     r_hip, l_hip, sdr_r, sdr_l = face_joint_indx

#     yaw_prev = estimate_forward_yaw(prev_last_j, r_hip, l_hip, sdr_r, sdr_l)
#     yaw_curr0 = estimate_forward_yaw(curr_j[0],   r_hip, l_hip, sdr_r, sdr_l)

#     dyaw = (yaw_prev - yaw_curr0).to(curr_j.device, curr_j.dtype)
#     curr_j = rotate_y_all(curr_j, dyaw)

#     # drop frame0 (was used only for alignment)
#     return curr_j[1:].contiguous()


# def recover_clip_alone(
#     data623: torch.Tensor,
#     joints_num: int,
#     use_root_loss: bool,
#     base_idx: int,
# ):
#     """
#     data623: (1,T,623)
#     returns: (T,J,3)
#     """
#     j = recover_from_ric(
#         data623,
#         joints_num=joints_num,
#         use_root_loss=use_root_loss,
#         base_idx=base_idx,
#     )[0]
#     if j.dim() == 4:
#         j = j[0]
#     return j


# # =========================
# # Main
# # =========================
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", type=str,
#                     default="/large/naru/EgoHand/BodyTokenize/runs/token_40_0115_fingertips/config.yaml")
#     ap.add_argument("--ckpt", type=str,
#                     default="/large/naru/EgoHand/BodyTokenize/runs/token_40_0115_fingertips/ckpt_epoch700.pt")
#     ap.add_argument("--split", type=str, default="train")

#     ap.add_argument("--recon", action="store_true",
#                     help="if set, run recover+stitch and save mp4 per sequence")

#     # NOTE: overlap is kept for dataloader slicing, but stitching policy is fixed:
#     #       we ALWAYS use curr[0] for yaw alignment and drop it.
#     ap.add_argument("--clip_len", type=int, default=20)
#     ap.add_argument("--overlap", type=int, default=1)
#     ap.add_argument("--fps", type=int, default=10)
#     ap.add_argument("--save_name", type=str, default="tok_pose_recon")

#     ap.add_argument("--base_idx_override", type=int, default=-1,
#                     help="if >=0, override base_idx from config")
#     ap.add_argument("--one_sequence", action="store_true",
#                     help="process only one sequence and exit")

#     args_cli = ap.parse_args()

#     # if int(args_cli.overlap) != 1:
#     #     print(f"[warn] This script assumes overlap=1 for stitching. You set overlap={args_cli.overlap}.")

#     DATA_ROOT = os.getenv("DATA_ROOT")
#     if not DATA_ROOT:
#         raise RuntimeError("DATA_ROOT env is not set.")

#     video_base_dir = os.path.join(DATA_ROOT, args_cli.split, "takes_clipped", "egoexo", "videos")
#     data_save_dir = os.path.join(DATA_ROOT, args_cli.split, "takes_clipped", "egoexo")
#     human_pose_dir = os.path.join(
#         DATA_ROOT, "ee4d", "ee4d_motion_uniegomotion", "uniegomotion",
#         f"ee_{args_cli.split}_joints_tips.pt"
#     )

#     args = OmegaConf.load(args_cli.config)

#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#     model = build_model_from_args(args, device)
#     ckpt = torch.load(args_cli.ckpt, map_location="cpu", weights_only=False)
#     model.load_state_dict(ckpt["model"], strict=True)
#     model = model.to(device).eval()

#     mean = torch.from_numpy(np.load(args.mean_path)).to(device).float()
#     std = torch.from_numpy(np.load(args.std_path)).to(device).float()
#     print(f"mean.shape={tuple(mean.shape)} std.shape={tuple(std.shape)}")

#     video_paths = sorted(glob.glob(os.path.join(video_base_dir, "**/*.mp4"), recursive=True))
#     print(f"Found {len(video_paths)} videos under {video_base_dir}")

#     include_fingertips = bool(getattr(args, "include_fingertips", False))
#     base_idx = int(getattr(args, "base_idx", 0))
#     if args_cli.base_idx_override >= 0:
#         base_idx = int(args_cli.base_idx_override)

#     use_root_loss = bool(getattr(args, "use_root_loss", True))
#     joints_num = 62 if include_fingertips else 52

#     overlap = int(args_cli.overlap)
#     clip_len = int(args_cli.clip_len)
#     assert 0 <= overlap < clip_len, f"bad overlap={overlap} for clip_len={clip_len}"
#     stride = clip_len - overlap

#     # face joints consistent with kp3d_to_623.py default
#     face_joint_indx = (2, 1, 17, 16)

#     with torch.no_grad():
#         for video_path in tqdm(video_paths, desc="Processing videos"):
#             sample_name = os.path.basename(os.path.dirname(video_path))
#             stem = os.path.splitext(os.path.basename(video_path))[0]  # "start___end"
#             parts = stem.split("___")
#             if len(parts) != 2:
#                 print(f"[skip] unexpected filename (need start___end.mp4): {video_path}")
#                 continue
#             start, end = parts
#             key = f"{sample_name}___{start}___{end}"

#             save_dir = os.path.join(data_save_dir, "tok_pose", f"base_{base_idx}", sample_name)
#             os.makedirs(save_dir, exist_ok=True)

#             if args_cli.recon:
#                 recon_dir = os.path.join(
#                     data_save_dir,
#                     args_cli.save_name,
#                     f"base_{base_idx}",
#                     sample_name,
#                 )
#                 os.makedirs(recon_dir, exist_ok=True)

#             ds_inf = MotionInferenceDataset(
#                 pt_path=human_pose_dir,
#                 key=key,
#                 clip_len=clip_len,
#                 overlap=overlap,
#                 include_fingertips=include_fingertips,
#             )
#             dl = DataLoader(
#                 ds_inf,
#                 batch_size=1,
#                 shuffle=False,
#                 num_workers=int(getattr(args, "num_workers", 0)),
#                 pin_memory=True,
#                 drop_last=False,
#                 collate_fn=collate_stack,
#             )

#             # baseline "all at once" (optional for visualization compare)
#             ds_inf_all = MotionInferenceDataset(
#                 pt_path=human_pose_dir,
#                 key=key,
#                 clip_len=406,
#                 overlap=0,
#                 include_fingertips=include_fingertips,
#             )
#             dl_all = DataLoader(
#                 ds_inf_all,
#                 batch_size=1,
#                 shuffle=False,
#                 num_workers=int(getattr(args, "num_workers", 0)),
#                 pin_memory=True,
#                 drop_last=False,
#                 collate_fn=collate_stack,
#             )

#             # ========== stitch in JOINTS space (yaw-only, drop curr[0]) ==========
#             full_j_gt = []
#             full_j_pr = []
#             first = True

#             prev_last_gt = None  # (J,3) last kept frame
#             prev_last_pr = None

#             Tfull = int(getattr(ds_inf, "Tfull", 0))

#             for i, batch in enumerate(dl):
#                 mB = batch["mB"]
#                 mH = batch["mH"]
#                 if not torch.is_tensor(mB):
#                     mB = torch.as_tensor(mB)
#                 if not torch.is_tensor(mH):
#                     mH = torch.as_tensor(mH)

#                 mB = mB.float().to(device, non_blocking=True)
#                 mH = mH.float().to(device, non_blocking=True)

#                 gt623 = torch.cat([mB, mH], dim=-1)  # (1,T,623)

#                 # normalize -> encode
#                 motion_n = (gt623 - mean) / (std + 1e-8)
#                 mBn = motion_n[..., :mB.shape[-1]]
#                 mHn = motion_n[..., mB.shape[-1]:]

#                 _, _, idx = model(mBn, mHn)

#                 # save indices per-clip
#                 idxH = idx["idxH"].detach().cpu().numpy()
#                 idxB = idx["idxB"].detach().cpu().numpy()
#                 idx_all = np.concatenate([idxB, idxH], axis=-1).reshape(-1)
#                 save_path = os.path.join(save_dir, f"{start}___{end}_{i:04d}.npz")
#                 np.savez_compressed(save_path, idx=idx_all)

#                 if not args_cli.recon:
#                     continue

#                 # decode -> denorm 623
#                 pr623_n = model.decode_from_ids(
#                     idxH=torch.from_numpy(idxH).to(device).long(),
#                     idxB=torch.from_numpy(idxB).to(device).long(),
#                 )
#                 pr623 = pr623_n * (std + 1e-8) + mean

#                 # valid_len
#                 if ("start" in batch) and ("end" in batch):
#                     clip_start = _to_int(batch["start"])
#                     clip_end = _to_int(batch["end"])
#                     valid_len = max(0, clip_end - clip_start)
#                 else:
#                     clip_start = i * stride
#                     if Tfull > 0:
#                         valid_len = max(0, min(clip_len, Tfull - clip_start))
#                     else:
#                         valid_len = gt623.shape[1]

#                 valid_len = min(valid_len, gt623.shape[1], pr623.shape[1])
#                 if valid_len <= 1:
#                     continue

#                 gt623_v = gt623[:, :valid_len].contiguous()
#                 pr623_v = pr623[:, :valid_len].contiguous()

#                 # 1) recover each clip independently
#                 j_gt = recover_clip_alone(gt623_v, joints_num, use_root_loss, base_idx)  # (T,J,3)
#                 j_pr = recover_clip_alone(pr623_v, joints_num, use_root_loss, base_idx)

#                 # 2) stitch (yaw-only) and drop curr[0] for non-first clips
#                 if first:
#                     j_gt_keep = j_gt
#                     j_pr_keep = j_pr
#                     first = False
#                 else:
#                     j_gt_keep = stitch_yaw_only_and_drop1(
#                         j_gt, prev_last_gt, face_joint_indx=face_joint_indx
#                     )
#                     j_pr_keep = stitch_yaw_only_and_drop1(
#                         j_pr, prev_last_pr, face_joint_indx=face_joint_indx
#                     )

#                 if j_gt_keep.shape[0] == 0:
#                     continue

#                 # 3) carry last kept frame
#                 prev_last_gt = j_gt_keep[-1].detach()
#                 prev_last_pr = j_pr_keep[-1].detach()

#                 full_j_gt.append(j_gt_keep)
#                 full_j_pr.append(j_pr_keep)

#             # baseline recover (all-at-once), only for vis compare
#             gt623_all = None
#             for batch_all in dl_all:
#                 mB_all = batch_all["mB"]
#                 mH_all = batch_all["mH"]
#                 if not torch.is_tensor(mB_all):
#                     mB_all = torch.as_tensor(mB_all)
#                 if not torch.is_tensor(mH_all):
#                     mH_all = torch.as_tensor(mH_all)

#                 mB_all = mB_all.float().to(device, non_blocking=True)
#                 mH_all = mH_all.float().to(device, non_blocking=True)
#                 gt623_all = torch.cat([mB_all, mH_all], dim=-1)  # (1,T,623)

#             if args_cli.recon and len(full_j_gt) > 0:
#                 j_gt_full = torch.cat(full_j_gt, dim=0)  # (T,J,3)
#                 j_pr_full = torch.cat(full_j_pr, dim=0)

#                 if gt623_all is None:
#                     raise RuntimeError("gt623_all is None (dl_all empty?)")

#                 # baseline "all-at-once"
#                 j_gt_all_trans = recover_from_ric(
#                     gt623_all,
#                     joints_num=joints_num,
#                     use_root_loss=False,
#                     base_idx=base_idx,
#                 )[0]
#                 if j_gt_all_trans.dim() == 4:
#                     j_gt_all_trans = j_gt_all_trans[0]

#                 # align length
#                 T_vis = min(j_gt_all_trans.shape[0], j_gt_full.shape[0])

#                 out_mp4 = os.path.join(recon_dir, f"{start}___{end}.mp4")
#                 visualize_two_motions(
#                     j_gt_all_trans[:T_vis],
#                     j_gt_full[:T_vis],
#                     save_path=out_mp4,
#                     fps=args_cli.fps,
#                     view="body",
#                     rotate=False,
#                     include_fingertips=include_fingertips,
#                     only_gt=False,
#                     origin_align=False,
#                     base_idx=base_idx,
#                 )
#                 print(f"[yaw_only_drop1] saved: {out_mp4}  (T={j_gt_full.shape[0]})")

#             if args_cli.one_sequence:
#                 exit()


# if __name__ == "__main__":
#     main()

# import sys
# sys.path.append(".")

# from src.train.utils import build_model_from_args
# from src.dataset.infer_loader import MotionInferenceDataset
# from src.dataset.collate import collate_stack
# from torch.utils.data import DataLoader

# import torch
# import argparse
# import os
# import numpy as np
# from omegaconf import OmegaConf
# from tqdm import tqdm
# import glob

# from src.evaluate.utils import recover_from_ric
# from src.evaluate.vis import visualize_two_motions

# # quaternion utils
# from common.quaternion import qrot


# def _to_int(x):
#     if torch.is_tensor(x):
#         return int(x.item()) if x.numel() == 1 else int(x.view(-1)[0].item())
#     if isinstance(x, (list, tuple)):
#         return int(x[0])
#     return int(x)


# # =========================================================
# # Forward-based yaw-only stitch
# # generalized for variable overlap:
# #   - align using curr_j[overlap - 1] (which matches prev_last_j)
# #   - drop curr_j[:overlap]
# # =========================================================
# def estimate_forward_yaw(
#     joints_xyz: torch.Tensor,
#     r_hip: int = 2,
#     l_hip: int = 1,
#     sdr_r: int = 17,
#     sdr_l: int = 16,
#     eps: float = 1e-8,
# ):
#     """
#     joints_xyz: (J,3) world coords
#     return yaw (scalar, rad), where yaw=0 means facing +Z
#     """
#     across1 = joints_xyz[r_hip] - joints_xyz[l_hip]
#     across2 = joints_xyz[sdr_r] - joints_xyz[sdr_l]
#     across = across1 + across2  # (3,)

#     across_xz = across[[0, 2]]
#     across_xz = across_xz / (across_xz.norm() + eps)

#     # forward = up x across  => (-az, ax) in xz
#     fwd_xz = torch.stack([-across_xz[1], across_xz[0]], dim=0)
#     fwd_xz = fwd_xz / (fwd_xz.norm() + eps)

#     yaw = torch.atan2(fwd_xz[0], fwd_xz[1])  # atan2(x,z)
#     return yaw


# def yaw_quat(dyaw: torch.Tensor):
#     """
#     dyaw: scalar full yaw [rad]
#     quaternion about +Y:
#       q = [cos(dyaw/2), 0, sin(dyaw/2), 0]
#     """
#     half = 0.5 * dyaw
#     q = torch.zeros((4,), device=dyaw.device, dtype=dyaw.dtype)
#     q[0] = torch.cos(half)
#     q[2] = torch.sin(half)
#     return q


# def rotate_y_all(joints: torch.Tensor, dyaw: torch.Tensor):
#     """
#     joints: (T,J,3)
#     dyaw: scalar
#     """
#     q = yaw_quat(dyaw)  # (4,)
#     q = q.view(1, 1, 4).expand(joints.shape[0], joints.shape[1], 4)
#     return qrot(q, joints)


# def stitch_yaw_only_and_drop(
#     curr_j: torch.Tensor,      # (T,J,3) unstitched clip
#     prev_last_j: torch.Tensor, # (J,3) stitched last frame (after drops)
#     overlap: int,
#     face_joint_indx=(2, 1, 17, 16),
# ):
#     """
#     Handles variable overlap stitching.
    
#     Logic:
#       If overlap=k, then curr_j[k-1] corresponds to the same instant as prev_last_j.
#       We use curr_j[k-1] for alignment, and then drop curr_j[:k].
      
#       Example (overlap=1): align curr[0], drop curr[:1] (return curr[1:])
#       Example (overlap=2): align curr[1], drop curr[:2] (return curr[2:])
#     """
#     if curr_j.shape[0] <= overlap:
#         # Not enough frames to drop 'overlap' and keep something
#         return curr_j[:0]

#     r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    
#     # Alignment frame index in the new clip
#     align_idx = overlap - 2
    
#     # 1. Calculate yaw of the previous clip's last frame
#     yaw_prev = estimate_forward_yaw(prev_last_j, r_hip, l_hip, sdr_r, sdr_l)
    
#     # 2. Calculate yaw of the corresponding frame in the current clip
#     yaw_curr_align = estimate_forward_yaw(curr_j[align_idx], r_hip, l_hip, sdr_r, sdr_l)

#     # 3. Rotate the entire current clip to match
#     dyaw = (yaw_prev - yaw_curr_align).to(curr_j.device, curr_j.dtype)
#     curr_j = rotate_y_all(curr_j, dyaw)

#     # 4. Drop the overlapping frames (0 to overlap-1)
#     return curr_j[overlap-1:].contiguous()


# def recover_clip_alone(
#     data623: torch.Tensor,
#     joints_num: int,
#     use_root_loss: bool,
#     base_idx: int,
# ):
#     """
#     data623: (1,T,623)
#     returns: (T,J,3)
#     """
#     j = recover_from_ric(
#         data623,
#         joints_num=joints_num,
#         use_root_loss=use_root_loss,
#         base_idx=base_idx,
#     )[0]
#     if j.dim() == 4:
#         j = j[0]
#     return j


# # =========================
# # Main
# # =========================
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", type=str,
#                     default="/large/naru/EgoHand/BodyTokenize/runs/token_40_0115_fingertips/config.yaml")
#     ap.add_argument("--ckpt", type=str,
#                     default="/large/naru/EgoHand/BodyTokenize/runs/token_40_0115_fingertips/ckpt_epoch700.pt")
#     ap.add_argument("--split", type=str, default="train")

#     ap.add_argument("--recon", action="store_true",
#                     help="if set, run recover+stitch and save mp4 per sequence")

#     ap.add_argument("--clip_len", type=int, default=20)
#     ap.add_argument("--overlap", type=int, default=1)
#     ap.add_argument("--fps", type=int, default=10)
#     ap.add_argument("--save_name", type=str, default="tok_pose_recon")

#     ap.add_argument("--base_idx_override", type=int, default=-1,
#                     help="if >=0, override base_idx from config")
#     ap.add_argument("--one_sequence", action="store_true",
#                     help="process only one sequence and exit")

#     args_cli = ap.parse_args()

#     # NOTE: warning removed as we now support arbitrary overlap
    
#     DATA_ROOT = os.getenv("DATA_ROOT")
#     if not DATA_ROOT:
#         raise RuntimeError("DATA_ROOT env is not set.")

#     video_base_dir = os.path.join(DATA_ROOT, args_cli.split, "takes_clipped", "egoexo", "videos")
#     data_save_dir = os.path.join(DATA_ROOT, args_cli.split, "takes_clipped", "egoexo")
#     human_pose_dir = os.path.join(
#         DATA_ROOT, "ee4d", "ee4d_motion_uniegomotion", "uniegomotion",
#         f"ee_{args_cli.split}_joints_tips.pt"
#     )

#     args = OmegaConf.load(args_cli.config)

#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#     model = build_model_from_args(args, device)
#     ckpt = torch.load(args_cli.ckpt, map_location="cpu", weights_only=False)
#     model.load_state_dict(ckpt["model"], strict=True)
#     model = model.to(device).eval()

#     mean = torch.from_numpy(np.load(args.mean_path)).to(device).float()
#     std = torch.from_numpy(np.load(args.std_path)).to(device).float()
#     print(f"mean.shape={tuple(mean.shape)} std.shape={tuple(std.shape)}")

#     video_paths = sorted(glob.glob(os.path.join(video_base_dir, "**/*.mp4"), recursive=True))
#     print(f"Found {len(video_paths)} videos under {video_base_dir}")

#     include_fingertips = bool(getattr(args, "include_fingertips", False))
#     base_idx = int(getattr(args, "base_idx", 0))
#     if args_cli.base_idx_override >= 0:
#         base_idx = int(args_cli.base_idx_override)

#     use_root_loss = bool(getattr(args, "use_root_loss", True))
#     joints_num = 62 if include_fingertips else 52

#     overlap = int(args_cli.overlap)
#     clip_len = int(args_cli.clip_len)
#     assert 0 <= overlap < clip_len, f"bad overlap={overlap} for clip_len={clip_len}"
#     stride = clip_len - overlap

#     # face joints consistent with kp3d_to_623.py default
#     face_joint_indx = (2, 1, 17, 16)

#     with torch.no_grad():
#         for video_path in tqdm(video_paths, desc="Processing videos"):
#             sample_name = os.path.basename(os.path.dirname(video_path))
#             stem = os.path.splitext(os.path.basename(video_path))[0]  # "start___end"
#             parts = stem.split("___")
#             if len(parts) != 2:
#                 print(f"[skip] unexpected filename (need start___end.mp4): {video_path}")
#                 continue
#             start, end = parts
#             key = f"{sample_name}___{start}___{end}"

#             save_dir = os.path.join(data_save_dir, "tok_pose", f"base_{base_idx}", sample_name)
#             os.makedirs(save_dir, exist_ok=True)

#             if args_cli.recon:
#                 recon_dir = os.path.join(
#                     data_save_dir,
#                     args_cli.save_name,
#                     f"base_{base_idx}",
#                     sample_name,
#                 )
#                 os.makedirs(recon_dir, exist_ok=True)

#             ds_inf = MotionInferenceDataset(
#                 pt_path=human_pose_dir,
#                 key=key,
#                 clip_len=clip_len,
#                 overlap=overlap,
#                 include_fingertips=include_fingertips,
#             )
#             dl = DataLoader(
#                 ds_inf,
#                 batch_size=1,
#                 shuffle=False,
#                 num_workers=int(getattr(args, "num_workers", 0)),
#                 pin_memory=True,
#                 drop_last=False,
#                 collate_fn=collate_stack,
#             )

#             # baseline "all at once" (optional for visualization compare)
#             ds_inf_all = MotionInferenceDataset(
#                 pt_path=human_pose_dir,
#                 key=key,
#                 clip_len=406,
#                 overlap=0,
#                 include_fingertips=include_fingertips,
#             )
#             dl_all = DataLoader(
#                 ds_inf_all,
#                 batch_size=1,
#                 shuffle=False,
#                 num_workers=int(getattr(args, "num_workers", 0)),
#                 pin_memory=True,
#                 drop_last=False,
#                 collate_fn=collate_stack,
#             )

#             # ========== stitch in JOINTS space (yaw-only, dynamic drop based on overlap) ==========
#             full_j_gt = []
#             full_j_pr = []
#             first = True

#             prev_last_gt = None  # (J,3) last kept frame
#             prev_last_pr = None

#             Tfull = int(getattr(ds_inf, "Tfull", 0))

#             for i, batch in enumerate(dl):
#                 mB = batch["mB"]
#                 mH = batch["mH"]
#                 if not torch.is_tensor(mB):
#                     mB = torch.as_tensor(mB)
#                 if not torch.is_tensor(mH):
#                     mH = torch.as_tensor(mH)

#                 mB = mB.float().to(device, non_blocking=True)
#                 mH = mH.float().to(device, non_blocking=True)

#                 gt623 = torch.cat([mB, mH], dim=-1)  # (1,T,623)

#                 # normalize -> encode
#                 motion_n = (gt623 - mean) / (std + 1e-8)
#                 mBn = motion_n[..., :mB.shape[-1]]
#                 mHn = motion_n[..., mB.shape[-1]:]

#                 _, _, idx = model(mBn, mHn)

#                 # save indices per-clip
#                 idxH = idx["idxH"].detach().cpu().numpy()
#                 idxB = idx["idxB"].detach().cpu().numpy()
#                 idx_all = np.concatenate([idxB, idxH], axis=-1).reshape(-1)
#                 save_path = os.path.join(save_dir, f"{start}___{end}_{i:04d}.npz")
#                 np.savez_compressed(save_path, idx=idx_all)

#                 if not args_cli.recon:
#                     continue

#                 # decode -> denorm 623
#                 pr623_n = model.decode_from_ids(
#                     idxH=torch.from_numpy(idxH).to(device).long(),
#                     idxB=torch.from_numpy(idxB).to(device).long(),
#                 )
#                 pr623 = pr623_n * (std + 1e-8) + mean

#                 # valid_len
#                 if ("start" in batch) and ("end" in batch):
#                     clip_start = _to_int(batch["start"])
#                     clip_end = _to_int(batch["end"])
#                     valid_len = max(0, clip_end - clip_start)
#                 else:
#                     clip_start = i * stride
#                     if Tfull > 0:
#                         valid_len = max(0, min(clip_len, Tfull - clip_start))
#                     else:
#                         valid_len = gt623.shape[1]

#                 valid_len = min(valid_len, gt623.shape[1], pr623.shape[1])
#                 if valid_len <= 1:
#                     continue

#                 gt623_v = gt623[:, :valid_len].contiguous()
#                 pr623_v = pr623[:, :valid_len].contiguous()

#                 # 1) recover each clip independently
#                 j_gt = recover_clip_alone(gt623_v, joints_num, use_root_loss, base_idx)  # (T,J,3)
#                 j_pr = recover_clip_alone(pr623_v, joints_num, use_root_loss, base_idx)

#                 # 2) stitch (yaw-only) and drop 'overlap' frames for non-first clips
#                 if first:
#                     j_gt_keep = j_gt
#                     j_pr_keep = j_pr
#                     first = False
#                 else:
#                     j_gt_keep = stitch_yaw_only_and_drop(
#                         j_gt, prev_last_gt, overlap=overlap, face_joint_indx=face_joint_indx
#                     )
#                     j_pr_keep = stitch_yaw_only_and_drop(
#                         j_pr, prev_last_pr, overlap=overlap, face_joint_indx=face_joint_indx
#                     )

#                 if j_gt_keep.shape[0] == 0:
#                     continue

#                 # 3) carry last kept frame
#                 prev_last_gt = j_gt_keep[-1].detach()
#                 prev_last_pr = j_pr_keep[-1].detach()

#                 full_j_gt.append(j_gt_keep)
#                 full_j_pr.append(j_pr_keep)

#             # baseline recover (all-at-once), only for vis compare
#             gt623_all = None
#             for batch_all in dl_all:
#                 mB_all = batch_all["mB"]
#                 mH_all = batch_all["mH"]
#                 if not torch.is_tensor(mB_all):
#                     mB_all = torch.as_tensor(mB_all)
#                 if not torch.is_tensor(mH_all):
#                     mH_all = torch.as_tensor(mH_all)

#                 mB_all = mB_all.float().to(device, non_blocking=True)
#                 mH_all = mH_all.float().to(device, non_blocking=True)
#                 gt623_all = torch.cat([mB_all, mH_all], dim=-1)  # (1,T,623)

#             if args_cli.recon and len(full_j_gt) > 0:
#                 j_gt_full = torch.cat(full_j_gt, dim=0)  # (T,J,3)
#                 j_pr_full = torch.cat(full_j_pr, dim=0)

#                 if gt623_all is None:
#                     raise RuntimeError("gt623_all is None (dl_all empty?)")

#                 # baseline "all-at-once"
#                 j_gt_all_trans = recover_from_ric(
#                     gt623_all,
#                     joints_num=joints_num,
#                     use_root_loss=False,
#                     base_idx=base_idx,
#                 )[0]
#                 if j_gt_all_trans.dim() == 4:
#                     j_gt_all_trans = j_gt_all_trans[0]

#                 # align length
#                 T_vis = min(j_gt_all_trans.shape[0], j_gt_full.shape[0])

#                 out_mp4 = os.path.join(recon_dir, f"{start}___{end}.mp4")
#                 visualize_two_motions(
#                     j_gt_all_trans[:T_vis],
#                     j_gt_full[:T_vis],
#                     save_path=out_mp4,
#                     fps=args_cli.fps,
#                     view="body",
#                     rotate=False,
#                     include_fingertips=include_fingertips,
#                     only_gt=False,
#                     origin_align=False,
#                     base_idx=base_idx,
#                 )
#                 print(f"[yaw_only_drop{overlap}] saved: {out_mp4}  (T={j_gt_full.shape[0]})")

#             if args_cli.one_sequence:
#                 exit()


# if __name__ == "__main__":
#     main()


# import sys
# sys.path.append(".")

# from src.train.utils import build_model_from_args
# from src.dataset.infer_loader import MotionInferenceDataset
# from src.dataset.collate import collate_stack
# from torch.utils.data import DataLoader

# import torch
# import argparse
# import os
# import numpy as np
# from omegaconf import OmegaConf
# from tqdm import tqdm
# import glob

# from src.evaluate.utils import recover_from_ric
# from src.evaluate.vis import visualize_two_motions

# # quaternion utils
# from common.quaternion import qrot


# def _to_int(x):
#     if torch.is_tensor(x):
#         return int(x.item()) if x.numel() == 1 else int(x.view(-1)[0].item())
#     if isinstance(x, (list, tuple)):
#         return int(x[0])
#     return int(x)


# # =========================================================
# # Forward-based yaw-only stitch
# # generalized for variable overlap:
# #   - align using curr_j[overlap - 1] (which matches prev_last_j)
# #   - drop curr_j[:overlap]
# # =========================================================
# def estimate_forward_yaw(
#     joints_xyz: torch.Tensor,
#     r_hip: int = 2,
#     l_hip: int = 1,
#     sdr_r: int = 17,
#     sdr_l: int = 16,
#     eps: float = 1e-8,
# ):
#     """
#     joints_xyz: (J,3) world coords
#     return yaw (scalar, rad), where yaw=0 means facing +Z
#     """
#     across1 = joints_xyz[r_hip] - joints_xyz[l_hip]
#     across2 = joints_xyz[sdr_r] - joints_xyz[sdr_l]
#     across = across1 + across2  # (3,)

#     across_xz = across[[0, 2]]
#     across_xz = across_xz / (across_xz.norm() + eps)

#     # forward = up x across  => (-az, ax) in xz
#     fwd_xz = torch.stack([-across_xz[1], across_xz[0]], dim=0)
#     fwd_xz = fwd_xz / (fwd_xz.norm() + eps)

#     yaw = torch.atan2(fwd_xz[0], fwd_xz[1])  # atan2(x,z)
#     return yaw


# def yaw_quat(dyaw: torch.Tensor):
#     """
#     dyaw: scalar full yaw [rad]
#     quaternion about +Y:
#       q = [cos(dyaw/2), 0, sin(dyaw/2), 0]
#     """
#     half = 0.5 * dyaw
#     q = torch.zeros((4,), device=dyaw.device, dtype=dyaw.dtype)
#     q[0] = torch.cos(half)
#     q[2] = torch.sin(half)
#     return q


# def rotate_y_all(joints: torch.Tensor, dyaw: torch.Tensor):
#     """
#     joints: (T,J,3)
#     dyaw: scalar
#     """
#     q = yaw_quat(dyaw)  # (4,)
#     q = q.view(1, 1, 4).expand(joints.shape[0], joints.shape[1], 4)
#     return qrot(q, joints)


# def stitch_yaw_only_and_drop(
#     curr_j: torch.Tensor,      # (T,J,3) unstitched clip
#     prev_last_j: torch.Tensor, # (J,3) stitched last frame (after drops)
#     overlap: int,
#     face_joint_indx=(2, 1, 17, 16),
# ):
#     """
#     Handles variable overlap stitching.
    
#     Logic:
#       If overlap=k, then curr_j[k-1] corresponds to the same instant as prev_last_j.
#       We use curr_j[k-1] for alignment, and then drop curr_j[:k].
      
#       Example (overlap=1): align curr[0], drop curr[:1] (return curr[1:])
#     """
#     if curr_j.shape[0] <= overlap:
#         # Not enough frames to drop 'overlap' and keep something
#         return curr_j[:0]

#     r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    
#     # Alignment frame index in the new clip
#     # The last frame of previous clip corresponds to 'overlap - 1' in the new clip.
#     align_idx = overlap - 1
    
#     # 1. Calculate yaw of the previous clip's last frame
#     yaw_prev = estimate_forward_yaw(prev_last_j, r_hip, l_hip, sdr_r, sdr_l)
    
#     # 2. Calculate yaw of the corresponding frame in the current clip
#     yaw_curr_align = estimate_forward_yaw(curr_j[align_idx], r_hip, l_hip, sdr_r, sdr_l)

#     # 3. Rotate the entire current clip to match
#     dyaw = (yaw_prev - yaw_curr_align).to(curr_j.device, curr_j.dtype)
#     curr_j = rotate_y_all(curr_j, dyaw)

#     # 4. Drop the overlapping frames (0 to overlap-1)
#     # We keep from 'overlap' onwards.
#     return curr_j[overlap:].contiguous()


# def recover_clip_alone(
#     data623: torch.Tensor,
#     joints_num: int,
#     use_root_loss: bool,
#     base_idx: int,
# ):
#     """
#     data623: (1,T,623)
#     returns: (T,J,3)
#     """
#     j = recover_from_ric(
#         data623,
#         joints_num=joints_num,
#         use_root_loss=use_root_loss,
#         base_idx=base_idx,
#     )[0]
#     if j.dim() == 4:
#         j = j[0]
#     return j


# # =========================
# # Main
# # =========================
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--config", type=str,
#                     default="/large/naru/EgoHand/BodyTokenize/runs/token_40_0115_fingertips/config.yaml")
#     ap.add_argument("--ckpt", type=str,
#                     default="/large/naru/EgoHand/BodyTokenize/runs/token_40_0115_fingertips/ckpt_epoch700.pt")
#     ap.add_argument("--split", type=str, default="train")

#     ap.add_argument("--recon", action="store_true",
#                     help="if set, run recover+stitch and save mp4 per sequence")

#     ap.add_argument("--clip_len", type=int, default=20)
#     ap.add_argument("--overlap", type=int, default=2, help="Raw overlap in dataloader (must be >= 2 for stitching)")
#     ap.add_argument("--fps", type=int, default=10)
#     ap.add_argument("--save_name", type=str, default="tok_pose_recon")

#     ap.add_argument("--base_idx_override", type=int, default=-1,
#                     help="if >=0, override base_idx from config")
#     ap.add_argument("--one_sequence", action="store_true",
#                     help="process only one sequence and exit")

#     args_cli = ap.parse_args()

#     # Safety check: input features are velocity-based, so we lose 1 frame.
#     # To stitch with at least 1 frame overlap in pose space, we need raw overlap >= 2.
#     if int(args_cli.overlap) < 2:
#         raise ValueError(f"For stitching, raw overlap must be >= 2 (because 1 frame is lost to deltas). Got {args_cli.overlap}")
    
#     DATA_ROOT = os.getenv("DATA_ROOT")
#     if not DATA_ROOT:
#         raise RuntimeError("DATA_ROOT env is not set.")

#     video_base_dir = os.path.join(DATA_ROOT, args_cli.split, "takes_clipped", "egoexo", "videos")
#     data_save_dir = os.path.join(DATA_ROOT, args_cli.split, "takes_clipped", "egoexo")
#     human_pose_dir = os.path.join(
#         DATA_ROOT, "ee4d", "ee4d_motion_uniegomotion", "uniegomotion",
#         f"ee_{args_cli.split}_joints_tips.pt"
#     )

#     args = OmegaConf.load(args_cli.config)

#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#     model = build_model_from_args(args, device)
#     ckpt = torch.load(args_cli.ckpt, map_location="cpu", weights_only=False)
#     model.load_state_dict(ckpt["model"], strict=True)
#     model = model.to(device).eval()

#     mean = torch.from_numpy(np.load(args.mean_path)).to(device).float()
#     std = torch.from_numpy(np.load(args.std_path)).to(device).float()
#     print(f"mean.shape={tuple(mean.shape)} std.shape={tuple(std.shape)}")

#     video_paths = sorted(glob.glob(os.path.join(video_base_dir, "**/*.mp4"), recursive=True))
#     print(f"Found {len(video_paths)} videos under {video_base_dir}")

#     include_fingertips = bool(getattr(args, "include_fingertips", False))
#     base_idx = int(getattr(args, "base_idx", 0))
#     if args_cli.base_idx_override >= 0:
#         base_idx = int(args_cli.base_idx_override)

#     use_root_loss = bool(getattr(args, "use_root_loss", True))
#     joints_num = 62 if include_fingertips else 52

#     overlap = int(args_cli.overlap)
#     clip_len = int(args_cli.clip_len)
#     assert 0 <= overlap < clip_len, f"bad overlap={overlap} for clip_len={clip_len}"
    
#     # Effective overlap in POSE space (after velocity conversion losses 1 frame)
#     effective_overlap = overlap - 1
#     print(f"Raw overlap: {overlap}, Effective pose overlap: {effective_overlap}")

#     # face joints consistent with kp3d_to_623.py default
#     face_joint_indx = (2, 1, 17, 16)

#     with torch.no_grad():
#         for video_path in tqdm(video_paths, desc="Processing videos"):
#             sample_name = os.path.basename(os.path.dirname(video_path))
#             stem = os.path.splitext(os.path.basename(video_path))[0]  # "start___end"
#             parts = stem.split("___")
#             if len(parts) != 2:
#                 print(f"[skip] unexpected filename (need start___end.mp4): {video_path}")
#                 continue
#             start, end = parts
#             key = f"{sample_name}___{start}___{end}"

#             save_dir = os.path.join(data_save_dir, "tok_pose", f"base_{base_idx}", sample_name)
#             os.makedirs(save_dir, exist_ok=True)

#             if args_cli.recon:
#                 recon_dir = os.path.join(
#                     data_save_dir,
#                     args_cli.save_name,
#                     f"base_{base_idx}",
#                     sample_name,
#                 )
#                 os.makedirs(recon_dir, exist_ok=True)

#             ds_inf = MotionInferenceDataset(
#                 pt_path=human_pose_dir,
#                 key=key,
#                 clip_len=clip_len,
#                 overlap=overlap,
#                 include_fingertips=include_fingertips,
#             )
#             dl = DataLoader(
#                 ds_inf,
#                 batch_size=1,
#                 shuffle=False,
#                 num_workers=int(getattr(args, "num_workers", 0)),
#                 pin_memory=True,
#                 drop_last=False,
#                 collate_fn=collate_stack,
#             )

#             # baseline "all at once" (optional for visualization compare)
#             ds_inf_all = MotionInferenceDataset(
#                 pt_path=human_pose_dir,
#                 key=key,
#                 clip_len=406,
#                 overlap=0,
#                 include_fingertips=include_fingertips,
#             )
#             dl_all = DataLoader(
#                 ds_inf_all,
#                 batch_size=1,
#                 shuffle=False,
#                 num_workers=int(getattr(args, "num_workers", 0)),
#                 pin_memory=True,
#                 drop_last=False,
#                 collate_fn=collate_stack,
#             )

#             # ========== stitch in JOINTS space (yaw-only, dynamic drop based on overlap) ==========
#             full_j_gt = []
#             full_j_pr = []
#             first = True

#             prev_last_gt = None  # (J,3) last kept frame
#             prev_last_pr = None

#             Tfull = int(getattr(ds_inf, "Tfull", 0))

#             for i, batch in enumerate(dl):
#                 mB = batch["mB"]
#                 mH = batch["mH"]
#                 if not torch.is_tensor(mB):
#                     mB = torch.as_tensor(mB)
#                 if not torch.is_tensor(mH):
#                     mH = torch.as_tensor(mH)

#                 mB = mB.float().to(device, non_blocking=True)
#                 mH = mH.float().to(device, non_blocking=True)

#                 gt623 = torch.cat([mB, mH], dim=-1)  # (1,T,623)

#                 # normalize -> encode
#                 motion_n = (gt623 - mean) / (std + 1e-8)
#                 mBn = motion_n[..., :mB.shape[-1]]
#                 mHn = motion_n[..., mB.shape[-1]:]

#                 _, _, idx = model(mBn, mHn)

#                 # save indices per-clip
#                 idxH = idx["idxH"].detach().cpu().numpy()
#                 idxB = idx["idxB"].detach().cpu().numpy()
#                 idx_all = np.concatenate([idxB, idxH], axis=-1).reshape(-1)
#                 save_path = os.path.join(save_dir, f"{start}___{end}_{i:04d}.npz")
#                 np.savez_compressed(save_path, idx=idx_all)

#                 if not args_cli.recon:
#                     continue

#                 # decode -> denorm 623
#                 pr623_n = model.decode_from_ids(
#                     idxH=torch.from_numpy(idxH).to(device).long(),
#                     idxB=torch.from_numpy(idxB).to(device).long(),
#                 )
#                 pr623 = pr623_n * (std + 1e-8) + mean

#                 # valid_len
#                 if ("start" in batch) and ("end" in batch):
#                     clip_start = _to_int(batch["start"])
#                     clip_end = _to_int(batch["end"])
#                     valid_len = max(0, clip_end - clip_start)
#                 else:
#                     clip_start = i * (clip_len - overlap) # stride = clip_len - overlap
#                     if Tfull > 0:
#                         valid_len = max(0, min(clip_len, Tfull - clip_start))
#                     else:
#                         valid_len = gt623.shape[1]

#                 valid_len = min(valid_len, gt623.shape[1], pr623.shape[1])
#                 if valid_len <= 1:
#                     continue

#                 gt623_v = gt623[:, :valid_len].contiguous()
#                 pr623_v = pr623[:, :valid_len].contiguous()

#                 # 1) recover each clip independently
#                 j_gt = recover_clip_alone(gt623_v, joints_num, use_root_loss, base_idx)  # (T,J,3)
#                 j_pr = recover_clip_alone(pr623_v, joints_num, use_root_loss, base_idx)

#                 # 2) stitch (yaw-only) and drop 'overlap' frames for non-first clips
#                 if first:
#                     j_gt_keep = j_gt
#                     j_pr_keep = j_pr
#                     first = False
#                 else:
#                     # Pass EFFECTIVE overlap (e.g. 1 if raw overlap was 2)
#                     j_gt_keep = stitch_yaw_only_and_drop(
#                         j_gt, prev_last_gt, overlap=effective_overlap, face_joint_indx=face_joint_indx
#                     )
#                     j_pr_keep = stitch_yaw_only_and_drop(
#                         j_pr, prev_last_pr, overlap=effective_overlap, face_joint_indx=face_joint_indx
#                     )

#                 if j_gt_keep.shape[0] == 0:
#                     continue

#                 # 3) carry last kept frame
#                 prev_last_gt = j_gt_keep[-1].detach()
#                 prev_last_pr = j_pr_keep[-1].detach()

#                 full_j_gt.append(j_gt_keep)
#                 full_j_pr.append(j_pr_keep)

#             # baseline recover (all-at-once), only for vis compare
#             gt623_all = None
#             for batch_all in dl_all:
#                 mB_all = batch_all["mB"]
#                 mH_all = batch_all["mH"]
#                 if not torch.is_tensor(mB_all):
#                     mB_all = torch.as_tensor(mB_all)
#                 if not torch.is_tensor(mH_all):
#                     mH_all = torch.as_tensor(mH_all)

#                 mB_all = mB_all.float().to(device, non_blocking=True)
#                 mH_all = mH_all.float().to(device, non_blocking=True)
#                 gt623_all = torch.cat([mB_all, mH_all], dim=-1)  # (1,T,623)

#             if args_cli.recon and len(full_j_gt) > 0:
#                 j_gt_full = torch.cat(full_j_gt, dim=0)  # (T,J,3)
#                 j_pr_full = torch.cat(full_j_pr, dim=0)

#                 if gt623_all is None:
#                     raise RuntimeError("gt623_all is None (dl_all empty?)")

#                 # baseline "all-at-once"
#                 j_gt_all_trans = recover_from_ric(
#                     gt623_all,
#                     joints_num=joints_num,
#                     use_root_loss=False,
#                     base_idx=base_idx,
#                 )[0]
#                 if j_gt_all_trans.dim() == 4:
#                     j_gt_all_trans = j_gt_all_trans[0]

#                 # align length
#                 T_vis = min(j_gt_all_trans.shape[0], j_gt_full.shape[0])

#                 out_mp4 = os.path.join(recon_dir, f"{start}___{end}.mp4")
#                 visualize_two_motions(
#                     j_gt_all_trans[:T_vis],
#                     j_gt_full[:T_vis],
#                     save_path=out_mp4,
#                     fps=args_cli.fps,
#                     view="body",
#                     rotate=False,
#                     include_fingertips=include_fingertips,
#                     only_gt=False,
#                     origin_align=False,
#                     base_idx=base_idx,
#                 )
#                 print(f"[yaw_only_drop{overlap}(eff{effective_overlap})] saved: {out_mp4}  (T={j_gt_full.shape[0]})")

#             if args_cli.one_sequence:
#                 exit()


# if __name__ == "__main__":
#     main()



import sys
sys.path.append(".")

from src.train.utils import build_model_from_args
from src.dataset.infer_loader import MotionInferenceDataset
from src.dataset.collate import collate_stack
from torch.utils.data import DataLoader

import torch
import argparse
import os
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import glob

from src.evaluate.utils import recover_from_ric
from src.evaluate.vis import visualize_two_motions

# quaternion utils
from common.quaternion import qrot


def _to_int(x):
    if torch.is_tensor(x):
        return int(x.item()) if x.numel() == 1 else int(x.view(-1)[0].item())
    if isinstance(x, (list, tuple)):
        return int(x[0])
    return int(x)


# =========================================================
# Yaw utils
# =========================================================
def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """
    Wrap angle to (-pi, pi]
    """
    return (angle + torch.pi) % (2 * torch.pi) - torch.pi


def estimate_forward_yaw(
    joints_xyz: torch.Tensor,
    r_hip: int = 2,
    l_hip: int = 1,
    sdr_r: int = 17,
    sdr_l: int = 16,
    eps: float = 1e-8,
):
    """
    joints_xyz: (J,3) world coords
    return yaw (scalar, rad), where yaw=0 means facing +Z
    """
    across1 = joints_xyz[r_hip] - joints_xyz[l_hip]
    across2 = joints_xyz[sdr_r] - joints_xyz[sdr_l]
    across = across1 + across2  # (3,)

    across_xz = across[[0, 2]]
    across_xz = across_xz / (across_xz.norm() + eps)

    # forward = up x across  => (-az, ax) in xz
    fwd_xz = torch.stack([-across_xz[1], across_xz[0]], dim=0)
    fwd_xz = fwd_xz / (fwd_xz.norm() + eps)

    yaw = torch.atan2(fwd_xz[0], fwd_xz[1])  # atan2(x,z)
    return yaw


def yaw_quat_batch(dyaw: torch.Tensor) -> torch.Tensor:
    """
    dyaw: (T,) yaw angles [rad]
    quaternion about +Y:
      q = [cos(dyaw/2), 0, sin(dyaw/2), 0]
    returns: (T,4)
    """
    half = 0.5 * dyaw
    q = torch.zeros((dyaw.shape[0], 4), device=dyaw.device, dtype=dyaw.dtype)
    q[:, 0] = torch.cos(half)
    q[:, 2] = torch.sin(half)
    return q


def rotate_y_per_frame(joints: torch.Tensor, dyaw_t: torch.Tensor) -> torch.Tensor:
    """
    joints: (T,J,3)
    dyaw_t: (T,) per-frame yaw rotation
    """
    q = yaw_quat_batch(dyaw_t)  # (T,4)
    q = q.view(joints.shape[0], 1, 4).expand(joints.shape[0], joints.shape[1], 4)
    return qrot(q, joints)


def compute_yaw_raw_sequence(
    joints: torch.Tensor,  # (T,J,3)
    face_joint_indx=(2, 1, 17, 16),
) -> torch.Tensor:
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    yaws = []
    for t in range(joints.shape[0]):
        yaws.append(estimate_forward_yaw(joints[t], r_hip, l_hip, sdr_r, sdr_l))
    return torch.stack(yaws, dim=0)  # (T,)


def integrate_yaw_from_raw(yaw_raw: torch.Tensor) -> torch.Tensor:
    """
    yaw_raw: (T,)
    returns yaw_int: (T,) where yaw_int[0]=0 and yaw_int[t]=sum wrap(dyaw)
    """
    T = yaw_raw.shape[0]
    if T == 0:
        return yaw_raw

    yaw_int = torch.zeros_like(yaw_raw)
    if T == 1:
        return yaw_int

    dyaw = wrap_to_pi(yaw_raw[1:] - yaw_raw[:-1])  # (T-1,)
    yaw_int[1:] = torch.cumsum(dyaw, dim=0)
    return yaw_int


# =========================================================
# Forward-based yaw INTEGRATION stitch (variable overlap)
#   - compute yaw_raw[t] per frame
#   - integrate dyaw to yaw_int (relative yaw, yaw_int[0]=0)
#   - decide init yaw offset by matching at curr[overlap-1] to prev_last
#   - apply per-frame delta rotation so that yaw becomes yaw_target
#   - drop curr[:overlap]
# =========================================================
def stitch_yaw_integrated_and_drop(
    curr_j: torch.Tensor,      # (T,J,3) unstitched clip
    prev_last_j: torch.Tensor, # (J,3) stitched last frame (after drops)
    overlap: int,
    face_joint_indx=(2, 1, 17, 16),
):
    """
    If overlap=k, then curr_j[k-1] corresponds to the same instant as prev_last_j.
    We use curr_j[k-1] to decide init yaw offset (i.e., absolute yaw gauge),
    then drop curr_j[:k].
    """
    if overlap <= 0:
        return curr_j.contiguous()

    if curr_j.shape[0] <= overlap:
        return curr_j[:0]

    align_idx = overlap - 1

    # 1) yaw_raw per frame for current clip
    yaw_raw = compute_yaw_raw_sequence(curr_j, face_joint_indx=face_joint_indx)  # (T,)

    # 2) integrate dyaw to get relative yaw (starts at 0)
    yaw_int = integrate_yaw_from_raw(yaw_raw)  # (T,)

    # 3) yaw at previous stitched last frame (absolute gauge)
    yaw_prev = estimate_forward_yaw(prev_last_j, *face_joint_indx)

    # 4) decide offset so that yaw_target[align_idx] == yaw_prev
    #    yaw_target[t] = yaw_int[t] + offset
    offset = yaw_prev - yaw_int[align_idx]
    yaw_target = yaw_int + offset  # (T,)

    # 5) rotate each frame so that yaw becomes yaw_target
    #    After rotation by delta[t], yaw_raw[t] -> yaw_raw[t] + delta[t]
    delta = wrap_to_pi(yaw_target - yaw_raw)  # (T,)
    curr_j_aligned = rotate_y_per_frame(curr_j, delta)

    # 6) drop overlap frames
    return curr_j_aligned[overlap:].contiguous()


def recover_clip_alone(
    data623: torch.Tensor,
    joints_num: int,
    use_root_loss: bool,
    base_idx: int,
):
    """
    data623: (1,T,623)
    returns: (T,J,3)
    """
    j = recover_from_ric(
        data623,
        joints_num=joints_num,
        use_root_loss=use_root_loss,
        base_idx=base_idx,
    )[0]
    if j.dim() == 4:
        j = j[0]
    return j


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str,
                    default="/large/naru/EgoHand/BodyTokenize/runs/token_40_0115_fingertips/config.yaml")
    ap.add_argument("--ckpt", type=str,
                    default="/large/naru/EgoHand/BodyTokenize/runs/token_40_0115_fingertips/ckpt_epoch700.pt")
    ap.add_argument("--split", type=str, default="train")

    ap.add_argument("--recon", action="store_true",
                    help="if set, run recover+stitch and save mp4 per sequence")

    ap.add_argument("--clip_len", type=int, default=20)
    ap.add_argument("--overlap", type=int, default=2, help="Raw overlap in dataloader (must be >= 2 for stitching)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--save_name", type=str, default="tok_pose_recon")

    ap.add_argument("--base_idx_override", type=int, default=-1,
                    help="if >=0, override base_idx from config")
    ap.add_argument("--one_sequence", action="store_true",
                    help="process only one sequence and exit")

    args_cli = ap.parse_args()

    # Safety check: input features are velocity-based, so we lose 1 frame.
    # To stitch with at least 1 frame overlap in pose space, we need raw overlap >= 2.
    if int(args_cli.overlap) < 2:
        raise ValueError(f"For stitching, raw overlap must be >= 2 (because 1 frame is lost to deltas). Got {args_cli.overlap}")

    DATA_ROOT = os.getenv("DATA_ROOT")
    if not DATA_ROOT:
        raise RuntimeError("DATA_ROOT env is not set.")

    video_base_dir = os.path.join(DATA_ROOT, args_cli.split, "takes_clipped", "egoexo", "videos")
    data_save_dir = os.path.join(DATA_ROOT, args_cli.split, "takes_clipped", "egoexo")
    human_pose_dir = os.path.join(
        DATA_ROOT, "ee4d", "ee4d_motion_uniegomotion", "uniegomotion",
        f"ee_{args_cli.split}_joints_tips.pt"
    )

    args = OmegaConf.load(args_cli.config)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model_from_args(args, device)
    ckpt = torch.load(args_cli.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device).eval()

    mean = torch.from_numpy(np.load(args.mean_path)).to(device).float()
    std = torch.from_numpy(np.load(args.std_path)).to(device).float()
    print(f"mean.shape={tuple(mean.shape)} std.shape={tuple(std.shape)}")

    video_paths = sorted(glob.glob(os.path.join(video_base_dir, "**/*.mp4"), recursive=True))
    print(f"Found {len(video_paths)} videos under {video_base_dir}")

    include_fingertips = bool(getattr(args, "include_fingertips", False))
    base_idx = int(getattr(args, "base_idx", 0))
    if args_cli.base_idx_override >= 0:
        base_idx = int(args_cli.base_idx_override)

    use_root_loss = bool(getattr(args, "use_root_loss", True))
    joints_num = 62 if include_fingertips else 52

    overlap = int(args_cli.overlap)
    clip_len = int(args_cli.clip_len)
    assert 0 <= overlap < clip_len, f"bad overlap={overlap} for clip_len={clip_len}"

    # Effective overlap in POSE space (after velocity conversion loses 1 frame)
    effective_overlap = overlap - 1
    print(f"Raw overlap: {overlap}, Effective pose overlap: {effective_overlap}")

    # face joints consistent with kp3d_to_623.py default
    face_joint_indx = (2, 1, 17, 16)

    with torch.no_grad():
        for video_path in tqdm(video_paths, desc="Processing videos"):
            sample_name = os.path.basename(os.path.dirname(video_path))
            stem = os.path.splitext(os.path.basename(video_path))[0]  # "start___end"
            parts = stem.split("___")
            if len(parts) != 2:
                print(f"[skip] unexpected filename (need start___end.mp4): {video_path}")
                continue
            start, end = parts
            key = f"{sample_name}___{start}___{end}"

            save_dir = os.path.join(data_save_dir, "tok_pose", f"base_{base_idx}", sample_name)
            os.makedirs(save_dir, exist_ok=True)

            if args_cli.recon:
                recon_dir = os.path.join(
                    data_save_dir,
                    args_cli.save_name,
                    f"base_{base_idx}",
                    sample_name,
                )
                os.makedirs(recon_dir, exist_ok=True)

            ds_inf = MotionInferenceDataset(
                pt_path=human_pose_dir,
                key=key,
                clip_len=clip_len,
                overlap=overlap,
                include_fingertips=include_fingertips,
            )
            dl = DataLoader(
                ds_inf,
                batch_size=1,
                shuffle=False,
                num_workers=int(getattr(args, "num_workers", 0)),
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_stack,
            )

            # baseline "all at once" (optional for visualization compare)
            ds_inf_all = MotionInferenceDataset(
                pt_path=human_pose_dir,
                key=key,
                clip_len=406,
                overlap=0,
                include_fingertips=include_fingertips,
            )
            dl_all = DataLoader(
                ds_inf_all,
                batch_size=1,
                shuffle=False,
                num_workers=int(getattr(args, "num_workers", 0)),
                pin_memory=True,
                drop_last=False,
                collate_fn=collate_stack,
            )

            # ========== stitch in JOINTS space (yaw-integrated + overlap-gauge) ==========
            full_j_gt = []
            full_j_pr = []
            first = True

            prev_last_gt = None  # (J,3) last kept frame
            prev_last_pr = None

            Tfull = int(getattr(ds_inf, "Tfull", 0))

            for i, batch in enumerate(dl):
                mB = batch["mB"]
                mH = batch["mH"]
                if not torch.is_tensor(mB):
                    mB = torch.as_tensor(mB)
                if not torch.is_tensor(mH):
                    mH = torch.as_tensor(mH)

                mB = mB.float().to(device, non_blocking=True)
                mH = mH.float().to(device, non_blocking=True)

                gt623 = torch.cat([mB, mH], dim=-1)  # (1,T,623)

                # normalize -> encode
                motion_n = (gt623 - mean) / (std + 1e-8)
                mBn = motion_n[..., :mB.shape[-1]]
                mHn = motion_n[..., mB.shape[-1]:]

                _, _, idx = model(mBn, mHn)

                # save indices per-clip
                idxH = idx["idxH"].detach().cpu().numpy()
                idxB = idx["idxB"].detach().cpu().numpy()
                idx_all = np.concatenate([idxB, idxH], axis=-1).reshape(-1)
                save_path = os.path.join(save_dir, f"{start}___{end}_{i:04d}.npz")
                np.savez_compressed(save_path, idx=idx_all)

                if not args_cli.recon:
                    continue

                # decode -> denorm 623
                pr623_n = model.decode_from_ids(
                    idxH=torch.from_numpy(idxH).to(device).long(),
                    idxB=torch.from_numpy(idxB).to(device).long(),
                )
                pr623 = pr623_n * (std + 1e-8) + mean

                # valid_len
                if ("start" in batch) and ("end" in batch):
                    clip_start = _to_int(batch["start"])
                    clip_end = _to_int(batch["end"])
                    valid_len = max(0, clip_end - clip_start)
                else:
                    clip_start = i * (clip_len - overlap)  # stride = clip_len - overlap
                    if Tfull > 0:
                        valid_len = max(0, min(clip_len, Tfull - clip_start))
                    else:
                        valid_len = gt623.shape[1]

                valid_len = min(valid_len, gt623.shape[1], pr623.shape[1])
                if valid_len <= 1:
                    continue

                gt623_v = gt623[:, :valid_len].contiguous()
                pr623_v = pr623[:, :valid_len].contiguous()

                # 1) recover each clip independently
                j_gt = recover_clip_alone(gt623_v, joints_num, use_root_loss, base_idx)  # (T,J,3)
                j_pr = recover_clip_alone(pr623_v, joints_num, use_root_loss, base_idx)

                # 2) stitch (yaw-integrated) and drop 'effective_overlap' frames for non-first clips
                if first:
                    # first clip: keep as-is (no gauge alignment)
                    j_gt_keep = j_gt
                    j_pr_keep = j_pr
                    first = False
                else:
                    j_gt_keep = stitch_yaw_integrated_and_drop(
                        j_gt, prev_last_gt, overlap=effective_overlap, face_joint_indx=face_joint_indx
                    )
                    j_pr_keep = stitch_yaw_integrated_and_drop(
                        j_pr, prev_last_pr, overlap=effective_overlap, face_joint_indx=face_joint_indx
                    )

                if j_gt_keep.shape[0] == 0:
                    continue

                # 3) carry last kept frame
                prev_last_gt = j_gt_keep[-1].detach()
                prev_last_pr = j_pr_keep[-1].detach()

                full_j_gt.append(j_gt_keep)
                full_j_pr.append(j_pr_keep)

            # baseline recover (all-at-once), only for vis compare
            gt623_all = None
            for batch_all in dl_all:
                mB_all = batch_all["mB"]
                mH_all = batch_all["mH"]
                if not torch.is_tensor(mB_all):
                    mB_all = torch.as_tensor(mB_all)
                if not torch.is_tensor(mH_all):
                    mH_all = torch.as_tensor(mH_all)

                mB_all = mB_all.float().to(device, non_blocking=True)
                mH_all = mH_all.float().to(device, non_blocking=True)
                gt623_all = torch.cat([mB_all, mH_all], dim=-1)  # (1,T,623)

            if args_cli.recon and len(full_j_gt) > 0:
                j_gt_full = torch.cat(full_j_gt, dim=0)  # (T,J,3)
                j_pr_full = torch.cat(full_j_pr, dim=0)

                if gt623_all is None:
                    raise RuntimeError("gt623_all is None (dl_all empty?)")

                # baseline "all-at-once"
                j_gt_all_trans = recover_from_ric(
                    gt623_all,
                    joints_num=joints_num,
                    use_root_loss=False,
                    base_idx=base_idx,
                )[0]
                if j_gt_all_trans.dim() == 4:
                    j_gt_all_trans = j_gt_all_trans[0]

                # align length
                T_vis = min(j_gt_all_trans.shape[0], j_gt_full.shape[0])

                out_mp4 = os.path.join(recon_dir, f"{start}___{end}.mp4")
                visualize_two_motions(
                    j_gt_all_trans[:T_vis],
                    j_gt_full[:T_vis],
                    save_path=out_mp4,
                    fps=args_cli.fps,
                    view="body",
                    rotate=False,
                    include_fingertips=include_fingertips,
                    only_gt=False,
                    origin_align=False,
                    base_idx=base_idx,
                )
                print(f"[yaw_integrated_drop{overlap}(eff{effective_overlap})] saved: {out_mp4}  (T={j_gt_full.shape[0]})")

            if args_cli.one_sequence:
                exit()


if __name__ == "__main__":
    main()