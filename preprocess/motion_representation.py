# ee4d_pt_to_humanml_features.py
# Convert ee_train_joints.pt (dict of clips with kp3d) -> HumanML3D-style joint_vecs + recovered joints

import os
import re
import numpy as np
import torch
from tqdm import tqdm
import sys
sys.path.append("/large/naru/EgoHand/BodyTokenize")
from common.skeleton import Skeleton
from common.quaternion import (
    qbetween_np, qrot_np, qfix, qmul_np, qinv_np,
    quaternion_to_cont6d_np, quaternion_to_cont6d,
    qrot, qinv
)
from preprocess.paramUtil import t2m_raw_offsets, t2m_body_hand_kinematic_chain
import numpy as np
import torch

FOOT_IDS = [7, 8, 10, 11]  # ankle/foot（あなたの52j定義のまま）

FOOT_IDS = [7, 8, 10, 11]

def axis_stats(kp3d, joint_ids=FOOT_IDS):
    """
    kp3d: (T,J,3) torch.Tensor or np.ndarray
    """
    if torch.is_tensor(kp3d):
        p = kp3d[:, joint_ids, :]                 # (T,F,3)
        stats = {}
        for ax, c in zip(["x","y","z"], [0,1,2]):
            v = p[..., c]                         # (T,F)
            min_t = v.min(dim=1).values           # (T,)  ★ここが重要
            stats[ax] = {
                "mean_min": float(min_t.mean().item()),
                "std_min":  float(min_t.std(unbiased=False).item()),
                "range_min": float((min_t.max() - min_t.min()).item()),
            }
        return stats
    else:
        p = kp3d[:, joint_ids, :]
        stats = {}
        for ax, c in zip(["x","y","z"], [0,1,2]):
            v = p[..., c]
            min_t = v.min(axis=1)
            stats[ax] = {
                "mean_min": float(min_t.mean()),
                "std_min":  float(min_t.std()),
                "range_min": float(min_t.max() - min_t.min()),
            }
        return stats

def pick_up_axis(stats):
    # 安定性優先（std, range）、最後に mean_min
    return sorted(stats.keys(), key=lambda a: (stats[a]["std_min"], stats[a]["range_min"], stats[a]["mean_min"]))[0]

def to_y_up(kp3d, up_axis):
    """
    kp3d: (T,J,3) torch or np
    返り値: same type, Y-up に統一
    """
    is_torch = torch.is_tensor(kp3d)
    p = kp3d.clone() if is_torch else np.array(kp3d, copy=True)

    x = p[..., 0].copy() if not is_torch else p[...,0].clone()
    y = p[..., 1].copy() if not is_torch else p[...,1].clone()
    z = p[..., 2].copy() if not is_torch else p[...,2].clone()

    if up_axis == "y":
        return p

    if up_axis == "z":
        # Z-up -> Y-up: X'=X, Y'=Z, Z'=-Y  （あなたが前に書いてたやつ）
        p[..., 0] = x
        p[..., 1] = z
        p[..., 2] = -y
        return p

    if up_axis == "x":
        # X-up -> Y-up の一例
        # ここは「床が +x 方向にある」前提で回す
        # X becomes up => Y' = X
        # 残りは右手系を保つように： X' = -Y, Z' = Z など
        p[..., 0] = -y
        p[..., 1] = x
        p[..., 2] = z
        return p

    raise ValueError(up_axis)

# def unify_clip_to_y_up(kp3d):
#     if torch.is_tensor(kp3d):
#         kp3d = kp3d.detach().cpu().numpy()
#     st = axis_stats(kp3d, joint_ids=FOOT_IDS)
#     up = pick_up_axis(st)
#     kp_yup = to_y_up(kp3d, up)
#     return kp_yup, up, st
import numpy as np
import torch

face_joint_indx = [2, 1, 17, 16]  # r_hip, l_hip, sdr_r, sdr_l
FOOT_IDS = [7, 8, 10, 11]         # L_ankle, R_ankle, L_foot, R_foot

AXES = {
    "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
}

def _norm_np(v, eps=1e-8):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + eps)

def pick_up_axis_from_geom(kp3d_np, face_joint_indx, FOOT_IDS, eps=1e-8):
    """
    kp3d_np: (T,J,3) numpy
    return: up_axis(str), scores(dict), debug(dict)
    """
    rhip, lhip, rsdr, lsdr = face_joint_indx
    lank, rank, lfoot, rfoot = FOOT_IDS  # ankle/foot

    # across (left-right)
    across1 = kp3d_np[:, rhip] - kp3d_np[:, lhip]   # (T,3)
    across2 = kp3d_np[:, rsdr] - kp3d_np[:, lsdr]   # (T,3)
    v_across = across1 + across2
    v_across = _norm_np(v_across, eps)

    # foot direction (ankle -> foot), use both feet
    vL = kp3d_np[:, lfoot] - kp3d_np[:, lank]
    vR = kp3d_np[:, rfoot] - kp3d_np[:, rank]
    v_foot = _norm_np(vL, eps) + _norm_np(vR, eps)
    v_foot = _norm_np(v_foot, eps)

    # normal candidate (should be close to up)
    n = np.cross(v_foot, v_across)                 # (T,3)
    w = np.linalg.norm(n, axis=-1)                 # (T,)
    n = _norm_np(n, eps)

    # keep only reliable frames (cross magnitude not tiny)
    good = w > (0.05 * np.median(w[w > eps]) if np.any(w > eps) else 0.0)
    if good.sum() < 5:
        # too few reliable frames
        good = w > eps

    if good.sum() < 5:
        # still too few
        return None, {k: 0.0 for k in AXES}, {"good_frames": int(good.sum())}

    ng = n[good]
    wg = w[good]

    scores = {}
    for k, a in AXES.items():
        # weighted |cos|
        scores[k] = float(np.sum(wg * np.abs(ng @ a)))

    up_axis = max(scores.keys(), key=lambda k: scores[k])
    dbg = {
        "good_frames": int(good.sum()),
        "w_mean": float(wg.mean()),
        "w_median": float(np.median(wg)),
    }
    return up_axis, scores, dbg


def unify_clip_to_y_up(kp3d, face_joint_indx=[2, 1, 17, 16], FOOT_IDS=[7, 8, 10, 11]):
    """
    kp3d: (T,J,3) torch or np
    return: kp_yup(np.ndarray), up_axis(str), info(dict)
    """
    # to numpy
    if torch.is_tensor(kp3d):
        kp3d_np = kp3d.detach().cpu().numpy()
    else:
        kp3d_np = np.asarray(kp3d)

    # 1) geometry-based up-axis
    up_geom, scores, dbg = pick_up_axis_from_geom(
        kp3d_np, face_joint_indx=face_joint_indx, FOOT_IDS=FOOT_IDS
    )

    # 2) fallback: your old stat method
    st = axis_stats(kp3d_np, joint_ids=FOOT_IDS)
    up_stat = pick_up_axis(st)

    # choose
    if up_geom is None:
        up = up_stat
        method = "stat_fallback"
    else:
        # if geom vote is very weak/ambiguous, fallback to stat
        # (e.g. top score close to second)
        svals = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        ratio = (svals[0][1] + 1e-8) / (svals[1][1] + 1e-8) if len(svals) > 1 else 999.0
        if ratio < 1.05:  # almost tie -> unreliable
            up = up_stat
            method = "stat_tie_fallback"
        else:
            up = up_geom
            method = "geom"

    kp_yup = to_y_up(kp3d_np, up)

    info = {
        "method": method,
        "up_geom": up_geom,
        "up_stat": up_stat,
        "scores_geom": scores,
        "stats_minfoot": st,
        "dbg": dbg,
    }
    return kp_yup, up, info

# -----------------------------
# Helpers
# -----------------------------
def safe_name(s: str) -> str:
    s = s.replace("\\", "/")
    s = re.sub(r"[^0-9a-zA-Z_\-\/]+", "_", s)
    s = s.strip("_")
    return s

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def find_first_clip_with_kp3d(db: dict):
    for k, v in db.items():
        if isinstance(v, dict) and ("kp3d" in v):
            return k, v
    raise RuntimeError("No item with 'kp3d' found in the pt file.")


# -----------------------------
# Joint selection (SMPL-X "orig joints" style)
# We assume out.joints order begins like:
# 0 pelvis, 1 L_Hip, 2 R_Hip, 3 Spine1, 4 L_Knee, 5 R_Knee, 6 Spine2, 7 L_Ankle, 8 R_Ankle,
# 9 Spine3, 10 L_Foot, 11 R_Foot, 12 Neck, 13 L_Collar, 14 R_Collar, 15 Head,
# 16 L_Shoulder, 17 R_Shoulder, 18 L_Elbow, 19 R_Elbow, 20 L_Wrist, 21 R_Wrist,
# 22..36 left hand 15 joints, 37..51 right hand 15 joints  => total 52
# -----------------------------
BODY_JOINTS_ID = list(range(0, 22))
# LHAND_JOINTS_ID = list(range(22, 37))   # 15
# RHAND_JOINTS_ID = list(range(37, 52))   # 15
LHAND_JOINTS_ID = list(range(24, 39))   # 15
RHAND_JOINTS_ID = list(range(39, 54))   # 15
JOINTS_52_ID = BODY_JOINTS_ID + LHAND_JOINTS_ID + RHAND_JOINTS_ID
JOINTS_NUM = 52

# Lower legs indices in this 52-joint set (HumanML3D convention uses 5 and 8 for legs)
l_idx1, l_idx2 = 5, 8
# Foot joint ids (same as your pasted script)
fid_r, fid_l = [8, 11], [7, 10]
# Facing direction joints: r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]


# -----------------------------
# Core functions (ported from your pasted file, minimal changes)
# -----------------------------
def uniform_skeleton(positions: np.ndarray, target_offset: torch.Tensor, n_raw_offsets, kinematic_chain):
    """
    positions: (T, J, 3) numpy
    target_offset: (J, 3) torch
    """
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')

    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0])).numpy()
    tgt_offset = target_offset.numpy()

    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()
    scale_rt = tgt_leg_len / (src_leg_len + 1e-8)

    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def process_file(positions: np.ndarray, feet_thre: float, tgt_offsets: torch.Tensor, n_raw_offsets, kinematic_chain):
    """
    positions: (T, J, 3) numpy
    return:
      data: (T-1, feat_dim) numpy
      global_positions: (T, J, 3)
      positions_local: (T, J, 3) local+facing normalized
      l_velocity: (T-1, 2)
    """
    positions = uniform_skeleton(positions, tgt_offsets, n_raw_offsets, kinematic_chain)

    # floor
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # center XZ by first frame root
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # face Z+ by first frame across direction
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / (np.linalg.norm(across) + 1e-8)

    forward_init = np.cross(np.array([[0, 1, 0]]), across[None, :], axis=-1)

    forward_init = forward_init / (np.linalg.norm(forward_init, axis=-1, keepdims=True) + 1e-8)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)                    # (1,4)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init  # (T,J,4)

    positions = qrot_np(root_quat_init, positions)
    global_positions = positions.copy()

    # foot contacts
    def foot_detect(pos, thres):
        velfactor = np.array([thres, thres])
        feet_l = (((pos[1:, fid_l, 0] - pos[:-1, fid_l, 0]) ** 2
                 + (pos[1:, fid_l, 1] - pos[:-1, fid_l, 1]) ** 2
                 + (pos[1:, fid_l, 2] - pos[:-1, fid_l, 2]) ** 2) < velfactor).astype(np.float32)

        feet_r = (((pos[1:, fid_r, 0] - pos[:-1, fid_r, 0]) ** 2
                 + (pos[1:, fid_r, 1] - pos[:-1, fid_r, 1]) ** 2
                 + (pos[1:, fid_r, 2] - pos[:-1, fid_r, 2]) ** 2) < velfactor).astype(np.float32)
        return feet_l, feet_r

    feet_l, feet_r = foot_detect(positions, feet_thre)

    # IK -> cont6d
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)
    quat_params = qfix(quat_params)

    r_rot = quat_params[:, 0].copy()  # (T,4)

    cont_6d_params = quaternion_to_cont6d_np(quat_params)  # (T,J,6)

    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()   # (T-1,3)
    velocity = qrot_np(r_rot[1:], velocity)                    # (T-1,3)

    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))       # (T-1,4)

    # local pose + faceZ+ for ALL frames (your get_rifke)
    positions_local = positions.copy()
    positions_local[..., 0] -= positions_local[:, 0:1, 0]
    positions_local[..., 2] -= positions_local[:, 0:1, 2]
    positions_local = qrot_np(np.repeat(r_rot[:, None], positions_local.shape[1], axis=1), positions_local)

    root_y = positions_local[:, 0, 1:2]                        # (T,1)
    r_velocity_y = np.arcsin(r_velocity[:, 2:3])               # (T-1,1)
    
    l_velocity = velocity[:, [0, 2]]                           # (T-1,2)

    root_data = np.concatenate([r_velocity_y, l_velocity, root_y[:-1]], axis=-1)  # (T-1,4)

    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)             # (T,(J-1)*6)
    ric_data = positions_local[:, 1:].reshape(len(positions_local), -1)           # (T,(J-1)*3)

    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1]).reshape(len(global_positions)-1, -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data.astype(np.float32), global_positions.astype(np.float32), positions_local.astype(np.float32), l_velocity.astype(np.float32)


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


def recover_from_ric(data_t: torch.Tensor, joints_num: int):
    r_rot_quat, r_pos = recover_root_rot_pos(data_t)
    positions = data_t[..., 4:(joints_num - 1) * 3 + 4].view(data_t.shape[:-1] + (-1, 3))

    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
    return positions


import numpy as np
import torch

def convert_zup_to_yup(kp3d):
    """
    kp3d: (T, J, 3)
      - torch.Tensor or np.ndarray
    Z-up -> Y-up:
      X' = X
      Y' = Z
      Z' = -Y
    """
    if torch.is_tensor(kp3d):
        kp = kp3d.clone()
        x = kp[..., 0]
        y = kp[..., 1]
        z = kp[..., 2]
        kp[..., 0] = x
        kp[..., 1] = z
        kp[..., 2] = -y
        return kp
    else:
        kp = np.array(kp3d, copy=True)
        x = kp[..., 0]
        y = kp[..., 1]
        z = kp[..., 2]
        kp[..., 0] = x
        kp[..., 1] = z
        kp[..., 2] = -y
        return kp

# -----------------------------
# Main
# -----------------------------
def main(in_pt: str, out_root: str, feet_thre: float = 0.002, limit: int = -1):
    ensure_dir(out_root)
    out_vec_dir = os.path.join(out_root, "new_joint_vecs")
    out_jnt_dir = os.path.join(out_root, "new_joints")
    ensure_dir(out_vec_dir)
    ensure_dir(out_jnt_dir)

    db = torch.load(in_pt, map_location="cpu", weights_only=False)

    # skeleton setup
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets).float()
    kinematic_chain = t2m_body_hand_kinematic_chain

    # choose target offsets from the FIRST clip's first frame (no external npy needed)
    k0, item0 = find_first_clip_with_kp3d(db)
    kp0 = item0["kp3d"]
    if torch.is_tensor(kp0):
        kp0 = kp0.cpu().numpy()
    kp0 = kp0[:, JOINTS_52_ID, :]  # (T,52,3)
    kp0, up, st = unify_clip_to_y_up(kp0)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(kp0[0]).float())  # (52,3)

    keys = list(db.keys())
    if limit > 0:
        keys = keys[:limit]

    total_frames = 0
    ok = 0
    # l = 0
    for k in tqdm(keys, desc="process clips"):
        item = db[k]
        if not isinstance(item, dict) or ("kp3d" not in item):
            continue

        kp = item["kp3d"]
        # kp = convert_zup_to_yup(kp)
        if torch.is_tensor(kp):
            kp = kp.cpu().numpy()

        if kp.ndim != 3 or kp.shape[-1] != 3:
            print(f"[skip] {k}: kp3d shape unexpected: {kp.shape}")
            continue
        if kp.shape[1] < 52:
            print(f"[skip] {k}: kp3d joints too small: J={kp.shape[1]}")
            continue

        positions = kp[:, JOINTS_52_ID, :]  # (T,52,3)
        positions, up, st = unify_clip_to_y_up(positions)


        try:
            data, global_pos, local_pos, l_vel = process_file(
                positions,
                feet_thre=feet_thre,
                tgt_offsets=tgt_offsets,
                n_raw_offsets=n_raw_offsets,
                kinematic_chain=kinematic_chain
            )

            # save vecs
            name = safe_name(str(k))
            vec_path = os.path.join(out_vec_dir, f"{name}.npy")
            np.save(vec_path, data)

            # recover joints from RIC (like HumanML3D script)
            rec = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), JOINTS_NUM)  # (1,T-1,52,3)
            rec = rec.squeeze(0).cpu().numpy()
            jnt_path = os.path.join(out_jnt_dir, f"{name}.npy")
            np.save(jnt_path, rec)

            total_frames += data.shape[0]
            ok += 1
            # l += 1
            # if l > 10:
            #     break
        except Exception as e:
            print(f"[error] {k}: {e}")

    print(f"done. clips={ok}, frames(T-1 sum)={total_frames}, minutes@20fps={total_frames/20/60:.2f}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_pt", default="/large/naru/EgoHand/data/ee4d/ee4d_motion_uniegomotion/uniegomotion/ee_train_joints.pt", help="ee_train_joints.pt (dict with kp3d)")
    ap.add_argument("--out_root", default="/large/naru/EgoHand/data/ee4d/ee4d_motion_uniegomotion/uniegomotion/ee_train_joints_motion_representation", help="output root dir (will create new_joint_vecs/ new_joints/)")
    ap.add_argument("--feet_thre", type=float, default=0.002)
    ap.add_argument("--limit", type=int, default=-1, help="process only first N keys (debug)")
    args = ap.parse_args()
    main(args.in_pt, args.out_root, feet_thre=args.feet_thre, limit=args.limit)




# ee4d_pt_to_humanml_features.py
# Convert ee_train_joints.pt (dict of clips with kp3d) -> HumanML3D-style joint_vecs + recovered joints
# + add fingertip joints from SMPL-X vertices (10 joints: 5 left + 5 right)

# import os
# import re
# import sys
# import numpy as np
# import torch
# from tqdm import tqdm

# sys.path.append("/large/naru/EgoHand/BodyTokenize")
# from common.skeleton import Skeleton
# from common.quaternion import (
#     qbetween_np, qrot_np, qfix, qmul_np, qinv_np,
#     quaternion_to_cont6d_np, qrot, qinv
# )
# from preprocess.paramUtil import t2m_raw_offsets, t2m_body_hand_kinematic_chain

# import smplx

# # -----------------------------
# # fingertip vertex indices (from your make_hand_regressor)
# # -----------------------------
# TIP_VERTS_L = [5361, 4933, 5058, 5169, 5286]
# TIP_VERTS_R = [8079, 7669, 7794, 7905, 8022]
# TIP_NAMES = [
#     "L_tip_thumb", "L_tip_index", "L_tip_middle", "L_tip_ring", "L_tip_pinky",
#     "R_tip_thumb", "R_tip_index", "R_tip_middle", "R_tip_ring", "R_tip_pinky",
# ]

# HUMAN_MODEL_PATH = "./models"

# def get_smplx_layer(device, gender="neutral", hand_pca_comps=12, batch_size=1):
#     g = {"neutral": "NEUTRAL", "male": "MALE", "female": "FEMALE"}[gender.lower()]
#     layer = smplx.create(
#         HUMAN_MODEL_PATH,
#         "smplx",
#         gender=g,
#         use_pca=True,
#         num_pca_comps=hand_pca_comps,
#         use_face_contour=False,   # face不要
#         batch_size=batch_size,
#     ).to(device)
#     layer.eval()
#     return layer

# def rot6d_to_rotmat(x6d: torch.Tensor) -> torch.Tensor:
#     a1 = x6d[..., 0:3]
#     a2 = x6d[..., 3:6]
#     b1 = torch.nn.functional.normalize(a1, dim=-1)
#     b2 = torch.nn.functional.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
#     b3 = torch.cross(b1, b2, dim=-1)
#     return torch.stack([b1, b2, b3], dim=-1)  # [...,3,3]

# def rotmat_to_axis_angle(rotmat: torch.Tensor):
#     R = rotmat
#     cos = (R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2] - 1) / 2
#     cos = torch.clamp(cos, -1 + 1e-6, 1 - 1e-6)
#     angle = torch.acos(cos)
#     rx = R[..., 2, 1] - R[..., 1, 2]
#     ry = R[..., 0, 2] - R[..., 2, 0]
#     rz = R[..., 1, 0] - R[..., 0, 1]
#     axis = torch.stack([rx, ry, rz], dim=-1)
#     axis = axis / (2 * torch.sin(angle)[..., None] + 1e-8)
#     return axis * angle[..., None]

# @torch.no_grad()
# def smpl_params_to_vertices_and_joints(smplx_layer, smpl_params: dict, device):
#     """
#     smpl_params:
#       global_orient [T,6]
#       body_pose     [T,21,6]
#       left/right_hand_pose [T,12] (PCA coeffs)
#       transl [T,3], betas [1,10]
#     """
#     go6 = smpl_params["global_orient"].to(device)        # [T,6]
#     bp6 = smpl_params["body_pose"].to(device)            # [T,21,6]
#     lh  = smpl_params["left_hand_pose"].to(device)       # [T,12]
#     rh  = smpl_params["right_hand_pose"].to(device)      # [T,12]
#     tr  = smpl_params["transl"].to(device)               # [T,3]
#     bt  = smpl_params["betas"].to(device)                # [1,10]

#     T = go6.shape[0]
#     betas = bt.expand(T, -1)                             # [T,10]

#     goR = rot6d_to_rotmat(go6).transpose(-1, -2)         # [...,3,3]
#     bpR = rot6d_to_rotmat(bp6).transpose(-1, -2)         # [...,3,3]
#     goA = rotmat_to_axis_angle(goR)                      # [T,3]
#     bpA = rotmat_to_axis_angle(bpR)                      # [T,21,3]

#     out = smplx_layer(
#         betas=betas,
#         global_orient=goA,
#         body_pose=bpA,
#         left_hand_pose=lh,
#         right_hand_pose=rh,
#         transl=tr,
#         pose2rot=True,
#     )
#     # out.vertices: [T,V,3], out.joints: [T,J,3]
#     return out.vertices, out.joints

# @torch.no_grad()
# def add_fingertips_to_kp3d(item: dict, device="cuda:0", gender="neutral"):
#     """
#     item must contain 'smpl_params' and 'kp3d'
#     returns: kp3d_aug (np.ndarray): [T, J+10, 3]
#     """
#     kp = item["kp3d"]
#     if torch.is_tensor(kp):
#         kp_np = kp.detach().cpu().numpy()
#     else:
#         kp_np = np.asarray(kp)

#     smpl_params = item.get("smpl_params", None)
#     if smpl_params is None:
#         return kp_np  # can't add

#     T = kp_np.shape[0]
#     smplx_layer = get_smplx_layer(device, gender=gender, hand_pca_comps=12, batch_size=T)
#     verts, _ = smpl_params_to_vertices_and_joints(smplx_layer, smpl_params, device=device)
#     verts_np = verts.detach().cpu().numpy()  # [T,V,3]

#     tips_L = verts_np[:, TIP_VERTS_L, :]  # [T,5,3]
#     tips_R = verts_np[:, TIP_VERTS_R, :]  # [T,5,3]
#     tips = np.concatenate([tips_L, tips_R], axis=1)  # [T,10,3]

#     # ★ ここで「joint軸の末尾」に追加
#     kp_aug = np.concatenate([kp_np, tips], axis=1)   # [T,J+10,3]
#     return kp_aug


# # -----------------------------
# # up-axis unify (your code, unchanged)
# # -----------------------------
# FOOT_IDS = [7, 8, 10, 11]
# face_joint_indx = [2, 1, 17, 16]  # r_hip, l_hip, sdr_r, sdr_l

# AXES = {
#     "x": np.array([1.0, 0.0, 0.0], dtype=np.float32),
#     "y": np.array([0.0, 1.0, 0.0], dtype=np.float32),
#     "z": np.array([0.0, 0.0, 1.0], dtype=np.float32),
# }
# def _norm_np(v, eps=1e-8):
#     n = np.linalg.norm(v, axis=-1, keepdims=True)
#     return v / (n + eps)

# def axis_stats(kp3d, joint_ids=FOOT_IDS):
#     p = kp3d[:, joint_ids, :]
#     stats = {}
#     for ax, c in zip(["x","y","z"], [0,1,2]):
#         v = p[..., c]
#         min_t = v.min(axis=1)
#         stats[ax] = {
#             "mean_min": float(min_t.mean()),
#             "std_min":  float(min_t.std()),
#             "range_min": float(min_t.max() - min_t.min()),
#         }
#     return stats

# def pick_up_axis(stats):
#     return sorted(stats.keys(), key=lambda a: (stats[a]["std_min"], stats[a]["range_min"], stats[a]["mean_min"]))[0]

# def to_y_up(kp3d, up_axis):
#     p = np.array(kp3d, copy=True)
#     x = p[..., 0].copy()
#     y = p[..., 1].copy()
#     z = p[..., 2].copy()

#     if up_axis == "y":
#         return p
#     if up_axis == "z":
#         p[..., 0] = x
#         p[..., 1] = z
#         p[..., 2] = -y
#         return p
#     if up_axis == "x":
#         p[..., 0] = -y
#         p[..., 1] = x
#         p[..., 2] = z
#         return p
#     raise ValueError(up_axis)

# def pick_up_axis_from_geom(kp3d_np, face_joint_indx, FOOT_IDS, eps=1e-8):
#     rhip, lhip, rsdr, lsdr = face_joint_indx
#     lank, rank, lfoot, rfoot = FOOT_IDS

#     across1 = kp3d_np[:, rhip] - kp3d_np[:, lhip]
#     across2 = kp3d_np[:, rsdr] - kp3d_np[:, lsdr]
#     v_across = _norm_np(across1 + across2, eps)

#     vL = kp3d_np[:, lfoot] - kp3d_np[:, lank]
#     vR = kp3d_np[:, rfoot] - kp3d_np[:, rank]
#     v_foot = _norm_np(vL, eps) + _norm_np(vR, eps)
#     v_foot = _norm_np(v_foot, eps)

#     n = np.cross(v_foot, v_across)
#     w = np.linalg.norm(n, axis=-1)
#     n = _norm_np(n, eps)

#     good = w > (0.05 * np.median(w[w > eps]) if np.any(w > eps) else 0.0)
#     if good.sum() < 5:
#         good = w > eps
#     if good.sum() < 5:
#         return None, {k: 0.0 for k in AXES}, {"good_frames": int(good.sum())}

#     ng = n[good]
#     wg = w[good]

#     scores = {}
#     for k, a in AXES.items():
#         scores[k] = float(np.sum(wg * np.abs(ng @ a)))

#     up_axis = max(scores.keys(), key=lambda k: scores[k])
#     dbg = {"good_frames": int(good.sum()), "w_mean": float(wg.mean()), "w_median": float(np.median(wg))}
#     return up_axis, scores, dbg

# def unify_clip_to_y_up(kp3d, face_joint_indx=[2, 1, 17, 16], FOOT_IDS=[7, 8, 10, 11]):
#     kp3d_np = np.asarray(kp3d)

#     up_geom, scores, dbg = pick_up_axis_from_geom(kp3d_np, face_joint_indx, FOOT_IDS)
#     st = axis_stats(kp3d_np, joint_ids=FOOT_IDS)
#     up_stat = pick_up_axis(st)

#     if up_geom is None:
#         up = up_stat
#     else:
#         svals = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#         ratio = (svals[0][1] + 1e-8) / (svals[1][1] + 1e-8) if len(svals) > 1 else 999.0
#         up = up_stat if ratio < 1.05 else up_geom

#     return to_y_up(kp3d_np, up), up, {"up_geom": up_geom, "up_stat": up_stat, "scores_geom": scores, "stats_minfoot": st, "dbg": dbg}


# # -----------------------------
# # misc helpers
# # -----------------------------
# def safe_name(s: str) -> str:
#     s = s.replace("\\", "/")
#     s = re.sub(r"[^0-9a-zA-Z_\-\/]+", "_", s)
#     s = s.strip("_")
#     return s

# def ensure_dir(p: str):
#     os.makedirs(p, exist_ok=True)

# def find_first_clip_with_kp3d(db: dict):
#     for k, v in db.items():
#         if isinstance(v, dict) and ("kp3d" in v):
#             return k, v
#     raise RuntimeError("No item with 'kp3d' found in the pt file.")


# # -----------------------------
# # Joint selection
# # -----------------------------
# # NOTE:
# # - This part depends on your kp3d joint order.
# # - Since we APPEND tips at the end, existing indices stay the same.
# BODY_JOINTS_ID = list(range(0, 22))
# LHAND_JOINTS_ID = list(range(24, 39))   # your current mapping
# RHAND_JOINTS_ID = list(range(39, 54))   # your current mapping

# JOINTS_52_ID = BODY_JOINTS_ID + LHAND_JOINTS_ID + RHAND_JOINTS_ID
# JOINTS_NUM = 52

# # HumanML skeleton params
# l_idx1, l_idx2 = 5, 8
# fid_r, fid_l = [8, 11], [7, 10]
# face_joint_indx = [2, 1, 17, 16]


# def uniform_skeleton(positions: np.ndarray, target_offset: torch.Tensor, n_raw_offsets, kinematic_chain):
#     src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
#     src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0])).numpy()
#     tgt_offset = target_offset.numpy()

#     src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
#     tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()
#     scale_rt = tgt_leg_len / (src_leg_len + 1e-8)

#     src_root_pos = positions[:, 0]
#     tgt_root_pos = src_root_pos * scale_rt

#     quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
#     src_skel.set_offset(target_offset)
#     new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
#     return new_joints


# def process_file(positions: np.ndarray, feet_thre: float, tgt_offsets: torch.Tensor, n_raw_offsets, kinematic_chain):
#     positions = uniform_skeleton(positions, tgt_offsets, n_raw_offsets, kinematic_chain)

#     floor_height = positions.min(axis=0).min(axis=0)[1]
#     positions[:, :, 1] -= floor_height

#     root_pos_init = positions[0]
#     root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
#     positions = positions - root_pose_init_xz

#     r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
#     across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
#     across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
#     across = across1 + across2
#     across = across / (np.linalg.norm(across) + 1e-8)

#     forward_init = np.cross(np.array([[0, 1, 0]]), across[None, :], axis=-1)
#     forward_init = forward_init / (np.linalg.norm(forward_init, axis=-1, keepdims=True) + 1e-8)

#     target = np.array([[0, 0, 1]])
#     root_quat_init = qbetween_np(forward_init, target)
#     root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

#     positions = qrot_np(root_quat_init, positions)
#     global_positions = positions.copy()

#     def foot_detect(pos, thres):
#         velfactor = np.array([thres, thres])
#         feet_l = (((pos[1:, fid_l, 0] - pos[:-1, fid_l, 0]) ** 2
#                  + (pos[1:, fid_l, 1] - pos[:-1, fid_l, 1]) ** 2
#                  + (pos[1:, fid_l, 2] - pos[:-1, fid_l, 2]) ** 2) < velfactor).astype(np.float32)

#         feet_r = (((pos[1:, fid_r, 0] - pos[:-1, fid_r, 0]) ** 2
#                  + (pos[1:, fid_r, 1] - pos[:-1, fid_r, 1]) ** 2
#                  + (pos[1:, fid_r, 2] - pos[:-1, fid_r, 2]) ** 2) < velfactor).astype(np.float32)
#         return feet_l, feet_r

#     feet_l, feet_r = foot_detect(positions, feet_thre)

#     skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
#     quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)
#     quat_params = qfix(quat_params)

#     r_rot = quat_params[:, 0].copy()
#     cont_6d_params = quaternion_to_cont6d_np(quat_params)

#     velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
#     velocity = qrot_np(r_rot[1:], velocity)

#     r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))

#     positions_local = positions.copy()
#     positions_local[..., 0] -= positions_local[:, 0:1, 0]
#     positions_local[..., 2] -= positions_local[:, 0:1, 2]
#     positions_local = qrot_np(np.repeat(r_rot[:, None], positions_local.shape[1], axis=1), positions_local)

#     root_y = positions_local[:, 0, 1:2]
#     r_velocity_y = np.arcsin(r_velocity[:, 2:3])
#     l_velocity = velocity[:, [0, 2]]

#     root_data = np.concatenate([r_velocity_y, l_velocity, root_y[:-1]], axis=-1)

#     rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)
#     ric_data = positions_local[:, 1:].reshape(len(positions_local), -1)

#     local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
#                         global_positions[1:] - global_positions[:-1]).reshape(len(global_positions)-1, -1)

#     data = root_data
#     data = np.concatenate([data, ric_data[:-1]], axis=-1)
#     data = np.concatenate([data, rot_data[:-1]], axis=-1)
#     data = np.concatenate([data, local_vel], axis=-1)
#     data = np.concatenate([data, feet_l, feet_r], axis=-1)

#     return data.astype(np.float32), global_positions.astype(np.float32), positions_local.astype(np.float32), l_velocity.astype(np.float32)


# def recover_root_rot_pos(data_t: torch.Tensor):
#     rot_vel = data_t[..., 0]
#     r_rot_ang = torch.zeros_like(rot_vel)
#     r_rot_ang[..., 1:] = rot_vel[..., :-1]
#     r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

#     r_rot_quat = torch.zeros(data_t.shape[:-1] + (4,), device=data_t.device)
#     r_rot_quat[..., 0] = torch.cos(r_rot_ang)
#     r_rot_quat[..., 2] = torch.sin(r_rot_ang)

#     r_pos = torch.zeros(data_t.shape[:-1] + (3,), device=data_t.device)
#     r_pos[..., 1:, [0, 2]] = data_t[..., :-1, 1:3]
#     r_pos = qrot(qinv(r_rot_quat), r_pos)
#     r_pos = torch.cumsum(r_pos, dim=-2)
#     r_pos[..., 1] = data_t[..., 3]
#     return r_rot_quat, r_pos

# def recover_from_ric(data_t: torch.Tensor, joints_num: int):
#     r_rot_quat, r_pos = recover_root_rot_pos(data_t)
#     positions = data_t[..., 4:(joints_num - 1) * 3 + 4].view(data_t.shape[:-1] + (-1, 3))

#     positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)
#     positions[..., 0] += r_pos[..., 0:1]
#     positions[..., 2] += r_pos[..., 2:3]
#     positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)
#     return positions


# # -----------------------------
# # Main
# # -----------------------------
# def main(in_pt: str, out_root: str, feet_thre: float = 0.002, limit: int = -1, device="cuda:0", gender="neutral"):
#     ensure_dir(out_root)
#     out_vec_dir = os.path.join(out_root, "new_joint_vecs")
#     out_jnt_dir = os.path.join(out_root, "new_joints")
#     ensure_dir(out_vec_dir)
#     ensure_dir(out_jnt_dir)

#     db = torch.load(in_pt, map_location="cpu", weights_only=False)

#     n_raw_offsets = torch.from_numpy(t2m_raw_offsets).float()
#     kinematic_chain = t2m_body_hand_kinematic_chain

#     # target offsets from first clip
#     k0, item0 = find_first_clip_with_kp3d(db)
#     kp0 = add_fingertips_to_kp3d(item0, device=device, gender=gender)  # ★追加
#     kp0 = kp0[:, JOINTS_52_ID, :]
#     kp0, up, info = unify_clip_to_y_up(kp0)

#     tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
#     tgt_offsets = tgt_skel.get_offsets_joints(torch.from_numpy(kp0[0]).float())

#     keys = list(db.keys())
#     if limit > 0:
#         keys = keys[:limit]

#     total_frames = 0
#     ok = 0

#     for k in tqdm(keys, desc="process clips"):
#         item = db[k]
#         if not isinstance(item, dict) or ("kp3d" not in item):
#             continue

#         kp_aug = add_fingertips_to_kp3d(item, device=device, gender=gender)  # ★追加
#         if kp_aug.ndim != 3 or kp_aug.shape[-1] != 3:
#             print(f"[skip] {k}: kp3d shape unexpected: {kp_aug.shape}")
#             continue

#         if kp_aug.shape[1] < 52:
#             print(f"[skip] {k}: kp3d joints too small: J={kp_aug.shape[1]}")
#             continue

#         positions = kp_aug[:, JOINTS_52_ID, :]
#         positions, up, info = unify_clip_to_y_up(positions)

#         try:
#             data, global_pos, local_pos, l_vel = process_file(
#                 positions,
#                 feet_thre=feet_thre,
#                 tgt_offsets=tgt_offsets,
#                 n_raw_offsets=n_raw_offsets,
#                 kinematic_chain=kinematic_chain
#             )

#             name = safe_name(str(k))
#             np.save(os.path.join(out_vec_dir, f"{name}.npy"), data)

#             rec = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), JOINTS_NUM)
#             rec = rec.squeeze(0).cpu().numpy()
#             np.save(os.path.join(out_jnt_dir, f"{name}.npy"), rec)

#             total_frames += data.shape[0]
#             ok += 1
#         except Exception as e:
#             print(f"[error] {k}: {e}")

#     print(f"done. clips={ok}, frames(T-1 sum)={total_frames}, minutes@20fps={total_frames/20/60:.2f}")
#     print("Added fingertip joints (appended at the END of kp3d joint axis):")
#     print("  +10 joints in this order:", TIP_NAMES)


# if __name__ == "__main__":
#     import argparse
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--in_pt", default="/large/naru/EgoHand/data/ee4d/ee4d_motion_uniegomotion/uniegomotion/ee_train_joints.pt")
#     ap.add_argument("--out_root", default="/large/naru/EgoHand/data/ee4d/ee4d_motion_uniegomotion/uniegomotion/ee_train_joints_new.pt")
#     ap.add_argument("--feet_thre", type=float, default=0.002)
#     ap.add_argument("--limit", type=int, default=-1)
#     ap.add_argument("--device", default="cuda:0")
#     ap.add_argument("--gender", default="neutral")
#     args = ap.parse_args()
#     main(args.in_pt, args.out_root, feet_thre=args.feet_thre, limit=args.limit, device=args.device, gender=args.gender)