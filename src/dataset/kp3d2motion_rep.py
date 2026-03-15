# src/dataset/kp3d_to_623.py
import numpy as np
import torch

from common.skeleton import Skeleton
from common.quaternion import (
    qbetween_np, qrot_np, qfix, qmul_np, qinv_np,
    quaternion_to_cont6d_np,
)
import sys
import os
sys.path.append(os.getcwd())
from preprocess.motion_representation import unify_clip_to_y_up

def zup_to_yup(kp):
    """
    kp: (T,52,3) numpy, Z-up
    return: (T,52,3) numpy, Y-up
    """
    kp_yup = kp.copy()
    kp_yup[..., 1] = kp[..., 2]      # y = z
    kp_yup[..., 2] = -kp[..., 1]     # z = -y
    return kp_yup

def _build_parent_map(kinematic_chain):
    """Build child→parent mapping from kinematic chain lists."""
    parent = {}
    for chain in kinematic_chain:
        for k in range(1, len(chain)):
            parent[chain[k]] = chain[k - 1]
    return parent


def _chain_to_root(parent_map, joint_idx):
    """Return the chain from root to joint_idx (inclusive)."""
    chain = [joint_idx]
    j = joint_idx
    while j in parent_map:
        j = parent_map[j]
        chain.append(j)
    chain.reverse()
    return chain


def _compute_hand_root(quat_params, global_positions, r_rot_inv,
                       kinematic_chain, lh_wrist_idx, rh_wrist_idx,
                       base_idx=0):
    """
    Compute per-hand root representation: 9D = wrist_vel(3) + wrist_rot6d(6).
    All values are relative to base_idx joint (body-relative).

    - wrist_vel: velocity of (wrist - base) in base-local frame
    - wrist_rot6d: wrist rotation relative to base joint rotation

    Args:
        quat_params: (T, J, 4) local quaternion rotations
        global_positions: (T, J, 3) world positions (after face-forward + floor removal)
        r_rot_inv: (T, 4) inverse of base joint rotation
        kinematic_chain: list of joint chains
        lh_wrist_idx, rh_wrist_idx: wrist joint indices (20, 21)
        base_idx: body reference joint index (e.g. 15 for head)

    Returns:
        lh_root: (T-1, 9) = [wrist_vel(3), wrist_rot6d(6)]
        rh_root: (T-1, 9)
    """
    parent_map = _build_parent_map(kinematic_chain)

    # Build global rotations via FK chain accumulation
    T, J, _ = quat_params.shape
    glob_rot = np.zeros_like(quat_params)  # (T, J, 4)
    glob_rot[:, 0] = quat_params[:, 0]
    # BFS through chains
    visited = set([0])
    queue = [0]
    while queue:
        p = queue.pop(0)
        for chain in kinematic_chain:
            for k in range(1, len(chain)):
                if chain[k - 1] == p and chain[k] not in visited:
                    c = chain[k]
                    glob_rot[:, c] = qmul_np(glob_rot[:, p], quat_params[:, c])
                    visited.add(c)
                    queue.append(c)

    # Base joint rotation (for body-relative computation)
    glob_base_rot = glob_rot[:, base_idx]  # (T, 4)
    glob_base_rot_inv = qinv_np(glob_base_rot)  # (T, 4)

    results = []
    for wrist_idx in [lh_wrist_idx, rh_wrist_idx]:
        # Wrist position relative to base joint: (T, 3)
        wrist_rel_pos = global_positions[:, wrist_idx] - global_positions[:, base_idx]

        # Velocity of relative position, rotated to base-local frame: (T-1, 3)
        wrist_rel_vel = wrist_rel_pos[1:] - wrist_rel_pos[:-1]  # (T-1, 3)
        wrist_rel_vel = qrot_np(glob_base_rot_inv[1:], wrist_rel_vel)  # (T-1, 3)

        # Wrist rotation relative to base joint as cont6d: (T-1, 6)
        wrist_rot_rel = qmul_np(glob_base_rot_inv[:-1], glob_rot[:-1, wrist_idx])  # (T-1, 4)
        wrist_rot6d = quaternion_to_cont6d_np(wrist_rot_rel)  # (T-1, 6)

        hand_root = np.concatenate([wrist_rel_vel, wrist_rot6d], axis=-1)  # (T-1, 9)
        results.append(hand_root)

    return results[0], results[1]


def process_file(positions: np.ndarray, feet_thre: float, tgt_offsets: torch.Tensor, n_raw_offsets, kinematic_chain,
                 fid_l=(7,10), fid_r=(8,11), face_joint_indx=(2,1,17,16), l_idx1=5, l_idx2=8, base_idx: int = 0,
                 hand_local: bool = False, lh_wrist_idx: int = 20, rh_wrist_idx: int = 21,
                 compute_hand_root: bool = False):
    """
    positions: (T,52,3) numpy, already Y-up
    return: data (T-1,623) numpy
    """
    # ---- uniform skeleton ----
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0])).numpy()
    tgt_offset = tgt_offsets.numpy()

    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()
    scale_rt = tgt_leg_len / (src_leg_len + 1e-8)

    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    src_skel.set_offset(tgt_offsets)
    positions = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)

    # ---- floor ----
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height

    # ---- center XZ by first frame root ----
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[base_idx] * np.array([1, 0, 1]) # maybe base_idx
    positions = positions - root_pose_init_xz

    # ---- face Z+ by first frame ----
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = (across1 + across2)
    across = across / (np.linalg.norm(across) + 1e-8)

    forward_init = np.cross(np.array([[0, 1, 0]]), across[None, :], axis=-1)
    forward_init = forward_init / (np.linalg.norm(forward_init, axis=-1, keepdims=True) + 1e-8)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)  # (1,4)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions = qrot_np(root_quat_init, positions)
    global_positions = positions.copy()

    # ---- foot contacts ----
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


    global_positions = positions.copy()


    # ---- IK -> cont6d ----
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)
    quat_params = qfix(quat_params)

    cont_6d_params = quaternion_to_cont6d_np(quat_params)  # jrはlocalでOK

    # base joint の global 回転 r_rot を作る
    # if base_idx == 0:
    #     r_rot = quat_params[:, 0].copy()
    # else:
    #     glob_quat = np.zeros_like(quat_params)
    #     glob_quat[:, 0] = quat_params[:, 0]
    #     for chain in skel._kinematic_tree:
    #         for k in range(1, len(chain)):
    #             p = chain[k - 1]
    #             c = chain[k]
    #             glob_quat[:, c] = qmul_np(glob_quat[:, p], quat_params[:, c])
    #     r_rot = glob_quat[:, base_idx].copy()
    rot_ref_idx = 0
    trans_ref_idx = base_idx

    r_rot = quat_params[:, rot_ref_idx].copy()

    r_rot_inv = qinv_np(r_rot)

    # ---- base linear velocity (world -> base) ----
    velocity = (positions[1:, trans_ref_idx] - positions[:-1, trans_ref_idx]).copy()   # (T-1,3)
    velocity = qrot_np(r_rot_inv[1:], velocity)

    # ---- base angular velocity ----
    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))  # (T-1,4)

    # ---- local pose ----
    positions_local = positions.copy()
    positions_local[..., 0] -= positions_local[:, trans_ref_idx:trans_ref_idx+1, 0]
    positions_local[..., 2] -= positions_local[:, trans_ref_idx:trans_ref_idx+1, 2]
    positions_local = qrot_np(np.repeat(r_rot_inv[:, None], positions_local.shape[1], axis=1), positions_local)

    # ---- hand local: RIC positions relative to wrist ----
    if hand_local:
        J = positions_local.shape[1]
        # determine hand joint ranges based on total joint count
        if J > 52:  # with fingertips (62 joints)
            lh_joints = list(range(22, 37)) + list(range(52, 57))   # left hand + left tips
            rh_joints = list(range(37, 52)) + list(range(57, 62))   # right hand + right tips
        else:       # without fingertips (52 joints)
            lh_joints = list(range(22, 37))
            rh_joints = list(range(37, 52))
        positions_local[:, lh_joints] -= positions_local[:, lh_wrist_idx:lh_wrist_idx+1]
        positions_local[:, rh_joints] -= positions_local[:, rh_wrist_idx:rh_wrist_idx+1]

    root_y = positions_local[:, trans_ref_idx, 1:2]                 # (T,1)
    r_velocity_y = np.arcsin(r_velocity[:, 2:3])        # (T-1,1)
    l_velocity = velocity[:, [0, 2]]                    # (T-1,2)

    root_data = np.concatenate([r_velocity_y, l_velocity, root_y[:-1]], axis=-1)  # (T-1,4)
    J = cont_6d_params.shape[1]  # 52 or 62
    keep_idxs = [j for j in range(J) if j != trans_ref_idx]
    rot_data = cont_6d_params[:, keep_idxs].reshape(len(cont_6d_params), -1)      # (T, (J-1)*6)
    ric_data = positions_local[:, keep_idxs].reshape(len(positions_local), -1)     # (T, (J-1)*3)

    local_vel = qrot_np(np.repeat(r_rot_inv[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])  # (T-1, J, 3)

    # ---- hand local: velocity relative to wrist ----
    if hand_local:
        local_vel[:, lh_joints] -= local_vel[:, lh_wrist_idx:lh_wrist_idx+1]
        local_vel[:, rh_joints] -= local_vel[:, rh_wrist_idx:rh_wrist_idx+1]

    local_vel = local_vel.reshape(len(global_positions)-1, -1)  # (T-1, J*3)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    # assert data.shape[1] == 623, data.shape
    # r_velocity: (T-1,4) with [w,x,y,z]
    d_half_from_quat = np.arctan2(r_velocity[:, 2], r_velocity[:, 0])  # (T-1,)
    rot_vel = r_velocity_y[:, 0]                                       # (T-1,)

    if not compute_hand_root:
        return data.astype(np.float32)

    # ---- hand root: wrist velocity + rotation per hand (body-relative) ----
    lh_root, rh_root = _compute_hand_root(
        quat_params, global_positions, r_rot_inv,
        kinematic_chain, lh_wrist_idx, rh_wrist_idx,
        base_idx=base_idx)
    return data.astype(np.float32), lh_root.astype(np.float32), rh_root.astype(np.float32)


def kp3d_to_motion_rep(
    kp3d_52_yup: np.ndarray,
    feet_thre: float,
    tgt_offsets: torch.Tensor,
    n_raw_offsets,
    kinematic_chain,
    base_idx=0,
    hand_local: bool = False,
    input_up_axis: str = "z",
    compute_hand_root: bool = False,
):
    """
    kp3d_52_yup: (T,52,3) numpy
      - input_up_axis="z": input is Z-up (legacy default, backward compatible)
      - input_up_axis="y": input is already Y-up
      - input_up_axis="auto": detect and convert to Y-up
    returns:
      - compute_hand_root=False: (T-1, motion_rep_dim) numpy
      - compute_hand_root=True:  ((T-1, motion_rep_dim), (T-1, 9), (T-1, 9)) numpy
    """
    if input_up_axis == "z":
        kp3d_yup = zup_to_yup(kp3d_52_yup)
    elif input_up_axis == "y":
        kp3d_yup = np.asarray(kp3d_52_yup, dtype=np.float32)
    elif input_up_axis == "auto":
        kp3d_yup, _, _ = unify_clip_to_y_up(kp3d_52_yup)
        kp3d_yup = np.asarray(kp3d_yup, dtype=np.float32)
    else:
        raise ValueError(f"Unknown input_up_axis={input_up_axis}. Expected one of: z, y, auto")

    return process_file(
        kp3d_yup,
        feet_thre,
        tgt_offsets,
        n_raw_offsets,
        kinematic_chain,
        base_idx=base_idx,
        hand_local=hand_local,
        compute_hand_root=compute_hand_root,
    )
