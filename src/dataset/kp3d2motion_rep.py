# src/dataset/kp3d_to_623.py
import numpy as np
import torch

from common.skeleton import Skeleton
from common.quaternion import (
    qbetween_np, qrot_np, qfix, qmul_np, qinv_np,
    quaternion_to_cont6d_np
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

def process_file(positions: np.ndarray, feet_thre: float, tgt_offsets: torch.Tensor, n_raw_offsets, kinematic_chain,
                 fid_l=(7,10), fid_r=(8,11), face_joint_indx=(2,1,17,16), l_idx1=5, l_idx2=8):
    """
    positions: (T,52,3) numpy, already Y-up
    return: data (T-1,623) numpy
    """
    # ---- uniform skeleton (same as your code) ----
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
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
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

    # ---- IK -> cont6d ----
    skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)
    quat_params = qfix(quat_params)

    r_rot = quat_params[:, 0].copy()                     # (T,4)
    cont_6d_params = quaternion_to_cont6d_np(quat_params) # (T,52,6)

    velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
    velocity = qrot_np(r_rot[1:], velocity)

    r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))  # (T-1,4)

    # ---- local pose ----
    positions_local = positions.copy()
    positions_local[..., 0] -= positions_local[:, 0:1, 0]
    positions_local[..., 2] -= positions_local[:, 0:1, 2]
    positions_local = qrot_np(np.repeat(r_rot[:, None], positions_local.shape[1], axis=1), positions_local)

    root_y = positions_local[:, 0, 1:2]                 # (T,1)
    r_velocity_y = np.arcsin(r_velocity[:, 2:3])        # (T-1,1)
    l_velocity = velocity[:, [0, 2]]                    # (T-1,2)

    root_data = np.concatenate([r_velocity_y, l_velocity, root_y[:-1]], axis=-1)  # (T-1,4)

    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)             # (T,(51)*6)
    ric_data = positions_local[:, 1:].reshape(len(positions_local), -1)           # (T,(51)*3)

    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1]).reshape(len(global_positions)-1, -1)  # (T-1, 52*3)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    # assert data.shape[1] == 623, data.shape
    return data.astype(np.float32)


def kp3d_to_motion_rep(kp3d_52_yup: np.ndarray, feet_thre: float, tgt_offsets: torch.Tensor, n_raw_offsets, kinematic_chain):
    """
    kp3d_52_yup: (T,52,3) numpy, Y-up
    returns: (T-1,motion_rep_dim) numpy
    """
    kp3d_52_yup = zup_to_yup(kp3d_52_yup)
    return process_file(kp3d_52_yup, feet_thre, tgt_offsets, n_raw_offsets, kinematic_chain)