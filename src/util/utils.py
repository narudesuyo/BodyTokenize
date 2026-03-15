
import random
import numpy as np
import torch

def count_params(m: torch.nn.Module):
    n_all = sum(p.numel() for p in m.parameters())
    n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return n_all, n_train
    
def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



# body (263)
B_ROOT0, B_ROOT1 = 0, 4
B_RIC0,  B_RIC1  = 4, 67      # 21*3
B_ROT0,  B_ROT1  = 67, 193    # 21*6
B_VEL0,  B_VEL1  = 193, 259   # 22*3
B_FEET0, B_FEET1 = 259, 263   # 4

# hand (360)  ※ recon/target上では bodyの後ろに来るので +263 して使う
H0 = 263
H_RIC0, H_RIC1 = H0 + 0,   H0 + 120
H_ROT0, H_ROT1 = H0 + 120, H0 + 360
H_VEL0, H_VEL1 = H0 + 360, H0 + 480

def mse(x, y):
    return torch.mean((x - y) ** 2)

@torch.no_grad()
def compute_part_losses(recon, target, hand_root_dim: int = 0):
    """
    recon/target: (B,T, body_dim + hand_dim)
    hand_root_dim: 0 (no hand root) or 18 (9 per hand) — offset for hand local blocks
    returns dict of scalar tensors
    """
    out = {}

    # ----- body (always 263D at front) -----
    out["loss/body_root"] = mse(recon[:, :, B_ROOT0:B_ROOT1], target[:, :, B_ROOT0:B_ROOT1])
    out["loss/body_ric"]  = mse(recon[:, :, B_RIC0:B_RIC1],   target[:, :, B_RIC0:B_RIC1])
    out["loss/body_rot6d"]= mse(recon[:, :, B_ROT0:B_ROT1],   target[:, :, B_ROT0:B_ROT1])
    out["loss/body_vel"]  = mse(recon[:, :, B_VEL0:B_VEL1],   target[:, :, B_VEL0:B_VEL1])
    out["loss/body_feet"] = mse(recon[:, :, B_FEET0:B_FEET1], target[:, :, B_FEET0:B_FEET1])

    # root detail
    out["loss/root_yaw"]   = mse(recon[:, :, 0:1], target[:, :, 0:1])
    out["loss/root_vx"]    = mse(recon[:, :, 1:2], target[:, :, 1:2])
    out["loss/root_vz"]    = mse(recon[:, :, 2:3], target[:, :, 2:3])
    out["loss/root_rooty"] = mse(recon[:, :, 3:4], target[:, :, 3:4])

    # ----- hand (after body, with optional hand root prefix) -----
    h0 = H0  # 263
    hrd = hand_root_dim  # 0 or 18
    if hrd > 0:
        out["loss/hand_root"] = mse(recon[:, :, h0:h0+hrd], target[:, :, h0:h0+hrd])

    # Hand local blocks: [ric_lh, ric_rh, rot_lh, rot_rh, vel_lh, vel_rh]
    # ric: nj*3 per hand, rot: nj*6 per hand, vel: nj*3 per hand
    # ratio ric:rot:vel = 1:2:1 → total = 4 parts
    total_hand_dim = recon.shape[-1] - H0
    hand_local_dim = total_hand_dim - hrd
    if hand_local_dim > 0:
        hl0 = h0 + hrd
        n_ric = hand_local_dim // 4       # ric (both hands)
        n_rot = hand_local_dim // 4 * 2   # rot (both hands) = 2x ric
        n_vel = hand_local_dim - n_ric - n_rot  # vel (both hands) = same as ric
        out["loss/hand_ric"]   = mse(recon[:, :, hl0:hl0+n_ric],
                                      target[:, :, hl0:hl0+n_ric])
        out["loss/hand_rot6d"] = mse(recon[:, :, hl0+n_ric:hl0+n_ric+n_rot],
                                      target[:, :, hl0+n_ric:hl0+n_ric+n_rot])
        out["loss/hand_vel"]   = mse(recon[:, :, hl0+n_ric+n_rot:hl0+n_ric+n_rot+n_vel],
                                      target[:, :, hl0+n_ric+n_rot:hl0+n_ric+n_rot+n_vel])

    return out