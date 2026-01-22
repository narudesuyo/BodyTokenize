import torch

@torch.no_grad()
def codebook_stats(idx: torch.Tensor, K: int):
    flat = idx.reshape(-1)
    hist = torch.bincount(flat, minlength=K).float()
    p = hist / (hist.sum() + 1e-8)
    usage = (hist > 0).float().mean().item()
    perplexity = torch.exp(-(p * (p + 1e-12).log()).sum()).item()
    return usage, perplexity

def batch_procrustes_align(pred, gt, eps=1e-8):
    """
    pred, gt: (B, T, J, 3)
    return: aligned_pred (B, T, J, 3)
    """
    B, T, J, _ = pred.shape
    pred = pred.reshape(B*T, J, 3)
    gt   = gt.reshape(B*T, J, 3)

    mu_pred = pred.mean(dim=1, keepdim=True)
    mu_gt   = gt.mean(dim=1, keepdim=True)

    pred_c = pred - mu_pred
    gt_c   = gt - mu_gt

    H = pred_c.transpose(1, 2) @ gt_c        # (BT,3,3)
    U, S, Vt = torch.linalg.svd(H)

    R = Vt.transpose(1, 2) @ U.transpose(1, 2)

    # reflection fix
    det = torch.det(R)
    Vt[det < 0, -1, :] *= -1
    R = Vt.transpose(1, 2) @ U.transpose(1, 2)

    var_pred = (pred_c ** 2).sum(dim=(1, 2))
    scale = (S.sum(dim=1) / (var_pred + eps)).view(-1, 1, 1)

    pred_aligned = scale * (pred_c @ R.transpose(1, 2)) + mu_gt
    return pred_aligned.view(B, T, J, 3)

def batch_procrustes_align_sequence(
    pred, gt, eps=1e-8, return_transform=False
):
    """
    pred, gt: (B, T, J, 3)
    return:
        aligned_pred: (B, T, J, 3)
        (optional) R: (B, 3, 3)
        (optional) scale: (B, 1, 1)
        (optional) t: (B, 1, 3)
    シーケンスごとに 1 つの similarity transform を推定
    """
    B, T, J, _ = pred.shape

    # (B, N, 3), N = T * J
    pred_flat = pred.reshape(B, T * J, 3)
    gt_flat   = gt.reshape(B, T * J, 3)

    mu_pred = pred_flat.mean(dim=1, keepdim=True)  # (B,1,3)
    mu_gt   = gt_flat.mean(dim=1, keepdim=True)

    pred_c = pred_flat - mu_pred
    gt_c   = gt_flat - mu_gt

    # cross-covariance
    H = pred_c.transpose(1, 2) @ gt_c              # (B,3,3)
    U, S, Vt = torch.linalg.svd(H)

    R = Vt.transpose(1, 2) @ U.transpose(1, 2)

    # reflection fix
    det = torch.det(R)
    Vt[det < 0, -1, :] *= -1
    R = Vt.transpose(1, 2) @ U.transpose(1, 2)

    # scale
    var_pred = (pred_c ** 2).sum(dim=(1, 2))       # (B,)
    scale = (S.sum(dim=1) / (var_pred + eps)).view(B, 1, 1)

    # translation
    t = mu_gt - scale * (mu_pred @ R.transpose(1, 2))  # (B,1,3)

    # apply transform
    pred_aligned_flat = scale * (pred_flat @ R.transpose(1, 2)) + t
    pred_aligned = pred_aligned_flat.view(B, T, J, 3)

    if return_transform:
        return pred_aligned, R, scale, t
    else:
        return pred_aligned

def mpjpe_bt(jp, jg, slc):
    d = torch.norm(jp[..., slc, :] - jg[..., slc, :], dim=-1)  # (B,T,Jpart)
    return d.mean(dim=2)  # (B,T)

def wa_mpjpe(jp, jg, slc):
    B = jp.shape[0]
    jp = jp[..., slc, :]
    jg = jg[..., slc, :]
    B, T, J, _ = jp.shape

    jp_pa = batch_procrustes_align_sequence(jp, jg)
    return torch.norm(jp_pa.reshape(B, T*J, 3) - jg.reshape(B, T*J, 3), dim=-1).mean(dim=1)  # (B,)

def w_mpjpe_firstk(jp, jg, slc, num_align_frames=1):
    """
    Weighted MPJPE where Procrustes alignment is computed
    using the first K frames, and applied to all frames.

    jp, jg: (B, T, J, 3)
    slc: joint slice
    num_align_frames: K (int)
    return: (B,)
    """
    jp = jp[..., slc, :]
    jg = jg[..., slc, :]
    B, T, J, _ = jp.shape

    K = min(num_align_frames, T)
    if K < 1:
        raise ValueError("num_align_frames must be >= 1")

    # --- use first K frames to estimate PA ---
    jp_init = jp[:, :K]   # (B, K, J, 3)
    jg_init = jg[:, :K]

    # flatten time & joints for PA estimation
    jp_init_f = jp_init.reshape(B, -1, 3)  # (B, K*J, 3)
    jg_init_f = jg_init.reshape(B, -1, 3)

    # estimate Procrustes transform
    _, R, s, t = batch_procrustes_align_sequence(
        jp_init_f.view(B, 1, -1, 3),  # fake T=1 to reuse function
        jg_init_f.view(B, 1, -1, 3),
        return_transform=True
    )

    # --- apply transform to all frames ---
    jp_all = jp.reshape(B, -1, 3)  # (B, T*J, 3)
    jp_all_pa = s * (jp_all @ R.transpose(-1, -2)) + t
    jp_all_pa = jp_all_pa.reshape(B, T, J, 3)

    # --- compute sequence MPJPE ---
    err = torch.norm(jp_all_pa - jg, dim=-1)  # (B, T, J)
    return err.mean(dim=(1, 2))  # (B,)

# def rte_joint(jp, jg, joint_idx: int):
#     """
#     Root Translation Error for a specific joint trajectory.
#     jp, jg: (B, T, J, 3)
#     return: (B,)
#     """
#     p = jp[:, :, joint_idx]  # (B,T,3)
#     g = jg[:, :, joint_idx]
#     return torch.norm(p - g, dim=-1).mean(dim=1)

import torch

def root_translation_error(
    jp: torch.Tensor,
    jg: torch.Tensor,
    joint_idx: int,
    *,
    use_scale: bool = False,   # 論文が "rigid alignment" なら False 推奨（R,t のみ）
    eps: float = 1e-8,
):
    """
    RTE(%) for a specific joint trajectory, normalized by GT trajectory displacement Δ.
    jp, jg: (B, T, J, 3)
    return: (B,)  # percent
    """
    assert jp.shape == jg.shape and jp.ndim == 4
    B, T, J, _ = jp.shape
    assert 0 <= joint_idx < J

    # 1) Align pred->gt with a single transform per sequence
    #    Your function estimates Sim(3): (R, scale, t)
    aligned, R, scale, t = batch_procrustes_align_sequence(jp, jg, eps=eps, return_transform=True)

    if not use_scale:
        # Re-apply only rigid part (R,t) using the R,t that came out.
        # (We recompute t as mu_gt - (mu_pred R^T) to remove scale effect.)
        pred_flat = jp.reshape(B, T * J, 3)
        gt_flat   = jg.reshape(B, T * J, 3)
        mu_pred = pred_flat.mean(dim=1, keepdim=True)  # (B,1,3)
        mu_gt   = gt_flat.mean(dim=1, keepdim=True)    # (B,1,3)
        t_rigid = mu_gt - (mu_pred @ R.transpose(1, 2))  # (B,1,3)

        aligned_flat = (pred_flat @ R.transpose(1, 2)) + t_rigid
        aligned = aligned_flat.view(B, T, J, 3)

    # 2) Take the joint trajectory
    p = aligned[:, :, joint_idx, :]   # (B,T,3)
    g = jg[:, :, joint_idx, :]        # (B,T,3)

    # 3) Numerator: mean per-frame L2 error
    num = torch.norm(p - g, dim=-1).mean(dim=1)  # (B,)

    # 4) Denominator Δ: total GT displacement along time
    disp = torch.norm(g[:, 1:] - g[:, :-1], dim=-1).sum(dim=1)  # (B,)

    rte = (num / (disp + eps)) * 100.0
    return rte

def relative_translation_error(
    jp: torch.Tensor,
    jg: torch.Tensor,
    joint_idx: int,
    *,
    use_scale: bool = False,   # 論文は rigid alignment → False 推奨
    eps: float = 1e-8,
):
    """
    Relative Translation Error (RTE %)
    jp, jg: (B, T, J, 3)
    return: (B,)  # percent
    """
    assert jp.shape == jg.shape and jp.ndim == 4
    B, T, J, _ = jp.shape
    assert 0 <= joint_idx < J

    # 1) Sequence-wise rigid alignment (pred -> gt)
    aligned, R, scale, t = batch_procrustes_align_sequence(
        jp, jg, eps=eps, return_transform=True
    )

    if not use_scale:
        # remove scale: keep only R,t
        pred_flat = jp.reshape(B, T * J, 3)
        gt_flat   = jg.reshape(B, T * J, 3)

        mu_pred = pred_flat.mean(dim=1, keepdim=True)
        mu_gt   = gt_flat.mean(dim=1, keepdim=True)

        t_rigid = mu_gt - (mu_pred @ R.transpose(1, 2))

        aligned_flat = (pred_flat @ R.transpose(1, 2)) + t_rigid
        aligned = aligned_flat.view(B, T, J, 3)

    # 2) Take joint trajectories
    p = aligned[:, :, joint_idx, :]   # (B,T,3)
    g = jg[:, :, joint_idx, :]        # (B,T,3)

    # 3) Frame-to-frame displacements
    dp = p[:, 1:] - p[:, :-1]         # (B,T-1,3)
    dg = g[:, 1:] - g[:, :-1]

    # 4) Numerator: mean per-step displacement error
    num = torch.norm(dp - dg, dim=-1).mean(dim=1)   # (B,)

    # 5) Denominator Δ: total GT displacement
    disp = torch.norm(dg, dim=-1).sum(dim=1)        # (B,)

    rte_rel = (num / (disp + eps)) * 100.0
    return rte_rel

def accel_joint(jp, jg, joint_idx: int, fps=10):
    """
    Acceleration error for a specific joint trajectory.
    jp, jg: (B, T, J, 3)
    fps: frames per second
    return: (B,)
    """

    dt = 1 / fps
    p = jp[:, :, joint_idx]  # (B,T,3)
    g = jg[:, :, joint_idx]

    ap = (p[:, 2:] - 2 * p[:, 1:-1] + p[:, :-2]) / dt**2   # (B,T-2,3)
    ag = (g[:, 2:] - 2 * g[:, 1:-1] + g[:, :-2]) / dt**2
    return torch.norm(ap - ag, dim=-1).mean(dim=1)  # (B,)


def accel_all_joints(jp, jg, fps):
    dt = 1.0 / fps
    ap = (jp[:, 2:] - 2*jp[:, 1:-1] + jp[:, :-2]) / (dt**2)
    ag = (jg[:, 2:] - 2*jg[:, 1:-1] + jg[:, :-2]) / (dt**2)

    err = torch.abs(
        torch.norm(ap, dim=-1) - torch.norm(ag, dim=-1)
    )  # (B, T-2, J)

    return err.mean(dim=(1,2))
# def accel_all_joints(jp, jg, fps):  # (B,T,J,3) -> (B,)
#     dt = 1.0 / fps
#     ap = (jp[:, 2:] - 2*jp[:, 1:-1] + jp[:, :-2]) / (dt**2)  # (B,T-2,J,3)
#     ag = (jg[:, 2:] - 2*jg[:, 1:-1] + jg[:, :-2]) / (dt**2)
#     return torch.norm(ap - ag, dim=-1).mean(dim=(1,2))  # mean over (T-2,J)