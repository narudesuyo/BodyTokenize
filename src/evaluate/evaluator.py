import torch
import os, sys
sys.path.append(".")
import numpy as np
from torch.utils.data import DataLoader
from src.dataset.dataloader import MotionDataset
from src.dataset.collate import collate_stack
from src.evaluate.utils import reconstruct_623_from_body_hand, recover_from_ric, batch_procrustes_align
from src.evaluate.vis import visualize_two_motions
from src.evaluate.metric import codebook_stats, mpjpe

def build_eval_loader(args, batch_size: int, shuffle: bool = True):
    ds = MotionDataset(
        pt_path=args.data_dir,
        feet_thre=getattr(args, "feet_thre", 0.002),
        kp_field=getattr(args, "kp_field", "kp3d"),
        clip_len=getattr(args, "clip_len", 81),
        random_crop=False,  # ★evalは固定が良い。毎回同じcropにしたいなら dataset側で start=0 等にする
        pad_if_short=getattr(args, "pad_if_short", True),
        to_torch=True,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_stack,
    )
    return dl


@torch.no_grad()
def evaluate_model(model, dl, args, device, num_batches=50, save_vis_every=0, viz_dir="./eval", vis=False):
    """
    model: H2VQ_CNNTransformer
    dl: returns {"mB":(B,T,263), "mH":(B,T,360), ...}
    """
    model.eval()

    BODY = slice(0, 22)
    LH   = slice(22, 37)
    RH   = slice(37, 52)

    T = args.T
    feat_mse_sum = 0.0

    frame_count = torch.zeros(T, device=device)
    mpjpe_all_t_sum  = torch.zeros(T, device=device)
    mpjpe_body_t_sum = torch.zeros(T, device=device)
    mpjpe_lh_t_sum   = torch.zeros(T, device=device)
    mpjpe_rh_t_sum   = torch.zeros(T, device=device)

    pampjpe_all_t_sum = torch.zeros(T, device=device)
    pampjpe_body_t_sum = torch.zeros(T, device=device)
    pampjpe_lh_t_sum = torch.zeros(T, device=device)
    pampjpe_rh_t_sum = torch.zeros(T, device=device)

    usageH_sum = usageB_sum = 0.0
    pplH_sum = pplB_sum = 0.0
    nb = 0


    for it, batch in enumerate(dl):
        if num_batches > 0 and it >= num_batches:
            break

        mB = batch["mB"].to(device, non_blocking=True)  # (B,T,263)
        mH = batch["mH"].to(device, non_blocking=True)  # (B,T,360)

        if args.normalize:
            motion = torch.cat([mB, mH], dim=-1)
            mean = torch.from_numpy(np.load(args.mean_path)).to(device) 
            std = torch.from_numpy(np.load(args.std_path)).to(device)
            motion = (motion - mean) / std
            mB = motion[:, :, :263]
            mH = motion[:, :, 263:]
        recon, losses, idx = model(mB, mH)

        if args.normalize:
            recon = (recon * std + mean)

        # (B,T,623)
        gt_motion = torch.cat([mB, mH], dim=-1)
        if args.normalize:
            gt_motion = gt_motion * std + mean
            mB_gt = gt_motion[:, :, :263]
            mH_gt = gt_motion[:, :, 263:]
        else:
            mB_gt = mB
            mH_gt = mH

        gt623 = reconstruct_623_from_body_hand(mB_gt, mH_gt)
        pr623 = reconstruct_623_from_body_hand(recon[:, :, :263], recon[:, :, 263:])

        feat_mse = torch.mean((pr623 - gt623) ** 2)
        feat_mse_sum += float(feat_mse.item())

        # (B,T,52,3)
        j_gt = recover_from_ric(gt623, joints_num=52)
        # j_gt = batch["kp52"][:,:80,:,:].to(j_gt.device)
        j_pr = recover_from_ric(pr623, joints_num=52)
        

        if vis:
            part = ["full", "body", "hands", "left_hand", "right_hand"]
            for p in part:
                visualize_two_motions(
                    j_gt[0], j_pr[0],
                    save_path=f"{viz_dir}/{it:03d}/{p}.mp4",
                    fps = 10,
                    view=p,
                )

        # root align
        j_gt = j_gt - j_gt[..., :1, :]
        j_pr = j_pr - j_pr[..., :1, :]



        j_pr_all_pa = batch_procrustes_align(j_pr, j_gt)
        j_pr_body_pa = batch_procrustes_align(j_pr[..., BODY, :], j_gt[..., BODY, :])
        j_pr_lh_pa = batch_procrustes_align(j_pr[..., LH, :], j_gt[..., LH, :])
        j_pr_rh_pa = batch_procrustes_align(j_pr[..., RH, :], j_gt[..., RH, :])


        def mpjpe_bt(jp, jg, slc):
            d = torch.norm(jp[..., slc, :] - jg[..., slc, :], dim=-1)  # (B,T,Jpart)
            return d.mean(dim=2)  # (B,T)

        err_all  = mpjpe_bt(j_pr, j_gt, slice(None))
        err_body = mpjpe_bt(j_pr, j_gt, BODY)
        err_lh   = mpjpe_bt(j_pr, j_gt, LH)
        err_rh   = mpjpe_bt(j_pr, j_gt, RH)

        mpjpe_all_t_sum  += err_all.sum(dim=0)
        mpjpe_body_t_sum += err_body.sum(dim=0)
        mpjpe_lh_t_sum   += err_lh.sum(dim=0)
        mpjpe_rh_t_sum   += err_rh.sum(dim=0)
        frame_count += err_all.shape[0]  # +B

        err_all_pa  = mpjpe_bt(j_pr_all_pa, j_gt, slice(None))
        err_body_pa = mpjpe_bt(j_pr_body_pa, j_gt[..., BODY, :], slice(None))
        err_lh_pa   = mpjpe_bt(j_pr_lh_pa, j_gt[..., LH, :], slice(None))
        err_rh_pa   = mpjpe_bt(j_pr_rh_pa, j_gt[..., RH, :], slice(None))

        pampjpe_all_t_sum  += err_all_pa.sum(dim=0)
        pampjpe_body_t_sum += err_body_pa.sum(dim=0)
        pampjpe_lh_t_sum   += err_lh_pa.sum(dim=0)
        pampjpe_rh_t_sum   += err_rh_pa.sum(dim=0)

        usageH, pplH = codebook_stats(idx["idxH"], args.K)
        usageB, pplB = codebook_stats(idx["idxB"], args.K)
        usageH_sum += usageH
        usageB_sum += usageB
        pplH_sum += pplH
        pplB_sum += pplB

        nb += 1

    mpjpe_all_t  = mpjpe_all_t_sum  / frame_count.clamp_min(1)
    mpjpe_body_t = mpjpe_body_t_sum / frame_count.clamp_min(1)
    mpjpe_lh_t   = mpjpe_lh_t_sum   / frame_count.clamp_min(1)
    mpjpe_rh_t   = mpjpe_rh_t_sum   / frame_count.clamp_min(1)

    pampjpe_all_t = pampjpe_all_t_sum / frame_count.clamp_min(1)
    pampjpe_body_t = pampjpe_body_t_sum / frame_count.clamp_min(1)
    pampjpe_lh_t = pampjpe_lh_t_sum / frame_count.clamp_min(1)
    pampjpe_rh_t = pampjpe_rh_t_sum / frame_count.clamp_min(1)

    metrics = {
        "feat_mse": feat_mse_sum / max(nb, 1),
        "mpjpe_all_mm":  mpjpe_all_t.mean().item()  * 1000,
        "mpjpe_body_mm": mpjpe_body_t.mean().item() * 1000,
        "mpjpe_lh_mm":   mpjpe_lh_t.mean().item()   * 1000,
        "mpjpe_rh_mm":   mpjpe_rh_t.mean().item()   * 1000,
        "pampjpe_all_mm": pampjpe_all_t.mean().item() * 1000,
        "pampjpe_body_mm": pampjpe_body_t.mean().item() * 1000,
        "pampjpe_lh_mm": pampjpe_lh_t.mean().item() * 1000,
        "pampjpe_rh_mm": pampjpe_rh_t.mean().item() * 1000,
        "usageH": usageH_sum / max(nb, 1),
        "pplH":   pplH_sum / max(nb, 1),
        "usageB": usageB_sum / max(nb, 1),
        "pplB":   pplB_sum / max(nb, 1),
    }
    return metrics