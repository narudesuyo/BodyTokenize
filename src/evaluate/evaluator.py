import torch
import sys
sys.path.append(".")
import numpy as np
from src.evaluate.utils import reconstruct_623_from_body_hand, recover_from_ric
from src.evaluate.vis import visualize_two_motions
from src.evaluate.metric import (
    codebook_stats,
    batch_procrustes_align,
    mpjpe_bt,
    wa_mpjpe,
    w_mpjpe_firstk,
    rte_joint,
    accel_all_joints,
)
from tqdm import tqdm


@torch.no_grad()
def evaluate_model(
    model,
    dl,
    args,
    device,
    num_save_samples: int = 50,
    viz_dir: str = "./eval",
    vis: bool = False,
    fps: int = 10,
):
    model.eval()

    parts = {
        "all":  slice(None),
        "body": slice(0, 22),
        "lh":   slice(22, 37),
        "rh":   slice(37, 52),
    }

    ROOT_IDX, LH_WRIST_IDX, RH_WRIST_IDX = 0, 20, 21

    # --- accumulators (sum over batches) ---
    sums = {f"{m}_{p}": 0.0 for m in ["pampjpe", "wa_mpjpe", "w_mpjpe"] for p in parts.keys()}
    sums.update({
        "feat_mse": 0.0,
        "rte_root": 0.0,
        "rte_lh_wrist": 0.0,
        "rte_rh_wrist": 0.0,
        "accel": 0.0,
    })
    cb_stats = {"usageH": 0.0, "pplH": 0.0, "usageB": 0.0, "pplB": 0.0}

    nb = 0

    if getattr(args, "normalize", False):
        mean = torch.from_numpy(np.load(args.mean_path)).to(device)
        std  = torch.from_numpy(np.load(args.std_path)).to(device)

    for it, batch in tqdm(enumerate(dl), total=len(dl), leave=False):
        mB = batch["mB"].to(device, non_blocking=True)
        mH = batch["mH"].to(device, non_blocking=True)

        if getattr(args, "normalize", False):
            motion = torch.cat([mB, mH], dim=-1)
            motion = (motion-mean) / std
            mB = motion[..., :263]
            mH = motion[..., 263:]

        recon, losses, idx = model(mB, mH)   # recon: (B,T,623)

        # --- denormalize if needed ---
        gt623 = torch.cat([mB, mH], dim=-1)  # (B,T,623)
        pr623 = recon

        if getattr(args, "normalize", False):
            gt623 = gt623 * std + mean
            pr623 = pr623 * std + mean

        sums["feat_mse"] += torch.mean((pr623 - gt623) ** 2).item()

        # --- joints (B,T,52,3) ---
        gt_rec = reconstruct_623_from_body_hand(gt623[..., :263], gt623[..., 263:], include_fingertips=args.include_fingertips)
        pr_rec = reconstruct_623_from_body_hand(pr623[..., :263], pr623[..., 263:], include_fingertips=args.include_fingertips)

        joints_num = 52 if not args.include_fingertips else 62

        j_gt = recover_from_ric(gt_rec, joints_num=joints_num, use_root_loss=getattr(args, "use_root_loss", True))
        j_pr = recover_from_ric(pr_rec, joints_num=joints_num, use_root_loss=getattr(args, "use_root_loss", True))

        # --- visualize ---
        if vis and it < num_save_samples:
            view_names = ["all", "body", "hands", "lh", "rh"]
            for vname in view_names:
                visualize_two_motions(
                    j_gt[0], j_pr[0],
                    save_path=f"{viz_dir}/{it:03d}/{vname}.mp4",
                    fps=fps,
                    view=vname,
                    include_fingertips=args.include_fingertips,
                    only_gt=args.eval_check,
                )

        # --- pose metrics (PA / WA / W-firstK) ---
        for name, slc in parts.items():
            jp_part = j_pr[..., slc, :]
            jg_part = j_gt[..., slc, :]

            # PA-MPJPE: per-frame Procrustes
            jp_pa = batch_procrustes_align(jp_part, jg_part)
            sums[f"pampjpe_{name}"] += mpjpe_bt(jp_pa, jg_part, slice(None)).mean().item()

            # WA: sequence-level Procrustes
            sums[f"wa_mpjpe_{name}"] += wa_mpjpe(jp_part, jg_part, slice(None)).mean().item()

            # W: first-K Procrustes (default K=1)
            sums[f"w_mpjpe_{name}"] += w_mpjpe_firstk(
                jp_part, jg_part, slice(None),
                num_align_frames=1
            ).mean().item()

        # --- trajectory metrics ---
        sums["rte_root"]      += rte_joint(j_pr, j_gt, ROOT_IDX).mean().item()
        sums["rte_lh_wrist"]  += rte_joint(j_pr, j_gt, LH_WRIST_IDX).mean().item()
        sums["rte_rh_wrist"]  += rte_joint(j_pr, j_gt, RH_WRIST_IDX).mean().item()
        sums["accel"]         += accel_all_joints(j_pr, j_gt, fps=fps).mean().item()

        # --- codebook stats ---
        uH, pH = codebook_stats(idx["idxH"], args.K)
        uB, pB = codebook_stats(idx["idxB"], args.K)
        cb_stats["usageH"] += uH; cb_stats["pplH"] += pH
        cb_stats["usageB"] += uB; cb_stats["pplB"] += pB

        nb += 1

    nb = max(nb, 1)

    # --- build hierarchical metrics keys for wandb ---
    metrics = {}

    # pose (mm)
    for p in parts.keys():
        metrics[f"EVAL/PA_MPJPE/{p}"]    = (sums[f"pampjpe_{p}"] / nb) * 1000.0
        metrics[f"EVAL/WA_MPJPE/{p}"]  = (sums[f"wa_mpjpe_{p}"] / nb) * 1000.0
        metrics[f"EVAL/W_MPJPE/{p}"]   = (sums[f"w_mpjpe_{p}"] / nb) * 1000.0

    # recon / traj
    metrics["EVAL/RECON/feat_mse"]       = sums["feat_mse"] / nb
    metrics["EVAL/RTE/root"]             = sums["rte_root"] * 1000.0 / nb
    metrics["EVAL/RTE/lh_wrist"]         = sums["rte_lh_wrist"] * 1000.0 / nb
    metrics["EVAL/RTE/rh_wrist"]         = sums["rte_rh_wrist"] * 1000.0 / nb
    metrics["EVAL/ACCEL/all"]            = sums["accel"] * 1000.0 / nb

    # codebook
    metrics["EVAL/CODEBOOK/H_usage"]     = cb_stats["usageH"] / nb
    metrics["EVAL/CODEBOOK/H_ppl"]       = cb_stats["pplH"] / nb
    metrics["EVAL/CODEBOOK/B_usage"]     = cb_stats["usageB"] / nb
    metrics["EVAL/CODEBOOK/B_ppl"]       = cb_stats["pplB"] / nb

    return metrics