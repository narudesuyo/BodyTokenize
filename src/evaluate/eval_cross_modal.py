"""
Evaluate cross-modal prediction: hand-only → full body, body-only → full body.

Loads a dual-decoder checkpoint, zeros out one modality's tokens after
quantization, and measures how well the other modality can reconstruct
the full motion.

Usage:
  python src/evaluate/eval_cross_modal.py \
      --ckpt runs/.../ckpt_best.pt \
      --config config/motion_vqvae.yaml \
      --modes hand2body body2hand normal \
      --vis --num_vis 5
"""

import sys
sys.path.append(".")

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from src.dataset.dataloader import MotionDataset
from src.dataset.collate import collate_stack
from src.train.utils import build_model_from_args
from src.evaluate.utils import reconstruct_623_from_body_hand, recover_from_ric
from src.evaluate.vis import visualize_two_motions
from src.evaluate.metric import (
    codebook_stats,
    batch_procrustes_align,
    mpjpe_bt,
    wa_mpjpe,
    w_mpjpe_firstk,
    accel_all_joints,
    relative_translation_error,
    root_translation_error,
)
from src.util.utils import set_seed


@torch.no_grad()
def evaluate_cross_modal(
    model,
    dl,
    args,
    device,
    mode="hand2body",
    num_save_samples=5,
    viz_dir="./eval_cross",
    vis=False,
    fps=10,
):
    """
    mode:
      "hand2body" - zero body tokens, predict from hand only
      "body2hand" - zero hand tokens, predict from body only
      "normal"    - no masking (baseline)
    """
    model.eval()
    assert model.dec_mode == "dual", "Cross-modal eval requires dec_mode='dual'"

    parts = {
        "all":  slice(None),
        "body": slice(0, 22),
        "lh":   slice(22, 37),
        "rh":   slice(37, 52),
    }
    ROOT_IDX, LH_WRIST_IDX, RH_WRIST_IDX = 0, 20, 21

    sums = {f"{m}_{p}": 0.0 for m in ["pampjpe", "wa_mpjpe", "w_mpjpe"] for p in parts}
    sums.update({
        "feat_mse": 0.0,
        "relative_translation_error_pelvis": 0.0,
        "relative_translation_error_lh_wrist": 0.0,
        "relative_translation_error_rh_wrist": 0.0,
        "root_translation_error_pelvis": 0.0,
        "root_translation_error_lh_wrist": 0.0,
        "root_translation_error_rh_wrist": 0.0,
        "accel": 0.0,
    })
    cb_stats = {"usageH": 0.0, "pplH": 0.0, "usageB": 0.0, "pplB": 0.0}
    nb = 0

    if getattr(args, "normalize", False):
        mean = torch.from_numpy(np.load(args.mean_path)).to(device)
        std = torch.from_numpy(np.load(args.std_path)).to(device)

    for it, batch in tqdm(enumerate(dl), total=len(dl), leave=False, desc=mode):
        mB = batch["mB"].to(device, non_blocking=True)
        mH = batch["mH"].to(device, non_blocking=True)

        if getattr(args, "normalize", False):
            motion = torch.cat([mB, mH], dim=-1)
            motion = (motion - mean) / std
            mB = motion[..., :263]
            mH = motion[..., 263:]

        # --- encode & quantize (reuse model internals) ---
        zH = model.encH(mH)
        zB = model.encB(mB)
        Tm = min(zH.size(1), zB.size(1))
        zH, zB = zH[:, :Tm], zB[:, :Tm]

        zH_tok = model._split_tokens(zH, model.hand_tokens_per_t)
        zH_q_tok, idxH = model.qH(zH_tok)
        zH_q = model._merge_tokens(zH_q_tok)

        zH_proj = model.hand_proj(zH_q)
        z_fused = model.fuse_proj(torch.cat([zH_proj, zB], dim=-1))
        zB_tok = model._split_tokens(z_fused, model.body_tokens_per_t)
        zB_q_tok, idxB = model.qB(zB_tok)
        zB_q = model._merge_tokens(zB_q_tok)

        # --- apply cross-modal masking ---
        if mode == "hand2body":
            zB_q = torch.zeros_like(zB_q)
        elif mode == "body2hand":
            zH_q = torch.zeros_like(zH_q)
        # else: normal, no masking

        # --- decode ---
        recon = model._decode(zB_q, zH_q)

        # --- metrics ---
        gt623 = torch.cat([mB, mH], dim=-1)[:, :recon.shape[1]]
        pr623 = recon

        if getattr(args, "normalize", False):
            gt623 = gt623 * std + mean
            pr623 = pr623 * std + mean

        sums["feat_mse"] += torch.mean((pr623 - gt623) ** 2).item()

        gt_rec = reconstruct_623_from_body_hand(gt623[..., :263], gt623[..., 263:], include_fingertips=args.include_fingertips)
        pr_rec = reconstruct_623_from_body_hand(pr623[..., :263], pr623[..., 263:], include_fingertips=args.include_fingertips)

        joints_num = 62 if args.include_fingertips else 52
        j_gt = recover_from_ric(gt_rec, joints_num=joints_num, use_root_loss=getattr(args, "use_root_loss", True),
                                base_idx=args.base_idx, hand_local=getattr(args, "hand_local", False))
        j_pr = recover_from_ric(pr_rec, joints_num=joints_num, use_root_loss=getattr(args, "use_root_loss", True),
                                base_idx=args.base_idx, hand_local=getattr(args, "hand_local", False))

        if vis and it < num_save_samples:
            for vname in ["all", "body", "hands", "lh", "rh"]:
                visualize_two_motions(
                    j_gt[0], j_pr[0],
                    save_path=f"{viz_dir}/{mode}/{it:03d}/{vname}.mp4",
                    fps=fps, view=vname, rotate=False,
                    include_fingertips=args.include_fingertips,
                    origin_align=True, base_idx=args.base_idx,
                )

        for name, slc in parts.items():
            jp_part = j_pr[..., slc, :]
            jg_part = j_gt[..., slc, :]
            jp_pa = batch_procrustes_align(jp_part, jg_part)
            sums[f"pampjpe_{name}"] += mpjpe_bt(jp_pa, jg_part, slice(None)).mean().item()
            sums[f"wa_mpjpe_{name}"] += wa_mpjpe(jp_part, jg_part, slice(None)).mean().item()
            sums[f"w_mpjpe_{name}"] += w_mpjpe_firstk(jp_part, jg_part, slice(None), num_align_frames=1).mean().item()

        sums["relative_translation_error_pelvis"] += relative_translation_error(j_pr, j_gt, ROOT_IDX).mean().item()
        sums["relative_translation_error_lh_wrist"] += relative_translation_error(j_pr, j_gt, LH_WRIST_IDX).mean().item()
        sums["relative_translation_error_rh_wrist"] += relative_translation_error(j_pr, j_gt, RH_WRIST_IDX).mean().item()
        sums["root_translation_error_pelvis"] += root_translation_error(j_pr, j_gt, ROOT_IDX).mean().item()
        sums["root_translation_error_lh_wrist"] += root_translation_error(j_pr, j_gt, LH_WRIST_IDX).mean().item()
        sums["root_translation_error_rh_wrist"] += root_translation_error(j_pr, j_gt, RH_WRIST_IDX).mean().item()
        sums["accel"] += accel_all_joints(j_pr, j_gt, fps=fps).mean().item()

        uH, pH = codebook_stats(idxH, args.K)
        uB, pB = codebook_stats(idxB, args.K)
        cb_stats["usageH"] += uH; cb_stats["pplH"] += pH
        cb_stats["usageB"] += uB; cb_stats["pplB"] += pB
        nb += 1

    nb = max(nb, 1)
    metrics = {}
    for p in parts:
        metrics[f"PA_MPJPE/{p}(mm)"] = (sums[f"pampjpe_{p}"] / nb) * 1000.0
        metrics[f"WA_MPJPE/{p}(mm)"] = (sums[f"wa_mpjpe_{p}"] / nb) * 1000.0
        metrics[f"W_MPJPE/{p}(mm)"] = (sums[f"w_mpjpe_{p}"] / nb) * 1000.0
    metrics["RECON/feat_mse"] = sums["feat_mse"] / nb
    metrics["ACCEL/all(mm/s^2)"] = sums["accel"] * 1000.0 / nb
    metrics["CB/H_usage"] = cb_stats["usageH"] / nb
    metrics["CB/B_usage"] = cb_stats["usageB"] / nb
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default="config/motion_vqvae.yaml")
    ap.add_argument("--modes", type=str, nargs="+", default=["hand2body", "body2hand", "normal"])
    ap.add_argument("--vis", action="store_true")
    ap.add_argument("--num_vis", type=int, default=5)
    args_cli = ap.parse_args()

    args = OmegaConf.load(args_cli.config)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Dataset
    _use_cache = getattr(args, "use_cache", False)
    ds_eval = MotionDataset(
        pt_path=args.cache_pt_eval if _use_cache else args.data_dir_eval,
        feet_thre=getattr(args, "feet_thre", 0.002),
        kp_field=getattr(args, "kp_field", "kp3d"),
        clip_len=getattr(args, "T", 81),
        random_crop=False,
        pad_if_short=getattr(args, "pad_if_short", True),
        include_fingertips=getattr(args, "include_fingertips", False),
        to_torch=True,
        base_idx=args.base_idx,
        hand_local=getattr(args, "hand_local", False),
        use_cache=_use_cache,
    )
    dl_eval = DataLoader(
        ds_eval,
        batch_size=getattr(args, "eval_batch_size", args.batch_size),
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_stack,
    )

    # Model
    ckpt = torch.load(args_cli.ckpt, weights_only=False, map_location=device)
    model = build_model_from_args(args, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    print(f"Loaded: {args_cli.ckpt}  (epoch={epoch})")

    viz_base = f"{args.save_dir}/eval_cross"

    # Key metrics to display
    key_metrics = [
        "PA_MPJPE/all(mm)", "PA_MPJPE/body(mm)", "PA_MPJPE/lh(mm)", "PA_MPJPE/rh(mm)",
        "WA_MPJPE/all(mm)", "WA_MPJPE/body(mm)", "WA_MPJPE/lh(mm)", "WA_MPJPE/rh(mm)",
        "RECON/feat_mse", "ACCEL/all(mm/s^2)",
    ]

    results = {}
    for mode in args_cli.modes:
        print(f"\n{'='*60}")
        print(f"  Mode: {mode}")
        print(f"{'='*60}")
        metrics = evaluate_cross_modal(
            model, dl_eval, args, device,
            mode=mode,
            num_save_samples=args_cli.num_vis,
            viz_dir=viz_base,
            vis=args_cli.vis,
        )
        results[mode] = metrics
        for k in key_metrics:
            if k in metrics:
                print(f"  {k}: {metrics[k]:.4f}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"  Summary")
    print(f"{'='*80}")
    header = f"{'mode':>12}"
    for k in key_metrics:
        short = k.split("/")[-1]
        header += f"  {short:>12}"
    print(header)
    print("-" * len(header))
    for mode in args_cli.modes:
        row = f"{mode:>12}"
        for k in key_metrics:
            val = results[mode].get(k, float("nan"))
            row += f"  {val:>12.2f}"
        print(row)


if __name__ == "__main__":
    main()
