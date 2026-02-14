import sys
sys.path.append(".")

from src.dataset.dataloader import MotionDataset
from src.evaluate.utils import reconstruct_623_from_body_hand, recover_from_ric
from src.evaluate.vis import visualize_two_motions
from src.dataset.collate import collate_stack
from src.util.utils import count_params, set_seed, compute_part_losses
from src.evaluate.metric import codebook_stats

# ★ flow evaluator に切替
from src.evaluate.evaluator_flow import evaluate_model

# ★ flow model builder に切替（build_model_from_args が vqvae_flow を返す想定）
from src.train.utils import build_model_from_args_flow

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf
import wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/motion_vqvae_flow.yaml")  # ★ flow用に推奨
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--resume", type=str, default=None)
    args_cli = ap.parse_args()

    args = OmegaConf.load(args_cli.config)
    args.resume = args_cli.resume
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    config_save_path = os.path.join(args.save_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        OmegaConf.save(args, config_save_path)

    # ===== Dataset / Loader =====
    ds = MotionDataset(
        pt_path=args.data_dir,
        feet_thre=getattr(args, "feet_thre", 0.002),
        kp_field=getattr(args, "kp_field", "kp3d"),
        clip_len=getattr(args, "T", 81),
        random_crop=getattr(args, "random_crop", True),
        pad_if_short=getattr(args, "pad_if_short", True),
        include_fingertips=getattr(args, "include_fingertips", False),
        to_torch=True,
        base_idx=args.base_idx,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_stack,
    )

    ds_eval = MotionDataset(
        pt_path=args.data_dir_eval,
        feet_thre=getattr(args, "feet_thre", 0.002),
        kp_field=getattr(args, "kp_field", "kp3d"),
        clip_len=getattr(args, "T", 81),
        random_crop=False,
        pad_if_short=getattr(args, "pad_if_short", True),
        include_fingertips=getattr(args, "include_fingertips", False),
        to_torch=True,
        base_idx=args.base_idx,
    )
    dl_eval = DataLoader(
        ds_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_stack,
    )

    # ===== Model =====
    start_epoch = 1
    global_step = 0

    if args_cli.resume is not None:
        ckpt = torch.load(args_cli.resume, weights_only=False)
        model = build_model_from_args_flow(args, device)
        model.load_state_dict(ckpt["model"])
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        opt.load_state_dict(ckpt["opt"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
    else:
        model = build_model_from_args_flow(args, device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    n_all, n_train = count_params(model)
    print("========== MODEL ==========")
    print(model)
    print("========== PARAMS =========")
    print(f"Total params     : {n_all:,}")
    print("===========================")

    # ===== wandb =====
    wandb.init(project=args.project, name=args.name, config=OmegaConf.to_container(args, resolve=True))
    wandb.watch(model, log="gradients", log_freq=200)

    # ===== normalization =====
    use_norm = getattr(args, "normalize", False)
    if use_norm:
        mean = torch.from_numpy(np.load(args.mean_path)).to(device)
        std = torch.from_numpy(np.load(args.std_path)).to(device)
        mean[0:1] = 0
        std[0:1] = 1

    # ===== flow sampling knobs (for logging sanity metrics) =====
    log_sample_every = getattr(args, "log_sample_every", 0)  # 0なら無効
    flow_steps = getattr(args, "flow_sample_steps", 30)
    flow_solver = getattr(args, "flow_solver", "heun")

    # joints loss sampling (super expensive)
    joints_loss_on_sample = getattr(args, "joints_loss_on_sample", False)  # ★flowでは基本False推奨
    joints_loss_weight = getattr(args, "joints_loss_weight", 0.0)

    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Epochs"):
        model.train()
        t0 = time.time()

        for it, batch in tqdm(enumerate(dl), desc="Training", leave=False):
            if args.eval_check:
                break

            mB = batch["mB"].to(device, non_blocking=True)  # (B,T,263)
            mH = batch["mH"].to(device, non_blocking=True)  # (B,T,360)

            motion = torch.cat([mB, mH], dim=-1)
            if use_norm:
                motion = (motion - mean) / std
            mB = motion[:, :, :263]
            mH = motion[:, :, 263:]

            # shape check
            if mB.shape[1] != (args.T - 1) or mH.shape[1] != (args.T - 1):
                raise RuntimeError(
                    f"Time length mismatch: got {mB.shape[1]} but args.T={args.T}. "
                    f"(dataset T={args.T} -> expected T={args.T-1})"
                )

            # ===== forward (flow-only; recon is None) =====
            recon, losses, idx = model(mB, mH)
            loss = losses["loss"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            # codebook stats
            usageH, pplH = codebook_stats(idx["idxH"].detach(), args.K)
            usageB, pplB = codebook_stats(idx["idxB"].detach(), args.K)

            # ===== optional sampling for sanity metrics (expensive) =====
            part_losses = {}
            joints_loss_val = 0.0

            if log_sample_every and (global_step % log_sample_every == 0):
                with torch.no_grad():
                    pr = model.sample_from_ids(
                        idx["idxH"].detach(),
                        idx["idxB"].detach(),
                        target_T=mB.shape[1],
                        steps=flow_steps,
                        solver=flow_solver,
                    )  # pr: (B,T,623) in feature space (normalized if input normalized)

                target = torch.cat([mB, mH], dim=-1)
                part_losses = compute_part_losses(pr, target)

                # optional: joints loss computed only on sampled pr (not recommended for training objective)
                if joints_loss_on_sample and joints_loss_weight > 0:
                    if use_norm:
                        pr_dn = pr * std + mean
                        gt_dn = target * std + mean
                    else:
                        pr_dn = pr
                        gt_dn = target

                    gt_623 = reconstruct_623_from_body_hand(gt_dn[:, :, :263], gt_dn[:, :, 263:])
                    pred_623 = reconstruct_623_from_body_hand(pr_dn[:, :, :263], pr_dn[:, :, 263:])
                    gt_joints = recover_from_ric(gt_623, joints_num=52, base_idx=args.base_idx)
                    pred_joints = recover_from_ric(pred_623, joints_num=52, base_idx=args.base_idx)

                    gt_joints = gt_joints - gt_joints[..., :1, :]
                    pred_joints = pred_joints - pred_joints[..., :1, :]

                    joints_loss_val = float((joints_loss_weight * torch.mean((pred_joints - gt_joints) ** 2)).detach())

            # ===== logging =====
            if global_step % args.log_every == 0:
                log = {
                    "step": global_step,
                    "epoch": epoch,

                    "loss": float(loss.detach()),
                    "flow_loss": float(losses["flow_loss"].detach()) if "flow_loss" in losses else 0.0,
                    "commit_loss": float(losses["commit_loss"].detach()) if "commit_loss" in losses else 0.0,
                    "entropy_loss": float(losses["entropy_loss"].detach()) if "entropy_loss" in losses else 0.0,
                    "commit_H": float(losses["commit_H"].detach()) if "commit_H" in losses else 0.0,
                    "commit_B": float(losses["commit_B"].detach()) if "commit_B" in losses else 0.0,

                    "joints_loss_sample": joints_loss_val,
                    "code_usage_H": usageH,
                    "code_usage_B": usageB,
                    "perplexity_H": pplH,
                    "perplexity_B": pplB,
                    "lr": opt.param_groups[0]["lr"],
                }
                if part_losses:
                    log.update({k: float(v.detach()) for k, v in part_losses.items()})

                wandb.log(log, step=global_step)

            global_step += 1

        # ===== checkpoint =====
        if (epoch % args.ckpt_every) == 0:
            ckpt = {
                "epoch": epoch,
                "step": global_step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "args": dict(args),
            }
            ckpt_path = os.path.join(args.save_dir, f"ckpt_epoch{epoch:03d}.pt")
            torch.save(ckpt, ckpt_path)
            print("saved:", ckpt_path)

        # ===== eval =====
        eval_every = args.eval_every
        if eval_every > 0 and (epoch % eval_every) == 0:
            model.eval()
            metrics = evaluate_model(
                model,
                dl_eval,
                args,
                device=device,
                num_save_samples=args.eval_num_save_samples,
                viz_dir=f"{args.eval_vis_dir}/epoch_{epoch:03d}",
                vis=True if args.eval_save_vis_every > 0 and (epoch % args.eval_save_vis_every) == 0 else False,
            )

            print(
                f"[E{epoch:03d} EVAL]\n"
                f"  feat_mse={metrics['EVAL/RECON/feat_mse']:.6f}\n"
                f"  pampjpe(mm)  all/body/lh/rh="
                f"{metrics['EVAL/PA_MPJPE/all(mm)']:.2f}/"
                f"{metrics['EVAL/PA_MPJPE/body(mm)']:.2f}/"
                f"{metrics['EVAL/PA_MPJPE/lh(mm)']:.2f}/"
                f"{metrics['EVAL/PA_MPJPE/rh(mm)']:.2f}\n"
                f"  wa_mpjpe(mm) all/body/lh/rh="
                f"{metrics['EVAL/WA_MPJPE/all(mm)']:.2f}/"
                f"{metrics['EVAL/WA_MPJPE/body(mm)']:.2f}/"
                f"{metrics['EVAL/WA_MPJPE/lh(mm)']:.2f}/"
                f"{metrics['EVAL/WA_MPJPE/rh(mm)']:.2f}\n"
                f"  w_mpjpe(mm)  all/body/lh/rh="
                f"{metrics['EVAL/W_MPJPE/all(mm)']:.2f}/"
                f"{metrics['EVAL/W_MPJPE/body(mm)']:.2f}/"
                f"{metrics['EVAL/W_MPJPE/lh(mm)']:.2f}/"
                f"{metrics['EVAL/W_MPJPE/rh(mm)']:.2f}\n"
                f"  relative_translation_error(%) pelvis/lh/rh="
                f"{metrics['EVAL/RelativeTranslationError/pelvis(%)']:.2f}/"
                f"{metrics['EVAL/RelativeTranslationError/lh_wrist(%)']:.2f}/"
                f"{metrics['EVAL/RelativeTranslationError/rh_wrist(%)']:.2f}\n"
                f"  root_translation_error(%) pelvis/lh/rh="
                f"{metrics['EVAL/RootTranslationError/pelvis(%)']:.2f}/"
                f"{metrics['EVAL/RootTranslationError/lh_wrist(%)']:.2f}/"
                f"{metrics['EVAL/RootTranslationError/rh_wrist(%)']:.2f}\n"
                f"  accel(mm/s^2) all={metrics['EVAL/ACCEL/all(mm/s^2)']:.2f}\n"
                f"  codebook H usage/ppl={metrics['EVAL/CODEBOOK/H_usage']:.3f}/{metrics['EVAL/CODEBOOK/H_ppl']:.1f} "
                f"B usage/ppl={metrics['EVAL/CODEBOOK/B_usage']:.3f}/{metrics['EVAL/CODEBOOK/B_ppl']:.1f}"
            )

            wandb.log(metrics, step=global_step)
            model.train()

        wandb.log({"epoch_time_sec": time.time() - t0}, step=global_step)

    wandb.finish()


if __name__ == "__main__":
    main()