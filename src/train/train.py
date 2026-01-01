import sys
sys.path.append(".")

from src.dataset.dataloader import MotionDataset
from src.evaluate.utils import reconstruct_623_from_body_hand, recover_from_ric
from src.evaluate.vis import visualize_two_motions
from src.dataset.collate import collate_stack
from src.util.utils import load_config, count_params, set_seed, compute_part_losses
from src.evaluate.metric import codebook_stats
from src.evaluate.evaluator import build_eval_loader, evaluate_model
from src.train.utils import build_model_from_args
import os
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import omegaconf
from omegaconf import OmegaConf
import wandb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/motion_vqvae.yaml")
    ap.add_argument("--name", type=str, default=None)
    args_cli = ap.parse_args()

    args = OmegaConf.load(args_cli.config)
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    config_save_path = os.path.join(args.save_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        OmegaConf.save(args, config_save_path)

    # ===== Dataset / Loader =====
    # args.data_dir が「ptパス」になってる前提（必要ならyaml側で名前変えて）
    ds = MotionDataset(
        pt_path=args.data_dir,            # ★ここがpt
        feet_thre=getattr(args, "feet_thre", 0.002),
        kp_field=getattr(args, "kp_field", "kp3d"),
        clip_len=getattr(args, "clip_len", 81),          # ★80 crop
        random_crop=getattr(args, "random_crop", True),  # ★trainならTrue推奨
        pad_if_short=getattr(args, "pad_if_short", True),
        to_torch=True,
    )

    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        # shuffle=True,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_stack,   # ★ここ変更
    )

    ds_eval = MotionDataset(
        pt_path=args.data_dir_eval,
        feet_thre=getattr(args, "feet_thre", 0.002),
        kp_field=getattr(args, "kp_field", "kp3d"),
        clip_len=getattr(args, "clip_len", 81),
        random_crop=False,
        pad_if_short=getattr(args, "pad_if_short", True),
        to_torch=True,
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
    model = build_model_from_args(args, device)
    n_all, n_train = count_params(model)
    print("========== MODEL ==========")
    print(model)  # architecture (full)
    print("========== PARAMS =========")
    print(f"Total params     : {n_all:,}")
    print(f"Trainable params : {n_train:,}")
    print("===========================")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # ===== wandb =====
    wandb.init(project=args.project, name=args.name, config=vars(args))
    wandb.watch(model, log="gradients", log_freq=200)

    mean = torch.from_numpy(np.load(args.mean_path)).to(device)
    std = torch.from_numpy(np.load(args.std_path)).to(device)

    mean[0:1] = 0
    std[0:1] = 1

    global_step = 0
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        model.train()
        t0 = time.time()

        for it, batch in tqdm(enumerate(dl), desc="Training", leave=False):
            mB = batch["mB"].to(device, non_blocking=True)  # (B,T,263)
            mH = batch["mH"].to(device, non_blocking=True)  # (B,T,360)
            motion = torch.cat([mB, mH], dim=-1)
            if args.normalize:
                motion = (motion - mean) / std
            mB = motion[:, :, :263]
            mH = motion[:, :, 263:]

            # 念のため shape check（args.T と一致してないと落とす）
            if mB.shape[1] != args.T or mH.shape[1] != args.T:
                raise RuntimeError(
                    f"Time length mismatch: got {mB.shape[1]} but args.T={args.T}. "
                    f"(dataset clip_len={getattr(args,'clip_len',None)} -> expected T=clip_len-1)"
                )


            recon, losses, idx = model(mB, mH)

            target = torch.cat([mB, mH], dim=-1)
            part_losses = compute_part_losses(recon, target)
            loss = losses["loss"]


            recon_denorm = recon * std + mean
            gt_denorm = target * std + mean
            gt_623 = reconstruct_623_from_body_hand(mB, mH)
            pred_623 = reconstruct_623_from_body_hand(recon_denorm[:, :, :263], recon_denorm[:, :, 263:])
            gt_joints = recover_from_ric(gt_623, joints_num=52)
            pred_joints = recover_from_ric(pred_623, joints_num=52)
            gt_joints = gt_joints - gt_joints[..., :1, :]
            pred_joints = pred_joints - pred_joints[..., :1, :]

            if args.joints_loss:
                joints_loss_weight = args.joints_loss_weight
                joints_loss = joints_loss_weight * torch.mean((pred_joints - gt_joints) ** 2)
                loss += joints_loss

            sample = recon[0]

            if it == 0:
                for i in range(sample.shape[0]):
                    p = sample[i, :4]
                    g = mB[0, i, :4]

                    print(f"  yaw   : pred={p[0]: .5f} | gt={g[0]: .5f}")
                    print(f"  vel_x : pred={p[1]: .5f} | gt={g[1]: .5f}")
                    print(f"  vel_z : pred={p[2]: .5f} | gt={g[2]: .5f}")
                    print(f"  root_y: pred={p[3]: .5f} | gt={g[3]: .5f}")
                    print("-" * 40)
                    break


            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            usageH, pplH = codebook_stats(idx["idxH"].detach(), args.K)
            usageB, pplB = codebook_stats(idx["idxB"].detach(), args.K)

            if global_step % args.log_every == 0:
                log = {
                    "step": global_step,
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "recon_loss": float(losses["recon_loss"].detach()),
                    "commit_loss": float(losses["commit_loss"].detach()),
                    "commit_H": float(losses["commit_H"].detach()),
                    "commit_B": float(losses["commit_B"].detach()),
                    "joints_loss": float(joints_loss.detach()) if args.joints_loss else 0,
                    "code_usage_H": usageH,
                    "code_usage_B": usageB,
                    "perplexity_H": pplH,
                    "perplexity_B": pplB,
                    "lr": opt.param_groups[0]["lr"],
                }
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
                "args": vars(args),
            }
            ckpt_path = os.path.join(args.save_dir, f"ckpt_epoch{epoch:03d}.pt")
            torch.save(ckpt, ckpt_path)
            print("saved:", ckpt_path)


        # ===== eval (optional) =====
        eval_every = args.eval_every
        if eval_every > 0 and (epoch % eval_every) == 0:
            model.eval()
            metrics = evaluate_model(
                model,
                dl_eval,
                args,
                device=device,
                num_batches=args.eval_num_batches,
                save_vis_every=args.eval_save_vis_every,
                viz_dir=f"{args.eval_vis_dir}/epoch_{epoch:03d}",
                vis=True if args.eval_save_vis_every > 0 and (epoch % args.eval_save_vis_every) == 0 else False,
            )

            # print
            print(
                f"[E{epoch:03d} EVAL]\n"
                f"  feat_mse={metrics['feat_mse']:.6f}\n"
                f"  mpjpe(mm) all/body/lh/rh="
                f"{metrics['mpjpe_all_mm']:.2f}/"
                f"{metrics['mpjpe_body_mm']:.2f}/"
                f"{metrics['mpjpe_lh_mm']:.2f}/"
                f"{metrics['mpjpe_rh_mm']:.2f}\n"
                f"  pampjpe(mm) all/body/lh/rh="
                f"{metrics['pampjpe_all_mm']:.2f}/"
                f"{metrics['pampjpe_body_mm']:.2f}/"
                f"{metrics['pampjpe_lh_mm']:.2f}/"
                f"{metrics['pampjpe_rh_mm']:.2f}\n"
                f"  codebook H usage/ppl={metrics['usageH']:.3f}/{metrics['pplH']:.1f} "
                f"B usage/ppl={metrics['usageB']:.3f}/{metrics['pplB']:.1f}"
            )

            # wandb
            wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=global_step)

            model.train()

        wandb.log({"epoch_time_sec": time.time() - t0}, step=global_step)

    wandb.finish()


if __name__ == "__main__":
    main()