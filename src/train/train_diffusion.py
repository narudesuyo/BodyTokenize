"""
Training script for standalone unconditional motion diffusion model.
Reuses MotionDataset with precomputed cache.
"""
import sys
sys.path.append(".")

import os
import glob
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from omegaconf import OmegaConf
import wandb

from src.dataset.dataloader import MotionDataset
from src.dataset.collate import collate_stack
from src.model.motion_diffusion import MotionDiffusion
from src.util.utils import count_params, set_seed
from src.evaluate.utils import recover_joints_from_body_hand
from src.evaluate.vis import visualize_two_motions
from src.evaluate.evaluator_diffusion import evaluate_diffusion


def build_model(args, device):
    model = MotionDiffusion(
        x_dim=args.x_dim,
        body_dim=getattr(args, "body_dim", 263),
        model_dim=args.model_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_ratio=getattr(args, "mlp_ratio", 4.0),
        drop=getattr(args, "drop", 0.0),
        attn_drop=getattr(args, "attn_drop", 0.0),
        t_dim=getattr(args, "t_dim", args.model_dim),
        max_T=getattr(args, "max_T", 256),
        diffusion_timesteps=args.diffusion_timesteps,
        alpha_root=getattr(args, "alpha_root", 5.0),
        alpha_body=getattr(args, "alpha_body", 1.0),
        alpha_hand=getattr(args, "alpha_hand", 5.0),
        joints_loss=getattr(args, "joints_loss", False),
        alpha_joints=getattr(args, "alpha_joints", 5.0),
        alpha_joints_hand=getattr(args, "alpha_joints_hand", 5.0),
        alpha_bone_length=getattr(args, "alpha_bone_length", 0.0),
        include_fingertips=getattr(args, "include_fingertips", True),
        hand_root=getattr(args, "hand_root", False),
        hand_root_dim=getattr(args, "hand_root_dim", 9),
        base_idx=getattr(args, "base_idx", 15),
        hand_local=getattr(args, "hand_local", False),
        use_root_loss=getattr(args, "use_root_loss", False),
        prediction_type=getattr(args, "prediction_type", "x0"),
        velocity_loss=getattr(args, "velocity_loss", False),
        alpha_velocity=getattr(args, "alpha_velocity", 1.0),
        foot_contact_loss=getattr(args, "foot_contact_loss", False),
        alpha_foot_contact=getattr(args, "alpha_foot_contact", 1.0),
    ).to(device)
    return model


@torch.no_grad()
def _compute_target_bone_lengths(ds, args, device):
    """Compute fixed GT bone lengths from one dataset sample."""
    from src.evaluate.utils import get_bone_pairs, compute_bone_lengths

    sample = ds[0]
    body = sample["body"].unsqueeze(0).to(device)
    hand = sample["hand"].unsqueeze(0).to(device)

    body_dim = getattr(args, "body_dim", 263)
    hand_root_dim_total = getattr(args, "hand_root_dim", 9) * 2 if getattr(args, "hand_root", False) else 0
    include_fingertips = getattr(args, "include_fingertips", True)
    joints_num = 62 if include_fingertips else 52

    joints = recover_joints_from_body_hand(
        body, hand,
        include_fingertips=include_fingertips,
        hand_root_dim=hand_root_dim_total,
        joints_num=joints_num,
        use_root_loss=getattr(args, "use_root_loss", False),
        base_idx=args.base_idx,
        hand_local=getattr(args, "hand_local", False),
        hand_only=getattr(args, "hand_only", False),
    )

    if include_fingertips:
        from paramUtil_add_tips import t2m_body_hand_kinematic_chain_with_tips as kc
    else:
        from paramUtil import t2m_body_hand_kinematic_chain as kc
    bone_pairs = get_bone_pairs(kc)
    bl = compute_bone_lengths(joints, bone_pairs)  # (1, T, num_bones)
    return bl.mean(dim=(0, 1))  # (num_bones,)


@torch.no_grad()
def sample_and_visualize(model, args, device, mean, std, epoch, global_step, num_samples=4):
    """Generate samples and save visualizations."""
    model.eval()
    T = args.T - 1  # output frames
    samples = model.sample_ddim(
        B=num_samples, T=T,
        num_steps=getattr(args, "sample_steps", 50),
        device=device,
    )  # (B, T, x_dim)

    # Denormalize
    if args.normalize:
        samples = samples * std + mean

    body_dim = getattr(args, "body_dim", 263)
    hand_root_dim = getattr(args, "hand_root_dim", 9) * 2 if getattr(args, "hand_root", False) else 0
    include_fingertips = getattr(args, "include_fingertips", True)
    joints_num = 62 if include_fingertips else 52

    mb = samples[:, :, :body_dim]
    mh = samples[:, :, body_dim:]

    joints = recover_joints_from_body_hand(
        mb, mh,
        include_fingertips=include_fingertips,
        hand_root_dim=hand_root_dim,
        joints_num=joints_num,
        use_root_loss=getattr(args, "use_root_loss", False),
        base_idx=args.base_idx,
        hand_local=getattr(args, "hand_local", False),
        hand_only=getattr(args, "hand_only", False),
    )  # (B, T, J, 3)

    vis_dir = os.path.join(args.save_dir, "samples", f"epoch_{epoch:04d}")
    os.makedirs(vis_dir, exist_ok=True)

    vis_videos = {}
    for i in range(min(num_samples, joints.shape[0])):
        save_path = os.path.join(vis_dir, f"sample_{i:02d}.mp4")
        j = joints[i].cpu()
        visualize_two_motions(
            j, j,
            save_path=save_path,
            fps=10,
            title=f"Generated (epoch {epoch})",
            include_fingertips=include_fingertips,
            only_gt=True,
            base_idx=args.base_idx,
        )
        try:
            vis_videos[f"VIS/sample/{i}"] = wandb.Video(save_path, format="mp4")
        except Exception:
            pass

    return vis_videos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--resume", type=str, default=None)
    args_cli = ap.parse_args()

    args = OmegaConf.load(args_cli.config)
    args.resume = args_cli.resume
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    config_save_path = os.path.join(args.save_dir, "config.yaml")
    OmegaConf.save(args, config_save_path)

    # ===== Dataset =====
    _use_cache = getattr(args, "use_cache", True)
    _hand_root = getattr(args, "hand_root", False)
    _debug_max = getattr(args, "debug_max_samples", 0)
    _use_root_loss = getattr(args, "use_root_loss", False)

    ds = MotionDataset(
        pt_path=args.cache_pt,
        clip_len=args.T,
        random_crop=True,
        pad_if_short=True,
        include_fingertips=getattr(args, "include_fingertips", True),
        to_torch=True,
        base_idx=args.base_idx,
        hand_local=getattr(args, "hand_local", False),
        use_cache=_use_cache,
        hand_root=_hand_root,
    )
    if _debug_max > 0:
        ds.keys = ds.keys[:_debug_max]
        ds.db = {k: ds.db[k] for k in ds.keys}
        print(f"[DEBUG] train dataset truncated to {len(ds.keys)} samples")

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
        pt_path=args.cache_pt_eval,
        clip_len=args.T,
        random_crop=False,
        pad_if_short=True,
        include_fingertips=getattr(args, "include_fingertips", True),
        to_torch=True,
        base_idx=args.base_idx,
        hand_local=getattr(args, "hand_local", False),
        use_cache=_use_cache,
        hand_root=_hand_root,
    )
    if _debug_max > 0:
        ds_eval.keys = ds_eval.keys[:_debug_max]
        ds_eval.db = {k: ds_eval.db[k] for k in ds_eval.keys}
        print(f"[DEBUG] eval dataset truncated to {len(ds_eval.keys)} samples")

    dl_eval = DataLoader(
        ds_eval,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_stack,
    )

    # Extra vis DataLoader: sources missing from val cache (e.g. hot3d from train)
    from src.evaluate.evaluator import _parse_source_list, _source_from_key
    _extra_vis_sources = set()
    _val_sources = set(k.split("::")[0] for k in ds_eval.keys if "::" in k)
    _vis_sources_cfg = _parse_source_list(getattr(args, "eval_vis_sources", None))
    if _vis_sources_cfg:
        for s in _vis_sources_cfg:
            if s not in _val_sources:
                _extra_vis_sources.add(s)
    dl_extra_vis = None
    if _extra_vis_sources and _use_cache:
        import copy
        _ds_extra = copy.copy(ds)
        _extra_keys = [k for k in ds.keys if "::" in k and k.split("::")[0] in _extra_vis_sources]
        if _extra_keys:
            _ds_extra.keys = _extra_keys
            _ds_extra.db = {k: ds.db[k] for k in _extra_keys}
            dl_extra_vis = DataLoader(
                _ds_extra, batch_size=min(32, len(_extra_keys)),
                shuffle=True, num_workers=0, pin_memory=True,
                drop_last=False, collate_fn=collate_stack,
            )
            print(f"[ExtraVis] {len(_extra_keys)} samples from {_extra_vis_sources} (train set)")

    # ===== Normalization =====
    from src.train.utils import compute_norm_stats
    body_dim = getattr(args, "body_dim", 263)
    _stats_cache_dir = os.path.join(args.save_dir, "norm_stats")
    mean_full, std_full = compute_norm_stats(
        args.cache_pt, body_dim=body_dim, root_dim=1,
        stats_cache_dir=_stats_cache_dir,
    )
    mean_full = mean_full[:args.x_dim].to(device)
    std_full = std_full[:args.x_dim].to(device)

    # ===== Model =====
    start_epoch = 1
    global_step = 0
    best_loss = float("inf")
    warmup_epochs = getattr(args, "warmup_epochs", 50)
    min_lr = getattr(args, "min_lr", 1e-6)

    model = build_model(args, device)

    # Pass norm stats for geometric losses (joints, velocity, foot contact)
    _need_geo = (getattr(args, "joints_loss", False) or getattr(args, "velocity_loss", False)
                 or getattr(args, "foot_contact_loss", False) or getattr(args, "alpha_bone_length", 0.0) > 0)
    if _need_geo:
        model.set_norm_stats(mean_full.clone(), std_full.clone())

    if args_cli.resume is not None:
        ckpt = torch.load(args_cli.resume, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        # best_loss is now WA_MPJPE(mm) based; reset if resuming from old val_mse checkpoint
        _old_best = ckpt.get("best_loss", float("inf"))
        best_loss = _old_best if _old_best > 1.0 else float("inf")

    # Precompute fixed GT bone lengths from one sample (after resume so it always overwrites)
    if getattr(args, "alpha_bone_length", 0.0) > 0:
        _bl = _compute_target_bone_lengths(ds, args, device)
        model.set_target_bone_lengths(_bl)
        print(f"[BoneLength] Registered {_bl.shape[0]} target bone lengths")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=getattr(args, "wd", 1e-6))
    if args_cli.resume is not None and "opt" in ckpt:
        opt.load_state_dict(ckpt["opt"])

    warmup_sched = LambdaLR(opt, lr_lambda=lambda ep: min(1.0, (ep + 1) / warmup_epochs))
    cosine_sched = CosineAnnealingLR(opt, T_max=args.epochs - warmup_epochs, eta_min=min_lr)
    scheduler = SequentialLR(opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

    if args_cli.resume is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        for _ in range(1, start_epoch):
            scheduler.step()

    n_all, n_train = count_params(model)
    print("========== DIFFUSION MODEL ==========")
    print(model)
    print("========== PARAMS =========")
    print(f"Total params     : {n_all:,}")
    print(f"Trainable params : {n_train:,}")
    print(f"x_dim            : {args.x_dim}")
    print(f"Diffusion steps  : {args.diffusion_timesteps}")
    print(f"prediction_type  : {getattr(args, 'prediction_type', 'x0')}")
    print("===========================")

    # ===== wandb =====
    wandb.init(project=args.project, name=args.name, config=OmegaConf.to_container(args, resolve=True))
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    # ===== Training Loop =====
    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Epochs"):
        model.train()
        t0 = time.time()
        epoch_loss = 0
        epoch_steps = 0
        _epoch_comp = {}

        for it, batch in tqdm(enumerate(dl), total=len(dl), desc="Training", leave=False):
            mB = batch["mB"].to(device, non_blocking=True)
            mH = batch["mH"].to(device, non_blocking=True)
            motion = torch.cat([mB, mH], dim=-1)

            if args.normalize:
                motion = (motion - mean_full) / std_full

            # Zero out unsupervised root dims (yaw, vx, vz) when not using root loss
            if not _use_root_loss:
                motion[..., :3] = 0.0

            losses = model(motion)
            loss = losses["loss"]

            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_clip = getattr(args, "grad_clip", 1.0)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            epoch_loss += loss.item()
            epoch_steps += 1

            # Accumulate per-component losses for epoch averaging
            for k in ("mse", "root_loss", "body_loss", "hand_loss",
                       "joints_loss", "joints_loss_hand", "bone_length_loss",
                       "velocity_loss", "foot_contact_loss"):
                if k in losses:
                    v = losses[k]
                    _epoch_comp[k] = _epoch_comp.get(k, 0.0) + (float(v.detach()) if torch.is_tensor(v) else float(v))

            global_step += 1

        scheduler.step()
        avg_loss = epoch_loss / max(epoch_steps, 1)
        epoch_log = {
            "epoch": epoch,
            "epoch_avg_loss": avg_loss,
            "lr": opt.param_groups[0]["lr"],
            "epoch_time_sec": time.time() - t0,
        }
        for k, v in _epoch_comp.items():
            epoch_log[f"epoch_avg_{k}"] = v / max(epoch_steps, 1)
        wandb.log(epoch_log)
        print(f"[E{epoch:04d}] avg_loss={avg_loss:.6f} lr={opt.param_groups[0]['lr']:.2e}")

        # ===== Eval: compute val loss =====
        eval_every = getattr(args, "eval_every", 10)
        if eval_every > 0 and (epoch % eval_every == 0 or epoch == start_epoch):
            model.eval()
            val_loss_sum = 0
            val_steps = 0
            with torch.no_grad():
                for batch in dl_eval:
                    mB = batch["mB"].to(device, non_blocking=True)
                    mH = batch["mH"].to(device, non_blocking=True)
                    motion = torch.cat([mB, mH], dim=-1)
                    if args.normalize:
                        motion = (motion - mean_full) / std_full
                    if not _use_root_loss:
                        motion[..., :3] = 0.0
                    losses = model(motion)
                    val_loss_sum += losses["mse"].item()
                    val_steps += 1

            val_mse = val_loss_sum / max(val_steps, 1)

            log_dict = {"val_mse": val_mse}

            # MPJPE evaluation (noise→denoise→compare with GT)
            _eval_noise_t = getattr(args, "eval_noise_t", 200)
            _eval_ddim_steps = getattr(args, "eval_ddim_steps", 50)
            _eval_num_save = getattr(args, "eval_num_save_samples", 1)
            _eval_vis_dir = getattr(args, "eval_vis_dir", os.path.join(args.save_dir, "eval"))
            _eval_save_vis_every = getattr(args, "eval_save_vis_every", 50)
            _eval_batch_size = getattr(args, "eval_batch_size", args.batch_size)

            # Build eval dataloader with eval_batch_size if different
            if _eval_batch_size != args.batch_size:
                dl_eval_mpjpe = DataLoader(
                    ds_eval,
                    batch_size=_eval_batch_size,
                    shuffle=True,
                    num_workers=args.num_workers,
                    pin_memory=True,
                    drop_last=True,
                    collate_fn=collate_stack,
                )
            else:
                dl_eval_mpjpe = dl_eval

            _do_vis = (_eval_save_vis_every > 0 and (epoch % _eval_save_vis_every == 0 or epoch == start_epoch))
            _vis_dir_epoch = os.path.join(_eval_vis_dir, f"epoch_{epoch:04d}") if _do_vis else None

            _eval_max_batches = getattr(args, "eval_max_batches", 10)

            _vis_num_per_source = getattr(args, "eval_vis_num_per_source", _eval_num_save)

            eval_metrics, eval_vis_videos = evaluate_diffusion(
                model, dl_eval_mpjpe, args, device,
                mean=mean_full, std=std_full,
                noise_t=_eval_noise_t,
                ddim_steps=_eval_ddim_steps,
                num_save_samples=_vis_num_per_source,
                viz_dir=_vis_dir_epoch or _eval_vis_dir,
                vis=_do_vis,
                fps=10,
                max_batches=_eval_max_batches,
                extra_vis_dl=dl_extra_vis if _do_vis else None,
            )
            log_dict.update(eval_metrics)
            for vk, vpath in eval_vis_videos.items():
                try:
                    log_dict[vk] = wandb.Video(vpath, format="mp4")
                except Exception:
                    pass

            wa_mpjpe_all = eval_metrics.get("EVAL/WA_MPJPE/all(mm)", float("inf"))
            _m = eval_metrics
            print(
                f"[E{epoch:04d} EVAL]\n"
                f"  feat_mse={_m.get('EVAL/RECON/feat_mse', 0):.6f}\n"
                f"  pampjpe(mm)  all/body/lh/rh="
                f"{_m.get('EVAL/PA_MPJPE/all(mm)', 0):.2f}/"
                f"{_m.get('EVAL/PA_MPJPE/body(mm)', 0):.2f}/"
                f"{_m.get('EVAL/PA_MPJPE/lh(mm)', 0):.2f}/"
                f"{_m.get('EVAL/PA_MPJPE/rh(mm)', 0):.2f}\n"
                f"  wa_mpjpe(mm) all/body/lh/rh="
                f"{_m.get('EVAL/WA_MPJPE/all(mm)', 0):.2f}/"
                f"{_m.get('EVAL/WA_MPJPE/body(mm)', 0):.2f}/"
                f"{_m.get('EVAL/WA_MPJPE/lh(mm)', 0):.2f}/"
                f"{_m.get('EVAL/WA_MPJPE/rh(mm)', 0):.2f}\n"
                f"  w_mpjpe(mm)  all/body/lh/rh="
                f"{_m.get('EVAL/W_MPJPE/all(mm)', 0):.2f}/"
                f"{_m.get('EVAL/W_MPJPE/body(mm)', 0):.2f}/"
                f"{_m.get('EVAL/W_MPJPE/lh(mm)', 0):.2f}/"
                f"{_m.get('EVAL/W_MPJPE/rh(mm)', 0):.2f}\n"
                f"  relative_translation_error(mm) pelvis/lh/rh="
                f"{_m.get('EVAL/RelativeTranslationError/pelvis(%)', 0):.2f}/"
                f"{_m.get('EVAL/RelativeTranslationError/lh_wrist(%)', 0):.2f}/"
                f"{_m.get('EVAL/RelativeTranslationError/rh_wrist(%)', 0):.2f}\n"
                f"  root_translation_error(%) pelvis/lh/rh="
                f"{_m.get('EVAL/RootTranslationError/pelvis(%)', 0):.2f}/"
                f"{_m.get('EVAL/RootTranslationError/lh_wrist(%)', 0):.2f}/"
                f"{_m.get('EVAL/RootTranslationError/rh_wrist(%)', 0):.2f}\n"
                f"  accel(mm/s^2) all={_m.get('EVAL/ACCEL/all(mm/s^2)', 0):.2f}"
            )

            # Sampling & visualization (unconditional generation)
            sample_every = getattr(args, "sample_vis_every", 50)
            if sample_every > 0 and (epoch % sample_every == 0):
                vis_videos = sample_and_visualize(
                    model, args, device, mean_full, std_full,
                    epoch, global_step,
                    num_samples=getattr(args, "num_vis_samples", 4),
                )
                log_dict.update(vis_videos)

            log_dict["epoch"] = epoch
            wandb.log(log_dict)

            # Best checkpoint based on WA_MPJPE
            if wa_mpjpe_all < best_loss:
                best_loss = wa_mpjpe_all
                best_path = os.path.join(args.save_dir, "ckpt_best.pt")
                torch.save({
                    "epoch": epoch,
                    "step": global_step,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "args": OmegaConf.to_container(args, resolve=True),
                }, best_path)
                print(f"[BEST] WA_MPJPE={best_loss:.2f}mm saved: {best_path}")

            model.train()

        # ===== Periodic checkpoint =====
        ckpt_every = getattr(args, "ckpt_every", 100)
        if ckpt_every > 0 and epoch % ckpt_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"ckpt_epoch{epoch:04d}.pt")
            torch.save({
                "epoch": epoch,
                "step": global_step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_loss": best_loss,
                "args": OmegaConf.to_container(args, resolve=True),
            }, ckpt_path)
            print(f"saved: {ckpt_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
