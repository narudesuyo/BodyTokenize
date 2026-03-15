import sys
sys.path.append(".")

from src.dataset.dataloader import MotionDataset
from src.evaluate.utils import recover_joints_from_body_hand
from src.evaluate.vis import visualize_two_motions
from src.dataset.collate import collate_stack
from src.util.utils import count_params, set_seed, compute_part_losses
from src.evaluate.metric import codebook_stats
from src.evaluate.evaluator import evaluate_model, _parse_source_list
from src.train.utils import build_model_from_args
import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import omegaconf
from omegaconf import OmegaConf
import wandb
from collections import OrderedDict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config/motion_vqvae.yaml")
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
    # args.data_dir が「ptパス」になってる前提（必要ならyaml側で名前変えて）

    _use_cache = getattr(args, "use_cache", False)
    _hand_root = getattr(args, "hand_root", False)
    _hand_only = getattr(args, "hand_only", False)
    _debug_max = getattr(args, "debug_max_samples", 0)

    ds = MotionDataset(
        pt_path=args.cache_pt if _use_cache else args.data_dir,
        feet_thre=getattr(args, "feet_thre", 0.002),
        kp_field=getattr(args, "kp_field", "kp3d"),
        clip_len=getattr(args, "T", 81),          # ★80 crop
        random_crop=getattr(args, "random_crop", True),  # ★trainならTrue推奨
        pad_if_short=getattr(args, "pad_if_short", True),
        include_fingertips=getattr(args, "include_fingertips", False),
        to_torch=True,
        base_idx=args.base_idx,
        hand_local=getattr(args, "hand_local", False),
        use_cache=_use_cache,
        hand_root=_hand_root,
        hand_only=_hand_only,
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
        collate_fn=collate_stack,   # ★ここ変更
    )

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
        hand_root=_hand_root,
        hand_only=_hand_only,
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

    # ===== Model =====
    start_epoch = 1
    global_step = 0
    best_metric = float("inf")
    warmup_epochs = getattr(args, "warmup_epochs", 50)
    min_lr = getattr(args, "min_lr", 1e-6)

    _decoder_type = getattr(args, "decoder_type", "regressor")
    _is_flow = _decoder_type in ("flow", "diffusion")

    if args_cli.resume is not None:
        ckpt = torch.load(args_cli.resume, weights_only=False)
        model = build_model_from_args(args, device)
        # Allow loading regressor ckpt into flow model (strict=False for new flow params)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"[WARN] Missing keys (expected for new decoder): {missing[:10]}{'...' if len(missing) > 10 else ''}")
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        if not missing:  # Only load opt state if model loaded fully
            opt.load_state_dict(ckpt["opt"])

        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        best_metric = ckpt.get("best_metric", float("inf"))
    else:
        model = build_model_from_args(args, device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # ===== LR Scheduler: Linear Warmup + Cosine Annealing =====
    warmup_sched = LambdaLR(opt, lr_lambda=lambda ep: min(1.0, (ep + 1) / warmup_epochs))
    cosine_sched = CosineAnnealingLR(opt, T_max=args.epochs - warmup_epochs, eta_min=min_lr)
    scheduler = SequentialLR(opt, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

    if args_cli.resume is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    else:
        # fast-forward scheduler to start_epoch
        for _ in range(1, start_epoch):
            scheduler.step()
    n_all, n_train = count_params(model)
    print("========== MODEL ==========")
    print(model)  # architecture (full)
    print("========== PARAMS =========")
    print(f"Total params     : {n_all:,}")
    print(f"Trainable params : {n_train:,}")
    print(f"Decoder type     : {_decoder_type}")
    print("===========================")

    # ===== wandb =====
    wandb.init(project=args.project, name=args.name, config=OmegaConf.to_container(args, resolve=True))
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    wandb.watch(model, log=None)

    # Determine body/hand dims from model
    _body_in_dim = model.body_in_dim
    _hand_in_dim = model.hand_in_dim
    _total_dim = _body_in_dim + _hand_in_dim

    # Compute or load normalization stats from cache
    from src.train.utils import compute_norm_stats
    _stats_cache_dir = os.path.join(args.save_dir, "norm_stats")
    mean, std = compute_norm_stats(
        args.cache_pt, body_dim=_body_in_dim, root_dim=1,
        stats_cache_dir=_stats_cache_dir,
    )
    mean = mean[:_total_dim].to(device)
    std = std[:_total_dim].to(device)
    _hand_root_dim_total = getattr(args, "hand_root_dim", 9) * 2 if _hand_root else 0
    _use_token_sep = getattr(args, "use_token_separation", False)

    for epoch in tqdm(range(start_epoch, args.epochs + 1), desc="Epochs"):
        model.train()
        t0 = time.time()
        _epoch_loss = 0.0
        _epoch_steps = 0
        _epoch_comp = {}

        for it, batch in tqdm(enumerate(dl), total=len(dl), desc="Training", leave=False):
            if args.eval_check:
                break

            mB = batch["mB"].to(device, non_blocking=True)  # (B,T,body_in_dim)
            mH = batch["mH"].to(device, non_blocking=True)  # (B,T,hand_in_dim)
            if _hand_only:
                # hand_only: body is zeros, only normalize hand part
                if args.normalize:
                    hand_mean = mean[_body_in_dim:_total_dim] if mean.shape[0] >= _total_dim else mean[_body_in_dim:]
                    hand_std = std[_body_in_dim:_total_dim] if std.shape[0] >= _total_dim else std[_body_in_dim:]
                    if hand_mean.shape[0] < mH.shape[-1]:
                        pad_len = mH.shape[-1] - hand_mean.shape[0]
                        hand_mean = torch.cat([hand_mean, torch.zeros(pad_len, device=device)])
                        hand_std = torch.cat([hand_std, torch.ones(pad_len, device=device)])
                    mH = (mH - hand_mean) / hand_std
                # mB stays zeros
            else:
                motion = torch.cat([mB, mH], dim=-1)
                if args.normalize:
                    motion = (motion - mean) / std
                mB = motion[:, :, :_body_in_dim]
                mH = motion[:, :, _body_in_dim:]

            # shape check
            if mH.shape[1] != (args.T-1) or (not _hand_only and mB.shape[1] != (args.T-1)):
                raise RuntimeError(
                    f"Time length mismatch: got mB={mB.shape[1]} mH={mH.shape[1]} but args.T={args.T}. "
                    f"(dataset T={args.T} -> expected T={args.T-1})"
                )


            recon, losses, idx = model(mB, mH)

            target = torch.cat([mB, mH], dim=-1)
            loss = losses["loss"]
            part_losses = {}
            joints_loss = torch.tensor(0.0, device=device)

            if not _is_flow:
                # regressor: compute part losses and optional joints loss
                part_losses = compute_part_losses(recon, target, hand_root_dim=_hand_root_dim_total)

                if args.joints_loss:
                    if args.normalize:
                        if mean.shape[0] < _total_dim:
                            mean_ext = torch.cat([mean, torch.zeros(_total_dim - mean.shape[0], device=device)])
                            std_ext = torch.cat([std, torch.ones(_total_dim - std.shape[0], device=device)])
                            recon_denorm = recon * std_ext + mean_ext
                            gt_denorm = target * std_ext + mean_ext
                        else:
                            recon_denorm = recon * std[:_total_dim] + mean[:_total_dim]
                            gt_denorm = target * std[:_total_dim] + mean[:_total_dim]
                    else:
                        recon_denorm = recon
                        gt_denorm = target
                    joints_num = 62 if args.include_fingertips else 52
                    gt_joints = recover_joints_from_body_hand(
                        gt_denorm[..., :_body_in_dim], gt_denorm[..., _body_in_dim:],
                        include_fingertips=args.include_fingertips,
                        hand_root_dim=_hand_root_dim_total,
                        joints_num=joints_num,
                        base_idx=args.base_idx,
                        hand_local=getattr(args, "hand_local", False),
                        hand_only=_hand_only,
                    )
                    pred_joints = recover_joints_from_body_hand(
                        recon_denorm[..., :_body_in_dim], recon_denorm[..., _body_in_dim:],
                        include_fingertips=args.include_fingertips,
                        hand_root_dim=_hand_root_dim_total,
                        joints_num=joints_num,
                        base_idx=args.base_idx,
                        hand_local=getattr(args, "hand_local", False),
                        hand_only=_hand_only,
                    )
                    gt_joints = gt_joints - gt_joints[..., :1, :]
                    pred_joints = pred_joints - pred_joints[..., :1, :]

                    joints_loss_weight = args.joints_loss_weight
                    joints_loss = joints_loss_weight * torch.mean((pred_joints - gt_joints) ** 2)
                    loss += joints_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            # Codebook stats (handle both 2-codebook and 4-codebook)
            cb_log = {}
            if _hand_only:
                if _use_token_sep:
                    for cb_name in ["HR", "HL"]:
                        key = f"idx{cb_name}"
                        if key in idx:
                            u, p = codebook_stats(idx[key].detach(), args.K)
                            cb_log[f"code_usage_{cb_name}"] = u
                            cb_log[f"perplexity_{cb_name}"] = p
                else:
                    usageH, pplH = codebook_stats(idx["idxH"].detach(), args.K)
                    cb_log = {"code_usage_H": usageH, "perplexity_H": pplH}
            elif _use_token_sep:
                for cb_name in ["BR", "BL", "HR", "HL"]:
                    key = f"idx{cb_name}"
                    if key in idx:
                        u, p = codebook_stats(idx[key].detach(), args.K)
                        cb_log[f"code_usage_{cb_name}"] = u
                        cb_log[f"perplexity_{cb_name}"] = p
            else:
                usageH, pplH = codebook_stats(idx["idxH"].detach(), args.K)
                usageB, pplB = codebook_stats(idx["idxB"].detach(), args.K)
                cb_log = {
                    "code_usage_H": usageH, "code_usage_B": usageB,
                    "perplexity_H": pplH, "perplexity_B": pplB,
                }
            # Hand trajectory codebook
            if "idxHT" in idx:
                u, p = codebook_stats(idx["idxHT"].detach(), args.K)
                cb_log["code_usage_HT"] = u
                cb_log["perplexity_HT"] = p

            _epoch_loss += float(loss.detach())
            _epoch_steps += 1

            # Accumulate per-component losses
            _comp_keys = ["recon_loss", "commit_loss", "flow_loss", "entropy_loss",
                          "commit_H", "commit_B", "commit_BR", "commit_BL", "commit_HR", "commit_HL", "commit_HT",
                          "loss_body_dec", "loss_hand_dec", "loss_full_dec",
                          "loss_full_hand_masked", "loss_full_body_masked",
                          "joints_loss", "joints_loss_hand", "bone_length_loss", "loss_handonly_samples"]
            for ck in _comp_keys:
                if ck in losses:
                    v = losses[ck]
                    _epoch_comp[ck] = _epoch_comp.get(ck, 0.0) + (float(v.detach()) if torch.is_tensor(v) else float(v))
            if not _is_flow and args.joints_loss:
                _epoch_comp["joints_loss"] = _epoch_comp.get("joints_loss", 0.0) + float(joints_loss.detach())
            for k, v in part_losses.items():
                _epoch_comp[k] = _epoch_comp.get(k, 0.0) + float(v.detach())
            for k, v in cb_log.items():
                _epoch_comp[k] = _epoch_comp.get(k, 0.0) + float(v)

            global_step += 1

        scheduler.step()
        _n = max(_epoch_steps, 1)
        epoch_log = {
            "epoch": epoch,
            "epoch_avg_loss": _epoch_loss / _n,
            "lr": opt.param_groups[0]["lr"],
            "epoch_time_sec": time.time() - t0,
        }
        for k, v in _epoch_comp.items():
            epoch_log[f"epoch_avg_{k}"] = v / _n
        wandb.log(epoch_log)
        _extra = " ".join(f"{k}={v/_n:.4f}" for k, v in _epoch_comp.items() if v > 0)
        print(f"[E{epoch:04d}] avg_loss={_epoch_loss / _n:.6f} lr={opt.param_groups[0]['lr']:.2e} {_extra}")

        # ===== checkpoint =====
        if (epoch % args.ckpt_every) == 0:
            ckpt = {
                "epoch": epoch,
                "step": global_step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_metric": best_metric,
                "args": vars(args),
            }
            ckpt_path = os.path.join(args.save_dir, f"ckpt_epoch{epoch:03d}.pt")
            torch.save(ckpt, ckpt_path)
            print("saved:", ckpt_path)


        # ===== eval (optional) =====
        eval_every = args.eval_every
        if eval_every > 0 and ((epoch % eval_every) == 0 or epoch == start_epoch):
            model.eval()
            _do_vis = args.eval_save_vis_every > 0 and ((epoch % args.eval_save_vis_every) == 0 or epoch == start_epoch)
            metrics = evaluate_model(
                model,
                dl_eval,
                args,
                device=device,
                num_save_samples=args.eval_num_save_samples,
                viz_dir=f"{args.eval_vis_dir}/epoch_{epoch:03d}",
                vis=_do_vis,
                mean=mean,
                std=std,
                extra_vis_dl=dl_extra_vis if _do_vis else None,
            )

            # print
            if _hand_only:
                _m = metrics
                print(
                    f"[E{epoch:04d} EVAL (hand_only)]\n"
                    f"  feat_mse={_m.get('EVAL/RECON/feat_mse', 0):.6f}\n"
                    f"  pampjpe(mm)  lh/rh="
                    f"{_m.get('EVAL/PA_MPJPE/lh(mm)', 0):.2f}/"
                    f"{_m.get('EVAL/PA_MPJPE/rh(mm)', 0):.2f}\n"
                    f"  wa_mpjpe(mm) lh/rh="
                    f"{_m.get('EVAL/WA_MPJPE/lh(mm)', 0):.2f}/"
                    f"{_m.get('EVAL/WA_MPJPE/rh(mm)', 0):.2f}\n"
                    f"  w_mpjpe(mm)  lh/rh="
                    f"{_m.get('EVAL/W_MPJPE/lh(mm)', 0):.2f}/"
                    f"{_m.get('EVAL/W_MPJPE/rh(mm)', 0):.2f}\n"
                    f"  codebook H usage/ppl={_m.get('EVAL/CODEBOOK/H_usage', 0):.3f}/{_m.get('EVAL/CODEBOOK/H_ppl', 0):.1f}"
                )
            else:
                _m = metrics
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
                    f"  accel(mm/s^2) all={_m.get('EVAL/ACCEL/all(mm/s^2)', 0):.2f}\n"
                    f"  codebook H usage/ppl={_m.get('EVAL/CODEBOOK/H_usage', 0):.3f}/{_m.get('EVAL/CODEBOOK/H_ppl', 0):.1f} "
                    f"B usage/ppl={_m.get('EVAL/CODEBOOK/B_usage', 0):.3f}/{_m.get('EVAL/CODEBOOK/B_ppl', 0):.1f}"
                )

            # wandb — vis mp4s (pick up to N per source)
            _vis_dir = f"{args.eval_vis_dir}/epoch_{epoch:03d}"
            _vis_mp4s = sorted(glob.glob(os.path.join(_vis_dir, "**/*.mp4"), recursive=True))
            _max_per_src = getattr(args, "eval_vis_wandb_max_per_source", 10)
            _vis_by_src = {}
            for _mp4 in _vis_mp4s:
                _rel = os.path.relpath(_mp4, _vis_dir)
                _src = _rel.split(os.sep)[0]
                _vis_by_src.setdefault(_src, []).append(_mp4)
            for _src, _mp4s in sorted(_vis_by_src.items()):
                for _mp4 in _mp4s[:_max_per_src]:
                    _vkey = f"VIS/{os.path.relpath(_mp4, _vis_dir).replace(os.sep, '/')}"
                    try:
                        metrics[_vkey] = wandb.Video(_mp4, format="mp4")
                    except Exception as e:
                        print(f"[WARN] wandb.Video failed for {_mp4}: {e}")
            metrics["epoch"] = epoch
            wandb.log(metrics)

            # ===== best checkpoint =====
            if _hand_only:
                cur_metric = (metrics.get("EVAL/WA_MPJPE/lh(mm)", float("inf")) + metrics.get("EVAL/WA_MPJPE/rh(mm)", float("inf"))) / 2.0
            else:
                cur_metric = metrics.get("EVAL/WA_MPJPE/all(mm)", float("inf"))
            if cur_metric < best_metric:
                best_metric = cur_metric
                best_ckpt = {
                    "epoch": epoch,
                    "step": global_step,
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_metric": best_metric,
                    "args": vars(args),
                }
                best_path = os.path.join(args.save_dir, "ckpt_best.pt")
                torch.save(best_ckpt, best_path)
                print(f"[BEST] WA_MPJPE={best_metric:.2f}mm saved: {best_path}")

            model.train()

        # epoch_time already logged in epoch_log above

    wandb.finish()


if __name__ == "__main__":
    main()
