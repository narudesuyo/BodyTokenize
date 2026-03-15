"""
Evaluator for unconditional motion diffusion model.
Reconstruction = noise GT at timestep t → DDIM denoise → compare with GT.
Analogous to VQ-VAE encode→decode evaluation.
"""
import torch
import sys
sys.path.append(".")
import numpy as np
from src.evaluate.utils import recover_joints_from_body_hand
from src.evaluate.vis import visualize_two_motions
from src.evaluate.evaluator import _compute_sample_metrics, _source_from_key
from collections import defaultdict
from src.evaluate.metric import (
    batch_procrustes_align,
    batch_procrustes_align_sequence,
    w_align_firstk,
    mpjpe_bt,
    wa_mpjpe,
    w_mpjpe_firstk,
    accel_all_joints,
    relative_translation_error,
    root_translation_error,
)
from tqdm import tqdm


@torch.no_grad()
def ddim_denoise_from_t(model, x_t, t_start, num_steps=50, eta=0.0):
    """
    Partial DDIM denoising from timestep t_start back to t=0.
    model: MotionDiffusion instance (has .net, .alphas_cumprod, etc.)
    x_t:   (B, T, x_dim) noised input at timestep t_start
    t_start: integer timestep (e.g. 200)
    num_steps: number of DDIM steps for the partial trajectory
    """
    device = x_t.device
    B = x_t.shape[0]

    # Build sub-sequence of timesteps from t_start down to 0
    # Evenly spaced within [0, t_start]
    if num_steps >= t_start:
        timesteps = list(range(t_start, -1, -1))
    else:
        step_size = max(t_start // num_steps, 1)
        timesteps = list(range(t_start, 0, -step_size))
        if timesteps[-1] != 0:
            timesteps.append(0)

    prediction_type = getattr(model, "prediction_type", "eps")

    x = x_t
    for i, t_cur in enumerate(timesteps):
        t_batch = torch.full((B,), t_cur, device=device, dtype=torch.long)
        net_out = model.net(x, t_batch)

        ac = model.alphas_cumprod[t_cur]
        if i + 1 < len(timesteps):
            ac_prev = model.alphas_cumprod[timesteps[i + 1]]
        else:
            ac_prev = torch.tensor(1.0, device=device)

        if prediction_type == "x0":
            x0_pred = net_out
            eps = (x - torch.sqrt(ac) * x0_pred) / torch.sqrt(1 - ac)
        else:
            eps = net_out
            x0_pred = (x - torch.sqrt(1 - ac) * eps) / torch.sqrt(ac)
        x0_pred = torch.clamp(x0_pred, -5, 5)

        if t_cur == 0:
            x = x0_pred
            break

        sigma = eta * torch.sqrt((1 - ac_prev) / (1 - ac) * (1 - ac / ac_prev))
        dir_xt = torch.sqrt(1 - ac_prev - sigma ** 2) * eps
        noise = torch.randn_like(x) if t_cur > 0 else 0
        x = torch.sqrt(ac_prev) * x0_pred + dir_xt + sigma * noise

    return x


@torch.no_grad()
def evaluate_diffusion(
    model,
    dl,
    args,
    device,
    mean,
    std,
    noise_t=200,
    ddim_steps=20,
    num_save_samples: int = 1,
    viz_dir: str = "./eval",
    vis: bool = False,
    fps: int = 10,
    max_batches: int = 0,
    extra_vis_dl=None,
):
    """
    Evaluate diffusion model by noising GT → denoising → comparing with GT.
    Returns dict of EVAL/ metrics compatible with VQ-VAE evaluator format.
    """
    model.eval()

    parts = {
        "all":  slice(None),
        "body": slice(0, 22),
        "lh":   slice(22, 37),
        "rh":   slice(37, 52),
    }
    ROOT_IDX, LH_WRIST_IDX, RH_WRIST_IDX = 0, 20, 21

    sums = {f"{m}_{p}": 0.0 for m in ["pampjpe", "wa_mpjpe", "w_mpjpe"] for p in parts.keys()}
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
    nb = 0

    use_norm = getattr(args, "normalize", False)
    body_dim = getattr(args, "body_dim", 263)
    _hand_root = getattr(args, "hand_root", False)
    _hand_root_dim_total = getattr(args, "hand_root_dim", 9) * 2 if _hand_root else 0
    include_fingertips = getattr(args, "include_fingertips", True)
    joints_num = 62 if include_fingertips else 52

    vis_saved_by_source = defaultdict(int)
    vis_num_per_source = num_save_samples
    vis_videos = {}
    view_names = ["all", "body", "hands", "lh", "rh"]

    _body_in_dim = body_dim

    total = min(len(dl), max_batches) if max_batches > 0 else len(dl)
    for it, batch in tqdm(enumerate(dl), total=total, desc="Eval(diffusion)", leave=False):
        if max_batches > 0 and it >= max_batches:
            break
        mB = batch["mB"].to(device, non_blocking=True)
        mH = batch["mH"].to(device, non_blocking=True)
        motion = torch.cat([mB, mH], dim=-1)

        # Detect body=zeros samples (HOT3D hand-only)
        _body_is_zero = (mB.abs().sum(dim=(-1, -2)) < 1e-6)  # (B,)

        # Zero out unsupervised root dims (yaw, vx, vz) BEFORE normalization
        if not getattr(args, "use_root_loss", True):
            motion[..., :3] = 0.0

        # Normalize
        if use_norm:
            motion_norm = (motion - mean) / std
        else:
            motion_norm = motion

        # Noise at timestep noise_t
        B_cur = motion_norm.shape[0]
        t_tensor = torch.full((B_cur,), noise_t, device=device, dtype=torch.long)
        x_t, _ = model.q_sample(motion_norm, t_tensor)

        # Denoise via partial DDIM
        recon_norm = ddim_denoise_from_t(model, x_t, noise_t, num_steps=ddim_steps)

        # Denormalize
        if use_norm:
            gt_dn = motion
            pr_dn = recon_norm * std + mean
        else:
            gt_dn = motion
            pr_dn = recon_norm

        sums["feat_mse"] += torch.mean((pr_dn - gt_dn) ** 2).item()

        # Reconstruct joints (per-sample hand_only based on body=zeros)
        _hand_only = False  # default for batch-level call
        j_gt = recover_joints_from_body_hand(
            gt_dn[..., :body_dim], gt_dn[..., body_dim:],
            include_fingertips=include_fingertips,
            hand_root_dim=_hand_root_dim_total,
            joints_num=joints_num,
            use_root_loss=getattr(args, "use_root_loss", False),
            base_idx=args.base_idx,
            hand_local=getattr(args, "hand_local", False),
            hand_only=_hand_only,
        )
        j_pr = recover_joints_from_body_hand(
            pr_dn[..., :body_dim], pr_dn[..., body_dim:],
            include_fingertips=include_fingertips,
            hand_root_dim=_hand_root_dim_total,
            joints_num=joints_num,
            use_root_loss=getattr(args, "use_root_loss", False),
            base_idx=args.base_idx,
            hand_local=getattr(args, "hand_local", False),
            hand_only=_hand_only,
        )

        # Visualization (WA/W for body, hand-PA for hand views, source-balanced)
        if vis:
            keys = batch.get("keys", None)
            B_cur_vis = int(j_gt.shape[0])
            j_pr_wa = batch_procrustes_align_sequence(j_pr, j_gt)
            j_pr_w = w_align_firstk(j_pr, j_gt, num_align_frames=1)
            j_pr_hpa = j_pr_wa.clone()
            j_pr_hpa[..., 22:, :] = batch_procrustes_align(
                j_pr[..., 22:, :], j_gt[..., 22:, :])
            _hand_views = {"hands", "lh", "rh"}
            _view_align = {}
            for vn in view_names:
                if vn in _hand_views:
                    _view_align[vn] = [("PA_hand", j_pr_hpa)]
                else:
                    _view_align[vn] = [("WA", j_pr_wa), ("W", j_pr_w)]
            for b in range(B_cur_vis):
                key_b = keys[b] if isinstance(keys, list) and b < len(keys) else f"it{it:03d}_b{b:03d}"
                src_b = _source_from_key(key_b)
                if vis_saved_by_source[src_b] >= vis_num_per_source:
                    continue
                _jg_vis = j_gt[b]
                vis_idx = sum(vis_saved_by_source.values())
                for vname in view_names:
                    for aname, j_pr_a in _view_align[vname]:
                        _jp_vis = j_pr_a[b]
                        _mo = _compute_sample_metrics(_jg_vis, _jp_vis, parts)
                        save_path = f"{viz_dir}/{src_b}/{vis_idx:02d}/{aname}/{vname}.mp4"
                        visualize_two_motions(
                            _jg_vis, _jp_vis,
                            save_path=save_path,
                            fps=fps,
                            view=vname,
                            rotate=False,
                            include_fingertips=include_fingertips,
                            origin_align=True,
                            base_idx=args.base_idx,
                            metrics_overlay=_mo,
                        )
                        vis_videos[f"VIS/{src_b}/{vis_idx:02d}/{aname}/{vname}"] = save_path
                vis_saved_by_source[src_b] += 1

        # Pose metrics
        for name, slc in parts.items():
            jp_part = j_pr[..., slc, :]
            jg_part = j_gt[..., slc, :]

            jp_pa = batch_procrustes_align(jp_part, jg_part)
            sums[f"pampjpe_{name}"] += mpjpe_bt(jp_pa, jg_part, slice(None)).mean().item()
            sums[f"wa_mpjpe_{name}"] += wa_mpjpe(jp_part, jg_part, slice(None)).mean().item()
            sums[f"w_mpjpe_{name}"] += w_mpjpe_firstk(jp_part, jg_part, slice(None), num_align_frames=1).mean().item()

        # Trajectory metrics
        sums["relative_translation_error_pelvis"] += relative_translation_error(j_pr, j_gt, ROOT_IDX, use_scale=False).mean().item()
        sums["relative_translation_error_lh_wrist"] += relative_translation_error(j_pr, j_gt, LH_WRIST_IDX, use_scale=False).mean().item()
        sums["relative_translation_error_rh_wrist"] += relative_translation_error(j_pr, j_gt, RH_WRIST_IDX, use_scale=False).mean().item()
        sums["root_translation_error_pelvis"] += root_translation_error(j_pr, j_gt, ROOT_IDX, use_scale=False).mean().item()
        sums["root_translation_error_lh_wrist"] += root_translation_error(j_pr, j_gt, LH_WRIST_IDX, use_scale=False).mean().item()
        sums["root_translation_error_rh_wrist"] += root_translation_error(j_pr, j_gt, RH_WRIST_IDX, use_scale=False).mean().item()
        sums["accel"] += accel_all_joints(j_pr, j_gt, fps=fps).mean().item()

        nb += 1

    nb = max(nb, 1)
    metrics = {}

    for p in parts.keys():
        metrics[f"EVAL/PA_MPJPE/{p}(mm)"] = (sums[f"pampjpe_{p}"] / nb) * 1000.0
        metrics[f"EVAL/WA_MPJPE/{p}(mm)"] = (sums[f"wa_mpjpe_{p}"] / nb) * 1000.0
        metrics[f"EVAL/W_MPJPE/{p}(mm)"] = (sums[f"w_mpjpe_{p}"] / nb) * 1000.0

    metrics["EVAL/RECON/feat_mse"] = sums["feat_mse"] / nb
    metrics["EVAL/RelativeTranslationError/pelvis(%)"] = sums["relative_translation_error_pelvis"] / nb
    metrics["EVAL/RelativeTranslationError/lh_wrist(%)"] = sums["relative_translation_error_lh_wrist"] / nb
    metrics["EVAL/RelativeTranslationError/rh_wrist(%)"] = sums["relative_translation_error_rh_wrist"] / nb
    metrics["EVAL/RootTranslationError/pelvis(%)"] = sums["root_translation_error_pelvis"] / nb
    metrics["EVAL/RootTranslationError/lh_wrist(%)"] = sums["root_translation_error_lh_wrist"] / nb
    metrics["EVAL/RootTranslationError/rh_wrist(%)"] = sums["root_translation_error_rh_wrist"] / nb
    metrics["EVAL/ACCEL/all(mm/s^2)"] = sums["accel"] * 1000.0 / nb

    # --- extra vis from additional DataLoader (e.g. hot3d from train set) ---
    if vis and extra_vis_dl is not None:
        import os
        _extra_hand_views = ["hands", "lh", "rh"]
        _extra_saved = defaultdict(int)
        for batch_ex in extra_vis_dl:
            mB_ex = batch_ex["mB"].to(device, non_blocking=True)
            mH_ex = batch_ex["mH"].to(device, non_blocking=True)
            keys_ex = batch_ex.get("keys", None)
            motion_ex = torch.cat([mB_ex, mH_ex], dim=-1)

            if not getattr(args, "use_root_loss", True):
                motion_ex[..., :3] = 0.0

            if use_norm:
                motion_ex_n = (motion_ex - mean) / std
            else:
                motion_ex_n = motion_ex

            B_ex = motion_ex_n.shape[0]
            t_tensor_ex = torch.full((B_ex,), noise_t, device=device, dtype=torch.long)
            x_t_ex, _ = model.q_sample(motion_ex_n, t_tensor_ex)
            recon_ex_n = ddim_denoise_from_t(model, x_t_ex, noise_t, num_steps=ddim_steps)

            if use_norm:
                gt_ex = motion_ex
                pr_ex = recon_ex_n * std + mean
            else:
                gt_ex = motion_ex
                pr_ex = recon_ex_n

            _hand_only_ex = False
            j_gt_ex = recover_joints_from_body_hand(
                gt_ex[..., :body_dim], gt_ex[..., body_dim:],
                include_fingertips=include_fingertips,
                hand_root_dim=_hand_root_dim_total,
                joints_num=joints_num,
                use_root_loss=getattr(args, "use_root_loss", False),
                base_idx=args.base_idx,
                hand_local=getattr(args, "hand_local", False),
                hand_only=_hand_only_ex,
            )
            j_pr_ex = recover_joints_from_body_hand(
                pr_ex[..., :body_dim], pr_ex[..., body_dim:],
                include_fingertips=include_fingertips,
                hand_root_dim=_hand_root_dim_total,
                joints_num=joints_num,
                use_root_loss=getattr(args, "use_root_loss", False),
                base_idx=args.base_idx,
                hand_local=getattr(args, "hand_local", False),
                hand_only=_hand_only_ex,
            )

            j_pr_hpa_ex = batch_procrustes_align_sequence(j_pr_ex, j_gt_ex).clone()
            j_pr_hpa_ex[..., 22:, :] = batch_procrustes_align(
                j_pr_ex[..., 22:, :], j_gt_ex[..., 22:, :])

            for b in range(j_gt_ex.shape[0]):
                key_b = keys_ex[b] if isinstance(keys_ex, list) and b < len(keys_ex) else f"extra_b{b:03d}"
                src_b = _source_from_key(key_b)
                if _extra_saved[src_b] >= num_save_samples:
                    continue
                _src_idx = _extra_saved[src_b]
                _mo = _compute_sample_metrics(j_gt_ex[b], j_pr_hpa_ex[b],
                    {"lh": parts["lh"], "rh": parts["rh"]})
                for vname in _extra_hand_views:
                    save_path = os.path.join(viz_dir, src_b, f"{_src_idx:02d}", "PA_hand", f"{vname}.mp4")
                    visualize_two_motions(
                        j_gt_ex[b], j_pr_hpa_ex[b],
                        save_path=save_path,
                        fps=fps, view=vname, rotate=False,
                        include_fingertips=include_fingertips,
                        origin_align=True, base_idx=args.base_idx,
                        metrics_overlay=_mo,
                    )
                    vis_videos[f"VIS/{src_b}/{_src_idx:02d}/PA_hand/{vname}"] = save_path
                _extra_saved[src_b] += 1

            if all(v >= num_save_samples for v in _extra_saved.values()):
                break

    return metrics, vis_videos
