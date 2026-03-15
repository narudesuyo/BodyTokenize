"""
Evaluator for unconditional motion flow matching model.
Reconstruction = interpolate GT toward noise at t_start -> ODE denoise -> compare with GT.
Analogous to diffusion evaluator's noise->denoise approach.
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
def evaluate_flow(
    model,
    dl,
    args,
    device,
    mean,
    std,
    noise_t: float = 0.5,
    ode_steps: int = 30,
    solver: str = "heun",
    num_save_samples: int = 1,
    viz_dir: str = "./eval",
    vis: bool = False,
    fps: int = 10,
    max_batches: int = 0,
    extra_vis_dl=None,
):
    """
    Evaluate flow matching model by interpolating GT toward noise -> ODE denoise -> compare.
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

    total = min(len(dl), max_batches) if max_batches > 0 else len(dl)
    for it, batch in tqdm(enumerate(dl), total=total, desc="Eval(flow)", leave=False):
        if max_batches > 0 and it >= max_batches:
            break
        mB = batch["mB"].to(device, non_blocking=True)
        mH = batch["mH"].to(device, non_blocking=True)
        motion = torch.cat([mB, mH], dim=-1)

        # Zero out unsupervised root dims (yaw, vx, vz) BEFORE normalization
        if not getattr(args, "use_root_loss", False):
            motion[..., :3] = 0.0

        # Normalize
        if use_norm:
            motion_norm = (motion - mean) / std
        else:
            motion_norm = motion

        # Interpolate toward noise at t_start
        noise = torch.randn_like(motion_norm)
        x_t = (1.0 - noise_t) * motion_norm + noise_t * noise

        # ODE denoise from t_start to 0
        recon_norm = model.denoise_from_t(x_t, t_start=noise_t, steps=ode_steps, solver=solver)

        # Denormalize
        if use_norm:
            gt_dn = motion
            pr_dn = recon_norm * std + mean
        else:
            gt_dn = motion
            pr_dn = recon_norm

        sums["feat_mse"] += torch.mean((pr_dn - gt_dn) ** 2).item()

        # Reconstruct joints
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

            if not getattr(args, "use_root_loss", False):
                motion_ex[..., :3] = 0.0

            if use_norm:
                motion_ex_n = (motion_ex - mean) / std
            else:
                motion_ex_n = motion_ex

            noise_ex = torch.randn_like(motion_ex_n)
            x_t_ex = (1.0 - noise_t) * motion_ex_n + noise_t * noise_ex
            recon_ex_n = model.denoise_from_t(x_t_ex, t_start=noise_t, steps=ode_steps, solver=solver)

            if use_norm:
                gt_ex = motion_ex
                pr_ex = recon_ex_n * std + mean
            else:
                gt_ex = motion_ex
                pr_ex = recon_ex_n

            j_gt_ex = recover_joints_from_body_hand(
                gt_ex[..., :body_dim], gt_ex[..., body_dim:],
                include_fingertips=include_fingertips,
                hand_root_dim=_hand_root_dim_total,
                joints_num=joints_num,
                use_root_loss=getattr(args, "use_root_loss", False),
                base_idx=args.base_idx,
                hand_local=getattr(args, "hand_local", False),
                hand_only=_hand_only,
            )
            j_pr_ex = recover_joints_from_body_hand(
                pr_ex[..., :body_dim], pr_ex[..., body_dim:],
                include_fingertips=include_fingertips,
                hand_root_dim=_hand_root_dim_total,
                joints_num=joints_num,
                use_root_loss=getattr(args, "use_root_loss", False),
                base_idx=args.base_idx,
                hand_local=getattr(args, "hand_local", False),
                hand_only=_hand_only,
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
