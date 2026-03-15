import torch
import sys
sys.path.append(".")
import os
import re
import numpy as np
from collections import defaultdict
from src.evaluate.utils import recover_joints_from_body_hand
from src.evaluate.vis import visualize_two_motions
from src.evaluate.metric import (
    codebook_stats,
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


def _source_from_key(key):
    if not isinstance(key, str):
        return "unknown"
    if "::" in key:
        return key.split("::", 1)[0]
    return "default"


def _zup_to_yup(joints):
    """Convert z-up joints to y-up: (x,y,z) -> (x,z,-y). Works for any (..., 3) shape."""
    out = torch.empty_like(joints)
    out[..., 0] = joints[..., 0]
    out[..., 1] = joints[..., 2]
    out[..., 2] = -joints[..., 1]
    return out


def _safe_path_token(text, fallback="sample"):
    if not isinstance(text, str) or len(text) == 0:
        return fallback
    token = re.sub(r"[^a-zA-Z0-9._-]+", "_", text)
    token = token.strip("._-")
    if len(token) == 0:
        token = fallback
    return token[:160]


def _compute_sample_metrics(j_gt, j_pr, parts):
    """
    Compute per-sample metrics for MP4 overlay.
    j_gt, j_pr: (T, J, 3) numpy or tensor
    parts: dict {name: slice}
    Returns: {"static": {"WA-MPJPE": {part: mm}, "W-MPJPE": {part: mm}},
              "dynamic": {"PA-MPJPE": {part: np.array(T)}}}
    """
    if isinstance(j_gt, np.ndarray):
        j_gt = torch.from_numpy(j_gt).float()
    if isinstance(j_pr, np.ndarray):
        j_pr = torch.from_numpy(j_pr).float()
    gt = j_gt.unsqueeze(0)  # (1,T,J,3)
    pr = j_pr.unsqueeze(0)

    static = {"WA-MPJPE": {}, "W-MPJPE": {}}
    dynamic = {"PA-MPJPE": {}}

    for name, slc in parts.items():
        gp = gt[..., slc, :]
        pp = pr[..., slc, :]
        # WA-MPJPE (mm)
        static["WA-MPJPE"][name] = wa_mpjpe(pp, gp, slice(None)).item() * 1000.0
        # W-MPJPE (mm)
        static["W-MPJPE"][name] = w_mpjpe_firstk(pp, gp, slice(None), num_align_frames=1).item() * 1000.0
        # PA-MPJPE per-frame (mm) -> (T,)
        pp_pa = batch_procrustes_align(pp, gp)
        pa_per_frame = mpjpe_bt(pp_pa, gp, slice(None)).squeeze(0).cpu().numpy() * 1000.0
        dynamic["PA-MPJPE"][name] = pa_per_frame

    return {"static": static, "dynamic": dynamic}


def _parse_source_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",")]
        out = [p for p in parts if p]
        return out if out else None
    if isinstance(value, (list, tuple)):
        out = [str(v).strip() for v in value if str(v).strip()]
        return out if out else None
    s = str(value).strip()
    return [s] if s else None


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
    mean: torch.Tensor = None,
    std: torch.Tensor = None,
    extra_vis_dl=None,
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
        "relative_translation_error_pelvis": 0.0,
        "relative_translation_error_lh_wrist": 0.0,
        "relative_translation_error_rh_wrist": 0.0,
        "root_translation_error_pelvis": 0.0,
        "root_translation_error_lh_wrist": 0.0,
        "root_translation_error_rh_wrist": 0.0,
        "accel": 0.0,
    })
    cb_stats = {"usageH": 0.0, "pplH": 0.0, "usageB": 0.0, "pplB": 0.0,
                 "usageBR": 0.0, "pplBR": 0.0, "usageBL": 0.0, "pplBL": 0.0,
                 "usageHR": 0.0, "pplHR": 0.0, "usageHL": 0.0, "pplHL": 0.0}

    # hand-only / body-only decoder accumulators (three decoders only)
    handonly_parts = {"lh": slice(22, 37), "rh": slice(37, 52)}
    bodyonly_parts = {"body": slice(0, 22)}
    sums_handonly = {f"{m}_{p}": 0.0 for m in ["pampjpe", "wa_mpjpe", "w_mpjpe"] for p in handonly_parts}
    sums_bodyonly = {f"{m}_{p}": 0.0 for m in ["pampjpe", "wa_mpjpe", "w_mpjpe"] for p in bodyonly_parts}

    nb = 0

    # Visualization policy (backward compatible by default)
    vis_balance_sources = bool(getattr(args, "eval_vis_balance_sources", False))
    vis_sources = _parse_source_list(getattr(args, "eval_vis_sources", None))
    vis_sources_set = set(vis_sources) if vis_sources is not None else None
    vis_num_per_source = int(getattr(args, "eval_vis_num_per_source", num_save_samples))
    _hand_root = getattr(args, "hand_root", False)
    _hand_root_dim_total = getattr(args, "hand_root_dim", 9) * 2 if _hand_root else 0
    _body_in_dim = getattr(args, "body_in_dim", 263)
    _use_token_sep = getattr(args, "use_token_separation", False)
    _use_three_decoders = getattr(args, "use_three_decoders", False)
    _hand_only = getattr(args, "hand_only", False)
    _decoder_type = getattr(args, "decoder_type", "regressor")
    _is_flow = _decoder_type in ("flow", "diffusion")

    vis_saved_total = 0
    vis_saved_by_source = defaultdict(int)
    view_names = ["hands", "lh", "rh"] if _hand_only else ["all", "body", "hands", "lh", "rh"]

    # Determine hand_in_dim
    if hasattr(args, "hand_in_dim"):
        _hand_in_dim = args.hand_in_dim
    elif args.include_fingertips:
        _hand_in_dim = 498 if _hand_root else 480
    else:
        _hand_in_dim = 378 if _hand_root else 360
    _total_dim = _body_in_dim + _hand_in_dim

    if getattr(args, "normalize", False):
        if mean is None or std is None:
            mean = torch.from_numpy(np.load(args.mean_path)).float().to(device)
            std  = torch.from_numpy(np.load(args.std_path)).float().to(device)
        else:
            mean = mean.to(device)
            std = std.to(device)

    for it, batch in tqdm(enumerate(dl), total=len(dl), leave=False):
        mB = batch["mB"].to(device, non_blocking=True)
        mH = batch["mH"].to(device, non_blocking=True)

        # Detect body=zeros BEFORE normalization (raw data, exact zeros)
        _body_is_zero = (mB.abs().sum(dim=(-1, -2)) < 1e-6)  # (B,)

        if getattr(args, "normalize", False):
            if _hand_only:
                # hand_only: body is zeros, only normalize hand
                hand_mean = mean[_body_in_dim:_total_dim] if mean.shape[0] >= _total_dim else mean[_body_in_dim:]
                hand_std = std[_body_in_dim:_total_dim] if std.shape[0] >= _total_dim else std[_body_in_dim:]
                if hand_mean.shape[0] < mH.shape[-1]:
                    pad_len = mH.shape[-1] - hand_mean.shape[0]
                    hand_mean = torch.cat([hand_mean, torch.zeros(pad_len, device=device)])
                    hand_std = torch.cat([hand_std, torch.ones(pad_len, device=device)])
                mH = (mH - hand_mean) / hand_std
            else:
                motion = torch.cat([mB, mH], dim=-1)
                if mean.shape[0] < motion.shape[-1]:
                    pad_len = motion.shape[-1] - mean.shape[0]
                    mean_ext = torch.cat([mean, torch.zeros(pad_len, device=device)])
                    std_ext = torch.cat([std, torch.ones(pad_len, device=device)])
                    motion = (motion - mean_ext) / std_ext
                else:
                    motion = (motion - mean[:_total_dim]) / std[:_total_dim]
                mB = motion[..., :_body_in_dim]
                mH = motion[..., _body_in_dim:]

        recon, losses, idx = model(mB, mH)   # recon: (B,T,total_dim) or None for flow

        if _is_flow and recon is None:
            # Flow decoder: reconstruct via ODE sampling from quantized indices
            target_T = mB.shape[1]
            recon = model.sample_from_ids(
                idx, target_T=target_T,
                steps=getattr(args, "flow_sample_steps", 30),
                solver=getattr(args, "flow_solver", "heun"),
            )

        # --- denormalize if needed ---
        gt623 = torch.cat([mB, mH], dim=-1)
        pr623 = recon

        if getattr(args, "normalize", False):
            if _hand_only:
                # hand_only: body is zeros (not normalized), only denorm hand part
                hand_mean = mean[_body_in_dim:_total_dim] if mean.shape[0] >= _total_dim else mean[_body_in_dim:]
                hand_std = std[_body_in_dim:_total_dim] if std.shape[0] >= _total_dim else std[_body_in_dim:]
                if hand_mean.shape[0] < mH.shape[-1]:
                    pad_len = mH.shape[-1] - hand_mean.shape[0]
                    hand_mean = torch.cat([hand_mean, torch.zeros(pad_len, device=device)])
                    hand_std = torch.cat([hand_std, torch.ones(pad_len, device=device)])
                gt_hand_denorm = gt623[..., _body_in_dim:] * hand_std + hand_mean
                pr_hand_denorm = pr623[..., _body_in_dim:] * hand_std + hand_mean
                gt623 = torch.cat([gt623[..., :_body_in_dim], gt_hand_denorm], dim=-1)
                pr623 = torch.cat([pr623[..., :_body_in_dim], pr_hand_denorm], dim=-1)
            else:
                if mean.shape[0] < _total_dim:
                    pad_len = _total_dim - mean.shape[0]
                    mean_ext = torch.cat([mean, torch.zeros(pad_len, device=device)])
                    std_ext = torch.cat([std, torch.ones(pad_len, device=device)])
                    gt623 = gt623 * std_ext + mean_ext
                    pr623 = pr623 * std_ext + mean_ext
                else:
                    gt623 = gt623 * std[:_total_dim] + mean[:_total_dim]
                    pr623 = pr623 * std[:_total_dim] + mean[:_total_dim]

        sums["feat_mse"] += torch.mean((pr623 - gt623) ** 2).item()

        # --- joints (B,T,52,3) ---
        joints_num = 52 if not args.include_fingertips else 62
        j_gt = recover_joints_from_body_hand(
            gt623[..., :_body_in_dim], gt623[..., _body_in_dim:],
            include_fingertips=args.include_fingertips,
            hand_root_dim=_hand_root_dim_total,
            joints_num=joints_num,
            use_root_loss=getattr(args, "use_root_loss", False),
            base_idx=args.base_idx,
            hand_local=getattr(args, "hand_local", False),
            hand_only=_hand_only,
        )
        j_pr = recover_joints_from_body_hand(
            pr623[..., :_body_in_dim], pr623[..., _body_in_dim:],
            include_fingertips=args.include_fingertips,
            hand_root_dim=_hand_root_dim_total,
            joints_num=joints_num,
            use_root_loss=getattr(args, "use_root_loss", False),
            base_idx=args.base_idx,
            hand_local=getattr(args, "hand_local", False),
            hand_only=_hand_only,
        )

        # --- per-sample hand-only override for body=zeros samples (HOT3D in mix) ---
        _has_wrist_world = "lh_wrist_world" in batch and "rh_wrist_world" in batch
        if _body_is_zero.any() and not _hand_only:
            from src.evaluate.utils import recover_hand_only_joints
            _lh_ww = batch["lh_wrist_world"].to(device) if _has_wrist_world else None
            _rh_ww = batch["rh_wrist_world"].to(device) if _has_wrist_world else None
            for bi in range(int(gt623.shape[0])):
                if not _body_is_zero[bi]:
                    continue
                _bi_lh_ww = _lh_ww[bi:bi+1] if _lh_ww is not None else None
                _bi_rh_ww = _rh_ww[bi:bi+1] if _rh_ww is not None else None
                # GT: use wrist world positions if available, else velocity integration
                j_gt[bi] = recover_hand_only_joints(
                    gt623[bi:bi+1, :, _body_in_dim:],
                    include_fingertips=args.include_fingertips,
                    hand_root_dim=_hand_root_dim_total,
                    joints_num=joints_num,
                    hand_local=getattr(args, "hand_local", False),
                    lh_wrist_world=_bi_lh_ww,
                    rh_wrist_world=_bi_rh_ww,
                )[0]
                # Pred: use same GT wrist positions (model can't predict world wrist)
                j_pr[bi] = recover_hand_only_joints(
                    pr623[bi:bi+1, :, _body_in_dim:],
                    include_fingertips=args.include_fingertips,
                    hand_root_dim=_hand_root_dim_total,
                    joints_num=joints_num,
                    hand_local=getattr(args, "hand_local", False),
                    lh_wrist_world=_bi_lh_ww,
                    rh_wrist_world=_bi_rh_ww,
                )[0]

        # --- visualize (WA + W for body/all, hand-PA for hand views) ---
        if vis:
            keys = batch.get("keys", None)
            B = int(j_gt.shape[0])
            j_pr_wa = batch_procrustes_align_sequence(j_pr, j_gt)
            j_pr_w = w_align_firstk(j_pr, j_gt, num_align_frames=1)
            # Hand-PA: per-frame Procrustes on hand joints only, body from WA
            j_pr_hpa = j_pr_wa.clone()
            j_pr_hpa[..., 22:, :] = batch_procrustes_align(
                j_pr[..., 22:, :], j_gt[..., 22:, :])
            _body_views = {"all", "body"}
            _hand_views = {"hands", "lh", "rh"}
            # body/all → WA/W, hand → hand-PA
            _view_align = {}
            for vn in view_names:
                if vn in _hand_views:
                    _view_align[vn] = [("PA_hand", j_pr_hpa)]
                else:
                    _view_align[vn] = [("WA", j_pr_wa), ("W", j_pr_w)]

            _vis_saved_this_batch = []  # (b, src_b, src_idx) for three-decoder vis
            if vis_balance_sources:
                for b in range(B):
                    key_b = keys[b] if isinstance(keys, list) and b < len(keys) else f"it{it:03d}_b{b:03d}"
                    src_b = _source_from_key(key_b)
                    if vis_sources_set is not None and src_b not in vis_sources_set:
                        continue
                    if vis_saved_by_source[src_b] >= vis_num_per_source:
                        continue

                    _jg_vis = j_gt[b]
                    _src_idx = vis_saved_by_source[src_b]
                    _only_gt = True if args.eval_check and not args.resume else False
                    # HOT3D (body=zeros): hand views only
                    _is_body_zero = _body_is_zero[b].item()
                    _vnames_b = [v for v in view_names if v in _hand_views] if _is_body_zero else view_names
                    for vname in _vnames_b:
                        for aname, j_pr_a in _view_align[vname]:
                            _jp_vis = j_pr_a[b]
                            out_dir = os.path.join(viz_dir, src_b, f"{_src_idx:02d}", aname)
                            _mo = None if _only_gt else _compute_sample_metrics(_jg_vis, _jp_vis, parts)
                            visualize_two_motions(
                                _jg_vis, _jp_vis,
                                save_path=os.path.join(out_dir, f"{vname}.mp4"),
                                fps=fps,
                                view=vname,
                                rotate=False,
                                include_fingertips=args.include_fingertips,
                                only_gt=_only_gt,
                                origin_align=True,
                                base_idx=args.base_idx,
                                metrics_overlay=_mo,
                            )
                    _vis_saved_this_batch.append((b, src_b, _src_idx))
                    vis_saved_by_source[src_b] += 1
            else:
                if vis_saved_total < num_save_samples:
                    _only_gt = True if args.eval_check and not args.resume else False
                    key_0 = keys[0] if isinstance(keys, list) and len(keys) > 0 else ""
                    src_0 = _source_from_key(key_0) if key_0 else "default"
                    _jg_vis = j_gt[0]
                    for vname in view_names:
                        for aname, j_pr_a in _view_align[vname]:
                            _jp_vis = j_pr_a[0]
                            _mo = None if _only_gt else _compute_sample_metrics(_jg_vis, _jp_vis, parts)
                            visualize_two_motions(
                                _jg_vis, _jp_vis,
                                save_path=f"{viz_dir}/{src_0}/{vis_saved_total:02d}/{aname}/{vname}.mp4",
                                fps=fps,
                                view=vname,
                                rotate=False,
                                include_fingertips=args.include_fingertips,
                                only_gt=_only_gt,
                                origin_align=True,
                                base_idx=args.base_idx,
                                metrics_overlay=_mo,
                            )
                    vis_saved_total += 1

        # --- pose metrics (PA / WA / W-firstK) ---
        _eval_parts = parts
        if _hand_only:
            # hand_only: only compute hand metrics (body is zeros → meaningless)
            _eval_parts = {"lh": parts["lh"], "rh": parts["rh"]}
        for name, slc in _eval_parts.items():
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

        # --- hand-only / body-only decoder metrics (three decoders, skip for hand_only mode) ---
        _has_partonly = (not _hand_only and _use_three_decoders and
                         ("recon_hand_only" in losses or _is_flow))
        if _has_partonly:
            if _is_flow:
                # Flow/diffusion: sample from body-only / hand-only decoders
                target_T = mB.shape[1]
                _steps = getattr(args, "flow_sample_steps", 30)
                _solver = getattr(args, "flow_solver", "heun")
                recon_ho = model.sample_from_ids(
                    idx, target_T=target_T, steps=_steps, solver=_solver, mode="hand_only")
                recon_bo = model.sample_from_ids(
                    idx, target_T=target_T, steps=_steps, solver=_solver, mode="body_only")
            else:
                recon_ho = losses["recon_hand_only"]  # (B,T,hand_in_dim)
                recon_bo = losses["recon_body_only"]  # (B,T,body_in_dim)

            # denormalize hand-only / body-only
            gt_body = gt623[..., :_body_in_dim]
            gt_hand = gt623[..., _body_in_dim:]
            pr_ho = recon_ho
            pr_bo = recon_bo
            if getattr(args, "normalize", False):
                if mean.shape[0] < _total_dim:
                    mean_ext = torch.cat([mean, torch.zeros(_total_dim - mean.shape[0], device=device)])
                    std_ext = torch.cat([std, torch.ones(_total_dim - std.shape[0], device=device)])
                else:
                    mean_ext = mean[:_total_dim]
                    std_ext = std[:_total_dim]
                pr_ho = pr_ho * std_ext[_body_in_dim:] + mean_ext[_body_in_dim:]
                pr_bo = pr_bo * std_ext[:_body_in_dim] + mean_ext[:_body_in_dim]

            # hand-only: GT body + pred hand → joints → hand metrics
            j_ho = recover_joints_from_body_hand(
                gt_body, pr_ho,
                include_fingertips=args.include_fingertips,
                hand_root_dim=_hand_root_dim_total,
                joints_num=joints_num,
                use_root_loss=getattr(args, "use_root_loss", False),
                base_idx=args.base_idx,
                hand_local=getattr(args, "hand_local", False),
            )

            # Override wrist for body=zeros samples (HOT3D) in j_ho
            if _body_is_zero.any() and _has_wrist_world:
                from src.evaluate.utils import recover_hand_only_joints
                for bi in range(int(gt623.shape[0])):
                    if not _body_is_zero[bi]:
                        continue
                    _bi_lh_ww = batch["lh_wrist_world"][bi:bi+1].to(device)
                    _bi_rh_ww = batch["rh_wrist_world"][bi:bi+1].to(device)
                    j_ho[bi] = recover_hand_only_joints(
                        pr_ho[bi:bi+1],
                        include_fingertips=args.include_fingertips,
                        hand_root_dim=_hand_root_dim_total,
                        joints_num=joints_num,
                        hand_local=getattr(args, "hand_local", False),
                        lh_wrist_world=_bi_lh_ww,
                        rh_wrist_world=_bi_rh_ww,
                    )[0]

            for name, slc in handonly_parts.items():
                jp_part = j_ho[..., slc, :]
                jg_part = j_gt[..., slc, :]
                jp_pa = batch_procrustes_align(jp_part, jg_part)
                sums_handonly[f"pampjpe_{name}"] += mpjpe_bt(jp_pa, jg_part, slice(None)).mean().item()
                sums_handonly[f"wa_mpjpe_{name}"] += wa_mpjpe(jp_part, jg_part, slice(None)).mean().item()
                sums_handonly[f"w_mpjpe_{name}"] += w_mpjpe_firstk(jp_part, jg_part, slice(None), num_align_frames=1).mean().item()

            # body-only: pred body + GT hand → joints → body metrics
            j_bo = recover_joints_from_body_hand(
                pr_bo, gt_hand,
                include_fingertips=args.include_fingertips,
                hand_root_dim=_hand_root_dim_total,
                joints_num=joints_num,
                use_root_loss=getattr(args, "use_root_loss", False),
                base_idx=args.base_idx,
                hand_local=getattr(args, "hand_local", False),
            )

            for name, slc in bodyonly_parts.items():
                jp_part = j_bo[..., slc, :]
                jg_part = j_gt[..., slc, :]
                jp_pa = batch_procrustes_align(jp_part, jg_part)
                sums_bodyonly[f"pampjpe_{name}"] += mpjpe_bt(jp_pa, jg_part, slice(None)).mean().item()
                sums_bodyonly[f"wa_mpjpe_{name}"] += wa_mpjpe(jp_part, jg_part, slice(None)).mean().item()
                sums_bodyonly[f"w_mpjpe_{name}"] += w_mpjpe_firstk(jp_part, jg_part, slice(None), num_align_frames=1).mean().item()

            # --- visualize hand-only / body-only decoder outputs ---
            if vis:
                ho_views = ["hands", "lh", "rh"]
                bo_views = ["body"]

                if vis_balance_sources:
                    # Reuse the same samples saved by main vis
                    for b, src_b, _src_idx in _vis_saved_this_batch:
                        out_dir = os.path.join(viz_dir, src_b, f"{_src_idx:02d}")
                        _is_bz = _body_is_zero[b].item()
                        _mo_ho = _compute_sample_metrics(j_gt[b], j_ho[b], handonly_parts)
                        for vname in ho_views:
                            visualize_two_motions(
                                j_gt[b], j_ho[b],
                                save_path=os.path.join(out_dir, f"{vname}_handonly.mp4"),
                                fps=fps, view=vname, rotate=False,
                                include_fingertips=args.include_fingertips,
                                origin_align=True, base_idx=args.base_idx,
                                metrics_overlay=_mo_ho,
                            )
                        if not _is_bz:
                            _mo_bo = _compute_sample_metrics(j_gt[b], j_bo[b], bodyonly_parts)
                            for vname in bo_views:
                                visualize_two_motions(
                                    j_gt[b], j_bo[b],
                                    save_path=os.path.join(out_dir, f"{vname}_bodyonly.mp4"),
                                    fps=fps, view=vname, rotate=False,
                                    include_fingertips=args.include_fingertips,
                                    origin_align=True, base_idx=args.base_idx,
                                    metrics_overlay=_mo_bo,
                                )
                else:
                    # Use the same index as main vis (vis_saved_total was already
                    # incremented in the main vis block, so use vis_saved_total-1)
                    _td_idx = vis_saved_total - 1
                    if _td_idx >= 0 and _td_idx < num_save_samples:
                        key_0 = keys[0] if isinstance(keys, list) and len(keys) > 0 else ""
                        src_0 = _source_from_key(key_0) if key_0 else "default"
                        _is_bz_0 = _body_is_zero[0].item()
                        _mo_ho = _compute_sample_metrics(j_gt[0], j_ho[0], handonly_parts)
                        for vname in ho_views:
                            visualize_two_motions(
                                j_gt[0], j_ho[0],
                                save_path=f"{viz_dir}/{src_0}/{_td_idx:02d}/{vname}_handonly.mp4",
                                fps=fps, view=vname, rotate=False,
                                include_fingertips=args.include_fingertips,
                                origin_align=True, base_idx=args.base_idx,
                                metrics_overlay=_mo_ho,
                            )
                        if not _is_bz_0:
                            _mo_bo = _compute_sample_metrics(j_gt[0], j_bo[0], bodyonly_parts)
                            for vname in bo_views:
                                visualize_two_motions(
                                    j_gt[0], j_bo[0],
                                    save_path=f"{viz_dir}/{src_0}/{_td_idx:02d}/{vname}_bodyonly.mp4",
                                    fps=fps, view=vname, rotate=False,
                                    include_fingertips=args.include_fingertips,
                                    origin_align=True, base_idx=args.base_idx,
                                    metrics_overlay=_mo_bo,
                                )

        # --- trajectory metrics (skip pelvis for hand_only) ---
        if not _hand_only:
            sums["relative_translation_error_pelvis"]      += relative_translation_error(j_pr, j_gt, ROOT_IDX, use_scale=False).mean().item()
            sums["root_translation_error_pelvis"]          += root_translation_error(j_pr, j_gt, ROOT_IDX, use_scale=False).mean().item()
        sums["relative_translation_error_lh_wrist"]  += relative_translation_error(j_pr, j_gt, LH_WRIST_IDX, use_scale=False).mean().item()
        sums["relative_translation_error_rh_wrist"]  += relative_translation_error(j_pr, j_gt, RH_WRIST_IDX, use_scale=False).mean().item()
        sums["root_translation_error_lh_wrist"]      += root_translation_error(j_pr, j_gt, LH_WRIST_IDX, use_scale=False).mean().item()
        sums["root_translation_error_rh_wrist"]      += root_translation_error(j_pr, j_gt, RH_WRIST_IDX, use_scale=False).mean().item()

        if not _hand_only:
            sums["accel"]         += accel_all_joints(j_pr, j_gt, fps=fps).mean().item()

        # --- codebook stats ---
        if _hand_only:
            if _use_token_sep:
                for cb_name in ["HR", "HL"]:
                    key = f"idx{cb_name}"
                    if key in idx:
                        u, p = codebook_stats(idx[key], args.K)
                        cb_stats[f"usage{cb_name}"] += u
                        cb_stats[f"ppl{cb_name}"] += p
            else:
                uH, pH = codebook_stats(idx["idxH"], args.K)
                cb_stats["usageH"] += uH; cb_stats["pplH"] += pH
        elif _use_token_sep:
            for cb_name in ["BR", "BL", "HR", "HL"]:
                key = f"idx{cb_name}"
                if key in idx:
                    u, p = codebook_stats(idx[key], args.K)
                    cb_stats[f"usage{cb_name}"] += u
                    cb_stats[f"ppl{cb_name}"] += p
        else:
            uH, pH = codebook_stats(idx["idxH"], args.K)
            uB, pB = codebook_stats(idx["idxB"], args.K)
            cb_stats["usageH"] += uH; cb_stats["pplH"] += pH
            cb_stats["usageB"] += uB; cb_stats["pplB"] += pB

        nb += 1

    nb = max(nb, 1)

    # --- build hierarchical metrics keys for wandb ---
    metrics = {}

    if _hand_only:
        # hand_only: only hand metrics
        for p in ["lh", "rh"]:
            metrics[f"EVAL/PA_MPJPE/{p}(mm)"]  = (sums[f"pampjpe_{p}"] / nb) * 1000.0
            metrics[f"EVAL/WA_MPJPE/{p}(mm)"]  = (sums[f"wa_mpjpe_{p}"] / nb) * 1000.0
            metrics[f"EVAL/W_MPJPE/{p}(mm)"]   = (sums[f"w_mpjpe_{p}"] / nb) * 1000.0

        metrics["EVAL/RECON/feat_mse"]       = sums["feat_mse"] / nb
        metrics["EVAL/RelativeTranslationError/lh_wrist(%)"]  = sums["relative_translation_error_lh_wrist"] / nb
        metrics["EVAL/RelativeTranslationError/rh_wrist(%)"]  = sums["relative_translation_error_rh_wrist"] / nb
        metrics["EVAL/RootTranslationError/lh_wrist(%)"]      = sums["root_translation_error_lh_wrist"] / nb
        metrics["EVAL/RootTranslationError/rh_wrist(%)"]      = sums["root_translation_error_rh_wrist"] / nb

        # codebook
        if _use_token_sep:
            for cb_name in ["HR", "HL"]:
                metrics[f"EVAL/CODEBOOK/{cb_name}_usage"] = cb_stats[f"usage{cb_name}"] / nb
                metrics[f"EVAL/CODEBOOK/{cb_name}_ppl"]   = cb_stats[f"ppl{cb_name}"] / nb
            metrics["EVAL/CODEBOOK/H_usage"] = (cb_stats["usageHR"] + cb_stats["usageHL"]) / (2 * nb)
            metrics["EVAL/CODEBOOK/H_ppl"]   = (cb_stats["pplHR"] + cb_stats["pplHL"]) / (2 * nb)
        else:
            metrics["EVAL/CODEBOOK/H_usage"] = cb_stats["usageH"] / nb
            metrics["EVAL/CODEBOOK/H_ppl"]   = cb_stats["pplH"] / nb
        # Set B_usage/B_ppl to 0 for compat
        metrics["EVAL/CODEBOOK/B_usage"] = 0.0
        metrics["EVAL/CODEBOOK/B_ppl"]   = 0.0
    else:
        # pose (mm)
        for p in parts.keys():
            metrics[f"EVAL/PA_MPJPE/{p}(mm)"]    = (sums[f"pampjpe_{p}"] / nb) * 1000.0
            metrics[f"EVAL/WA_MPJPE/{p}(mm)"]  = (sums[f"wa_mpjpe_{p}"] / nb) * 1000.0
            metrics[f"EVAL/W_MPJPE/{p}(mm)"]   = (sums[f"w_mpjpe_{p}"] / nb) * 1000.0

        # recon / traj
        metrics["EVAL/RECON/feat_mse"]       = sums["feat_mse"] / nb
        metrics["EVAL/RelativeTranslationError/pelvis(%)"]             = sums["relative_translation_error_pelvis"] / nb
        metrics["EVAL/RelativeTranslationError/lh_wrist(%)"]         = sums["relative_translation_error_lh_wrist"] / nb
        metrics["EVAL/RelativeTranslationError/rh_wrist(%)"]         = sums["relative_translation_error_rh_wrist"] / nb
        metrics["EVAL/RootTranslationError/pelvis(%)"]             = sums["root_translation_error_pelvis"] / nb
        metrics["EVAL/RootTranslationError/lh_wrist(%)"]             = sums["root_translation_error_lh_wrist"] / nb
        metrics["EVAL/RootTranslationError/rh_wrist(%)"]             = sums["root_translation_error_rh_wrist"] / nb

        metrics["EVAL/ACCEL/all(mm/s^2)"]            = sums["accel"] * 1000.0 / nb

        # codebook
        if _use_token_sep:
            for cb_name in ["BR", "BL", "HR", "HL"]:
                metrics[f"EVAL/CODEBOOK/{cb_name}_usage"] = cb_stats[f"usage{cb_name}"] / nb
                metrics[f"EVAL/CODEBOOK/{cb_name}_ppl"]   = cb_stats[f"ppl{cb_name}"] / nb
            # Also provide H/B aggregates for compatibility with train.py logging
            metrics["EVAL/CODEBOOK/H_usage"] = (cb_stats["usageHR"] + cb_stats["usageHL"]) / (2 * nb)
            metrics["EVAL/CODEBOOK/H_ppl"]   = (cb_stats["pplHR"] + cb_stats["pplHL"]) / (2 * nb)
            metrics["EVAL/CODEBOOK/B_usage"] = (cb_stats["usageBR"] + cb_stats["usageBL"]) / (2 * nb)
            metrics["EVAL/CODEBOOK/B_ppl"]   = (cb_stats["pplBR"] + cb_stats["pplBL"]) / (2 * nb)
        else:
            metrics["EVAL/CODEBOOK/H_usage"] = cb_stats["usageH"] / nb
            metrics["EVAL/CODEBOOK/H_ppl"]   = cb_stats["pplH"] / nb
            metrics["EVAL/CODEBOOK/B_usage"] = cb_stats["usageB"] / nb
            metrics["EVAL/CODEBOOK/B_ppl"]   = cb_stats["pplB"] / nb

    # hand-only / body-only decoder metrics (three decoders only)
    if _use_three_decoders:
        for p in handonly_parts:
            metrics[f"EVAL/PA_MPJPE/{p}_handonly(mm)"]  = (sums_handonly[f"pampjpe_{p}"] / nb) * 1000.0
            metrics[f"EVAL/WA_MPJPE/{p}_handonly(mm)"]  = (sums_handonly[f"wa_mpjpe_{p}"] / nb) * 1000.0
            metrics[f"EVAL/W_MPJPE/{p}_handonly(mm)"]   = (sums_handonly[f"w_mpjpe_{p}"] / nb) * 1000.0
        for p in bodyonly_parts:
            metrics[f"EVAL/PA_MPJPE/{p}_bodyonly(mm)"]  = (sums_bodyonly[f"pampjpe_{p}"] / nb) * 1000.0
            metrics[f"EVAL/WA_MPJPE/{p}_bodyonly(mm)"]  = (sums_bodyonly[f"wa_mpjpe_{p}"] / nb) * 1000.0
            metrics[f"EVAL/W_MPJPE/{p}_bodyonly(mm)"]   = (sums_bodyonly[f"w_mpjpe_{p}"] / nb) * 1000.0

    # --- extra vis from additional DataLoader (e.g. hot3d from train set, hand views only) ---
    if vis and extra_vis_dl is not None:
        _extra_hand_views = ["hands", "lh", "rh"]
        _extra_saved = defaultdict(int)
        for batch_ex in extra_vis_dl:
            mB_ex = batch_ex["mB"].to(device, non_blocking=True)
            mH_ex = batch_ex["mH"].to(device, non_blocking=True)
            keys_ex = batch_ex.get("keys", None)
            _ex_bz = (mB_ex.abs().sum(dim=(-1, -2)) < 1e-6)  # before normalize

            if getattr(args, "normalize", False):
                motion_ex = torch.cat([mB_ex, mH_ex], dim=-1)
                if mean.shape[0] < _total_dim:
                    pad_len = _total_dim - mean.shape[0]
                    mean_ext = torch.cat([mean, torch.zeros(pad_len, device=device)])
                    std_ext = torch.cat([std, torch.ones(pad_len, device=device)])
                    motion_ex = (motion_ex - mean_ext) / std_ext
                else:
                    motion_ex = (motion_ex - mean[:_total_dim]) / std[:_total_dim]
                mB_ex = motion_ex[..., :_body_in_dim]
                mH_ex = motion_ex[..., _body_in_dim:]

            recon_ex, _, _ = model(mB_ex, mH_ex)
            gt_ex = torch.cat([mB_ex, mH_ex], dim=-1)
            pr_ex = recon_ex

            if getattr(args, "normalize", False):
                if mean.shape[0] < _total_dim:
                    gt_ex = gt_ex * std_ext + mean_ext
                    pr_ex = pr_ex * std_ext + mean_ext
                else:
                    gt_ex = gt_ex * std[:_total_dim] + mean[:_total_dim]
                    pr_ex = pr_ex * std[:_total_dim] + mean[:_total_dim]

            j_gt_ex = recover_joints_from_body_hand(
                gt_ex[..., :_body_in_dim], gt_ex[..., _body_in_dim:],
                include_fingertips=args.include_fingertips,
                hand_root_dim=_hand_root_dim_total,
                joints_num=joints_num,
                use_root_loss=getattr(args, "use_root_loss", False),
                base_idx=args.base_idx,
                hand_local=getattr(args, "hand_local", False),
                hand_only=_hand_only,
            )
            j_pr_ex = recover_joints_from_body_hand(
                pr_ex[..., :_body_in_dim], pr_ex[..., _body_in_dim:],
                include_fingertips=args.include_fingertips,
                hand_root_dim=_hand_root_dim_total,
                joints_num=joints_num,
                use_root_loss=getattr(args, "use_root_loss", False),
                base_idx=args.base_idx,
                hand_local=getattr(args, "hand_local", False),
                hand_only=_hand_only,
            )

            # Override hand-only samples with hand-only reconstruction
            _ex_has_ww = "lh_wrist_world" in batch_ex and "rh_wrist_world" in batch_ex
            if _ex_bz.any():
                from src.evaluate.utils import recover_hand_only_joints
                _ex_lh_ww = batch_ex["lh_wrist_world"].to(device) if _ex_has_ww else None
                _ex_rh_ww = batch_ex["rh_wrist_world"].to(device) if _ex_has_ww else None
                for bi in range(int(gt_ex.shape[0])):
                    if not _ex_bz[bi]:
                        continue
                    _bi_lh = _ex_lh_ww[bi:bi+1] if _ex_lh_ww is not None else None
                    _bi_rh = _ex_rh_ww[bi:bi+1] if _ex_rh_ww is not None else None
                    j_gt_ex[bi] = recover_hand_only_joints(
                        gt_ex[bi:bi+1, :, _body_in_dim:],
                        include_fingertips=args.include_fingertips,
                        hand_root_dim=_hand_root_dim_total,
                        joints_num=joints_num,
                        hand_local=getattr(args, "hand_local", False),
                        lh_wrist_world=_bi_lh,
                        rh_wrist_world=_bi_rh,
                    )[0]
                    j_pr_ex[bi] = recover_hand_only_joints(
                        pr_ex[bi:bi+1, :, _body_in_dim:],
                        include_fingertips=args.include_fingertips,
                        hand_root_dim=_hand_root_dim_total,
                        joints_num=joints_num,
                        hand_local=getattr(args, "hand_local", False),
                        lh_wrist_world=_bi_lh,
                        rh_wrist_world=_bi_rh,
                    )[0]

            j_pr_hpa_ex = batch_procrustes_align_sequence(j_pr_ex, j_gt_ex).clone()
            j_pr_hpa_ex[..., 22:, :] = batch_procrustes_align(
                j_pr_ex[..., 22:, :], j_gt_ex[..., 22:, :])

            for b in range(j_gt_ex.shape[0]):
                key_b = keys_ex[b] if isinstance(keys_ex, list) and b < len(keys_ex) else f"extra_b{b:03d}"
                src_b = _source_from_key(key_b)
                if _extra_saved[src_b] >= vis_num_per_source:
                    continue
                _src_idx = _extra_saved[src_b]
                _mo = _compute_sample_metrics(j_gt_ex[b], j_pr_hpa_ex[b],
                    {"lh": parts["lh"], "rh": parts["rh"]})
                for vname in _extra_hand_views:
                    visualize_two_motions(
                        j_gt_ex[b], j_pr_hpa_ex[b],
                        save_path=os.path.join(viz_dir, src_b, f"{_src_idx:02d}", "PA_hand", f"{vname}.mp4"),
                        fps=fps, view=vname, rotate=False,
                        include_fingertips=args.include_fingertips,
                        origin_align=True, base_idx=args.base_idx,
                        metrics_overlay=_mo,
                    )
                _extra_saved[src_b] += 1

            if all(v >= vis_num_per_source for v in _extra_saved.values()):
                break

    return metrics
