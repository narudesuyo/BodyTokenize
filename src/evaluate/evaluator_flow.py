import torch
import sys
sys.path.append(".")
import numpy as np
from src.evaluate.utils import recover_joints_from_body_hand
from src.evaluate.vis import visualize_two_motions
from src.evaluate.evaluator import _compute_sample_metrics, _source_from_key
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
    cb_stats = {"usageH": 0.0, "pplH": 0.0, "usageB": 0.0, "pplB": 0.0}
    nb = 0

    use_norm = getattr(args, "normalize", False)
    if use_norm:
        mean = torch.from_numpy(np.load(args.mean_path)).to(device)
        std  = torch.from_numpy(np.load(args.std_path)).to(device)
        mean[0:1] = 0
        std[0:1] = 1

    steps = getattr(args, "flow_sample_steps_eval", getattr(args, "flow_sample_steps", 30))
    solver = getattr(args, "flow_solver_eval", getattr(args, "flow_solver", "heun"))

    for it, batch in tqdm(enumerate(dl), total=len(dl), leave=False):
        mB = batch["mB"].to(device, non_blocking=True)
        mH = batch["mH"].to(device, non_blocking=True)

        if use_norm:
            motion = torch.cat([mB, mH], dim=-1)
            motion = (motion - mean) / std
            mB = motion[..., :263]
            mH = motion[..., 263:]

        # forward to get ids
        _, losses, idx = model(mB, mH)

        gt623 = torch.cat([mB, mH], dim=-1)  # feature space (maybe normalized)
        pr623 = model.sample_from_ids(
            idx["idxH"], idx["idxB"],
            target_T=gt623.shape[1],
            steps=steps,
            solver=solver,
        )

        gt_dn, pr_dn = gt623, pr623
        if use_norm:
            gt_dn = gt623 * std + mean
            pr_dn = pr623 * std + mean

        sums["feat_mse"] += torch.mean((pr_dn - gt_dn) ** 2).item()

        joints_num = 52 if not args.include_fingertips else 62
        _hand_root_dim_total = getattr(args, "hand_root_dim", 9) * 2 if getattr(args, "hand_root", False) else 0
        j_gt = recover_joints_from_body_hand(
            gt_dn[..., :263], gt_dn[..., 263:],
            include_fingertips=args.include_fingertips,
            hand_root_dim=_hand_root_dim_total,
            joints_num=joints_num,
            use_root_loss=getattr(args, "use_root_loss", True),
            base_idx=args.base_idx,
            hand_local=getattr(args, "hand_local", False),
            hand_only=getattr(args, "hand_only", False),
        )
        j_pr = recover_joints_from_body_hand(
            pr_dn[..., :263], pr_dn[..., 263:],
            include_fingertips=args.include_fingertips,
            hand_root_dim=_hand_root_dim_total,
            joints_num=joints_num,
            use_root_loss=getattr(args, "use_root_loss", True),
            base_idx=args.base_idx,
            hand_local=getattr(args, "hand_local", False),
            hand_only=getattr(args, "hand_only", False),
        )

        if vis and it < num_save_samples:
            view_names = ["all", "body", "hands", "lh", "rh"]
            _only_gt = True if args.eval_check and not args.resume else False
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
            keys = batch.get("keys", None)
            key_0 = keys[0] if isinstance(keys, list) and len(keys) > 0 else ""
            _jg_vis = j_gt[0]
            for vname in view_names:
                for aname, j_pr_a in _view_align[vname]:
                    _jp_vis = j_pr_a[0]
                    _mo = None if _only_gt else _compute_sample_metrics(_jg_vis, _jp_vis, parts)
                    visualize_two_motions(
                        _jg_vis, _jp_vis,
                        save_path=f"{viz_dir}/{it:03d}/{aname}/{vname}.mp4",
                        fps=fps,
                        view=vname,
                        rotate=False,
                        include_fingertips=args.include_fingertips,
                        only_gt=_only_gt,
                        origin_align=True,
                        base_idx=args.base_idx,
                        metrics_overlay=_mo,
                    )

        for name, slc in parts.items():
            jp_part = j_pr[..., slc, :]
            jg_part = j_gt[..., slc, :]

            jp_pa = batch_procrustes_align(jp_part, jg_part)
            sums[f"pampjpe_{name}"] += mpjpe_bt(jp_pa, jg_part, slice(None)).mean().item()
            sums[f"wa_mpjpe_{name}"] += wa_mpjpe(jp_part, jg_part, slice(None)).mean().item()
            sums[f"w_mpjpe_{name}"] += w_mpjpe_firstk(jp_part, jg_part, slice(None), num_align_frames=1).mean().item()

        sums["relative_translation_error_pelvis"] += relative_translation_error(j_pr, j_gt, ROOT_IDX, use_scale=False).mean().item()
        sums["relative_translation_error_lh_wrist"] += relative_translation_error(j_pr, j_gt, LH_WRIST_IDX, use_scale=False).mean().item()
        sums["relative_translation_error_rh_wrist"] += relative_translation_error(j_pr, j_gt, RH_WRIST_IDX, use_scale=False).mean().item()
        sums["root_translation_error_pelvis"] += root_translation_error(j_pr, j_gt, ROOT_IDX, use_scale=False).mean().item()
        sums["root_translation_error_lh_wrist"] += root_translation_error(j_pr, j_gt, LH_WRIST_IDX, use_scale=False).mean().item()
        sums["root_translation_error_rh_wrist"] += root_translation_error(j_pr, j_gt, RH_WRIST_IDX, use_scale=False).mean().item()
        sums["accel"] += accel_all_joints(j_pr, j_gt, fps=fps).mean().item()

        uH, pH = codebook_stats(idx["idxH"], args.K)
        uB, pB = codebook_stats(idx["idxB"], args.K)
        cb_stats["usageH"] += uH; cb_stats["pplH"] += pH
        cb_stats["usageB"] += uB; cb_stats["pplB"] += pB

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

    metrics["EVAL/CODEBOOK/H_usage"] = cb_stats["usageH"] / nb
    metrics["EVAL/CODEBOOK/H_ppl"] = cb_stats["pplH"] / nb
    metrics["EVAL/CODEBOOK/B_usage"] = cb_stats["usageB"] / nb
    metrics["EVAL/CODEBOOK/B_ppl"] = cb_stats["pplB"] / nb

    return metrics
