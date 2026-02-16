"""
Evaluate a flow checkpoint with different sampling steps/solver.

Usage:
  python src/evaluate/eval_flow_steps.py \
      --ckpt runs/.../ckpt_epoch100.pt \
      --config config/motion_vqvae_flow.yaml \
      --steps 10 20 30 50 100 \
      --solver heun
"""

import sys
sys.path.append(".")

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from tqdm import tqdm

from src.dataset.dataloader import MotionDataset
from src.dataset.collate import collate_stack
from src.train.utils import build_model_from_args_flow
from src.evaluate.evaluator_flow import evaluate_model
from src.util.utils import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--config", type=str, default="config/motion_vqvae_flow.yaml")
    ap.add_argument("--steps", type=int, nargs="+", default=[10, 20, 30, 50, 100])
    ap.add_argument("--solver", type=str, default="heun", choices=["euler", "heun"])
    ap.add_argument("--vis", action="store_true", help="Save visualization")
    ap.add_argument("--num_vis", type=int, default=5)
    args_cli = ap.parse_args()

    args = OmegaConf.load(args_cli.config)
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ===== Dataset =====
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

    # ===== Model =====
    ckpt = torch.load(args_cli.ckpt, weights_only=False, map_location=device)
    model = build_model_from_args_flow(args, device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    epoch = ckpt.get("epoch", "?")
    step = ckpt.get("step", "?")
    print(f"Loaded checkpoint: {args_cli.ckpt}  (epoch={epoch}, step={step})")

    # ===== Run eval for each step count =====
    key_metrics = [
        "EVAL/PA_MPJPE/all(mm)",
        "EVAL/PA_MPJPE/body(mm)",
        "EVAL/PA_MPJPE/lh(mm)",
        "EVAL/PA_MPJPE/rh(mm)",
        "EVAL/WA_MPJPE/all(mm)",
        "EVAL/W_MPJPE/all(mm)",
        "EVAL/RECON/feat_mse",
        "EVAL/ACCEL/all(mm/s^2)",
        "EVAL/CODEBOOK/H_usage",
        "EVAL/CODEBOOK/B_usage",
    ]

    results = {}
    for n_steps in args_cli.steps:
        print(f"\n{'='*60}")
        print(f"  solver={args_cli.solver}  steps={n_steps}")
        print(f"{'='*60}")

        # Override eval settings
        args.flow_sample_steps_eval = n_steps
        args.flow_solver_eval = args_cli.solver

        viz_dir = f"{args.save_dir}/eval_steps/{args_cli.solver}_{n_steps}"
        metrics = evaluate_model(
            model, dl_eval, args,
            device=device,
            num_save_samples=args_cli.num_vis,
            viz_dir=viz_dir,
            vis=args_cli.vis,
        )

        results[n_steps] = metrics
        for k in key_metrics:
            if k in metrics:
                print(f"  {k}: {metrics[k]:.4f}")

    # ===== Summary table =====
    print(f"\n{'='*60}")
    print(f"  Summary  (solver={args_cli.solver})")
    print(f"{'='*60}")

    header = f"{'steps':>6}"
    for k in key_metrics:
        short = k.split("/")[-1]
        header += f"  {short:>14}"
    print(header)
    print("-" * len(header))

    for n_steps in args_cli.steps:
        row = f"{n_steps:>6}"
        for k in key_metrics:
            val = results[n_steps].get(k, float("nan"))
            row += f"  {val:>14.4f}"
        print(row)


if __name__ == "__main__":
    main()
