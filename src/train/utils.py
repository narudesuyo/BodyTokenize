import torch
from src.model.vqvae import H2VQ
import os
def build_model_from_args(args, device):
    if args.include_fingertips:
        body_in_dim = 263
        hand_in_dim = 480
    else:
        body_in_dim = 263
        hand_in_dim = 360
    model = H2VQ(
        T=args.T, 
        body_in_dim=body_in_dim,
        hand_in_dim=hand_in_dim,
        code_dim=args.code_dim,
        K=args.K,
        ema_decay=args.ema_decay,
        alpha_commit=args.alpha_commit,
        body_tokens_per_t=args.body_tokens_per_t,
        hand_tokens_per_t=args.hand_tokens_per_t,
        body_down=args.body_down,
        hand_down=args.hand_down,
        enc_type_B=args.enc_type_B,  # "xformer" or "cnn"
        enc_type_H=args.enc_type_H,  # "xformer" or "cnn"
        enc_use_attn_B=args.enc_use_attn_B,
        enc_use_attn_H=args.enc_use_attn_H,
        enc_depth=args.enc_depth,
        enc_heads=args.enc_heads,
        mlp_ratio=args.mlp_ratio,
        enc_use_pos=args.enc_use_pos,
        enc_post_mlp=args.enc_post_mlp,
        cnn_width_B=args.cnn_width_B,
        cnn_depth_B=args.cnn_depth_B,
        cnn_width_H=args.cnn_width_H,
        cnn_depth_H=args.cnn_depth_H,
        cnn_dilation_max=args.cnn_dilation_max,
        dec_hid=args.dec_hid,
        alpha_root=args.alpha_root,
        alpha_body=args.alpha_body,
        alpha_hand=args.alpha_hand,        
        use_root_loss=args.use_root_loss,
        include_fingertips=args.include_fingertips,
    ).to(device)
    return model

# src/train/utils.py

import torch


def build_model_from_args_flow(args, device):
    """
    Build flow-only VQ model.
    Requires: src.model.vqvae_flow.H2VQFlow (or adjust import path).
    """
    # ここはあなたの配置に合わせて import を調整してね
    # 例: from src.model.vqvae_flow import H2VQFlow
    from src.model.vqvae_flow import H2VQFlow

    # ---- core dims (固定前提ならそのままでもOK) ----
    T = getattr(args, "T", 81)  # あなたのtrainが (args.T-1) を期待してるので合わせる
    body_in_dim = getattr(args, "body_in_dim", 263)
    hand_in_dim = getattr(args, "hand_in_dim", 360)

    # ---- VQ + tokens/downsample ----
    code_dim = getattr(args, "code_dim", 512)
    K = getattr(args, "K", 512)
    ema_decay = getattr(args, "ema_decay", 0.99)

    body_tokens_per_t = getattr(args, "body_tokens_per_t", 2)
    hand_tokens_per_t = getattr(args, "hand_tokens_per_t", 4)
    body_down = getattr(args, "body_down", 4)
    hand_down = getattr(args, "hand_down", 4)

    # ---- enc type ----
    enc_type_B = getattr(args, "enc_type_B", "xformer")
    enc_type_H = getattr(args, "enc_type_H", "xformer")

    # ---- xformer enc ----
    enc_depth = getattr(args, "enc_depth", 6)
    enc_heads = getattr(args, "enc_heads", 8)
    enc_mlp_ratio = getattr(args, "enc_mlp_ratio", getattr(args, "mlp_ratio", 4.0))
    enc_drop = getattr(args, "enc_drop", 0.0)
    enc_attn_drop = getattr(args, "enc_attn_drop", 0.0)
    enc_use_pos = getattr(args, "enc_use_pos", True)
    enc_post_mlp = getattr(args, "enc_post_mlp", True)
    enc_use_attn_B = getattr(args, "enc_use_attn_B", True)
    enc_use_attn_H = getattr(args, "enc_use_attn_H", True)

    # ---- cnn enc ----
    cnn_width_B = getattr(args, "cnn_width_B", 512)
    cnn_depth_B = getattr(args, "cnn_depth_B", 8)
    cnn_width_H = getattr(args, "cnn_width_H", 512)
    cnn_depth_H = getattr(args, "cnn_depth_H", 8)
    cnn_kernel = getattr(args, "cnn_kernel", 3)
    cnn_dilation_max = getattr(args, "cnn_dilation_max", 8)
    cnn_drop = getattr(args, "cnn_drop", 0.0)

    # ---- fuse ----
    use_fuse = getattr(args, "use_fuse", True)

    # ---- flow decoder ----
    flow_model_dim = getattr(args, "flow_model_dim", getattr(args, "dec_dim", 512))
    flow_depth = getattr(args, "flow_depth", 8)
    flow_heads = getattr(args, "flow_heads", 8)
    flow_mlp_ratio = getattr(args, "flow_mlp_ratio", 4.0)
    flow_drop = getattr(args, "flow_drop", 0.0)
    flow_attn_drop = getattr(args, "flow_attn_drop", 0.0)
    flow_t_dim = getattr(args, "flow_t_dim", 512)
    flow_use_x_pos = getattr(args, "flow_use_x_pos", True)
    flow_use_rope = getattr(args, "flow_use_rope", False)

    # ---- losses ----
    lambda_flow = getattr(args, "lambda_flow", 1.0)
    alpha_commit = getattr(args, "alpha_commit", 0.02)
    lambda_entropy = getattr(args, "lambda_entropy", 1e-3)

    # ---- part weights + root selection ----
    alpha_root = getattr(args, "alpha_root", 1.0)
    alpha_body = getattr(args, "alpha_body", 1.0)
    alpha_hand = getattr(args, "alpha_hand", 1.0)
    use_root_loss = getattr(args, "use_root_loss", True)
    root_keep_idx = getattr(args, "root_keep_idx", (0, 3))

    # ---- stability ----
    mask_input_dims = getattr(args, "mask_input_dims", True)

    # ---- quant ----
    do_reset = getattr(args, "do_reset", True)

    cond_type = getattr(args, "cond_type", "baseline")

    model = H2VQFlow(
        T=T,
        body_in_dim=body_in_dim,
        hand_in_dim=hand_in_dim,

        code_dim=code_dim,
        K=K,
        ema_decay=ema_decay,
        alpha_commit=alpha_commit,
        lambda_flow=lambda_flow,
        lambda_entropy=lambda_entropy,

        body_tokens_per_t=body_tokens_per_t,
        hand_tokens_per_t=hand_tokens_per_t,
        body_down=body_down,
        hand_down=hand_down,

        enc_type_B=enc_type_B,
        enc_type_H=enc_type_H,

        enc_depth=enc_depth,
        enc_heads=enc_heads,
        enc_mlp_ratio=enc_mlp_ratio,
        enc_drop=enc_drop,
        enc_attn_drop=enc_attn_drop,
        enc_use_pos=enc_use_pos,
        enc_post_mlp=enc_post_mlp,
        enc_use_attn_B=enc_use_attn_B,
        enc_use_attn_H=enc_use_attn_H,

        cnn_width_B=cnn_width_B,
        cnn_depth_B=cnn_depth_B,
        cnn_width_H=cnn_width_H,
        cnn_depth_H=cnn_depth_H,
        cnn_kernel=cnn_kernel,
        cnn_dilation_max=cnn_dilation_max,
        cnn_drop=cnn_drop,

        use_fuse=use_fuse,

        flow_model_dim=flow_model_dim,
        flow_depth=flow_depth,
        flow_heads=flow_heads,
        flow_mlp_ratio=flow_mlp_ratio,
        flow_drop=flow_drop,
        flow_attn_drop=flow_attn_drop,
        flow_t_dim=flow_t_dim,
        flow_use_x_pos=flow_use_x_pos,
        flow_use_rope=flow_use_rope,

        alpha_root=alpha_root,
        alpha_body=alpha_body,
        alpha_hand=alpha_hand,
        use_root_loss=use_root_loss,
        root_keep_idx=root_keep_idx,

        mask_input_dims=mask_input_dims,
        do_reset=do_reset,
        cond_type=cond_type,
    ).to(device)

    return model

def _maybe_load_ckpt(path: str):
    if path is None:
        return None
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _safe_merge_args_from_ckpt(args, ckpt):
    if ckpt is None:
        return args

    ckpt_args = ckpt.get("args", None)
    if not isinstance(ckpt_args, dict):
        try:
            ckpt_args = vars(ckpt_args)
        except Exception:
            ckpt_args = None

    if not isinstance(ckpt_args, dict):
        return args

    keys_to_sync = [
        "K", "T", "include_fingertips",
        "model_type", "n_layers", "n_heads", "d_model", "d_ff",
        "mean_path", "std_path", "normalize",
        "joints_loss", "joints_loss_weight",
        "lr", "wd",
        "eval_every", "eval_num_save_samples", "eval_vis_dir", "eval_save_vis_every",
        "log_every", "ckpt_every",
        "project", "name",
        "epochs", "batch_size", "num_workers",
        "data_dir", "data_dir_eval",
    ]

    for k in keys_to_sync:
        if k in ckpt_args:
            try:
                args[k] = ckpt_args[k]
            except Exception:
                pass

    return args